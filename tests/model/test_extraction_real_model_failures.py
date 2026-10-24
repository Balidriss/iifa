"""Extraction against the failures real local models show in a manual run
(SPEC 7.4b, 4.2, 4.4b, 5.1, 5.2): answers capped and never parsed when
truncated, JSON mode optional, extra request parameters, complete statements
that name their subject, aliases grounded in the source, co-occurrence within
one sentence."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest.jobs import turn_text
from iifa.ingest.queue import FAILED
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply
from tests.helpers import auth, remember
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)
TURN = "Karim dirige la planification de l'installation du projet solaire depuis mars 2026."


def karim_fact(statement: str = "Karim dirige la planification du projet solaire.") -> Any:
    return {
        "subject": "Karim",
        "statement": statement,
        "quote": "dirige la planification",
    }


def karim_answer(statement: str = "Karim dirige la planification du projet solaire.") -> Reply:
    return answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[karim_fact(statement)])


@pytest.fixture
def overrides() -> dict[str, Any]:
    return {}


@pytest.fixture
def app(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock, overrides: dict[str, Any]
) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai, **overrides), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def run(client: TestClient) -> dict[str, Any]:
    response = client.post("/v1/background/run", json={"budget": {"seconds": 60}})
    assert response.status_code == 200, response.text
    report: dict[str, Any] = response.json()
    return report


# --------------------------------------------------------------------------- caps and truncation


def test_every_extraction_call_is_capped_from_its_source(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    long_turn = " ".join([TURN] * 30)
    fake_openai.script(chat=[karim_answer(), karim_answer()])
    post_turn(client, TURN)
    post_turn(client, long_turn)
    run_all(client, app)
    pipeline = app.state.worker.pipeline
    short_cap, long_cap = (c.body["max_tokens"] for c in fake_openai.chat_calls)
    assert short_cap == pipeline.answer_cap(turn_text({"user": TURN}))
    assert long_cap == pipeline.answer_cap(turn_text({"user": long_turn}))
    assert short_cap < long_cap <= pipeline.answer_max_tokens


def test_a_truncated_answer_is_asked_again_with_twice_the_cap_and_never_parsed(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    # the truncated answer is itself valid JSON naming a node: parsing it would write it
    truncated = Reply(
        json.dumps({"nodes": [{"name": "Truncated Node"}], "facts": []}),
        finish_reason="length",
    )
    fake_openai.script(chat=[truncated, karim_answer()])
    post_turn(client, TURN)
    report = run_all(client, app)
    first, second = (c.body["max_tokens"] for c in fake_openai.chat_calls)
    assert second == 2 * first
    store = app.state.store
    assert store.find_node("Truncated Node", VIEW) is None
    assert store.find_node("Karim", VIEW) is not None
    assert report["jobs"][0]["result"]["truncated"] == 1


def test_an_answer_truncated_twice_fails_the_job_and_writes_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    cut = Reply('{"nodes": [{"name": "Kar', finish_reason="length")
    fake_openai.script(chat=[cut, cut])
    job_id = post_turn(client, TURN)
    report = run(client)
    assert len(fake_openai.chat_calls) == 2
    job = app.state.queue.get(job_id)
    assert job.status == FAILED and job.error == "length"  # never retried: it would truncate again
    assert report["jobs"][0]["outcome"] == "failed"
    assert app.state.store.counts()["sources"] == 0
    assert app.state.store.counts()["nodes"] == 0


@pytest.mark.parametrize("overrides", [{"answer_max_tokens": 600}])
def test_a_cap_already_at_its_bound_is_not_asked_again(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[Reply("{", finish_reason="length")])
    job_id = post_turn(client, TURN)
    run(client)
    assert [c.body["max_tokens"] for c in fake_openai.chat_calls] == [600]
    assert app.state.queue.get(job_id).status == FAILED


def test_a_truncated_summary_is_asked_again_then_left_unchanged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            karim_answer(),
            Reply("Karim dirige la plani", finish_reason="length"),
            Reply("Karim dirige la plani", finish_reason="length"),
        ]
    )
    post_turn(client, TURN, session_id="s1")
    client.post("/v1/session/end", json={"scope": SCOPE, "session_id": "s1"})
    report = run_all(client, app)
    caps = [c.body["max_tokens"] for c in fake_openai.chat_calls[1:]]
    assert caps[1] == 2 * caps[0]
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None and karim.summary == ""
    assert report["jobs"][1]["result"]["summaries"]["truncated"] == 1


# --------------------------------------------------------------------------- JSON mode and extras


@pytest.mark.parametrize(
    "overrides", [{"chat_json_mode": False, "chat_extra": {"reasoning_effort": "none"}}]
)
def test_json_mode_off_reads_the_object_out_of_a_plain_answer(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    plain = (
        '<think>The source says Karim leads; maybe {"nodes": []}?</think>\n'
        "Voici l'extraction :\n```json\n"
        + json.dumps(
            {"nodes": [{"name": "Karim", "kind": "person"}], "facts": [karim_fact()]},
            ensure_ascii=False,
        )
        + "\n```"
    )
    fake_openai.script(chat=[Reply(plain)])
    post_turn(client, TURN)
    run_all(client, app)
    call = fake_openai.chat_calls[0]
    assert "response_format" not in call.body
    assert call.body["reasoning_effort"] == "none"
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None
    assert [f.statement for f in app.state.store.facts([karim.id], VIEW)] == [
        "Karim dirige la planification du projet solaire."
    ]


# --------------------------------------------------------------------------- complete statements


def test_a_fragment_is_asked_again_as_a_sentence(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[karim_answer("dirige"), karim_answer()])
    post_turn(client, TURN)
    report = run_all(client, app)
    assert len(fake_openai.chat_calls) == 2
    note = fake_openai.chat_calls[1].messages[0]["content"]
    assert "complete sentence" in note and "names its subject" in note
    # not a script problem: the note never asks for spellings among the aliases
    assert "list that spelling" not in note
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None
    facts = app.state.store.facts([karim.id], VIEW)
    assert [f.statement for f in facts] == ["Karim dirige la planification du projet solaire."]
    # a complete sentence is long enough for P12 to check it
    assert facts[0].language == "fr"
    assert report["jobs"][0]["result"]["retries"] == 1


def test_a_statement_that_does_not_name_its_subject_is_a_fragment_too(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    unnamed = "Il dirige la planification du projet solaire."
    fake_openai.script(chat=[karim_answer(unnamed), karim_answer(unnamed)])
    post_turn(client, TURN)
    report = run_all(client, app)
    result = report["jobs"][0]["result"]
    assert result["facts"]["written"] == 0 and result["facts"]["incomplete"] == 1
    assert result["rejected"]["facts"] >= 1
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None and app.state.store.facts([karim.id], VIEW) == []


def test_a_fragment_asked_again_is_dropped_and_counted(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[karim_answer("déteste"), karim_answer("déteste")])
    post_turn(client, TURN)
    report = run_all(client, app)
    result = report["jobs"][0]["result"]
    assert result["facts"]["incomplete"] == 1 and result["facts"]["written"] == 0
    assert app.state.store.counts()["facts"] == 0


def test_a_free_standing_fact_is_checked_for_completeness_only(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(
                facts=[
                    {"subject": None, "statement": "Réponds toujours en français."},
                    {"subject": None, "statement": "court"},
                ]
            ),
            answer(facts=[{"subject": None, "statement": "Réponds toujours en français."}]),
        ]
    )
    post_turn(client, "Réponds toujours en français, s'il te plaît, et fais court.")
    report = run_all(client, app)
    assert len(fake_openai.chat_calls) == 2  # "court" asked again; the sentence was never a subject
    assert report["jobs"][0]["result"]["facts"]["written"] == 1


# --------------------------------------------------------------------------- grounded aliases


def test_an_alias_absent_from_the_source_becomes_a_proposal(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ivan = remember(client, "Иван руководит проектом.", node={"name": "Иван", "kind": "person"})
    ivan_id = ivan["node"]["id"]
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Иван", "kind": "person", "aliases": ["Карим", "Ваня"]}],
                facts=[
                    {
                        "subject": "Иван",
                        "statement": "Иван, которого зовут Ваня, руководит проектом.",
                        "quote": "Ваня",
                    }
                ],
            ),
            answer(nodes=[{"name": "Карим", "kind": "person"}]),
        ]
    )
    post_turn(client, "Иван (все зовут его Ваня) руководит проектом.")
    report = run_all(client, app)
    store = app.state.store
    assert store.get_node(ivan_id).aliases == ("Ваня",)  # found in the source: kept
    proposals = store.proposals([ivan_id], VIEW)
    assert [(p.kind, p.value, p.status) for p in proposals] == [("alias", "Карим", "pending")]
    assert proposals[0].scopes == (SCOPE,) and proposals[0].source_id is not None
    assert report["jobs"][0]["result"]["aliases"] == {
        "kept": 1,
        "proposed": 1,
        "invented": 0,
        "descriptive": 0,
        "near": 0,
    }

    # resolution never uses a proposal: the real Karim gets his own node
    post_turn(client, "Карим пришёл вчера.")
    run_all(client, app)
    karim = store.find_node("Карим", VIEW)
    assert karim is not None and karim.id != ivan_id

    # the node view shows it; accepting it makes it an alias
    seen = client.get(f"/v1/nodes/{ivan_id}", params={"scope": SCOPE, "budget_tokens": 2000}).json()
    assert [p["value"] for p in seen["proposals"]] == ["Карим"]
    decided = client.post(f"/v1/proposals/{proposals[0].id}", json={"decision": "accept"})
    assert decided.status_code == 200, decided.text
    assert decided.json()["status"] == "accepted"
    assert set(store.get_node(ivan_id).aliases) == {"Ваня", "Карим"}
    again = client.post(f"/v1/proposals/{proposals[0].id}", json={"decision": "reject"})
    assert again.status_code == 409


def test_a_new_node_keeps_only_the_aliases_its_source_holds(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Émilie Durand", "kind": "person", "aliases": ["Mimi", "Milou"]}],
                facts=[
                    {
                        "subject": "Émilie Durand",
                        "statement": "Émilie Durand préfère les réponses courtes.",
                        "quote": "préfère les réponses courtes",
                    }
                ],
            )
        ]
    )
    post_turn(client, "Émilie Durand, dite Mimi, préfère les réponses courtes.")
    run_all(client, app)
    store = app.state.store
    emilie = store.find_node("Émilie Durand", VIEW)
    assert emilie is not None and emilie.aliases == ("Mimi",)
    assert [p.value for p in store.proposals([emilie.id], VIEW)] == ["Milou"]
    assert store.find_node("Milou", VIEW) is None


def test_a_rejected_proposal_changes_nothing_but_its_status(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[answer(nodes=[{"name": "Karim", "kind": "person", "aliases": ["K."]}])]
    )
    post_turn(client, TURN)
    run_all(client, app)
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None and karim.aliases == ()
    (proposal,) = store.proposals([karim.id], VIEW)
    response = client.post(f"/v1/proposals/{proposal.id}", json={"decision": "reject"})
    assert response.status_code == 200 and response.json()["status"] == "rejected"
    assert store.get_node(karim.id).aliases == ()
    assert store.proposals([karim.id], VIEW) == []
    bad = client.post(f"/v1/proposals/{proposal.id}", json={"decision": "maybe"})
    assert bad.status_code == 422


# --------------------------------------------------------------------------- co-occurrence


def test_co_occurrence_links_only_subjects_of_the_same_sentence(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {"name": "Karim", "kind": "person"},
                    {"name": "Projet solaire", "kind": "project"},
                    {"name": "Émilie", "kind": "person"},
                ],
                facts=[
                    karim_fact(),
                    {
                        "subject": "Émilie",
                        "statement": "Émilie préfère le thé vert.",
                        "quote": "préfère le thé vert",
                    },
                ],
            )
        ]
    )
    post_turn(client, "Karim dirige le projet solaire.", "Noté. Émilie préfère le thé vert.")
    run_all(client, app)
    store = app.state.store
    ids = {name: store.find_node(name, VIEW).id for name in ("Karim", "Projet solaire", "Émilie")}
    pairs = {
        frozenset((e.src, e.dst))
        for e in store.edges_between(list(ids.values()), VIEW)
        if e.type == "co_occurs"
    }
    assert pairs == {frozenset((ids["Karim"], ids["Projet solaire"]))}


def test_an_adjudication_truncated_twice_leaves_every_mention_new(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    remember(client, "Karim Haddad is an accountant", node={"name": "Karim Haddad"})
    remember(client, "Karim Benali is an electrician", node={"name": "Karim Benali"})
    cut = Reply('{"decisions": [{"mention": 0, "choice": "node_', finish_reason="length")
    fake_openai.script(
        chat=[
            answer(facts=[{"subject": "Karim", "statement": "Karim met Nadia.", "quote": "met"}]),
            cut,
            cut,
        ]
    )
    post_turn(client, "Karim met Nadia yesterday.", "ok")
    report = run_all(client, app)
    result = report["jobs"][0]["result"]
    first, second = (c.body["max_tokens"] for c in fake_openai.chat_calls[1:])
    assert second == 2 * first and result["truncated"] == 1
    # the safe default: a new node, never a guess between the two Karims
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None and karim.name == "Karim"


def test_a_proposal_whose_node_left_the_view_is_not_accepted(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[answer(nodes=[{"name": "Karim", "kind": "person", "aliases": ["K."]}])]
    )
    post_turn(client, TURN)
    run_all(client, app)
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None
    (proposal,) = store.proposals([karim.id], VIEW)
    store.archive(karim.id)
    refused = client.post(f"/v1/proposals/{proposal.id}", json={"decision": "accept"})
    assert refused.status_code == 404 and refused.json()["detail"] == "no such node"
    assert store.get_proposal(proposal.id).status == "pending"
    # rejecting needs no node: it only changes the status
    rejected = client.post(f"/v1/proposals/{proposal.id}", json={"decision": "reject"})
    assert rejected.status_code == 200


# --------------------------------------------------------------------------- naming across scripts

SCRIPT_NOTE = "script in which their subject's name does not appear"


@pytest.mark.parametrize(
    ("source", "node", "subject", "statement"),
    [
        # a real model's answer: an English sentence stored under Иван without naming him
        ("Иван works on the solar project.", "Иван", "Иван", "works on the solar project"),
        # a real model's answer: an Arabic fragment stored under Karim (a Latin name)
        (
            "كريم يقود تخطيط تركيب المشروع الشمسي.",
            "Karim",
            "Karim",
            "يقود تخطيط تركيب المشروع الشمسي",
        ),
    ],
    ids=["english-under-ivan", "arabic-under-karim"],
)
def test_a_statement_that_names_no_one_in_its_script_is_a_fragment(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    source: str,
    node: str,
    subject: str,
    statement: str,
) -> None:
    reply = answer(
        nodes=[{"name": node, "kind": "person"}],
        facts=[{"subject": subject, "statement": statement, "quote": statement}],
    )
    fake_openai.script(chat=[reply, reply])
    post_turn(client, source)
    result = run_all(client, app)["jobs"][0]["result"]
    assert SCRIPT_NOTE in fake_openai.chat_calls[1].messages[0]["content"]
    assert result["facts"]["written"] == 0
    assert result["facts"]["incomplete"] == 1 and result["facts"]["cross_script"] == 1
    assert app.state.store.counts()["facts"] == 0


def test_a_retry_that_supplies_the_source_spelling_keeps_the_fact(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The first answer names Karim only in Latin; the retry note says why, and the
    second answer lists the source's spelling "Карим" among Karim's aliases."""
    source = "Karim leads the solar project. Карим переехал в Лион."
    facts = [
        {
            "subject": "Karim",
            "statement": "Karim leads the solar project.",
            "quote": "Karim leads the solar project",
        },
        {
            "subject": "Karim",
            "statement": "Карим переехал в Лион.",
            "quote": "Карим переехал в Лион",
        },
    ]
    fake_openai.script(
        chat=[
            answer(nodes=[{"name": "Karim", "kind": "person"}], facts=facts),
            answer(nodes=[{"name": "Karim", "kind": "person", "aliases": ["Карим"]}], facts=facts),
        ]
    )
    post_turn(client, source)
    result = run_all(client, app)["jobs"][0]["result"]
    note = fake_openai.chat_calls[1].messages[0]["content"]
    assert SCRIPT_NOTE in note and "list that spelling in the subject's aliases" in note
    assert result["retries"] == 1 and result["facts"]["written"] == 2
    assert result["facts"]["incomplete"] == 0 and result["facts"]["cross_script"] == 0
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None and karim.aliases == ("Карим",)
    assert {f.statement for f in app.state.store.facts([karim.id], VIEW)} == {
        "Karim leads the solar project.",
        "Карим переехал в Лион.",
    }


def test_a_mention_of_the_answer_that_resolves_to_the_subject_names_it(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The subject is written "Karim", the node is "Karim Benali": it reaches
    the node only through resolution (the words of one known name, SPEC 5.1),
    and so does the edge end "Карим" (the node's stored alias). Both are
    mentions of the same node in this answer, so "Карим" names the subject in
    the statement's script; no retry is needed. (A typo would not do: it is
    only a candidate for the adjudication, which comes after this check.)"""
    remember(client, "Karim lives in Lyon.", node={"name": "Karim Benali", "aliases": ["Карим"]})
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Лион", "kind": "place"}],
                facts=[
                    {
                        "subject": "Karim",
                        "statement": "Карим переехал в Лион.",
                        "quote": "Карим переехал в Лион",
                    }
                ],
                edges=[{"src": "Карим", "dst": "Лион", "type": "lives in"}],
            )
        ]
    )
    post_turn(client, "Карим переехал в Лион.")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["retries"] == 0 and result["facts"]["written"] == 1


# --------------------------------------------------------------------------- aliases of someone else

CORPUS_1 = (
    "Youssef 00 prefers short answers (note 3). Léa 00 dislikes early meetings (note 4). "
    "Иван 00 works on the solar project (note 5)."
)


def test_an_alias_that_names_another_node_of_the_answer_is_a_proposal(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """On the manual run's corpus:1, qwen3.5 attaches "lea 00" and "youssef 00" to
    Иван. Both are in the source, but they are the names of two other nodes of the
    same answer: proposals (merge candidates), never aliases (SPEC 5.1)."""
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {"name": "Youssef 00", "kind": "person"},
                    {"name": "Léa 00", "kind": "person"},
                    {"name": "Иван 00", "kind": "person", "aliases": ["lea 00", "youssef 00"]},
                ],
                facts=[
                    {
                        "subject": "Youssef 00",
                        "statement": "Youssef 00 prefers short answers.",
                        "quote": "Youssef 00 prefers short answers",
                    },
                    {
                        "subject": "Léa 00",
                        "statement": "Léa 00 dislikes early meetings.",
                        "quote": "Léa 00 dislikes early meetings",
                    },
                ],
            ),
            answer(nodes=[{"name": "Youssef 00", "kind": "person"}]),
        ]
    )
    post_turn(client, CORPUS_1)
    result = run_all(client, app)["jobs"][0]["result"]
    store = app.state.store
    ivan = store.find_node("Иван 00", VIEW)
    youssef = store.find_node("Youssef 00", VIEW)
    lea = store.find_node("Léa 00", VIEW)
    assert ivan is not None and youssef is not None and lea is not None
    assert len({ivan.id, youssef.id, lea.id}) == 3
    assert ivan.aliases == ()
    assert sorted(p.value for p in store.proposals([ivan.id], VIEW)) == ["lea 00", "youssef 00"]
    assert result["aliases"] == {
        "kept": 0,
        "proposed": 2,
        "invented": 0,
        "descriptive": 0,
        "near": 0,
    }
    assert [f.statement for f in store.facts([youssef.id], VIEW)] == [
        "Youssef 00 prefers short answers."
    ]
    # resolution still finds Youssef, never Иван
    post_turn(client, "Youssef 00 called.")
    run_all(client, app)
    again = store.find_node("Youssef 00", VIEW)
    assert again is not None and again.id == youssef.id


def test_an_alias_that_names_a_known_node_is_a_proposal(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    nadia = remember(client, "Nadia works on Solar.", node={"name": "Nadia", "kind": "person"})
    fake_openai.script(
        chat=[answer(nodes=[{"name": "Karim", "kind": "person", "aliases": ["Nadia"]}])]
    )
    post_turn(client, "Karim met Nadia yesterday.")
    run_all(client, app)
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None and karim.aliases == ()
    assert [p.value for p in store.proposals([karim.id], VIEW)] == ["Nadia"]
    found = store.find_node("Nadia", VIEW)
    assert found is not None and found.id == nadia["node"]["id"]


def test_an_alias_in_a_script_absent_from_the_source_is_dropped_and_counted(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """ "카림" from a source without Hangul is invention, not a proposal: it would
    only flood the review queue. "Карим", in a script the source does use, stays a
    proposal (it is not in the source itself)."""
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Karim", "kind": "person", "aliases": ["카림", "Карим"]}],
                facts=[
                    {
                        "subject": "Karim",
                        "statement": "Karim leads the solar project.",
                        "quote": "Karim leads the solar project",
                    }
                ],
            )
        ]
    )
    post_turn(client, "Karim leads the solar project with Иван.")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["aliases"] == {
        "kept": 0,
        "proposed": 1,
        "invented": 1,
        "descriptive": 0,
        "near": 0,
    }
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None and karim.aliases == ()
    assert [p.value for p in store.proposals([karim.id], VIEW)] == ["Карим"]


MIXED_NOTE = "mix two scripts in one word"
MIXED = "انتقل يوسف إلى كazan office."
MIXED_SOURCE = "يوسف انتقل إلى كازان في الربيع."


def youssef_answer(statement: str) -> Reply:
    return answer(
        nodes=[{"name": "يوسف", "kind": "person"}],
        facts=[{"subject": "يوسف", "statement": statement, "quote": "يوسف انتقل إلى كازان"}],
    )


def test_a_word_mixing_scripts_is_rejected_then_dropped_and_counted(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """SPEC 4.2: a malformed statement from the trap corpus, asked again once with a
    note naming the cause, still malformed: dropped, counted as mixed_script."""
    fake_openai.script(chat=[youssef_answer(MIXED), youssef_answer(MIXED)])
    post_turn(client, MIXED_SOURCE)
    result = run_all(client, app)["jobs"][0]["result"]
    assert MIXED_NOTE in fake_openai.chat_calls[1].messages[0]["content"]
    assert "كazan" in fake_openai.chat_calls[1].messages[0]["content"]
    assert result["facts"]["written"] == 0 and result["retries"] == 1
    assert result["facts"]["incomplete"] == 1 and result["facts"]["mixed_script"] == 1
    assert result["facts"]["cross_script"] == 0
    assert app.state.store.counts()["facts"] == 0


def test_a_retry_in_one_script_keeps_the_fact(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fixed = "انتقل يوسف إلى كازان في الربيع."
    fake_openai.script(chat=[youssef_answer(MIXED), youssef_answer(fixed)])
    post_turn(client, MIXED_SOURCE)
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["facts"]["written"] == 1 and result["facts"]["mixed_script"] == 0
    stored = app.state.store.facts_by_id(
        [r["id"] for r in app.state.store.execute("SELECT id FROM facts").fetchall()],
        Visibility.of_scope(SCOPE),
    )
    assert [f.statement for f in stored.values()] == [fixed]


def test_a_free_standing_fact_is_checked_too(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    free = answer(facts=[{"subject": None, "statement": MIXED, "quote": "يوسف انتقل"}])
    fake_openai.script(chat=[free, free])
    post_turn(client, MIXED_SOURCE)
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["facts"]["written"] == 0 and result["facts"]["mixed_script"] == 1
