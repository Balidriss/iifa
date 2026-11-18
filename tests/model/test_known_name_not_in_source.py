"""A known subject the source does not name (SPEC 5.1). The store holds one
author's report; a later turn speaks of another author's report by the same
generic words. Whatever path the model's subject takes to the old node - its
exact name copied from the known subjects, a generic word contained in it, or
a name the model made up - the new fact never lands there on its own: the
adjudication sees the source sentence and decides, and what it does not
attach is a fact proposal on the known node (``facts.unnamed``)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.scopes import Visibility
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.tier2 import SCOPE, answer, post_turn, run_all

VIEW = Visibility.of_scope(SCOPE)
OLD = "Rapport de Nadia sur le budget"
FIRST = f"J'ai relu le {OLD} : le résumé fait 2 pages."
OLD_FACT = {
    "subject": OLD,
    "statement": f"Le résumé du {OLD} fait 2 pages.",
    "attribute": "summary_length",
    "value": "2 pages",
    "quote": "le résumé fait 2 pages",
}
SECOND = (
    "Relis le résumé de ce rapport de Bruno. "
    "Le tableau des chiffres est en annexe, à la fin du document."
)
QUOTE = "Le tableau des chiffres est en annexe"
NEW = chat_json({"decisions": [{"mention": 0, "choice": "new"}]})


def attach(node_id: str) -> Any:
    return chat_json({"decisions": [{"mention": 0, "choice": node_id}]})


def lesson(subject: str, **extra: Any) -> dict[str, Any]:
    return {
        "subject": subject,
        "statement": "Le tableau des chiffres du rapport est en annexe.",
        "attribute": "table_location",
        "value": "en annexe",
        "quote": QUOTE,
        **extra,
    }


def ingest(
    client: TestClient,
    app: FastAPI,
    fake: FakeOpenAIServer,
    text: str,
    *replies: Any,
    **extra: Any,
) -> Any:
    fake.script(chat=list(replies))
    post_turn(client, text, **extra)
    jobs = run_all(client, app)["jobs"]
    return next(j["result"] for j in jobs if j["kind"] == "turn")


def old_node(app: FastAPI) -> Any:
    node = app.state.store.find_node(OLD, VIEW)
    assert node is not None
    return node


def statements(app: FastAPI, node_id: str) -> list[str]:
    return sorted(f.statement for f in app.state.store.facts([node_id], VIEW))


def fact_proposals(app: FastAPI) -> list[tuple[str, str]]:
    rows = app.state.store.execute(
        "SELECT p.node_id, p.value FROM proposals p"
        " WHERE p.kind = 'fact' AND p.status = 'pending' ORDER BY p.value"
    ).fetchall()
    return [(r["node_id"], r["value"]) for r in rows]


def adjudication_prompt(fake: FakeOpenAIServer) -> str:
    """The user message of the last adjudication call."""
    calls = [c for c in fake.chat_calls if "Decide, for each mention" in json.dumps(c.messages)]
    assert calls, "no adjudication call"
    return str(calls[-1].messages[-1]["content"])


@pytest.fixture
def known(client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer) -> str:
    ingest(
        client,
        app,
        fake_openai,
        FIRST,
        answer(nodes=[{"name": OLD, "kind": "object"}], facts=[OLD_FACT]),
    )
    return str(old_node(app).id)


# ------------------------------------------------------------------ the shape


@pytest.mark.parametrize(
    "subject",
    [
        OLD,  # the known name copied from the prompt's known subjects
        "rapport",  # a generic word, whole words of exactly one known name
        "rapport de Nadia",  # whole words of it too, and not what the source says
    ],
)
def test_a_fact_about_another_report_never_lands_on_the_known_one(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    known: str,
    subject: str,
) -> None:
    result = ingest(client, app, fake_openai, SECOND, answer(facts=[lesson(subject)]), NEW)
    assert statements(app, known) == [OLD_FACT["statement"]]
    assert fact_proposals(app) == [(known, lesson(subject)["statement"])]
    assert result["facts"]["unnamed"] == 1 and result["facts"]["written"] == 0
    assert result["adjudication_calls"] == 1
    # the adjudication saw the sentence the fact quotes and the known report
    prompt = adjudication_prompt(fake_openai)
    assert QUOTE in prompt and f"candidate {known}: {OLD}" in prompt


def test_no_node_is_created_under_a_name_memory_already_holds(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, known: str
) -> None:
    ingest(client, app, fake_openai, SECOND, answer(facts=[lesson(OLD)]), NEW)
    rows = app.state.store.execute("SELECT id FROM nodes WHERE name = ?", (OLD,)).fetchall()
    assert [r["id"] for r in rows] == [known]


def test_a_truncated_or_missing_adjudication_attaches_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, known: str
) -> None:
    """An answer that is no decision leaves the mention undecided: a proposal."""
    junk = chat_json({"decisions": []})
    result = ingest(client, app, fake_openai, SECOND, answer(facts=[lesson(OLD)]), junk)
    assert statements(app, known) == [OLD_FACT["statement"]]
    assert result["facts"]["unnamed"] == 1


def test_the_answer_node_copied_from_memory_gives_it_no_alias(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, known: str
) -> None:
    """The answer proposes the known report with the other report's words as an
    alias the source holds: kept, it would resolve every later mention of
    Bruno's report to Nadia's. It is an alias proposal at most."""
    reply = answer(
        nodes=[{"name": OLD, "kind": "object", "aliases": ["rapport de Bruno"]}],
        facts=[lesson(OLD, subject_as_written="rapport de Bruno")],
    )
    ingest(client, app, fake_openai, SECOND, reply, NEW)
    assert old_node(app).aliases == ()
    assert statements(app, known) == [OLD_FACT["statement"]]


def test_an_edge_to_a_known_subject_the_source_does_not_name_is_not_written(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, known: str
) -> None:
    reply = answer(
        nodes=[{"name": "Bruno", "kind": "person"}],
        edges=[{"src": "Bruno", "dst": OLD, "type": "authored"}],
    )
    ingest(client, app, fake_openai, SECOND, reply, NEW)
    rows = app.state.store.execute(
        "SELECT id FROM edges WHERE src = ? OR dst = ?", (known, known)
    ).fetchall()
    assert rows == []


# ------------------------------------------------------------------ what still attaches


def test_the_adjudication_may_attach_a_subject_the_source_does_not_name(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, known: str
) -> None:
    """A name the model translated, or a pronoun: the adjudication reads the
    sentence and attaches it."""
    text = "Il a été mis à jour hier : le tableau des chiffres est en annexe."
    result = ingest(client, app, fake_openai, text, answer(facts=[lesson(OLD)]), attach(known))
    assert statements(app, known) == sorted([OLD_FACT["statement"], lesson(OLD)["statement"]])
    assert result["facts"]["unnamed"] == 0 and result["facts"]["written"] == 1


def test_a_source_that_names_the_subject_needs_no_adjudication(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, known: str
) -> None:
    text = f"Sur le {OLD}, {QUOTE}."
    result = ingest(client, app, fake_openai, text, answer(facts=[lesson(OLD)]))
    assert result["adjudication_calls"] == 0 and result["facts"]["written"] == 1


def test_a_subject_named_earlier_in_the_session_needs_no_adjudication(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ingest(
        client,
        app,
        fake_openai,
        FIRST,
        answer(nodes=[{"name": OLD, "kind": "object"}], facts=[OLD_FACT]),
        session_id="s1",
    )
    text = f"Et son tableau ? {QUOTE}."
    result = ingest(client, app, fake_openai, text, answer(facts=[lesson(OLD)]), session_id="s1")
    assert result["adjudication_calls"] == 0 and result["facts"]["written"] == 1
    # another session never saw it named: the adjudication decides there
    other = ingest(
        client, app, fake_openai, text, answer(facts=[lesson(OLD)]), NEW, session_id="s2"
    )
    assert other["adjudication_calls"] == 1 and other["facts"]["unnamed"] == 1


def test_a_first_name_alone_in_a_new_source_is_adjudicated_not_attached_alone(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """ "Karim" is whole words of "Karim Benali", as "rapport" is of the report's
    name: across sources, a word of a name does not name the node."""
    benali = {"name": "Karim Benali", "kind": "person"}
    lives = {
        "subject": "Karim Benali",
        "statement": "Karim Benali lives in Lyon.",
        "quote": "Karim Benali lives in Lyon",
    }
    ingest(
        client,
        app,
        fake_openai,
        "Karim Benali lives in Lyon.",
        answer(nodes=[benali], facts=[lives]),
    )
    karim = app.state.store.find_node("Karim Benali", VIEW)
    tea = {"subject": "Karim", "statement": "Karim prefers tea.", "quote": "Karim prefers tea"}
    result = ingest(
        client, app, fake_openai, "Karim prefers tea.", answer(facts=[tea]), attach(karim.id)
    )
    assert result["adjudication_calls"] == 1 and result["facts"]["written"] == 1
    assert "Karim prefers tea." in statements(app, karim.id)


def test_a_name_in_a_script_written_without_spaces_names_its_node(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    kenji = {"name": "健二", "kind": "person"}
    first = {
        "subject": "健二",
        "statement": "健二はベルリンに住んでいる。",
        "quote": "健二はベルリンに住んでいる",
    }
    ingest(
        client,
        app,
        fake_openai,
        "健二はベルリンに住んでいる。",
        answer(nodes=[kenji], facts=[first]),
    )
    later = {"subject": "健二", "statement": "健二は紅茶が好きだ。", "quote": "健二は紅茶が好きだ"}
    result = ingest(client, app, fake_openai, "健二は紅茶が好きだ。", answer(facts=[later]))
    assert result["adjudication_calls"] == 0 and result["facts"]["written"] == 1
