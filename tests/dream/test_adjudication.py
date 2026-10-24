"""Merge candidates and their adjudication (SPEC 5.6): a merge is never decided
on a name - one model call per pair reads both nodes whole and answers a closed
verdict; ``same`` queues a ``merge_nodes`` for review, ``different`` changes
nothing, ``unsure`` leaves a visible proposal; the verdict and its reason are
shown to the reviewer beside the pair (SPEC 4.4b, 8.6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.dream.adjudicate import names_close, names_fit
from iifa.scopes import Visibility
from iifa.store.repo import NAME, node_name_text
from tests.conftest import FakeClock
from tests.dream.conftest import (
    MEMORY_END,
    MEMORY_START,
    SCOPE,
    Memory,
    adjudication_calls,
    build_memory,
    dream,
    dream_model,
    dream_report,
    dream_settings,
    plan_of,
    quiet_model,
    run_queue,
    start_dream,
    unit_calls,
)
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth


@pytest.fixture
def memory(dream_app: FastAPI) -> Memory:
    return build_memory(dream_app.state.store)


def _propose(app: FastAPI, a: str, b: str) -> Any:
    return app.state.store.add_proposal(scopes=[SCOPE], kind="merge", node_id=a, value=b)


def _merges(app: FastAPI) -> list[Any]:
    proposals: list[Any] = app.state.store.pending_proposals(
        Visibility.of_scope(SCOPE), kind="merge"
    )
    return proposals


def _says(verdict: str, reason: str = "because") -> Any:
    return dream_model(lambda unit: [], lambda pair: {"verdict": verdict, "reason": reason})


def _review(client: TestClient) -> list[dict[str, Any]]:
    response = client.get(
        "/v1/review", params={"scope": SCOPE, "budget_tokens": 16000}, headers=auth("review")
    )
    assert response.status_code == 200, response.text
    items: list[dict[str, Any]] = response.json()["items"]
    return items


def _merge_items(client: TestClient) -> list[dict[str, Any]]:
    return [
        i
        for i in _review(client)
        if i["kind"] == "merge" or (i["kind"] == "dream_operation" and i["op"] == "merge_nodes")
    ]


# --------------------------------------------------------------------------- the call


def test_each_side_is_a_timeline_per_key_learned_in_one_order(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The model read a change over time as a contradiction ("architect vs
    accountant"): it sees, per key, each value with when memory learned it - one
    order for A and B together - its date when known, and what it replaced."""
    store = dream_app.state.store
    ivan_ru = store.create_node(scopes=[SCOPE], name="Иван", kind="person")
    ivan = store.create_node(scopes=[SCOPE], name="Ivan", kind="person")

    def add(node: Any, statement: str, key: str | None, value: str | None, **kw: Any) -> Any:
        return store.add_fact(
            scopes=[SCOPE],
            node_id=node.id,
            statement=statement,
            origin="extracted",
            attribute=key,
            value=value,
            **kw,
        )

    add(ivan_ru, "Иван работает архитектором.", "occupation", "архитектор")
    add(ivan_ru, "Иван живёт в Казани.", "home", "Казань")
    add(ivan, "Ivan works as an accountant.", "occupation", "accountant", valid_from="2026-01")
    paris = add(ivan, "Ivan lives in Paris.", "home", "Paris")
    add(ivan, "Ivan likes hiking.", None, None)
    lyon = add(ivan, "Ivan lives in Lyon.", "home", "Lyon")
    store.close_fact(paris.id, superseded_by=lyon.id)
    _propose(dream_app, ivan_ru.id, ivan.id)
    fake_openai.set_default(chat=_says("unsure"))
    dream(dream_client)
    [pair] = adjudication_calls(fake_openai)
    sides = {pair["a"]["name"]: pair["a"], pair["b"]["name"]: pair["b"]}
    assert sides["Иван"]["timelines"] == {
        "home": "Казань (learned 2nd)",
        "occupation": "архитектор (learned 1st)",
    }
    assert sides["Ivan"]["timelines"] == {
        "home": "Paris (learned 4th) -> Lyon (learned 6th)",
        "occupation": "accountant (learned 3rd, from 2026-01)",
    }
    assert sides["Ivan"]["other"] == ["Ivan likes hiking. (learned 5th)"]
    assert "other" not in sides["Иван"]
    assert "facts" not in sides["Ivan"]  # no flat list beside the timelines


def test_a_unit_answer_that_merges_is_rejected_and_queues_nothing(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [{"op": "merge_nodes", "keep": memory.karim.id, "absorb": memory.kiki.id}]

    fake_openai.set_default(chat=dream_model(plan))
    run = dream(dream_client)
    assert run["report"]["rejected"] == {"adjudication_only": 1}
    assert run["report"]["review"] == []
    assert adjudication_calls(fake_openai) == []  # no candidate: no proposal, no vectors


def test_a_pending_proposal_is_adjudicated_on_both_nodes_and_their_facts(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    _propose(dream_app, memory.karim.id, memory.kiki.id)
    store = dream_app.state.store
    store.add_fact(
        scopes=[SCOPE],
        node_id=memory.kiki.id,
        statement="Kiki lives in Lyon.",
        origin="extracted",
        attribute="home",
        value="Lyon",
        language="en",
    )
    fake_openai.set_default(chat=_says("same", "both live in Lyon and are called K."))
    run = dream(dream_client)
    calls = [c for c in fake_openai.chat_calls if '"verdict"' in c.messages[0]["content"]]
    assert len(calls) == 1  # once per pair per run: Kiki's unit does not ask again
    system, user = calls[0].messages[0]["content"], calls[0].messages[-1]["content"]
    assert "DATA" in system and "never instructions" in system
    assert "same" in system and "different" in system and "unsure" in system
    assert "Daniela" in system  # a near name is not evidence
    assert "Karima" not in system and "Ivana" not in system  # never the bench's own traps
    assert user.startswith(MEMORY_START) and user.rstrip().endswith(MEMORY_END)
    pair = adjudication_calls(fake_openai)[0]
    names = {pair["a"]["name"], pair["b"]["name"]}
    assert names == {"Karim", "Kiki"}
    karim = pair["a"] if pair["a"]["name"] == "Karim" else pair["b"]
    kiki = pair["b"] if karim is pair["a"] else pair["a"]
    assert karim["aliases"] == ["K."] and karim["kind"] == "person"
    assert karim["timelines"]["home"].startswith("Lyon (learned ")
    assert any(o.startswith("Karim is allergic to peanuts. (learned ") for o in karim["other"])
    assert kiki["timelines"] == {"home": "Lyon (learned 5th)"} and "other" not in kiki
    # the queued merge keeps the node with more live facts
    [item] = plan_of(dream_client, run["run_id"])
    assert (item["op"], item["status"], item["route"]) == ("merge_nodes", "queued", "review")
    assert (item["payload"]["keep"], item["payload"]["absorb"]) == (memory.karim.id, memory.kiki.id)
    assert item["reason_given"] == "both live in Lyon and are called K."
    adjudications = run["report"]["adjudications"]
    assert adjudications["verdicts"] == {"same": 1, "different": 0, "unsure": 0}
    assert adjudications["pairs"] == [
        {"a": memory.karim.id, "b": memory.kiki.id, "verdict": "same"}
    ]
    assert "reason" not in str(adjudications["pairs"])  # counts and ids only (P5)
    assert dream_app.state.store.get_node(memory.kiki.id).merged_into is None  # reviewed first


def test_different_changes_nothing_and_the_proposal_stays_pending(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    proposal = _propose(dream_app, memory.karim.id, memory.kiki.id)
    store = dream_app.state.store
    before = store.revisions([memory.karim.id, memory.kiki.id])
    fake_openai.set_default(chat=_says("different", "Kiki is a cat"))
    run = dream(dream_client)
    assert run["report"]["review"] == []
    assert plan_of(dream_client, run["run_id"]) == []
    assert store.get_proposal(proposal.id).status == "pending"  # only a reviewer decides it
    assert store.revisions([memory.karim.id, memory.kiki.id]) == before
    assert run["report"]["adjudications"]["verdicts"]["different"] == 1


def test_unsure_on_a_pair_without_proposal_leaves_one(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    dream_app.state.dreamer.name_neighbours = lambda node_id: (
        [(memory.kiki.id, 0.7)] if node_id == memory.karim.id else []
    )
    fake_openai.set_default(chat=_says("unsure", "nothing to tell them apart"))
    run = dream(dream_client)
    proposals = _merges(dream_app)
    assert [(p.node_id, p.value) for p in proposals] == [(memory.karim.id, memory.kiki.id)]
    assert run["report"]["adjudications"]["proposals_created"] == 1
    [item] = _merge_items(dream_client)
    assert item["kind"] == "merge" and item["id"] == proposals[0].id
    assert item["adjudication"]["verdict"] == "unsure"
    assert item["adjudication"]["reason"] == "nothing to tell them apart"


def test_unsure_on_a_proposed_pair_creates_no_second_proposal(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    _propose(dream_app, memory.kiki.id, memory.karim.id)
    fake_openai.set_default(chat=_says("unsure"))
    run = dream(dream_client)
    assert len(_merges(dream_app)) == 1
    assert run["report"]["adjudications"]["proposals_created"] == 0


# --------------------------------------------------------------------------- names that fit


def _story(store: Any) -> tuple[Any, Any]:
    """Two people whose facts read like one life: Ivan in Kazan, Youssef from
    Tunis, both on one team - the shape that drew a `same` from qwen3.5:9b."""
    ivan = store.create_node(scopes=[SCOPE], name="Иван", kind="person")
    youssef = store.create_node(scopes=[SCOPE], name="يوسف", kind="person")
    for node, statement, key, value, language in (
        (youssef, "يوسف يعيش في تونس.", "home", "تونس", "ar"),
        (youssef, "يوسف صحفي.", "role", "صحفي", "ar"),
        (ivan, "Иван живёт в Казани.", "home", "Казань", "ru"),
        (ivan, "Иван работает архитектором.", "role", "архитектор", "ru"),
        (ivan, "Иван работает в команде Атлас.", "team", "Атлас", "ru"),
    ):
        store.add_fact(
            scopes=[SCOPE],
            node_id=node.id,
            statement=statement,
            origin="extracted",
            attribute=key,
            value=value,
            language=language,
        )
    return ivan, youssef


def test_a_same_whose_names_do_not_fit_becomes_unsure(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """SPEC 5.6: the service enforces the half of the rule the model dropped -
    `same` only when the names fit. A downgrade of label, never a merge
    decision: the pair goes to review as `unsure`, with the model's reason."""
    store = dream_app.state.store
    ivan, youssef = _story(store)
    dreamer = dream_app.state.dreamer
    dreamer.name_neighbours = lambda node_id: [(youssef.id, 0.54)] if node_id == ivan.id else []
    dreamer.adjudicator.names_close = lambda a, b: False  # below the calibrated merge
    reason = "the same person who moved from Tunisia to Kazan"
    fake_openai.set_default(chat=_says("same", reason))
    run = dream(dream_client)
    adjudications = run["report"]["adjudications"]
    assert adjudications["verdicts"] == {"same": 0, "different": 0, "unsure": 1}
    assert adjudications["names_unfit"] == 1
    [pair] = adjudications["pairs"]
    assert (pair["verdict"], pair["model_verdict"]) == ("unsure", "same")
    assert [i["op"] for i in plan_of(dream_client, run["run_id"])] == []  # no merge_nodes
    [item] = _merge_items(dream_client)
    assert item["kind"] == "merge" and {item["node_id"], item["value"]} == {ivan.id, youssef.id}
    assert (item["adjudication"]["verdict"], item["adjudication"]["reason"]) == ("unsure", reason)
    assert store.get_node(youssef.id).merged_into is None


@pytest.mark.parametrize(
    ("first", "second", "aliases", "close"),
    [
        ("Atlas", "Атлас", [], False),  # one spelling in two scripts
        ("Orion", "Orion One", ["O1"], False),  # a shared alias
        ("Bob", "Robert", [], True),  # the name cosine reaches the calibrated merge
    ],
)
def test_a_same_whose_names_fit_stays_same(
    dream_client: TestClient,
    dream_app: FastAPI,
    fake_openai: FakeOpenAIServer,
    first: str,
    second: str,
    aliases: list[str],
    close: bool,
) -> None:
    store = dream_app.state.store
    a = store.create_node(scopes=[SCOPE], name=first, kind="project", aliases=aliases)
    b = store.create_node(scopes=[SCOPE], name=second, kind="project", aliases=aliases)
    dreamer = dream_app.state.dreamer
    dreamer.name_neighbours = lambda node_id: [(b.id, 0.6)] if node_id == a.id else []
    dreamer.adjudicator.names_close = lambda x, y: close and {x, y} == {a.id, b.id}
    fake_openai.set_default(chat=_says("same", "one project"))
    run = dream(dream_client)
    adjudications = run["report"]["adjudications"]
    assert adjudications["verdicts"]["same"] == 1 and adjudications["names_unfit"] == 0
    assert "model_verdict" not in adjudications["pairs"][0]
    [item] = plan_of(dream_client, run["run_id"])
    assert (item["op"], item["status"]) == ("merge_nodes", "queued")


def test_names_are_close_only_at_the_calibrated_merge_threshold(dream_app: FastAPI) -> None:
    """The name cosine counts at ``merge``, the threshold calibrated for the
    model (SPEC 6.1) - not at ``resolve``, which only makes a candidate."""
    store = dream_app.state.store
    bob = store.create_node(scopes=[SCOPE], name="Bob", kind="person")
    robert = store.create_node(scopes=[SCOPE], name="Robert", kind="person")
    roberta = store.create_node(scopes=[SCOPE], name="Roberta", kind="person")
    for node, values in ((bob, [1.0, 0.0]), (robert, [0.95, 0.31]), (roberta, [0.6, 0.8])):
        store.put_vector(node.id, NAME, model="m", text=node_name_text(node), values=values)
    close = names_close(store, "m", 0.9)
    assert close(bob.id, robert.id) and not close(bob.id, roberta.id)
    assert names_fit(bob, robert, close) and not names_fit(bob, roberta, close)
    assert not names_fit(bob, robert, None)  # no calibrated model: spelling and aliases only


# --------------------------------------------------------------------------- candidates


def test_name_vector_candidates_need_the_same_kind_and_identical_scopes(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    store = dream_app.state.store
    shared = store.create_node(scopes=[SCOPE, "team:solar"], name="Karim", kind="person")
    elsewhere = store.create_node(scopes=["user:bob"], name="Karim", kind="person")
    dream_app.state.dreamer.name_neighbours = lambda node_id: (
        [(memory.lyon.id, 0.9), (shared.id, 0.95), (elsewhere.id, 0.99), (memory.kiki.id, 0.6)]
        if node_id == memory.karim.id
        else []
    )
    fake_openai.set_default(chat=_says("different"))
    dream(dream_client)
    pairs = adjudication_calls(fake_openai)
    assert [{p["a"]["name"], p["b"]["name"]} for p in pairs] == [{"Karim", "Kiki"}]


def test_proposals_come_first_and_the_calls_per_unit_are_capped(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = dream_settings(tmp_path, fake_openai, dream_adjudications_per_unit=1)
    app = create_app(settings, clock=clock)
    with TestClient(app, headers=auth("all")) as client:
        memory = build_memory(app.state.store)
        ana = app.state.store.create_node(scopes=[SCOPE], name="Ana", kind="person")
        _propose(app, memory.karim.id, memory.lyon.id)  # two kinds: skipped, costs no call
        _propose(app, memory.karim.id, ana.id)
        app.state.dreamer.name_neighbours = lambda node_id: (
            [(memory.kiki.id, 0.99)] if node_id == memory.karim.id else []
        )
        fake_openai.set_default(chat=_says("different"))
        run = dream(client)
        app.state.store.close()
    # Karim's unit asks about Ana (a proposal, first) and, capped at one, not Kiki
    pairs = adjudication_calls(fake_openai)
    assert [{p["a"]["name"], p["b"]["name"]} for p in pairs] == [{"Karim", "Ana"}]
    assert run["report"]["adjudications"]["skipped"]["kind"] == 1


def test_zero_adjudications_per_unit_turns_merges_off(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = dream_settings(tmp_path, fake_openai, dream_adjudications_per_unit=0)
    app = create_app(settings, clock=clock)
    with TestClient(app, headers=auth("all")) as client:
        memory = build_memory(app.state.store)
        _propose(app, memory.karim.id, memory.kiki.id)
        fake_openai.set_default(chat=_says("same"))
        dream(client)
        app.state.store.close()
    assert adjudication_calls(fake_openai) == []


def test_without_merge_nodes_requested_nothing_is_adjudicated(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    _propose(dream_app, memory.karim.id, memory.kiki.id)
    fake_openai.set_default(chat=_says("same"))
    dream(dream_client, operations=["refresh_summary"])
    assert adjudication_calls(fake_openai) == []
    assert len(unit_calls(fake_openai)) == 3


# --------------------------------------------------------------------------- asking once


def test_a_pair_is_not_asked_again_until_one_side_moves(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    _propose(dream_app, memory.karim.id, memory.kiki.id)
    fake_openai.set_default(chat=_says("different"))
    dream(dream_client)
    again = dream(dream_client)
    assert len(adjudication_calls(fake_openai)) == 1
    assert again["report"]["adjudications"]["skipped"]["unchanged"] == 1
    dream_app.state.store.add_fact(
        scopes=[SCOPE],
        node_id=memory.kiki.id,
        statement="Kiki is a cat.",
        origin="extracted",
        language="en",
    )
    dream(dream_client)
    assert len(adjudication_calls(fake_openai)) == 2  # a new fact: worth asking again


def test_a_pair_a_reviewer_rejected_is_never_asked_again(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    proposal = _propose(dream_app, memory.karim.id, memory.kiki.id)
    response = dream_client.post(
        f"/v1/review/{proposal.id}", json={"decision": "reject"}, headers=auth("review")
    )
    assert response.status_code == 200, response.text
    dream_app.state.dreamer.name_neighbours = lambda node_id: (
        [(memory.kiki.id, 0.9)] if node_id == memory.karim.id else []
    )
    fake_openai.set_default(chat=_says("same"))
    run = dream(dream_client)
    assert adjudication_calls(fake_openai) == []
    assert run["report"]["adjudications"]["skipped"]["rejected"] == 1


def test_a_rejected_merge_operation_is_never_asked_again(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    dream_app.state.dreamer.name_neighbours = lambda node_id: (
        [(memory.kiki.id, 0.9)] if node_id == memory.karim.id else []
    )
    fake_openai.set_default(chat=_says("same"))
    op_id = dream(dream_client)["report"]["review"][0]
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "reject"}, headers=auth("review")
    )
    assert response.status_code == 200, response.text
    dream_app.state.store.add_fact(  # even after one side moved
        scopes=[SCOPE], node_id=memory.kiki.id, statement="Kiki is tall.", origin="extracted"
    )
    dream(dream_client)
    assert len(adjudication_calls(fake_openai)) == 1


def test_a_queued_merge_is_not_asked_again(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    _propose(dream_app, memory.karim.id, memory.kiki.id)
    fake_openai.set_default(chat=_says("same"))
    dream(dream_client)
    run = dream(dream_client)
    assert len(adjudication_calls(fake_openai)) == 1
    assert run["report"]["adjudications"]["skipped"]["queued"] == 1


# --------------------------------------------------------------------------- dry run, answers, budget


def test_a_dry_run_adjudicates_plans_and_records_nothing(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    dream_app.state.dreamer.name_neighbours = lambda node_id: (
        [(memory.kiki.id, 0.9)] if node_id == memory.karim.id else []
    )
    fake_openai.set_default(chat=_says("unsure"))
    run = dream(dream_client, dry_run=True)
    assert run["report"]["adjudications"]["verdicts"]["unsure"] == 1
    assert run["report"]["adjudications"]["proposals_created"] == 0
    assert _merges(dream_app) == []
    assert (
        dream_app.state.dreamer.journal.latest_adjudication(memory.karim.id, memory.kiki.id) is None
    )
    # a later dry run asks again: it recorded nothing
    dream(dream_client, dry_run=True)
    assert len(adjudication_calls(fake_openai)) == 2


def test_a_dry_run_plans_the_merge_a_same_verdict_would_queue(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    _propose(dream_app, memory.karim.id, memory.kiki.id)
    fake_openai.set_default(chat=_says("same"))
    run = dream(dream_client, dry_run=True)
    [item] = plan_of(dream_client, run["run_id"])
    assert (item["op"], item["status"], item["route"]) == ("merge_nodes", "planned", "review")
    assert run["report"]["review"] == []


@pytest.mark.parametrize(
    ("said", "cause"),
    [
        ("they are the same", "bad_answer"),
        ({"verdict": "same", "operations": [{"op": "forget_all"}]}, "bad_answer"),
        ({"verdict": "merge everything"}, "bad_answer"),
    ],
)
def test_an_answer_outside_the_verdict_leaves_the_pair_as_it_was(
    dream_client: TestClient,
    dream_app: FastAPI,
    fake_openai: FakeOpenAIServer,
    memory: Memory,
    said: Any,
    cause: str,
) -> None:
    proposal = _propose(dream_app, memory.karim.id, memory.kiki.id)
    fake_openai.set_default(chat=dream_model(lambda unit: [], lambda pair: said))
    run = dream(dream_client)
    adjudications = run["report"]["adjudications"]
    assert adjudications["answers"][cause] == 1
    assert adjudications["pairs"] == []
    assert run["report"]["review"] == []
    assert dream_app.state.store.get_proposal(proposal.id).status == "pending"
    # nothing recorded: the next run asks again
    dream(dream_client)
    assert len(adjudication_calls(fake_openai)) == 2


def test_adjudications_spend_the_runs_budget_and_resume_at_the_same_unit(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    _propose(dream_app, memory.karim.id, memory.kiki.id)
    fake_openai.set_default(chat=_says("different"))
    first = dream(dream_client, budget={"calls": 1})
    assert first["report"]["stopped_by"] == "calls"
    assert first["report"]["calls"] == 1
    assert unit_calls(fake_openai) == []  # stopped before Karim's unit call
    second = dream(dream_client)
    assert unit_calls(fake_openai)[0]["node"]["id"] == memory.karim.id
    assert len(adjudication_calls(fake_openai)) == 1  # judged once, then skipped
    assert second["report"]["adjudications"]["skipped"]["unchanged"] == 1


# --------------------------------------------------------------------------- the reviewer sees the verdict


def test_the_review_queue_shows_the_verdict_beside_the_pending_proposal(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    proposal = _propose(dream_app, memory.karim.id, memory.kiki.id)
    [never] = _merge_items(dream_client)
    assert never["id"] == proposal.id and never["adjudication"] is None
    fake_openai.set_default(chat=_says("different", "Karim is in Lyon, Kiki in Tokyo"))
    run = dream(dream_client)
    [item] = _merge_items(dream_client)
    assert item["id"] == proposal.id
    assert item["adjudication"]["verdict"] == "different"
    assert item["adjudication"]["reason"] == "Karim is in Lyon, Kiki in Tokyo"
    assert item["adjudication"]["run_id"] == run["run_id"]
    assert item["adjudication"]["at"]


def test_accepting_a_merge_proposal_directly_says_where_merges_happen(
    dream_client: TestClient, dream_app: FastAPI, memory: Memory
) -> None:
    proposal = _propose(dream_app, memory.karim.id, memory.kiki.id)
    accept = dream_client.post(
        f"/v1/proposals/{proposal.id}", json={"decision": "accept"}, headers=auth("write")
    )
    assert accept.status_code == 409
    assert accept.json()["detail"] == (
        "a merge proposal is never accepted directly: a dream's merge_nodes, "
        "approved in review, merges"
    )
    assert dream_app.state.store.get_node(memory.kiki.id).merged_into is None


def test_a_merge_proposal_can_be_rejected_in_review_never_accepted(
    dream_client: TestClient, dream_app: FastAPI, memory: Memory
) -> None:
    proposal = _propose(dream_app, memory.karim.id, memory.kiki.id)
    accept = dream_client.post(
        f"/v1/review/{proposal.id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert accept.status_code == 409
    assert dream_app.state.store.get_node(memory.kiki.id).merged_into is None
    reject = dream_client.post(
        f"/v1/review/{proposal.id}", json={"decision": "reject"}, headers=auth("review")
    )
    assert (reject.status_code, reject.json()["status"]) == (200, "rejected")
    assert _merge_items(dream_client) == []


def test_forget_deletes_the_adjudications_that_read_the_record(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    """P5: a verdict's reason may quote what it read."""
    _propose(dream_app, memory.karim.id, memory.kiki.id)
    fake_openai.set_default(chat=_says("different", "Karim drinks green tea every morning"))
    dream(dream_client)
    journal = dream_app.state.dreamer.journal
    assert journal.latest_adjudication(memory.karim.id, memory.kiki.id) is not None
    response = dream_client.post("/v1/forget", json={"id": memory.green_tea.id})
    assert response.status_code == 200, response.text
    assert response.json()["removed"]["adjudications_deleted"] == 1
    assert journal.latest_adjudication(memory.karim.id, memory.kiki.id) is None
    [item] = _merge_items(dream_client)
    assert item["adjudication"] is None
    store = dream_app.state.store
    raw = store.execute("SELECT count(*) FROM dream_adjudication_records").fetchone()[0]
    assert raw == 0


def test_undoing_a_run_deletes_the_proposals_its_unsure_verdicts_left(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    dream_app.state.dreamer.name_neighbours = lambda node_id: (
        [(memory.kiki.id, 0.9)] if node_id == memory.karim.id else []
    )
    fake_openai.set_default(chat=_says("unsure"))
    run = dream(dream_client)
    assert len(_merges(dream_app)) == 1
    response = dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={}, headers=auth("dream"))
    assert response.status_code == 200, response.text
    assert _merges(dream_app) == []
    # the verdict stays, as a record of what the model said
    assert (
        dream_app.state.dreamer.journal.latest_adjudication(memory.karim.id, memory.kiki.id)
        is not None
    )


def test_a_proposal_left_by_unsure_follows_the_ordinary_rules_once_decided(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    """A reviewer who rejected it is not undone by undoing the run."""
    dream_app.state.dreamer.name_neighbours = lambda node_id: (
        [(memory.kiki.id, 0.9)] if node_id == memory.karim.id else []
    )
    fake_openai.set_default(chat=_says("unsure"))
    run = dream(dream_client)
    store = dream_app.state.store
    [proposal] = _merges(dream_app)
    dream_client.post(
        f"/v1/review/{proposal.id}", json={"decision": "reject"}, headers=auth("review")
    )
    dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={}, headers=auth("dream"))
    assert store.get_proposal(proposal.id).status == "rejected"


def test_a_quiet_run_reports_empty_adjudications(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    run_queue(dream_client)
    report = dream_report(dream_client, run_id)["report"]
    assert report["adjudications"] == {
        "calls": 0,
        "verdicts": {"same": 0, "different": 0, "unsure": 0},
        "answers": {"bad_answer": 0, "truncated": 0},
        "skipped": {"kind": 0, "scope": 0, "unchanged": 0, "rejected": 0, "queued": 0},
        "proposals_created": 0,
        "names_unfit": 0,
        "pairs": [],
    }
