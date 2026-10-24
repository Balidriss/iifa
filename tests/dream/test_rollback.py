"""Must not make recall worse (SPEC 5.6 rule 9, 11.2): the probe sets of the
run's scope and of the scopes below it are scored before and after; a
regression or an inconclusive comparison rolls the run back."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.recall.probes import run_probe_set
from tests.dream.conftest import (
    MEDIUM_AUTO,
    SCOPE,
    Memory,
    build_memory,
    dream,
    dream_model,
    plan_of,
)
from tests.fakes.openai_server import FakeOpenAIServer

# what the report shows per set, before and after (SPEC 5.6 rule 9)
NUMBERS = {"recall", "recall_per_1k_tokens", "compared_per_1k_tokens", "summary_tokens", "tier"}
LONG_SUMMARY = (
    "Karim lives in Lyon, likes tea in the morning and drinks green tea every morning; "
    "he is the one everybody calls K. and he keeps a pinned note about his routine."
)


@pytest.fixture
def dream_review() -> tuple[tuple[str, str, str], ...]:
    """These tests apply consolidations: the operator lets ``medium`` apply."""
    return MEDIUM_AUTO


def _probe_set(
    client: TestClient, scope: str, query: str, expected: list[str], budget: int = 400
) -> str:
    response = client.post(
        "/v1/probes",
        json={
            "scope": scope,
            "name": "rollback",
            "budget_tokens": budget,
            "probes": [{"query": query, "expected": expected}],
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _consolidation(memory: Memory, statement: str) -> Any:
    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [
            {
                "op": "consolidate_facts",
                "node": memory.karim.id,
                "facts": [memory.tea.id, memory.green_tea.id],
                "statement": statement,
            }
        ]

    return dream_model(plan)


def test_a_dream_that_keeps_recall_is_kept_and_reports_the_scores(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    set_id = _probe_set(dream_client, SCOPE, "green tea morning", [memory.green_tea.id])
    fake_openai.set_default(chat=_consolidation(memory, "Karim drinks green tea every morning."))
    run = dream(dream_client)
    assert run["status"] == "finished"
    probes = run["report"]["probes"]
    assert probes["measured"] is True
    [measured] = probes["sets"]
    assert measured["set_id"] == set_id
    assert measured["verdict"] == "ok"
    assert set(measured["before"]) == NUMBERS
    assert "probes" not in measured  # numbers only, never the per-probe ids
    assert dream_app.state.store.get_fact(memory.tea.id).consolidated_into is not None


def _summaries(summary: str) -> Any:
    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["name"] != "Karim":
            return []
        return [{"op": "refresh_summary", "node": unit["node"]["id"], "summary": summary}]

    return dream_model(plan)


def test_a_dream_that_only_refreshes_a_summary_is_kept(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A first summary costs more than the line it replaces: priced as that line,
    a bounded summary never rolls a dream back on its own; both numbers are
    shown, so its price stays visible."""
    store = dream_app.state.store
    memory = build_memory(store)
    _probe_set(dream_client, SCOPE, "Karim", [memory.karim.id], budget=1000)
    fake_openai.set_default(chat=_summaries(LONG_SUMMARY))
    run = dream(dream_client)
    assert run["status"] == "finished", run["report"]["probes"]
    [measured] = run["report"]["probes"]["sets"]
    before, after = measured["before"], measured["after"]
    assert set(before) == set(after) == NUMBERS
    assert after["recall_per_1k_tokens"] < before["recall_per_1k_tokens"] * 0.98  # the raw price
    assert after["summary_tokens"] > 0 == before["summary_tokens"]
    assert measured["verdict"] == "ok"
    assert store.get_node(memory.karim.id).summary == LONG_SUMMARY  # stored whole

    # a second dream rewrites it: both scores set the summaries aside alike
    fake_openai.set_default(chat=_summaries(LONG_SUMMARY.replace("Lyon", "Lyon (France)")))
    again = dream(dream_client)
    assert again["status"] == "finished", again["report"]["probes"]
    [measured] = again["report"]["probes"]["sets"]
    assert measured["before"]["summary_tokens"] > 0 and measured["verdict"] == "ok"


def test_a_dream_that_lowers_recall_rolls_itself_back(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    _probe_set(dream_client, SCOPE, "green tea", [memory.green_tea.id])
    fake_openai.set_default(chat=_consolidation(memory, "Karim enjoys a hot drink at dawn."))
    run = dream(dream_client)
    assert run["status"] == "rolled_back"
    assert run["report"]["rolled_back"] == {"reason": "regressed", "incomplete": []}
    assert store.get_fact(memory.green_tea.id).valid_to is None
    assert store.get_fact(memory.green_tea.id).consolidated_into is None
    assert [i["status"] for i in plan_of(dream_client, run["run_id"])] == ["undone"]


def test_an_inconclusive_comparison_rolls_the_run_back(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Other probes scored after than before (one lost its answer to a forget in
    between): a dream that cannot be shown harmless is not kept."""
    store = dream_app.state.store
    memory = build_memory(store)
    response = dream_client.post(
        "/v1/probes",
        json={
            "scope": SCOPE,
            "name": "two",
            "probes": [
                {"query": "green tea", "expected": [memory.green_tea.id]},
                {"query": "Lyon", "expected": [memory.lyon.id]},
            ],
        },
    )
    assert response.status_code == 201
    fake_openai.set_default(chat=_consolidation(memory, "Karim drinks green tea every morning."))
    dream_app.state.dreamer._before_probes_after = lambda: store.forget(memory.lyon.id)
    run = dream(dream_client)
    assert run["status"] == "rolled_back"
    assert run["report"]["rolled_back"]["reason"] == "inconclusive"
    assert store.get_fact(memory.tea.id).consolidated_into is None


def test_a_rollback_names_what_a_later_write_keeps(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    _probe_set(dream_client, SCOPE, "green tea", [memory.green_tea.id])
    fake_openai.set_default(chat=_consolidation(memory, "Karim enjoys a hot drink at dawn."))

    def owner_pins_the_new_fact() -> None:
        new = store.get_fact(memory.tea.id).consolidated_into
        store.attach_fact(new, memory.karim.id, pinned=True)

    dream_app.state.dreamer._before_probes_after = owner_pins_the_new_fact
    run = dream(dream_client)
    assert run["status"] == "rolled_back"
    rolled = run["report"]["rolled_back"]
    assert rolled["reason"] == "regressed"
    assert rolled["incomplete"] == [plan_of(dream_client, run["run_id"])[0]["id"]]


def test_queued_operations_of_a_rolled_back_run_are_dropped(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    _probe_set(dream_client, SCOPE, "green tea", [memory.green_tea.id])
    store.add_proposal(scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id)

    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [
            {
                "op": "consolidate_facts",
                "node": memory.karim.id,
                "facts": [memory.tea.id, memory.green_tea.id],
                "statement": "Karim enjoys a hot drink at dawn.",
            },
        ]

    fake_openai.set_default(
        chat=dream_model(plan, lambda pair: {"verdict": "same", "reason": "one person"})
    )
    run = dream(dream_client)
    assert run["status"] == "rolled_back"
    statuses = [(i["op"], i["status"], i["reason"]) for i in plan_of(dream_client, run["run_id"])]
    assert statuses == [
        ("merge_nodes", "dropped", "rolled_back"),
        ("consolidate_facts", "undone", None),
    ]


def test_probe_sets_below_the_scope_are_measured_too(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    _probe_set(dream_client, f"{SCOPE}/agent:coder", "green tea", [memory.green_tea.id])
    fake_openai.set_default(chat=_consolidation(memory, "Karim enjoys a hot drink at dawn."))
    run = dream(dream_client)
    assert run["status"] == "rolled_back"


def test_without_a_probe_set_nothing_is_measured_and_the_report_says_so(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    fake_openai.set_default(chat=_consolidation(memory, "Karim enjoys a hot drink at dawn."))
    run = dream(dream_client)
    assert run["status"] == "finished"
    assert run["report"]["probes"] == {"measured": False, "sets": []}


def test_a_dry_run_measures_nothing(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    _probe_set(dream_client, SCOPE, "green tea", [memory.green_tea.id])
    fake_openai.set_default(chat=_consolidation(memory, "Karim enjoys a hot drink at dawn."))
    run = dream(dream_client, dry_run=True)
    assert run["status"] == "finished"
    assert run["report"]["probes"]["measured"] is False


def test_a_probe_finds_what_a_dream_consolidated_or_merged(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    """SPEC 11.2: an expected fact is found through ``consolidated_into``, an
    expected node through ``merged_into``."""
    store, engine = dream_app.state.store, dream_app.state.engine
    memory = build_memory(store)
    new = store.add_fact(
        scopes=[SCOPE],
        node_id=memory.karim.id,
        statement="Karim drinks green tea daily.",
        origin="extracted",
    )
    probe_set = store.create_probe_set(
        scope=SCOPE,
        name="follow",
        budget_tokens=400,
        probes=[("green tea daily", [memory.green_tea.id]), ("Kiki", [memory.kiki.id])],
    )
    store.consolidate_facts([memory.green_tea.id], into=new.id)
    store.update_node(memory.karim.id, add_aliases=["Kiki"])  # as a merge does
    store.merge_node(memory.kiki.id, into=memory.karim.id)
    score = run_probe_set(engine, store, probe_set)
    assert score.recall == 1.0
