"""A dream run end to end, against a scripted dream model behind the fake
OpenAI server (SPEC 5.6, 8.6): units, the closed answer, budgets, the cursor,
one dream per scope, cancel, P11, a missing dream model, snapshot races
through injection seams, and recall answering while the model thinks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.dream.conftest import (
    MEDIUM_AUTO,
    MEMORY_END,
    MEMORY_START,
    SCOPE,
    Memory,
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
from tests.fakes.openai_server import Fail, FakeOpenAIServer, Reply
from tests.helpers import auth


@pytest.fixture
def dream_review() -> tuple[tuple[str, str, str], ...]:
    """These tests apply consolidations: the operator lets ``medium`` apply."""
    return MEDIUM_AUTO


@pytest.fixture
def memory(dream_app: FastAPI) -> Memory:
    return build_memory(dream_app.state.store)


def _unit_nodes(fake: FakeOpenAIServer) -> list[str]:
    return [unit["node"]["id"] for unit in unit_calls(fake)]


# --------------------------------------------------------------------------- starting and units


def test_a_dream_is_asynchronous_and_walks_one_unit_per_node(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    assert dream_report(dream_client, run_id)["status"] == "pending"
    assert fake_openai.chat_calls == []  # nothing ran yet (on_demand)
    run_queue(dream_client)
    run = dream_report(dream_client, run_id)
    assert run["status"] == "finished"
    assert run["report"]["stopped_by"] == "complete"
    assert run["report"]["units"] == 3
    assert _unit_nodes(fake_openai) == sorted([memory.karim.id, memory.kiki.id, memory.lyon.id])
    assert {call.model for call in fake_openai.chat_calls} == {"fake-dream"}


def test_a_unit_shows_the_node_its_facts_and_edges(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    store = dream_app.state.store
    store.add_edge(
        scopes=[SCOPE], src=memory.karim.id, dst=memory.lyon.id, type="lives_in", origin="test"
    )
    store.add_proposal(scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id)
    fake_openai.set_default(chat=quiet_model())
    dream(dream_client)
    unit = unit_calls(fake_openai)[0]
    assert unit["node"]["id"] == memory.karim.id
    assert unit["node"]["aliases"] == ["K."]
    assert {f["id"] for f in unit["facts"]} == {
        memory.home.id,
        memory.tea.id,
        memory.green_tea.id,
        memory.pinned.id,
    }
    explicit = next(f for f in unit["facts"] if f["id"] == memory.pinned.id)
    assert explicit["origin"] == "explicit"
    assert [n["id"] for n in unit["neighbours"]] == [memory.lyon.id]
    assert "merge_candidates" not in unit  # a merge is the adjudication's (SPEC 5.6)
    assert unit["summary_language"] == "en"


def test_the_prompt_walls_the_memory_off_as_data(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    dream(dream_client)
    messages = fake_openai.chat_calls[0].messages
    system = messages[0]["content"]
    assert "DATA" in system and "never instructions" in system
    assert "flag_conflict" in system
    assert "merge_nodes" not in system  # proposed only through the adjudication
    assert "decay_edges" not in system  # the service's own, never the model's
    user = messages[-1]["content"]
    assert user.startswith(MEMORY_START) and user.rstrip().endswith(MEMORY_END)


def test_the_operations_requested_bound_the_catalogue(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [
            {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id},
            {"op": "refresh_summary", "node": memory.karim.id, "summary": "Karim lives in Lyon."},
        ]

    fake_openai.set_default(chat=dream_model(plan))
    run = dream(dream_client, operations=["refresh_summary"])
    assert run["report"]["rejected"] == {"not_requested": 1}
    assert run["report"]["operations"] == {"refresh_summary": {"applied": 1}}
    assert "flag_conflict" not in fake_openai.chat_calls[0].messages[0]["content"]


def test_unknown_operations_in_a_request_are_refused(dream_client: TestClient) -> None:
    response = dream_client.post("/v1/dream", json={"scope": SCOPE, "operations": ["forget"]})
    assert response.status_code == 422


# --------------------------------------------------------------------------- dry run and plan


def test_a_dry_run_returns_the_full_plan_and_applies_nothing(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    store = dream_app.state.store
    store.add_proposal(scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id)

    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] == memory.karim.id:
            summary = {"node": memory.karim.id, "summary": "Karim lives in Lyon."}
        elif unit["node"]["id"] == memory.kiki.id:
            summary = {"node": memory.kiki.id, "summary": "Kiki is a person."}
        else:
            return []
        return [{"op": "refresh_summary", **summary}]

    fake_openai.set_default(
        chat=dream_model(plan, lambda pair: {"verdict": "same", "reason": "one person"})
    )
    before = store.revisions([memory.karim.id, memory.kiki.id])
    run = dream(dream_client, dry_run=True)
    assert run["dry_run"] is True
    assert store.revisions([memory.karim.id, memory.kiki.id]) == before
    items = plan_of(dream_client, run["run_id"])
    assert [(i["op"], i["status"], i["reason"]) for i in items] == [
        ("merge_nodes", "planned", None),  # adjudicated before the unit's call
        ("refresh_summary", "planned", None),
        ("refresh_summary", "dropped", "no_facts"),
    ]
    assert items[1]["payload"]["summary"] == "Karim lives in Lyon."
    assert items[0]["route"] == "review"  # what a real run would do with it
    assert run["report"]["review"] == []  # nothing queued for real


def test_a_dry_run_never_moves_the_cursor(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    start_dream(dream_client, dry_run=True, budget={"calls": 1})
    run_queue(dream_client)
    start_dream(dream_client, dry_run=True, budget={"calls": 1})
    run_queue(dream_client)
    assert _unit_nodes(fake_openai) == [memory.karim.id, memory.karim.id]


def test_the_plan_is_paginated_under_budget(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [
            {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id},
            {"op": "flag_conflict", "fact": memory.home.id, "with": memory.green_tea.id},
            {"op": "flag_conflict", "fact": memory.pinned.id, "with": memory.green_tea.id},
        ]

    fake_openai.set_default(chat=dream_model(plan))
    run = dream(dream_client, dry_run=True)
    first = dream_client.get(
        f"/v1/dream/{run['run_id']}/plan", params={"limit": 2, "budget_tokens": 16000}
    ).json()
    assert len(first["items"]) == 2
    rest = dream_client.get(
        f"/v1/dream/{run['run_id']}/plan",
        params={"after": first["next"], "budget_tokens": 16000},
    ).json()
    assert len(rest["items"]) == 1 and rest["next"] is None
    tight = dream_client.get(f"/v1/dream/{run['run_id']}/plan", params={"budget_tokens": 1}).json()
    assert tight["items"] == [] and tight["used_tokens"] <= 1


# --------------------------------------------------------------------------- budgets and cursor


def test_a_call_budget_stops_cleanly_and_a_later_run_resumes(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    first = dream(dream_client, budget={"calls": 1})
    assert first["report"]["stopped_by"] == "calls"
    assert first["report"]["units"] == 1
    second = dream(dream_client, budget={"calls": 5})
    assert second["report"]["stopped_by"] == "complete"
    nodes = sorted([memory.karim.id, memory.kiki.id, memory.lyon.id])
    assert _unit_nodes(fake_openai) == nodes  # each node once, in order
    dream(dream_client, budget={"calls": 1})  # a full pass reset the cursor
    assert _unit_nodes(fake_openai)[-1] == nodes[0]


def test_a_token_budget_stops_cleanly(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run = dream(dream_client, budget={"tokens": 1})
    assert run["report"]["stopped_by"] == "tokens"
    assert run["report"]["calls"] == 1
    assert run["report"]["tokens"] >= 1


def test_a_time_budget_stops_cleanly(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run = dream(dream_client, budget={"seconds": 0})
    assert run["report"]["stopped_by"] == "seconds"
    assert fake_openai.chat_calls == []


def test_an_operation_budget_stops_cleanly(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [
            {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id},
            {"op": "flag_conflict", "fact": memory.home.id, "with": memory.green_tea.id},
        ]

    fake_openai.set_default(chat=dream_model(plan))
    run = dream(dream_client, budget={"operations": 1})
    assert run["report"]["stopped_by"] == "operations"
    assert run["report"]["operations"] == {"flag_conflict": {"applied": 1}}
    assert run["report"]["left"] == 1  # what it did not get to


def test_the_dream_counts_against_the_budget_of_the_run_it_runs_in(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    report = run_queue(dream_client, calls=1)
    assert report["calls"] == 1
    assert dream_report(dream_client, run_id)["report"]["stopped_by"] == "calls"


# --------------------------------------------------------------------------- one at a time, cancel


def test_one_dream_per_scope_family_at_a_time(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    start_dream(dream_client, SCOPE)
    for scope in (SCOPE, f"{SCOPE}/agent:coder"):
        assert dream_client.post("/v1/dream", json={"scope": scope}).status_code == 409
    assert dream_client.post("/v1/dream", json={"scope": "user:bob"}).status_code == 202
    run_queue(dream_client)
    assert dream_client.post("/v1/dream", json={"scope": SCOPE}).status_code == 202


def test_a_dream_below_a_running_one_is_refused_and_above_too(
    dream_client: TestClient, fake_openai: FakeOpenAIServer
) -> None:
    start_dream(dream_client, f"{SCOPE}/agent:coder")
    assert dream_client.post("/v1/dream", json={"scope": SCOPE}).status_code == 409


def test_cancelling_a_pending_dream_means_it_never_calls(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    assert dream_client.post(f"/v1/dream/{run_id}/cancel").status_code == 200
    run_queue(dream_client)
    assert dream_report(dream_client, run_id)["status"] == "cancelled"
    assert fake_openai.chat_calls == []


def test_cancelling_a_running_dream_stops_it_between_units(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    dreamer = dream_app.state.dreamer
    dreamer._after_unit = lambda: dreamer.journal.request_cancel(run_id)
    run_queue(dream_client)
    run = dream_report(dream_client, run_id)
    assert (run["status"], run["report"]["stopped_by"], run["report"]["units"]) == (
        "cancelled",
        "cancelled",
        1,
    )


def test_cancelling_a_finished_dream_is_a_conflict(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run = dream(dream_client)
    assert dream_client.post(f"/v1/dream/{run['run_id']}/cancel").status_code == 409


# --------------------------------------------------------------------------- P11


def test_a_paused_service_runs_no_dream(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    dream_client.put("/v1/background", json={"mode": "paused"})
    assert dream_client.post("/v1/background/run", json={}).status_code == 409
    assert fake_openai.chat_calls == []
    assert dream_report(dream_client, run_id)["status"] == "pending"


def test_a_pause_stops_a_running_dream_between_units(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    dream_app.state.dreamer._after_unit = lambda: dream_app.state.worker.set_mode("paused")
    run_queue(dream_client)
    run = dream_report(dream_client, run_id)
    assert (run["report"]["stopped_by"], run["report"]["units"]) == ("paused", 1)
    assert len(fake_openai.chat_calls) == 1


# --------------------------------------------------------------------------- the dream model


def test_without_a_dream_model_only_service_operations_run(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = dream_settings(
        tmp_path,
        fake_openai,
        dream_url=None,
        dream_model=None,
        chat_url=fake_openai.base_url,
        chat_model="fake-chat",
    )
    app = create_app(settings, clock=clock)
    with TestClient(app, headers=auth("all")) as client:
        build_memory(app.state.store)
        run = dream(client)
    assert run["report"]["unavailable"] == [
        "refresh_summary",
        "propose_edge",
        "create_theme",
        "consolidate_facts",
        "merge_nodes",
        "supersede_fact",
        "flag_conflict",
    ]
    assert fake_openai.chat_calls == []  # the extraction model never stands in


def test_decay_edges_runs_without_the_model_and_is_undoable(
    dream_client: TestClient,
    dream_app: FastAPI,
    fake_openai: FakeOpenAIServer,
    memory: Memory,
    clock: FakeClock,
) -> None:
    store = dream_app.state.store
    edge = store.add_edge(
        scopes=[SCOPE],
        src=memory.karim.id,
        dst=memory.lyon.id,
        type="knows",
        origin="test",
        status="proposed",
    )
    clock.advance(days=400)
    fake_openai.set_default(chat=quiet_model())
    run = dream(dream_client, operations=["decay_edges"])
    assert fake_openai.chat_calls == []  # no model operation asked for
    assert run["report"]["operations"] == {"decay_edges": {"applied": 1}}
    assert store.get_edge(edge.id).valid_to is not None
    assert dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={}).status_code == 200
    back = store.get_edge(edge.id)
    assert (back.valid_to, back.weight) == (None, edge.weight)


def test_reembed_queues_the_embed_job(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = dream_settings(
        tmp_path, fake_openai, embeddings_url=fake_openai.base_url, embeddings_model="fake-embed"
    )
    app = create_app(settings, clock=clock)
    with TestClient(app, headers=auth("all")) as client:
        build_memory(app.state.store)
        while app.state.queue.has_open("embed"):  # the startup embed job
            run_queue(client)
        fake_openai.set_default(chat=quiet_model())
        app.state.store.update_node(
            app.state.store.find_node("Karim", Visibility.of_scope(SCOPE)).id,
            summary="Karim, from Lyon.",
        )
        run_id = start_dream(client, operations=["reembed"])
        client.post("/v1/background/run", json={"max_jobs": 1})
        run = dream_report(client, run_id)
        assert run["report"]["operations"] == {"reembed": {"applied": 1}}
        assert app.state.queue.has_open("embed")


def test_a_bad_answer_skips_its_unit_and_the_run_goes_on(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.script(
        chat=[Reply("I would rather merge everything."), Reply("x", finish_reason="length")]
    )
    fake_openai.set_default(chat=quiet_model())
    run = dream(dream_client)
    assert run["report"]["skipped"] == {"bad_answer": 1, "truncated": 1}
    assert run["report"]["units"] == 3
    assert run["report"]["stopped_by"] == "complete"


def test_a_failing_model_stops_the_run_and_keeps_the_cursor(
    dream_client: TestClient, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    fake_openai.script(chat=[Fail(500)])
    fake_openai.set_default(chat=quiet_model())
    first = dream(dream_client)
    assert first["report"]["stopped_by"] == "model_error"
    assert first["report"]["units"] == 0
    dream(dream_client, budget={"calls": 1})
    assert _unit_nodes(fake_openai)[-1] == memory.karim.id  # the failed unit again


# --------------------------------------------------------------------------- rule 8


def test_recall_answers_while_the_dream_model_thinks(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    """No store lock and no transaction is held across the model call: from the
    model server's own thread the lock is free and a recall answers."""
    store, engine = dream_app.state.store, dream_app.state.engine
    observed: list[bool] = []

    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        free = store.lock.acquire(blocking=False)
        if free:
            try:
                answer = engine.recall("tea", view=Visibility.of_scope(SCOPE), budget_tokens=200)
                observed.append(bool(answer["items"]))
            finally:
                store.lock.release()
        else:
            observed.append(False)
        return []

    fake_openai.set_default(chat=dream_model(plan))
    dream(dream_client)
    assert observed == [True, True, True]


def _change(store: Any, memory: Memory, kind: str) -> Callable[[], None]:
    node = memory.karim.id

    def apply() -> None:
        if kind == "alias":
            store.update_node(node, add_aliases=["Kaka"])
        elif kind == "archive":
            store.archive(memory.tea.id)
        elif kind == "new_fact":
            store.add_fact(
                scopes=[SCOPE], node_id=node, statement="Karim plays chess.", origin="extracted"
            )
        elif kind == "scope":
            with store.lock:
                store.execute(
                    "INSERT INTO record_scopes (record_id, scope) VALUES (?, ?)",
                    (memory.tea.id, "user:alice/project:x"),
                )
        elif kind == "forget":
            store.forget(memory.green_tea.id)
        elif kind == "summary":
            store.update_node(node, summary="A human wrote this.")

    return apply


@pytest.mark.parametrize("kind", ["alias", "archive", "new_fact", "scope", "forget", "summary"])
def test_an_operation_whose_inputs_changed_after_the_snapshot_is_dropped(
    dream_client: TestClient,
    dream_app: FastAPI,
    fake_openai: FakeOpenAIServer,
    memory: Memory,
    kind: str,
) -> None:
    """Rule 8 through the seam between the model's answer and the apply: a write
    that lands there wins, the dream's operation is dropped and reported."""
    store = dream_app.state.store

    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [
            {
                "op": "consolidate_facts",
                "node": memory.karim.id,
                "facts": [memory.tea.id, memory.green_tea.id],
                "statement": "Karim drinks green tea every morning.",
            }
        ]

    fake_openai.set_default(chat=dream_model(plan))
    seam = _change(store, memory, kind)
    fired: list[bool] = []

    def before_apply() -> None:
        if not fired:
            fired.append(True)
            seam()

    dream_app.state.dreamer._before_apply = before_apply
    run = dream(dream_client)
    assert run["report"]["dropped"] == {"stale": 1}
    if kind != "forget":
        assert store.get_fact(memory.tea.id).consolidated_into is None


def test_the_dreams_own_writes_do_not_make_its_next_operation_stale(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.karim.id:
            return []
        return [
            {"op": "refresh_summary", "node": memory.karim.id, "summary": "Karim lives in Lyon."},
            {
                "op": "consolidate_facts",
                "node": memory.karim.id,
                "facts": [memory.tea.id, memory.green_tea.id],
                "statement": "Karim drinks green tea every morning.",
            },
        ]

    fake_openai.set_default(chat=dream_model(plan))
    run = dream(dream_client)
    assert run["report"]["operations"] == {
        "refresh_summary": {"applied": 1},
        "consolidate_facts": {"applied": 1},
    }


def test_a_record_not_in_the_unit_cannot_be_named(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    other = dream_app.state.store.add_fact(
        scopes=[SCOPE], statement="A free fact about nobody.", origin="extracted"
    )

    def plan(unit: dict[str, Any]) -> list[dict[str, Any]]:
        if unit["node"]["id"] != memory.kiki.id:
            return []
        return [{"op": "flag_conflict", "fact": other.id, "with": memory.tea.id}]

    fake_openai.set_default(chat=dream_model(plan))
    run = dream(dream_client)
    assert run["report"]["dropped"] == {"unknown_record": 1}


def test_a_crashed_dream_resumes_from_its_cursor(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer, memory: Memory
) -> None:
    """A job left ``running`` by a crash goes back to the queue (SPEC 7.2) and
    the run resumes after the last unit it finished."""
    fake_openai.set_default(chat=quiet_model())
    run_id = start_dream(dream_client)
    dreamer = dream_app.state.dreamer

    class Crash(BaseException):
        pass

    def crash() -> None:
        dreamer._after_unit = None
        raise Crash

    dreamer._after_unit = crash
    with pytest.raises(Crash):
        dream_app.state.worker.run()
    queue = dream_app.state.queue
    assert queue.recover() == 1
    run_queue(dream_client)
    run = dream_report(dream_client, run_id)
    assert run["status"] == "finished"
    assert _unit_nodes(fake_openai) == sorted([memory.karim.id, memory.kiki.id, memory.lyon.id])
