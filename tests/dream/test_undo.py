"""Every run is a changeset (SPEC 5.6 rule 4): undo reverts a run or one
operation exactly, refuses - naming what depends on it - when a later write
touched what it changed, and cannot undo what forget wiped (P5)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from tests.helpers import auth


@pytest.fixture
def dream_review() -> tuple[tuple[str, str, str], ...]:
    """These tests apply consolidations: the operator lets ``medium`` apply."""
    return MEDIUM_AUTO


def _two_operations(memory: Memory) -> Any:
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

    return dream_model(plan)


def test_undo_reverts_a_whole_run(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    fake_openai.set_default(chat=_two_operations(memory))
    run = dream(dream_client)
    new_fact = store.get_fact(memory.tea.id).consolidated_into
    assert new_fact is not None
    response = dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={})
    assert response.status_code == 200, response.text
    assert response.json()["undone"] == 2
    assert store.get_node(memory.karim.id).summary == ""
    assert store.get_fact(memory.tea.id).valid_to is None
    assert store.revisions([new_fact]) == {}
    assert [i["status"] for i in plan_of(dream_client, run["run_id"])] == ["undone", "undone"]
    assert dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={}).json()["undone"] == 0


def test_undo_reverts_one_operation(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    fake_openai.set_default(chat=_two_operations(memory))
    run = dream(dream_client)
    first = plan_of(dream_client, run["run_id"])[0]["id"]
    response = dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={"op_id": first})
    assert (response.status_code, response.json()["undone"]) == (200, 1)
    assert store.get_node(memory.karim.id).summary == ""
    assert store.get_fact(memory.tea.id).consolidated_into is not None  # the other stays


def test_undo_is_refused_when_a_later_write_depends_on_it(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    fake_openai.set_default(chat=_two_operations(memory))
    run = dream(dream_client)
    new_fact = store.get_fact(memory.tea.id).consolidated_into
    store.flag_fact(new_fact, "language_mismatch")  # a later write on what it created
    response = dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert new_fact in detail["records"]
    assert len(detail["operations"]) == 1
    # all or nothing: the summary was not reverted either
    assert store.get_node(memory.karim.id).summary == "Karim lives in Lyon."


def test_one_operation_can_still_be_undone_when_another_is_blocked(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    fake_openai.set_default(chat=_two_operations(memory))
    run = dream(dream_client)
    new_fact = store.get_fact(memory.tea.id).consolidated_into
    store.flag_fact(new_fact, "language_mismatch")
    items = plan_of(dream_client, run["run_id"])
    blocked = dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={"op_id": items[1]["id"]})
    assert blocked.status_code == 409
    free = dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={"op_id": items[0]["id"]})
    assert free.status_code == 200


def test_an_operation_wiped_by_forget_cannot_be_undone(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    fake_openai.set_default(chat=_two_operations(memory))
    run = dream(dream_client)
    new_fact = store.get_fact(memory.tea.id).consolidated_into
    store.forget(new_fact)
    items = plan_of(dream_client, run["run_id"])
    consolidation = items[1]
    assert consolidation["forgotten"] is True
    assert consolidation["payload"] is None
    response = dream_client.post(
        f"/v1/dream/{run['run_id']}/undo", json={"op_id": consolidation["id"]}
    )
    assert response.status_code == 409
    assert "forgotten" in response.text
    # forget never brings back what it did not delete: the originals stay closed
    tea = store.get_fact(memory.tea.id)
    assert tea.valid_to is not None and tea.consolidated_into is None


def test_undo_of_an_unknown_operation_or_run_is_not_found(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    fake_openai.set_default(chat=_two_operations(memory))
    run = dream(dream_client)
    other = dream_client.post(f"/v1/dream/{run['run_id']}/undo", json={"op_id": "dop_" + "0" * 26})
    assert other.status_code == 404
    assert dream_client.post("/v1/dream/dream_" + "0" * 26 + "/undo", json={}).status_code == 404


def test_a_token_bound_elsewhere_does_not_reach_the_run(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    fake_openai.set_default(chat=_two_operations(memory))
    run = dream(dream_client)
    for path in ("", "/plan"):
        response = dream_client.get(f"/v1/dream/{run['run_id']}{path}", headers=auth("bob-dream"))
        assert response.status_code == 404
    response = dream_client.post(
        f"/v1/dream/{run['run_id']}/undo", json={}, headers=auth("bob-dream")
    )
    assert response.status_code == 404
    assert (
        dream_client.get(f"/v1/dream/{run['run_id']}", headers=auth("alice-dream")).status_code
        == 200
    )
    _ = SCOPE
