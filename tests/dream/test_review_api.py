"""One review queue (SPEC 4.4b, 8.6): pending alias and fact proposals and the
dream's queued operations in one listing, under the one ``review`` permission;
accepting a queued operation re-checks it against memory as it is now."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.dream.conftest import SCOPE, Memory, build_memory, dream, dream_model
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth


def _queue_a_merge(client: TestClient, fake: FakeOpenAIServer, memory: Memory) -> dict[str, Any]:
    # the pending merge proposal makes the pair a candidate; the adjudication's
    # "same" queues the merge (SPEC 5.6)
    store = client.app.state.store  # type: ignore[attr-defined]
    store.add_proposal(scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id)
    fake.set_default(
        chat=dream_model(
            lambda unit: [], lambda pair: {"verdict": "same", "reason": "the same person"}
        )
    )
    return dream(client)


def _review(client: TestClient, token: str = "review", scope: str = SCOPE) -> Any:
    return client.get(
        "/v1/review", params={"scope": scope, "budget_tokens": 16000}, headers=auth(token)
    )


def test_the_review_queue_lists_proposals_and_queued_operations_together(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    alias = store.add_proposal(scopes=[SCOPE], kind="alias", node_id=memory.karim.id, value="Kaka")
    fact = store.add_proposal(
        scopes=[SCOPE],
        kind="fact",
        node_id=None,
        value="The user lives in Moscow.",
        payload={"provenance": "external"},
    )
    merge = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    run = _queue_a_merge(dream_client, fake_openai, memory)
    response = _review(dream_client)
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    kinds = {(i["kind"], i["id"]) for i in items}
    queued = run["report"]["review"]
    assert kinds == {("alias", alias.id), ("fact", fact.id), ("dream_operation", queued[0])}
    assert merge.id not in {i["id"] for i in items}  # its queued merge_nodes stands for it
    operation = next(i for i in items if i["kind"] == "dream_operation")
    assert operation["op"] == "merge_nodes"
    assert operation["risk"] == "high"
    assert operation["reason_given"] == "the same person"
    assert operation["run_id"] == run["run_id"]
    assert operation["proposal_id"] == merge.id
    adjudication = operation["adjudication"]
    assert (adjudication["verdict"], adjudication["reason"]) == ("same", "the same person")
    assert adjudication["run_id"] == run["run_id"]


def test_the_review_queue_needs_the_review_permission(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    for token in ("dream", "read", "write", "admin", "alice", "alice-dream"):
        assert _review(dream_client, token).status_code == 403, token
    assert _review(dream_client, "alice-review").status_code == 200


def test_the_review_queue_shows_only_what_the_reviewer_sees(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    _queue_a_merge(dream_client, fake_openai, memory)
    assert len(_review(dream_client, "alice-review").json()["items"]) == 1
    response = _review(dream_client, "alice-review", scope="user:bob")
    assert response.status_code == 403
    assert _review(dream_client, "review", scope="user:bob").json()["items"] == []


def test_accepting_a_queued_operation_applies_it(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "applied"
    assert store.get_node(memory.kiki.id).merged_into == memory.karim.id
    again = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert again.status_code == 409
    assert _review(dream_client).json()["items"] == []


def test_rejecting_a_queued_operation_changes_nothing(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    before = store.revisions([memory.karim.id, memory.kiki.id])
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "reject"}, headers=auth("review")
    )
    assert (response.status_code, response.json()["status"]) == (200, "rejected")
    assert store.revisions([memory.karim.id, memory.kiki.id]) == before


def test_a_queued_operation_whose_inputs_moved_is_dropped_on_accept(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    store.update_node(memory.kiki.id, add_aliases=["Kiki B."])  # memory moved meanwhile
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert response.status_code == 409
    assert "stale" in response.text
    assert store.get_node(memory.kiki.id).merged_into is None


def test_a_dream_token_cannot_approve_what_its_dream_proposed(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    for token in ("dream", "alice-dream", "write"):
        response = dream_client.post(
            f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth(token)
        )
        assert response.status_code == 403, token


def test_a_reviewer_bound_elsewhere_does_not_see_the_operation(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("bob-review")
    )
    assert response.status_code == 404


def test_proposals_are_decided_through_the_review_queue_too(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    alias = store.add_proposal(scopes=[SCOPE], kind="alias", node_id=memory.karim.id, value="Kaka")
    fact = store.add_proposal(
        scopes=[SCOPE],
        kind="fact",
        node_id=memory.karim.id,
        value="Karim plays golf on Sundays.",
    )
    for proposal in (alias, fact):
        response = dream_client.post(
            f"/v1/review/{proposal.id}", json={"decision": "accept"}, headers=auth("review")
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "accepted"
    assert "Kaka" in store.get_node(memory.karim.id).aliases


def test_a_merge_proposal_is_not_decided_through_the_review_queue(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    merge = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    response = dream_client.post(
        f"/v1/review/{merge.id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert response.status_code == 409


def test_an_unknown_review_item_is_not_found(dream_client: TestClient) -> None:
    for item in ("dop_" + "0" * 26, "prop_" + "0" * 26, "node_" + "0" * 26, "x"):
        response = dream_client.post(
            f"/v1/review/{item}", json={"decision": "accept"}, headers=auth("review")
        )
        assert response.status_code == 404, item
