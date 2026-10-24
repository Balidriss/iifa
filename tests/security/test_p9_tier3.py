"""Tier 3 and P9: dense candidates are filtered by the view before any score,
exactly as lexical ones; semantic resolution, merge proposals and semantic
links never reach across a scope line (SPEC 6.1, dream rule 12)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth, remember
from tests.model.test_tier3 import NEAR, tier3_settings
from tests.tier2 import answer, post_turn, run_all

pytestmark = pytest.mark.security

ALICE, BOB = "user:alice", "user:bob"
QUERY = "who leads the photovoltaic work"


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier3_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def dense_view(client: TestClient) -> list[tuple[str, str]]:
    body = {"scope": ALICE, "query": QUERY, "budget_tokens": 2000}
    result = client.post("/v1/recall", json=body).json()
    assert result["tier"] == 3
    return [(i["id"], i["why"]) for i in result["items"]]


def test_p9_invisible_vectors_change_no_dense_rank_or_score(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    statements = [f"Alice note {i} about the roof" for i in range(6)]
    fake_openai.set_vectors(
        {QUERY: NEAR, **{s: [1.0, 0.1 * i, 0.3, 0, 0, 0, 0, 0] for i, s in enumerate(statements)}}
    )
    for statement in statements:
        remember(client, statement, scope=ALICE)
    run_all(client, app)
    before = dense_view(client)
    assert before and all(item_id.startswith("fact_") for item_id, _ in before)

    # Bob's vectors sit exactly on the query: if they leaked they would outrank all
    bob_statements = [f"Bob secret {i}" for i in range(20)]
    fake_openai.set_vectors({s: NEAR for s in bob_statements})
    bob_ids = {remember(client, s, scope=BOB)["fact"]["id"] for s in bob_statements}
    run_all(client, app)
    after = dense_view(client)
    assert after == before  # same items, same order, same scores in the why
    assert not bob_ids & {item_id for item_id, _ in after}


def test_p9_a_near_identical_node_in_another_scope_is_never_a_candidate(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Semantic resolution compares a mention only with the nodes the source's
    scopes see: Bob's "Karim", identical in vector space, is not offered."""
    fake_openai.set_vectors({"Karim": NEAR, "Карим": NEAR})
    remember(client, "Karim fixes roofs.", scope=BOB, node={"name": "Karim", "kind": "person"})
    run_all(client, app)
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Карим", "kind": "person"}],
                facts=[
                    {
                        "subject": "Карим",
                        "statement": "Карим переехал в Лион.",
                        "quote": "Карим переехал в Лион",
                    }
                ],
            )
        ]
    )
    post_turn(client, "Карим переехал в Лион.", scope=ALICE)
    report = run_all(client, app)
    turn = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert turn["mentions_adjudicated"] == 0
    assert len(fake_openai.chat_calls) == 1  # no adjudication call was needed
    store = app.state.store
    alice = store.find_node("Карим", Visibility.of_scope(ALICE))
    bob = store.find_node("Karim", Visibility.of_scope(BOB))
    assert alice is not None and bob is not None and alice.id != bob.id
    assert store.proposals([alice.id, bob.id], Visibility.of_subtree("user:alice")) == []


def test_rule_12_a_merge_proposal_needs_identical_scope_sets(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Two nodes close by name in different scopes are never proposed for a
    merge (dream rule 12: never widen visibility); in the same scope they are."""
    fake_openai.set_vectors({"Karim": NEAR, "Карим": NEAR, "Kareem": NEAR})
    remember(client, "Karim fixes roofs.", scope=ALICE, node={"name": "Karim", "kind": "person"})
    remember(
        client, "Kareem sells solar panels.", scope=BOB, node={"name": "Kareem", "kind": "person"}
    )
    remember(
        client, "Карим переехал в Лион.", scope=ALICE, node={"name": "Карим", "kind": "person"}
    )
    run_all(client, app)
    store = app.state.store
    everything = Visibility.of_subtree("user:alice")
    karim = store.find_node("Karim", Visibility.of_scope(ALICE))
    cyrillic = store.find_node("Карим", Visibility.of_scope(ALICE))
    kareem = store.find_node("Kareem", Visibility.of_scope(BOB))
    assert karim and cyrillic and kareem
    merges = [
        p
        for p in store.proposals([karim.id, cyrillic.id], everything)
        + store.proposals([kareem.id], Visibility.of_subtree("user:bob"))
        if p.kind == "merge"
    ]
    pairs = {frozenset((p.node_id, p.value)) for p in merges}
    assert pairs == {frozenset((karim.id, cyrillic.id))}
    # proposed, never applied: both nodes are still there, unmerged
    assert store.get_node(karim.id).archived_at is None
    assert store.get_node(cyrillic.id).archived_at is None
