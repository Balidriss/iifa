"""P9 - scope filters before scoring, and the graph never crosses a scope line."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iifa.recall.lexical import bm25
from iifa.recall.tokenizer import tokenize
from iifa.scopes import Visibility
from iifa.store.repo import Store
from tests.api.conftest import KEY, auth, remember
from tests.conftest import FakeClock

pytestmark = pytest.mark.security

ALICE = Visibility.of_scope("user:alice")
QUERY = "solar project budget"


def _scores(store: Store) -> list[tuple[str, float]]:
    hits = bm25(store.postings(tokenize(QUERY), ALICE), store.corpus_stats(ALICE))
    return [(h.doc_id, round(h.score, 12)) for h in hits]


def test_p9_invisible_corpus_changes_no_score_and_no_statistic(store: Store) -> None:
    """Adding thousands of documents to a scope the reader cannot see must leave
    every visible score, rank, document frequency and average length untouched."""
    for i in range(20):
        store.add_fact(
            scopes=["user:alice"],
            statement=f"note {i} about the solar project" + (" budget" if i % 3 == 0 else ""),
            origin="explicit",
        )
    before_scores = _scores(store)
    before_stats = store.corpus_stats(ALICE)
    assert before_scores

    for i in range(300):  # a corpus that would shift IDF and average length if it leaked
        store.add_fact(
            scopes=["user:bob"],
            statement=f"solar solar solar budget budget {i} " + "x " * (i % 40),
            origin="explicit",
        )
    store.create_node(scopes=["user:bob"], name="Solar budget committee", summary="solar budget")

    assert _scores(store) == before_scores
    assert store.corpus_stats(ALICE) == before_stats
    assert all(p.doc_id.startswith("fact_") for p in store.postings(["solar"], ALICE))
    assert len(store.postings(["solar"], ALICE)) == 20


def test_p9_api_ranking_and_omissions_are_identical_with_an_invisible_corpus(
    client: TestClient,
) -> None:
    for i in range(12):
        remember(client, f"alice note {i} on the solar project budget", scope="user:alice")

    def snapshot() -> dict[str, object]:
        body = client.post(
            "/v1/recall", json={"scope": "user:alice", "query": QUERY, "budget_tokens": 500}
        ).json()
        return {
            "ids": [i["id"] for i in body["items"]],
            "used": body["used_tokens"],
            "omitted": body["omitted"],
        }

    before = snapshot()
    for i in range(80):
        remember(
            client, f"bob note {i} solar solar budget " + "filler " * (i % 30), scope="user:bob"
        )
    assert snapshot() == before


def test_p9_graph_never_traverses_or_names_an_invisible_record(client: TestClient) -> None:
    """Amendment point 2: an edge to an invisible node, or an invisible edge, is
    not followed, and no ``why`` names the invisible node."""
    karim = remember(
        client,
        "Karim is an electrician",
        scope="user:alice",
        node={"name": "Karim", "kind": "person"},
    )
    secret = remember(
        client,
        "Zorblax project is confidential",
        scope="user:bob",
        node={"name": "Zorblax", "kind": "project"},
    )
    public = remember(
        client,
        "Windmill project is public",
        scope="user:alice",
        node={"name": "Windmill", "kind": "project"},
    )
    # edge visible to alice, target node invisible
    assert (
        client.post(
            "/v1/link",
            json={
                "scopes": ["user:alice"],
                "src": karim["node"]["id"],
                "dst": secret["node"]["id"],
                "type": "works_on",
            },
        ).status_code
        == 404
    )
    # so create it with a token that sees both, then check alice's view
    assert (
        client.post(
            "/v1/link",
            json={
                "scopes": ["user:alice", "user:bob"],
                "src": karim["node"]["id"],
                "dst": secret["node"]["id"],
                "type": "works_on",
            },
        ).status_code
        == 201
    )
    # edge invisible to alice between two visible nodes
    assert (
        client.post(
            "/v1/link",
            json={"scope": "user:bob"}
            | {"src": karim["node"]["id"], "dst": public["node"]["id"], "type": "works_on"},
        ).status_code
        == 404
    )
    admin_edge = client.post(
        "/v1/link",
        json={
            "scopes": ["user:bob"],
            "src": karim["node"]["id"],
            "dst": public["node"]["id"],
            "type": "works_on",
        },
        headers=auth("all"),
    )
    assert admin_edge.status_code == 404  # both ends must be visible in the write view too

    body = client.post(
        "/v1/recall",
        json={"scope": "user:alice", "query": "electrician", "budget_tokens": 4000},
        headers=auth("alice"),
    ).json()
    ids = {i["id"] for i in body["items"]}
    assert karim["fact"]["id"] in ids
    assert secret["fact"]["id"] not in ids and secret["node"]["id"] not in ids
    assert public["fact"]["id"] not in ids  # the only edge to it is invisible
    for item in body["items"]:
        assert (
            "Zorblax" not in item["why"]
            and "Zorblax" not in item["title"]
            and "Zorblax" not in item["line"]
        )
    view = client.get(
        f"/v1/nodes/{karim['node']['id']}", params={"scope": "user:alice"}, headers=auth("alice")
    ).json()
    assert view["edges"] == []
    # Bob, who sees the shared edge and Zorblax, does get the hop
    bob_view = client.get(
        f"/v1/nodes/{karim['node']['id']}", params={"scopes": ["user:alice", "user:bob"]}
    ).json()
    assert len(bob_view["edges"]) == 1


def test_p9_fact_of_an_invisible_node_never_identifies_the_node(client: TestClient) -> None:
    node = remember(
        client,
        "Zorblax is a project",
        scope="user:bob",
        node={"name": "Zorblax", "kind": "project"},
    )["node"]
    shared = client.post(
        "/v1/remember",
        json={
            "scopes": ["user:alice", "user:bob"],
            "statement": "budget approved",
            "node_id": node["id"],
        },
    )
    assert shared.status_code == 201
    body = client.post(
        "/v1/recall",
        json={"scope": "user:alice", "query": "budget approved"},
        headers=auth("alice"),
    ).json()
    assert [i["id"] for i in body["items"]] == [shared.json()["fact"]["id"]]
    assert body["items"][0]["title"] == "budget approved"  # no "Zorblax — " prefix
    expanded = client.post(
        "/v1/expand",
        json={"scope": "user:alice", "ids": [shared.json()["fact"]["id"]]},
        headers=auth("alice"),
    ).json()
    assert expanded["items"][0]["node_id"] is None and expanded["items"][0]["node_name"] is None
    bob = client.post(
        "/v1/expand",
        json={"scope": "user:bob", "ids": [shared.json()["fact"]["id"]]},
        headers=auth("bob"),
    ).json()
    assert bob["items"][0]["node_id"] == node["id"]


def test_p9_tokens_cannot_read_outside_their_scopes(client: TestClient) -> None:
    fact = remember(client, "alice only", scope="user:alice")["fact"]["id"]
    assert (
        client.post(
            "/v1/recall", json={"scope": "user:alice", "query": "alice"}, headers=auth("bob")
        ).status_code
        == 403
    )
    assert client.post(
        "/v1/expand", json={"scope": "user:bob", "ids": [fact]}, headers=auth("bob")
    ).json()["missing"] == [fact]
    assert (
        client.get(
            f"/v1/nodes/{fact}", params={"scope": "user:bob"}, headers=auth("bob")
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/recall",
            json={"scope": "user:alice", "query": "alice"},
            headers=auth("alice-coder"),
        ).status_code
        == 403
    )
    coder = client.post(
        "/v1/recall",
        json={"scope": "user:alice/agent:coder", "query": "alice"},
        headers=auth("alice-coder"),
    ).json()
    assert [i["id"] for i in coder["items"]] == [fact]  # ancestors are readable
    assert (
        client.post("/v1/archive", json={"id": fact}, headers=auth("alice-coder")).status_code
        == 403
    )  # but not writable


def test_p9_a_rescope_never_grants_or_takes_away_a_scope_the_token_does_not_own(
    client: TestClient,
) -> None:
    """SPEC 4.6: every scope a re-scope adds and every scope it removes must be the
    token's own; a scope it does not own is kept exactly as it was."""
    coder = remember(client, "Coder note on palettes", scope="user:alice/agent:coder")["fact"]
    widen = client.patch(
        f"/v1/facts/{coder['id']}", json={"scopes": ["user:alice"]}, headers=auth("alice-coder")
    )
    assert widen.status_code == 403  # bound below user:alice: cannot open it to the parent

    shared = remember(client, "Shared note", scopes=["user:alice", "user:bob"])["fact"]

    def patch(scopes: list[str], token: str) -> int:
        response = client.patch(
            f"/v1/facts/{shared['id']}", json={"scopes": scopes}, headers=auth(token)
        )
        return response.status_code

    assert patch(["user:alice"], "alice") == 403  # would take bob's scope away
    assert patch(["user:alice", "user:bob", "user:carol"], "alice") == 403  # would grant carol
    assert client.app.state.store.scopes_of(shared["id"]) == ("user:alice", "user:bob")
    assert patch(["user:alice/agent:coder", "user:bob"], "alice") == 200  # bob's kept as is
    assert client.app.state.store.scopes_of(shared["id"]) == ("user:alice/agent:coder", "user:bob")

    alice_only = remember(client, "Alice private note", scope="user:alice")["fact"]
    stolen = client.patch(
        f"/v1/facts/{alice_only['id']}", json={"scopes": ["user:bob"]}, headers=auth("bob")
    )
    assert stolen.status_code == 403  # bob does not own the record at all
    assert client.app.state.store.scopes_of(alice_only["id"]) == ("user:alice",)


def test_p9_a_rescope_never_leaves_a_fact_under_a_node_of_the_scope_it_left(
    client: TestClient,
) -> None:
    """SPEC 4.6: archiving or forgetting a node reaches its facts, so a fact never
    moves away from its node. A token of the scope the fact left cannot hide or
    delete it through the node."""
    coder, writer = "user:alice/agent:coder", "user:alice/agent:writer"
    written = remember(client, "Mira paints with cold palettes", scope=coder, node={"name": "Mira"})
    fact, node = written["fact"]["id"], written["node"]["id"]

    def move(**extra: object) -> int:
        body = {"scopes": [writer], **extra}
        return client.patch(f"/v1/facts/{fact}", json=body, headers=auth("alice")).status_code

    assert move() == 409  # the node would stay in coder, out of the fact's sight
    assert move(detach=True) == 200

    def recalled() -> list[str]:
        body = client.post(
            "/v1/recall", json={"scope": writer, "query": "cold palettes"}, headers=auth("alice")
        ).json()
        return [item["id"] for item in body["items"]]

    assert recalled() == [fact]
    archive = client.post("/v1/archive", json={"id": node}, headers=auth("alice-coder"))
    assert archive.status_code == 200
    assert recalled() == [fact]
    gone = client.post("/v1/forget", json={"id": node}, headers=auth("forget")).json()
    assert gone["removed"]["facts"] == 0
    assert recalled() == [fact]


def test_p9_a_rescoped_fact_moves_its_statistics_with_it(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    """After a re-scope, every score and statistic is the one of a store where the
    fact was written in its new scope from the start: the scope it left keeps no
    trace of it (no document count, no length, no posting)."""

    def fill(target: Store) -> None:
        for i in range(10):
            target.add_fact(
                scopes=["user:alice"], statement=f"alice solar note {i}", origin="explicit"
            )
            target.add_fact(
                scopes=["user:bob"],
                statement=f"bob solar budget {i} " + "x " * i,
                origin="explicit",
            )

    fill(store)
    moved = store.add_fact(
        scopes=["user:bob"], statement="solar solar budget budget review", origin="explicit"
    )
    store.attach_fact(moved.id, None, scopes=["user:alice"])

    with Store(tmp_path / "reference.db", KEY, clock=clock) as reference:
        fill(reference)
        reference.add_fact(
            scopes=["user:alice"], statement="solar solar budget budget review", origin="explicit"
        )
        for view in (ALICE, Visibility.of_scope("user:bob")):
            assert store.corpus_stats(view) == reference.corpus_stats(view)
            assert len(store.postings(["budget"], view)) == len(
                reference.postings(["budget"], view)
            )
        assert [s for _, s in _scores(store)] == [s for _, s in _scores(reference)]
    assert moved.id not in {
        p.doc_id for p in store.postings(["review"], Visibility.of_scope("user:bob"))
    }
