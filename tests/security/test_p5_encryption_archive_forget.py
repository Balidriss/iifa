"""P5 - encrypted at rest; archiving is reversible, forgetting is real."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from iifa.store.connection import WrongKeyError, open_database
from iifa.store.repo import Store
from tests.api.conftest import auth, remember
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.fixtures.corpus import CANARY
from tests.tier2 import answer, post_turn, run_all, tier2_settings

pytestmark = pytest.mark.security


def _raw_bytes(path: Path) -> bytes:
    data = b""
    for candidate in (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-journal"),
    ):
        if candidate.exists():
            data += candidate.read_bytes()
    return data


def test_p5_database_is_unreadable_without_the_key(tmp_path: Path) -> None:
    path = tmp_path / "m.db"
    with Store(path, "right-key") as store:
        store.add_fact(scopes=["user:alice"], statement=CANARY, origin="explicit")
    assert CANARY.encode() not in _raw_bytes(path)
    plain = sqlite3.connect(path)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            plain.execute("SELECT count(*) FROM sqlite_master").fetchall()
    finally:
        plain.close()
    with pytest.raises(WrongKeyError):
        open_database(path, "wrong-key")
    with pytest.raises(ValueError, match="key is required"):
        open_database(path, "")
    with Store(path, "right-key") as store:
        assert store.counts()["facts"] == 1


def test_p5_forget_leaves_no_trace_in_the_file(tmp_path: Path) -> None:
    """Scan the raw bytes of an *unencrypted* store: with encryption the bytes
    are unreadable anyway, so this proves the deletion itself is complete."""
    path = tmp_path / "plain.db"
    with Store(path, "", plaintext_for_tests=True) as store:
        node = store.create_node(
            scopes=["user:alice"], name=f"Node {CANARY}", aliases=[f"alias {CANARY}"]
        )
        source = store.add_source(scopes=["user:alice"], kind="turn", text=f"source text {CANARY}")
        fact = store.add_fact(
            scopes=["user:alice"],
            node_id=node.id,
            statement=f"fact {CANARY}",
            origin="extracted",
            source_id=source.id,
            attribute=f"attr {CANARY}",
            value=f"value {CANARY}",
        )
        keep = store.add_fact(
            scopes=["user:alice"], statement="unrelated fact stays", origin="explicit"
        )
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert CANARY.encode() in _raw_bytes(path)  # the test can see plaintext before forget
        removed = store.forget(node.id)
        assert removed == {
            "facts": 1,
            "sources": 1,
            "edges": 0,
            "nodes": 1,
            "proposals": 0,
            "vectors": 0,
            "probes_deleted": 0,
            "dream_ops_wiped": 0,
            "adjudications_deleted": 0,
        }
        assert store.postings(["canary"], Visibility.of_scope("user:alice")) == []
        assert store.counts() == {
            "nodes": 0,
            "nodes_archived": 0,
            "facts": 1,
            "facts_archived": 0,
            "edges": 0,
            "edges_archived": 0,
            "sources": 0,
            "lexical_documents": 1,
            "queue": 0,
        }
        assert store.get_fact(keep.id).statement == "unrelated fact stays"
        _ = fact
    assert CANARY.encode() not in _raw_bytes(path)
    assert CANARY.encode("utf-16-le") not in _raw_bytes(path)


def test_p5_archive_hides_everywhere_and_restore_brings_back_unchanged(client: TestClient) -> None:
    fact = remember(client, "Karim likes tea", node={"name": "Karim", "kind": "person"})
    fact_id, node_id = fact["fact"]["id"], fact["node"]["id"]
    before = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": [fact_id], "level": 2}
    ).json()["items"][0]

    assert client.post("/v1/archive", json={"id": fact_id}).status_code == 200
    assert (
        client.post("/v1/recall", json={"scope": "user:alice", "query": "tea"}).json()["items"]
        == []
    )
    expanded = client.post("/v1/expand", json={"scope": "user:alice", "ids": [fact_id]}).json()
    assert expanded["items"] == [] and expanded["missing"] == [fact_id]
    assert client.get(f"/v1/nodes/{node_id}", params={"scope": "user:alice"}).json()["facts"] == []
    assert [
        i["id"] for i in client.get("/v1/archive", params={"scope": "user:alice"}).json()["items"]
    ] == [fact_id]

    assert client.post("/v1/restore", json={"id": fact_id}).status_code == 200
    after = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": [fact_id], "level": 2}
    ).json()["items"][0]
    assert after == before
    assert (
        client.post("/v1/recall", json={"scope": "user:alice", "query": "tea"}).json()["items"][0][
            "id"
        ]
        == fact_id
    )

    # archiving the node hides its facts too; restoring the node brings them back
    client.post("/v1/archive", json={"id": node_id})
    assert (
        client.post("/v1/recall", json={"scope": "user:alice", "query": "tea"}).json()["items"]
        == []
    )
    assert client.get(f"/v1/nodes/{node_id}", params={"scope": "user:alice"}).status_code == 404
    client.post("/v1/restore", json={"id": node_id})
    assert (
        client.post("/v1/recall", json={"scope": "user:alice", "query": "tea"}).json()["items"][0][
            "id"
        ]
        == fact_id
    )


def test_p5_forget_needs_its_own_permission_on_every_route(client: TestClient) -> None:
    fact = remember(client, "Karim likes tea", node={"name": "Karim"})
    for token in ("read", "write", "admin", "dream", "alice", "alice-dream"):
        response = client.post("/v1/forget", json={"id": fact["fact"]["id"]}, headers=auth(token))
        assert response.status_code == 403, token
    assert client.get("/v1/stats").json()["counts"]["facts"] == 1
    # no other route deletes: archive + restore round-trips, PATCH cannot remove
    assert client.patch(f"/v1/facts/{fact['fact']['id']}", json={"detach": True}).status_code == 200
    assert client.get("/v1/stats").json()["counts"]["facts"] == 1
    removed = client.post(
        "/v1/forget", json={"id": fact["fact"]["id"]}, headers=auth("forget")
    ).json()["removed"]
    assert removed == {
        "facts": 1,
        "sources": 1,
        "vectors": 0,
        "probes_deleted": 0,
        "dream_ops_wiped": 0,
        "adjudications_deleted": 0,
    }
    assert client.get("/v1/stats").json()["counts"]["facts"] == 0


def test_p5_forget_is_bound_to_the_tokens_scopes(client: TestClient) -> None:
    alice_fact = remember(client, "Alice's secret", scope="user:alice")["fact"]["id"]
    assert (
        client.post("/v1/forget", json={"id": alice_fact}, headers=auth("bob")).status_code == 403
    )
    assert (
        client.post("/v1/forget", json={"id": alice_fact}, headers=auth("forget")).status_code
        == 200
    )


def test_p5_forget_works_without_archiving_first(client: TestClient) -> None:
    fact = remember(client, "to be forgotten in one call", node={"name": "Karim"})
    response = client.post("/v1/forget", json={"id": fact["node"]["id"]})
    assert response.json()["removed"] == {
        "facts": 1,
        "sources": 1,
        "edges": 0,
        "nodes": 1,
        "proposals": 0,
        "vectors": 0,
        "probes_deleted": 0,
        "dream_ops_wiped": 0,
        "adjudications_deleted": 0,
    }
    assert client.get("/v1/archive", params={"scope": "user:alice"}).json()["items"] == []


def test_p5_forget_deletes_every_probe_that_expected_the_record(tmp_path: Path) -> None:
    """A probe's question may carry the content it asks about ("Where does Karim
    live?"): forget deletes every probe that expected the record, even one that
    still expects another."""
    path = tmp_path / "plain.db"
    with Store(path, "", plaintext_for_tests=True) as store:
        fact = store.add_fact(scopes=["user:alice"], statement="a plain fact", origin="explicit")
        keep = store.add_fact(scopes=["user:alice"], statement="another fact", origin="explicit")
        created = store.create_probe_set(
            scope="user:alice",
            name="set",
            budget_tokens=100,
            probes=[
                (f"where does {CANARY} live", [fact.id, keep.id]),
                ("unrelated question", [keep.id]),
            ],
        )
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert CANARY.encode() in _raw_bytes(path)  # the test can see plaintext before forget
        assert store.forget(fact.id)["probes_deleted"] == 1
        assert [p.query for p in store.get_probe_set(created.id).probes] == ["unrelated question"]
    assert CANARY.encode() not in _raw_bytes(path)


def test_p5_forgetting_a_turns_source_leaves_no_trace_in_the_queue_or_anywhere(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """SPEC 12, the queue too: a turn's words wait in the queue until the turn
    is processed. Once its source is forgotten they are nowhere in the raw file:
    not in the queue, the tables or the lexical index. An unencrypted store, so
    the scan can see plaintext."""
    settings = tier2_settings(tmp_path, fake_openai, db_key="", plaintext_database_for_tests=True)
    statement = f"Karim keeps the {CANARY} notebook."
    fact = {
        "subject": "Karim",
        "statement": statement,
        "attribute": "keeps",
        "value": f"the {CANARY} notebook",
        "quote": statement,
    }
    fake_openai.script(chat=[answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact])])
    app = create_app(settings, clock=clock)
    path = settings.db_path
    try:
        with TestClient(app, headers=auth("all")) as client:
            post_turn(client, statement)
            app.state.store.checkpoint("TRUNCATE")
            assert CANARY.encode() in _raw_bytes(path)  # waiting in the queue
            run_all(client, app)
            (source_id,) = [r["id"] for r in app.state.store.execute("SELECT id FROM sources")]
            assert app.state.store.counts()["facts"] == 1
            response = client.post("/v1/forget", json={"id": source_id})
            assert response.status_code == 200, response.text
            assert response.json()["removed"]["sources"] == 1
            assert app.state.store.counts()["facts"] == 0
            assert CANARY.encode() not in _raw_bytes(path)
    finally:
        app.state.store.close()
