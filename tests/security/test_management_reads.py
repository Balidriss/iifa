"""Guardrails of the management routes: the archive
never names a node the view does not see, the dream list never shows a run
outside the token's scopes, a proposed edge is decided only by a ``write``
token that touches it, and ``forget`` wipes that decision from the journal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.store.corrections import Corrections
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.helpers import auth, make_settings

pytestmark = pytest.mark.security

ALICE, BOB = "user:alice", "user:bob"


def test_p9_the_archive_never_names_an_invisible_node(client: TestClient, app: FastAPI) -> None:
    """An archived edge of Alice's joins her node to one of Bob's; an archived
    fact of hers hangs off Bob's node. Alice's archive shows neither Bob's node's
    id nor its name."""
    store: Store = app.state.store
    karim = store.create_node(scopes=[ALICE], name="Karim", kind="person")
    secret = store.create_node(scopes=[BOB], name="Zebulon Secret", kind="person")
    edge = store.add_edge(
        scopes=[ALICE], src=karim.id, dst=secret.id, type="knows", origin="explicit"
    )
    fact = store.add_fact(
        scopes=[ALICE], node_id=secret.id, statement="Met at the fair.", origin="explicit"
    )
    store.archive(edge.id)
    store.archive(fact.id)
    response = client.get(
        "/v1/archive", params={"scope": ALICE, "budget_tokens": 16000}, headers=auth("alice")
    )
    assert response.status_code == 200, response.text
    text = response.text
    assert secret.id not in text and "Zebulon" not in text
    items = {i["kind"]: i for i in response.json()["items"]}
    assert (items["edge"]["src"], items["edge"]["src_name"]) == (karim.id, "Karim")
    assert (items["edge"]["dst"], items["edge"]["dst_name"]) == (None, None)
    assert (items["fact"]["node_id"], items["fact"]["node_name"]) == (None, None)
    # a view that sees both scopes names both
    both = client.get(
        "/v1/archive", params={"scopes": [ALICE, BOB], "budget_tokens": 16000}
    ).json()["items"]
    assert {i.get("dst_name") or i.get("node_name") for i in both} == {"Zebulon Secret"}


def test_p9_the_dream_list_never_shows_a_run_outside_the_tokens_scopes(
    tmp_path: Path, clock: FakeClock
) -> None:
    app = create_app(make_settings(tmp_path, background_mode="paused"), clock=clock)
    try:
        with TestClient(app) as client:
            bobs = client.post("/v1/dream", json={"scope": BOB}, headers=auth("bob-dream"))
            assert bobs.status_code == 202, bobs.text
            bob_run = bobs.json()["run_id"]
            mine = client.get("/v1/dreams", params={"scope": ALICE}, headers=auth("alice-dream"))
            assert mine.status_code == 200
            assert mine.json()["items"] == []
            theirs = client.get("/v1/dreams", params={"scope": BOB}, headers=auth("alice-dream"))
            assert theirs.status_code == 403 and bob_run not in theirs.text
            for token in ("alice", "review", "admin", "read", "write"):  # no dream permission
                refused = client.get("/v1/dreams", params={"scope": BOB}, headers=auth(token))
                assert refused.status_code == 403
            listed = client.get("/v1/dreams", params={"scope": BOB}, headers=auth("bob-dream"))
            assert [i["id"] for i in listed.json()["items"]] == [bob_run]
    finally:
        app.state.store.close()


def test_an_edge_is_decided_only_by_a_write_token_that_touches_it(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    karim = store.create_node(scopes=[ALICE], name="Karim", kind="person")
    lyon = store.create_node(scopes=[ALICE], name="Lyon", kind="place")
    edge = store.add_edge(
        scopes=[ALICE],
        src=karim.id,
        dst=lyon.id,
        type="lives_in",
        origin="extracted",
        status="proposed",
    )
    for token in ("read", "dream", "review", "admin", "forget", "alice-dream", "bob-review"):
        response = client.post(
            f"/v1/edges/{edge.id}", json={"decision": "confirm"}, headers=auth(token)
        )
        assert response.status_code == 403, token
    other = client.post(f"/v1/edges/{edge.id}", json={"decision": "reject"}, headers=auth("bob"))
    assert other.status_code == 403
    assert store.get_edge(edge.id).status == "proposed"
    assert store.get_edge(edge.id).archived_at is None
    done = client.post(f"/v1/edges/{edge.id}", json={"decision": "confirm"}, headers=auth("alice"))
    assert done.status_code == 200
    undo = f"/v1/corrections/{done.json()['correction_id']}/undo"
    assert client.post(undo, headers=auth("bob")).status_code == 404  # not his record
    assert client.post(undo, headers=auth("review")).status_code == 403  # no write
    assert client.post(undo, headers=auth("alice")).status_code == 200


def test_p5_forget_wipes_an_edge_decision_from_the_journal(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "plain.db"
    settings = make_settings(tmp_path, db_path=path, db_key="", plaintext_database_for_tests=True)
    app = create_app(settings, clock=clock)
    try:
        with TestClient(app, headers=auth("all")) as client:
            store: Store = app.state.store
            karim = store.create_node(scopes=[ALICE], name="Karim", kind="person")
            lyon = store.create_node(scopes=[ALICE], name="Lyon", kind="place")
            edge = store.add_edge(
                scopes=[ALICE],
                src=karim.id,
                dst=lyon.id,
                type="lives_in",
                origin="extracted",
                status="proposed",
            )
            response = client.post(f"/v1/edges/{edge.id}", json={"decision": "confirm"})
            correction = response.json()["correction_id"]
            store.checkpoint("TRUNCATE")
            assert edge.id.encode() in _raw_bytes(path)
            removed = client.post("/v1/forget", json={"id": edge.id}).json()["removed"]
            assert removed["corrections_wiped"] == 1
            entry = Corrections(store).get(correction)
            assert (entry.forgotten, entry.payload, entry.changes, entry.records) == (
                True,
                None,
                None,
                (),
            )
            assert edge.id not in json.dumps(entry.__dict__)
            assert client.post(f"/v1/corrections/{correction}/undo").status_code == 404
    finally:
        app.state.store.close()
    assert edge.id.encode() not in _raw_bytes(path)


def _raw_bytes(path: Path) -> bytes:
    data = b""
    for candidate in (path, path.with_name(path.name + "-wal")):
        if candidate.exists():
            data += candidate.read_bytes()
    return data
