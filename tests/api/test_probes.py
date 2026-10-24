"""Probe set routes (SPEC 8.7, 11.2): create and delete need ``dream``; list, read
and run need ``read``; a token reaches a set only when bound to its scope."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import auth, remember
from tests.fixtures.corpus import CANARY


def make_set(client: TestClient, expected: list[str], **extra: Any) -> dict[str, Any]:
    body = {
        "scope": "user:alice",
        "name": "where people live",
        "budget_tokens": 300,
        "probes": [{"query": "Where does Karim live? Lyon", "expected": expected}],
        **extra,
    }
    response = client.post("/v1/probes", json=body)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def test_create_read_list_run_and_delete(client: TestClient) -> None:
    fact = remember(client, "Karim lives in Lyon.", node={"name": "Karim", "kind": "person"})
    created = make_set(client, [fact["fact"]["id"]])
    set_id = created["id"]
    assert set_id.startswith("pset_") and created["probes"] == 1

    listed = client.get("/v1/probes", params={"scope": "user:alice"}, headers=auth("read"))
    assert listed.status_code == 200
    assert [s["id"] for s in listed.json()["sets"]] == [set_id]

    one = client.get(f"/v1/probes/{set_id}", headers=auth("read")).json()
    assert one["scope"] == "user:alice" and one["budget_tokens"] == 300
    assert one["probes"][0]["query"] == "Where does Karim live? Lyon"
    assert one["probes"][0]["expected"] == [fact["fact"]["id"]]

    run = client.post(f"/v1/probes/{set_id}/run", headers=auth("read"))
    assert run.status_code == 200, run.text
    score = run.json()
    assert score["recall"] == 1.0 and score["scored"] == 1 and score["tier"] == 0
    assert score["probes"][0]["found"] == [fact["fact"]["id"]]
    assert "Lyon" not in run.text  # ids and numbers only

    deleted = client.delete(f"/v1/probes/{set_id}", headers=auth("dream"))
    assert deleted.status_code == 200 and deleted.json() == {"id": set_id, "deleted": 1}
    assert client.get(f"/v1/probes/{set_id}").status_code == 404
    assert client.delete(f"/v1/probes/{set_id}").status_code == 404


def test_the_default_budget_applies_when_none_is_given(client: TestClient) -> None:
    fact = remember(client, "Karim lives in Lyon.")
    body = {
        "scope": "user:alice",
        "name": "n",
        "probes": [{"query": "q", "expected": [fact["fact"]["id"]]}],
    }
    response = client.post("/v1/probes", json=body)
    assert response.status_code == 201
    one = client.get(f"/v1/probes/{response.json()['id']}").json()
    assert one["budget_tokens"] == client.app.state.settings.default_budget_tokens  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "body",
    [
        {"scope": "user:alice", "name": "n", "probes": []},
        {"scope": "user:alice", "name": "", "probes": [{"query": "q", "expected": ["x"]}]},
        {"scope": "user:alice", "name": "n", "probes": [{"query": "", "expected": ["x"]}]},
        {"scope": "user:alice", "name": "n", "probes": [{"query": "q", "expected": []}]},
        {
            "scope": "user:alice",
            "name": "n",
            "budget_tokens": 0,
            "probes": [{"query": "q", "expected": ["x"]}],
        },
        {"scope": "bad scope", "name": "n", "probes": [{"query": "q", "expected": ["x"]}]},
    ],
)
def test_malformed_sets_are_refused(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/v1/probes", json=body).status_code == 422


def test_a_budget_above_the_maximum_is_refused(client: TestClient) -> None:
    fact = remember(client, "Karim lives in Lyon.")
    body = {
        "scope": "user:alice",
        "name": "n",
        "budget_tokens": 10**9,
        "probes": [{"query": "q", "expected": [fact["fact"]["id"]]}],
    }
    assert client.post("/v1/probes", json=body).status_code == 422


@pytest.mark.security
def test_p9_an_invisible_or_unknown_expected_id_is_refused_alike(client: TestClient) -> None:
    bobs = remember(client, "Bob lives in Oslo.", scope="user:bob")["fact"]["id"]
    details = set()
    for record_id in (bobs, "fact_" + "0" * 26):
        body = {
            "scope": "user:alice",
            "name": "n",
            "probes": [{"query": "q", "expected": [record_id]}],
        }
        response = client.post("/v1/probes", json=body)
        assert response.status_code == 422
        details.add(response.json()["detail"].replace(record_id, "<id>"))
    assert len(details) == 1


@pytest.mark.security
def test_p9_a_set_outside_the_token_scopes_does_not_exist_for_it(client: TestClient) -> None:
    fact = remember(client, "Karim lives in Lyon.")
    set_id = make_set(client, [fact["fact"]["id"]])["id"]
    for method, path in (
        ("GET", f"/v1/probes/{set_id}"),
        ("POST", f"/v1/probes/{set_id}/run"),
    ):
        assert client.request(method, path, headers=auth("bob")).status_code == 404
    listed = client.get("/v1/probes", params={"scope": "user:alice"}, headers=auth("bob"))
    assert listed.status_code == 403
    # the owner's token still sees it
    assert client.get(f"/v1/probes/{set_id}", headers=auth("alice")).status_code == 200


@pytest.mark.security
def test_creating_or_deleting_outside_the_token_scopes_is_refused(client: TestClient) -> None:
    """A ``dream`` token bound to ``user:alice`` cannot create a set for Bob, nor
    see (so nor delete) one of Bob's."""
    bobs = remember(client, "Bob lives in Oslo.", scope="user:bob")["fact"]["id"]
    body = {"scope": "user:bob", "name": "n", "probes": [{"query": "q", "expected": [bobs]}]}
    assert client.post("/v1/probes", json=body, headers=auth("alice-dream")).status_code == 403
    set_id = client.post("/v1/probes", json=body).json()["id"]
    assert client.delete(f"/v1/probes/{set_id}", headers=auth("alice-dream")).status_code == 404
    assert client.get(f"/v1/probes/{set_id}").status_code == 200


@pytest.mark.security
def test_probe_routes_log_no_content(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    fact = remember(client, f"Karim said {CANARY}")
    body = {
        "scope": "user:alice",
        "name": f"set {CANARY}",
        "probes": [{"query": f"what did Karim say {CANARY}", "expected": [fact["fact"]["id"]]}],
    }
    set_id = client.post("/v1/probes", json=body).json()["id"]
    client.get(f"/v1/probes/{set_id}")
    client.get("/v1/probes", params={"scope": "user:alice"})
    client.post(f"/v1/probes/{set_id}/run")
    client.delete(f"/v1/probes/{set_id}")
    assert any("probe" in r.getMessage() for r in caplog.records)
    assert CANARY not in caplog.text and "Karim" not in caplog.text


@pytest.mark.security
def test_p5_forget_through_the_api_deletes_the_probes_that_expected_it(client: TestClient) -> None:
    fact = remember(client, "Karim lives in Lyon.")
    set_id = make_set(client, [fact["fact"]["id"]])["id"]
    forgotten = client.post("/v1/forget", json={"id": fact["fact"]["id"]})
    assert forgotten.status_code == 200
    assert forgotten.json()["removed"]["probes_deleted"] == 1
    one = client.get(f"/v1/probes/{set_id}").json()
    assert one["probes"] == []  # the question named the forgotten content
    run = client.post(f"/v1/probes/{set_id}/run").json()
    assert run["scored"] == 0 and run["recall"] is None and "empty" not in run


def test_unknown_ids_are_404(client: TestClient) -> None:
    for path in ("/v1/probes/pset_" + "0" * 26, "/v1/probes/nonsense"):
        assert client.get(path).status_code == 404
        assert client.post(path + "/run").status_code == 404
        assert client.delete(path).status_code == 404
