"""``GET /v1/lookup?name=`` finds visible nodes by name or alias (what memory_about uses)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import auth, remember


def test_lookup_by_name_and_alias_is_case_and_accent_insensitive(client: TestClient) -> None:
    written = remember(
        client, "Émilie préfère le thé", node={"name": "Émilie", "aliases": ["Mimi"]}
    )
    node_id = written["node"]["id"]
    for name in ("Émilie", "emilie", "ÉMILIE", "mimi"):
        body = client.get("/v1/lookup", params={"scope": "user:alice", "name": name}).json()
        assert body["records"] == [{"id": node_id, "kind": "node", "name": "Émilie"}], name
        assert body["name"] == name and body["external_id"] is None
    assert (
        client.get("/v1/lookup", params={"scope": "user:alice", "name": "nobody"}).json()["records"]
        == []
    )


def test_lookup_needs_exactly_one_of_external_id_or_name(client: TestClient) -> None:
    assert client.get("/v1/lookup", params={"scope": "user:alice"}).status_code == 422
    response = client.get(
        "/v1/lookup", params={"scope": "user:alice", "name": "x", "external_id": "y"}
    )
    assert response.status_code == 422
    assert "exactly one" in response.json()["detail"]


def test_lookup_by_name_ignores_archived_and_invisible_nodes(client: TestClient) -> None:
    alice = remember(client, "Karim in alice's scope", node={"name": "Karim"})["node"]["id"]
    bob = remember(client, "Karim in bob's scope", scope="user:bob", node={"name": "Karim"})[
        "node"
    ]["id"]
    seen = client.get("/v1/lookup", params={"scope": "user:alice", "name": "Karim"}).json()
    assert [r["id"] for r in seen["records"]] == [alice]
    as_bob = client.get(
        "/v1/lookup", params={"scope": "user:bob", "name": "karim"}, headers=auth("bob")
    ).json()
    assert [r["id"] for r in as_bob["records"]] == [bob]
    client.post("/v1/archive", json={"id": alice})
    assert (
        client.get("/v1/lookup", params={"scope": "user:alice", "name": "Karim"}).json()["records"]
        == []
    )


@pytest.mark.security
def test_p9_lookup_by_name_outside_the_token_scopes_is_refused(client: TestClient) -> None:
    response = client.get(
        "/v1/lookup", params={"scope": "user:bob", "name": "Karim"}, headers=auth("alice")
    )
    assert response.status_code == 403
