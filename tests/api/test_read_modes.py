"""Amendment point 4: three read modes - scope + ancestors, an explicit list, a subtree (admin)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import auth, remember


def _ids(client: TestClient, **view: object) -> set[str]:
    body = {"query": "solar", "budget_tokens": 4000, **view}
    response = client.post("/v1/recall", json=body)
    assert response.status_code == 200, response.text
    return {i["id"] for i in response.json()["items"]}


def test_default_mode_sees_scope_and_ancestors_only(client: TestClient) -> None:
    root = remember(client, "solar at user level", scope="user:alice")["fact"]["id"]
    coder = remember(client, "solar in the coder agent", scope="user:alice/agent:coder")["fact"][
        "id"
    ]
    other = remember(client, "solar in another agent", scope="user:alice/agent:writer")["fact"][
        "id"
    ]
    bob = remember(client, "solar for bob", scope="user:bob")["fact"]["id"]
    assert _ids(client, scope="user:alice/agent:coder") == {root, coder}
    assert _ids(client, scope="user:alice") == {root}
    assert _ids(client, scope="user:bob") == {bob}
    assert other not in _ids(client, scope="user:alice/agent:coder")


def test_explicit_list_mode_sees_exactly_those_scopes(client: TestClient) -> None:
    root = remember(client, "solar at user level", scope="user:alice")["fact"]["id"]
    coder = remember(client, "solar in the coder agent", scope="user:alice/agent:coder")["fact"][
        "id"
    ]
    writer = remember(client, "solar in the writer agent", scope="user:alice/agent:writer")["fact"][
        "id"
    ]
    assert _ids(client, scopes=["user:alice/agent:coder", "user:alice/agent:writer"]) == {
        coder,
        writer,
    }
    assert root not in _ids(client, scopes=["user:alice/agent:coder"])
    # the token must be bound to every listed scope
    response = client.post(
        "/v1/recall",
        json={"scopes": ["user:alice", "user:bob"], "query": "solar"},
        headers=auth("alice"),
    )
    assert response.status_code == 403
    assert client.post("/v1/recall", json={"scopes": [], "query": "solar"}).status_code == 422


def test_subtree_mode_needs_admin_and_sees_everything_below(client: TestClient) -> None:
    root = remember(client, "solar at user level", scope="user:alice")["fact"]["id"]
    coder = remember(client, "solar in the coder agent", scope="user:alice/agent:coder")["fact"][
        "id"
    ]
    deep = remember(client, "solar deep down", scope="user:alice/agent:coder/task:42")["fact"]["id"]
    bob = remember(client, "solar for bob", scope="user:bob")["fact"]["id"]
    assert _ids(client, subtree="user:alice") == {root, coder, deep}
    assert bob not in _ids(client, subtree="user:alice")
    denied = client.post(
        "/v1/recall", json={"subtree": "user:alice", "query": "solar"}, headers=auth("alice")
    )
    assert denied.status_code == 403 and "admin" in denied.json()["detail"]
    outside = client.post(
        "/v1/recall", json={"subtree": "user:bob", "query": "solar"}, headers=auth("alice-admin")
    )
    assert outside.status_code == 403
    allowed = client.post(
        "/v1/recall",
        json={"subtree": "user:alice", "query": "solar", "budget_tokens": 4000},
        headers=auth("alice-admin"),
    )
    assert {i["id"] for i in allowed.json()["items"]} == {root, coder, deep}


def test_modes_apply_to_get_routes_too(client: TestClient) -> None:
    fact = remember(
        client, "solar deep down", scope="user:alice/agent:coder/task:42", external_id="ext-1"
    )
    lookup = client.get(
        "/v1/lookup", params={"external_id": "ext-1", "subtree": "user:alice"}
    ).json()
    assert [r["id"] for r in lookup["records"]] == [fact["fact"]["id"]]
    assert (
        client.get("/v1/lookup", params={"external_id": "ext-1", "scope": "user:alice"}).json()[
            "records"
        ]
        == []
    )
    listed = client.get(
        "/v1/lookup",
        params={"external_id": "ext-1", "scopes": ["user:alice/agent:coder/task:42", "user:bob"]},
    ).json()
    assert len(listed["records"]) == 1
    assert (
        client.get(
            "/v1/lookup",
            params={"external_id": "ext-1", "scope": "user:alice", "subtree": "user:alice"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/lookup",
            params={"external_id": "ext-1", "subtree": "user:alice"},
            headers=auth("alice"),
        ).status_code
        == 403
    )
