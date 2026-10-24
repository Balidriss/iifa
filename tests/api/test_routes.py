"""Route behaviour: remember, link, recall, expand, node view, archive, lookup, patch, errors."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.api.conftest import auth, remember


def test_health_and_stats(client: TestClient) -> None:
    health = client.get("/v1/health").json()
    assert health["status"] == "ok" and health["tier"] == 0
    assert health["backends"] == {"chat": None, "embeddings": None}
    assert health["queue"]["depth"] == 0 and health["queue"]["oldest_pending"] is None
    assert health["background"]["mode"] == "auto" and health["background"]["worker"] == "running"
    stats = client.get("/v1/stats").json()["counts"]
    assert stats["nodes"] == 0 and "facts" in stats and stats["queue"] == 0


def test_remember_creates_node_then_reuses_it(client: TestClient) -> None:
    first = remember(
        client, "Karim leads the solar project", node={"name": "Karim", "kind": "person"}
    )
    assert first["created_node"] and first["node"]["name"] == "Karim"
    assert first["fact"]["scopes"] == ["user:alice"]
    second = remember(client, "Karim moved to Lyon", node={"name": "karim", "aliases": ["K"]})
    assert not second["created_node"] and second["node"]["id"] == first["node"]["id"]
    third = remember(client, "Karim likes tea", node_id=first["node"]["id"])
    assert third["node"]["id"] == first["node"]["id"]
    view = client.get(f"/v1/nodes/{first['node']['id']}", params={"scope": "user:alice"}).json()
    assert view["aliases"] == ["K"]
    assert [f["statement"] for f in view["facts"]] == [
        "Karim likes tea",
        "Karim moved to Lyon",
        "Karim leads the solar project",
    ]


def test_remember_free_standing_fact_recall_and_attach(client: TestClient) -> None:
    """Amendment point 6 through the API."""
    fact = remember(client, "Prefers short answers and no emoji")
    assert fact["node"] is None
    recall = client.post(
        "/v1/recall", json={"scope": "user:alice", "query": "short answers"}
    ).json()
    assert [i["id"] for i in recall["items"]] == [fact["fact"]["id"]]
    assert recall["items"][0]["title"] == "Prefers short answers and no emoji"
    expanded = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": [fact["fact"]["id"]]}
    ).json()
    assert expanded["items"][0]["node_id"] is None and expanded["items"][0]["node_name"] is None

    alice = remember(client, "Alice is the user", node={"name": "Alice", "kind": "person"})
    patched = client.patch(
        f"/v1/facts/{fact['fact']['id']}", json={"node_id": alice["node"]["id"], "pinned": True}
    )
    assert patched.status_code == 200 and patched.json()["node_id"] == alice["node"]["id"]
    view = client.get(f"/v1/nodes/{alice['node']['id']}", params={"scope": "user:alice"}).json()
    assert view["facts"][0]["id"] == fact["fact"]["id"]  # pinned first
    handle = client.post(
        "/v1/recall", json={"scope": "user:alice", "query": "short answers"}
    ).json()["items"][0]
    assert handle["title"].startswith("Alice — ") and "pinned" in handle["flags"]
    detached = client.patch(f"/v1/facts/{fact['fact']['id']}", json={"detach": True}).json()
    assert detached["node_id"] is None
    assert (
        client.patch(
            f"/v1/facts/{fact['fact']['id']}", json={"detach": True, "node_id": "x"}
        ).status_code
        == 422
    )
    assert (
        client.patch(
            f"/v1/facts/{fact['fact']['id']}", json={"node_id": "node_" + "0" * 26}
        ).status_code
        == 404
    )
    assert client.patch("/v1/facts/fact_" + "0" * 26, json={"pinned": True}).status_code == 404


def test_recall_uses_edges_and_explains_why(client: TestClient) -> None:
    karim = remember(
        client, "Karim is an electrician in Lyon", node={"name": "Karim", "kind": "person"}
    )
    solar = remember(
        client,
        "The rooftop installation starts in March",
        node={"name": "Solar project", "kind": "project"},
    )
    link = client.post(
        "/v1/link",
        json={
            "scope": "user:alice",
            "src": karim["node"]["id"],
            "dst": solar["node"]["id"],
            "type": "works_on",
        },
    )
    assert link.status_code == 201 and link.json()["status"] == "confirmed"
    recall = client.post(
        "/v1/recall", json={"scope": "user:alice", "query": "electrician", "budget_tokens": 1000}
    ).json()
    whys = {i["id"]: i["why"] for i in recall["items"]}
    assert whys[karim["fact"]["id"]] == "query term 'electrician'"
    assert whys[solar["fact"]["id"]] == "edge path: Karim -works_on-> Solar project"
    assert recall["items"][0]["id"] == karim["fact"]["id"]  # direct hit outranks the hop
    # exclude_ids: what the harness already injected is never resent
    again = client.post(
        "/v1/recall",
        json={"scope": "user:alice", "query": "electrician", "exclude_ids": [karim["fact"]["id"]]},
    ).json()
    assert karim["fact"]["id"] not in {i["id"] for i in again["items"]}
    assert (
        client.post(
            "/v1/link",
            json={
                "scope": "user:alice",
                "src": karim["node"]["id"],
                "dst": "node_" + "0" * 26,
                "type": "x",
            },
        ).status_code
        == 404
    )


def test_recall_reasons_and_budget_validation(client: TestClient) -> None:
    empty = client.post("/v1/recall", json={"scope": "user:alice", "query": "anything"}).json()
    assert empty == {
        "items": [],
        "used_tokens": 0,
        "budget_tokens": 400,
        "omitted": 0,
        "reason": "no_match",
        "tier": 0,
        "dense": "off",
        "degraded": None,
        "query_vector": None,
    }
    assert (
        client.post("/v1/recall", json={"scope": "user:alice", "query": "..."}).json()["reason"]
        == "empty_query"
    )
    assert (
        client.post(
            "/v1/recall", json={"scope": "user:alice", "query": "x", "budget_tokens": 10**9}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/recall", json={"scope": "user:alice", "query": "x", "budget_tokens": -1}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/recall", json={"scope": "user:alice", "query": "x", "extra": 1}
        ).status_code
        == 422
    )
    assert client.post("/v1/recall", json={"scope": "user alice", "query": "x"}).status_code == 422
    assert client.post("/v1/recall", json={"query": "x"}).status_code == 422
    assert (
        client.post(
            "/v1/recall", json={"scope": "user:alice", "scopes": ["user:bob"], "query": "x"}
        ).status_code
        == 422
    )


def test_expand_levels_and_pagination(client: TestClient) -> None:
    ids = [
        remember(
            client,
            f"Fact number {i} about the budget of the solar project",
            source_reference=f"turn-{i}",
        )["fact"]["id"]
        for i in range(4)
    ]
    level2 = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": ids[:1], "level": 2}
    ).json()
    item = level2["items"][0]
    assert item["excerpt"] == "Fact number 0 about the budget of the solar project"
    assert item["source_id"].startswith("src_") and item["origin"] == "explicit"
    level1 = client.post("/v1/expand", json={"scope": "user:alice", "ids": ids[:1]}).json()
    assert "excerpt" not in level1["items"][0]

    page = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": ids, "budget_tokens": 250}
    ).json()
    assert 0 < len(page["items"]) < 4 and page["cursor"]
    seen = [i["id"] for i in page["items"]]
    while page["cursor"]:
        page = client.post(
            "/v1/expand",
            json={
                "scope": "user:alice",
                "ids": ids,
                "budget_tokens": 250,
                "cursor": page["cursor"],
            },
        ).json()
        seen += [i["id"] for i in page["items"]]
    assert seen == ids
    missing = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": ["fact_" + "0" * 26, "junk"]}
    ).json()
    assert missing["missing"] == ["fact_" + "0" * 26, "junk"] and missing["reason"] == "no_items"
    assert (
        client.post(
            "/v1/expand", json={"scope": "user:alice", "ids": ids, "cursor": "!!"}
        ).status_code
        == 422
    )
    assert (
        client.post("/v1/expand", json={"scope": "user:alice", "ids": ids, "level": 3}).status_code
        == 422
    )


def test_node_view_budget_and_not_found(client: TestClient) -> None:
    karim = remember(client, "Karim is an electrician", node={"name": "Karim"})
    node_id = karim["node"]["id"]
    tiny = client.get(
        f"/v1/nodes/{node_id}", params={"scope": "user:alice", "budget_tokens": 5}
    ).json()
    assert tiny["facts"] == [] and tiny["reason"] == "budget_too_small" and tiny["used_tokens"] == 0
    full = client.get(f"/v1/nodes/{node_id}", params={"scope": "user:alice"}).json()
    assert (
        full["facts"][0]["id"] == karim["fact"]["id"]
        and full["used_tokens"] <= full["budget_tokens"]
    )
    head_only = client.get(
        f"/v1/nodes/{node_id}",
        params={"scope": "user:alice", "budget_tokens": full["used_tokens"] - 1},
    ).json()
    assert head_only["facts"] == [] and head_only["omitted"] == 1 and head_only["name"] == "Karim"
    assert client.get(f"/v1/nodes/{node_id}", params={"scope": "user:bob"}).status_code == 404
    assert client.get(f"/v1/nodes/{node_id}").status_code == 422
    assert (
        client.get(
            f"/v1/nodes/{node_id}", params={"scope": "user:alice", "budget_tokens": 10**9}
        ).status_code
        == 422
    )


def test_archive_list_restore_and_errors(client: TestClient) -> None:
    fact = remember(client, "Karim likes tea", node={"name": "Karim"})
    assert client.post("/v1/archive", json={"id": fact["fact"]["id"]}).json() == {
        "id": fact["fact"]["id"],
        "kind": "fact",
        "archived": True,
    }
    listed = client.get("/v1/archive", params={"scope": "user:alice"}).json()
    assert [i["id"] for i in listed["items"]] == [fact["fact"]["id"]] and listed[
        "used_tokens"
    ] <= listed["budget_tokens"]
    assert (
        client.get("/v1/archive", params={"scope": "user:alice", "budget_tokens": 0}).json()[
            "reason"
        ]
        == "budget_too_small"
    )
    assert client.post("/v1/restore", json={"id": fact["fact"]["id"]}).json()["archived"] is False
    assert client.get("/v1/archive", params={"scope": "user:alice"}).json()["items"] == []
    for route in ("/v1/archive", "/v1/restore", "/v1/forget"):
        assert client.post(route, json={"id": "fact_" + "0" * 26}).status_code == 404
        assert client.post(route, json={"id": "junk"}).status_code == 404
    assert (
        client.post("/v1/archive", json={"id": fact["source_id"]}).status_code == 404
    )  # sources are not archivable


def test_lookup_by_external_id(client: TestClient) -> None:
    """Amendment point 5 through the API."""
    fact = remember(
        client,
        "Imported preference",
        external_id="crm-7",
        node={"name": "Karim", "external_id": "crm-7"},
    )
    found = client.get("/v1/lookup", params={"scope": "user:alice", "external_id": "crm-7"}).json()
    assert found["records"] == [
        {"id": fact["node"]["id"], "kind": "node"},
        {"id": fact["fact"]["id"], "kind": "fact"},
    ]
    assert (
        client.get("/v1/lookup", params={"scope": "user:bob", "external_id": "crm-7"}).json()[
            "records"
        ]
        == []
    )
    assert client.get("/v1/lookup", params={"scope": "user:alice"}).status_code == 422


def test_remember_write_scope_rules(client: TestClient) -> None:
    both = remember(client, "Shared with Bob", scopes=["user:alice/project:solar", "user:bob"])
    assert both["fact"]["scopes"] == ["user:alice/project:solar", "user:bob"]
    assert client.post("/v1/remember", json={"statement": "x"}).status_code == 422
    assert (
        client.post(
            "/v1/remember", json={"scope": "user:alice", "scopes": ["user:bob"], "statement": "x"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/remember",
            json={
                "scope": "user:alice",
                "statement": "x",
                "node": {"name": "N"},
                "node_id": "node_" + "0" * 26,
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/remember",
            json={"scope": "user:alice", "statement": "x", "node_id": "node_" + "0" * 26},
        ).status_code
        == 404
    )
    # A token bound to user:alice may not write into user:bob, nor touch Bob's records.
    assert (
        client.post(
            "/v1/remember", json={"scope": "user:bob", "statement": "x"}, headers=auth("alice")
        ).status_code
        == 403
    )
    bobs = remember(client, "Bob's own fact", scope="user:bob")
    assert (
        client.post(
            "/v1/archive", json={"id": bobs["fact"]["id"]}, headers=auth("alice")
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/archive", json={"id": both["fact"]["id"]}, headers=auth("alice")
        ).status_code
        == 200
    )
