"""The reading routes of SPEC 8.2: ``/v1/timeline`` (fact
changes in a date range), ``/v1/at`` (what was known at a date, P7) and
``/v1/path`` (how two nodes are related), each under budget (P1) and inside
the view (P9)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.recall.history import parse_bound
from tests.api.conftest import auth, remember
from tests.conftest import FakeClock

SCOPE = "user:alice"


def link(client: TestClient, src: str, dst: str, type_: str = "works_on", **extra: Any) -> str:
    body = {"scope": SCOPE, "src": src, "dst": dst, "type": type_, **extra}
    response = client.post("/v1/link", json=body)
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


# --------------------------------------------------------------------------- timeline


def test_timeline_lists_fact_changes_newest_first_inside_the_range(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    karim = remember(client, "Karim lives in Paris", node={"name": "Karim", "kind": "person"})
    node_id = karim["node"]["id"]
    clock.advance(days=10)
    moved = remember(client, "Karim lives in Lyon", node_id=node_id)
    app.state.store.close_fact(karim["fact"]["id"], superseded_by=moved["fact"]["id"])
    clock.advance(days=10)
    remember(client, "Karim prefers tea", node_id=node_id)

    body = client.post(
        "/v1/timeline", json={"scope": SCOPE, "node_id": node_id, "budget_tokens": 2000}
    ).json()
    assert body["node"] == {"id": node_id, "kind": "node", "name": "Karim", "node_kind": "person"}
    changes = [(i["change"], i["line"]) for i in body["items"]]
    assert changes == [
        ("recorded", "Karim prefers tea"),
        ("recorded", "Karim lives in Lyon"),
        ("closed", "Karim lives in Paris"),
        ("recorded", "Karim lives in Paris"),
    ]
    closed = body["items"][2]
    assert closed["superseded_by"] == moved["fact"]["id"] and closed["at"] == closed["valid_to"]
    assert closed["node_name"] == "Karim" and closed["origin"] == "explicit"
    assert body["used_tokens"] <= 2000 and body["reason"] is None and body["omitted"] == 0
    assert [i["at"] for i in body["items"]] == sorted(
        (i["at"] for i in body["items"]), reverse=True
    )

    # a range keeps only the changes inside it
    ranged = client.post(
        "/v1/timeline",
        json={
            "scope": SCOPE,
            "node_id": node_id,
            "since": "2026-10-01",
            "until": "2026-10-10",
            "budget_tokens": 2000,
        },
    ).json()
    assert [(i["change"], i["line"]) for i in ranged["items"]] == [
        ("recorded", "Karim lives in Lyon"),
        ("closed", "Karim lives in Paris"),
    ]
    assert ranged["since"] == parse_bound("2026-10-01")
    assert ranged["until"] == parse_bound("2026-10-10", end=True)
    # the whole view, without a node
    everything = client.post("/v1/timeline", json={"scope": SCOPE, "budget_tokens": 2000}).json()
    assert everything["node"] is None and len(everything["items"]) == 4
    empty = client.post("/v1/timeline", json={"scope": SCOPE, "since": "2030-01"}).json()
    assert empty["items"] == [] and empty["reason"] == "no_changes"


def test_timeline_validates_dates_nodes_and_budget(client: TestClient) -> None:
    karim = remember(client, "Karim lives in Paris", node={"name": "Karim"})
    node_id = karim["node"]["id"]
    assert (
        client.post("/v1/timeline", json={"scope": SCOPE, "since": "yesterday"}).status_code == 422
    )
    assert client.post("/v1/timeline", json={"scope": SCOPE, "until": "2026-13"}).status_code == 422
    assert (
        client.post("/v1/timeline", json={"scope": SCOPE, "node_id": "node_nope"}).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/timeline", json={"scope": SCOPE, "node_id": node_id}, headers=auth("bob")
        ).status_code
        == 403
    )
    small = client.post(
        "/v1/timeline", json={"scope": SCOPE, "node_id": node_id, "budget_tokens": 5}
    ).json()
    assert small["items"] == [] and small["reason"] == "budget_too_small" and small["omitted"] == 1
    assert client.post("/v1/timeline", json={"scope": SCOPE, "limit": 0}).status_code == 422


def test_parse_bound() -> None:
    assert parse_bound("2026-03") == "2026-03-01T00:00:00.000000+00:00"
    assert parse_bound("2026-02", end=True) == "2026-02-28T23:59:59.999999+00:00"
    assert parse_bound("2026-03-14") == "2026-03-14T00:00:00.000000+00:00"
    assert parse_bound("2026-03-14", end=True) == "2026-03-14T23:59:59.999999+00:00"
    assert parse_bound("2026-03-14T10:00:00") == "2026-03-14T10:00:00.000000+00:00"
    assert parse_bound("2026-03-14T12:00:00+02:00") == "2026-03-14T10:00:00.000000+00:00"
    for bad in ("2026-13", "2026-1", "soon", "2026-02-30"):
        with pytest.raises(ValueError):
            parse_bound(bad)


# --------------------------------------------------------------------------- at


def test_at_answers_what_was_known_at_a_date(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    """P7: history stays queryable. Recorded by D and not closed by D: known at D."""
    paris = remember(client, "Karim lives in Paris", node={"name": "Karim"})
    node_id = paris["node"]["id"]
    day_one = app.state.store.now()
    clock.advance(days=30)
    lyon = remember(client, "Karim lives in Lyon", node_id=node_id)
    app.state.store.close_fact(paris["fact"]["id"], superseded_by=lyon["fact"]["id"])

    def known(date: str) -> list[str]:
        body = client.post(
            "/v1/at", json={"scope": SCOPE, "node_id": node_id, "date": date, "budget_tokens": 2000}
        ).json()
        assert body["id"] == node_id and body["name"] == "Karim" and body["used_tokens"] <= 2000
        return [i["statement"] for i in body["items"]]

    assert known("2020-01-01") == []
    assert known(day_one) == ["Karim lives in Paris"]
    assert known("2026-10-10") == ["Karim lives in Paris"]
    assert known(clock().isoformat()) == ["Karim lives in Lyon"]
    assert known("2030-01-01") == ["Karim lives in Lyon"]
    body = client.post(
        "/v1/at", json={"scope": SCOPE, "node_id": node_id, "date": "2026-10-10"}
    ).json()
    item = body["items"][0]
    assert item["superseded_by"] == lyon["fact"]["id"] and item["valid_to"] is not None
    assert item["recorded_at"] == day_one and body["reason"] is None
    nothing = client.post("/v1/at", json={"scope": SCOPE, "node_id": node_id, "date": "2020-01"})
    assert nothing.json()["reason"] == "nothing_known"
    assert (
        client.post("/v1/at", json={"scope": SCOPE, "node_id": node_id, "date": "x"}).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/at", json={"scope": SCOPE, "node_id": "node_x", "date": "2026"}
        ).status_code
        == 404
    )
    tiny = client.post(
        "/v1/at",
        json={"scope": SCOPE, "node_id": node_id, "date": "2030-01-01", "budget_tokens": 3},
    ).json()
    assert tiny["items"] == [] and tiny["reason"] == "budget_too_small" and tiny["omitted"] == 2


# --------------------------------------------------------------------------- path


def test_path_finds_the_shortest_visible_route(client: TestClient) -> None:
    karim = remember(client, "Karim is an electrician", node={"name": "Karim"})["node"]["id"]
    solar = remember(client, "Solar is a project", node={"name": "Solar project"})["node"]["id"]
    budget = remember(client, "The budget topic", node={"name": "Budget"})["node"]["id"]
    nadia = remember(client, "Nadia is an accountant", node={"name": "Nadia"})["node"]["id"]
    alone = remember(client, "An island", node={"name": "Island"})["node"]["id"]
    link(client, karim, solar, "works_on")
    link(client, budget, solar, "about")
    link(client, nadia, budget, "manages")
    link(client, karim, nadia, "knows")  # the short way from Karim to Nadia

    body = client.post(
        "/v1/path", json={"scope": SCOPE, "src": karim, "dst": budget, "budget_tokens": 2000}
    ).json()
    assert body["found"] is True and body["hops"] == 2
    assert body["path"] in (
        "Karim -works_on-> Solar project <-about- Budget",
        "Karim -knows-> Nadia -manages-> Budget",
    )
    steps = body["steps"]
    assert steps[0]["from"] == karim and steps[-1]["to"] == budget
    assert {s["direction"] for s in steps} <= {"forward", "backward"}
    assert body["used_tokens"] <= 2000 and body["reason"] is None
    direct = client.post("/v1/path", json={"scope": SCOPE, "src": karim, "dst": nadia}).json()
    assert direct["hops"] == 1 and direct["path"] == "Karim -knows-> Nadia"
    reverse = client.post("/v1/path", json={"scope": SCOPE, "src": nadia, "dst": karim}).json()
    assert (
        reverse["path"] == "Nadia <-knows- Karim" and reverse["steps"][0]["direction"] == "backward"
    )
    none = client.post("/v1/path", json={"scope": SCOPE, "src": karim, "dst": alone}).json()
    assert none["found"] is False and none["hops"] is None and none["reason"] == "no_path"
    same = client.post("/v1/path", json={"scope": SCOPE, "src": karim, "dst": karim}).json()
    assert same["found"] is True and same["hops"] == 0 and same["reason"] == "same_node"
    bounded = client.post(
        "/v1/path", json={"scope": SCOPE, "src": karim, "dst": budget, "max_hops": 1}
    ).json()
    assert bounded["found"] is False
    assert (
        client.post("/v1/path", json={"scope": SCOPE, "src": karim, "dst": "node_x"}).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/path", json={"scope": SCOPE, "src": karim, "dst": budget, "max_hops": 9}
        ).status_code
        == 422
    )


def test_path_is_delivered_whole_or_not_at_all(client: TestClient) -> None:
    karim = remember(client, "Karim", node={"name": "Karim"})["node"]["id"]
    nadia = remember(client, "Nadia", node={"name": "Nadia"})["node"]["id"]
    link(client, karim, nadia, "knows")
    small = client.post(
        "/v1/path", json={"scope": SCOPE, "src": karim, "dst": nadia, "budget_tokens": 20}
    ).json()
    assert small["found"] is True and small["hops"] == 1
    assert small["steps"] == [] and small["path"] is None
    assert small["reason"] == "budget_too_small" and small["omitted"] == 1
    assert small["used_tokens"] == 0


@pytest.mark.security
def test_p9_path_never_crosses_an_invisible_node_or_edge(client: TestClient) -> None:
    """A path through an invisible node is a leak: with the only route running
    through Bob's node, Alice gets no path at all, and a route over an invisible
    edge is not taken either."""
    karim = remember(client, "Karim", scope="user:alice", node={"name": "Karim"})["node"]["id"]
    budget = remember(client, "Budget", scope="user:alice", node={"name": "Budget"})["node"]["id"]
    secret = remember(client, "Zorblax", scope="user:bob", node={"name": "Zorblax"})["node"]["id"]
    # edges Alice can see, through a node she cannot
    for src, dst in ((karim, secret), (secret, budget)):
        assert (
            client.post(
                "/v1/link",
                json={"scopes": ["user:alice", "user:bob"], "src": src, "dst": dst, "type": "via"},
            ).status_code
            == 201
        )
    alice = auth("alice")
    body = client.post(
        "/v1/path",
        json={"scope": "user:alice", "src": karim, "dst": budget, "max_hops": 4},
        headers=alice,
    ).json()
    assert body["found"] is False and "Zorblax" not in str(body)
    # an edge Alice cannot see, between two nodes she can
    assert (
        client.post(
            "/v1/link", json={"scope": "user:bob", "src": karim, "dst": budget, "type": "hidden"}
        ).status_code
        == 404  # both ends must be visible in the write view: Bob's scope alone sees neither
    )
    assert (
        client.post(
            "/v1/link",
            json={
                "scopes": ["user:bob", "user:alice"],
                "src": karim,
                "dst": budget,
                "type": "shared",
            },
        ).status_code
        == 201
    )
    found = client.post(
        "/v1/path", json={"scope": "user:alice", "src": karim, "dst": budget}, headers=alice
    ).json()
    assert found["found"] is True and found["path"] == "Karim -shared-> Budget"
    # Bob's view (both scopes) sees the route through Zorblax too, but the shared edge is shorter
    both = client.post(
        "/v1/path", json={"scopes": ["user:alice", "user:bob"], "src": karim, "dst": budget}
    ).json()
    assert both["hops"] == 1
    # an invisible end is "no such node", not a hint that it exists elsewhere
    assert (
        client.post(
            "/v1/path", json={"scope": "user:alice", "src": karim, "dst": secret}, headers=alice
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/at",
            json={"scope": "user:alice", "node_id": secret, "date": "2030-01-01"},
            headers=alice,
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/timeline", json={"scope": "user:alice", "node_id": secret}, headers=alice
        ).status_code
        == 404
    )


@pytest.mark.security
def test_p9_timeline_of_a_view_never_names_an_invisible_node(client: TestClient) -> None:
    shared = remember(
        client,
        "A shared fact about Zorblax",
        scopes=["user:alice", "user:bob"],
        node={"name": "Zorblax"},
    )
    # the fact is visible to Alice, its node (created in the same scopes) too; move the node
    # out of her sight by re-creating the situation: a fact in Alice's scope on Bob's node
    store = client.app.state.store  # type: ignore[attr-defined]
    bob_node = store.create_node(scopes=["user:bob"], name="Hidden subject")
    store.add_fact(
        scopes=["user:alice"], node_id=bob_node.id, statement="Alice knows this", origin="explicit"
    )
    body = client.post(
        "/v1/timeline", json={"scope": "user:alice", "budget_tokens": 2000}, headers=auth("alice")
    ).json()
    lines = {i["line"]: i for i in body["items"]}
    assert "Alice knows this" in lines
    assert lines["Alice knows this"]["node_id"] is None
    assert lines["Alice knows this"]["node_name"] is None
    assert lines["A shared fact about Zorblax"]["node_id"] == shared["node"]["id"]
    assert "Hidden subject" not in str(body)


# --------------------------------------------------------------------------- reinforcement (SPEC 5.3)


def test_expand_reinforces_the_edge_that_surfaced_the_item(
    client: TestClient, app: FastAPI
) -> None:
    karim = remember(client, "Karim is an electrician", node={"name": "Karim"})["node"]["id"]
    solar = remember(client, "The solar project starts in March", node={"name": "Solar project"})[
        "node"
    ]["id"]
    edge_id = link(client, karim, solar, "works_on")
    store = app.state.store
    before = store.get_edge(edge_id)
    assert before.use_count == 0 and before.last_used_at is None and before.weight == 1.0
    recalled = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "electrician", "budget_tokens": 2000}
    ).json()
    via_edge = [i for i in recalled["items"] if i["why"].startswith("edge path")]
    assert via_edge and "Solar project" in via_edge[0]["why"]
    expanded = client.post(
        "/v1/expand", json={"scope": SCOPE, "ids": [via_edge[0]["id"]], "budget_tokens": 2000}
    ).json()
    assert expanded["reinforced_edges"] == [edge_id]
    after = store.get_edge(edge_id)
    assert after.use_count == 1 and after.last_used_at is not None and after.weight == 1.25
    # once per recall: expanding again without a new recall changes nothing
    again = client.post(
        "/v1/expand", json={"scope": SCOPE, "ids": [via_edge[0]["id"]], "budget_tokens": 2000}
    ).json()
    assert again["reinforced_edges"] == [] and store.get_edge(edge_id).use_count == 1
    # an item that a query term surfaced directly reinforces nothing
    direct = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "solar project", "budget_tokens": 2000}
    ).json()
    direct_item = next(i for i in direct["items"] if i["why"].startswith("query term"))
    plain = client.post("/v1/expand", json={"scope": SCOPE, "ids": [direct_item["id"]]}).json()
    assert plain["reinforced_edges"] == []
