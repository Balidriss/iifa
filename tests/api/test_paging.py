"""The node list (``GET /v1/nodes``) and cursor pagination of every list route:
a walk delivers each item exactly once whatever is written between
its pages, a page stops at its limit or its budget, whichever comes first, and
a cursor is opaque, signed and bound to its route, view, sort and filters."""

from __future__ import annotations

import base64
import json
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.api.paging import Cursors
from iifa.ids import new_id
from iifa.recall.paging import take
from iifa.scopes import Visibility
from iifa.store.repo import Store
from tests.api.conftest import auth, make_settings, remember
from tests.conftest import FakeClock

SCOPE = "user:alice"
BIG = 16000


def node(client: TestClient, name: str, statement: str | None = None, **extra: Any) -> str:
    node_ref = {"name": name, **extra}
    body = remember(client, statement or f"{name} is known", node=node_ref)
    node_id: str = body["node"]["id"]
    return node_id


def nodes_page(client: TestClient, **params: Any) -> dict[str, Any]:
    response = client.get("/v1/nodes", params={"scope": SCOPE, "budget_tokens": BIG, **params})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def walk(
    fetch: Callable[[str | None], dict[str, Any]],
    between: Callable[[int], None] = lambda _: None,
) -> list[dict[str, Any]]:
    """Every item of a walk, calling ``between(page number)`` after each page."""
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    for number in range(100):
        body = fetch(cursor)
        items.extend(body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            return items
        between(number)
    raise AssertionError("the walk never ended")


# --------------------------------------------------------------------------- GET /v1/nodes


def test_nodes_lists_counts_and_last_change(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    store: Store = app.state.store
    karim = node(client, "Karim", "Karim lives in Lyon", kind="person")
    clock.advance(days=1)
    remember(client, "Karim likes tea", node_id=karim)
    closed = remember(client, "Karim lives in Paris", node_id=karim)
    store.close_fact(closed["fact"]["id"])
    clock.advance(days=1)
    solar = node(client, "Solar project", kind="project")
    store.add_proposal(scopes=[SCOPE], kind="alias", node_id=karim, value="Kiki")
    store.add_proposal(scopes=["user:bob"], kind="alias", node_id=karim, value="K.")  # not Alice's
    node_bob = remember(client, "Bob is known", scope="user:bob", node={"name": "Bob"})

    body = nodes_page(client)
    assert body["sort"] == "name" and body["next_cursor"] is None and body["reason"] is None
    by_id = {i["id"]: i for i in body["items"]}
    assert list(by_id) == [karim, solar]  # P9: Bob's node is not in Alice's view
    assert node_bob["node"]["id"] not in by_id
    assert by_id[karim] == {
        "id": karim,
        "kind": "person",
        "name": "Karim",
        "aliases": [],
        "live_facts": 2,  # the closed fact is not live
        "pending_proposals": 1,  # the proposal in Bob's scope is not counted
        "changed_at": store.get_fact(closed["fact"]["id"]).created_at,
        "archived_at": None,
        "merged_into": None,
    }
    assert by_id[solar]["kind"] == "project" and by_id[solar]["live_facts"] == 1
    changed = nodes_page(client, sort="changed")
    assert [i["id"] for i in changed["items"]] == [solar, karim]  # newest change first
    clock.advance(days=1)
    remember(client, "Karim drives a van", node_id=karim)
    assert [i["id"] for i in nodes_page(client, sort="changed")["items"]] == [karim, solar]


def test_nodes_sort_by_name_folds_case_and_diacritics(client: TestClient) -> None:
    ids = {name: node(client, name) for name in ("zoé", "Émile", "alice", "Bogdan", "Иван")}
    names = [i["name"] for i in nodes_page(client)["items"]]
    assert names == ["alice", "Bogdan", "Émile", "zoé", "Иван"]
    assert set(ids.values()) == {i["id"] for i in nodes_page(client)["items"]}


def test_nodes_prefix_matches_a_name_or_an_alias_whole_unicode(client: TestClient) -> None:
    karim = node(client, "Karim")
    karima = node(client, "Karima")
    robert = node(client, "Robert", aliases=["Bob"])
    emile = node(client, "Émile Zola")
    ivan = node(client, "Иван")
    node(client, "Nadia")

    def matched(prefix: str) -> list[str]:
        return [i["id"] for i in nodes_page(client, prefix=prefix)["items"]]

    assert matched("kar") == [karim, karima]
    assert matched("KARIM") == [karim, karima]
    assert matched("Karima") == [karima]
    assert matched("bo") == [robert]  # through the alias
    robert_item = nodes_page(client, prefix="bo")["items"][0]
    assert robert_item["name"] == "Robert" and robert_item["aliases"] == ["Bob"]
    assert matched("emile z") == [emile] and matched("ÉMILE") == [emile]
    assert matched("ив") == [ivan] and matched("ИВАН") == [ivan]
    assert matched("arim") == []  # a prefix, never a substring
    assert matched("x") == []
    bad = client.get("/v1/nodes", params={"scope": SCOPE, "prefix": "--"})
    assert bad.status_code == 422 and "letter" in bad.json()["detail"]


def test_nodes_leave_out_archived_and_merged_unless_asked(client: TestClient, app: FastAPI) -> None:
    store: Store = app.state.store
    kept = node(client, "Karim")
    gone = node(client, "Nadia")
    merged = node(client, "K.")
    store.archive(gone)
    store.merge_node(merged, into=kept)
    assert [i["id"] for i in nodes_page(client)["items"]] == [kept]
    every = {i["id"]: i for i in nodes_page(client, include_archived=True)["items"]}
    assert set(every) == {kept, gone, merged}
    assert every[gone]["archived_at"] is not None and every[gone]["merged_into"] is None
    assert every[merged]["merged_into"] == kept and every[merged]["archived_at"] is not None
    assert every[kept]["archived_at"] is None


@pytest.mark.parametrize("sort", ["name", "changed"])
def test_a_node_walk_covers_every_node_once_whatever_is_written_between_pages(
    client: TestClient, clock: FakeClock, sort: str
) -> None:
    """Keyset pages over a snapshot: new nodes (whatever their name) and new facts
    about nodes not yet delivered (which move their last change) never make a
    node of the walk appear twice or not at all."""
    names = [f"{letter}node" for letter in "acegikmoqsuw"]
    before: list[str] = []
    for name in names:
        before.append(node(client, name))
        clock.advance(minutes=1)
    added: list[str] = []

    def write(number: int) -> None:
        clock.advance(minutes=1)
        for letter in "bdfhjlnprtvx"[number * 3 : number * 3 + 3]:
            added.append(node(client, f"{letter}node"))
        for node_id in before:  # every node of the walk changes, seen or not
            remember(client, f"fact {number} about it", node_id=node_id)

    items = walk(lambda c: nodes_page(client, sort=sort, limit=5, cursor=c), write)
    counts = Counter(i["id"] for i in items)
    assert set(counts) == set(before) and set(counts.values()) == {1}
    assert added and not set(added) & set(counts)  # the walk is the nodes of its first page
    keys = [i["name"].lower() if sort == "name" else i["changed_at"] for i in items]
    assert keys == (sorted(keys) if sort == "name" else sorted(keys, reverse=True))
    fresh = walk(lambda c: nodes_page(client, sort=sort, limit=5, cursor=c))
    assert len(fresh) == len(before) + len(added)  # a new walk sees them


def test_a_node_walk_leaves_a_node_archived_while_it_runs(client: TestClient, app: FastAPI) -> None:
    ids = [node(client, f"n{i:02d}") for i in range(6)]
    first = nodes_page(client, limit=3)
    app.state.store.archive(ids[4])
    rest = nodes_page(client, limit=3, cursor=first["next_cursor"])
    seen = [i["id"] for i in first["items"] + rest["items"]]
    assert seen == ids[:4] + ids[5:] and rest["next_cursor"] is None


@pytest.mark.parametrize("sort", ["name", "changed"])
def test_the_node_list_reads_in_a_constant_number_of_statements(store: Store, sort: str) -> None:
    """No query per node: a page of 5 or of 40 nodes, with aliases, facts and
    proposals, costs the same number of SQL statements."""
    view = Visibility.of_scope(SCOPE)
    for i in range(40):
        made = store.create_node(scopes=[SCOPE], name=f"n{i:02d}", aliases=[f"alias{i}"])
        store.add_fact(scopes=[SCOPE], node_id=made.id, statement=f"fact {i}", origin="explicit")
        store.add_proposal(scopes=[SCOPE], kind="alias", node_id=made.id, value=f"other{i}")
    mark = new_id("node").split("_", 1)[1]

    def statements_for(limit: int) -> int:
        seen: list[str] = []
        store._conn.set_trace_callback(seen.append)
        try:
            rows = store.node_list(
                view,
                sort=sort,
                prefix=None,
                include_archived=False,
                mark=mark,
                after=None,
                limit=limit,
            )
        finally:
            store._conn.set_trace_callback(None)
        assert len(rows) == limit and all(r["live_facts"] == 1 for r in rows)
        assert all(r["pending_proposals"] == 1 and len(r["aliases"]) == 1 for r in rows)
        return len(seen)

    assert statements_for(5) == statements_for(40)


# --------------------------------------------------------------------------- budget and limit


def test_a_page_stops_at_the_limit_or_the_budget_whichever_comes_first(client: TestClient) -> None:
    ids = [node(client, f"n{i:02d}") for i in range(6)]
    by_limit = nodes_page(client, limit=2)
    assert [i["id"] for i in by_limit["items"]] == ids[:2] and by_limit["omitted"] == 0
    one = by_limit["used_tokens"] // 2
    by_budget = nodes_page(client, limit=5, budget_tokens=one * 3 + one // 2)
    assert [i["id"] for i in by_budget["items"]] == ids[:3]
    assert by_budget["used_tokens"] <= by_budget["budget_tokens"]
    assert by_budget["omitted"] == 2  # the rest of the limit window, for the next page
    nxt = nodes_page(client, limit=5, cursor=by_budget["next_cursor"])
    assert [i["id"] for i in nxt["items"]] == ids[3:]  # exactly after the last delivered item
    assert nxt["next_cursor"] is None


def test_a_budget_too_small_delivers_nothing_and_keeps_the_position(client: TestClient) -> None:
    ids = [node(client, f"n{i:02d}") for i in range(3)]
    first = nodes_page(client, limit=1)
    tiny = nodes_page(client, cursor=first["next_cursor"], budget_tokens=1)
    assert tiny["items"] == [] and tiny["reason"] == "budget_too_small"
    assert tiny["next_cursor"] is None and tiny["omitted"] == 2
    again = nodes_page(client, cursor=first["next_cursor"])  # same cursor, larger budget
    assert [i["id"] for i in again["items"]] == ids[1:]
    empty = client.get("/v1/nodes", params={"scope": "user:carol"}).json()
    assert empty["items"] == [] and empty["reason"] == "no_items" and empty["next_cursor"] is None
    assert client.get("/v1/nodes", params={"scope": SCOPE, "limit": 0}).status_code == 422
    assert client.get("/v1/nodes", params={"scope": SCOPE, "limit": 1001}).status_code == 422
    assert client.get("/v1/nodes", params={"scope": SCOPE, "sort": "id"}).status_code == 422


# --------------------------------------------------------------------------- the other lists


def test_a_timeline_walk_delivers_every_change_once_while_facts_arrive(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    karim = node(client, "Karim", "Karim fact 0")
    facts = [remember(client, f"Karim fact {i}", node_id=karim)["fact"]["id"] for i in range(1, 9)]
    app.state.store.close_fact(facts[0])
    before = client.post(
        "/v1/timeline", json={"scope": SCOPE, "budget_tokens": BIG, "limit": 1000}
    ).json()["items"]
    assert len(before) == 10  # nine recorded, one closed

    def fetch(cursor: str | None) -> dict[str, Any]:
        body = {"scope": SCOPE, "node_id": karim, "limit": 3, "budget_tokens": BIG}
        response = client.post("/v1/timeline", json={**body, "cursor": cursor})
        assert response.status_code == 200, response.text
        result: dict[str, Any] = response.json()
        return result

    def write(_: int) -> None:
        clock.advance(minutes=1)
        remember(client, "Karim fact later", node_id=karim)

    items = walk(fetch, write)
    assert [(i["id"], i["change"]) for i in items] == [(i["id"], i["change"]) for i in before]


def test_a_proposal_walk_delivers_every_pending_proposal_once(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    karim = node(client, "Karim")
    made = [
        store.add_proposal(scopes=[SCOPE], kind="alias", node_id=karim, value=f"k{i}").id
        for i in range(7)
    ]

    def fetch(cursor: str | None) -> dict[str, Any]:
        params = {"scope": SCOPE, "limit": 3, "budget_tokens": BIG, "cursor": cursor}
        result: dict[str, Any] = client.get("/v1/proposals", params=params).json()
        return result

    added: list[str] = []

    def write(number: int) -> None:
        added.append(
            store.add_proposal(scopes=[SCOPE], kind="alias", node_id=karim, value=f"n{number}").id
        )

    items = walk(fetch, write)
    ids = [i["id"] for i in items]
    assert ids == made + added  # oldest first: what arrives joins the end, once

    def review(cursor: str | None) -> dict[str, Any]:
        params = {"scope": SCOPE, "limit": 4, "budget_tokens": BIG, "cursor": cursor}
        response = client.get("/v1/review", params=params)
        assert response.status_code == 200, response.text
        result: dict[str, Any] = response.json()
        return result

    assert [i["id"] for i in walk(review)] == ids
    alias_only = client.get(
        "/v1/proposals", params={"scope": SCOPE, "kind": "alias", "limit": 2, "budget_tokens": BIG}
    ).json()
    assert [i["id"] for i in alias_only["items"]] == ids[:2]


def test_an_archive_walk_delivers_every_archived_record_once(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    store: Store = app.state.store
    karim = node(client, "Karim")
    archived: list[str] = []
    for i in range(7):
        fact = remember(client, f"Karim fact {i}", node_id=karim)["fact"]["id"]
        store.archive(fact)
        archived.append(fact)
        clock.advance(minutes=1)
    other = node(client, "Nadia")
    store.archive(other)
    archived.append(other)

    def fetch(cursor: str | None) -> dict[str, Any]:
        params = {"scope": SCOPE, "limit": 3, "budget_tokens": BIG, "cursor": cursor}
        result: dict[str, Any] = client.get("/v1/archive", params=params).json()
        return result

    def write(number: int) -> None:
        clock.advance(minutes=1)
        store.archive(remember(client, f"later {number}", node_id=karim)["fact"]["id"])

    items = walk(fetch, write)
    assert [i["id"] for i in items] == list(reversed(archived))  # newest first, each once
    assert {i["kind"] for i in items} == {"fact", "node"}


# --------------------------------------------------------------------------- cursors are opaque and bound


def _tamper(cursor: str) -> str:
    """The same cursor with its position moved back to the start of the list
    (an empty name and id), the signature left as it was."""
    head, tail = cursor.split(".")
    payload = json.loads(base64.urlsafe_b64decode(head + "=" * (-len(head) % 4)))
    payload["p"][1:] = ["", ""]
    raw = json.dumps(payload, separators=(",", ":")).encode()
    forged = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert forged != head
    return f"{forged}.{tail}"


@pytest.mark.security
def test_p9_a_cursor_never_crosses_a_view_a_route_or_a_filter(client: TestClient) -> None:
    """A cursor from Alice's scope, replayed on Bob's (by a token that may read
    both, or by Bob), on another route, sort or filter, is refused with 400 and
    returns nothing; an altered cursor is refused the same way."""
    for i in range(4):
        node(client, f"alice{i}")
        remember(client, f"bob fact {i}", scope="user:bob", node={"name": f"bob{i}"})
    cursor = nodes_page(client, limit=1)["next_cursor"]
    assert cursor is not None
    replays: list[tuple[str, dict[str, Any], dict[str, str] | None]] = [
        ("/v1/nodes", {"scope": "user:bob"}, None),
        ("/v1/nodes", {"scope": "user:bob"}, auth("bob")),
        ("/v1/nodes", {"scopes": [SCOPE, "user:bob"]}, None),
        ("/v1/nodes", {"scope": SCOPE + "/agent:coder"}, None),
        ("/v1/nodes", {"scope": SCOPE, "sort": "changed"}, None),
        ("/v1/nodes", {"scope": SCOPE, "prefix": "a"}, None),
        ("/v1/nodes", {"scope": SCOPE, "include_archived": True}, None),
        ("/v1/archive", {"scope": SCOPE}, None),
        ("/v1/proposals", {"scope": SCOPE}, None),
        ("/v1/review", {"scope": SCOPE}, None),
        ("/v1/dreams", {"scope": SCOPE}, None),
    ]
    for path, params, headers in replays:
        response = client.get(path, params={**params, "cursor": cursor}, headers=headers)
        assert response.status_code == 400, (path, params, response.text)
        assert "items" not in response.json()
    timeline = client.post("/v1/timeline", json={"scope": SCOPE, "cursor": cursor})
    assert timeline.status_code == 400
    for bad in (
        _tamper(cursor),
        cursor[:-2] + "AA",
        cursor + "x",
        "abc",
        "a.b.c",
        "x" * 3000,
        "é.é",
    ):
        response = client.get("/v1/nodes", params={"scope": SCOPE, "cursor": bad})
        assert response.status_code in (400, 422), bad
    # the right cursor still works
    assert client.get("/v1/nodes", params={"scope": SCOPE, "cursor": cursor}).status_code == 200


@pytest.mark.security
def test_a_cursor_signed_by_another_service_is_refused(client: TestClient, tmp_path: Any) -> None:
    for i in range(3):
        node(client, f"n{i}")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = create_app(make_settings(other_dir))
    try:
        with TestClient(other, headers=auth("all")) as other_client:
            for i in range(3):
                node(other_client, f"n{i}")
            foreign = nodes_page(other_client, limit=1)["next_cursor"]
    finally:
        other.state.store.close()
    response = client.get("/v1/nodes", params={"scope": SCOPE, "cursor": foreign})
    assert response.status_code == 400


def test_the_cursor_key_is_kept_in_the_store(store: Store) -> None:
    first = Cursors.for_store(store)
    again = Cursors.for_store(store)
    cursor = first.encode({"route": "x"}, ["a", 1])
    assert again.decode(cursor, {"route": "x"}) == ["a", 1]
    assert again.decode(None, {"route": "x"}) is None
    with pytest.raises(ValueError, match="16 bytes"):
        Cursors(b"short")


def test_a_well_signed_cursor_of_the_wrong_shape_is_refused(
    client: TestClient, app: FastAPI
) -> None:
    """A position the route does not expect (another route's shape) is a 400."""
    cursors: Cursors = app.state.cursors
    context = {
        "route": "archive",
        "view": {"exact": [SCOPE], "subtrees": []},
    }
    for position in (["only one"], [1, "x"], ["a", True]):
        cursor = cursors.encode(context, position)
        assert (
            client.get("/v1/archive", params={"scope": SCOPE, "cursor": cursor}).status_code == 400
        )


def test_a_cursor_binds_the_scopes_of_a_view_not_their_order(client: TestClient) -> None:
    for i in range(3):
        node(client, f"n{i}")
    both = [SCOPE, "user:bob"]
    first = client.get("/v1/nodes", params={"scopes": both, "limit": 1, "budget_tokens": BIG})
    swapped = client.get(
        "/v1/nodes",
        params={"scopes": list(reversed(both)), "cursor": first.json()["next_cursor"]},
    )
    assert swapped.status_code == 200 and len(swapped.json()["items"]) == 2


def test_a_change_dropped_between_two_reads_does_not_end_the_walk() -> None:
    """``more_after``: items read and then dropped (a fact forgotten between the
    two reads of a timeline page) still say that more follow."""
    page = take([1, 2], limit=2, budget_tokens=10, cost=lambda _: 1, more_after=True)
    assert page.items == [1, 2] and page.more and page.last == 2
    assert not take([1, 2], limit=2, budget_tokens=10, cost=lambda _: 1).more
