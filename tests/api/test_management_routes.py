"""What a management page needs beside the node corrections: a person decides a proposed
edge, an explicit link confirms the live
edge already there instead of adding a twin, the archive says what each record
was, and the dreams of a scope can be listed whichever client started them."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, get_args

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.api.dream import RunStatus
from iifa.dream import journal
from iifa.scopes import Visibility
from iifa.store.corrections import Corrections
from iifa.store.models import Edge, Node
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.helpers import auth, make_settings

SCOPE = "user:alice"
VIEW = Visibility.of_scope(SCOPE)
BIG = 16000


def _nodes(app: FastAPI) -> tuple[Node, Node]:
    store: Store = app.state.store
    karim = store.create_node(scopes=[SCOPE], name="Karim", kind="person")
    lyon = store.create_node(scopes=[SCOPE], name="Lyon", kind="place")
    return karim, lyon


def _proposed(app: FastAPI, src: Node, dst: Node, type_: str = "lives_in") -> Edge:
    store: Store = app.state.store
    return store.add_edge(
        scopes=[SCOPE], src=src.id, dst=dst.id, type=type_, origin="extracted", status="proposed"
    )


def _decide(client: TestClient, edge_id: str, token: str = "alice", **body: Any) -> Any:
    return client.post(f"/v1/edges/{edge_id}", json=body, headers=auth(token))


def _edges_of(client: TestClient, node_id: str) -> list[dict[str, Any]]:
    response = client.get(f"/v1/nodes/{node_id}", params={"scope": SCOPE, "budget_tokens": BIG})
    assert response.status_code == 200, response.text
    edges: list[dict[str, Any]] = response.json()["edges"]
    return edges


def _link(client: TestClient, src: str, dst: str, type_: str, token: str = "alice") -> Any:
    return client.post(
        "/v1/link",
        json={"scope": SCOPE, "src": src, "dst": dst, "type": type_},
        headers=auth(token),
    )


# --------------------------------------------------------------------------- a proposed edge


def test_a_person_confirms_a_proposed_edge(client: TestClient, app: FastAPI) -> None:
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    response = _decide(client, edge.id, decision="confirm", rev=edge.decision_rev)
    assert response.status_code == 200, response.text
    out = response.json()
    assert (out["id"], out["status"], out["archived"]) == (edge.id, "confirmed", False)
    assert out["correction_id"].startswith("corr_")
    edges = _edges_of(client, karim.id)
    assert [(e["id"], e["status"]) for e in edges] == [(edge.id, "confirmed")]
    entry = client.get(f"/v1/corrections/{out['correction_id']}").json()
    assert (entry["op"], entry["token"], entry["records"]) == ("confirm_edge", "alice", [edge.id])


def test_undoing_a_confirmation_puts_the_edge_back_proposed(
    client: TestClient, app: FastAPI
) -> None:
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    correction = _decide(client, edge.id, decision="confirm").json()["correction_id"]
    undone = client.post(f"/v1/corrections/{correction}/undo", headers=auth("alice"))
    assert undone.status_code == 200, undone.text
    assert undone.json()["status"] == "undone"
    assert app.state.store.get_edge(edge.id).status == "proposed"


def test_a_rejected_edge_is_archived_shown_in_the_archive_and_undone(
    client: TestClient, app: FastAPI
) -> None:
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    response = _decide(client, edge.id, decision="reject")
    assert response.status_code == 200, response.text
    assert response.json()["archived"] is True
    assert _edges_of(client, karim.id) == []
    archive = client.get("/v1/archive", params={"scope": SCOPE, "budget_tokens": BIG}).json()
    assert archive["items"] == [
        {
            "id": edge.id,
            "kind": "edge",
            "archived_at": archive["items"][0]["archived_at"],
            "type": "lives_in",
            "status": "proposed",
            "src": karim.id,
            "src_name": "Karim",
            "dst": lyon.id,
            "dst_name": "Lyon",
        }
    ]
    correction = response.json()["correction_id"]
    assert client.post(f"/v1/corrections/{correction}/undo").status_code == 200
    assert [(e["id"], e["status"]) for e in _edges_of(client, karim.id)] == [(edge.id, "proposed")]


def test_only_a_live_proposed_edge_is_decided(client: TestClient, app: FastAPI) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    confirmed = store.add_edge(
        scopes=[SCOPE], src=karim.id, dst=lyon.id, type="knows", origin="explicit"
    )
    response = _decide(client, confirmed.id, decision="reject")
    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "not_proposed"
    archived = _proposed(app, karim, lyon, "visits")
    store.archive(archived.id)
    refused = _decide(client, archived.id, decision="confirm")
    assert refused.json()["detail"]["reason"] == "not_live"
    closed = _proposed(app, karim, lyon, "likes")
    store.close_edge(closed.id)
    refused = _decide(client, closed.id, decision="confirm")
    assert refused.json()["detail"]["reason"] == "not_live"
    edge = _proposed(app, karim, lyon)
    stale = _decide(client, edge.id, decision="confirm", rev=edge.decision_rev - 1)
    assert stale.status_code == 409
    assert stale.json()["detail"] == {
        "reason": "stale",
        "records": [edge.id],
        "revs": {edge.id: edge.decision_rev},
    }
    assert _decide(client, edge.id, decision="maybe").status_code == 422
    assert _decide(client, "edge_" + "0" * 26, decision="confirm").status_code == 404
    assert _decide(client, karim.id, decision="confirm").status_code == 404
    assert store.get_edge(edge.id).status == "proposed"  # nothing refused wrote anything


# --------------------------------------------------------------------------- the edge's rev


def _read_rev(client: TestClient, node_id: str, edge_id: str) -> int:
    rev: int = next(e["rev"] for e in _edges_of(client, node_id) if e["id"] == edge_id)
    return rev


def test_the_rev_the_node_view_shows_is_the_one_a_decision_takes(
    client: TestClient, app: FastAPI
) -> None:
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    rev = _read_rev(client, karim.id, edge.id)
    assert rev == app.state.store.get_edge(edge.id).decision_rev
    response = _decide(client, edge.id, decision="confirm", rev=rev)
    assert response.status_code == 200, response.text
    assert response.json()["rev"] == _read_rev(client, karim.id, edge.id) > rev


def test_a_decision_after_another_write_changed_the_status_is_stale(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    rev = _read_rev(client, karim.id, edge.id)
    store.confirm_edge(edge.id)  # a second source confirmed it meanwhile
    store.set_edge_status(edge.id, "proposed")
    current = store.get_edge(edge.id).decision_rev
    response = _decide(client, edge.id, decision="reject", rev=rev)
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "reason": "stale",
        "records": [edge.id],
        "revs": {edge.id: current},
    }
    assert _decide(client, edge.id, decision="reject", rev=current).status_code == 200


def test_a_reinforcement_after_the_read_does_not_make_the_decision_stale(
    client: TestClient, app: FastAPI
) -> None:
    """The case a page meets: a person reads the node, a recall the agent
    expands reinforces the edge, the person decides with the rev read."""
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    rev = _read_rev(client, karim.id, edge.id)
    assert store.reinforce_edge(edge.id) is not None
    assert store.get_edge(edge.id).rev > edge.rev  # the full revision moved
    response = _decide(client, edge.id, decision="confirm", rev=rev)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "confirmed"


def test_a_decay_that_leaves_the_edge_open_does_not_make_the_decision_stale(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    rev = _read_rev(client, karim.id, edge.id)
    clock.advance(days=10)
    assert store.decay_edges(half_life_days=30, prune_below=0.01)["decayed"] == 1
    assert store.get_edge(edge.id).weight < edge.weight
    assert _decide(client, edge.id, decision="confirm", rev=rev).status_code == 200


def test_a_decay_that_closes_the_edge_moves_its_rev_and_the_decision_is_not_live(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    rev = _read_rev(client, karim.id, edge.id)
    clock.advance(days=400)
    assert store.decay_edges(half_life_days=30, prune_below=0.01)["pruned"] == 1
    assert store.get_edge(edge.id).decision_rev > rev
    refused = _decide(client, edge.id, decision="confirm")
    assert refused.json()["detail"]["reason"] == "not_live"


def test_undoing_a_decision_after_a_reinforcement_keeps_the_reinforcement(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    correction = _decide(client, edge.id, decision="confirm").json()["correction_id"]
    reinforced = store.reinforce_edge(edge.id)
    assert reinforced is not None and reinforced.weight > edge.weight
    undone = client.post(f"/v1/corrections/{correction}/undo", headers=auth("alice"))
    assert undone.status_code == 200, undone.text
    after = store.get_edge(edge.id)
    assert after.status == "proposed"
    assert (after.weight, after.use_count) == (reinforced.weight, reinforced.use_count)


def test_an_undo_after_a_later_change_of_the_edges_life_is_refused(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    correction = _decide(client, edge.id, decision="confirm").json()["correction_id"]
    store.close_edge(edge.id)
    refused = client.post(f"/v1/corrections/{correction}/undo")
    assert refused.status_code == 409
    assert refused.json()["detail"] == {"reason": "depends", "records": [edge.id]}


# --------------------------------------------------------------------------- link confirms, never twins


def test_a_link_on_a_proposed_edge_confirms_it_rather_than_adding_a_twin(
    client: TestClient, app: FastAPI
) -> None:
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    first = _link(client, karim.id, lyon.id, "lives_in")
    assert first.status_code == 200, first.text
    body = first.json()
    assert (body["id"], body["status"], body["created"]) == (edge.id, "confirmed", False)
    assert body["rev"] == _read_rev(client, karim.id, edge.id)
    assert body["correction_id"].startswith("corr_")
    again = _link(client, karim.id, lyon.id, "lives_in")
    assert again.status_code == 200
    assert (again.json()["id"], again.json()["correction_id"]) == (edge.id, None)
    assert [(e["id"], e["status"]) for e in _edges_of(client, karim.id)] == [(edge.id, "confirmed")]


def test_a_link_of_another_type_or_direction_is_a_new_edge(
    client: TestClient, app: FastAPI
) -> None:
    karim, lyon = _nodes(app)
    _proposed(app, karim, lyon)
    other_type = _link(client, karim.id, lyon.id, "works_in")
    reverse = _link(client, lyon.id, karim.id, "lives_in")
    assert (other_type.status_code, reverse.status_code) == (201, 201)
    assert other_type.json()["created"] is True
    created = other_type.json()
    assert created["rev"] == _read_rev(client, karim.id, created["id"])
    assert len(_edges_of(client, karim.id)) == 3


def test_a_co_occurrence_is_the_same_edge_either_way(client: TestClient, app: FastAPI) -> None:
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon, "co_occurs")
    response = _link(client, lyon.id, karim.id, "co_occurs")
    assert response.status_code == 200, response.text
    assert response.json()["id"] == edge.id


def test_a_link_never_confirms_an_edge_the_token_does_not_touch(
    client: TestClient, app: FastAPI
) -> None:
    """A token bound below the edge's scope sees it but may not change it: its
    link is its own edge, in its own scope."""
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    response = client.post(
        "/v1/link",
        json={"scope": SCOPE + "/agent:coder", "src": karim.id, "dst": lyon.id, "type": "lives_in"},
        headers=auth("alice-coder"),
    )
    assert response.status_code == 201, response.text
    assert response.json()["id"] != edge.id
    assert app.state.store.get_edge(edge.id).status == "proposed"


# --------------------------------------------------------------------------- the archive says what it holds


def test_archive_items_say_what_each_record_was(client: TestClient, app: FastAPI) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    fact = store.add_fact(
        scopes=[SCOPE], node_id=karim.id, statement="Karim plays the oud.", origin="explicit"
    )
    store.archive(fact.id)
    store.archive(lyon.id)
    items = client.get("/v1/archive", params={"scope": SCOPE, "budget_tokens": BIG}).json()["items"]
    by_kind = {i["kind"]: i for i in items}
    assert {k: v for k, v in by_kind["fact"].items() if k != "archived_at"} == {
        "id": fact.id,
        "kind": "fact",
        "statement": "Karim plays the oud.",
        "node_id": karim.id,
        "node_name": "Karim",
    }
    assert {k: v for k, v in by_kind["node"].items() if k != "archived_at"} == {
        "id": lyon.id,
        "kind": "node",
        "name": "Lyon",
        "node_kind": "place",
        "merged_into": None,
    }


def test_the_archive_page_still_obeys_its_budget(client: TestClient, app: FastAPI) -> None:
    store: Store = app.state.store
    karim, _ = _nodes(app)
    for i in range(3):
        fact = store.add_fact(
            scopes=[SCOPE], node_id=karim.id, statement=f"Karim fact {i} " * 20, origin="explicit"
        )
        store.archive(fact.id)
    params = {"scope": SCOPE, "limit": 1, "budget_tokens": BIG}
    one = client.get("/v1/archive", params=params).json()["used_tokens"]
    page = client.get("/v1/archive", params={"scope": SCOPE, "budget_tokens": one + 5}).json()
    assert page["used_tokens"] <= one + 5
    assert len(page["items"]) == 1 and page["next_cursor"] is not None
    assert page["omitted"] == 2


# --------------------------------------------------------------------------- the dreams of a scope


@pytest.fixture
def paused_app(tmp_path: Path, clock: FakeClock) -> Iterator[FastAPI]:
    """No background worker runs the dreams: a run stays as it was recorded."""
    application = create_app(make_settings(tmp_path, background_mode="paused"), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def paused(paused_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(paused_app, headers=auth("all")) as test_client:
        yield test_client


def _start(client: TestClient, token: str, scope: str = SCOPE, **body: Any) -> str:
    response = client.post("/v1/dream", json={"scope": scope, **body}, headers=auth(token))
    assert response.status_code == 202, response.text
    run_id: str = response.json()["run_id"]
    return run_id


def _dreams(client: TestClient, token: str = "dream", **params: Any) -> Any:
    return client.get(
        "/v1/dreams", params={"scope": SCOPE, "budget_tokens": BIG, **params}, headers=auth(token)
    )


def test_every_client_lists_the_dreams_of_a_scope(paused: TestClient, clock: FakeClock) -> None:
    first = _start(paused, "alice-dream", dry_run=True)
    assert paused.post(f"/v1/dream/{first}/cancel", headers=auth("alice-dream")).status_code == 200
    clock.advance(minutes=1)
    second = _start(paused, "dream", scope=SCOPE + "/agent:coder")
    for token in ("alice-dream", "dream"):
        response = _dreams(paused, token)
        assert response.status_code == 200, response.text
        items = response.json()["items"]
        assert [(i["id"], i["started_by"], i["status"], i["dry_run"]) for i in items] == [
            (second, "dream", "pending", False),
            (first, "alice-dream", "cancelled", True),
        ]
        assert items[0]["scope"] == SCOPE + "/agent:coder"
    run = paused.get(f"/v1/dream/{first}", headers=auth("alice-dream")).json()
    assert run["started_by"] == "alice-dream"


def test_the_dream_list_filters_by_status_and_pages_by_cursor(
    paused: TestClient, clock: FakeClock
) -> None:
    started = []
    for _ in range(3):
        run_id = _start(paused, "dream")
        started.append(run_id)
        paused.post(f"/v1/dream/{run_id}/cancel")
        clock.advance(minutes=1)
    last = _start(paused, "dream")
    pending = _dreams(paused, status="pending").json()["items"]
    assert [i["id"] for i in pending] == [last]
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        body = _dreams(paused, status="cancelled", limit=2, cursor=cursor).json()
        seen.extend(i["id"] for i in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert seen == list(reversed(started))  # newest first, each once
    assert _dreams(paused, status="sleeping").status_code == 422
    assert _dreams(paused, scope="not a scope").status_code == 422


def test_the_list_offers_every_run_status() -> None:
    statuses = {
        journal.PENDING,
        journal.RUNNING,
        journal.FINISHED,
        journal.CANCELLED,
        journal.ROLLED_BACK,
        journal.FAILED,
    }
    assert set(get_args(RunStatus)) == statuses


def test_the_store_refuses_an_unknown_edge_decision_or_status(app: FastAPI) -> None:
    store: Store = app.state.store
    karim, lyon = _nodes(app)
    edge = _proposed(app, karim, lyon)
    with pytest.raises(ValueError, match="unknown edge decision"):
        Corrections(store).decide_edge(edge.id, "maybe", token="alice")
    with pytest.raises(ValueError, match="unknown edge status"):
        store.set_edge_status(edge.id, "doubtful")
    assert store.get_edge(edge.id).status == "proposed"
