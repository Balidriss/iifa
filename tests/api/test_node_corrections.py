"""A person corrects a node - renames it, adds and removes aliases,
changes its kind - through ``PATCH /v1/nodes/{id}``, journalled with the token's
name and undoable exactly."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.recall.tokenizer import normalize
from iifa.scopes import Visibility
from iifa.store.models import Node
from iifa.store.repo import Store
from tests.helpers import auth

SCOPE = "user:alice"
VIEW = Visibility.of_scope(SCOPE)


def _node(app: FastAPI, name: str, aliases: tuple[str, ...] = ()) -> Node:
    store: Store = app.state.store
    return store.create_node(scopes=[SCOPE], name=name, kind="place", aliases=aliases)


def _unruled_node(app: FastAPI, name: str, aliases: tuple[str, ...] = ()) -> Node:
    """A node holding a name and aliases the name rule refuses,
    written straight into the tables, as an older store may hold them."""
    store: Store = app.state.store
    node = store.create_node(scopes=[SCOPE], name="Placeholder", kind="place")
    with store.lock:
        store.execute(
            "UPDATE nodes SET name = ?, name_norm = ? WHERE id = ?",
            (name, normalize(name), node.id),
        )
        for alias in aliases:
            store.execute(
                "INSERT INTO node_aliases (node_id, alias, alias_norm) VALUES (?, ?, ?)",
                (node.id, alias, normalize(alias)),
            )
    return store.get_node(node.id)


def _patch(client: TestClient, node_id: str, token: str = "alice", **body: Any) -> Any:
    return client.patch(f"/v1/nodes/{node_id}", json=body, headers=auth(token))


def test_a_rename_keeps_the_old_name_as_an_alias(client: TestClient, app: FastAPI) -> None:
    node = _unruled_node(app, "Atelier Lumen Press", ("rendez-vous du **mardi 3 mars",))
    response = _patch(
        client,
        node.id,
        name="Atelier Lumen",
        remove_aliases=["rendez-vous du **mardi 3 mars"],
        add_aliases=["Lumen Press"],
    )
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["name"] == "Atelier Lumen"
    assert sorted(out["aliases"]) == ["Atelier Lumen Press", "Lumen Press"]
    assert out["correction_id"].startswith("corr_")
    store: Store = app.state.store
    assert store.find_node("Atelier Lumen", VIEW) is not None
    assert store.lookup_name("Atelier Lumen Press", VIEW)[0]["id"] == node.id
    assert store.lookup_name("rendez-vous du mardi 3 mars", VIEW) == []
    assert out["rev"] == store.get_node(node.id).rev


def test_a_rename_does_not_keep_an_old_name_the_rule_refuses(
    client: TestClient, app: FastAPI
) -> None:
    node = _unruled_node(app, "Atelier Lumen (Lumen Press")
    response = _patch(client, node.id, name="Atelier Lumen")
    assert response.status_code == 200, response.text
    assert response.json()["aliases"] == []
    assert app.state.store.lookup_name("Atelier Lumen (Lumen Press", VIEW) == []


def test_keep_old_name_false_drops_the_old_name(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen Press")
    response = _patch(client, node.id, name="Atelier Lumen", keep_old_name=False)
    assert response.status_code == 200, response.text
    assert response.json()["aliases"] == []
    assert app.state.store.lookup_name("Atelier Lumen Press", VIEW) == []


def test_an_alias_spelled_like_the_new_name_is_dropped(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Report spring audit", ("the spring audit",))
    response = _patch(client, node.id, name="The spring audit", keep_old_name=False)
    assert response.status_code == 200, response.text
    assert (response.json()["name"], response.json()["aliases"]) == ("The spring audit", [])


def test_the_kind_changes(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    response = _patch(client, node.id, kind="organisation")
    assert (response.status_code, response.json()["kind"]) == (200, "organisation")
    assert app.state.store.get_node(node.id).kind == "organisation"


def test_a_patch_that_changes_nothing_writes_nothing(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen", ("Lumen",))
    rev = app.state.store.get_node(node.id).rev
    response = _patch(client, node.id, name="Atelier Lumen", add_aliases=["lumen"], kind="place")
    assert response.status_code == 200, response.text
    assert response.json()["correction_id"] is None
    assert app.state.store.get_node(node.id).rev == rev


def test_a_patch_must_ask_for_something(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    assert _patch(client, node.id).status_code == 422
    assert _patch(client, node.id, keep_old_name=False).status_code == 422


def test_removing_an_alias_the_node_lacks_is_refused(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen", ("Lumen",))
    response = _patch(client, node.id, remove_aliases=["Press"])
    assert response.status_code == 422
    assert response.json()["detail"] == {"reason": "no_alias", "records": ["Press"]}
    assert app.state.store.get_node(node.id).aliases == ("Lumen",)


def test_the_one_name_rule_refuses_an_empty_name(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    assert _patch(client, node.id, name="   ").status_code == 422
    assert _patch(client, node.id, add_aliases=[" \n "]).status_code == 422
    assert app.state.store.get_node(node.id).name == "Atelier Lumen"


def test_a_stale_revision_refuses_the_patch(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    seen = client.get(f"/v1/nodes/{node.id}", params={"scope": SCOPE}).json()["rev"]
    assert _patch(client, node.id, name="Lumen", rev=seen).status_code == 200
    response = _patch(client, node.id, name="Atelier Lumen", rev=seen)  # someone else's view
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["reason"] == "stale"
    assert detail["revs"] == {node.id: app.state.store.get_node(node.id).rev}
    assert app.state.store.get_node(node.id).name == "Lumen"


def test_a_patch_needs_write_and_a_token_that_touches_the_node(
    client: TestClient, app: FastAPI
) -> None:
    node = _node(app, "Atelier Lumen")
    assert _patch(client, node.id, "read", name="Lumen").status_code == 403
    assert _patch(client, node.id, "bob", name="Lumen").status_code == 403
    assert _patch(client, "node_" + "0" * 26, name="Lumen").status_code == 404
    assert _patch(client, "nope", name="Lumen").status_code == 404
    assert app.state.store.get_node(node.id).name == "Atelier Lumen"


def test_the_journal_says_who_and_when_and_undo_restores_exactly(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    node = _unruled_node(app, "Atelier Lumen (Lumen Press", ("rendez-vous du **mardi", "Lumen"))
    before = store.get_node(node.id)
    out = _patch(
        client,
        node.id,
        name="Atelier Lumen",
        kind="organisation",
        keep_old_name=False,
        remove_aliases=["rendez-vous du **mardi"],
        add_aliases=["Press"],
    ).json()
    entry = client.get(f"/v1/corrections/{out['correction_id']}", headers=auth("alice")).json()
    assert (entry["op"], entry["status"], entry["token"]) == ("update_node", "applied", "alice")
    assert entry["created_at"] and entry["records"] == [node.id]
    assert entry["payload"]["name"] == "Atelier Lumen"
    undone = client.post(
        f"/v1/corrections/{out['correction_id']}/undo", headers=auth("alice-review")
    )
    assert undone.status_code == 200, undone.text
    assert (undone.json()["status"], undone.json()["undone_by"]) == ("undone", "alice-review")
    after = store.get_node(node.id)
    assert (after.name, after.kind, after.aliases) == (before.name, before.kind, before.aliases)
    assert store.lookup_name("Atelier Lumen (Lumen Press", VIEW)[0]["id"] == node.id
    again = client.post(f"/v1/corrections/{out['correction_id']}/undo")
    assert (again.status_code, again.json()["detail"]["reason"]) == (409, "not_applied")


def test_undo_waits_for_the_later_correction(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    first = _patch(client, node.id, name="Lumen").json()["correction_id"]
    second = _patch(client, node.id, add_aliases=["Press"]).json()["correction_id"]
    blocked = client.post(f"/v1/corrections/{first}/undo")
    assert blocked.status_code == 409
    assert blocked.json()["detail"] == {"reason": "depends", "records": [node.id]}
    assert client.post(f"/v1/corrections/{second}/undo").status_code == 200
    assert client.post(f"/v1/corrections/{first}/undo").status_code == 200
    assert app.state.store.get_node(node.id).name == "Atelier Lumen"


def test_a_correction_is_not_seen_from_another_scope(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    correction = _patch(client, node.id, name="Lumen").json()["correction_id"]
    assert client.get(f"/v1/corrections/{correction}", headers=auth("bob")).status_code == 404
    undo = client.post(f"/v1/corrections/{correction}/undo", headers=auth("bob"))
    assert undo.status_code == 404
    for missing in ("corr_" + "0" * 26, "nope"):
        assert client.get(f"/v1/corrections/{missing}").status_code == 404


def test_undoing_a_patch_needs_write(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    correction = _patch(client, node.id, name="Lumen").json()["correction_id"]
    for token in ("read", "review", "dream"):
        undo = client.post(f"/v1/corrections/{correction}/undo", headers=auth(token))
        assert undo.status_code == 403, token
    assert app.state.store.get_node(node.id).name == "Lumen"


def test_an_archived_node_is_restored_before_it_is_corrected(
    client: TestClient, app: FastAPI
) -> None:
    node = _node(app, "Atelier Lumen")
    app.state.store.archive(node.id)
    response = _patch(client, node.id, name="Lumen")
    assert (response.status_code, response.json()["detail"]["reason"]) == (409, "not_live")
    assert app.state.store.get_node(node.id).name == "Atelier Lumen"


def test_an_unrelated_patch_keeps_an_alias_spelled_like_the_name(
    client: TestClient, app: FastAPI
) -> None:
    node = _node(app, "Atelier Lumen", ("café lumen",))
    response = _patch(client, node.id, add_aliases=["Lumen"])
    assert response.status_code == 200, response.text
    assert sorted(response.json()["aliases"]) == ["Lumen", "café lumen"]


def test_remember_refuses_a_name_the_rule_refuses(client: TestClient, app: FastAPI) -> None:
    for node in ({"name": "   "}, {"name": "Atelier Lumen", "aliases": ["x" * 301]}):
        body = {"scope": SCOPE, "statement": "It opens at nine.", "node": node}
        assert client.post("/v1/remember", json=body).status_code == 422, node
    assert client.get("/v1/stats").json()["counts"]["facts"] == 0


def test_the_node_view_shows_its_revision(client: TestClient, app: FastAPI) -> None:
    node = _node(app, "Atelier Lumen")
    view = client.get(f"/v1/nodes/{node.id}", params={"scope": SCOPE}).json()
    assert view["rev"] == app.state.store.get_node(node.id).rev
