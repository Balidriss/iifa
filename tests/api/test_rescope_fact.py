"""Re-scoping a fact (SPEC 4.6): share it with a parent scope or a chosen few."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.api.conftest import remember

ARTIST = "user:alice/agent:artist"
CODER = "user:alice/agent:coder"
WRITER = "user:alice/agent:writer"


def _recalled(client: TestClient, scope: str, query: str) -> list[str]:
    body = client.post("/v1/recall", json={"scope": scope, "query": query}).json()
    return [item["id"] for item in body["items"]]


def _patch(client: TestClient, fact_id: str, **body: Any) -> Any:
    return client.patch(f"/v1/facts/{fact_id}", json=body)


def test_a_fact_shared_with_the_parent_scope_is_recalled_from_every_child(
    client: TestClient,
) -> None:
    fact = remember(client, "Prefers cold palettes in illustrations", scope=ARTIST)["fact"]
    assert _recalled(client, CODER, "cold palettes") == []

    patched = _patch(client, fact["id"], scopes=["user:alice"])
    assert patched.status_code == 200, patched.text
    assert patched.json()["scopes"] == ["user:alice"]

    for scope in (ARTIST, CODER, WRITER, "user:alice"):
        assert _recalled(client, scope, "cold palettes") == [fact["id"]], scope


def test_a_fact_shared_with_a_chosen_few_stays_out_of_the_others(client: TestClient) -> None:
    fact = remember(client, "Prefers cold palettes in illustrations", scope=ARTIST)["fact"]
    patched = _patch(client, fact["id"], scopes=[ARTIST, CODER, CODER]).json()
    assert patched["scopes"] == [ARTIST, CODER]
    assert _recalled(client, CODER, "cold palettes") == [fact["id"]]
    assert _recalled(client, WRITER, "cold palettes") == []
    assert _recalled(client, "user:alice", "cold palettes") == []

    narrowed = _patch(client, fact["id"], scopes=[CODER]).json()
    assert narrowed["scopes"] == [CODER]
    assert _recalled(client, ARTIST, "cold palettes") == []


def test_a_rescope_that_would_hide_the_fact_s_node_is_refused(client: TestClient) -> None:
    """The fact stays where its node is: a new set that no longer sees the node is
    refused unless the patch detaches the fact or names a node it sees."""
    written = remember(
        client,
        "Mira paints with cold palettes",
        scope=ARTIST,
        node={"name": "Mira", "kind": "person"},
    )
    fact, mira = written["fact"], written["node"]

    refused = _patch(client, fact["id"], scopes=["user:alice"])
    assert refused.status_code == 409 and "detach" in refused.text
    assert _patch(client, fact["id"], scopes=[CODER]).status_code == 409
    assert client.app.state.store.scopes_of(fact["id"]) == (ARTIST,)

    kept = _patch(client, fact["id"], scopes=[ARTIST, CODER]).json()
    assert kept["node_id"] == mira["id"] and kept["scopes"] == [ARTIST, CODER]

    moved = _patch(client, fact["id"], scopes=["user:alice"], detach=True).json()
    assert moved["node_id"] is None and moved["scopes"] == ["user:alice"]


def test_a_rescoped_fact_shows_no_excerpt_the_reader_cannot_see(client: TestClient) -> None:
    fact = remember(client, "Prefers cold palettes in illustrations", scope=ARTIST)["fact"]
    _patch(client, fact["id"], scopes=["user:alice"])

    def expanded(scope: str) -> dict[str, Any]:
        body = client.post(
            "/v1/expand", json={"scope": scope, "ids": [fact["id"]], "level": 2}
        ).json()
        item: dict[str, Any] = body["items"][0]
        return item

    seen_from_coder = expanded(CODER)
    assert seen_from_coder["statement"] == "Prefers cold palettes in illustrations"
    assert seen_from_coder["excerpt"] is None  # its source stays in the artist scope
    assert expanded(ARTIST)["excerpt"] == "Prefers cold palettes in illustrations"


def test_rescope_is_validated(client: TestClient) -> None:
    fact = remember(client, "Prefers cold palettes", scope=ARTIST)["fact"]
    assert _patch(client, fact["id"], scopes=[]).status_code == 422
    malformed = _patch(client, fact["id"], scopes=["user:alice/artist"])
    assert malformed.status_code == 422 and "mode:artist" in malformed.text

    mira = remember(client, "Mira paints", scope=ARTIST, node={"name": "Mira"})["node"]
    # with both, the node must be visible from the NEW set
    assert _patch(client, fact["id"], scopes=[CODER], node_id=mira["id"]).status_code == 404
    both = _patch(client, fact["id"], scopes=["user:alice"], node_id=mira["id"])
    assert both.status_code == 404
    attached = _patch(client, fact["id"], scopes=[ARTIST, CODER], node_id=mira["id"]).json()
    assert attached["node_id"] == mira["id"] and attached["scopes"] == [ARTIST, CODER]


def test_rescope_bumps_the_revision(app: FastAPI, client: TestClient) -> None:
    """A dream that read the fact before the re-scope sees a newer revision and
    applies nothing on it (SPEC 4.4, 5.6)."""
    store = app.state.store
    fact = remember(client, "Prefers cold palettes", scope=ARTIST)["fact"]
    rev = store.get_fact(fact["id"]).rev
    _patch(client, fact["id"], scopes=["user:alice"])
    assert store.get_fact(fact["id"]).rev > rev
    assert store.scopes_of(fact["id"]) == ("user:alice",)
