"""SPEC 5.7 guardrails: two nodes are merged only by an explicit decision of a
token holding ``review`` - never by a ``write`` token, a dream token or a model -
never across scope sets, and a person's merge undoes exactly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from iifa.api import auth as api_auth
from iifa.api.app import create_app
from iifa.api.routes import _accept_merge
from iifa.mcp import CORE_TOOLS, FORGET_TOOL
from iifa.scopes import Visibility
from iifa.store.corrections import Corrections
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.dream.conftest import SCOPE, Memory, build_memory
from tests.helpers import auth, make_settings

pytestmark = pytest.mark.security

CANARY = "quokkacanary"
NOT_REVIEW = ("read", "write", "forget", "dream", "admin", "alice", "alice-dream")


def _with_history(store: Store, memory: Memory) -> dict[str, str]:
    """Kiki gets facts (one explicit), an alias, an edge to Lyon and an edge to
    Karim (left joining Karim to himself by a merge, so closed)."""
    explicit = store.add_fact(
        scopes=[SCOPE],
        node_id=memory.kiki.id,
        statement="Kiki is allergic to cats.",
        origin="explicit",
    )
    extracted = store.add_fact(
        scopes=[SCOPE],
        node_id=memory.kiki.id,
        statement="Kiki plays the cello.",
        origin="extracted",
    )
    store.update_node(memory.kiki.id, add_aliases=["Kiks"])
    to_lyon = store.add_edge(
        scopes=[SCOPE], src=memory.kiki.id, dst=memory.lyon.id, type="lives_in", origin="explicit"
    )
    to_karim = store.add_edge(
        scopes=[SCOPE], src=memory.karim.id, dst=memory.kiki.id, type="knows", origin="explicit"
    )
    return {
        "explicit": explicit.id,
        "extracted": extracted.id,
        "to_lyon": to_lyon.id,
        "to_karim": to_karim.id,
    }


def _state(store: Store, memory: Memory, ids: dict[str, str]) -> dict[str, Any]:
    """Everything a merge changes, without the revisions and update times."""
    nodes = {}
    for node_id in (memory.karim.id, memory.kiki.id, memory.lyon.id):
        node = store.get_node(node_id)
        nodes[node_id] = (
            node.name,
            node.kind,
            node.aliases,
            node.archived_at,
            node.merged_into,
            node.scopes,
        )
    facts = {
        f: (store.get_fact(f).node_id, store.get_fact(f).valid_to, store.get_fact(f).statement)
        for f in (*store.node_fact_ids(memory.karim.id), *store.node_fact_ids(memory.kiki.id))
    }
    edges = {}
    for key in ("to_lyon", "to_karim"):
        edge = store.get_edge(ids[key])
        edges[key] = (edge.src, edge.dst, edge.valid_to, edge.weight, edge.status)
    view = Visibility.of_scope(SCOPE)
    proposals = {p.id: p.status for p in store.proposals([memory.karim.id, memory.kiki.id], view)}
    return {"nodes": nodes, "facts": facts, "edges": edges, "proposals": proposals}


def _merge(client: TestClient, keep: str, absorb: str, token: str = "review", **extra: Any) -> Any:
    return client.post(
        "/v1/nodes/merge", json={"keep": keep, "absorb": absorb, **extra}, headers=auth(token)
    )


def test_only_a_review_token_merges_two_nodes(dream_client: TestClient, dream_app: FastAPI) -> None:
    store: Store = dream_app.state.store
    memory = build_memory(store)
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    before = store.revisions([memory.karim.id, memory.kiki.id])
    for token in NOT_REVIEW:
        assert _merge(dream_client, memory.karim.id, memory.kiki.id, token).status_code == 403
        decided = dream_client.post(
            f"/v1/review/{proposal.id}", json={"decision": "accept"}, headers=auth(token)
        )
        assert decided.status_code == 403, token
    # the write route an agent holds never merges, whoever calls it
    for token in ("write", "alice", "alice-review", "all"):
        accepted = dream_client.post(
            f"/v1/proposals/{proposal.id}", json={"decision": "accept"}, headers=auth(token)
        )
        assert accepted.status_code == 409, token
    assert store.revisions([memory.karim.id, memory.kiki.id]) == before
    assert store.get_proposal(proposal.id).status == "pending"
    response = _merge(dream_client, memory.karim.id, memory.kiki.id, "alice-review")
    assert response.status_code == 200, response.text
    assert store.get_node(memory.kiki.id).merged_into == memory.karim.id
    assert store.get_proposal(proposal.id).status == "accepted"  # the pending one, taken


def test_undoing_a_merge_needs_review_too(dream_client: TestClient, dream_app: FastAPI) -> None:
    store: Store = dream_app.state.store
    memory = build_memory(store)
    correction = _merge(dream_client, memory.karim.id, memory.kiki.id).json()["correction_id"]
    for token in ("write", "alice"):
        undo = dream_client.post(f"/v1/corrections/{correction}/undo", headers=auth(token))
        assert undo.status_code == 403, token
    assert store.get_node(memory.kiki.id).merged_into == memory.karim.id


def test_no_model_path_reaches_a_persons_merge(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    """The model's ways in are the MCP tools (an agent token: read + write) and
    the dream (``dream``): none holds ``review``, and the service's own merge
    path checks it again, whatever route called it."""
    assert not any("merge" in t or "correct" in t or "node" in t for t in CORE_TOOLS)
    assert "merge" not in FORGET_TOOL
    store: Store = dream_app.state.store
    memory = build_memory(store)
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    agent = api_auth.Principal("agent", ("*",), frozenset({"read", "write", "dream"}))

    class _Request:
        app = dream_app

    with pytest.raises(HTTPException) as refused:
        _accept_merge(_Request(), agent, proposal, None)  # type: ignore[arg-type]
    assert refused.value.status_code == 403
    assert store.get_node(memory.kiki.id).merged_into is None


def test_a_merge_across_scope_sets_is_refused(dream_client: TestClient, dream_app: FastAPI) -> None:
    store: Store = dream_app.state.store
    karim = store.create_node(scopes=[SCOPE], name="Karim", kind="person")
    shared = store.create_node(scopes=[SCOPE, "team:x"], name="Karim", kind="person")
    other = store.create_node(scopes=["user:bob"], name="Karim", kind="person")
    proposal = store.add_proposal(scopes=[SCOPE], kind="merge", node_id=karim.id, value=shared.id)
    before = store.revisions([karim.id, shared.id, other.id])
    response = _merge(dream_client, karim.id, shared.id)
    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "scope_mismatch"
    review = dream_client.post(
        f"/v1/review/{proposal.id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert (review.status_code, review.json()["detail"]["reason"]) == (409, "scope_mismatch")
    assert _merge(dream_client, karim.id, other.id).status_code == 409
    # a reviewer bound to one side only does not reach the other node
    assert _merge(dream_client, karim.id, other.id, "alice-review").status_code == 403
    assert store.revisions([karim.id, shared.id, other.id]) == before
    assert store.get_proposal(proposal.id).status == "pending"


def test_a_persons_merge_of_two_kinds_keeps_the_kept_kind_and_undoes_exactly(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    """A kind the extraction got wrong is no reason to refuse a merge a person
    decided: the kept node keeps its kind, the absorbed one is archived with
    its own, and undo restores both. The dream's merges still need one kind."""
    store: Store = dream_app.state.store
    memory = build_memory(store)
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.lyon.id
    )
    accepted = dream_client.post(
        f"/v1/review/{proposal.id}",
        json={"decision": "accept", "keep": memory.karim.id},
        headers=auth("review"),
    )
    assert accepted.status_code == 200, accepted.text
    karim, lyon = store.get_node(memory.karim.id), store.get_node(memory.lyon.id)
    assert (karim.kind, lyon.kind, lyon.merged_into) == ("person", "place", karim.id)
    assert "Lyon" in karim.aliases
    undo = dream_client.post(
        f"/v1/corrections/{accepted.json()['correction_id']}/undo", headers=auth("review")
    )
    assert undo.status_code == 200, undo.text
    lyon = store.get_node(memory.lyon.id)
    assert (lyon.kind, lyon.merged_into, lyon.archived_at) == ("place", None, None)
    assert "Lyon" not in store.get_node(memory.karim.id).aliases
    assert store.get_proposal(proposal.id).status == "pending"


def test_a_merge_of_two_kinds_names_the_node_kept(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    """The kept node decides the merged subject's kind: between two kinds the
    reviewer names it, never the fact count."""
    store: Store = dream_app.state.store
    memory = build_memory(store)
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.lyon.id
    )
    refused = dream_client.post(
        f"/v1/review/{proposal.id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["reason"] == "keep_required"
    assert store.get_node(memory.lyon.id).merged_into is None
    assert store.get_proposal(proposal.id).status == "pending"


def test_a_merge_of_one_node_or_of_an_archived_node_is_refused(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    store: Store = dream_app.state.store
    memory = build_memory(store)
    assert _merge(dream_client, memory.karim.id, memory.karim.id).status_code == 422
    store.archive(memory.kiki.id)
    response = _merge(dream_client, memory.karim.id, memory.kiki.id)
    assert (response.status_code, response.json()["detail"]["reason"]) == (409, "not_live")


def test_a_name_the_rule_refuses_stops_the_merge_before_anything_moves(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    """A name stored before the rule would join the kept node's names: the
    merge is refused whole (the same guard serves the dream)."""
    store: Store = dream_app.state.store
    memory = build_memory(store)
    with store.lock:
        store.execute(
            "INSERT INTO node_aliases (node_id, alias, alias_norm) VALUES (?, ?, ?)",
            (memory.kiki.id, "k" * 400, "k" * 400),
        )
    before = store.revisions([memory.karim.id, memory.kiki.id])
    response = _merge(dream_client, memory.karim.id, memory.kiki.id)
    assert (response.status_code, response.json()["detail"]["reason"]) == (409, "invalid_name")
    assert store.revisions([memory.karim.id, memory.kiki.id]) == before


def test_a_persons_merge_moves_everything_and_undo_restores_exactly(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    store: Store = dream_app.state.store
    memory = build_memory(store)
    ids = _with_history(store, memory)
    before = _state(store, memory, ids)
    revs = store.revisions([memory.karim.id, memory.kiki.id])
    response = _merge(dream_client, memory.karim.id, memory.kiki.id, revs=revs)
    assert response.status_code == 200, response.text
    correction = response.json()["correction_id"]
    karim, kiki = store.get_node(memory.karim.id), store.get_node(memory.kiki.id)
    assert (kiki.merged_into, kiki.archived_at is not None) == (karim.id, True)
    assert {"Kiki", "Kiks", "K."} <= set(karim.aliases)
    assert store.get_fact(ids["explicit"]).node_id == karim.id
    assert store.get_fact(ids["explicit"]).origin == "explicit"
    assert store.get_fact(ids["extracted"]).node_id == karim.id
    assert store.get_edge(ids["to_lyon"]).src == karim.id
    assert store.get_edge(ids["to_karim"]).valid_to is not None  # Karim knows Karim: closed
    entry = dream_client.get(f"/v1/corrections/{correction}").json()
    assert (entry["op"], entry["token"], entry["status"]) == ("merge_nodes", "review", "applied")
    undo = dream_client.post(f"/v1/corrections/{correction}/undo", headers=auth("review"))
    assert undo.status_code == 200, undo.text
    after = _state(store, memory, ids)
    assert after == before  # the proposal the direct merge recorded went with it
    assert store.merge_proposal_between(memory.karim.id, memory.kiki.id) is None


def test_an_accepted_proposal_is_pending_again_after_undo(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    store: Store = dream_app.state.store
    memory = build_memory(store)
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.kiki.id, value=memory.karim.id
    )
    accepted = dream_client.post(
        f"/v1/review/{proposal.id}",
        json={"decision": "accept", "keep": memory.kiki.id},
        headers=auth("review"),
    )
    assert accepted.status_code == 200, accepted.text
    assert store.get_node(memory.karim.id).merged_into == memory.kiki.id  # the person chose
    correction = accepted.json()["correction_id"]
    assert dream_client.post(f"/v1/corrections/{correction}/undo").status_code == 200
    assert store.get_proposal(proposal.id).status == "pending"
    assert store.get_node(memory.karim.id).merged_into is None
    bad_keep = dream_client.post(
        f"/v1/review/{proposal.id}",
        json={"decision": "accept", "keep": memory.lyon.id},
        headers=auth("review"),
    )
    assert bad_keep.status_code == 422


def test_a_later_write_blocks_the_undo_of_a_merge(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    store: Store = dream_app.state.store
    memory = build_memory(store)
    correction = _merge(dream_client, memory.karim.id, memory.kiki.id).json()["correction_id"]
    store.update_node(memory.karim.id, add_aliases=["Karimo"])  # a later write
    undo = dream_client.post(f"/v1/corrections/{correction}/undo")
    assert undo.status_code == 409
    assert undo.json()["detail"] == {"reason": "depends", "records": [memory.karim.id]}
    assert store.get_node(memory.kiki.id).merged_into == memory.karim.id


def test_a_stale_revision_refuses_the_merge(dream_client: TestClient, dream_app: FastAPI) -> None:
    store: Store = dream_app.state.store
    memory = build_memory(store)
    revs = store.revisions([memory.karim.id, memory.kiki.id])
    store.update_node(memory.kiki.id, add_aliases=["Kiks"])
    response = _merge(dream_client, memory.karim.id, memory.kiki.id, revs=revs)
    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "stale"
    assert response.json()["detail"]["records"] == [memory.kiki.id]
    assert store.get_node(memory.kiki.id).merged_into is None
    stranger = _merge(dream_client, memory.karim.id, memory.kiki.id, revs={memory.lyon.id: 1})
    assert stranger.status_code == 422


def test_p5_forget_wipes_the_corrections_journal(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "plain.db"
    settings = make_settings(tmp_path, db_path=path, db_key="", plaintext_database_for_tests=True)
    app = create_app(settings, clock=clock)
    try:
        with TestClient(app, headers=auth("all")) as client:
            store: Store = app.state.store
            node = store.create_node(scopes=[SCOPE], name="North garden", kind="place")
            response = client.patch(f"/v1/nodes/{node.id}", json={"add_aliases": [CANARY]})
            correction = response.json()["correction_id"]
            store.checkpoint("TRUNCATE")
            assert CANARY.encode() in _raw_bytes(path)
            removed = client.post("/v1/forget", json={"id": node.id}).json()["removed"]
            assert removed["corrections_wiped"] == 1
            entry = Corrections(store).get(correction)
            assert (entry.forgotten, entry.payload, entry.changes, entry.records) == (
                True,
                None,
                None,
                (),
            )
            assert node.id not in json.dumps(entry.__dict__)
            assert client.get(f"/v1/corrections/{correction}").status_code == 404
    finally:
        app.state.store.close()
    assert CANARY.encode() not in _raw_bytes(path)
    assert CANARY.encode("utf-16-le") not in _raw_bytes(path)


def _raw_bytes(path: Path) -> bytes:
    data = b""
    for candidate in (path, path.with_name(path.name + "-wal")):
        if candidate.exists():
            data += candidate.read_bytes()
    return data
