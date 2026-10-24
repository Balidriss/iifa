"""Proposals (SPEC 4.4b) are records like the others: they carry text from a
source, so they are read through the view (P9), decided only by a token bound
to one of their scopes, and hard-deleted with their node or their source (P5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iifa.scopes import Visibility
from iifa.store.repo import ProposalDecided, Store
from tests.api.conftest import auth
from tests.fixtures.corpus import CANARY
from tests.security.test_p5_encryption_archive_forget import _raw_bytes

pytestmark = pytest.mark.security

CODER = "user:alice/agent:coder"
OTHER = "user:alice/agent:other"


def _node_and_proposals(store: Store) -> tuple[str, str, str]:
    """A node every agent of Alice sees, and one alias proposal from each agent's source."""
    node = store.create_node(scopes=["user:alice"], name="Иван", kind="person")
    ids = []
    for scope, value in ((CODER, "Ваня"), (OTHER, "Карим")):
        source = store.add_source(scopes=[scope], kind="turn", text=f"{value} руководит проектом.")
        proposal = store.add_proposal(
            scopes=[scope],
            kind="alias",
            node_id=node.id,
            value=value,
            source_id=source.id,
            excerpt=(0, len(value)),
        )
        ids.append(proposal.id)
    return node.id, ids[0], ids[1]


def test_p9_a_proposal_is_seen_only_through_its_scopes(store: Store) -> None:
    node_id, coder, other = _node_and_proposals(store)
    assert [p.id for p in store.proposals([node_id], Visibility.of_scope(CODER))] == [coder]
    assert [p.id for p in store.proposals([node_id], Visibility.of_scope(OTHER))] == [other]
    # the node is visible from user:alice, the proposals from its agents are not
    assert store.proposals([node_id], Visibility.of_scope("user:alice")) == []
    assert store.proposals([node_id], Visibility.of_scope("user:bob")) == []
    both = Visibility.of_subtree("user:alice")
    assert {p.id for p in store.proposals([node_id], both)} == {coder, other}


def test_p9_the_node_view_lists_only_visible_pending_proposals(client: TestClient) -> None:
    store: Store = client.app.state.store  # type: ignore[attr-defined]
    node_id, coder, other = _node_and_proposals(store)
    seen = client.get(
        f"/v1/nodes/{node_id}", params={"scope": CODER}, headers=auth("alice-coder")
    ).json()
    assert [p["id"] for p in seen["proposals"]] == [coder]
    assert seen["proposals"][0]["value"] == "Ваня" and seen["proposals"][0]["kind"] == "alias"
    assert "Карим" not in str(seen)
    from_alice = client.get(
        f"/v1/nodes/{node_id}", params={"scope": "user:alice"}, headers=auth("alice")
    ).json()
    assert from_alice["proposals"] == []
    _ = other


def test_p9_a_proposal_is_decided_only_by_a_token_bound_to_its_scopes(
    client: TestClient,
) -> None:
    store: Store = client.app.state.store  # type: ignore[attr-defined]
    node_id, coder, other = _node_and_proposals(store)
    refused = client.post(
        f"/v1/proposals/{other}", json={"decision": "accept"}, headers=auth("alice-coder")
    )
    assert refused.status_code == 403
    assert refused.json()["detail"] == "record outside the token's scopes"
    read_only = client.post(
        f"/v1/proposals/{coder}", json={"decision": "accept"}, headers=auth("read")
    )
    assert read_only.status_code == 403
    assert store.get_node(node_id).aliases == ()
    assert store.get_proposal(other).status == "pending"
    accepted = client.post(
        f"/v1/proposals/{coder}", json={"decision": "accept"}, headers=auth("alice-coder")
    )
    assert accepted.status_code == 200, accepted.text
    assert store.get_node(node_id).aliases == ("Ваня",)


def test_p9_the_decision_route_does_not_tell_one_record_from_another(client: TestClient) -> None:
    for record in ("prop_" + "0" * 26, "node_" + "0" * 26, "nonsense"):
        response = client.post(f"/v1/proposals/{record}", json={"decision": "reject"})
        assert response.status_code == 404


def test_p5_forgetting_the_node_or_the_source_deletes_its_proposals(tmp_path: Path) -> None:
    path = tmp_path / "plain.db"
    with Store(path, "", plaintext_for_tests=True) as store:
        node = store.create_node(scopes=["user:alice"], name="Иван", kind="person")
        kept = store.create_node(scopes=["user:alice"], name="Emilie", kind="person")
        source = store.add_source(scopes=["user:alice"], kind="turn", text="Иван руководит.")
        other = store.add_source(scopes=["user:alice"], kind="turn", text="Emilie aime le thé.")
        store.add_proposal(
            scopes=["user:alice"],
            kind="alias",
            node_id=node.id,
            value=f"alias {CANARY}",
            source_id=source.id,
        )
        by_source = store.add_proposal(
            scopes=["user:alice"],
            kind="alias",
            node_id=kept.id,
            value=f"second {CANARY[::-1]}",
            source_id=other.id,
        )
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert CANARY.encode() in _raw_bytes(path)
        removed = store.forget(node.id)
        assert removed["proposals"] == 1
        assert store.proposals([node.id], Visibility.of_subtree("user:alice")) == []
        # forgetting a source takes the proposals it grounded with it
        removed = store.forget(other.id)
        assert removed["proposals"] == 1
        with pytest.raises(LookupError):
            store.get_proposal(by_source.id)
        assert store.get_node(kept.id).aliases == ()
    raw = _raw_bytes(path)
    assert CANARY.encode() not in raw and CANARY[::-1].encode() not in raw


def test_a_proposal_is_decided_once(store: Store) -> None:
    node_id, coder, other = _node_and_proposals(store)
    accepted = store.decide_proposal(coder, accept=True)
    assert accepted.status == "accepted" and accepted.decided_at is not None
    assert store.get_node(node_id).aliases == ("Ваня",)
    rejected = store.decide_proposal(other, accept=False)
    assert rejected.status == "rejected"
    assert store.get_node(node_id).aliases == ("Ваня",)
    with pytest.raises(ProposalDecided):
        store.decide_proposal(other, accept=True)
    # decided proposals leave the pending list, and resolution never used them
    assert store.proposals([node_id], Visibility.of_scope(OTHER)) == []
    assert store.find_node("Карим", Visibility.of_scope(OTHER)) is None


def test_the_same_pending_proposal_is_not_recorded_twice(store: Store) -> None:
    node = store.create_node(scopes=["user:alice"], name="Иван", kind="person")
    first = store.add_proposal(scopes=["user:alice"], kind="alias", node_id=node.id, value="Карим")
    again = store.add_proposal(scopes=["user:alice"], kind="alias", node_id=node.id, value="карим")
    assert again.id == first.id
    assert len(store.proposals([node.id], Visibility.of_scope("user:alice"))) == 1
