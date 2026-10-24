"""Fact proposals in the store (SPEC 4.4b): a fact about the owner the write
path would not write, kept with the rest of the fact, possibly without a node;
accepting one writes an explicit fact that keeps its provenance. Schema 8 keeps
every proposal a version-7 file held."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from iifa.scopes import Visibility
from iifa.store.connection import open_database
from iifa.store.repo import ProposalDecided, Store
from tests.fixtures.make_fixture_db import FIXTURE_DIR, FIXTURE_KEY

SCOPE = "user:alice"
PAYLOAD = {
    "attribute": "home",
    "value": "Moscow",
    "valid_from": "2026-03",
    "language": "en",
    "flags": ["speaker"],
    "provenance": "external",
    "confidence": 0.8,
}


def _proposal(
    store: Store, *, node_id: str | None = None, statement: str = "The user lives in Moscow."
) -> str:
    source = store.add_source(scopes=[SCOPE], kind="turn", text=f"Web: {statement}")
    return store.add_proposal(
        scopes=[SCOPE],
        kind="fact",
        node_id=node_id,
        value=statement,
        source_id=source.id,
        excerpt=(5, 5 + len(statement)),
        confidence=0.8,
        payload=PAYLOAD,
    ).id


def test_a_fact_proposal_keeps_the_fact_and_may_have_no_node(store: Store) -> None:
    proposal = store.get_proposal(_proposal(store))
    assert proposal.kind == "fact" and proposal.node_id is None
    assert proposal.value == "The user lives in Moscow." and proposal.payload == PAYLOAD
    again = _proposal(store)
    assert again == proposal.id  # the same pending proposal is recorded once
    listed = store.pending_proposals(Visibility.of_scope(SCOPE), kind="fact")
    assert [p.id for p in listed] == [proposal.id]
    assert store.pending_proposals(Visibility.of_scope(SCOPE), kind="alias") == []
    assert store.pending_proposals(Visibility.of_scope("user:bob")) == []


def test_only_a_fact_proposal_may_go_without_a_node(store: Store) -> None:
    with pytest.raises(ValueError, match="node"):
        store.add_proposal(scopes=[SCOPE], kind="alias", node_id=None, value="Ваня")


def test_accepting_a_fact_proposal_writes_an_explicit_fact_with_its_provenance(
    store: Store,
) -> None:
    owner = store.create_node(scopes=[SCOPE], name="Alice", kind="person")
    proposal_id = _proposal(store, node_id=owner.id)
    accepted = store.decide_proposal(proposal_id, accept=True)
    assert accepted.status == "accepted" and accepted.applied_id is not None
    fact = store.get_fact(accepted.applied_id)
    assert fact.origin == "explicit" and fact.proposal_id == proposal_id
    assert fact.node_id == owner.id and fact.scopes == (SCOPE,)
    assert fact.source_id == accepted.source_id
    assert (fact.excerpt_start, fact.excerpt_end) == (accepted.excerpt_start, accepted.excerpt_end)
    assert (fact.attribute, fact.value, fact.valid_from) == ("home", "Moscow", "2026-03")
    assert fact.provenance == "external" and fact.language == "en"
    assert fact.flags == ("speaker",) and fact.confidence == 0.8
    with pytest.raises(ProposalDecided):
        store.decide_proposal(proposal_id, accept=False)


def test_rejecting_a_fact_proposal_writes_nothing(store: Store) -> None:
    rejected = store.decide_proposal(_proposal(store), accept=False)
    assert rejected.status == "rejected" and rejected.applied_id is None
    assert store.execute("SELECT count(*) FROM facts").fetchone()[0] == 0


def test_schema_8_keeps_the_proposals_of_a_version_7_file(tmp_path: Path) -> None:
    copy = tmp_path / "v7.db"
    shutil.copy(FIXTURE_DIR / "v7.db", copy)
    conn = open_database(copy, FIXTURE_KEY)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 7
    node_id = conn.execute("SELECT id FROM nodes").fetchone()[0]
    conn.execute(
        "INSERT INTO proposals (id, kind, node_id, value, value_norm, status, created_at)"
        " VALUES ('prop_legacy', 'alias', ?, 'Kari', 'kari', 'pending', '2026-01-01T00:00:00Z')",
        (node_id,),
    )
    conn.execute(
        "INSERT INTO record_scopes (record_id, scope) VALUES ('prop_legacy', 'user:fixture')"
    )
    conn.close()
    with Store(copy, FIXTURE_KEY) as store:
        legacy = store.get_proposal("prop_legacy")
        assert (legacy.kind, legacy.node_id, legacy.value) == ("alias", node_id, "Kari")
        assert legacy.payload == {} and legacy.applied_id is None
        assert store.decide_proposal("prop_legacy", accept=True).status == "accepted"
        assert "Kari" in store.get_node(node_id).aliases
