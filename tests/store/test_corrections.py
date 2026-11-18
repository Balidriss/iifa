"""The corrections journal at the store level (SPEC 5.7): the kind rule, the
default node to keep, and undo when forget took part of what a correction
touched."""

from __future__ import annotations

import pytest

from iifa.store.corrections import CorrectionRefused, Corrections, default_keep
from iifa.store.repo import Store

SCOPE = "user:alice"


def test_a_kind_cannot_be_blank(store: Store) -> None:
    node = store.create_node(scopes=[SCOPE], name="North garden", kind="place")
    with pytest.raises(CorrectionRefused) as refused:
        Corrections(store).patch_node(node.id, token="t", kind="   ")
    assert refused.value.reason == "no_kind"
    assert store.get_node(node.id).kind == "place"


def test_the_node_with_more_live_facts_is_kept_else_the_older(store: Store) -> None:
    older = store.create_node(scopes=[SCOPE], name="Nadia", kind="person")
    newer = store.create_node(scopes=[SCOPE], name="Nadia B.", kind="person")
    keep, absorb = default_keep(store, newer, older)
    assert (keep.id, absorb.id) == (older.id, newer.id)
    store.add_fact(scopes=[SCOPE], node_id=newer.id, statement="Nadia B. sings.", origin="explicit")
    keep, absorb = default_keep(store, older, newer)
    assert (keep.id, absorb.id) == (newer.id, older.id)


def test_an_entry_wiped_by_forget_cannot_be_undone(store: Store) -> None:
    keep = store.create_node(scopes=[SCOPE], name="Nadia", kind="person")
    absorb = store.create_node(scopes=[SCOPE], name="Nadia B.", kind="person")
    fact = store.add_fact(
        scopes=[SCOPE], node_id=absorb.id, statement="Nadia B. sings.", origin="explicit"
    )
    corrections = Corrections(store)
    entry = corrections.merge(keep.id, absorb.id, token="t")
    assert store.forget(fact.id)["corrections_wiped"] == 1
    with pytest.raises(CorrectionRefused) as refused:
        corrections.undo(entry.id, token="t")
    assert refused.value.reason == "forgotten"
    assert store.get_node(absorb.id).merged_into == keep.id


def test_undo_goes_on_when_the_accepted_proposal_left_with_its_source(store: Store) -> None:
    keep = store.create_node(scopes=[SCOPE], name="Nadia", kind="person")
    absorb = store.create_node(scopes=[SCOPE], name="Nadia B.", kind="person")
    source = store.add_source(scopes=[SCOPE], kind="turn", text="Nadia and Nadia B.")
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=keep.id, value=absorb.id, source_id=source.id
    )
    corrections = Corrections(store)
    entry = corrections.merge(keep.id, absorb.id, token="t", proposal=proposal)
    store.forget(source.id)  # the proposal goes with its source (P5)
    undone = corrections.undo(entry.id, token="u")
    assert (undone.status, undone.undone_by) == ("undone", "u")
    assert store.get_node(absorb.id).merged_into is None
