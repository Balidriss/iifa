"""Nodes, facts, edges, sources, scope sets, archive/restore/forget, external ids."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from iifa.scopes import Visibility
from iifa.store.repo import NotFound, ScopeMismatch, Store
from tests.conftest import FakeClock

ALICE = Visibility.of_scope("user:alice")


def test_node_fact_edge_round_trip(store: Store) -> None:
    karim = store.create_node(
        scopes=["user:alice"], name="Karim", kind="person", aliases=["my brother", "K"]
    )
    solar = store.create_node(scopes=["user:alice"], name="Solar project", kind="project")
    source = store.add_source(
        scopes=["user:alice"], kind="remember", text="Karim works on the solar project"
    )
    fact = store.add_fact(
        scopes=["user:alice"],
        node_id=karim.id,
        statement="Karim works on the solar project",
        origin="explicit",
        source_id=source.id,
        excerpt=(0, 5),
        attribute="works_on",
        value="solar",
    )
    edge = store.add_edge(
        scopes=["user:alice"], src=karim.id, dst=solar.id, type="works_on", origin="explicit"
    )

    assert karim.aliases == ("K", "my brother")
    assert karim.scopes == ("user:alice",)
    # adding a fact touched the node, so only its revision moved (SPEC 4.4)
    assert store.get_node(karim.id) == replace(karim, rev=store.get_node(karim.id).rev)
    assert store.get_fact(fact.id) == fact
    assert fact.node_id == karim.id and fact.origin == "explicit" and fact.excerpt_end == 5
    assert store.get_edge(edge.id).status == "confirmed"
    assert store.get_source(source.id).text_sha256
    assert store.facts([karim.id], ALICE) == [fact]
    assert store.edges_of([solar.id], ALICE) == [edge]
    assert store.counts()["facts"] == 1


def test_find_node_by_name_or_alias_case_and_accent_insensitive(store: Store) -> None:
    emilie = store.create_node(scopes=["user:alice"], name="Émilie", aliases=["ma sœur"])
    assert store.find_node("emilie", ALICE) == emilie
    assert (
        store.find_node("MA SOEUR", ALICE) == emilie or store.find_node("ma sœur", ALICE) == emilie
    )
    assert store.find_node("Karim", ALICE) is None
    assert store.find_node("   ", ALICE) is None


def test_find_node_prefers_the_closest_scope(store: Store) -> None:
    shared = store.create_node(scopes=["user:alice"], name="Karim")
    mine = store.create_node(scopes=["user:alice/agent:coder"], name="Karim")
    view = Visibility.of_scope("user:alice/agent:coder")
    assert store.find_node("karim", view) == mine
    assert store.find_node("karim", ALICE) == shared


def test_record_may_belong_to_several_scopes_and_is_visible_through_any(store: Store) -> None:
    """Amendment point 1: a set of scopes, visible if any is visible."""
    fact = store.add_fact(
        scopes=["user:alice/project:solar", "user:bob", "user:bob"],
        statement="shared",
        origin="explicit",
    )
    assert fact.scopes == ("user:alice/project:solar", "user:bob")
    assert store.facts_by_id([fact.id], Visibility.of_scope("user:bob"))
    assert store.facts_by_id([fact.id], Visibility.of_scope("user:alice/project:solar/task:1"))
    assert not store.facts_by_id([fact.id], Visibility.of_scope("user:alice"))
    assert not store.facts_by_id([fact.id], Visibility.of_scope("user:carol"))
    assert store.facts_by_id([fact.id], Visibility.of_subtree("user:alice"))
    with pytest.raises(ValueError, match="at least one scope"):
        store.add_fact(scopes=[], statement="nowhere", origin="explicit")


def test_multi_scope_records_come_back_once_per_query(store: Store) -> None:
    """Regression: SQLite 3.51 duplicated rows for EXISTS + IN over a two-scope record."""
    both = ["user:alice", "user:bob"]
    view = Visibility.of_scopes(both)
    a = store.create_node(scopes=both, name="Karim", external_id="x")
    b = store.create_node(scopes=both, name="Solar")
    fact = store.add_fact(
        scopes=both, node_id=a.id, statement="shared solar fact", origin="explicit", external_id="x"
    )
    edge = store.add_edge(scopes=both, src=a.id, dst=b.id, type="works_on", origin="explicit")
    assert sorted(store.nodes_by_id([a.id, b.id], view)) == sorted([a.id, b.id])
    assert list(store.facts_by_id([fact.id], view)) == [fact.id]
    assert store.facts([a.id], view) == [fact]
    assert store.facts_top([a.id], view, 3) == [fact]
    assert store.edges_of([a.id], view) == [edge]
    assert [p.doc_id for p in store.postings(["solar"], view)] == [fact.id, b.id] or sorted(
        p.doc_id for p in store.postings(["solar"], view)
    ) == sorted([fact.id, b.id])
    assert len(store.postings(["solar"], view)) == 2
    assert store.lookup_external("x", view) == [
        {"id": a.id, "kind": "node"},
        {"id": fact.id, "kind": "fact"},
    ]
    found = store.find_node("karim", view)
    assert found is not None and found == replace(a, rev=found.rev)
    assert store.corpus_stats(view).documents == 6  # 3 documents, each in 2 scopes


def test_free_standing_fact_and_attach(store: Store) -> None:
    """Amendment point 6: node_id is optional and can be attached later."""
    fact = store.add_fact(
        scopes=["user:alice"], statement="Prefers short answers", origin="explicit"
    )
    assert fact.node_id is None
    assert store.facts_by_id([fact.id], ALICE)[fact.id] == fact
    alice = store.create_node(scopes=["user:alice"], name="Alice", kind="person")
    attached = store.attach_fact(fact.id, alice.id, pinned=True)
    assert attached.node_id == alice.id and attached.pinned
    assert store.facts([alice.id], ALICE) == [attached]
    detached = store.attach_fact(fact.id, None)
    assert detached.node_id is None and detached.pinned
    with pytest.raises(NotFound):
        store.attach_fact(fact.id, "node_" + "0" * 26)


def test_external_ids_are_kept_and_queryable(store: Store) -> None:
    """Amendment point 5."""
    node = store.create_node(scopes=["user:alice"], name="Karim", external_id="crm-42")
    fact = store.add_fact(
        scopes=["user:alice"], statement="x", origin="imported", external_id="crm-42"
    )
    store.add_source(scopes=["user:bob"], kind="import", text="t", external_id="crm-42")
    found = store.lookup_external("crm-42", ALICE)
    assert found == [{"id": node.id, "kind": "node"}, {"id": fact.id, "kind": "fact"}]
    assert store.lookup_external("crm-42", Visibility.of_scope("user:bob"))[0]["kind"] == "src"
    assert store.lookup_external("nope", ALICE) == []
    assert store.get_node(node.id).external_id == "crm-42"


def test_require_identical_scopes_guards_dream_merges(store: Store) -> None:
    """Amendment point 3: merging records with different scope sets would widen visibility."""
    a = store.add_fact(scopes=["user:alice", "user:bob"], statement="a", origin="extracted")
    b = store.add_fact(scopes=["user:bob", "user:alice"], statement="b", origin="extracted")
    c = store.add_fact(scopes=["user:alice"], statement="c", origin="extracted")
    assert store.require_identical_scopes([a.id, b.id]) == ("user:alice", "user:bob")
    with pytest.raises(ScopeMismatch, match="widen"):
        store.require_identical_scopes([a.id, c.id])
    with pytest.raises(ValueError, match="no records"):
        store.require_identical_scopes([])
    with pytest.raises(NotFound):
        store.scopes_of("fact_" + "0" * 26)


def test_archive_hides_restore_brings_back_unchanged(store: Store, clock: FakeClock) -> None:
    node = store.create_node(scopes=["user:alice"], name="Karim")
    fact = store.add_fact(
        scopes=["user:alice"], node_id=node.id, statement="Karim likes tea", origin="explicit"
    )
    other = store.create_node(scopes=["user:alice"], name="Beverages")
    edge = store.add_edge(
        scopes=["user:alice"], src=node.id, dst=other.id, type="likes", origin="explicit"
    )

    assert store.archive(fact.id) == "fact"
    assert store.facts([node.id], ALICE) == []
    assert store.facts_by_id([fact.id], ALICE) == {}
    assert store.postings(["tea"], ALICE) == []  # gone from lexical candidates too
    listed = store.archived(ALICE)
    assert [(a["id"], a["kind"]) for a in listed] == [(fact.id, "fact")]
    assert store.archive(fact.id) == "fact"  # idempotent
    clock.advance(days=1)
    assert store.restore(fact.id) == "fact"
    restored = store.get_fact(fact.id)
    assert restored == replace(fact, updated_at=restored.updated_at, rev=restored.rev)
    assert restored.rev > fact.rev  # archive and restore are changes (SPEC 4.4)
    assert restored.archived_at is None

    # Archiving a node takes its facts and edges out of every read.
    store.archive(node.id)
    assert store.facts([node.id], ALICE) == []
    assert store.edges_of([other.id], ALICE) == []
    assert store.nodes_by_id([node.id], ALICE) == {}
    assert store.find_node("karim", ALICE) is None
    store.restore(node.id)
    assert store.edges_of([other.id], ALICE) == [edge]
    assert store.facts([node.id], ALICE)[0].id == fact.id


def test_archive_and_restore_reject_unknown_and_unarchivable_ids(store: Store) -> None:
    with pytest.raises(NotFound):
        store.archive("fact_" + "0" * 26)
    with pytest.raises(NotFound):
        store.restore("node_" + "0" * 26)
    src = store.add_source(scopes=["user:alice"], kind="turn", text="hello")
    with pytest.raises(NotFound):
        store.archive(src.id)  # sources are not archivable, only forgotten
    with pytest.raises(NotFound):
        store.archive("junk")


def test_forget_cascades_and_counts(store: Store) -> None:
    karim = store.create_node(scopes=["user:alice"], name="Karim")
    solar = store.create_node(scopes=["user:alice"], name="Solar")
    shared = store.add_source(scopes=["user:alice"], kind="turn", text="Karim and Solar")
    f1 = store.add_fact(
        scopes=["user:alice"],
        node_id=karim.id,
        statement="one",
        origin="extracted",
        source_id=shared.id,
    )
    f2 = store.add_fact(
        scopes=["user:alice"],
        node_id=solar.id,
        statement="two",
        origin="extracted",
        source_id=shared.id,
    )
    store.add_edge(
        scopes=["user:alice"],
        src=karim.id,
        dst=solar.id,
        type="works_on",
        origin="extracted",
        source_id=shared.id,
    )

    assert store.forget(f1.id) == {
        "facts": 1,
        "sources": 0,
        "vectors": 0,
        "probes_deleted": 0,
        "dream_ops_wiped": 0,
        "adjudications_deleted": 0,
    }  # the source is still used by f2
    assert store.forget(karim.id) == {
        "facts": 0,
        "sources": 0,
        "edges": 1,
        "nodes": 1,
        "proposals": 0,
        "vectors": 0,
        "probes_deleted": 0,
        "dream_ops_wiped": 0,
        "adjudications_deleted": 0,
    }
    with pytest.raises(NotFound):
        store.get_node(karim.id)
    assert store.forget(shared.id) == {
        "facts": 1,
        "sources": 1,
        "edges": 0,
        "proposals": 0,
        "vectors": 0,
        "probes_deleted": 0,
        "dream_ops_wiped": 0,
        "adjudications_deleted": 0,
    }
    with pytest.raises(NotFound):
        store.get_fact(f2.id)
    counts = store.counts()
    assert counts["nodes"] == 1 and counts["facts"] == 0 and counts["sources"] == 0
    assert counts["lexical_documents"] == 1  # only the Solar node remains indexed
    assert store.scopes_of(solar.id) == ("user:alice",)
    for gone in (karim.id, f1.id, f2.id, shared.id):
        with pytest.raises(NotFound):
            store.scopes_of(gone)


def test_forget_of_edge_and_unknown(store: Store) -> None:
    a = store.create_node(scopes=["user:alice"], name="A")
    b = store.create_node(scopes=["user:alice"], name="B")
    edge = store.add_edge(
        scopes=["user:alice"], src=a.id, dst=b.id, type="knows", origin="explicit"
    )
    assert store.forget(edge.id) == {"edges": 1, "dream_ops_wiped": 0, "adjudications_deleted": 0}
    for bad in (
        edge.id,
        "edge_" + "0" * 26,
        "src_" + "0" * 26,
        "node_" + "0" * 26,
        "fact_" + "0" * 26,
        "junk",
    ):
        with pytest.raises(NotFound):
            store.forget(bad)


def test_sources_can_be_kept_as_hash_only(tmp_path: Path) -> None:
    with Store(tmp_path / "h.db", "k", retain_sources=False) as store:
        source = store.add_source(scopes=["user:alice"], kind="turn", text="secret text")
        assert source.text is None and len(source.text_sha256) == 64
        assert store.source_if_visible(source.id, ALICE) == source
        assert store.source_if_visible(source.id, Visibility.of_scope("user:bob")) is None


def test_update_node_aliases_reindex_and_the_summary_is_never_searched(store: Store) -> None:
    """SPEC 6.1: a node is found by its name and aliases; its summary is
    model-written text, shown in handles but never ranked by recall."""
    node = store.create_node(scopes=["user:alice"], name="Karim")
    assert store.postings(["kar"], ALICE) == []
    updated = store.update_node(node.id, summary="An electrician in Lyon", add_aliases=["Kar", " "])
    assert updated.summary == "An electrician in Lyon" and updated.aliases == ("Kar",)
    assert [p.doc_id for p in store.postings(["kar"], ALICE)] == [node.id]
    assert store.postings(["electrician"], ALICE) == []
    assert store.corpus_stats(ALICE).documents == 1
    assert store.corpus_stats(Visibility()).documents == 0
    assert store.postings(["x"], Visibility()) == []


def test_closed_facts_leave_recall_but_stay_readable(store: Store) -> None:
    node = store.create_node(scopes=["user:alice"], name="Karim")
    old = store.add_fact(
        scopes=["user:alice"],
        node_id=node.id,
        statement="Karim lives in Paris",
        origin="explicit",
        valid_to="2026-01-01",
    )
    assert store.postings(["paris"], ALICE) == []
    assert store.facts([node.id], ALICE) == []
    assert store.facts([node.id], ALICE, live_only=False) == [old]
    assert store.facts([], ALICE) == [] and store.edges_of([], ALICE) == []
    assert store.facts_by_id([], ALICE) == {} and store.nodes_by_id([], ALICE) == {}
