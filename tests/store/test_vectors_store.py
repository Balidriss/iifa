"""Schema version 5: vectors (SPEC 6, 6.1). Stamped with model, dimension and
text hash; what is missing or stale is listed for the background `embed` job;
a change of the embedded text drops the vector; comparisons only ever see live
records inside the view (P9); `forget` deletes vectors (P5)."""

from __future__ import annotations

from pathlib import Path

from iifa.scopes import Visibility
from iifa.store.repo import NAME, SUMMARY, TEXT, Store
from tests.security.test_p5_encryption_archive_forget import _raw_bytes

ALICE = Visibility.of_scope("user:alice")
MODEL = "embed-a"


def _setup(store: Store) -> tuple[str, str, str]:
    karim = store.create_node(scopes=["user:alice"], name="Karim", aliases=["K"])
    fact = store.add_fact(
        scopes=["user:alice"], node_id=karim.id, statement="Karim leads Solar", origin="explicit"
    )
    bob = store.create_node(scopes=["user:bob"], name="Bob", summary="Bob fixes roofs")
    return karim.id, fact.id, bob.id


def test_pending_lists_what_has_no_current_vector(store: Store) -> None:
    karim, fact, bob = _setup(store)
    pending = {(p.record_id, p.facet): p.text for p in store.pending_vectors(MODEL, limit=100)}
    # a node without summary has no summary vector to compute
    assert pending == {
        (karim, NAME): "Karim; K",
        (fact, TEXT): "Karim leads Solar",
        (bob, NAME): "Bob",
        (bob, SUMMARY): "Bob fixes roofs",
    }
    store.put_vector(fact, TEXT, model=MODEL, text="Karim leads Solar", values=[1.0, 0.0])
    left = {(p.record_id, p.facet) for p in store.pending_vectors(MODEL, limit=100)}
    assert (fact, TEXT) not in left and len(left) == 3
    assert store.vector_stats(MODEL) == {"total": 1, "current": 1, "stale": 0, "missing": 3}
    assert len(store.pending_vectors(MODEL, limit=2)) == 2


def test_a_model_change_makes_every_vector_stale(store: Store) -> None:
    karim, fact, _ = _setup(store)
    store.put_vector(fact, TEXT, model=MODEL, text="Karim leads Solar", values=[1.0, 0.0])
    store.put_vector(karim, NAME, model=MODEL, text="Karim; K", values=[0.0, 1.0])
    assert store.vector_stats("embed-b") == {"total": 2, "current": 0, "stale": 2, "missing": 2}
    assert {(p.record_id, p.facet) for p in store.pending_vectors("embed-b", limit=10)} >= {
        (fact, TEXT),
        (karim, NAME),
    }
    # stale vectors are never compared with fresh ones
    assert store.vector_rows(ALICE, facets=(TEXT, NAME), model="embed-b") == []
    store.put_vector(fact, TEXT, model="embed-b", text="Karim leads Solar", values=[1.0, 1.0])
    rows = store.vector_rows(ALICE, facets=(TEXT,), model="embed-b")
    assert [(r.record_id, r.facet) for r in rows] == [(fact, TEXT)]


def test_changing_the_embedded_text_drops_the_vector(store: Store) -> None:
    karim, _, _ = _setup(store)
    store.put_vector(karim, NAME, model=MODEL, text="Karim; K", values=[1.0, 0.0])
    store.update_node(karim, summary="Karim leads the solar project")
    # the summary is new: pending; the name vector did not change
    assert (karim, SUMMARY) in {
        (p.record_id, p.facet) for p in store.pending_vectors(MODEL, limit=10)
    }
    assert store.vector_stats(MODEL)["current"] == 1
    store.update_node(karim, add_aliases=["Карим"])
    pending = {(p.record_id, p.facet): p.text for p in store.pending_vectors(MODEL, limit=10)}
    assert pending[(karim, NAME)] == "Karim; K; Карим"


def test_vector_rows_see_only_live_records_inside_the_view(store: Store) -> None:
    """P9 before any score: an invisible, archived or closed record's vector is
    never handed to a comparison."""
    karim, fact, bob = _setup(store)
    closed = store.add_fact(
        scopes=["user:alice"], node_id=karim, statement="Karim lived in Paris", origin="explicit"
    )
    for record, facet, text in (
        (karim, NAME, "Karim; K"),
        (fact, TEXT, "Karim leads Solar"),
        (bob, NAME, "Bob"),
        (bob, SUMMARY, "Bob fixes roofs"),
        (closed.id, TEXT, "Karim lived in Paris"),
    ):
        store.put_vector(record, facet, model=MODEL, text=text, values=[1.0, 0.5])
    store.close_fact(closed.id)
    seen = {
        (r.record_id, r.facet)
        for r in store.vector_rows(ALICE, facets=(TEXT, NAME, SUMMARY), model=MODEL)
    }
    assert seen == {(karim, NAME), (fact, TEXT)}
    store.archive(karim)
    seen = {
        (r.record_id, r.facet) for r in store.vector_rows(ALICE, facets=(TEXT, NAME), model=MODEL)
    }
    assert seen == set()  # the archived node, and the fact of an archived node
    bob_view = Visibility.of_scope("user:bob")
    assert {
        r.record_id for r in store.vector_rows(bob_view, facets=(NAME, SUMMARY), model=MODEL)
    } == {bob}


def test_forget_deletes_the_vectors(tmp_path: Path) -> None:
    path = tmp_path / "plain.db"
    with Store(path, "", plaintext_for_tests=True, checkpoint_interval=0) as store:
        karim, fact, _ = _setup(store)
        marker = [123.456, 654.321, 111.111]  # a recognisable float32 pattern in the file
        store.put_vector(fact, TEXT, model=MODEL, text="Karim leads Solar", values=marker)
        store.put_vector(karim, NAME, model=MODEL, text="Karim; K", values=marker)
        blob = store.vector_rows(ALICE, facets=(TEXT,), model=MODEL)[0].blob
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert blob in _raw_bytes(path)
        removed = store.forget(karim)
        assert removed["vectors"] == 2
        assert store.vector_stats(MODEL)["total"] == 0
    assert blob not in _raw_bytes(path)
