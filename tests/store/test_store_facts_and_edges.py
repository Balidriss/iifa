"""Store additions of schema version 3: closing and flagging facts (P7, P6),
edge evidence, confirmation, reinforcement and decay (SPEC 5.2, 5.3), the node
index for resolution, what a session touched, and the history queries."""

from __future__ import annotations

import pytest

from iifa.scopes import Visibility
from iifa.store.repo import NotFound, Store
from tests.conftest import FakeClock

ALICE = Visibility.of_scope("user:alice")


def test_close_and_flag_facts(store: Store, clock: FakeClock) -> None:
    node = store.create_node(scopes=["user:alice"], name="Karim")
    old = store.add_fact(
        scopes=["user:alice"],
        node_id=node.id,
        statement="Karim lives in Paris",
        origin="explicit",
        attribute="city",
        value="Paris",
    )
    clock.advance(days=3)
    new = store.add_fact(
        scopes=["user:alice"],
        node_id=node.id,
        statement="Karim lives in Lyon",
        origin="extracted",
        attribute="city",
        value="Lyon",
        language="en",
    )
    closed = store.close_fact(old.id, superseded_by=new.id)
    assert closed.valid_to == store.now() and closed.superseded_by == new.id
    assert store.facts([node.id], ALICE) == [new]  # the closed one left recall
    assert store.postings(["paris"], ALICE) == []
    assert [f.id for f in store.facts([node.id], ALICE, live_only=False)] == [new.id, old.id]
    # closing twice keeps the first valid_to; an explicit valid_to is honoured
    clock.advance(days=1)
    assert store.close_fact(old.id).valid_to == closed.valid_to
    dated = store.add_fact(scopes=["user:alice"], statement="free", origin="extracted")
    assert store.close_fact(dated.id, valid_to="2026-03").valid_to == "2026-03"
    flagged = store.flag_fact(new.id, "conflict", "language_mismatch", conflicts_with=old.id)
    assert flagged.flags == ("conflict", "language_mismatch") and flagged.conflicts_with == old.id
    assert store.flag_fact(new.id, "conflict").flags == ("conflict", "language_mismatch")
    assert new.language == "en" and store.get_node(node.id).updated_at == store.now()
    with pytest.raises(NotFound):
        store.close_fact("fact_" + "0" * 26)
    with pytest.raises(NotFound):
        store.close_fact(new.id, superseded_by="fact_" + "0" * 26)


def test_edge_evidence_confirmation_and_reinforcement(store: Store, clock: FakeClock) -> None:
    a = store.create_node(scopes=["user:alice"], name="Karim")
    b = store.create_node(scopes=["user:alice"], name="Solar")
    s1 = store.add_source(scopes=["user:alice"], kind="turn", text="one", session_id="s1")
    s2 = store.add_source(scopes=["user:alice"], kind="turn", text="two", session_id="s1")
    edge = store.add_edge(
        scopes=["user:alice"],
        src=a.id,
        dst=b.id,
        type="works_on",
        origin="extracted",
        status="proposed",
        source_id=s1.id,
    )
    assert edge.evidence == (s1.id,) and edge.status == "proposed"
    assert store.add_evidence(edge.id, s1.id) == 1  # the same source counts once
    assert store.add_evidence(edge.id, s2.id) == 2
    assert store.get_edge(edge.id).evidence == tuple(sorted((s1.id, s2.id)))
    confirmed = store.confirm_edge(edge.id)
    assert confirmed.status == "confirmed"
    with pytest.raises(ValueError):
        store.add_edge(
            scopes=["user:alice"], src=a.id, dst=b.id, type="x", origin="explicit", status="maybe"
        )
    clock.advance(hours=1)
    reinforced = store.reinforce_edge(edge.id)
    assert reinforced is not None
    assert reinforced.use_count == 1 and reinforced.weight == 1.25
    assert reinforced.last_used_at == store.now()
    for _ in range(50):
        store.reinforce_edge(edge.id)
    assert store.get_edge(edge.id).weight == 10.0  # capped
    closed = store.close_edge(edge.id)
    assert closed.valid_to == store.now() and store.reinforce_edge(edge.id) is None
    assert store.edges_of([a.id], ALICE) == []
    assert (
        store.edges_between([a.id, b.id], ALICE) == [] and store.edges_between([a.id], ALICE) == []
    )
    with pytest.raises(NotFound):
        store.add_evidence(edge.id, "src_" + "0" * 26)
    assert store.nodes_touched_by_session("s1", ALICE) == [a.id, b.id]
    assert store.nodes_touched_by_session("nope", ALICE) == []
    assert store.nodes_touched_by_session("", ALICE) == []


def test_decay_uses_the_clock_never_touches_confirmed_edges_and_prunes_by_closing(
    store: Store, clock: FakeClock
) -> None:
    a = store.create_node(scopes=["user:alice"], name="A")
    b = store.create_node(scopes=["user:alice"], name="B")
    c = store.create_node(scopes=["user:bob"], name="C")
    proposed = store.add_edge(
        scopes=["user:alice"], src=a.id, dst=b.id, type="p", origin="extracted", status="proposed"
    )
    confirmed = store.add_edge(
        scopes=["user:alice"], src=a.id, dst=b.id, type="c", origin="explicit", status="confirmed"
    )
    bobs = store.add_edge(
        scopes=["user:bob"], src=c.id, dst=c.id, type="p", origin="extracted", status="proposed"
    )
    assert store.decay_edges(half_life_days=30, prune_below=0.1) == {"decayed": 0, "pruned": 0}
    clock.advance(days=30)
    counts = store.decay_edges(half_life_days=30, prune_below=0.1, view=ALICE)
    assert counts == {"decayed": 1, "pruned": 0}
    assert store.get_edge(proposed.id).weight == 0.5
    assert store.get_edge(confirmed.id).weight == 1.0
    assert store.get_edge(bobs.id).weight == 1.0  # outside the view
    # the same clock reading again changes nothing (a function of time, not of runs)
    assert store.decay_edges(half_life_days=30, prune_below=0.1, view=ALICE) == {
        "decayed": 0,
        "pruned": 0,
    }
    # a use resets the anchor: reinforcement fights decay
    store.reinforce_edge(proposed.id)
    clock.advance(days=30)
    assert store.decay_edges(half_life_days=30, prune_below=0.1, view=ALICE)["decayed"] == 1
    assert store.get_edge(proposed.id).weight == round(0.75 / 2, 6)
    clock.advance(days=120)
    counts = store.decay_edges(half_life_days=30, prune_below=0.1)
    assert counts == {"decayed": 0, "pruned": 2}  # Bob's edge too, without a view
    pruned = store.get_edge(proposed.id)
    assert pruned.valid_to == store.now() and pruned.archived_at is None  # closed, never deleted
    assert store.edges_of([a.id], ALICE) == [store.get_edge(confirmed.id)]
    with pytest.raises(ValueError):
        store.decay_edges(half_life_days=0, prune_below=0.1)


def test_node_index_and_history_queries(store: Store, clock: FakeClock) -> None:
    karim = store.create_node(scopes=["user:alice"], name="Karim", kind="person", aliases=["K"])
    store.create_node(scopes=["user:bob"], name="Secret")
    clock.advance(minutes=1)
    emilie = store.create_node(scopes=["user:alice"], name="Émilie", language="fr")
    index = store.node_index(ALICE)
    assert index == [
        (emilie.id, "other", "Émilie", "emilie", ()),
        (karim.id, "person", "Karim", "karim", ("k",)),
    ]
    assert store.node_index(ALICE, limit=1) == index[:1] and store.node_index(ALICE, limit=0) == []
    assert emilie.language == "fr" and store.get_node(emilie.id).flags == ()
    flagged = store.update_node(
        emilie.id, summary="Résumé", language="fr", flags=["language_mismatch"]
    )
    assert flagged.summary == "Résumé" and flagged.flags == ("language_mismatch",)
    assert store.update_node(emilie.id, language="en").language == "en"

    first = store.add_fact(
        scopes=["user:alice"], node_id=karim.id, statement="Karim lives in Paris", origin="explicit"
    )
    t1 = store.now()
    clock.advance(days=10)
    second = store.add_fact(
        scopes=["user:alice"], node_id=karim.id, statement="Karim lives in Lyon", origin="extracted"
    )
    store.close_fact(first.id, superseded_by=second.id)
    t2 = store.now()
    clock.advance(days=10)
    third = store.add_fact(
        scopes=["user:alice"], node_id=karim.id, statement="Karim prefers tea", origin="extracted"
    )
    store.archive(third.id)
    hidden = store.add_fact(scopes=["user:bob"], statement="Bob's", origin="explicit")

    assert [f.id for f in store.facts_known_at(karim.id, ALICE, t1)] == [first.id]
    assert [f.id for f in store.facts_known_at(karim.id, ALICE, t2)] == [second.id]
    assert store.facts_known_at(karim.id, ALICE, "2020-01-01") == []
    history = store.fact_history(ALICE, node_id=None, since=None, until=None, limit=10)
    assert [f.id for f in history] == [second.id, first.id]  # archived and invisible ones excluded
    assert hidden.id not in {f.id for f in history}
    ranged = store.fact_history(ALICE, node_id=karim.id, since=t2, until=t2, limit=10)
    assert [f.id for f in ranged] == [second.id, first.id]  # first was still open at t2's start
    later = store.fact_history(ALICE, node_id=karim.id, since=store.now(), until=None, limit=10)
    assert [f.id for f in later] == [second.id]
    assert store.fact_history(ALICE, node_id=karim.id, since=None, until=None, limit=1) == [second]
    assert store.fact_history(ALICE, node_id=karim.id, since=None, until=None, limit=0) == []
    assert store.fact_history(Visibility(), node_id=None, since=None, until=None, limit=5) == []
    assert store.node_index(Visibility()) == []


def test_sources_carry_session_and_language(store: Store) -> None:
    source = store.add_source(
        scopes=["user:alice"], kind="turn", text="Bonjour", session_id="s9", language="fr"
    )
    assert source.session_id == "s9" and source.language == "fr"
    again = store.get_source(source.id)
    assert again.session_id == "s9" and again.language == "fr"
    plain = store.add_source(scopes=["user:alice"], kind="remember", text="x")
    assert plain.session_id is None and plain.language is None
