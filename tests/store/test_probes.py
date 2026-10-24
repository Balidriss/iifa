"""Probe sets (SPEC 4.7, 11.2): stored questions with the records that answer
them, run through recall, scored, and compared before/after (what the dream's
rollback rule 9 needs)."""

from __future__ import annotations

import itertools

import pytest

from iifa.recall.engine import RecallEngine
from iifa.recall.probes import (
    INCONCLUSIVE,
    OK,
    REGRESSED,
    ProbeResult,
    ProbeScore,
    compare,
    run_probe_set,
)
from iifa.store.repo import NotFound, ScopeMismatch, Store
from iifa.tokens import ConservativeTokenizer

ALICE = "user:alice"


def engine(store: Store) -> RecallEngine:
    return RecallEngine(store, ConservativeTokenizer(), tier=2)


def test_a_set_is_created_read_listed_and_deleted(store: Store) -> None:
    node = store.create_node(scopes=[ALICE], name="Karim", kind="person")
    fact = store.add_fact(
        scopes=[ALICE], node_id=node.id, statement="Karim lives in Lyon.", origin="explicit"
    )
    created = store.create_probe_set(
        scope=ALICE,
        name="where people live",
        budget_tokens=300,
        probes=[("Where does Karim live?", [fact.id]), ("Who is Karim?", [node.id])],
    )
    assert created.id.startswith("pset_")
    assert created.scope == ALICE and created.budget_tokens == 300
    again = store.get_probe_set(created.id)
    assert [p.query for p in again.probes] == ["Where does Karim live?", "Who is Karim?"]
    assert [p.expected for p in again.probes] == [(fact.id,), (node.id,)]
    assert [(s.id, size) for s, size in store.probe_sets(ALICE)] == [(created.id, 2)]
    assert store.probe_sets("user:bob") == []
    assert store.delete_probe_set(created.id) == 1
    with pytest.raises(NotFound):
        store.get_probe_set(created.id)
    assert store.delete_probe_set(created.id) == 0


def test_an_ancestor_record_is_visible_from_a_child_scope(store: Store) -> None:
    fact = store.add_fact(scopes=[ALICE], statement="Alice likes tea.", origin="explicit")
    created = store.create_probe_set(
        scope=f"{ALICE}/agent:coder", name="n", budget_tokens=100, probes=[("tea?", [fact.id])]
    )
    assert created.probes[0].expected == (fact.id,)


@pytest.mark.security
def test_p9_a_set_cannot_name_a_record_its_scope_does_not_see(store: Store) -> None:
    """Invisible and unknown ids are refused the same way: no existence leak."""
    bobs = store.add_fact(scopes=["user:bob"], statement="Bob lives in Oslo.", origin="explicit")
    child = store.add_fact(
        scopes=[f"{ALICE}/agent:coder"], statement="A child scope fact.", origin="explicit"
    )
    unknown = "fact_" + "0" * 26
    messages = set()
    for record_id in (bobs.id, child.id, unknown, "not-an-id"):
        with pytest.raises(ScopeMismatch) as caught:
            store.create_probe_set(
                scope=ALICE, name="n", budget_tokens=100, probes=[("q", [record_id])]
            )
        messages.add(str(caught.value).replace(record_id, "<id>"))
    assert len(messages) == 1
    assert store.probe_sets(ALICE) == []


def test_an_archived_record_cannot_be_expected(store: Store) -> None:
    fact = store.add_fact(scopes=[ALICE], statement="Karim lives in Lyon.", origin="explicit")
    store.archive(fact.id)
    with pytest.raises(ScopeMismatch):
        store.create_probe_set(scope=ALICE, name="n", budget_tokens=100, probes=[("q", [fact.id])])


@pytest.mark.security
def test_p5_forget_removes_a_record_from_every_probe(store: Store) -> None:
    node = store.create_node(scopes=[ALICE], name="Karim", kind="person")
    fact = store.add_fact(
        scopes=[ALICE], node_id=node.id, statement="Karim lives in Lyon.", origin="explicit"
    )
    other = store.add_fact(scopes=[ALICE], statement="Léa likes tea.", origin="explicit")
    created = store.create_probe_set(
        scope=ALICE,
        name="n",
        budget_tokens=300,
        probes=[("Karim?", [fact.id, other.id]), ("Who?", [node.id])],
    )
    # "Karim?" still expected Léa's fact too: it goes anyway (its question may carry
    # the forgotten content, and without that answer it measures something else)
    assert store.forget(fact.id)["probes_deleted"] == 1
    assert [p.query for p in store.get_probe_set(created.id).probes] == ["Who?"]
    assert store.forget(node.id)["probes_deleted"] == 1
    assert store.get_probe_set(created.id).probes == ()
    assert store.forget(other.id)["probes_deleted"] == 0
    assert store.execute("SELECT count(*) FROM probes").fetchone()[0] == 0
    assert store.execute("SELECT count(*) FROM probe_expected").fetchone()[0] == 0


def test_a_probe_must_expect_something(store: Store) -> None:
    fact = store.add_fact(scopes=[ALICE], statement="Karim lives in Lyon.", origin="explicit")
    with pytest.raises(ValueError, match="expects"):
        store.create_probe_set(
            scope=ALICE, name="n", budget_tokens=100, probes=[("q", [fact.id]), ("q2", [])]
        )
    assert store.probe_sets(ALICE) == []


def test_running_a_set_scores_found_and_missing_ids(store: Store) -> None:
    karim = store.add_fact(scopes=[ALICE], statement="Karim lives in Lyon.", origin="explicit")
    lea = store.add_fact(scopes=[ALICE], statement="Léa is allergic to peanuts.", origin="explicit")
    created = store.create_probe_set(
        scope=ALICE,
        name="n",
        budget_tokens=400,
        probes=[("Where does Karim live? Lyon", [karim.id]), ("Karim Lyon", [karim.id, lea.id])],
    )
    score = run_probe_set(engine(store), store, store.get_probe_set(created.id))
    assert score.scored == 2
    first, second = score.probes
    assert first.found == (karim.id,) and first.missing == () and first.recall == 1.0
    assert second.found == (karim.id,) and second.missing == (lea.id,) and second.recall == 0.5
    assert score.recall == pytest.approx(0.75)
    assert score.used_tokens > 0
    assert score.recall_per_1k_tokens == pytest.approx(0.75 / score.used_tokens * 1000)
    assert score.tier == 2
    payload = score.as_dict()
    assert "Karim" not in repr(payload)  # ids and numbers only


def test_the_fact_that_superseded_an_expected_one_counts_as_found(store: Store) -> None:
    old = store.add_fact(
        scopes=[ALICE],
        statement="Karim lives in Lyon.",
        origin="extracted",
        attribute="city",
        value="Lyon",
    )
    created = store.create_probe_set(
        scope=ALICE, name="n", budget_tokens=400, probes=[("Where does Karim live", [old.id])]
    )
    new = store.add_fact(
        scopes=[ALICE],
        statement="Karim lives in Paris now.",
        origin="extracted",
        attribute="city",
        value="Paris",
    )
    store.close_fact(old.id, superseded_by=new.id)
    score = run_probe_set(engine(store), store, store.get_probe_set(created.id))
    assert score.probes[0].found == (old.id,)
    assert score.recall == 1.0


def test_a_set_emptied_by_forget_has_nothing_to_score(store: Store) -> None:
    fact = store.add_fact(scopes=[ALICE], statement="Karim lives in Lyon.", origin="explicit")
    created = store.create_probe_set(
        scope=ALICE, name="n", budget_tokens=400, probes=[("Karim", [fact.id])]
    )
    assert store.forget(fact.id)["probes_deleted"] == 1
    emptied = store.get_probe_set(created.id)
    assert emptied.probes == ()
    score = run_probe_set(engine(store), store, emptied)
    assert score.scored == 0 and score.probes == ()
    assert score.recall is None and score.recall_per_1k_tokens is None
    assert compare(score, score).verdict == INCONCLUSIVE


def test_an_archived_expected_record_is_missing(store: Store) -> None:
    fact = store.add_fact(scopes=[ALICE], statement="Karim lives in Lyon.", origin="explicit")
    created = store.create_probe_set(
        scope=ALICE, name="n", budget_tokens=400, probes=[("Karim Lyon", [fact.id])]
    )
    store.archive(fact.id)
    score = run_probe_set(engine(store), store, store.get_probe_set(created.id))
    assert score.probes[0].missing == (fact.id,) and score.recall == 0.0


# --------------------------------------------------------------------------- comparison


def _score(recall: float, tokens: float, *, tier: int | None = 2, ids: int = 2) -> ProbeScore:
    probes = tuple(
        ProbeResult(f"probe_{i}", ("fact_x",), ("fact_x",), (), int(tokens), recall)
        for i in range(ids)
    )
    return ProbeScore(
        set_id="pset_x",
        scope=ALICE,
        budget_tokens=400,
        tier=tier,
        recall=recall,
        used_tokens=tokens,
        recall_per_1k_tokens=recall / tokens * 1000 if tokens else None,
        scored=ids,
        probes=probes,
    )


def test_compare_same_or_better_is_ok() -> None:
    assert compare(_score(0.8, 200), _score(0.8, 200)).verdict == OK
    assert compare(_score(0.8, 200), _score(0.9, 150)).verdict == OK
    within = compare(_score(0.80, 200), _score(0.79, 200), tolerance=0.02)
    assert within.verdict == OK


def test_compare_a_recall_drop_beyond_tolerance_regresses() -> None:
    verdict = compare(_score(0.8, 200), _score(0.7, 100), tolerance=0.02)
    assert verdict.verdict == REGRESSED  # cheaper, but it lost answers
    assert verdict.reason == "recall"
    assert verdict.recall_delta == pytest.approx(-0.1)


def test_compare_a_per_token_drop_beyond_tolerance_regresses() -> None:
    verdict = compare(_score(0.8, 200), _score(0.8, 300), tolerance=0.02)
    assert verdict.verdict == REGRESSED
    assert verdict.reason == "recall_per_1k_tokens"


def test_compare_different_tiers_or_probes_is_inconclusive() -> None:
    assert compare(_score(0.8, 200, tier=3), _score(0.2, 200, tier=2)).verdict == INCONCLUSIVE
    assert compare(_score(0.8, 200, tier=None), _score(0.8, 200)).verdict == INCONCLUSIVE
    assert compare(_score(0.8, 200, ids=2), _score(0.8, 200, ids=3)).verdict == INCONCLUSIVE
    assert compare(_score(0.0, 0, ids=0), _score(0.0, 0, ids=0)).verdict == INCONCLUSIVE


def test_compare_from_zero_tokens_does_not_divide_by_zero() -> None:
    before = _score(0.0, 0)
    assert compare(before, _score(0.5, 100)).verdict == OK
    assert compare(_score(0.5, 100), before).verdict == REGRESSED


def test_a_missed_node_is_missing_and_a_verdict_serialises(store: Store) -> None:
    node = store.create_node(scopes=[ALICE], name="Zoé", kind="person")
    created = store.create_probe_set(
        scope=ALICE, name="n", budget_tokens=400, probes=[("nothing matches", [node.id])]
    )
    score = run_probe_set(engine(store), store, store.get_probe_set(created.id))
    assert score.probes[0].missing == (node.id,)
    verdict = compare(score, score).as_dict()
    assert verdict["verdict"] == OK and verdict["recall_delta"] == 0.0


def test_the_supersession_chain_is_bounded(store: Store) -> None:
    facts = [
        store.add_fact(scopes=[ALICE], statement=f"Karim lives in city {i}.", origin="extracted")
        for i in range(4)
    ]
    for old, new in itertools.pairwise(facts):
        store.close_fact(old.id, superseded_by=new.id)
    assert store.supersession_chain(facts[0].id) == [f.id for f in facts[1:]]
    assert store.supersession_chain(facts[0].id, limit=2) == [facts[1].id, facts[2].id]
    assert store.supersession_chain(facts[3].id) == []
