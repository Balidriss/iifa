"""Summary tokens (SPEC 11.2): per probe, what the summary lines of the node
handles returned cost above the line of a node without a summary - never
negative, at most the bound, none for a fact handle. The comparison uses
recall per 1k tokens without them (``compared_per_1k_tokens``), computed the
same way on both scores; recall stays strict and every other token counts."""

from __future__ import annotations

from typing import Any

from iifa.dream.run import _score_dict, _score_from
from iifa.recall.engine import RecallEngine
from iifa.recall.handles import SUMMARY_LINE_TOKENS
from iifa.recall.probes import OK, REGRESSED, compare, run_probe_set
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

ALICE = "user:alice"
VIEW = Visibility.of_scope(ALICE)
SUMMARY = (
    "Karim is Bob's colleague; he moved from Paris to Lyon in March 2026, works as a "
    "solar project engineer at Atlas, drinks green tea every morning and plays football."
)
REWRITTEN = (
    "Karim, Bob's colleague, lives in Lyon since March 2026 and works at Atlas as an "
    "engineer on the solar project; he drinks green tea and plays football on Sundays."
)


def engine(store: Store) -> RecallEngine:
    return RecallEngine(store, ConservativeTokenizer(), tier=2)


def memory(store: Store) -> tuple[str, str, str]:
    """Karim (a node handle answers "Karim") and the probe set that asks it."""
    node = store.create_node(scopes=[ALICE], name="Karim", kind="person")
    fact = store.add_fact(
        scopes=[ALICE], node_id=node.id, statement="Karim lives in Lyon.", origin="explicit"
    )
    probe_set = store.create_probe_set(
        scope=ALICE,
        name="karim",
        budget_tokens=600,
        probes=[("Karim", [node.id]), ("Karim Lyon", [fact.id])],
    )
    return node.id, fact.id, probe_set.id


def plain_cost(recall: RecallEngine, store: Store, item: dict[str, Any]) -> int:
    node = store.get_node(item["id"])
    count = store.fact_counts([node.id], VIEW)[node.id]
    return recall.cost({**item, "line": f"{node.kind}, {count} live facts"})


def test_summary_tokens_are_what_the_summary_line_costs_above_the_plain_line(
    store: Store,
) -> None:
    node_id, _, set_id = memory(store)
    store.update_node(node_id, summary=SUMMARY)
    recall = engine(store)
    answer = recall.recall("Karim", view=VIEW, budget_tokens=600)
    [item] = [i for i in answer["items"] if i["id"] == node_id]
    expected = recall.cost(item) - plain_cost(recall, store, item)
    assert 0 < expected <= SUMMARY_LINE_TOKENS
    score = run_probe_set(recall, store, store.get_probe_set(set_id))
    assert score.probes[0].summary_tokens == expected
    assert score.summary_tokens == sum(p.summary_tokens for p in score.probes) / 2
    assert (
        score.compared_per_1k_tokens
        == score.recall / (score.used_tokens - score.summary_tokens) * 1000
    )
    assert score.as_dict()["compared_per_1k_tokens"] == score.compared_per_1k_tokens


def test_without_a_summary_nothing_is_set_aside(store: Store) -> None:
    _, _, set_id = memory(store)
    score = run_probe_set(engine(store), store, store.get_probe_set(set_id))
    assert score.summary_tokens == 0
    assert score.compared_per_1k_tokens == score.recall_per_1k_tokens


def test_a_first_summary_alone_never_regresses_the_comparison(store: Store) -> None:
    node_id, _, set_id = memory(store)
    recall = engine(store)
    before = run_probe_set(recall, store, store.get_probe_set(set_id))
    store.update_node(node_id, summary=SUMMARY)
    after = run_probe_set(recall, store, store.get_probe_set(set_id))
    assert after.used_tokens > before.used_tokens  # the raw price stays visible
    assert (after.recall_per_1k_tokens or 0) < (before.recall_per_1k_tokens or 0) * 0.98
    assert compare(before, after).verdict == OK


def test_a_rewritten_summary_compares_like_with_like(store: Store) -> None:
    """A second dream: summaries already exist, both scores set them aside."""
    node_id, _, set_id = memory(store)
    store.update_node(node_id, summary="Karim lives in Lyon.")
    recall = engine(store)
    before = run_probe_set(recall, store, store.get_probe_set(set_id))
    assert before.summary_tokens > 0
    store.update_node(node_id, summary=REWRITTEN)
    after = run_probe_set(recall, store, store.get_probe_set(set_id))
    assert compare(before, after).verdict == OK


def test_a_lost_answer_still_regresses_whatever_the_summaries(store: Store) -> None:
    node_id, fact_id, set_id = memory(store)
    recall = engine(store)
    before = run_probe_set(recall, store, store.get_probe_set(set_id))
    store.update_node(node_id, summary=SUMMARY)
    store.archive(fact_id)
    verdict = compare(before, run_probe_set(recall, store, store.get_probe_set(set_id)))
    assert (verdict.verdict, verdict.reason) == (REGRESSED, "recall")


def test_a_longer_fact_statement_is_never_set_aside(store: Store) -> None:
    """Only a node handle's summary line is priced as the plain line: a fact's
    statement, however long, counts in full and still regresses."""
    node_id, _, set_id = memory(store)
    recall = engine(store)
    before = run_probe_set(recall, store, store.get_probe_set(set_id))
    store.add_fact(
        scopes=[ALICE],
        node_id=node_id,
        statement="Karim " + " ".join(["walks along the river in Lyon"] * 4) + ".",
        origin="explicit",
    )
    after = run_probe_set(recall, store, store.get_probe_set(set_id))
    assert after.summary_tokens == 0
    verdict = compare(before, after)
    assert (verdict.verdict, verdict.reason) == (REGRESSED, "recall_per_1k_tokens")


def test_a_before_saved_without_the_field_compares_both_raw(store: Store) -> None:
    """A "before" saved before summary tokens were counted carries no
    compared_per_1k_tokens: both scores are then compared raw, never a raw
    "before" against a compared "after" (that would always favour the dream)."""
    node_id, _, set_id = memory(store)
    recall = engine(store)
    saved = _score_dict(run_probe_set(recall, store, store.get_probe_set(set_id)))
    del saved["compared_per_1k_tokens"], saved["summary_tokens"]
    before = _score_from(saved)
    assert before.compared_per_1k_tokens is None
    store.update_node(node_id, summary=SUMMARY)  # a refresh-only dream
    after = run_probe_set(recall, store, store.get_probe_set(set_id))
    assert after.compared_per_1k_tokens is not None and after.summary_tokens > 0
    verdict = compare(before, after)
    assert (verdict.verdict, verdict.reason) == (REGRESSED, "recall_per_1k_tokens")
    assert (
        verdict.per_token_delta
        == (after.recall_per_1k_tokens or 0) / (before.recall_per_1k_tokens or 1) - 1
    )
