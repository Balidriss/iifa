"""The summary shown is bounded (SPEC 5.6): a node handle's line and a profile
entry's summary show at most SUMMARY_LINE_TOKENS tokens of the summary,
counted with the configured tokenizer, cut at a word boundary (a character
boundary without spaces) and ended with "…" inside the bound. The stored
summary stays whole and expand shows it in full."""

from __future__ import annotations

import pytest

from iifa.recall.engine import RecallEngine
from iifa.recall.handles import SUMMARY_LINE_TOKENS, summary_line
from iifa.recall.profile import profile_block
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

ALICE = "user:alice"
VIEW = Visibility.of_scope(ALICE)
TOKENS = ConservativeTokenizer()
ENGLISH = (
    "Karim is Bob's colleague; he moved from Paris to Lyon in March 2026, works as a "
    "solar project engineer at Atlas, drinks green tea every morning and plays football."
)
RUSSIAN = (
    "София — подруга Боба, живёт в Монреале с 2025 года, работает дизайнером в "
    "компании Orion и учит японский язык по вечерам."
)
CHINESE = (
    "美玲是鲍勃的同事，二零二五年搬到上海，在一家太阳能公司做工程师，每天早上喝绿茶，周末打羽毛球。"
)


@pytest.mark.parametrize("summary", [ENGLISH, RUSSIAN])
def test_a_long_summary_is_cut_at_a_word_inside_the_bound(summary: str) -> None:
    line = summary_line(summary, TOKENS)
    assert TOKENS.count(line) <= SUMMARY_LINE_TOKENS
    assert line.endswith("…")
    shown = line[:-1]
    assert summary.startswith(shown) and summary[len(shown)] == " "  # a whole word
    assert len(shown.split()) >= 2  # not cut to nothing


def test_a_summary_without_spaces_is_cut_at_a_character() -> None:
    line = summary_line(CHINESE, TOKENS)
    assert TOKENS.count(line) <= SUMMARY_LINE_TOKENS
    assert line.endswith("…") and CHINESE.startswith(line[:-1]) and len(line) > 2


def test_a_short_summary_is_shown_whole_and_flat() -> None:
    assert summary_line("Karim lives  in\nLyon.", TOKENS) == "Karim lives in Lyon."
    assert summary_line("", TOKENS) == ""


def summarised(store: Store) -> str:
    node = store.create_node(scopes=[ALICE], name="Karim", kind="person")
    store.add_fact(
        scopes=[ALICE], node_id=node.id, statement="Karim lives in Lyon.", origin="explicit"
    )
    store.update_node(node.id, summary=ENGLISH)
    return node.id


def test_a_node_handle_shows_the_bounded_summary_and_expand_the_whole(store: Store) -> None:
    node_id = summarised(store)
    engine = RecallEngine(store, TOKENS, tier=2)
    answer = engine.recall("Karim", view=VIEW, budget_tokens=600)
    [handle] = [i for i in answer["items"] if i["id"] == node_id]
    assert handle["line"] == summary_line(ENGLISH, TOKENS) != ENGLISH
    expanded = engine.expand([node_id], view=VIEW, budget_tokens=2000)
    assert expanded["items"][0]["summary"] == ENGLISH
    assert store.get_node(node_id).summary == ENGLISH  # stored whole


def test_a_node_without_a_summary_keeps_its_kind_and_count(store: Store) -> None:
    node = store.create_node(scopes=[ALICE], name="Karim", kind="person")
    store.add_fact(
        scopes=[ALICE], node_id=node.id, statement="Karim lives in Lyon.", origin="explicit"
    )
    engine = RecallEngine(store, TOKENS, tier=2)
    answer = engine.recall("Karim", view=VIEW, budget_tokens=600)
    [handle] = [i for i in answer["items"] if i["id"] == node.id]
    assert handle["line"] == "person, 1 live facts"


def test_the_profile_block_shows_the_same_bound(store: Store) -> None:
    node_id = summarised(store)
    lyon = store.create_node(scopes=[ALICE], name="Lyon", kind="place")
    store.add_edge(scopes=[ALICE], src=node_id, dst=lyon.id, type="lives_in", origin="explicit")
    block = profile_block(RecallEngine(store, TOKENS, tier=2), view=VIEW, budget_tokens=600)
    [entry] = [i for i in block["items"] if i["id"] == node_id]
    assert entry["summary"] == summary_line(ENGLISH, TOKENS)


class _Lumpy:
    """One token per character, but a cut ending in "d" costs ten more: a count
    that is not monotonic in the length."""

    name = "lumpy"
    per_item_overhead = 0

    def count(self, text: str) -> int:
        return len(text) + (10 if text.endswith("d…") else 0)


def test_a_count_that_is_not_monotonic_still_ends_inside_the_bound() -> None:
    lumpy = _Lumpy()
    for limit in range(1, 29):  # the whole text (29) fits from 29 on
        line = summary_line("abcd efgh ijkl mnop qrst uvwx", lumpy, limit=limit)
        assert lumpy.count(line) <= limit and line.endswith("…"), (limit, line)
