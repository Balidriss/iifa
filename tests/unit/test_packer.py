"""P1 - the budget packer. Property-based: for any items and any budget, the
output is at or under the budget and never cuts an item."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iifa.recall.packer import PackResult, pack

pytestmark = pytest.mark.security

costs_strategy = st.lists(st.integers(min_value=0, max_value=5000), max_size=60)


@given(costs=costs_strategy, budget=st.integers(min_value=0, max_value=20000))
@settings(max_examples=300, deadline=None)
def test_p1_budget_never_exceeded_and_items_never_cut(costs: list[int], budget: int) -> None:
    result = pack(costs, budget)
    assert result.used_tokens <= budget
    assert result.used_tokens == sum(costs[i] for i in result.chosen)
    assert list(result.chosen) == sorted(set(result.chosen))  # rank order kept, no duplicates
    assert all(0 <= i < len(costs) for i in result.chosen)
    assert result.omitted == len(costs) - len(result.chosen)
    if not costs:
        assert result.reason == "no_items"
    elif not result.chosen:
        assert result.reason == "budget_too_small"
        assert all(c > budget for c in costs)
    else:
        assert result.reason is None


@given(
    costs=costs_strategy,
    budget=st.integers(min_value=0, max_value=20000),
    reserved=st.integers(min_value=0, max_value=3000),
)
@settings(max_examples=200, deadline=None)
def test_p1_reserved_tokens_count_against_the_budget(
    costs: list[int], budget: int, reserved: int
) -> None:
    result = pack(costs, budget, reserved_tokens=reserved)
    if result.chosen:
        assert result.used_tokens <= budget
        assert result.used_tokens == reserved + sum(costs[i] for i in result.chosen)


@given(costs=costs_strategy, budget=st.integers(min_value=0, max_value=20000))
@settings(max_examples=200, deadline=None)
def test_p1_packing_is_first_fit_maximal(costs: list[int], budget: int) -> None:
    """Every omitted item would have overflowed at its position: nothing is left out for no reason."""
    result = pack(costs, budget)
    remaining = budget
    chosen = set(result.chosen)
    for i, cost in enumerate(costs):
        if i in chosen:
            remaining -= cost
        else:
            assert cost > remaining


def test_p1_skips_a_big_item_and_keeps_later_small_ones() -> None:
    assert pack([10, 500, 20], 40) == PackResult(chosen=(0, 2), used_tokens=30, omitted=1)


def test_p1_zero_budget_is_empty_with_a_reason() -> None:
    assert pack([1, 2], 0) == PackResult(omitted=2, reason="budget_too_small")
    assert pack([0], 0) == PackResult(chosen=(0,), used_tokens=0, omitted=0)


def test_p1_negative_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pack([1], -1)
    with pytest.raises(ValueError, match="non-negative"):
        pack([1], 5, reserved_tokens=-1)
    with pytest.raises(ValueError, match="non-negative"):
        pack([-3], 5)
