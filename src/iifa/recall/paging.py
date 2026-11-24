"""A page of a list: the longest run of items, in the list's order,
that holds at most ``limit`` items and fits under the budget (P1). An item that
does not fit is never skipped: it opens the next page, so paging by keyset
delivers every item exactly once."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from iifa.recall.packer import pack_prefix

T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    items: list[T]
    used_tokens: int
    omitted: int
    reason: str | None
    last: T | None
    more: bool


def take(
    fetched: Sequence[T],
    *,
    limit: int,
    budget_tokens: int,
    cost: Callable[[T], int],
    empty_reason: str | None = "no_items",
    more_after: bool = False,
) -> Page[T]:
    """The page out of ``fetched``: up to ``limit + 1`` items read after the
    cursor, the extra one only telling whether more follow. ``omitted`` counts
    the items of the ``limit`` window the budget left for the next page;
    ``more`` is true when an item follows ``last``, the last one delivered.
    A page that delivers nothing has ``more`` false and says why in ``reason``:
    the same cursor with a larger budget gets it. ``more_after`` says more
    follow whatever ``fetched`` holds (items read, then dropped)."""
    window = list(fetched[:limit])
    result = pack_prefix([cost(item) for item in window], budget_tokens)
    chosen = [window[i] for i in result.chosen]
    reason = empty_reason if result.reason == "no_items" else result.reason
    last = chosen[-1] if chosen else None
    more = bool(chosen) and (len(fetched) > limit or more_after or len(chosen) < len(window))
    return Page(chosen, result.used_tokens, result.omitted, reason, last, more)


__all__ = ["Page", "take"]
