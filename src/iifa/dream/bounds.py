"""What one dream call reads is bounded (SPEC 5.6).

A local server given a prompt longer than its context window truncates it
silently from the start - the system prompt, and with it the P8 wall, first.
So every unit and every adjudication is counted with the service's own counter
(SPEC 7.3) before it is sent and, over ``prompt_max_tokens``, carries fewer
records: pinned facts first, then explicit ones, then the most recent; the
data says how many were left out. A call whose fixed part alone is over the
budget is not sent at all (``oversized``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from iifa.store.models import Edge, Fact
from iifa.tokens import Tokenizer

OVERSIZED = "oversized"


def prompt_tokens(messages: Sequence[Mapping[str, Any]], tokenizer: Tokenizer) -> int:
    """Both messages, each with the counter's per-item overhead (its wrapper)."""
    return sum(tokenizer.count(str(m["content"])) + tokenizer.per_item_overhead for m in messages)


def by_priority(facts: Sequence[Fact]) -> list[Fact]:
    """Pinned first, then explicit, then the most recent (``valid_from``, then
    creation): the order in which facts are kept when not all of them fit."""
    newest = sorted(facts, key=lambda f: (f.valid_from or "", f.created_at, f.id), reverse=True)
    return sorted(newest, key=lambda f: (not f.pinned, f.origin != "explicit"))


def by_recency(edges: Sequence[Edge]) -> list[Edge]:
    return sorted(edges, key=lambda e: (e.created_at, e.id), reverse=True)


def largest(n: int, fits: Callable[[int], bool]) -> int:
    """The largest ``k`` in ``0..n`` with ``fits(k)`` (``fits`` is monotone:
    fewer records never cost more), or -1 when even ``fits(0)`` is false."""
    if fits(n):
        return n
    if not fits(0):
        return -1
    low, high = 0, n  # fits(low), not fits(high)
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def shown_in_order(chosen: Sequence[Fact], facts: Sequence[Fact]) -> list[Fact]:
    """The chosen facts, in the order the store gave them (the prompt's order)."""
    ids = {f.id for f in chosen}
    return [f for f in facts if f.id in ids]


__all__ = ["OVERSIZED", "by_priority", "by_recency", "largest", "prompt_tokens", "shown_in_order"]
