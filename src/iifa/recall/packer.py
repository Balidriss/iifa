"""The budget packer: the one place P1 is enforced.

Given item costs in rank order and a budget, choose items in order, skipping
any that would not fit, and never cutting one. The result is always at or
under the budget; when nothing fits the result is empty and says why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PackResult:
    chosen: tuple[int, ...] = field(default=())
    used_tokens: int = 0
    omitted: int = 0
    reason: str | None = None


def pack(costs: Sequence[int], budget_tokens: int, *, reserved_tokens: int = 0) -> PackResult:
    """Pick indices of ``costs`` in order so that their sum plus ``reserved_tokens``
    never exceeds ``budget_tokens``. An item that does not fit is skipped, not cut."""
    if budget_tokens < 0 or reserved_tokens < 0:
        raise ValueError("budget_tokens and reserved_tokens must be non-negative")
    if not costs:
        return PackResult(reason="no_items")
    remaining = budget_tokens - reserved_tokens
    chosen: list[int] = []
    used = 0
    for index, cost in enumerate(costs):
        if cost < 0:
            raise ValueError("item costs must be non-negative")
        if cost <= remaining:
            chosen.append(index)
            remaining -= cost
            used += cost
    if not chosen:
        return PackResult(omitted=len(costs), reason="budget_too_small")
    return PackResult(tuple(chosen), used + reserved_tokens, len(costs) - len(chosen), None)
