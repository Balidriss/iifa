"""Time as a first-class dimension (P7, SPEC 8.2): the timeline of a node or a
scope, and what was known about a node at a date. Both read closed facts as
well as live ones, apply the view before anything is counted (P9: a fact whose
node is invisible never names it) and pack under the budget (P1).
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from typing import Any

from iifa.recall.engine import RecallEngine
from iifa.recall.handles import LINE_CHARS, one_line, with_from
from iifa.recall.packer import pack
from iifa.recall.paging import take
from iifa.scopes import Visibility
from iifa.store.models import Fact, Node

RECORDED, CLOSED = "recorded", "closed"


def parse_bound(value: str, *, end: bool = False) -> str:
    """An ISO date (``2026-03``, ``2026-03-14``) or timestamp as the store writes
    timestamps (UTC, microseconds), at the start of the period or, with
    ``end``, at its last microsecond. Raises ``ValueError`` for anything else."""
    text = value.strip()
    if len(text) == 7 and text[4] == "-":
        year, month = int(text[:4]), int(text[5:7])
        if not 1 <= month <= 12:
            raise ValueError(f"invalid month in {value!r}")
        day = calendar.monthrange(year, month)[1] if end else 1
        text = f"{year:04d}-{month:02d}-{day:02d}"
    if len(text) == 10:
        moment = datetime.fromisoformat(text)
        moment = moment.replace(tzinfo=UTC)
        if end:
            moment = moment.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        moment = datetime.fromisoformat(text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _node_head(node: Node) -> dict[str, Any]:
    return {"id": node.id, "kind": "node", "name": node.name, "node_kind": node.kind}


def _change(fact: Fact, node: Node | None, change: str, at: str) -> dict[str, Any]:
    return with_from(
        {
            "id": fact.id,
            "kind": "fact",
            "change": change,
            "at": at,
            "node_id": node.id if node else None,
            "node_name": node.name if node else None,
            "line": one_line(fact.statement, LINE_CHARS),
            "attribute": fact.attribute,
            "value": fact.value,
            "valid_from": fact.valid_from,
            "valid_to": fact.valid_to,
            "origin": fact.origin,
            "superseded_by": fact.superseded_by,
            "flags": list(fact.flags),
        },
        fact,
    )


def timeline(
    engine: RecallEngine,
    *,
    view: Visibility,
    budget_tokens: int,
    node_id: str | None,
    since: str | None,
    until: str | None,
    limit: int,
    before: tuple[str, str, str] | None = None,
) -> dict[str, Any] | None:
    """Fact changes (recorded, closed) newest first, of one node or of the view,
    one page at a time (SPEC 8): at most ``limit`` changes after the keyset
    position ``before`` (``at``, fact id, change), the longest run of them that
    fits the budget. ``next`` is the position of the last change delivered when
    more follow, else ``None``. ``None`` when ``node_id`` is not a node the view
    may see."""
    store = engine.store
    node: Node | None = None
    if node_id is not None:
        node = store.nodes_by_id([node_id], view).get(node_id)
        if node is None:
            return None
    low = parse_bound(since) if since else None
    high = parse_bound(until, end=True) if until else None
    keys = store.fact_changes(
        view, node_id=node_id, since=low, until=high, before=before, limit=limit + 1
    )
    facts = store.facts_by_id(sorted({fact_id for _, fact_id, _ in keys}), view)
    subjects = store.nodes_by_id(sorted({f.node_id for f in facts.values() if f.node_id}), view)
    changes: list[tuple[tuple[str, str, str], dict[str, Any]]] = []
    for key in keys:
        at, fact_id, change = key
        fact = facts.get(fact_id)
        if fact is None:  # forgotten between the two reads
            continue
        subject = subjects.get(fact.node_id) if fact.node_id else None
        changes.append((key, _change(fact, subject, change, at)))
    page = take(
        changes,
        limit=limit,
        budget_tokens=budget_tokens,
        cost=lambda c: engine.cost(c[1]),
        empty_reason="no_changes",
        more_after=len(keys) > limit,
    )
    return {
        "node": _node_head(node) if node else None,
        "since": low,
        "until": high,
        "items": [c[1] for c in page.items],
        "used_tokens": page.used_tokens,
        "budget_tokens": budget_tokens,
        "omitted": page.omitted,
        "reason": page.reason,
        "next": page.last[0] if page.more and page.last else None,
    }


def _known(fact: Fact) -> dict[str, Any]:
    return with_from(
        {
            "id": fact.id,
            "kind": "fact",
            "statement": fact.statement,
            "attribute": fact.attribute,
            "value": fact.value,
            "valid_from": fact.valid_from,
            "valid_to": fact.valid_to,
            "origin": fact.origin,
            "confidence": fact.confidence,
            "recorded_at": fact.created_at,
            "superseded_by": fact.superseded_by,
            "flags": list(fact.flags),
        },
        fact,
    )


def known_at(
    engine: RecallEngine, *, view: Visibility, budget_tokens: int, node_id: str, date: str
) -> dict[str, Any] | None:
    """What was known about ``node_id`` at ``date``: the facts recorded by then and
    not closed by then, live or since superseded. ``None`` for an invisible node."""
    store = engine.store
    node = store.nodes_by_id([node_id], view).get(node_id)
    if node is None:
        return None
    at = parse_bound(date, end=True)
    items = [_known(f) for f in store.facts_known_at(node_id, view, at)]
    head = {**_node_head(node), "at": at}
    head_cost = engine.cost(head)
    if head_cost > budget_tokens:
        return {
            **head,
            "items": [],
            "used_tokens": 0,
            "budget_tokens": budget_tokens,
            "omitted": 1 + len(items),
            "reason": "budget_too_small",
        }
    result = pack([engine.cost(i) for i in items], budget_tokens, reserved_tokens=head_cost)
    reason = result.reason
    if not items:
        reason = "nothing_known"
    elif not result.chosen:
        reason = "budget_too_small"
    return {
        **head,
        "items": [items[i] for i in result.chosen],
        "used_tokens": result.used_tokens if result.chosen else head_cost,
        "budget_tokens": budget_tokens,
        "omitted": result.omitted,
        "reason": reason,
    }


__all__ = ["CLOSED", "RECORDED", "known_at", "parse_bound", "timeline"]
