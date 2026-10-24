"""``POST /v1/path`` (SPEC 8.2): how two nodes are related, by a bounded
breadth-first search over live edges. ``Store.edges_of`` applies the view to
every edge and both of its ends before a hop is taken, so a path never crosses
or names an invisible record (P9: a path through an invisible node is a leak).
"""

from __future__ import annotations

from typing import Any

from iifa.recall.engine import RecallEngine
from iifa.recall.packer import pack
from iifa.scopes import Visibility
from iifa.store.models import Edge, Node
from iifa.store.repo import Store

MAX_VISITED = 2000
FORWARD, BACKWARD = "forward", "backward"


def _search(
    store: Store, view: Visibility, src: str, dst: str, max_hops: int
) -> list[tuple[str, Edge, str]] | None:
    """Edges from ``src`` to ``dst`` as ``(from, edge, to)`` steps, or None."""
    if src == dst:
        return []
    parents: dict[str, tuple[str, Edge]] = {}
    frontier = [src]
    visited = {src}
    for _ in range(max_hops):
        if not frontier or len(visited) > MAX_VISITED:
            break
        next_frontier: list[str] = []
        for edge in store.edges_of(frontier, view):
            for here, there in ((edge.src, edge.dst), (edge.dst, edge.src)):
                if here not in visited or there in visited or here not in frontier:
                    continue
                visited.add(there)
                parents[there] = (here, edge)
                next_frontier.append(there)
                if there == dst:
                    return _unwind(parents, src, dst)
        frontier = next_frontier
    return None


def _unwind(
    parents: dict[str, tuple[str, Edge]], src: str, dst: str
) -> list[tuple[str, Edge, str]]:
    steps: list[tuple[str, Edge, str]] = []
    node = dst
    while node != src:
        here, edge = parents[node]
        steps.append((here, edge, node))
        node = here
    steps.reverse()
    return steps


def _step(here: str, edge: Edge, there: str, names: dict[str, Node]) -> dict[str, Any]:
    forward = edge.src == here
    return {
        "from": here,
        "from_name": names[here].name,
        "edge_id": edge.id,
        "type": edge.type,
        "status": edge.status,
        "weight": edge.weight,
        "direction": FORWARD if forward else BACKWARD,
        "to": there,
        "to_name": names[there].name,
    }


def _text(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    parts = [steps[0]["from_name"]]
    for step in steps:
        arrow = f" -{step['type']}-> " if step["direction"] == FORWARD else f" <-{step['type']}- "
        parts.append(arrow + step["to_name"])
    return "".join(parts)


def path(
    engine: RecallEngine,
    *,
    view: Visibility,
    budget_tokens: int,
    src: str,
    dst: str,
    max_hops: int,
) -> dict[str, Any] | None:
    """The shortest visible path between two visible nodes, under budget: the
    steps are delivered whole or not at all (a partial path would mislead).
    ``None`` when either end is not a node the view may see."""
    store = engine.store
    ends = store.nodes_by_id([src, dst], view)
    if src not in ends or dst not in ends:
        return None
    found = _search(store, view, src, dst, max_hops)
    head: dict[str, Any] = {
        "src": {"id": src, "name": ends[src].name},
        "dst": {"id": dst, "name": ends[dst].name},
        "max_hops": max_hops,
        "found": found is not None,
        "hops": len(found) if found is not None else None,
    }
    if not found:
        return {
            **head,
            "steps": [],
            "path": "" if found == [] else None,
            "used_tokens": 0,
            "budget_tokens": budget_tokens,
            "omitted": 0,
            "reason": "same_node" if found == [] else "no_path",
        }
    names = store.nodes_by_id(sorted({n for here, _, there in found for n in (here, there)}), view)
    steps = [_step(here, edge, there, names) for here, edge, there in found]
    text = _text(steps)
    payload = {**head, "steps": steps, "path": text}
    cost = engine.cost(payload)
    if cost > budget_tokens:
        return {
            **head,
            "steps": [],
            "path": None,
            "used_tokens": 0,
            "budget_tokens": budget_tokens,
            "omitted": len(steps),
            "reason": "budget_too_small",
        }
    packed = pack([cost], budget_tokens)
    return {
        **payload,
        "used_tokens": packed.used_tokens,
        "budget_tokens": budget_tokens,
        "omitted": 0,
        "reason": None,
    }


__all__ = ["BACKWARD", "FORWARD", "MAX_VISITED", "path"]
