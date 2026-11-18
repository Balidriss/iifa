"""The profile block (SPEC 8.1): what a session starts with.

Pinned facts first, then in the agent perspective the self's live rules (its
commitments and its preferences, SPEC 4.2), then the most central nodes of
the scope, packed under the budget. The block is **deterministic and
stable** while nothing changes: its items carry no age, no timestamp and no
score, its order depends only on the
records (creation order for pinned facts, centrality then creation order for
nodes), so the same store answers the same bytes and a harness can keep the
block in a cacheable prompt prefix. Visibility is applied before anything is
counted (P9): a node's fact and edge counts include only what the reader sees.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from iifa.ingest.keys import RULE_FAMILIES, canonical_key, family_of
from iifa.recall.engine import RecallEngine
from iifa.recall.handles import summary_line, with_from
from iifa.recall.packer import pack
from iifa.scopes import Visibility
from iifa.store.models import Fact, Node
from iifa.store.repo import Store
from iifa.tokens import Tokenizer

PROFILE_NODES = 20


def profile_fact(fact: Fact, node: Node | None, *, pinned: bool = True) -> dict[str, Any]:
    """A pinned fact, or a rule of the self (``pinned`` False). ``node``
    is None for a free-standing fact and for a fact whose node the reader may
    not see (P9)."""
    item = {
        "id": fact.id,
        "kind": "fact",
        "node_id": node.id if node else None,
        "node_name": node.name if node else None,
        "statement": fact.statement,
        "attribute": fact.attribute,
        "value": fact.value,
        "valid_from": fact.valid_from,
        "flags": sorted({*fact.flags, "pinned"} if pinned else set(fact.flags)),
    }
    return with_from(item, fact)


def self_rules(
    store: Store, view: Visibility, name: str, scope: str
) -> tuple[Node | None, list[Fact]]:
    """The self's node, when the reader sees it, and its live rules (SPEC 4.2:
    its commitments and its preferences) that are not pinned, oldest first:
    an order that depends only on the records."""
    node = store.find_node(name, Visibility.of_scope(scope))
    if node is None or node.id not in store.nodes_by_id([node.id], view):
        return None, []
    facts = [
        f
        for f in store.facts([node.id], view)
        if family_of(canonical_key(f.attribute)) in RULE_FAMILIES and not f.pinned
    ]
    return node, sorted(facts, key=lambda f: (f.created_at, f.id))


def profile_node(node: Node, facts: int, edges: int, tokenizer: Tokenizer) -> dict[str, Any]:
    return {
        "id": node.id,
        "kind": "node",
        "name": node.name,
        "node_kind": node.kind,
        "summary": summary_line(node.summary, tokenizer),  # bounded (SPEC 5.6)
        "aliases": list(node.aliases),
        "facts": facts,
        "edges": edges,
    }


def profile_block(
    engine: RecallEngine,
    *,
    view: Visibility,
    budget_tokens: int,
    exclude_ids: Sequence[str] = (),
    max_nodes: int = PROFILE_NODES,
    agent: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """``agent``: the self's name and declared scope (SPEC 4.6), None in the
    user perspective."""
    store = engine.store
    exclude = set(exclude_ids)
    pinned = [f for f in store.pinned_facts(view) if f.id not in exclude]
    subjects = store.nodes_by_id(sorted({f.node_id for f in pinned if f.node_id}), view)
    items: list[dict[str, Any]] = [
        profile_fact(f, subjects.get(f.node_id) if f.node_id else None) for f in pinned
    ]
    if agent is not None:
        me, rules = self_rules(store, view, *agent)
        items.extend(profile_fact(f, me, pinned=False) for f in rules if f.id not in exclude)
    items.extend(
        profile_node(node, facts, edges, engine.tokenizer)
        for node, facts, edges in store.central_nodes(view, max_nodes)
        if node.id not in exclude
    )
    result = pack([engine.cost(item) for item in items], budget_tokens)
    reason = result.reason
    if not items:
        reason = "empty_profile"
    elif not result.chosen:
        reason = "budget_too_small"
    return {
        "items": [items[i] for i in result.chosen],
        "used_tokens": result.used_tokens,
        "budget_tokens": budget_tokens,
        "omitted": result.omitted,
        "reason": reason,
        "tier": engine.tier,
        "degraded": None,
    }


__all__ = ["PROFILE_NODES", "profile_block", "profile_fact", "profile_node", "self_rules"]
