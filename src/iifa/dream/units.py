"""A dream's unit of work (SPEC 5.6): one node the scope may change, with what
the model needs to judge it - its live facts, its edges and the nodes at their
other end - read in one go with the revisions of every record, the snapshot its
operations are checked against (rule 8). Merge candidates are not part of it:
they go to the adjudication (``iifa.dream.adjudicate``)."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

from iifa.dream.apply import changeable, dream_view
from iifa.dream.bounds import by_priority, by_recency, largest, prompt_tokens, shown_in_order
from iifa.dream.prompts import dream_messages
from iifa.ingest.summaries import facts_language
from iifa.scopes import Visibility
from iifa.store.models import Edge, Fact, Node
from iifa.store.repo import Store
from iifa.tokens import Tokenizer

SCAN = 50  # node ids read at a time while looking for the next unit


@dataclass(frozen=True)
class Fitted:
    """A call that fits its budget: the messages, the records they show (the
    only ones the answer may name) and their size by the service's counter."""

    messages: list[dict[str, str]]
    ids: set[str]
    tokens: int
    reduced: bool


@dataclass(frozen=True)
class Unit:
    node: Node
    facts: tuple[Fact, ...]
    edges: tuple[Edge, ...]
    neighbours: tuple[Node, ...]
    summary_language: str
    snapshot: dict[str, int]

    @property
    def ids(self) -> set[str]:
        """Every record the model may name in its answer."""
        return set(self.snapshot)

    def as_prompt(
        self, facts: Sequence[Fact] | None = None, edges: Sequence[Edge] | None = None
    ) -> dict[str, Any]:
        """What the model reads: ids, names, statements - never a scope or a
        source. With ``facts`` / ``edges``, only those, and how many were left out
        (SPEC 5.6: what one call reads is bounded)."""
        shown_facts = self.facts if facts is None else tuple(facts)
        shown_edges = self.edges if edges is None else tuple(edges)
        ends = {e.src for e in shown_edges} | {e.dst for e in shown_edges}
        out: dict[str, Any] = {
            "node": _node(self.node, summary=True),
            "facts": [
                {
                    "id": f.id,
                    "statement": f.statement,
                    "attribute": f.attribute,
                    "value": f.value,
                    "origin": f.origin,
                    "language": f.language,
                    "valid_from": f.valid_from,
                    "pinned": f.pinned,
                    "flags": list(f.flags),
                }
                for f in shown_facts
            ],
        }
        if len(shown_facts) < len(self.facts):
            out["facts_left_out"] = len(self.facts) - len(shown_facts)
        out["edges"] = [
            {"id": e.id, "src": e.src, "dst": e.dst, "type": e.type, "status": e.status}
            for e in shown_edges
        ]
        if len(shown_edges) < len(self.edges):
            out["edges_left_out"] = len(self.edges) - len(shown_edges)
        out["neighbours"] = [_node(n) for n in self.neighbours if n.id in ends]
        out["summary_language"] = self.summary_language
        return out

    def fitted(self, allowed: Collection[str], tokenizer: Tokenizer, budget: int) -> Fitted | None:
        """The unit's call within ``budget`` tokens: every fact that fits by
        priority, then the most recent edges; None when the node alone is over."""
        facts = by_priority(self.facts)
        edges = by_recency(self.edges)

        def render(n_facts: int, n_edges: int) -> list[dict[str, str]]:
            shown = shown_in_order(facts[:n_facts], self.facts)
            return dream_messages(self.as_prompt(shown, edges[:n_edges]), allowed)

        def fits(n_facts: int, n_edges: int) -> bool:
            return prompt_tokens(render(n_facts, n_edges), tokenizer) <= budget

        n_facts = largest(len(facts), lambda k: fits(k, 0))
        if n_facts < 0:
            return None
        n_edges = largest(len(edges), lambda k: fits(n_facts, k)) if n_facts == len(facts) else 0
        messages = render(n_facts, n_edges)
        shown_edges = edges[:n_edges]
        ends = {e.src for e in shown_edges} | {e.dst for e in shown_edges}
        ids = {
            self.node.id,
            *(f.id for f in facts[:n_facts]),
            *(e.id for e in shown_edges),
            *(n.id for n in self.neighbours if n.id in ends),
        }
        return Fitted(
            messages=messages,
            ids=ids,
            tokens=prompt_tokens(messages, tokenizer),
            reduced=n_facts < len(facts) or n_edges < len(edges),
        )


def _node(node: Node, *, summary: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": node.id,
        "name": node.name,
        "kind": node.kind,
        "aliases": list(node.aliases),
    }
    if summary:
        out["summary"] = node.summary
    return out


def next_unit_node(store: Store, scope: str, after: str) -> Node | None:
    """The first live node after ``after`` (id order) that the dream may change."""
    view = Visibility.of_subtree(scope)
    cursor = after
    while True:
        ids = store.node_ids_after(view, cursor, SCAN)
        if not ids:
            return None
        nodes = store.nodes_by_id(ids, view)
        for node_id in ids:
            node = nodes.get(node_id)
            if node is not None and changeable(node.scopes, scope):
                return node
        cursor = ids[-1]


def read_unit(
    store: Store,
    scope: str,
    node: Node,
    forced_language: Callable[[Sequence[str]], str | None],
) -> Unit:
    """One unit, read under the store lock so that what the model sees and the
    revisions it is checked against are one consistent snapshot."""
    view = dream_view(scope)
    with store.lock:
        current = store.get_node(node.id)
        facts = store.facts([current.id], view)
        edges = store.edges_of([current.id], view)
        others = [e.dst if e.src == current.id else e.src for e in edges]
        nodes = store.nodes_by_id(list(dict.fromkeys(others)), view)
        neighbours = tuple(nodes[i] for i in dict.fromkeys(others) if i in nodes)
        ids = [
            current.id,
            *(f.id for f in facts),
            *(e.id for e in edges),
            *(n.id for n in neighbours),
        ]
        snapshot = store.revisions(ids)
    language = (
        facts_language(list(facts), forced_language(current.scopes))
        if facts
        else (forced_language(current.scopes) or "und")
    )
    return Unit(
        node=current,
        facts=tuple(facts),
        edges=tuple(edges),
        neighbours=neighbours,
        summary_language=language,
        snapshot=snapshot,
    )


__all__ = ["Fitted", "Unit", "next_unit_node", "read_unit"]
