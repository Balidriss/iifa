"""Graph expansion over explicit edges: one hop from the nodes a query touched.

P9 applies to the graph: ``Store.edges_of`` only returns edges whose edge record
and both endpoint nodes are visible, so no path here ever crosses or names an
invisible record.
"""

from __future__ import annotations

from dataclasses import dataclass

from iifa.scopes import Visibility
from iifa.store.models import Edge, Node
from iifa.store.repo import Store

PROPOSED_WEIGHT = 0.5
HOP_DAMPING = 0.6


@dataclass(frozen=True)
class Neighbour:
    node: Node
    via: Edge
    seed: Node
    score: float

    @property
    def path(self) -> str:
        """``Karim -works_on-> Solar project``, read from the seed toward the neighbour."""
        if self.via.src == self.seed.id:
            return f"{self.seed.name} -{self.via.type}-> {self.node.name}"
        return f"{self.seed.name} <-{self.via.type}- {self.node.name}"


def neighbours(
    store: Store, seeds: dict[str, tuple[Node, float]], view: Visibility
) -> list[Neighbour]:
    """Nodes one live, visible edge away from ``seeds`` (id -> (node, score)), best path each."""
    if not seeds:
        return []
    best: dict[str, Neighbour] = {}
    edges = store.edges_of(list(seeds), view)
    other_ids = {e.dst if e.src in seeds else e.src for e in edges}
    others = store.nodes_by_id(sorted(other_ids - set(seeds)), view)
    for edge in edges:
        for seed_id, far_id in ((edge.src, edge.dst), (edge.dst, edge.src)):
            if seed_id not in seeds or far_id in seeds or far_id not in others:
                continue
            seed_node, seed_score = seeds[seed_id]
            weight = edge.weight * (PROPOSED_WEIGHT if edge.status == "proposed" else 1.0)
            score = seed_score * HOP_DAMPING * weight
            current = best.get(far_id)
            if current is None or score > current.score:
                best[far_id] = Neighbour(others[far_id], edge, seed_node, score)
    return sorted(best.values(), key=lambda n: (-n.score, n.node.id))
