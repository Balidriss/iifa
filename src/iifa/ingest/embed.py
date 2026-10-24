"""The background ``embed`` job (SPEC 6.1): vectors for what has none of the
configured model (new records, changed text, a model change), in batches, under
the gate and the run budget (P11). Then, for the nodes it just embedded:

* **merge proposals** — two existing nodes close by their name vectors are never
  merged here; they become a ``merge`` proposal for the dream's review queue, and only
  when their scope sets are identical (dream rule 12: never widen visibility);
* **semantic links** (SPEC 5.2) — nodes whose summary vectors are close get a
  proposed ``similar_to`` edge at reduced weight.

Everything a comparison sees is what the node's own scopes see (P9).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from iifa import log, vectors
from iifa.ingest.extract import EXTRACTED, write_view
from iifa.store.repo import MERGE, NAME, PROPOSED, SUMMARY, PendingVector, Store

if TYPE_CHECKING:
    from iifa.ingest.queue import Job
    from iifa.ingest.worker import JobContext

EMBED = "embed"
SIMILAR_TO = "similar_to"
SIMILAR_WEIGHT = 0.5
MAX_ROUNDS = 1000  # a guard: every round stores a batch, so the queue shrinks

# A test seam between the model call and the writes (the window SPEC 6.1 closes);
# never set in service.
_after_embed: Callable[[list[PendingVector]], None] | None = None


def embed(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    store, config = ctx.store, ctx.config
    model = ctx.models.embeddings_model
    if model is None:
        return {
            "embedded": 0,
            "batches": 0,
            "vectors_discarded": 0,
            "merge_proposals": 0,
            "similar_edges": 0,
        }
    embedded = batches = discarded = 0
    names: list[str] = []
    summaries: list[str] = []
    for _ in range(MAX_ROUNDS):
        pending = store.pending_vectors(model, limit=config.embed_batch)
        if not pending:
            break
        result = ctx.models.embed([p.text for p in pending])
        if _after_embed is not None:
            _after_embed(pending)
        for item, values in zip(pending, result.vectors, strict=True):
            if not store.put_vector(
                item.record_id, item.facet, model=model, text=item.text, values=values
            ):
                discarded += 1  # changed or forgotten during the call (SPEC 6.1, P5)
                continue
            embedded += 1
            if item.facet == NAME:
                names.append(item.record_id)
            elif item.facet == SUMMARY:
                summaries.append(item.record_id)
        batches += 1
    # SPEC 6.1: thresholds are per model; uncalibrated, nothing semantic is proposed
    thresholds = config.thresholds(model)
    merges = links = 0
    if thresholds is not None:
        merges = sum(_propose_merges(store, n, model, thresholds.merge) for n in names)
        links = sum(_link_similar(store, n, model, thresholds.link) for n in summaries)
    log.event(
        "embedded",
        records=embedded,
        batches=batches,
        discarded=discarded,
        merges=merges,
        links=links,
    )
    return {
        "embedded": embedded,
        "batches": batches,
        "vectors_discarded": discarded,
        "merge_proposals": merges,
        "similar_edges": links,
    }


def neighbours(
    store: Store, node_id: str, facet: str, model: str, threshold: float
) -> list[tuple[str, float]]:
    """Other live nodes the node's scopes see, whose ``facet`` vector is at least
    ``threshold`` from the node's own, best first."""
    node = store.get_node(node_id)
    if node.archived_at is not None:
        return []
    rows = store.vector_rows(write_view(list(node.scopes)), facets=(facet,), model=model)
    own = next((r for r in rows if r.record_id == node_id), None)
    if own is None:
        return []
    ranked = vectors.rank(list(vectors.unpack(own.blob)), [r.blob for r in rows])
    return [
        (rows[i].record_id, score)
        for i, score in ranked
        if rows[i].record_id != node_id and score >= threshold
    ]


def _propose_merges(store: Store, node_id: str, model: str, threshold: float) -> int:
    node = store.get_node(node_id)
    proposed = 0
    for other_id, score in neighbours(store, node_id, NAME, model, threshold):
        if store.scopes_of(other_id) != node.scopes:
            continue  # dream rule 12: a merge never widens visibility
        if store.merge_proposed(node_id, other_id):
            continue
        store.add_proposal(
            scopes=node.scopes,
            kind=MERGE,
            node_id=node_id,
            value=other_id,
            confidence=round(score, 4),
        )
        proposed += 1
    return proposed


def _link_similar(store: Store, node_id: str, model: str, threshold: float) -> int:
    node = store.get_node(node_id)
    linked = 0
    for other_id, _score in neighbours(store, node_id, SUMMARY, model, threshold):
        view = write_view(list(node.scopes))
        existing = store.edges_between([node_id, other_id], view)
        if any(e.type == SIMILAR_TO and {e.src, e.dst} == {node_id, other_id} for e in existing):
            continue
        src, dst = sorted((node_id, other_id))
        store.add_edge(
            scopes=node.scopes,
            src=src,
            dst=dst,
            type=SIMILAR_TO,
            origin=EXTRACTED,
            status=PROPOSED,
            weight=SIMILAR_WEIGHT,
        )
        linked += 1
    return linked


__all__ = ["EMBED", "SIMILAR_TO", "SIMILAR_WEIGHT", "embed"]
