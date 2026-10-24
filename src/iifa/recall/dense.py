"""Dense retrieval for tier 3 (SPEC 6.1).

The query is embedded once, in the foreground (it does not wait at the
background gate, and holds new background calls while in flight), bounded by a
short timeout; a caller may supply the vector instead, if it carries the
configured model and the stored dimension. Repeated queries hit a small cache.
Candidates are the vectors of the live records the view sees, filtered in SQL
before any comparison (P9). Any failure is reported, never raised: the recall is
then served at tier 2 and says why (P10).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from iifa import vectors
from iifa.llm import ModelError, QueryEmbedder
from iifa.scopes import Visibility
from iifa.store.repo import NAME, TEXT, Store

ON, OFF, UNAVAILABLE = "on", "off", "unavailable"
CACHE_SIZE = 256


@dataclass(frozen=True)
class SuppliedVector:
    model: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class QueryVector:
    values: list[float] | None
    dense: str  # on | off | unavailable
    degraded: str | None = None  # why a configured dense recall was not served
    note: str | None = None  # what became of a supplied vector


@dataclass(frozen=True)
class DenseHit:
    record_id: str
    kind: str
    score: float


class Dense:
    def __init__(
        self,
        store: Store,
        embedder: QueryEmbedder,
        *,
        enabled: bool = True,
        timeout: float = 1.5,
        cache_size: int = CACHE_SIZE,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.enabled = enabled
        self.timeout = timeout
        self._cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @property
    def model(self) -> str | None:
        return self.embedder.model

    @property
    def status(self) -> str:
        return ON if self.enabled and self.model is not None else OFF

    def query_vector(self, query: str, supplied: SuppliedVector | None = None) -> QueryVector:
        model = self.model
        if model is None:
            return QueryVector(None, OFF)
        note: str | None = None
        if supplied is not None:
            note = self._check(supplied, model)
            if note == "used":
                return QueryVector(list(supplied.values), ON, note=note)
        if not self.enabled:
            return QueryVector(None, OFF, note=note)
        key = (model, query)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return QueryVector(cached, ON, note=note)
        try:
            values = self.embedder.embed_query(query, timeout=self.timeout)
        except ModelError as exc:
            return QueryVector(None, UNAVAILABLE, degraded=f"embeddings {exc.reason}", note=note)
        with self._lock:
            self._cache[key] = values
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return QueryVector(values, ON, note=note)

    def _check(self, supplied: SuppliedVector, model: str) -> str:
        if supplied.model != model:
            return f"ignored: model {supplied.model}"
        dimension = self.store.vector_dimension(model)
        if dimension is not None and len(supplied.values) != dimension:
            return f"ignored: dimension {len(supplied.values)}"
        try:
            vectors.normalise(supplied.values)
        except ValueError:
            return "ignored: not a direction"
        return "used"

    def search(self, values: Sequence[float], view: Visibility, limit: int) -> list[DenseHit]:
        """The best ``limit`` records (facts by their statement, nodes by their
        name) for the query vector, inside the view. Never by a summary (SPEC
        6.1): its vector serves link proposals only."""
        model = self.model
        if model is None or limit <= 0:
            return []
        rows = self.store.vector_rows(view, facets=(TEXT, NAME), model=model)
        best: dict[str, DenseHit] = {}
        for index, score in vectors.rank(values, [r.blob for r in rows]):
            record_id = rows[index].record_id
            if record_id not in best:
                kind = "fact" if rows[index].facet == TEXT else "node"
                best[record_id] = DenseHit(record_id, kind, score)
        hits = sorted(best.values(), key=lambda h: (-h.score, h.record_id))
        return hits[:limit]


__all__ = ["OFF", "ON", "UNAVAILABLE", "Dense", "DenseHit", "QueryVector", "SuppliedVector"]
