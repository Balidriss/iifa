"""The bench's measure of matching an activity by vectors (SPEC 5.5, 11.1):
per probe, the cosine of each task node and whether a word already matches
it; per threshold, which positives' rules come in a block and how many
negatives bring one."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from bench.self_bench import CASES, VECTOR_PROBES, vector_sweep

from iifa.ingest.activity import matched_words
from iifa.recall.dense import ON, Dense, DenseHit, QueryVector
from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

SCOPE = "user:alice/agent:nova"
VIEW = Visibility.of_scope(SCOPE)
QUERIES = [query for _, query, _ in VECTOR_PROBES]


class _Vectors:
    """Each probe finds the map activity at its own closeness."""

    def __init__(self, record: str, closeness: dict[str, float]) -> None:
        self.record, self.closeness = record, closeness
        self.model = "bge-m3"

    def query_vector(self, query: str, supplied: Any = None) -> QueryVector:
        return QueryVector([float(QUERIES.index(query))], ON)

    def search(self, values: Sequence[float], view: Visibility, limit: int) -> list[DenseHit]:
        score = self.closeness.get(QUERIES[int(values[0])])
        return [] if score is None else [DenseHit(self.record, "node", score)]


def test_the_sweep_reports_cosines_and_matches_per_threshold(store: Store) -> None:
    task = store.create_node(scopes=[SCOPE], name="drawing a map", kind="task")
    rule = store.add_fact(
        scopes=[SCOPE],
        node_id=task.id,
        statement="Drawing a map: draw north at the top.",
        origin="extracted",
        attribute="method",
        value="north at the top",
        activities=[task.id],
    )
    positive, negative = QUERIES[0], QUERIES[3]  # map in other words; the capital
    engine = RecallEngine(
        store,
        ConservativeTokenizer(),
        tier=3,
        dense=cast(Dense, _Vectors(task.id, {positive: 0.62, negative: 0.48})),
    )
    written = {"activity": [{"id": rule.id, "key": "method"}]}
    report = vector_sweep(store, engine, VIEW, written, thresholds=(0.45, 0.5, 0.65))
    assert report["scores"]["map-other-words"] == [
        {"activity": "drawing a map", "cosine": 0.62, "word": False}
    ]
    sweep = report["sweep"]
    assert sweep["words only"]["positives"]["map-other-words"] == "0/1"
    assert sweep["0.45"]["negatives_with_block"] == 1  # the capital matched the map
    assert sweep["0.5"]["negatives_with_block"] == 0
    assert sweep["0.5"]["positives"]["map-other-words"] == "1/1"
    assert sweep["0.65"]["positives"]["map-other-words"] == "0/1"


def test_the_positive_probes_share_no_word_with_any_activity_name() -> None:
    """A positive probe matched by a word measures nothing about vectors: none
    shares a word with the gold activities nor with the names a real model
    gave them."""
    names = [c.activity[0] for c in CASES if c.activity]
    names += [  # as a Qwen3.8 27B variant named them on the --self corpus
        "drawing a map of the balcony planters",
        "writing Lumen 2.2 release notes",
        "restoring a backup",
    ]
    for key, query, target in VECTOR_PROBES:
        assert all(matched_words(name, query) == 0 for name in names), key
        assert target is None or target in {c.key for c in CASES}
