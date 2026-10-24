"""A small-scale latency check with p95 thresholds (SPEC 7.4, 13.1). The full
10k-fact measurement is ``bench/recall_latency.py``."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer
from tests.fixtures.corpus import people, statements

pytestmark = pytest.mark.perf

FACTS = 1000
QUERIES = [
    "solar project",
    "Lyon",
    "réponses courtes",
    "бюджетом",
    "budget early meetings",
    "physique Karim",
]


def test_recall_p95_under_threshold_at_small_scale(tmp_path: Path) -> None:
    with Store(tmp_path / "perf.db", "k") as store:
        nodes = {
            name: store.create_node(scopes=["user:perf"], name=name, kind="person")
            for name in people(50)
        }
        enqueue: list[float] = []
        for text in statements(FACTS):
            subject = next(n for n in nodes if text.startswith(n))
            started = time.perf_counter()
            store.add_fact(
                scopes=["user:perf"], node_id=nodes[subject].id, statement=text, origin="extracted"
            )
            enqueue.append(time.perf_counter() - started)
        names = list(nodes.values())
        for i in range(0, len(names) - 1, 2):
            store.add_edge(
                scopes=["user:perf"],
                src=names[i].id,
                dst=names[i + 1].id,
                type="works_with",
                origin="extracted",
            )
        engine = RecallEngine(store, ConservativeTokenizer())
        view = Visibility.of_scope("user:perf/agent:bench")
        latencies: list[float] = []
        for i in range(60):
            started = time.perf_counter()
            result = engine.recall(QUERIES[i % len(QUERIES)], view=view, budget_tokens=600)
            latencies.append(time.perf_counter() - started)
            assert result["items"]
    p95 = statistics.quantiles(latencies, n=20)[18]
    write_p95 = statistics.quantiles(enqueue, n=20)[18]
    assert p95 < 0.150, f"recall p95 {p95 * 1000:.1f} ms at {FACTS} facts"
    assert write_p95 < 0.050, f"remember p95 {write_p95 * 1000:.1f} ms"
