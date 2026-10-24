"""Measure tier-0 recall latency and the write path at scale (SPEC 7.4).

    python bench/recall_latency.py --facts 10000 --queries 200

Builds an encrypted store in a temporary directory from the synthetic
multilingual corpus, times every ``store.add_fact`` call (the write behind
``POST /v1/remember``, without HTTP), then runs recall with a 600-token budget.
Both are reported as full distributions - p50, p95, p99, max, mean and total -
because a p95 alone hides a tail: a WAL checkpoint is rare and expensive (two
fsyncs), and on Windows it is where most of an ingestion's wall-clock time
goes. The tail line names the writes above ten times the median and their
share of the total. No network, no model.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.corpus import people, statements

from iifa import __version__
from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.connection import BACKSTOP_AUTOCHECKPOINT_PAGES
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

QUERIES = [
    "solar project",
    "who moved to Lyon",
    "réponses courtes",
    "бюджет",
    "early meetings budget",
    "physique",
    "Karim",
    "Émilie déteste",
    "Иван изучает",
    "manages the budget",
]

TAIL_FACTOR = 10


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    """The distribution of ``values`` (seconds): count, total, mean, p50, p95, p99, max."""
    if not values:
        raise ValueError("no values")
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def tail(values: list[float], factor: int = TAIL_FACTOR) -> dict[str, float]:
    """The values above ``factor`` times the median: how many, their time, their share."""
    threshold = percentile(values, 50) * factor
    slow = [v for v in values if v > threshold]
    total = sum(values)
    return {
        "threshold": threshold,
        "count": len(slow),
        "total": sum(slow),
        "share": (sum(slow) / total) if total else 0.0,
    }


def format_summary(label: str, values: list[float]) -> str:
    s = summarize(values)
    return (
        f"{label} ({s['count']} calls): total {s['total']:.2f} s, mean {s['mean'] * 1000:.2f} ms, "
        f"p50 {s['p50'] * 1000:.2f} ms, p95 {s['p95'] * 1000:.2f} ms, "
        f"p99 {s['p99'] * 1000:.2f} ms, max {s['max'] * 1000:.1f} ms"
    )


def format_tail(values: list[float]) -> str:
    t = tail(values)
    return (
        f"  tail: {t['count']} calls above {t['threshold'] * 1000:.1f} ms "
        f"({TAIL_FACTOR}x the median) took {t['total']:.2f} s = {t['share'] * 100:.1f} % of the total"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--budget", type=int, default=600)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp, Store(Path(tmp) / "bench.db", "bench-key") as store:
        names = people(50)
        nodes = {n: store.create_node(scopes=["user:bench"], name=n, kind="person") for n in names}
        ids = list(nodes.values())
        for i in range(0, len(ids) - 1, 2):
            store.add_edge(
                scopes=["user:bench"],
                src=ids[i].id,
                dst=ids[i + 1].id,
                type="works_with",
                origin="explicit",
            )
        writes: list[float] = []
        started_all = time.perf_counter()
        for text in statements(args.facts):
            subject = next(n for n in names if text.startswith(n))
            t0 = time.perf_counter()
            store.add_fact(
                scopes=["user:bench"],
                node_id=nodes[subject].id,
                statement=text,
                origin="extracted",
            )
            writes.append(time.perf_counter() - t0)
        ingest_seconds = time.perf_counter() - started_all
        checkpoints = store.checkpointer.stats if store.checkpointer else None

        engine = RecallEngine(store, ConservativeTokenizer())
        view = Visibility.of_scope("user:bench/agent:x")
        for q in QUERIES:  # warm-up
            engine.recall(q, view=view, budget_tokens=args.budget)
        latencies: list[float] = []
        items = 0
        for i in range(args.queries):
            t0 = time.perf_counter()
            result = engine.recall(QUERIES[i % len(QUERIES)], view=view, budget_tokens=args.budget)
            latencies.append(time.perf_counter() - t0)
            items += len(result["items"])
        counts = store.counts()

    print(f"iifa {__version__} tier 0 benchmark")
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(
        f"facts {counts['facts']}  nodes {counts['nodes']}  edges {counts['edges']}  scope user:bench"
        f"  writer autocheckpoint backstop {BACKSTOP_AUTOCHECKPOINT_PAGES} pages"
    )
    print(
        f"ingestion loop: {args.facts} facts in {ingest_seconds:.2f} s wall clock "
        f"({sum(writes):.2f} s inside store.add_fact)"
    )
    print(format_summary("writes: store.add_fact", writes))
    print(format_tail(writes))
    if checkpoints is not None:
        print(
            f"  background checkpoints during ingestion: {checkpoints.runs} runs moved "
            f"{checkpoints.frames} WAL frames in {checkpoints.seconds:.2f} s off the write path "
            f"(slowest {checkpoints.slowest * 1000:.1f} ms)"
        )
    print(
        format_summary(
            f"recall: budget {args.budget} tokens, {items / args.queries:.1f} handles avg",
            latencies,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
