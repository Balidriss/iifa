"""The benchmark reports the whole distribution of a write, not a p95 that hides
the tail (10k writes took 144 s on Windows at a reported p95 of 1.2 ms)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

BENCH = Path(__file__).resolve().parents[2] / "bench" / "recall_latency.py"


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recall_latency_bench", BENCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_p99_max_mean_and_total(bench: ModuleType) -> None:
    values = [0.001] * 98 + [1.0, 1.0]  # a p95 of 1 ms hides two seconds
    summary = bench.summarize(values)
    assert summary["p95"] == pytest.approx(0.001)
    assert summary["p99"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(1.0)
    assert summary["mean"] == pytest.approx(2.098 / 100)
    assert summary["total"] == pytest.approx(2.098)
    assert summary["count"] == 100
    line = bench.format_summary("writes", values)
    for word in ("p50", "p95", "p99", "max", "mean", "total"):
        assert word in line


def test_tail_names_the_slow_calls_and_their_share(bench: ModuleType) -> None:
    values = [0.001] * 98 + [1.0, 1.0]
    tail = bench.tail(values)
    assert tail["count"] == 2
    assert tail["threshold"] == pytest.approx(0.01)
    assert tail["share"] == pytest.approx(2.0 / 2.098)
    assert "%" in bench.format_tail(values)


def test_summary_refuses_no_values(bench: ModuleType) -> None:
    with pytest.raises(ValueError):
        bench.summarize([])
