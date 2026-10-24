"""SPEC 7.4: the turn ingestion call only enqueues and answers in under 10 ms.
Measured at the queue (one insert) and through the route in-process."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iifa.ingest.queue import Queue
from iifa.store.repo import Store
from tests.fixtures.corpus import statements

pytestmark = pytest.mark.perf

TURNS = 200


def test_enqueue_p95_under_10_ms(tmp_path: Path) -> None:
    with Store(tmp_path / "q.db", "k") as store:
        queue = Queue(store)
        texts = statements(TURNS)
        latencies: list[float] = []
        for i, text in enumerate(texts):
            started = time.perf_counter()
            queue.enqueue(
                "turn", {"user": text, "assistant": text * 3}, scopes=["user:perf"], session_id="s"
            )
            latencies.append(time.perf_counter() - started)
            _ = i
        assert queue.depth() == TURNS
    p95 = statistics.quantiles(latencies, n=20)[18]
    assert p95 < 0.010, f"enqueue p95 {p95 * 1000:.2f} ms"


def test_turn_route_answers_in_milliseconds(client: TestClient) -> None:
    """The call through the app in-process: p50 under 10 ms, and a tail bound loose
    enough for a shared CI runner (the queue insert above is the real cost)."""
    client.put("/v1/background", json={"mode": "paused"})  # measure the enqueue alone
    latencies: list[float] = []
    for text in statements(100):
        started = time.perf_counter()
        response = client.post(
            "/v1/turns", json={"scope": "user:alice", "user": text, "assistant": text * 3}
        )
        latencies.append(time.perf_counter() - started)
        assert response.status_code == 202
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]
    assert p50 < 0.010, f"POST /v1/turns p50 {p50 * 1000:.2f} ms"
    assert p95 < 0.025, f"POST /v1/turns p95 {p95 * 1000:.2f} ms"
