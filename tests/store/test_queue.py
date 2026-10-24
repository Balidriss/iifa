"""The persisted ingestion queue (SPEC 7.2): rows survive a restart, a claim is
atomic, an interrupted job runs again, a done job keeps no content (P5)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from iifa.ingest.queue import DONE, FAILED, KEEP_DONE, PENDING, RUNNING, Queue
from iifa.store.repo import NotFound, Store
from tests.conftest import FakeClock
from tests.fixtures.corpus import CANARY
from tests.helpers import KEY


def test_enqueue_claim_complete_round_trip(store: Store) -> None:
    queue = Queue(store)
    first = queue.enqueue(
        "turn", {"user": "hi"}, scopes=["user:alice", "user:alice"], session_id="s1"
    )
    second = queue.enqueue("noop", {}, scopes=["user:bob"])
    assert first.id.startswith("job_") and first.status == PENDING and first.seq < second.seq
    assert first.scopes == ("user:alice",) and first.session_id == "s1"
    assert queue.stats() == {
        "depth": 2,
        "pending": 2,
        "running": 0,
        "failed": 0,
        "done": 0,
        "oldest_pending": first.created_at,
    }
    claimed = queue.claim()
    assert claimed is not None and claimed.id == first.id  # oldest first
    assert claimed.status == RUNNING and claimed.started_at is not None
    assert claimed.payload == {"user": "hi"}
    assert queue.stats()["depth"] == 2 and queue.stats()["running"] == 1
    done = queue.complete(first.id)
    assert done.status == DONE and done.finished_at and done.payload == {}
    assert queue.stats()["oldest_pending"] == second.created_at
    assert [j.id for j in queue.jobs(status=PENDING)] == [second.id]
    assert queue.claim() is not None and queue.claim() is None


def test_enqueue_validates_its_input(store: Store) -> None:
    queue = Queue(store)
    with pytest.raises(ValueError, match="scope"):
        queue.enqueue("turn", {}, scopes=[])
    with pytest.raises(ValueError, match="kind"):
        queue.enqueue("", {}, scopes=["user:alice"])
    with pytest.raises(NotFound):
        queue.get("job_" + "0" * 26)
    with pytest.raises(ValueError, match="never content"):
        queue.fail(queue.enqueue("noop", {}, scopes=["user:alice"]).id, "x" * 121, retry_at=None)


def test_queue_survives_a_restart_and_recovers_interrupted_jobs(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "q.db"
    with Store(path, KEY, clock=clock) as store:
        queue = Queue(store)
        kept = queue.enqueue("turn", {"user": "remember this"}, scopes=["user:alice"])
        interrupted = queue.enqueue("turn", {"user": "was running"}, scopes=["user:alice"])
        queue.complete(queue.enqueue("noop", {}, scopes=["user:alice"]).id)
        assert queue.claim() is not None  # kept -> running
        assert queue.claim() is not None  # interrupted -> running
        # the process dies here: no complete, no release
    with Store(path, KEY, clock=clock) as store:
        queue = Queue(store)
        assert queue.stats()["running"] == 2
        assert queue.recover() == 2
        pending = queue.jobs(status=PENDING)
        assert [j.id for j in pending] == [kept.id, interrupted.id]
        assert pending[0].payload == {"user": "remember this"}
        assert all(j.started_at is None for j in pending)
        assert queue.recover() == 0


def test_fail_retries_until_told_otherwise_and_claim_waits_for_not_before(
    store: Store, clock: FakeClock
) -> None:
    queue = Queue(store)
    job = queue.enqueue("model_noop", {"calls": 1}, scopes=["user:alice"])
    assert queue.claim() is not None
    later = store.now_plus(timedelta(seconds=30))
    retried = queue.fail(job.id, "http_503", retry_at=later)
    assert retried.status == PENDING and retried.attempts == 1 and retried.error == "http_503"
    assert retried.not_before == later and retried.payload == {"calls": 1}
    assert queue.claim() is None  # not yet
    clock.advance(seconds=31)
    claimed = queue.claim()
    assert claimed is not None and claimed.id == job.id and claimed.not_before is None
    failed = queue.fail(job.id, "http_400", retry_at=None)
    assert failed.status == FAILED and failed.attempts == 2 and failed.finished_at
    assert failed.payload == {"calls": 1}  # kept for an operator's retry
    assert queue.claim() is None
    assert queue.stats()["failed"] == 1 and queue.stats()["depth"] == 0
    again = queue.retry(job.id)
    assert again.status == PENDING and again.error is None
    with pytest.raises(ValueError, match="failed"):
        queue.retry(job.id)


def test_release_puts_a_job_back_untouched(store: Store) -> None:
    queue = Queue(store)
    job = queue.enqueue("model_noop", {"calls": 3}, scopes=["user:alice"])
    assert queue.claim() is not None
    released = queue.release(job.id)
    assert released.status == PENDING and released.attempts == 0 and released.started_at is None
    assert released.payload == {"calls": 3}
    # releasing a pending job is a no-op
    assert queue.release(job.id).status == PENDING


def test_done_jobs_keep_no_content_and_are_pruned(tmp_path: Path) -> None:
    path = tmp_path / "plain.db"
    with Store(path, "", plaintext_for_tests=True, checkpoint_interval=0) as store:
        queue = Queue(store)
        job = queue.enqueue("turn", {"user": f"secret {CANARY}"}, scopes=["user:alice"])
        store.checkpoint("TRUNCATE")
        assert CANARY.encode() in path.read_bytes()
        assert queue.claim() is not None
        queue.complete(job.id)
        assert queue.get(job.id).payload == {}
        for _ in range(KEEP_DONE + 5):
            queue.complete(queue.enqueue("noop", {}, scopes=["user:alice"]).id)
        assert queue.stats()["done"] == KEEP_DONE
        with pytest.raises(NotFound):
            queue.get(job.id)  # the oldest done rows are gone
        store.checkpoint("TRUNCATE")
    assert CANARY.encode() not in path.read_bytes()


def test_service_state_round_trip(store: Store) -> None:
    assert store.get_state("background_mode") is None
    store.set_state("background_mode", "paused")
    store.set_state("background_mode", "on_demand")
    assert store.get_state("background_mode") == "on_demand"
    assert store.counts()["queue"] == 0
