"""The background worker (P11) at the object level: the gate, budgets, the three
modes, runs and their budgets, retries, recovery and shutdown. The fake model
server's call log is the witness for every "no model call" claim."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from iifa.ingest.jobs import HANDLERS, MODEL_NOOP, NOOP
from iifa.ingest.queue import DONE, FAILED, PENDING, RUNNING, Job, Queue
from iifa.ingest.worker import (
    AUTO,
    KEPT_RUNS,
    ON_DEMAND,
    PAUSED,
    STATE_MODE,
    Budget,
    BudgetExhausted,
    Gate,
    JobContext,
    ModelSession,
    RunRefused,
    Worker,
)
from iifa.llm import CHAT, EMBEDDINGS, Endpoint, ModelClient
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import Fail, FakeOpenAIServer, Reply
from tests.helpers import KEY, wait_until

SCOPES = ["user:alice"]


# --------------------------------------------------------------------------- the gate


def test_gate_limits_calls_per_endpoint() -> None:
    gate = Gate({CHAT: 1, EMBEDDINGS: 2})
    assert gate.acquire(CHAT, timeout=0) and not gate.acquire(CHAT, timeout=0.02)
    assert gate.acquire(EMBEDDINGS, timeout=0) and gate.acquire(EMBEDDINGS, timeout=0)
    assert not gate.acquire(EMBEDDINGS, timeout=0)
    gate.release(CHAT)
    assert gate.acquire(CHAT, timeout=0)
    assert gate.active() == {CHAT: 1, EMBEDDINGS: 2}
    gate.set_limit(CHAT, 2)
    assert gate.acquire(CHAT, timeout=0) and gate.limits() == {CHAT: 2, EMBEDDINGS: 2}
    with pytest.raises(ValueError, match="at least 1"):
        gate.set_limit(CHAT, 0)
    with pytest.raises(KeyError):
        gate.set_limit("vision", 1)
    with pytest.raises(ValueError):
        Gate({CHAT: 0})
    gate.release(CHAT)
    gate.release(CHAT)
    gate.release(CHAT)  # over-release is harmless
    assert gate.active()[CHAT] == 0


def test_gate_yield_blocks_until_resume_or_timeout() -> None:
    gate = Gate({CHAT: 1})
    gate.yield_for(5.0)
    assert gate.yield_remaining() is not None
    assert not gate.acquire(CHAT, timeout=0.05)
    acquired: list[float] = []

    def waiter() -> None:
        started = time.monotonic()
        assert gate.acquire(CHAT, timeout=5.0)
        acquired.append(time.monotonic() - started)

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.1)
    assert acquired == []  # still yielded
    gate.resume()
    thread.join(timeout=5)
    assert acquired and acquired[0] < 2.0
    assert gate.yield_remaining() is None
    gate.release(CHAT)
    gate.yield_for(0.1)
    started = time.monotonic()
    assert gate.acquire(CHAT, timeout=5.0)  # the yield expires on its own
    assert 0.05 < time.monotonic() - started < 2.0
    with pytest.raises(ValueError):
        gate.yield_for(0)


def test_gate_close_wakes_every_waiter() -> None:
    gate = Gate({CHAT: 1})
    assert gate.acquire(CHAT, timeout=0)
    results: list[bool] = []
    thread = threading.Thread(target=lambda: results.append(gate.acquire(CHAT, timeout=None)))
    thread.start()
    time.sleep(0.05)
    gate.close()
    thread.join(timeout=5)
    assert results == [False]
    assert not gate.acquire(CHAT, timeout=0)  # closed stays closed
    gate.open()
    gate.release(CHAT)
    assert gate.acquire(CHAT, timeout=0)


# --------------------------------------------------------------------------- budgets


def test_budget_limits_in_order() -> None:
    budget = Budget(max_calls=2, max_tokens=100, max_seconds=10.0, started=100.0)
    assert budget.exhausted(100.0) is None and budget.remaining_seconds(105.0) == 5.0
    budget.charge(60)
    budget.charge(60)
    assert budget.exhausted(100.0) == "calls"
    budget.max_calls = None
    assert budget.exhausted(100.0) == "tokens" and budget.tokens == 120
    budget.max_tokens = None
    assert budget.exhausted(109.9) is None and budget.exhausted(110.0) == "seconds"
    assert budget.remaining_seconds(200.0) == 0.0
    assert Budget().exhausted(1e9) is None and Budget().remaining_seconds(0.0) is None
    with pytest.raises(ValueError):
        Budget(max_calls=-1)
    with pytest.raises(ValueError):
        Budget(max_seconds=-0.1)


def test_model_session_refuses_a_call_the_budget_cannot_afford(
    fake_openai: FakeOpenAIServer,
) -> None:
    client = ModelClient(Endpoint(CHAT, fake_openai.base_url, "fake-chat"))
    session = ModelSession(client, Budget(max_calls=1, started=0.0), lambda: 0.0)
    session.chat([{"role": "user", "content": "one"}])
    with pytest.raises(BudgetExhausted) as stopped:
        session.chat([{"role": "user", "content": "two"}])
    assert stopped.value.limit == "calls" and session.last_limit == "calls"
    assert len(fake_openai.chat_calls) == 1
    none = ModelSession(None, Budget(), lambda: 0.0)
    with pytest.raises(Exception, match="no_endpoint"):
        none.chat([{"role": "user", "content": "x"}])
    with pytest.raises(Exception, match="no_endpoint"):
        ModelSession(client, Budget(), lambda: 0.0).embed(["x"])
    client.close()


# --------------------------------------------------------------------------- the worker


@dataclass
class Harness:
    store: Store
    queue: Queue
    worker: Worker
    fake: FakeOpenAIServer
    clock: FakeClock

    def enqueue(self, kind: str = MODEL_NOOP, **payload: Any) -> Job:
        return self.worker.enqueue(kind, payload, scopes=SCOPES)

    def status_of(self, job: Job) -> str:
        return self.queue.get(job.id).status

    def wait_done(self, *jobs: Job, timeout: float = 5.0) -> None:
        assert wait_until(lambda: all(self.status_of(j) == DONE for j in jobs), timeout=timeout), [
            self.queue.get(j.id) for j in jobs
        ]


def make_worker(
    store: Store,
    fake: FakeOpenAIServer,
    *,
    mode: str = AUTO,
    monotonic: Any = time.monotonic,
    max_attempts: int = 3,
    retry_delay: float = 30.0,
    handlers: dict[str, Any] | None = None,
    gate: Gate | None = None,
) -> Worker:
    gate = gate or Gate({CHAT: 1, EMBEDDINGS: 1})
    models = ModelClient(
        Endpoint(CHAT, fake.base_url, "fake-chat", timeout=5.0),
        Endpoint(EMBEDDINGS, fake.base_url, "fake-embed", timeout=5.0),
        gate=gate,
    )
    return Worker(
        store,
        Queue(store),
        mode=mode,
        gate=gate,
        models=models,
        handlers=handlers if handlers is not None else HANDLERS,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        idle_poll=0.02,
        monotonic=monotonic,
    )


@pytest.fixture
def harness(tmp_path: Path, clock: FakeClock, fake_openai: FakeOpenAIServer) -> Iterator[Harness]:
    store = Store(tmp_path / "w.db", KEY, clock=clock)
    worker = make_worker(store, fake_openai)
    yield Harness(store, worker.queue, worker, fake_openai, clock)
    worker.stop()
    assert worker.models is not None
    worker.models.close()
    store.close()


def test_auto_mode_processes_the_queue_as_it_fills(harness: Harness) -> None:
    harness.worker.start()
    assert harness.worker.status()["worker"] == "running"
    first = harness.enqueue(calls=2)
    second = harness.enqueue(endpoint=EMBEDDINGS)
    third = harness.enqueue(NOOP)
    harness.wait_done(first, second, third)
    assert len(harness.fake.chat_calls) == 2 and len(harness.fake.embedding_calls) == 1
    assert harness.fake.chat_calls[0].messages == [{"role": "user", "content": "ping 1/2"}]
    assert {c.model for c in harness.fake.calls} == {"fake-chat", "fake-embed"}
    status = harness.worker.status()
    assert status["queue"]["done"] == 3 and status["queue"]["depth"] == 0
    assert status["current_job"] is None and status["mode_source"] == "config"


def test_paused_accepts_work_calls_nothing_and_comes_back_paused(harness: Harness) -> None:
    harness.worker.start()
    harness.worker.set_mode(PAUSED)
    job = harness.enqueue()
    time.sleep(0.2)
    assert harness.fake.calls == [] and harness.status_of(job) == PENDING
    assert harness.worker.status()["queue"]["oldest_pending"] == job.created_at
    with pytest.raises(RunRefused, match="paused"):
        harness.worker.run()
    assert harness.fake.calls == []
    assert harness.store.get_state(STATE_MODE) == PAUSED
    # a restart with the configuration saying "auto" still comes back paused
    again = make_worker(harness.store, harness.fake, mode=AUTO)
    assert again.mode == PAUSED and again.mode_source == "persisted"
    assert again.queue.stats()["depth"] == 1
    harness.worker.set_mode(AUTO)
    harness.wait_done(job)
    assert len(harness.fake.chat_calls) == 1
    with pytest.raises(ValueError, match="unknown background mode"):
        harness.worker.set_mode("sometimes")


def test_on_demand_calls_the_model_only_inside_a_run(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    harness.worker.start()
    jobs = [harness.enqueue() for _ in range(3)]
    time.sleep(0.2)
    assert harness.fake.calls == [] and all(harness.status_of(j) == PENDING for j in jobs)
    report = harness.worker.run(Budget(max_calls=2))
    assert report.id.startswith("run_") and report.stopped_by == "calls"
    assert report.calls == 2 and report.tokens > 0 and report.seconds >= 0
    assert [j["outcome"] for j in report.jobs] == [DONE, DONE]
    assert report.counts == {DONE: 2, FAILED: 0, "retry": 0, "released": 0}
    assert report.queue["depth"] == 1 and len(harness.fake.chat_calls) == 2
    time.sleep(0.1)
    assert len(harness.fake.chat_calls) == 2  # nothing outside the run
    second = harness.worker.run()
    assert second.stopped_by == "queue_empty" and second.calls == 1
    assert harness.worker.status()["runs"] == 2
    assert all(harness.status_of(j) == DONE for j in jobs)


def test_a_run_stops_inside_a_job_and_a_later_run_resumes_it(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    job = harness.enqueue(calls=3)
    report = harness.worker.run(Budget(max_calls=2))
    assert report.stopped_by == "calls" and report.counts["released"] == 1
    assert report.jobs == [
        {"id": job.id, "kind": MODEL_NOOP, "outcome": "released", "result": None}
    ]
    released = harness.queue.get(job.id)
    assert released.status == PENDING and released.attempts == 0
    assert len(harness.fake.chat_calls) == 2
    again = harness.worker.run(Budget(max_calls=10))
    assert again.stopped_by == "queue_empty" and again.calls == 3
    assert harness.status_of(job) == DONE and len(harness.fake.chat_calls) == 5


def test_run_budgets_tokens_seconds_and_jobs(
    tmp_path: Path, clock: FakeClock, fake_openai: FakeOpenAIServer
) -> None:
    ticks = iter(range(0, 100_000))
    store = Store(tmp_path / "b.db", KEY, clock=clock)
    worker = make_worker(store, fake_openai, mode=ON_DEMAND, monotonic=lambda: next(ticks) * 0.4)
    try:
        jobs = [worker.enqueue(MODEL_NOOP, {}, scopes=SCOPES) for _ in range(6)]
        by_tokens = worker.run(Budget(max_tokens=1))
        assert by_tokens.stopped_by == "tokens" and by_tokens.calls == 1 and by_tokens.tokens >= 1
        by_jobs = worker.run(Budget(), max_jobs=2)
        assert by_jobs.stopped_by == "max_jobs" and by_jobs.calls == 2
        by_seconds = worker.run(Budget(max_seconds=1.0))
        assert by_seconds.stopped_by == "seconds"
        assert 0 < by_seconds.calls < 3 and by_seconds.seconds >= 1.0
        rest = worker.run()
        assert rest.stopped_by == "queue_empty"
        assert all(worker.queue.get(j.id).status == DONE for j in jobs)
        assert len(fake_openai.chat_calls) == 6
    finally:
        worker.stop()
        store.close()


def test_run_is_refused_while_another_run_is_in_progress(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    held: list[Any] = []
    harness.worker._spawn = held.append
    first = harness.worker.start_run()
    assert harness.worker.status()["run_in_progress"] is True
    with pytest.raises(RunRefused, match="busy") as refused:
        harness.worker.run()
    assert refused.value.run_id == first.id
    held[0]()
    assert harness.worker.status()["run_in_progress"] is False


def test_a_run_that_cannot_be_recorded_leaves_no_run_behind(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    real_now = harness.store.now
    harness.store.now = lambda: (_ for _ in ()).throw(RuntimeError("closed"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        harness.worker.run()
    harness.store.now = real_now  # type: ignore[method-assign]
    assert harness.worker.status()["run"] is None
    assert harness.worker.run().stopped_by == "queue_empty"


def test_a_run_ends_even_when_its_closing_read_fails(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    real_stats = harness.queue.stats
    calls = {"n": 0}

    def stats_once_broken() -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("closed")
        return real_stats()

    harness.queue.stats = stats_once_broken  # type: ignore[method-assign]
    report = harness.worker.run()
    assert report.status == "done" and report.queue == {} and report.finished_at is None
    assert harness.worker.status()["run"] is None
    assert harness.worker.run().stopped_by == "queue_empty"


def test_a_shutdown_while_a_run_waits_at_the_gate_says_shutdown(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    worker = harness.worker

    def stopped_then_calls(job: Job, ctx: JobContext) -> None:
        worker.stop()  # closes the gate: the next call cannot start
        ctx.models.chat([{"role": "user", "content": "x"}], max_tokens=1)

    worker.handlers["stop_then_call"] = stopped_then_calls
    job = harness.enqueue("stop_then_call")
    report = worker.run()
    assert report.stopped_by == "shutdown" and report.counts["released"] == 1
    assert harness.status_of(job) == PENDING and harness.fake.calls == []


def test_an_async_run_stops_at_shutdown_and_ends_cleanly(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    job = harness.enqueue()
    held: list[Any] = []
    harness.worker._spawn = held.append
    report = harness.worker.start_run(started_by="host")
    assert harness.worker.status()["run"]["run_id"] == report.id
    harness.worker.stop()
    held[0]()  # the run's thread, after the stop
    assert report.stopped_by == "shutdown" and report.status == "done" and report.jobs == []
    assert harness.status_of(job) == PENDING and harness.fake.calls == []
    assert harness.worker.status()["run"] is None
    assert harness.worker.status()["run_in_progress"] is False


def test_an_async_run_runs_in_its_own_thread(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    job = harness.enqueue(NOOP)
    report = harness.worker.start_run()
    thread = harness.worker._run_thread
    assert thread is not None and thread.name == "iifa-run"
    thread.join(timeout=30)
    assert report.status == "done" and report.stopped_by == "queue_empty"
    assert harness.status_of(job) == DONE


def test_a_run_whose_thread_cannot_start_is_ended(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)

    def broken(work: Any) -> None:
        raise RuntimeError("can't start new thread")

    harness.worker._spawn = broken
    with pytest.raises(RuntimeError):
        harness.worker.start_run()
    assert harness.worker.status()["run"] is None
    assert harness.worker.run().stopped_by == "queue_empty"  # the lock was released


def test_finished_runs_kept_in_memory_are_bounded(harness: Harness) -> None:
    harness.worker.set_mode(ON_DEMAND)
    reports = [harness.worker.run() for _ in range(KEPT_RUNS + 2)]
    assert harness.worker.run_report(reports[0].id) is None
    assert harness.worker.run_report(reports[1].id) is None
    kept = harness.worker.run_report(reports[2].id)
    assert kept is not None and kept["status"] == "done"
    assert harness.worker.run_report(reports[-1].id) is not None


def test_transient_failures_retry_later_and_give_up_after_max_attempts(
    tmp_path: Path, clock: FakeClock, fake_openai: FakeOpenAIServer
) -> None:
    store = Store(tmp_path / "r.db", KEY, clock=clock)
    worker = make_worker(store, fake_openai, mode=ON_DEMAND, max_attempts=3, retry_delay=10.0)
    try:
        fake_openai.script(chat=[Fail(503), Fail(503), Fail(503), Fail(400)])
        job = worker.enqueue(MODEL_NOOP, {}, scopes=SCOPES)
        assert worker.run().counts["retry"] == 1
        after_first = worker.queue.get(job.id)
        assert after_first.status == PENDING and after_first.attempts == 1
        assert after_first.error == "http_503"
        assert after_first.not_before == store.now_plus(timedelta(seconds=10))
        assert worker.run().stopped_by == "queue_empty"  # not before its time
        clock.advance(seconds=11)
        assert worker.run().counts["retry"] == 1
        assert worker.queue.get(job.id).not_before == store.now_plus(timedelta(seconds=20))
        clock.advance(seconds=21)
        assert worker.run().counts[FAILED] == 1  # third attempt: give up
        failed = worker.queue.get(job.id)
        assert failed.status == FAILED and failed.attempts == 3 and failed.payload == {}
        assert len(fake_openai.chat_calls) == 3
        # a non-retryable failure gives up at once
        other = worker.enqueue(MODEL_NOOP, {}, scopes=SCOPES)
        assert worker.run().counts[FAILED] == 1
        assert worker.queue.get(other.id).error == "http_400"
        assert worker.queue.get(other.id).attempts == 1
        assert worker.status()["queue"]["failed"] == 2
    finally:
        worker.stop()
        store.close()


def test_unknown_kinds_and_handler_bugs_fail_the_job_not_the_worker(
    tmp_path: Path, clock: FakeClock, fake_openai: FakeOpenAIServer
) -> None:
    def broken(job: Job, ctx: JobContext) -> None:
        raise ValueError("bug")

    def needs_a_model(job: Job, ctx: JobContext) -> None:
        ctx.models.chat([{"role": "user", "content": "x"}])

    store = Store(tmp_path / "h.db", KEY, clock=clock)
    worker = make_worker(
        store,
        fake_openai,
        handlers={"broken": broken, "later": needs_a_model, NOOP: HANDLERS[NOOP]},
    )
    try:
        worker.start()
        bogus = worker.enqueue("bogus", {}, scopes=SCOPES)
        crash = worker.enqueue("broken", {}, scopes=SCOPES)
        fine = worker.enqueue(NOOP, {}, scopes=SCOPES)
        assert wait_until(lambda: worker.queue.get(fine.id).status == DONE)
        assert worker.queue.get(bogus.id).error == "unknown_kind:bogus"
        assert worker.queue.get(crash.id).error == "error:ValueError"
        assert worker.status()["worker"] == "running"
        assert fake_openai.calls == []
    finally:
        worker.stop()
        store.close()


def test_yield_holds_new_calls_until_resume_or_timeout(harness: Harness) -> None:
    harness.worker.start()
    harness.worker.yield_for(5.0)
    assert harness.worker.status()["yielded"] is True
    job = harness.enqueue()
    time.sleep(0.2)
    assert harness.fake.calls == []
    assert harness.status_of(job) == RUNNING  # claimed, waiting at the gate
    assert harness.worker.status()["current_job"] == job.id
    harness.worker.resume()
    harness.wait_done(job)
    assert len(harness.fake.chat_calls) == 1
    harness.worker.yield_for(0.2)
    timed = harness.enqueue()
    harness.wait_done(timed)  # the yield expired on its own
    assert len(harness.fake.chat_calls) == 2
    assert harness.worker.status()["yielded"] is False


def test_a_call_in_flight_finishes_and_the_next_one_waits(harness: Harness) -> None:
    harness.fake.set_default(chat=Reply("slow", delay=0.3))
    harness.worker.start()
    job = harness.enqueue(calls=2)
    assert wait_until(lambda: len(harness.fake.chat_calls) == 1)
    harness.worker.yield_for(5.0)  # arrives while the first call is in flight
    time.sleep(0.6)
    assert len(harness.fake.chat_calls) == 1  # the first finished, the second waited
    assert harness.fake.chat_calls[0].status == 200
    harness.worker.resume()
    harness.wait_done(job)
    assert len(harness.fake.chat_calls) == 2


def test_concurrency_one_never_overlaps_calls_from_the_thread_and_a_run(
    harness: Harness,
) -> None:
    harness.fake.set_default(chat=Reply("slow", delay=0.25))
    harness.worker.start()
    jobs = [harness.enqueue() for _ in range(3)]
    report = harness.worker.run()  # competes with the thread for the same queue
    harness.wait_done(*jobs)
    starts = sorted(c.started_at for c in harness.fake.chat_calls)
    assert len(starts) == 3 and report.calls >= 1
    assert all(b - a >= 0.2 for a, b in pairwise(starts))


def test_concurrency_two_lets_calls_overlap(
    tmp_path: Path, clock: FakeClock, fake_openai: FakeOpenAIServer
) -> None:
    store = Store(tmp_path / "c.db", KEY, clock=clock)
    worker = make_worker(store, fake_openai, gate=Gate({CHAT: 2, EMBEDDINGS: 1}))
    try:
        fake_openai.set_default(chat=Reply("slow", delay=0.3))
        worker.start()
        jobs = [worker.enqueue(MODEL_NOOP, {}, scopes=SCOPES) for _ in range(2)]
        worker.run(max_jobs=1)
        assert wait_until(lambda: all(worker.queue.get(j.id).status == DONE for j in jobs))
        starts = sorted(c.started_at for c in fake_openai.chat_calls)
        assert len(starts) == 2 and starts[1] - starts[0] < 0.25
        worker.set_concurrency(CHAT, 1)
        assert worker.concurrency() == {CHAT: 1, EMBEDDINGS: 1}
    finally:
        worker.stop()
        store.close()


def test_start_recovers_and_stop_releases_the_current_job(harness: Harness) -> None:
    orphan = harness.enqueue()
    assert harness.queue.claim() is not None  # as if a previous process died here
    harness.worker.yield_for(30.0)
    harness.worker.start()
    assert wait_until(lambda: harness.worker.status()["current_job"] == orphan.id)
    assert harness.fake.calls == []
    harness.worker.stop()
    assert harness.worker.status()["worker"] == "stopped"
    assert harness.status_of(orphan) == PENDING and harness.queue.get(orphan.id).attempts == 0
    assert harness.fake.calls == []  # loses nothing, calls nothing
    harness.worker.resume()
    harness.worker.start()  # the same worker again: the gate is open once more
    harness.wait_done(orphan)
    assert len(harness.fake.chat_calls) == 1
