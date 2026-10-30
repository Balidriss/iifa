"""The background worker: modes, the gate, budgets and runs (P11).

The harness owns the model's time. Every job runs under a **mode**:

* ``auto`` — the worker thread processes the queue as it fills;
* ``paused`` — work is accepted and persisted, no job runs, no model is called;
* ``on_demand`` — jobs run only inside ``run``, then the worker waits again.

The mode set at runtime is persisted in the store, so a paused service comes
back paused (SPEC 7.2). Every model call, from the thread or from a run, passes
the **gate**: at most ``concurrency`` calls in flight per endpoint (default 1),
and none at all while the harness has said ``yield`` — the call in flight
finishes, the next one waits until ``resume`` or the yield's timeout.

A **run** processes jobs now, in the caller's thread or in a thread of its own,
until the queue is empty or a budget (calls, tokens, seconds, jobs) is reached;
it stops cleanly, puts an interrupted job back untouched, and reports what it
did. At most one run at a time. The current run and the last ``KEPT_RUNS`` are
kept in memory, so any client can follow a run it did not start.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from iifa import log
from iifa.config import BACKGROUND_MODES
from iifa.ids import new_id
from iifa.ingest.extract import PipelineConfig
from iifa.ingest.queue import Job, Queue
from iifa.llm import (
    CHAT,
    EMBEDDINGS,
    TRUNCATED,
    ChatResult,
    EmbeddingsResult,
    ModelClient,
    ModelError,
    ModelWaitTimeout,
)
from iifa.store.repo import Store

MODES = BACKGROUND_MODES
AUTO, PAUSED, ON_DEMAND = MODES
STATE_MODE = "background_mode"

DEFAULT_YIELD_TIMEOUT = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 30.0
DEFAULT_IDLE_POLL = 1.0


class BudgetExhausted(RuntimeError):
    """A run reached one of its limits before the job could make its next call."""

    def __init__(self, limit: str) -> None:
        super().__init__(f"budget exhausted: {limit}")
        self.limit = limit


class RunRefused(RuntimeError):
    """``run`` is not possible now: the service is paused, or a run is in progress
    (``run_id`` names it)."""

    def __init__(self, reason: str, run_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.run_id = run_id


# --------------------------------------------------------------------------- the gate


class Gate:
    """The P11 gate every model call passes: a resizable per-endpoint concurrency
    limit and the harness's yield. Wall-clock by nature: a yield lasts a number
    of real seconds whatever the injectable clock says."""

    def __init__(self, limits: Mapping[str, int]) -> None:
        self._cond = threading.Condition()
        self._limits: dict[str, int] = {}
        self._active: dict[str, int] = {}
        for kind, limit in limits.items():
            self._limits[kind] = _positive(limit, kind)
            self._active[kind] = 0
        self._yield_until: float | None = None
        self._closed = False
        self._foreground = 0
        # test seam: called (lock held) each time a call is about to wait, with
        # the endpoint and why - "yield", "foreground" or "full"
        self._waiting: Callable[[str, str], None] | None = None

    def limits(self) -> dict[str, int]:
        with self._cond:
            return dict(self._limits)

    def set_limit(self, kind: str, limit: int) -> None:
        with self._cond:
            if kind not in self._limits:
                raise KeyError(kind)
            self._limits[kind] = _positive(limit, kind)
            self._cond.notify_all()

    def active(self) -> dict[str, int]:
        with self._cond:
            return dict(self._active)

    def yield_for(self, seconds: float) -> float:
        """No new call starts for ``seconds`` (or until ``resume``). Returns the deadline."""
        if seconds <= 0:
            raise ValueError("yield needs a positive number of seconds")
        with self._cond:
            self._yield_until = time.monotonic() + seconds
            return self._yield_until

    def resume(self) -> None:
        with self._cond:
            self._yield_until = None
            self._cond.notify_all()

    def yield_remaining(self) -> float | None:
        """Seconds left on the current yield, or None when not yielded."""
        with self._cond:
            return self._remaining_locked()

    def _remaining_locked(self) -> float | None:
        if self._yield_until is None:
            return None
        left = self._yield_until - time.monotonic()
        if left <= 0:
            self._yield_until = None
            return None
        return left

    @contextmanager
    def foreground(self) -> Iterator[None]:
        """A foreground model call is in flight (the query embedding, SPEC 6.1): no
        background call starts until it ends. It does not take a slot itself, so
        it never waits behind iifa's own background work."""
        with self._cond:
            self._foreground += 1
        try:
            yield
        finally:
            with self._cond:
                self._foreground -= 1
                self._cond.notify_all()

    def foreground_active(self) -> int:
        with self._cond:
            return self._foreground

    def close(self) -> None:
        """Shutdown: every waiter gives up at once, and nothing passes until ``open``."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def open(self) -> None:
        with self._cond:
            self._closed = False
            self._cond.notify_all()

    def acquire(self, kind: str, *, timeout: float | None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                if self._closed:
                    return False
                paused_for = self._remaining_locked()
                idle = paused_for is None and self._foreground == 0
                if idle and self._active[kind] < self._limits[kind]:
                    self._active[kind] += 1
                    return True
                waits = [w for w in (paused_for,) if w is not None]
                if deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    waits.append(left)
                if self._waiting is not None:
                    self._waiting(kind, self._why_waiting(paused_for))
                self._cond.wait(min(waits) if waits else None)

    def _why_waiting(self, paused_for: float | None) -> str:
        if paused_for is not None:
            return "yield"
        return "foreground" if self._foreground else "full"

    def release(self, kind: str) -> None:
        with self._cond:
            self._active[kind] = max(0, self._active[kind] - 1)
            self._cond.notify_all()


def _positive(value: int, kind: str) -> int:
    if int(value) < 1:
        raise ValueError(f"{kind} concurrency must be at least 1")
    return int(value)


# --------------------------------------------------------------------------- budgets


@dataclass
class Budget:
    """What a run may spend. ``None`` means no limit on that axis."""

    max_calls: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    calls: int = 0
    tokens: int = 0
    started: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("calls", self.max_calls), ("tokens", self.max_tokens)):
            if value is not None and value < 0:
                raise ValueError(f"max_{name} must be non-negative")
        if self.max_seconds is not None and self.max_seconds < 0:
            raise ValueError("max_seconds must be non-negative")

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started)

    def exhausted(self, now: float) -> str | None:
        """The first limit reached, or None."""
        if self.max_calls is not None and self.calls >= self.max_calls:
            return "calls"
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            return "tokens"
        if self.max_seconds is not None and self.elapsed(now) >= self.max_seconds:
            return "seconds"
        return None

    def remaining_seconds(self, now: float) -> float | None:
        if self.max_seconds is None:
            return None
        return max(0.0, self.max_seconds - self.elapsed(now))

    def charge(self, tokens: int) -> None:
        self.calls += 1
        self.tokens += max(0, tokens)


class ModelSession:
    """The model client as a job sees it: bound to a budget and to the gate.
    Every call checks the budget first; a call the budget cannot afford raises
    ``BudgetExhausted`` and the job goes back to the queue untouched."""

    def __init__(
        self, client: ModelClient | None, budget: Budget, monotonic: Callable[[], float]
    ) -> None:
        self._client = client
        self.budget = budget
        self._monotonic = monotonic
        self.last_limit: str | None = None

    def _before(self) -> float | None:
        now = self._monotonic()
        limit = self.budget.exhausted(now)
        if limit is not None:
            self.last_limit = limit
            raise BudgetExhausted(limit)
        return self.budget.remaining_seconds(now)

    def _client_or_fail(self, kind: str) -> ModelClient:
        if self._client is None:
            raise ModelError(kind, "no_endpoint", retryable=False)
        return self._client

    @property
    def has_chat(self) -> bool:
        """Is a chat endpoint configured (tier 2)? Asking never calls it."""
        return self._client is not None and self._client.chat_endpoint is not None

    @property
    def has_embeddings(self) -> bool:
        return self._client is not None and self._client.embeddings_endpoint is not None

    @property
    def embeddings_model(self) -> str | None:
        """The configured embedding model, the stamp of every vector (SPEC 6)."""
        endpoint = self._client.embeddings_endpoint if self._client is not None else None
        return endpoint.model if endpoint is not None else None

    def chat(self, messages: Sequence[Mapping[str, Any]], **params: Any) -> ChatResult:
        client = self._client_or_fail(CHAT)
        endpoint = client.chat_endpoint
        if endpoint is None:
            raise ModelError(CHAT, "no_endpoint", retryable=False)
        wait = self._before()
        try:
            result = client.chat(messages, model=endpoint.model, wait_timeout=wait, **params)
        except ModelWaitTimeout as exc:
            self.last_limit = "seconds"
            raise BudgetExhausted("seconds") from exc
        except ModelError as exc:
            if exc.reason == TRUNCATED:
                # the model spent the whole cap: the call counts, at that many tokens
                self.budget.charge(int(params.get("max_tokens") or 0))
            raise
        self.budget.charge(result.usage.total)
        return result

    def embed(self, inputs: Sequence[str]) -> EmbeddingsResult:
        client = self._client_or_fail(EMBEDDINGS)
        endpoint = client.embeddings_endpoint
        if endpoint is None:
            raise ModelError(EMBEDDINGS, "no_endpoint", retryable=False)
        wait = self._before()
        try:
            result = client.embed(inputs, model=endpoint.model, wait_timeout=wait)
        except ModelWaitTimeout as exc:
            self.last_limit = "seconds"
            raise BudgetExhausted("seconds") from exc
        self.budget.charge(result.usage.total)
        return result


def _never() -> str | None:
    return None


@dataclass
class JobContext:
    store: Store
    queue: Queue
    models: ModelSession
    clock: Callable[[], Any]
    config: PipelineConfig = field(default_factory=PipelineConfig)
    # the dream runner (SPEC 5.6), for the ``dream`` job only
    dream: Any = None
    # why a long job should stop at its next step: "paused", "shutdown", or None
    interrupt: Callable[[], str | None] = _never


Handler = Callable[[Job, JobContext], dict[str, Any] | None]

DONE, FAILED, RETRY, RELEASED = "done", "failed", "retry", "released"
EMBED_KIND = "embed"
EMBED_SCOPE = "system:vectors"  # the embed job works on every record; a queue row needs a scope
# jobs that write embedded text: an embed job follows them (SPEC 6.1)
WRITING_KINDS = frozenset({"turn", "precompact", "session_end", "dream"})


RUN_RUNNING, RUN_DONE = "running", "done"
KEPT_RUNS = 20  # finished runs kept in memory beside the current one


@dataclass
class RunReport:
    id: str
    started_by: str | None = None
    started_at: str | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    status: str = RUN_RUNNING
    finished_at: str | None = None
    jobs: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {DONE: 0, FAILED: 0, RETRY: 0, RELEASED: 0}
    )
    calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    stopped_by: str | None = None
    queue: dict[str, Any] = field(default_factory=dict)
    current_job: str | None = None

    def record(self, job: Job, outcome: str, result: dict[str, Any] | None = None) -> None:
        """``result`` is what the handler returned: counts and short words, never content."""
        self.jobs.append({"id": job.id, "kind": job.kind, "outcome": outcome, "result": result})
        self.counts[outcome] += 1

    @property
    def jobs_done(self) -> int:
        """Jobs this run took to an end (done, failed, or failed for a later retry);
        a job put back at a budget limit is still to do."""
        return sum(n for outcome, n in self.counts.items() if outcome != RELEASED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "status": self.status,
            "started_by": self.started_by,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "budget": dict(self.budget),
            "jobs_done": self.jobs_done,
            "jobs": list(self.jobs),
            "processed": dict(self.counts),
            "calls": self.calls,
            "tokens": self.tokens,
            "seconds": round(self.seconds, 3),
            "stopped_by": self.stopped_by,
            "current_job": self.current_job,
            "queue": dict(self.queue),
        }


def _jobs_left(report: RunReport, pending: int) -> int:
    """What the run may still take: the pending jobs, capped by ``max_jobs``. An
    estimate: turns arriving meanwhile add to it, a budget may stop the run first."""
    max_jobs = report.budget.get("max_jobs")
    if max_jobs is None:
        return pending
    taken = len(report.jobs) + (report.current_job is not None)
    return max(0, min(pending, int(max_jobs) - taken))


# --------------------------------------------------------------------------- the worker


class Worker:
    def __init__(
        self,
        store: Store,
        queue: Queue,
        *,
        mode: str,
        gate: Gate,
        models: ModelClient | None,
        handlers: Mapping[str, Handler],
        yield_timeout: float = DEFAULT_YIELD_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        idle_poll: float = DEFAULT_IDLE_POLL,
        monotonic: Callable[[], float] = time.monotonic,
        pipeline: PipelineConfig | None = None,
        dream: Any = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown background mode {mode!r}")
        self.store = store
        self.queue = queue
        self.gate = gate
        self.models = models
        self.handlers = dict(handlers)
        self.pipeline = pipeline or PipelineConfig()
        self.dream = dream
        self.yield_timeout = yield_timeout
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay = max(0.0, float(retry_delay))
        self.idle_poll = max(0.01, float(idle_poll))
        self._monotonic = monotonic
        self._cond = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._running_job: str | None = None
        # the run in progress (its report and budget; at most one, set and cleared
        # under ``_cond``) and the runs kept in memory
        self._live: tuple[RunReport, Budget] | None = None
        self._runs: dict[str, RunReport] = {}
        self._run_thread: threading.Thread | None = None
        # a test seam: how ``start_run`` hands the run to a thread of its own
        self._spawn: Callable[[Callable[[], None]], None] = self._spawn_thread
        # a test seam between the mode check and the claim (see _step); never set in service
        self._before_claim: Callable[[], None] | None = None
        # a test seam: the loop is about to wait because the mode is not auto
        self._waiting_for_auto: Callable[[], None] | None = None
        self.runs = 0
        persisted = store.get_state(STATE_MODE)
        if persisted in MODES:
            self._mode = str(persisted)
            self.mode_source = "persisted"
        else:
            self._mode = mode
            self.mode_source = "config"
        log.event("background_mode", mode=self._mode, source=self.mode_source)

    # ------------------------------------------------------------------ controls (§8.4)

    @property
    def mode(self) -> str:
        with self._cond:
            return self._mode

    def set_mode(self, mode: str) -> str:
        """Change the mode now and persist it: it survives a restart."""
        if mode not in MODES:
            raise ValueError(f"unknown background mode {mode!r}")
        self.store.set_state(STATE_MODE, mode)
        with self._cond:
            self._mode = mode
            self.mode_source = "runtime"
            self._cond.notify_all()
        log.event("background_mode", mode=mode, source="runtime")
        return mode

    def concurrency(self) -> dict[str, int]:
        return self.gate.limits()

    def set_concurrency(self, kind: str, limit: int) -> None:
        self.gate.set_limit(kind, limit)
        log.event("background_concurrency", endpoint=kind, limit=int(limit))

    def yield_for(self, seconds: float | None = None) -> float:
        """A foreground turn starts: no new model call until ``resume`` or ``seconds``."""
        duration = self.yield_timeout if seconds is None else float(seconds)
        self.gate.yield_for(duration)
        log.event("background_yield", seconds=round(duration, 3))
        return duration

    def resume(self) -> None:
        self.gate.resume()
        with self._cond:
            self._cond.notify_all()
        log.event("background_resume")

    def enqueue(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        scopes: Sequence[str],
        session_id: str | None = None,
    ) -> Job:
        job = self.queue.enqueue(kind, payload, scopes=scopes, session_id=session_id)
        with self._cond:
            self._cond.notify_all()
        log.event("enqueue", job=job.id, kind=kind, seq=job.seq)
        return job

    def status(self) -> dict[str, Any]:
        with self._cond:
            mode = self._mode
            alive = self._thread is not None and self._thread.is_alive()
            running_job = self._running_job
        remaining = self.gate.yield_remaining()
        queue = self.queue.stats()
        run = self._run_summary(queue, yielded=remaining is not None)
        return {
            "mode": mode,
            "mode_source": self.mode_source,
            "concurrency": self.gate.limits(),
            "in_flight": self.gate.active(),
            "yielded": remaining is not None,
            "yield_remaining_seconds": None if remaining is None else round(remaining, 3),
            "worker": "running" if alive else "stopped",
            "current_job": running_job,
            "run_in_progress": run is not None,
            "run": run,
            "runs": self.runs,
            "queue": queue,
        }

    def _run_summary(self, queue: Mapping[str, Any], *, yielded: bool) -> dict[str, Any] | None:
        """The run in progress, as any client may see it: ids and counts, no content."""
        with self._cond:
            if self._live is None:
                return None
            report = self._live[0]
            done = report.jobs_done
            left = _jobs_left(report, int(queue["pending"]))
        return {
            "run_id": report.id,
            "started_by": report.started_by,
            "started_at": report.started_at,
            "budget": dict(report.budget),
            "jobs_done": done,
            "jobs_left": left,
            "yielded": yielded,
        }

    def run_report(self, run_id: str) -> dict[str, Any] | None:
        """A kept run's report, live while it runs; None when it is not kept.
        ``jobs_left`` and ``yielded`` describe a running run and are None after."""
        with self._cond:
            report = self._runs.get(run_id)
            if report is None:
                return None
            if not self._is_live(report):
                return {**report.as_dict(), "jobs_left": None, "yielded": None}
        queue = self.queue.stats()  # only for the live run, outside the lock
        yielded = self.gate.yield_remaining() is not None
        with self._cond:
            data = report.as_dict()
            if not self._is_live(report):  # it ended meanwhile
                return {**data, "jobs_left": None, "yielded": None}
            assert self._live is not None
            budget = self._live[1]
            data.update(
                calls=budget.calls,
                tokens=budget.tokens,
                seconds=round(budget.elapsed(self._monotonic()), 3),
                queue=queue,
                jobs_left=_jobs_left(report, int(queue["pending"])),
                yielded=yielded,
            )
            return data

    def _is_live(self, report: RunReport) -> bool:
        return self._live is not None and self._live[0] is report

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> Worker:
        recovered = self.queue.recover()
        if recovered:
            log.event("queue_recovered", jobs=recovered)
        endpoint = self.models.embeddings_endpoint if self.models is not None else None
        if endpoint is not None and self.store.pending_vectors(endpoint.model, limit=1):
            # a model change, or writes made without an embeddings endpoint
            self.request_embeddings()
        with self._cond:
            self._stop = False
        self.gate.open()
        self._thread = threading.Thread(target=self._loop, name="iifa-worker", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 30.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self.gate.close()
        deadline = time.monotonic() + timeout  # one timeout for both threads
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        run_thread = self._run_thread
        if run_thread is not None and run_thread.is_alive():
            run_thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _loop(self) -> None:
        while self._step() != "stopped":
            pass

    def _step(self, *, wait: bool = True) -> str:
        """One turn of the auto loop: wait for ``auto``, claim the oldest job, run it.

        P11: the decision to run a claimed job is taken under the mode lock, after
        the claim. A mode change that lands between the mode check and the claim
        (``_before_claim`` is the seam a test uses to put it exactly there) sends
        the job back to the queue untouched: once ``set_mode`` returns, no new job
        starts, so a paused service makes no call. A job already running when the
        pause arrives finishes (it is in flight, not new)."""
        with self._cond:
            while wait and not self._stop and self._mode != AUTO:
                if self._waiting_for_auto is not None:
                    self._waiting_for_auto()
                self._cond.wait(self.idle_poll)
            if self._stop:
                return "stopped"
            if self._mode != AUTO:
                return "not_auto"
        if self._before_claim is not None:
            self._before_claim()
        job = self.queue.claim()
        if job is None:
            if wait:
                with self._cond:
                    if not self._stop:
                        self._cond.wait(self.idle_poll)
            return "idle"
        with self._cond:
            go = not self._stop and self._mode == AUTO
            if go:
                self._running_job = job.id
        if not go:
            self.queue.release(job.id)
            log.event("job_released", job=job.id, kind=job.kind, limit="mode")
            return "released"
        session = ModelSession(self.models, Budget(started=self._monotonic()), self._monotonic)
        try:
            self._process(job, session)
        except Exception as exc:  # pragma: no cover - the loop must survive anything
            log.event("worker_error", job=job.id, error=type(exc).__name__)
        return "processed"

    # ------------------------------------------------------------------ run (§8.4)

    def run(
        self,
        budget: Budget | None = None,
        *,
        max_jobs: int | None = None,
        started_by: str | None = None,
    ) -> RunReport:
        """Process now, in this thread, until the queue is empty or a limit is
        reached. Refused while paused (P11: a paused service calls no model) and
        while another run is in progress."""
        report, budget = self._begin(budget, max_jobs, started_by)
        self._execute(report, budget, max_jobs)
        return report

    def start_run(
        self,
        budget: Budget | None = None,
        *,
        max_jobs: int | None = None,
        started_by: str | None = None,
    ) -> RunReport:
        """As ``run``, but in a thread of its own: returns at once with the run's
        report, which fills as it goes (``run_report``). Refused the same way."""
        report, budget = self._begin(budget, max_jobs, started_by)
        try:
            self._spawn(lambda: self._execute(report, budget, max_jobs))
        except BaseException:
            report.stopped_by = "error"
            self._end(report, budget)
            raise
        return report

    def _spawn_thread(self, work: Callable[[], None]) -> None:
        thread = threading.Thread(target=work, name="iifa-run", daemon=True)
        self._run_thread = thread
        thread.start()

    def _begin(
        self, budget: Budget | None, max_jobs: int | None, started_by: str | None
    ) -> tuple[RunReport, Budget]:
        """Record the run as the current one, unless one is in progress. Nothing
        is held before the check-and-set, so a failure here leaks nothing."""
        if self.mode == PAUSED:
            raise RunRefused("paused")
        budget = budget or Budget()
        budget.started = self._monotonic()
        report = RunReport(
            new_id("run"),
            started_by=started_by,
            started_at=self.store.now(),
            budget={
                "calls": budget.max_calls,
                "tokens": budget.max_tokens,
                "seconds": budget.max_seconds,
                "max_jobs": max_jobs,
            },
        )
        with self._cond:
            if self._live is not None:
                raise RunRefused("busy", self._live[0].id)
            self._live = (report, budget)
            self._runs[report.id] = report
        log.event("run_start", run=report.id)
        return report, budget

    def _execute(self, report: RunReport, budget: Budget, max_jobs: int | None) -> None:
        """The run's loop; it always ends the run, whatever happens inside it."""
        try:
            session = ModelSession(self.models, budget, self._monotonic)
            while True:
                limit = budget.exhausted(self._monotonic())
                if limit is not None:
                    report.stopped_by = limit
                    break
                if max_jobs is not None and len(report.jobs) >= max_jobs:
                    report.stopped_by = "max_jobs"
                    break
                with self._cond:
                    stopping = self._stop
                if stopping:
                    report.stopped_by = "shutdown"
                    break
                if self.mode == PAUSED:
                    report.stopped_by = "paused"
                    break
                job = self.queue.claim()
                if job is None:
                    report.stopped_by = "queue_empty"
                    break
                with self._cond:
                    report.current_job = job.id
                outcome, result = self._process(job, session)
                with self._cond:
                    report.current_job = None
                    report.record(job, outcome, result)
                if outcome == RELEASED:
                    # a call held at a gate the shutdown closed is put back too
                    with self._cond:
                        stopping = self._stop
                    report.stopped_by = "shutdown" if stopping else session.last_limit or "seconds"
                    break
        finally:
            if report.stopped_by is None:
                report.stopped_by = "error"
            self._end(report, budget)

    def _end(self, report: RunReport, budget: Budget) -> None:
        """Close the run; it always stops being the current one."""
        queue: dict[str, Any] = {}
        finished: str | None = None
        try:
            queue = self.queue.stats()
            finished = self.store.now()
        except Exception as exc:  # e.g. a store closed under a run at shutdown
            log.event("run_end_error", run=report.id, error=type(exc).__name__)
        with self._cond:
            report.calls = budget.calls
            report.tokens = budget.tokens
            report.seconds = budget.elapsed(self._monotonic())
            report.queue = queue
            report.finished_at = finished
            report.status = RUN_DONE
            self._live = None
            self.runs += 1
            while len(self._runs) > KEPT_RUNS:  # the oldest finished runs go
                del self._runs[next(iter(self._runs))]
        log.event(
            "run_end",
            run=report.id,
            jobs=len(report.jobs),
            calls=report.calls,
            tokens=report.tokens,
            stopped_by=report.stopped_by,
        )

    # ------------------------------------------------------------------ one job

    def _process(self, job: Job, session: ModelSession) -> tuple[str, dict[str, Any] | None]:
        """Run one job: its outcome and, when it completed, what the handler returned."""
        handler = self.handlers.get(job.kind)
        if handler is None:
            self.queue.fail(job.id, f"unknown_kind:{job.kind}"[:120], retry_at=None)
            log.event("job_failed", job=job.id, kind=job.kind, reason="unknown_kind")
            return FAILED, None
        with self._cond:
            self._running_job = job.id
        try:
            context = JobContext(
                self.store,
                self.queue,
                session,
                self.store.clock,
                self.pipeline,
                dream=self.dream,
                interrupt=self.interrupted,
            )
            result = handler(job, context)
        except BudgetExhausted as exc:
            self.queue.release(job.id)
            log.event("job_released", job=job.id, kind=job.kind, limit=exc.limit)
            return RELEASED, None
        except ModelError as exc:
            return self._failed(job, exc.reason, retryable=exc.retryable), None
        except Exception as exc:  # a bug in a handler never kills the worker
            return self._failed(job, f"error:{type(exc).__name__}", retryable=False), None
        finally:
            with self._cond:
                self._running_job = None
        self.queue.complete(job.id)
        log.event("job_done", job=job.id, kind=job.kind)
        if job.kind in WRITING_KINDS:
            self.request_embeddings(job.scopes)
        return DONE, result

    def interrupted(self) -> str | None:
        """Why a long job (a dream) should stop at its next step (P11): the service
        was paused or is shutting down. A job in flight is never cut mid-call."""
        with self._cond:
            if self._stop:
                return "shutdown"
            if self._mode == PAUSED:
                return "paused"
            return None

    def request_embeddings(self, scopes: Sequence[str] = ()) -> bool:
        """Queue the background ``embed`` job (SPEC 6.1) after a write, unless one is
        already open or no embeddings endpoint is configured. Enqueue only."""
        if self.models is None or self.models.embeddings_endpoint is None:
            return False
        if self.queue.has_open(EMBED_KIND):
            return False
        self.queue.enqueue(EMBED_KIND, {}, scopes=list(scopes) or [EMBED_SCOPE])
        return True

    def _failed(self, job: Job, reason: str, *, retryable: bool) -> str:
        attempts = job.attempts + 1
        if retryable and attempts < self.max_attempts:
            delay = timedelta(seconds=self.retry_delay * attempts)
            retry_at = self.store.now_plus(delay)
            self.queue.fail(job.id, reason[:120], retry_at=retry_at)
            log.event("job_retry", job=job.id, kind=job.kind, reason=reason[:60], attempt=attempts)
            return RETRY
        self.queue.fail(job.id, reason[:120], retry_at=None)
        log.event("job_failed", job=job.id, kind=job.kind, reason=reason[:60], attempt=attempts)
        return FAILED
