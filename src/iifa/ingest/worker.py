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

A **run** processes jobs now, in the caller's thread, until the queue is empty
or a budget (calls, tokens, seconds, jobs) is reached; it stops cleanly, puts an
interrupted job back untouched, and reports what it did.
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
    """``run`` is not possible now: the service is paused, or a run is in progress."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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


@dataclass
class RunReport:
    id: str
    jobs: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(
        default_factory=lambda: {DONE: 0, FAILED: 0, RETRY: 0, RELEASED: 0}
    )
    calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    stopped_by: str = "queue_empty"
    queue: dict[str, Any] = field(default_factory=dict)

    def record(self, job: Job, outcome: str, result: dict[str, Any] | None = None) -> None:
        """``result`` is what the handler returned: counts and short words, never content."""
        self.jobs.append({"id": job.id, "kind": job.kind, "outcome": outcome, "result": result})
        self.counts[outcome] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "jobs": list(self.jobs),
            "processed": dict(self.counts),
            "calls": self.calls,
            "tokens": self.tokens,
            "seconds": round(self.seconds, 3),
            "stopped_by": self.stopped_by,
            "queue": dict(self.queue),
        }


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
        self._run_lock = threading.Lock()
        self._running_job: str | None = None
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
        return {
            "mode": mode,
            "mode_source": self.mode_source,
            "concurrency": self.gate.limits(),
            "in_flight": self.gate.active(),
            "yielded": remaining is not None,
            "yield_remaining_seconds": None if remaining is None else round(remaining, 3),
            "worker": "running" if alive else "stopped",
            "current_job": running_job,
            "run_in_progress": self._run_lock.locked(),
            "runs": self.runs,
            "queue": self.queue.stats(),
        }

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
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

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

    def run(self, budget: Budget | None = None, *, max_jobs: int | None = None) -> RunReport:
        """Process now, in this thread, until the queue is empty or a limit is
        reached. Refused while paused (P11: a paused service calls no model) and
        while another run is in progress."""
        if self.mode == PAUSED:
            raise RunRefused("paused")
        if not self._run_lock.acquire(blocking=False):
            raise RunRefused("busy")
        try:
            budget = budget or Budget()
            budget.started = self._monotonic()
            report = RunReport(new_id("run"))
            session = ModelSession(self.models, budget, self._monotonic)
            log.event("run_start", run=report.id)
            while True:
                limit = budget.exhausted(self._monotonic())
                if limit is not None:
                    report.stopped_by = limit
                    break
                if max_jobs is not None and len(report.jobs) >= max_jobs:
                    report.stopped_by = "max_jobs"
                    break
                if self.mode == PAUSED:
                    report.stopped_by = "paused"
                    break
                job = self.queue.claim()
                if job is None:
                    report.stopped_by = "queue_empty"
                    break
                outcome, result = self._process(job, session)
                report.record(job, outcome, result)
                if outcome == RELEASED:
                    report.stopped_by = session.last_limit or "seconds"
                    break
            report.calls = budget.calls
            report.tokens = budget.tokens
            report.seconds = budget.elapsed(self._monotonic())
            report.queue = self.queue.stats()
            self.runs += 1
            log.event(
                "run_end",
                run=report.id,
                jobs=len(report.jobs),
                calls=report.calls,
                tokens=report.tokens,
                stopped_by=report.stopped_by,
            )
            return report
        finally:
            self._run_lock.release()

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
