"""A dream run (SPEC 5.6): started through the API, executed as a job of the
persistent queue so that it obeys the background mode, the gate and ``yield``
(P11), unit by unit, within its budget, and measured against the probe sets of
its scope (rule 9).

Test seams (never set in service): ``_after_unit`` runs after each unit,
``_before_apply`` between the model's answer and the first operation of a
unit, ``_before_probes_after`` before the probes are scored again.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from iifa import log
from iifa.dream.adjudicate import Adjudicator, NameNeighbours, NamesClose, empty_report
from iifa.dream.apply import Applier, changeable
from iifa.dream.bounds import OVERSIZED
from iifa.dream.catalogue import MODEL_OPERATIONS, OPERATIONS, UNIT_OPERATIONS, parse_answer
from iifa.dream.journal import (
    ACTIVE,
    APPLIED,
    CANCELLED,
    DROPPED,
    FINISHED,
    PENDING,
    PLANNED,
    QUEUED,
    ROLLED_BACK,
    RUNNING,
    Journal,
    Run,
)
from iifa.dream.policy import ReviewPolicy, resolve_policy
from iifa.dream.units import next_unit_node, read_unit
from iifa.ingest.queue import Job
from iifa.ingest.worker import Budget, BudgetExhausted, JobContext
from iifa.llm import TRUNCATED, ChatResult, ModelClient, ModelError, ModelWaitTimeout
from iifa.recall.probes import INCONCLUSIVE, OK, ProbeResult, ProbeScore, compare
from iifa.scopes import within
from iifa.store.models import ProbeSet
from iifa.store.repo import NotFound, Store
from iifa.tokens import ConservativeTokenizer, Tokenizer

DREAM_JOB = "dream"
DEFAULT_BUDGET: dict[str, int | float] = {
    "calls": 50,
    "tokens": 200_000,
    "seconds": 1800,
    "operations": 200,
}
FINISHED_STATES = (FINISHED, CANCELLED, ROLLED_BACK, "failed")


class DreamBusy(RuntimeError):
    """Another dream is pending or running on the scope, an ancestor or a descendant."""


@dataclass
class DreamConfig:
    answer_max_tokens: int = 2048
    json_mode: bool = True
    probe_tolerance: float = 0.02
    adjudications_per_unit: int = 3
    prompt_max_tokens: int = 4000  # what one call reads (SPEC 5.6), by ``tokenizer``
    tokenizer: Tokenizer = field(default_factory=ConservativeTokenizer)
    edge_half_life_days: float = 30.0
    edge_prune_below: float = 0.1
    review: tuple[tuple[str, str, str], ...] = ()
    forced_language: Callable[[Sequence[str]], str | None] = field(default=lambda scopes: None)


class DreamSession:
    """The dream model as a run sees it: every call checks the run's budget and
    the budget of the ``run`` (SPEC 8.4) the job executes in, passes the gate,
    and is charged to both."""

    def __init__(
        self,
        client: ModelClient,
        budgets: Sequence[Budget],
        monotonic: Callable[[], float],
    ) -> None:
        self._client = client
        self._budgets = list(budgets)
        self._monotonic = monotonic

    def chat(self, messages: Sequence[Mapping[str, Any]], **params: Any) -> ChatResult:
        endpoint = self._client.chat_endpoint
        assert endpoint is not None  # a dream client always has its endpoint
        now = self._monotonic()
        waits: list[float] = []
        for budget in self._budgets:
            limit = budget.exhausted(now)
            if limit is not None:
                raise BudgetExhausted(limit)
            left = budget.remaining_seconds(now)
            if left is not None:
                waits.append(left)
        try:
            result = self._client.chat(
                messages,
                model=endpoint.model,
                wait_timeout=min(waits) if waits else None,
                **params,
            )
        except ModelWaitTimeout as exc:
            raise BudgetExhausted("seconds") from exc
        except ModelError as exc:
            if exc.reason == TRUNCATED:
                for budget in self._budgets:
                    budget.charge(int(params.get("max_tokens") or 0))
            raise
        for budget in self._budgets:
            budget.charge(result.usage.total)
        return result


class Dreamer:
    def __init__(
        self,
        store: Store,
        *,
        client: ModelClient | None,
        config: DreamConfig,
        probe_scorer: Callable[[ProbeSet], ProbeScore] | None = None,
        request_embeddings: Callable[[Sequence[str]], bool] | None = None,
        has_embeddings: bool = False,
        name_neighbours: NameNeighbours | None = None,
        names_close: NamesClose | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.client = client
        self.config = config
        self.journal = Journal(store)
        self.applier = Applier(store, self.journal, forced_language=config.forced_language)
        self.adjudicator = Adjudicator(
            store,
            self.journal,
            self.applier,
            per_unit=config.adjudications_per_unit,
            answer_max_tokens=config.answer_max_tokens,
            json_mode=config.json_mode,
            tokenizer=config.tokenizer,
            prompt_max_tokens=config.prompt_max_tokens,
            names_close=names_close,
        )
        # the nodes whose name vector reaches ``resolve`` (tier 3, calibrated), else None
        self.name_neighbours = name_neighbours
        self.probe_scorer = probe_scorer
        self.request_embeddings = request_embeddings
        self.has_embeddings = has_embeddings
        self._monotonic = monotonic
        self._after_unit: Callable[[], object] | None = None
        self._before_apply: Callable[[], object] | None = None
        self._before_probes_after: Callable[[], object] | None = None

    # ------------------------------------------------------------------ starting

    def policy_for(self, scope: str, requested: Mapping[str, str | None]) -> ReviewPolicy:
        """The configured policy of ``scope``, tightened by a request (PolicyError
        when it would loosen it)."""
        return resolve_policy(scope, self.config.review).tightened(requested)

    def start(
        self,
        scope: str,
        *,
        operations: Sequence[str] | None,
        budget: Mapping[str, Any],
        dry_run: bool,
        policy: ReviewPolicy,
    ) -> Run:
        """Record a run; refused while another is active in the scope's family (rule 7)."""
        with self.store.lock:
            for active in self.journal.active_runs():
                if within(scope, active.scope) or within(active.scope, scope):
                    raise DreamBusy(active.id)
            limits = {**DEFAULT_BUDGET, **{k: v for k, v in budget.items() if v is not None}}
            return self.journal.create_run(
                scope=scope, dry_run=dry_run, operations=operations, budget=limits, policy=policy
            )

    # ------------------------------------------------------------------ the job

    def handle(self, job: Job, ctx: JobContext) -> dict[str, Any]:
        run_id = str(job.payload.get("run_id", ""))
        report = self.execute(run_id, outer=ctx.models.budget, interrupt=ctx.interrupt)
        return {"run_id": run_id, "stopped_by": report.get("stopped_by")}

    def execute(  # noqa: PLR0912, PLR0915 - the run loop reads best in one piece
        self,
        run_id: str,
        *,
        outer: Budget | None = None,
        interrupt: Callable[[], str | None] = lambda: None,
    ) -> dict[str, Any]:
        journal = self.journal
        try:
            run = journal.get_run(run_id)
        except NotFound:
            return {"stopped_by": "unknown_run"}
        if run.status in FINISHED_STATES:
            return run.report or {}
        if run.cancel_requested and run.status == PENDING:
            return (
                journal.finish_run(run.id, CANCELLED, self._report(run, {}, "cancelled")).report
                or {}
            )
        state = dict(run.state)
        if run.status == PENDING:
            run = journal.start_run(run.id)
        if not state.get("setup"):
            state = self._setup(run)
            journal.save_state(run.id, state)
            run = journal.get_run(run.id)
        state.setdefault("adjudications", empty_report())
        state.setdefault("adjudicated", [])
        state.setdefault("prompts", self._prompts_report())
        log.event("dream_start", run=run.id, dry_run=run.dry_run)

        own = Budget(
            max_calls=_int(run.budget.get("calls")),
            max_tokens=_int(run.budget.get("tokens")),
            max_seconds=_float(run.budget.get("seconds")),
            started=self._monotonic(),
        )
        budgets = [own] + ([outer] if outer is not None else [])
        allowed = [o for o in UNIT_OPERATIONS if o in run.operations]
        adjudicate = "merge_nodes" in run.operations and self.adjudicator.per_unit > 0
        max_ops = _int(run.budget.get("operations"))
        stopped_by = "complete"
        if self.client is None:
            allowed, adjudicate = [], False
        session = DreamSession(self.client, budgets, self._monotonic) if self.client else None

        while (allowed or adjudicate) and session is not None:
            run = journal.get_run(run.id)
            reason = self._stop_reason(run, budgets, interrupt, state, max_ops)
            if reason is not None:
                stopped_by = reason
                break
            node = next_unit_node(self.store, run.scope, str(state["after"]))
            if node is None:
                stopped_by = "complete"
                state["after"] = ""
                if not run.dry_run:
                    journal.set_cursor(run.scope, "")
                break
            if adjudicate:
                try:
                    state["counted"] += self.adjudicator.unit(
                        run, node, session, state, self.name_neighbours
                    )
                except BudgetExhausted as exc:
                    stopped_by = exc.limit
                    break
                except ModelError:
                    stopped_by = "model_error"
                    break
                finally:
                    journal.save_state(run.id, state)
            if not allowed:
                self._unit_done(run, node.id, state)
                continue
            unit = read_unit(self.store, run.scope, node, self.config.forced_language)
            fitted = unit.fitted(allowed, self.config.tokenizer, self.config.prompt_max_tokens)
            prompts = state["prompts"]
            if fitted is None:
                prompts[OVERSIZED] += 1  # never cut from the start by the server
                self._unit_done(run, node.id, state)
                continue
            prompts["units_reduced"] += int(fitted.reduced)
            prompts["largest_prompt_tokens"] = max(prompts["largest_prompt_tokens"], fitted.tokens)
            try:
                result = session.chat(
                    fitted.messages,
                    max_tokens=self.config.answer_max_tokens,
                    json_object=self.config.json_mode,
                )
            except BudgetExhausted as exc:
                stopped_by = exc.limit
                break
            except ModelError as exc:
                if exc.reason != TRUNCATED:
                    stopped_by = "model_error"
                    break
                state["skipped"]["truncated"] += 1
                result = None
            if result is not None:
                parsed = parse_answer(result.content, allowed=allowed)
                if parsed.error is not None:
                    state["skipped"][parsed.error] += 1
                else:
                    for cause, count in parsed.rejected.items():
                        state["rejected"][cause] = state["rejected"].get(cause, 0) + count
                    if self._before_apply is not None:
                        self._before_apply()
                    snapshot = dict(unit.snapshot)
                    for index, op in enumerate(parsed.operations):
                        if max_ops is not None and state["counted"] >= max_ops:
                            state["left"] += len(parsed.operations) - index
                            break
                        entry = self.applier.submit(run, op, snapshot, unit_ids=fitted.ids)
                        if entry.status in (APPLIED, QUEUED, PLANNED):
                            state["counted"] += 1
            self._unit_done(run, node.id, state)

        state["calls"] = own.calls + int(state.get("calls_before", 0))
        state["tokens"] = own.tokens + int(state.get("tokens_before", 0))
        state["calls_before"], state["tokens_before"] = state["calls"], state["tokens"]
        state["seconds"] = round(own.elapsed(self._monotonic()), 3)
        journal.save_state(run.id, state)
        run = journal.get_run(run.id)
        status = CANCELLED if stopped_by == "cancelled" else FINISHED
        probes, rolled_back = self._measure(run)
        if rolled_back is not None:
            status = ROLLED_BACK
        report = self._report(run, state, stopped_by, probes=probes, rolled_back=rolled_back)
        journal.finish_run(run.id, status, report)
        log.event(
            "dream_end",
            run=run.id,
            status=status,
            stopped_by=stopped_by,
            units=state.get("units", 0),
            calls=state["calls"],
        )
        return report

    # ------------------------------------------------------------------ pieces

    def _prompts_report(self) -> dict[str, int]:
        return {
            "budget": self.config.prompt_max_tokens,
            "largest_prompt_tokens": 0,
            "units_reduced": 0,
            "pairs_reduced": 0,
            OVERSIZED: 0,
        }

    def _unit_done(self, run: Run, node_id: str, state: dict[str, Any]) -> None:
        state["units"] += 1
        state["after"] = node_id
        self.journal.save_state(run.id, state)
        if not run.dry_run:
            self.journal.set_cursor(run.scope, node_id)
        if self._after_unit is not None:
            self._after_unit()

    def _setup(self, run: Run) -> dict[str, Any]:
        """Once per run: where it starts, the probe scores before, the service's
        own operations (they need no model)."""
        state: dict[str, Any] = {
            "setup": True,
            "after": self.journal.cursor(run.scope),
            "units": 0,
            "counted": 0,
            "left": 0,
            "rejected": {},
            "skipped": Counter({"bad_answer": 0, "truncated": 0}),
            "adjudications": empty_report(),
            "adjudicated": [],
        }
        if not run.dry_run:
            sets = self.store.probe_sets_within(run.scope)
            if sets and self.probe_scorer is not None:
                self.journal.save_probes_before(
                    run.id, [_score_dict(self.probe_scorer(s)) for s in sets]
                )
        if "decay_edges" in run.operations:
            plan = self.applier.decay_plan(
                run.scope,
                half_life_days=self.config.edge_half_life_days,
                prune_below=self.config.edge_prune_below,
                now=self.store.clock(),
            )
            if plan:
                self.applier.record_service(
                    run,
                    "decay_edges",
                    apply=lambda: self.applier.apply_decay(plan),
                    watch=[e.id for e, _, _ in plan],
                    summary={"edges": len(plan), "pruned": sum(1 for _, _, p in plan if p)},
                )
                state["counted"] += 1
        if "reembed" in run.operations and self.has_embeddings and self.request_embeddings:
            request = self.request_embeddings

            def queue_embed() -> tuple[dict[str, Any], list[str]]:
                request([run.scope])
                return {}, []

            self.applier.record_service(run, "reembed", apply=queue_embed)
            state["counted"] += 1
        return state

    def _stop_reason(
        self,
        run: Run,
        budgets: Sequence[Budget],
        interrupt: Callable[[], str | None],
        state: Mapping[str, Any],
        max_ops: int | None,
    ) -> str | None:
        if run.cancel_requested:
            return "cancelled"
        reason = interrupt()
        if reason is not None:
            return reason
        now = self._monotonic()
        for budget in budgets:
            limit = budget.exhausted(now)
            if limit is not None:
                return limit
        if max_ops is not None and int(state["counted"]) >= max_ops:
            return "operations"
        return None

    def _measure(self, run: Run) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Rule 9: compare the probe scores before and after; a regression or an
        inconclusive comparison rolls the run back."""
        before = run.probes_before
        if run.dry_run or not before or self.probe_scorer is None:
            return {"measured": False, "sets": []}, None
        applied = any(e.status == APPLIED for e in self.journal.ops_of_run(run.id))
        sets: list[dict[str, Any]] = []
        failure: str | None = None
        if applied and self._before_probes_after is not None:
            self._before_probes_after()
        for saved in before:
            entry: dict[str, Any] = {"set_id": saved["set_id"], "before": _numbers(saved)}
            if not applied:
                entry.update(after=None, verdict="skipped", reason="nothing_applied")
                sets.append(entry)
                continue
            try:
                probe_set = self.store.get_probe_set(saved["set_id"])
            except NotFound:
                entry.update(after=None, verdict=INCONCLUSIVE, reason="probes")
                failure = failure or INCONCLUSIVE
                sets.append(entry)
                continue
            after = self.probe_scorer(probe_set)
            verdict = compare(_score_from(saved), after, tolerance=self.config.probe_tolerance)
            entry.update(
                after=_numbers(_score_dict(after)), verdict=verdict.verdict, reason=verdict.reason
            )
            if verdict.verdict != OK:
                failure = failure or verdict.verdict
            sets.append(entry)
        if failure is None:
            return {"measured": True, "sets": sets}, None
        incomplete = self.applier.rollback(run.id)
        return {"measured": True, "sets": sets}, {"reason": failure, "incomplete": incomplete}

    def _report(
        self,
        run: Run,
        state: Mapping[str, Any],
        stopped_by: str,
        *,
        probes: Mapping[str, Any] | None = None,
        rolled_back: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Counts and ids only (P5): what the run did, sent to review, dropped."""
        operations: dict[str, dict[str, int]] = {}
        dropped: Counter[str] = Counter()
        review: list[str] = []
        for entry in self.journal.ops_of_run(run.id):
            counts = operations.setdefault(entry.op, {})
            counts[entry.status] = counts.get(entry.status, 0) + 1
            if entry.status == DROPPED and entry.reason:
                dropped[entry.reason] += 1
            if entry.status == QUEUED:
                review.append(entry.id)
        requested_model = [o for o in MODEL_OPERATIONS if o in run.operations]
        return {
            "stopped_by": stopped_by,
            "units": int(state.get("units", 0)),
            "operations": {o: operations[o] for o in OPERATIONS if o in operations},
            "rejected": dict(state.get("rejected", {})),
            "dropped": dict(dropped),
            "skipped": dict(state.get("skipped", {"bad_answer": 0, "truncated": 0})),
            "review": review,
            "left": int(state.get("left", 0)),
            "unavailable": requested_model if self.client is None else [],
            "calls": int(state.get("calls", 0)),
            "tokens": int(state.get("tokens", 0)),
            "seconds": float(state.get("seconds", 0.0)),
            "adjudications": dict(state.get("adjudications") or empty_report()),
            "prompts": dict(state.get("prompts") or self._prompts_report()),
            "probes": dict(probes or {"measured": False, "sets": []}),
            "rolled_back": dict(rolled_back) if rolled_back is not None else None,
        }


def dream_job(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    """The queue handler of a dream run (SPEC 5.6, P11)."""
    dreamer = ctx.dream
    if not isinstance(dreamer, Dreamer):
        raise RuntimeError("no dreamer configured")
    return dreamer.handle(job, ctx)


def _int(value: Any) -> int | None:
    return None if value is None else int(value)


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _score_dict(score: ProbeScore) -> dict[str, Any]:
    return {
        "set_id": score.set_id,
        "scope": score.scope,
        "budget_tokens": score.budget_tokens,
        "tier": score.tier,
        "recall": score.recall,
        "used_tokens": score.used_tokens,
        "recall_per_1k_tokens": score.recall_per_1k_tokens,
        "summary_tokens": score.summary_tokens,
        "compared_per_1k_tokens": score.compared_per_1k_tokens,
        "probe_ids": sorted(score.scored_ids),
    }


def _numbers(saved: Mapping[str, Any]) -> dict[str, Any]:
    """What the report shows per set (SPEC 5.6 rule 9): the raw rate beside the
    compared one, so the price of summaries stays visible."""
    return {
        "recall": saved["recall"],
        "recall_per_1k_tokens": saved["recall_per_1k_tokens"],
        "compared_per_1k_tokens": saved.get("compared_per_1k_tokens"),
        "summary_tokens": saved.get("summary_tokens", 0.0),
        "tier": saved["tier"],
    }


def _score_from(saved: Mapping[str, Any]) -> ProbeScore:
    probes = tuple(ProbeResult(p, (), (), (), 0, 0.0) for p in saved["probe_ids"])
    return ProbeScore(
        set_id=saved["set_id"],
        scope=saved["scope"],
        budget_tokens=int(saved["budget_tokens"]),
        tier=saved["tier"],
        recall=saved["recall"],
        used_tokens=float(saved["used_tokens"]),
        recall_per_1k_tokens=saved["recall_per_1k_tokens"],
        scored=len(probes),
        probes=probes,
        summary_tokens=float(saved.get("summary_tokens", 0.0)),
        compared_per_1k_tokens=saved.get("compared_per_1k_tokens"),
    )


__all__ = [
    "ACTIVE",
    "DEFAULT_BUDGET",
    "DREAM_JOB",
    "RUNNING",
    "DreamBusy",
    "DreamConfig",
    "Dreamer",
    "changeable",
    "dream_job",
]
