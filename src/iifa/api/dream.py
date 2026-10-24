"""Dream and review routes (SPEC 8.6). Starting, following, cancelling and
undoing a dream need the ``dream`` permission; listing and deciding the review
queue need ``review``, which ``dream`` does not imply. A token reaches a run only
when it is bound to the run's scope or above (otherwise the run does not exist
for it, P9), and a queued operation only when it sees every record it touches."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request

from iifa import log
from iifa.api import auth
from iifa.api.routes import (
    ScopeQ,
    ScopesQ,
    _budget,
    _engine,
    _query_view,
    _store,
    _worker,
    decide_proposal_item,
    proposal_item,
)
from iifa.api.schemas import DreamStart, DreamUndo, ReviewDecision
from iifa.dream.apply import DecisionError, UndoBlocked
from iifa.dream.journal import APPLIED, CANCELLED, PENDING, QUEUED, RUNNING, Entry, Journal, Run
from iifa.dream.policy import PolicyError
from iifa.dream.run import DREAM_JOB, DreamBusy, Dreamer
from iifa.ids import kind_of
from iifa.recall.packer import pack
from iifa.scopes import Visibility
from iifa.store.repo import NotFound

router = APIRouter(prefix="/v1")

PLAN_LIMIT = 1000


def _dreamer(request: Request) -> Dreamer:
    dreamer: Dreamer = request.app.state.dreamer
    return dreamer


def _run(request: Request, principal: auth.Principal, run_id: str) -> Run:
    """The run, or 404 when it does not exist or the token is not bound to its scope."""
    if kind_of(run_id) != "dream":
        raise HTTPException(404, "no such dream")
    try:
        run = _dreamer(request).journal.get_run(run_id)
    except NotFound as exc:
        raise HTTPException(404, "no such dream") from exc
    if not principal.may_write([run.scope]):
        raise HTTPException(404, "no such dream")
    return run


def _run_out(run: Run) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "scope": run.scope,
        "status": run.status,
        "dry_run": run.dry_run,
        "operations": list(run.operations),
        "budget": dict(run.budget),
        "review_policy": run.policy.as_dict(),
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "report": run.report,
    }


def _entry_out(entry: Entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "seq": entry.seq,
        "op": entry.op,
        "risk": entry.risk,
        "route": entry.route,
        "status": entry.status,
        "reason": entry.reason,
        "reason_given": entry.reason_given,
        "payload": entry.payload,
        "records": list(entry.records),
        "forgotten": entry.forgotten,
        "created_at": entry.created_at,
        "decided_at": entry.decided_at,
        "undone_at": entry.undone_at,
    }


@router.post("/dream", status_code=202)
def start_dream(
    request: Request,
    body: DreamStart,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/dream", "dream")],
) -> dict[str, Any]:
    """Start a run (SPEC 5.6): recorded and queued at once, executed by the
    background worker under the mode, the gate and ``yield`` (P11)."""
    auth.require_write_scopes(principal, [body.scope])
    dreamer = _dreamer(request)
    requested = body.review_policy.model_dump() if body.review_policy else {}
    try:
        policy = dreamer.policy_for(body.scope, requested)
    except PolicyError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        run = dreamer.start(
            body.scope,
            operations=body.operations,
            budget=body.budget.model_dump() if body.budget else {},
            dry_run=body.dry_run,
            policy=policy,
        )
    except DreamBusy as exc:
        raise HTTPException(409, "a dream is already pending or running on this scope") from exc
    job = _worker(request).enqueue(DREAM_JOB, {"run_id": run.id}, scopes=[body.scope])
    dreamer.journal.set_job(run.id, job.id)
    log.event("dream_queued", run=run.id, job=job.id, dry_run=run.dry_run)
    return {"run_id": run.id, "status": run.status, "job_id": job.id}


@router.get("/dream/{run_id}")
def get_dream(
    request: Request,
    run_id: str,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/dream/{run_id}", "dream")],
) -> dict[str, Any]:
    return _run_out(_run(request, principal, run_id))


@router.get("/dream/{run_id}/plan")
def dream_plan(
    request: Request,
    run_id: str,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/dream/{run_id}/plan", "dream")],
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=PLAN_LIMIT)] = 100,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """The run's operations in order, each with what became of it and why
    (content: the model's texts and reasons), paginated and under budget."""
    run = _run(request, principal, run_id)
    budget = _budget(request, budget_tokens)
    engine = _engine(request)
    entries = _dreamer(request).journal.ops_of_run(run.id, after=after, limit=limit + 1)
    page, more = entries[:limit], len(entries) > limit
    items = [_entry_out(e) for e in page]
    # a page is the longest prefix that fits (P1): pagination never skips an item
    chosen: list[dict[str, Any]] = []
    used = 0
    for item in items:
        cost = engine.cost(item)
        if used + cost > budget:
            break
        chosen.append(item)
        used += cost
    has_next = bool(chosen) and (more or len(chosen) < len(items))
    return {
        "items": chosen,
        "next": chosen[-1]["seq"] if has_next else None,
        "used_tokens": used,
        "budget_tokens": budget,
        "omitted": len(items) - len(chosen),
        "reason": "budget_too_small" if items and not chosen else None,
    }


@router.post("/dream/{run_id}/undo")
def undo_dream(
    request: Request,
    run_id: str,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/dream/{run_id}/undo", "dream")],
    body: DreamUndo | None = None,
) -> dict[str, Any]:
    """Revert a run (all or nothing) or one of its operations (SPEC 5.6 rule 4)."""
    run = _run(request, principal, run_id)
    dreamer = _dreamer(request)
    op_id = body.op_id if body else None
    try:
        if op_id is None:
            undone = dreamer.applier.undo_run(run.id)
        else:
            try:
                entry = dreamer.journal.get_op(op_id)
            except NotFound as exc:
                raise HTTPException(404, "no such operation") from exc
            if entry.run_id != run.id:
                raise HTTPException(404, "no such operation")
            dreamer.applier.undo_op(op_id)
            undone = 1
    except UndoBlocked as exc:
        raise HTTPException(
            409,
            {"reason": exc.reason, "operations": exc.operations, "records": exc.records},
        ) from exc
    log.event("dream_undo", run=run.id, op=op_id, undone=undone)
    return {"run_id": run.id, "undone": undone}


@router.post("/dream/{run_id}/cancel")
def cancel_dream(
    request: Request,
    run_id: str,
    principal: Annotated[
        auth.Principal, auth.declare("POST", "/v1/dream/{run_id}/cancel", "dream")
    ],
) -> dict[str, Any]:
    """Stop a dream cleanly: a pending one never starts, a running one stops at
    its next unit, as at a budget limit."""
    run = _run(request, principal, run_id)
    if run.status not in (PENDING, RUNNING):
        raise HTTPException(409, f"the dream is {run.status}")
    run = _dreamer(request).journal.request_cancel(run.id)
    return {"run_id": run.id, "status": run.status if run.status == CANCELLED else "cancelling"}


# --------------------------------------------------------------------------- the review queue


def _sees_all(store: Any, view: Visibility, records: tuple[str, ...]) -> bool:
    if not records:
        return False
    for record_id in records:
        try:
            if not view.sees(store.scopes_of(record_id)):
                return False
        except NotFound:
            return False
    return True


def _adjudication(journal: Journal, a: str, b: str) -> dict[str, Any] | None:
    """The latest verdict on a pair (SPEC 5.6), shown to the reviewer beside it."""
    latest = journal.latest_adjudication(a, b)
    if latest is None:
        return None
    return {
        "verdict": latest.verdict,
        "reason": latest.reason,
        "run_id": latest.run_id,
        "at": latest.created_at,
    }


def _operation_item(entry: Entry, journal: Journal, store: Any) -> dict[str, Any]:
    item = {
        "kind": "dream_operation",
        "id": entry.id,
        "run_id": entry.run_id,
        "op": entry.op,
        "risk": entry.risk,
        "reason_given": entry.reason_given,
        "payload": entry.payload,
        "records": list(entry.records),
        "created_at": entry.created_at,
    }
    if entry.op == "merge_nodes" and entry.payload is not None:
        keep, absorb = str(entry.payload["keep"]), str(entry.payload["absorb"])
        proposal = store.merge_proposal_between(keep, absorb)
        item["proposal_id"] = proposal.id if proposal is not None else None
        item["adjudication"] = _adjudication(journal, keep, absorb)
    return item


@router.get("/review")
def review_queue(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/review", "review")],
    scope: ScopeQ = None,
    scopes: ScopesQ = None,
    subtree: ScopeQ = None,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """One queue (SPEC 4.4b, 8.6): pending alias, fact and merge proposals and
    queued dream operations the view sees, oldest first, under budget. A merge
    proposal carries the latest adjudication of its pair, and stands aside when
    its merge_nodes is queued (that item carries the same verdict)."""
    view = _query_view(principal, scope, scopes, subtree)
    budget = _budget(request, budget_tokens)
    store = _store(request)
    journal = _dreamer(request).journal
    items: list[dict[str, Any]] = []
    for proposal in store.pending_proposals(view):
        item = proposal_item(proposal)
        if proposal.kind == "merge" and proposal.node_id is not None:
            if journal.merge_ops(proposal.node_id, proposal.value, status=QUEUED):
                continue
            item["adjudication"] = _adjudication(journal, proposal.node_id, proposal.value)
        items.append(item)
    items.extend(
        _operation_item(e, journal, store)
        for e in journal.queued()
        if _sees_all(store, view, e.records)
    )
    items.sort(key=lambda i: (i["created_at"], i["id"]))
    engine = _engine(request)
    result = pack([engine.cost(i) for i in items], budget)
    return {
        "items": [items[i] for i in result.chosen],
        "used_tokens": result.used_tokens,
        "budget_tokens": budget,
        "omitted": result.omitted,
        "reason": result.reason,
    }


@router.post("/review/{item_id}")
def decide_review_item(
    request: Request,
    item_id: str,
    body: ReviewDecision,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/review/{item_id}", "review")],
) -> dict[str, Any]:
    """Accept or reject one item of the queue: a proposal, or a queued operation
    - checked again against memory as it is now (rule 8: ``stale`` is a 409)."""
    accept = body.decision == "accept"
    kind = kind_of(item_id)
    if kind == "prop":
        return decide_proposal_item(request, principal, item_id, accept, from_review=True)
    if kind != "dop":
        raise HTTPException(404, "no such item")
    dreamer = _dreamer(request)
    store = _store(request)
    try:
        entry = dreamer.journal.get_op(item_id)
        run = dreamer.journal.get_run(entry.run_id)
    except NotFound as exc:
        raise HTTPException(404, "no such item") from exc
    if not principal.may_write([run.scope]) or not all(
        principal.may_touch(store.scopes_of(r)) for r in entry.records if store.revisions([r])
    ):
        raise HTTPException(404, "no such item")
    try:
        decided = dreamer.applier.decide(item_id, accept=accept)
    except DecisionError as exc:
        raise HTTPException(409, f"the operation is {exc.status}") from exc
    log.event("dream_review", op=decided.id, status=decided.status)
    if decided.status not in (APPLIED, "rejected"):
        raise HTTPException(409, {"status": decided.status, "reason": decided.reason})
    if decided.status == APPLIED:
        _worker(request).request_embeddings([run.scope])
    return {"id": decided.id, "op": decided.op, "status": decided.status}
