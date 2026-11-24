"""Dream and review routes. Starting, following, cancelling and
undoing a dream need the ``dream`` permission; listing and deciding the review
queue need ``review``, which ``dream`` does not imply. A token reaches a run only
when it is bound to the run's scope or above (otherwise the run does not exist
for it, P9), and a queued operation only when it sees every record it touches."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from iifa import log
from iifa.api import auth
from iifa.api.paging import read_cursor, view_context
from iifa.api.routes import (
    LIST_LIMIT,
    LIST_LIMIT_MAX,
    CursorQ,
    LimitQ,
    ScopeQ,
    ScopesQ,
    _budget,
    _cursors,
    _engine,
    _next_cursor,
    _query_view,
    _store,
    _worker,
    decide_proposal_item,
    proposal_item,
)
from iifa.api.schemas import DreamStart, DreamUndo, ReviewDecision
from iifa.dream.apply import DecisionError, UndoBlocked
from iifa.dream.catalogue import named_ids
from iifa.dream.journal import (
    APPLIED,
    CANCELLED,
    PENDING,
    QUEUED,
    RUNNING,
    Entry,
    Journal,
    Run,
)
from iifa.dream.policy import PolicyError
from iifa.dream.run import DREAM_JOB, DreamBusy, Dreamer
from iifa.ids import kind_of
from iifa.recall.paging import take
from iifa.scopes import ScopeError, Visibility, validate_scope
from iifa.store.repo import NotFound

router = APIRouter(prefix="/v1")


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
        "started_by": run.started_by,
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
    """Start a run: recorded and queued at once, executed by the
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
            started_by=principal.name,
        )
    except DreamBusy as exc:
        raise HTTPException(409, "a dream is already pending or running on this scope") from exc
    job = _worker(request).enqueue(DREAM_JOB, {"run_id": run.id}, scopes=[body.scope])
    dreamer.journal.set_job(run.id, job.id)
    log.event("dream_queued", run=run.id, job=job.id, dry_run=run.dry_run)
    return {"run_id": run.id, "status": run.status, "job_id": job.id}


RunStatus = Literal["pending", "running", "finished", "cancelled", "rolled_back", "failed"]


def _run_item(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "kind": "dream",
        "scope": run.scope,
        "status": run.status,
        "dry_run": run.dry_run,
        "started_by": run.started_by,
        "cancel_requested": run.cancel_requested,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
    }


@router.get("/dreams")
def list_dreams(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/dreams", "dream")],
    scope: Annotated[str, Query(min_length=1, max_length=512)],
    status: Annotated[RunStatus | None, Query()] = None,
    cursor: CursorQ = None,
    limit: LimitQ = LIST_LIMIT,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """The runs on ``scope`` and below it, newest first, a page at a time:
    every run the token reaches, whichever client started it."""
    try:
        validate_scope(scope)
    except ScopeError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not principal.may_write([scope]):
        raise HTTPException(403, "scope outside the token's scopes")
    budget = _budget(request, budget_tokens)
    context = {"route": "dreams", "scope": scope, "status": status}
    position = read_cursor(_cursors(request), cursor, context, str, str)
    before = (position[0], position[1]) if position else None
    runs = _dreamer(request).journal.runs_within(
        scope, status=status, before=before, limit=limit + 1
    )
    items = [_run_item(r) for r in runs]
    page = take(items, limit=limit, budget_tokens=budget, cost=_engine(request).cost)
    last = page.last if page.more else None
    return {
        "items": page.items,
        "next_cursor": _next_cursor(
            request, context, [last["created_at"], last["id"]] if last else None
        ),
        "used_tokens": page.used_tokens,
        "budget_tokens": budget,
        "omitted": page.omitted,
        "reason": page.reason,
    }


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
    after: Annotated[int | None, Query(ge=0)] = None,
    cursor: CursorQ = None,
    limit: Annotated[int, Query(ge=1, le=LIST_LIMIT_MAX)] = LIST_LIMIT,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """The run's operations in order, each with what became of it and why
    (content: the model's texts and reasons), a page at a time under budget.
    ``after`` is the same position as ``cursor``, as a plain ``seq``."""
    run = _run(request, principal, run_id)
    if after is not None and cursor:
        raise HTTPException(422, "give after or cursor, not both")
    budget = _budget(request, budget_tokens)
    context = {"route": "plan", "run": run.id}
    position = read_cursor(_cursors(request), cursor, context, int)
    start = position[0] if position else (after or 0)
    engine = _engine(request)
    entries = _dreamer(request).journal.ops_of_run(run.id, after=start, limit=limit + 1)
    items = [_entry_out(e) for e in entries]
    page = take(items, limit=limit, budget_tokens=budget, cost=engine.cost, empty_reason=None)
    last = page.last if page.more else None
    return {
        "items": page.items,
        "next": last["seq"] if last else None,
        "next_cursor": _next_cursor(request, context, [last["seq"]] if last else None),
        "used_tokens": page.used_tokens,
        "budget_tokens": budget,
        "omitted": page.omitted,
        "reason": page.reason,
    }


@router.post("/dream/{run_id}/undo")
def undo_dream(
    request: Request,
    run_id: str,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/dream/{run_id}/undo", "dream")],
    body: DreamUndo | None = None,
) -> dict[str, Any]:
    """Revert a run (all or nothing) or one of its operations."""
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
    """The latest verdict on a pair, shown to the reviewer beside it."""
    latest = journal.latest_adjudication(a, b)
    if latest is None:
        return None
    return {
        "verdict": latest.verdict,
        "reason": latest.reason,
        "run_id": latest.run_id,
        "at": latest.created_at,
    }


REVIEW_NODE_FACTS = 3  # facts shown per node of a review item (pinned first, then newest)


def _item_node_ids(item: dict[str, Any]) -> list[str]:
    """The nodes a review item is about, in the order a reviewer reads them: a
    merge proposal's two sides (``node_id`` then ``value``), the nodes a queued
    operation's payload names (a ``merge_nodes``'s ``keep`` then ``absorb``, a
    grouping's members), an alias's or a fact's node. A merge without its
    ``node_id`` has no pair (no adjudication either): it names nothing."""
    if item["kind"] == "dream_operation":
        return [i for i in named_ids(item["payload"] or {}) if kind_of(i) == "node"]
    if not item["node_id"]:
        return []
    if item["kind"] == "merge":
        return [item["node_id"], item["value"]]
    return [item["node_id"]]


def _item_fact_ids(item: dict[str, Any]) -> list[str]:
    """The facts a queued operation's payload names (the facts a consolidation
    replaces, the two facts of a supersession). A proposal names none: a fact
    proposal carries its own statement."""
    if item["kind"] != "dream_operation":
        return []
    return [i for i in named_ids(item["payload"] or {}) if kind_of(i) == "fact"]


def _node_briefs(store: Any, view: Visibility, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Each node by name, kind, aliases and its first few facts, by id, so a
    reviewer reads a question about two subjects without opening them: two
    queries for a whole page. Only what the view sees (P9): a node it does not
    see, or no longer live, is left out, and so is a fact it does not see."""
    nodes = store.nodes_by_id(ids, view)
    briefs = {
        node.id: {
            "id": node.id,
            "name": node.name,
            "kind": node.kind,
            "aliases": list(node.aliases),
            "facts": [],
        }
        for node in nodes.values()
    }
    for fact in store.facts_top(list(briefs), view, REVIEW_NODE_FACTS):
        briefs[fact.node_id]["facts"].append({"id": fact.id, "statement": fact.statement})
    return briefs


def _fact_statements(store: Any, view: Visibility, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Each fact the view sees (P9) by its statement, by id."""
    return {
        fact.id: {"id": fact.id, "statement": fact.statement}
        for fact in store.facts_by_id(ids, view).values()
    }


def _replaced(store: Any, view: Visibility, proposal: Any) -> list[dict[str, Any]]:
    """The live facts accepting a fact proposal would close, by
    their statement: an accepted commitment of the self replaces the one of
    its topic, and the topic is the extraction's reading - a reviewer must see
    what goes before deciding. Only what the view sees (P9)."""
    closing = store.replaced_by(proposal)
    if not closing:
        return []
    seen = store.facts_by_id([f.id for f in closing], view)
    return [{"id": f.id, "statement": f.statement} for f in closing if f.id in seen]


def _name_nodes(
    page: list[dict[str, Any]],
    briefs: Mapping[str, dict[str, Any]],
    spare: int,
    cost: Callable[[Any], int],
    statements: Mapping[str, dict[str, Any]] | None = None,
) -> None:
    """Attach ``nodes``, and the ``facts`` a queued operation names, to the
    items of a page, in order, out of the ``spare`` tokens the page leaves: the
    items on the page stay exactly the ones it holds without names. Where the
    spare runs short the facts under an item's nodes go first, then the facts
    it names, then its nodes' names."""
    statements = statements or {}
    for item in page:
        named = [briefs[i] for i in dict.fromkeys(_item_node_ids(item)) if i in briefs]
        stated = [statements[i] for i in dict.fromkeys(_item_fact_ids(item)) if i in statements]
        names = [{**b, "facts": []} for b in named]
        bare = cost(item)
        for nodes, facts in ((named, stated), (names, stated), (names, [])):
            added = {k: v for k, v in (("nodes", nodes), ("facts", facts)) if v}
            if not added:
                continue
            extra = cost({**item, **added}) - bare
            if extra <= spare:
                item.update(added)
                spare -= extra
                break


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
    cursor: CursorQ = None,
    limit: LimitQ = LIST_LIMIT,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """One queue: pending alias, fact and merge proposals and
    queued dream operations the view sees, oldest first, a page at a time under
    budget. A merge proposal carries the latest adjudication of its
    pair, and stands aside when its merge_nodes is queued (that item carries the
    same verdict). A fact proposal whose accept would close live facts carries
    ``replaces``, each one by its statement. An item about nodes carries
    ``nodes``: each one the view sees,
    by name with its first few facts; a queued operation that names facts
    carries ``facts``, each one the view sees by its statement."""
    view = _query_view(principal, scope, scopes, subtree)
    budget = _budget(request, budget_tokens)
    context = {"route": "review", "view": view_context(view)}
    position = read_cursor(_cursors(request), cursor, context, str, str)
    store = _store(request)
    journal = _dreamer(request).journal
    items: list[dict[str, Any]] = []
    for proposal in store.pending_proposals(view):
        item = proposal_item(proposal)
        if proposal.kind == "merge" and proposal.node_id is not None:
            if journal.merge_ops(proposal.node_id, proposal.value, status=QUEUED):
                continue
            item["adjudication"] = _adjudication(journal, proposal.node_id, proposal.value)
        if replaced := _replaced(store, view, proposal):
            item["replaces"] = replaced
        items.append(item)
    items.extend(
        _operation_item(e, journal, store)
        for e in journal.queued()
        if _sees_all(store, view, e.records)
    )
    items.sort(key=lambda i: (i["created_at"], i["id"]))
    if position is not None:
        items = [i for i in items if (i["created_at"], i["id"]) > position]
    items = items[: limit + 1]
    engine = _engine(request)
    # the page as it stands without names, then the names out of what it leaves
    bare = take(items, limit=limit, budget_tokens=budget, cost=engine.cost)
    ids = list(dict.fromkeys(i for item in bare.items for i in _item_node_ids(item)))
    fact_ids = list(dict.fromkeys(i for item in bare.items for i in _item_fact_ids(item)))
    if ids or fact_ids:
        _name_nodes(
            bare.items,
            _node_briefs(store, view, ids),
            budget - bare.used_tokens,
            engine.cost,
            _fact_statements(store, view, fact_ids),
        )
    page = take(items, limit=limit, budget_tokens=budget, cost=engine.cost)
    last = page.last if page.more else None
    return {
        "items": page.items,
        "next_cursor": _next_cursor(
            request, context, [last["created_at"], last["id"]] if last else None
        ),
        "used_tokens": page.used_tokens,
        "budget_tokens": budget,
        "omitted": page.omitted,
        "reason": page.reason,
    }


@router.post("/review/{item_id}")
def decide_review_item(
    request: Request,
    item_id: str,
    body: ReviewDecision,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/review/{item_id}", "review")],
) -> dict[str, Any]:
    """Accept or reject one item of the queue: a proposal (accepting a merge
    proposal merges the two nodes as a person's decision), or a queued
    operation - checked again against memory as it is now (``stale`` is
    a 409)."""
    accept = body.decision == "accept"
    kind = kind_of(item_id)
    if kind == "prop":
        return decide_proposal_item(
            request, principal, item_id, accept, from_review=True, keep=body.keep
        )
    if kind != "dop":
        raise HTTPException(404, "no such item")
    if body.keep is not None:
        raise HTTPException(422, "keep is given only to accept a merge proposal")
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
