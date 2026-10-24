"""Routes under ``/v1``. Every route declares its permission; every content-returning
route resolves a view and a budget before touching the store."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from iifa import __version__, calibration, log, vectors
from iifa.api import auth
from iifa.api.schemas import (
    AtRequest,
    BackgroundUpdate,
    ExpandRequest,
    FactPatch,
    ForgetRequest,
    LinkRequest,
    PathRequest,
    PrecompactRequest,
    ProbeSetCreate,
    ProposalDecision,
    RecallRequest,
    RecordRequest,
    RememberRequest,
    RememberResponse,
    RunRequest,
    SessionEndRequest,
    SessionStartRequest,
    TimelineRequest,
    TurnRequest,
    YieldRequest,
)
from iifa.ids import kind_of
from iifa.ingest.jobs import PRECOMPACT, SESSION_END, TURN
from iifa.ingest.queue import Job
from iifa.ingest.worker import Budget, RunRefused, Worker
from iifa.llm import ModelClient
from iifa.recall import history, paths
from iifa.recall.dense import SuppliedVector
from iifa.recall.engine import RecallEngine
from iifa.recall.packer import pack
from iifa.recall.probes import run_probe_set
from iifa.recall.profile import profile_block
from iifa.scopes import Visibility
from iifa.store.models import ProbeSet, Proposal
from iifa.store.repo import NotFound, ProposalDecided, ProposalNotApplicable, ScopeMismatch, Store

router = APIRouter(prefix="/v1")


def _store(request: Request) -> Store:
    store: Store = request.app.state.store
    return store


def _engine(request: Request) -> RecallEngine:
    engine: RecallEngine = request.app.state.engine
    return engine


def _worker(request: Request) -> Worker:
    worker: Worker = request.app.state.worker
    return worker


def _models(request: Request) -> ModelClient:
    models: ModelClient = request.app.state.models
    return models


def _budget(request: Request, requested: int | None) -> int:
    settings = request.app.state.settings
    if requested is None:
        return int(settings.default_budget_tokens)
    if requested > settings.max_budget_tokens:
        raise HTTPException(
            422, f"budget_tokens above the configured maximum {settings.max_budget_tokens}"
        )
    return requested


def _view(principal: auth.Principal, body: Any) -> Visibility:
    return auth.resolve_view(principal, scope=body.scope, scopes=body.scopes, subtree=body.subtree)


ScopeQ = Annotated[str | None, Query()]
ScopesQ = Annotated[list[str] | None, Query()]


def _query_view(
    principal: auth.Principal, scope: str | None, scopes: list[str] | None, subtree: str | None
) -> Visibility:
    try:
        return auth.resolve_view(principal, scope=scope, scopes=scopes, subtree=subtree)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# --------------------------------------------------------------------------- operations


@router.get("/health")
def health(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/health", auth.ANY)],
) -> dict[str, Any]:
    """Tier, configured backends, the queue (depth and oldest pending item, so a
    paused queue never fills in silence, P11) and the background mode. Answering
    makes no model call."""
    status = _worker(request).status()
    vectors_status = _vectors_status(request)
    return {
        "status": "ok",
        "version": __version__,
        "tier": _engine(request).tier,
        "backends": _models(request).status(),
        "queue": status["queue"],
        "background": {
            "mode": status["mode"],
            "yielded": status["yielded"],
            "concurrency": status["concurrency"],
            "worker": status["worker"],
        },
        "stale_vectors": vectors_status["stale"] if vectors_status else 0,
        "vectors": vectors_status,
        "tokenizer": request.app.state.engine.tokenizer.name,
    }


def _vectors_status(request: Request) -> dict[str, Any] | None:
    """Tier 3 as health reports it (SPEC 6.1, P10): the model, the counts, whether
    recall embeds the query, and which arithmetic path runs (and past which size
    the pure-Python path misses the latency target). Counts only, no model call."""
    dense = _engine(request).dense
    if dense is None or dense.model is None:
        return None
    counts = _store(request).vector_stats(dense.model)
    pipeline = _worker(request).pipeline
    thresholds = pipeline.thresholds(dense.model)
    return {
        "model": dense.model,
        **counts,
        "calibration": calibration.describe(thresholds, key=pipeline.key_threshold_of(dense.model)),
        "dense": dense.status,
        "path": vectors.active_path(),
        "python_path_limit": vectors.python_path_limit(),
    }


@router.get("/stats")
def stats(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/stats", "admin")],
) -> dict[str, Any]:
    return {"counts": _store(request).counts()}


# --------------------------------------------------------------------------- reading


@router.post("/recall")
def recall(
    request: Request,
    body: RecallRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/recall", "read")],
) -> dict[str, Any]:
    view = _view(principal, body)
    budget = _budget(request, body.budget_tokens)
    supplied = (
        SuppliedVector(body.query_vector.model, tuple(body.query_vector.values))
        if body.query_vector is not None
        else None
    )
    result = _engine(request).recall(
        body.query,
        view=view,
        budget_tokens=budget,
        exclude_ids=body.exclude_ids,
        query_vector=supplied,
    )
    log.event(
        "recall",
        items=len(result["items"]),
        omitted=result["omitted"],
        used=result["used_tokens"],
        budget=budget,
    )
    return result


@router.post("/expand")
def expand(
    request: Request,
    body: ExpandRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/expand", "read")],
) -> dict[str, Any]:
    view = _view(principal, body)
    budget = _budget(request, body.budget_tokens)
    try:
        result = _engine(request).expand(
            body.ids, view=view, budget_tokens=budget, level=body.level, cursor=body.cursor
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    log.event(
        "expand", items=len(result["items"]), omitted=result["omitted"], used=result["used_tokens"]
    )
    return result


@router.get("/nodes/{node_id}")
def get_node(
    request: Request,
    node_id: str,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/nodes/{node_id}", "read")],
    scope: ScopeQ = None,
    scopes: ScopesQ = None,
    subtree: ScopeQ = None,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    view = _query_view(principal, scope, scopes, subtree)
    result = _engine(request).node_view(
        node_id, view=view, budget_tokens=_budget(request, budget_tokens)
    )
    if result is None:
        raise HTTPException(404, "no such node")
    return result


@router.post("/timeline")
def timeline(
    request: Request,
    body: TimelineRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/timeline", "read")],
) -> dict[str, Any]:
    """Fact changes of a node, or of the whole view, inside a date range (P7)."""
    view = _view(principal, body)
    budget = _budget(request, body.budget_tokens)
    try:
        result = history.timeline(
            _engine(request),
            view=view,
            budget_tokens=budget,
            node_id=body.node_id,
            since=body.since,
            until=body.until,
            limit=body.limit,
        )
    except ValueError as exc:
        raise HTTPException(422, f"invalid date: {exc}") from exc
    if result is None:
        raise HTTPException(404, "no such node")
    log.event("timeline", items=len(result["items"]), omitted=result["omitted"], budget=budget)
    return result


@router.post("/at")
def known_at(
    request: Request,
    body: AtRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/at", "read")],
) -> dict[str, Any]:
    """What was known about a node at a date: recorded by then, not closed by then."""
    view = _view(principal, body)
    budget = _budget(request, body.budget_tokens)
    try:
        result = history.known_at(
            _engine(request), view=view, budget_tokens=budget, node_id=body.node_id, date=body.date
        )
    except ValueError as exc:
        raise HTTPException(422, f"invalid date: {exc}") from exc
    if result is None:
        raise HTTPException(404, "no such node")
    log.event("at", items=len(result["items"]), omitted=result["omitted"], budget=budget)
    return result


@router.post("/path")
def find_path(
    request: Request,
    body: PathRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/path", "read")],
) -> dict[str, Any]:
    """How two nodes are related: the shortest path over visible, live edges,
    at most ``max_hops`` long, delivered whole or not at all under the budget."""
    view = _view(principal, body)
    budget = _budget(request, body.budget_tokens)
    result = paths.path(
        _engine(request),
        view=view,
        budget_tokens=budget,
        src=body.src,
        dst=body.dst,
        max_hops=body.max_hops,
    )
    if result is None:
        raise HTTPException(404, "no such node")
    log.event("path", found=result["found"], hops=result["hops"], budget=budget)
    return result


@router.get("/lookup")
def lookup(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/lookup", "read")],
    external_id: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    name: Annotated[str | None, Query(min_length=1, max_length=300)] = None,
    scope: ScopeQ = None,
    scopes: ScopesQ = None,
    subtree: ScopeQ = None,
) -> dict[str, Any]:
    """``external_id`` -> the visible records carrying it; ``name`` -> the visible
    nodes with that name or alias (exactly one of the two)."""
    if (external_id is None) == (name is None):
        raise HTTPException(422, "give exactly one of external_id or name")
    view = _query_view(principal, scope, scopes, subtree)
    store = _store(request)
    records = (
        store.lookup_external(external_id, view)
        if external_id is not None
        else store.lookup_name(name or "", view)
    )
    return {"external_id": external_id, "name": name, "records": records}


@router.get("/archive")
def list_archived(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/archive", "read")],
    scope: ScopeQ = None,
    scopes: ScopesQ = None,
    subtree: ScopeQ = None,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    view = _query_view(principal, scope, scopes, subtree)
    budget = _budget(request, budget_tokens)
    engine = _engine(request)
    items = _store(request).archived(view)
    result = pack([engine.cost(i) for i in items], budget)
    return {
        "items": [items[i] for i in result.chosen],
        "used_tokens": result.used_tokens,
        "budget_tokens": budget,
        "omitted": result.omitted,
        "reason": result.reason,
    }


# --------------------------------------------------------------------------- writing


@router.post("/remember", status_code=201)
def remember(
    request: Request,
    body: RememberRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/remember", "write")],
) -> RememberResponse:
    store = _store(request)
    try:
        scopes = auth.require_write_scopes(principal, body.target_scopes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.node and body.node_id:
        raise HTTPException(422, "give node or node_id, not both")
    view = Visibility.of_scope(scopes[0]) if len(scopes) == 1 else Visibility.of_scopes(scopes)

    node = None
    created_node = False
    with store.lock:
        if body.node_id:
            node = store.nodes_by_id([body.node_id], view).get(body.node_id)
            if node is None:
                raise HTTPException(404, "no such node")
        elif body.node:
            node = store.find_node(body.node.name, view)
            if node is None:
                node = store.create_node(
                    scopes=scopes,
                    name=body.node.name,
                    kind=body.node.kind,
                    aliases=body.node.aliases,
                    summary=body.node.summary,
                    external_id=body.node.external_id,
                )
                created_node = True
            elif body.node.aliases or body.node.summary:
                node = store.update_node(node.id, add_aliases=body.node.aliases)
        source = store.add_source(
            scopes=scopes, kind="remember", text=body.statement, reference=body.source_reference
        )
        fact = store.add_fact(
            scopes=scopes,
            statement=body.statement,
            origin="explicit",
            node_id=node.id if node else None,
            source_id=source.id,
            excerpt=(0, len(body.statement)),
            attribute=body.attribute,
            value=body.value,
            valid_from=body.valid_from,
            valid_to=body.valid_to,
            confidence=body.confidence,
            pinned=body.pinned,
            external_id=body.external_id,
        )
    _worker(request).request_embeddings(scopes)  # tier 3: enqueue only (SPEC 6.1)
    log.event(
        "remember",
        fact=fact.id,
        node=node.id if node else None,
        created_node=created_node,
        scopes=len(scopes),
    )
    return RememberResponse(
        fact={
            "id": fact.id,
            "kind": "fact",
            "scopes": list(fact.scopes),
            "external_id": fact.external_id,
        },
        node={"id": node.id, "kind": "node", "scopes": list(node.scopes), "name": node.name}
        if node
        else None,
        source_id=source.id,
        created_node=created_node,
    )


@router.post("/link", status_code=201)
def link(
    request: Request,
    body: LinkRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/link", "write")],
) -> dict[str, Any]:
    store = _store(request)
    try:
        scopes = auth.require_write_scopes(principal, body.target_scopes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    view = Visibility.of_scope(scopes[0]) if len(scopes) == 1 else Visibility.of_scopes(scopes)
    with store.lock:
        ends = store.nodes_by_id([body.src, body.dst], view)
        if body.src not in ends or body.dst not in ends:
            raise HTTPException(404, "no such node")
        edge = store.add_edge(
            scopes=scopes,
            src=body.src,
            dst=body.dst,
            type=body.type,
            origin="explicit",
            status="confirmed",
            weight=body.weight,
            external_id=body.external_id,
        )
    log.event("link", edge=edge.id, src=edge.src, dst=edge.dst)
    return {
        "id": edge.id,
        "kind": "edge",
        "scopes": list(edge.scopes),
        "status": edge.status,
        "external_id": edge.external_id,
    }


def _touchable(store: Store, principal: auth.Principal, record_id: str) -> tuple[str, ...]:
    try:
        scopes = store.scopes_of(record_id)
    except NotFound as exc:
        raise HTTPException(404, "no such record") from exc
    auth.require_touch(principal, scopes)
    return scopes


@router.post("/archive")
def archive(
    request: Request,
    body: RecordRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/archive", "write")],
) -> dict[str, Any]:
    store = _store(request)
    with store.lock:
        _touchable(store, principal, body.id)
        try:
            kind = store.archive(body.id)
        except NotFound as exc:
            raise HTTPException(404, "no such record") from exc
    log.event("archive", record=body.id, kind=kind)
    return {"id": body.id, "kind": kind, "archived": True}


@router.post("/restore")
def restore(
    request: Request,
    body: RecordRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/restore", "write")],
) -> dict[str, Any]:
    store = _store(request)
    with store.lock:
        _touchable(store, principal, body.id)
        try:
            kind = store.restore(body.id)
        except NotFound as exc:
            raise HTTPException(404, "no such record") from exc
    log.event("restore", record=body.id, kind=kind)
    return {"id": body.id, "kind": kind, "archived": False}


@router.post("/forget")
def forget(
    request: Request,
    body: ForgetRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/forget", "forget")],
) -> dict[str, Any]:
    store = _store(request)
    with store.lock:
        _touchable(store, principal, body.id)
        try:
            removed = store.forget(body.id)
        except NotFound as exc:
            raise HTTPException(404, "no such record") from exc
    log.event("forget", record=body.id, **{f"removed_{k}": v for k, v in removed.items()})
    return {"id": body.id, "removed": removed}


@router.patch("/facts/{fact_id}")
def patch_fact(
    request: Request,
    fact_id: str,
    body: FactPatch,
    principal: Annotated[auth.Principal, auth.declare("PATCH", "/v1/facts/{fact_id}", "write")],
) -> dict[str, Any]:
    store = _store(request)
    if body.node_id and body.detach:
        raise HTTPException(422, "give node_id or detach, not both")
    with store.lock:
        scopes = _touchable(store, principal, fact_id)
        try:
            fact = store.get_fact(fact_id)
        except NotFound as exc:
            raise HTTPException(404, "no such record") from exc
        node_id = fact.node_id
        if body.detach:
            node_id = None
        elif body.node_id:
            view = Visibility.of_scopes(scopes)
            if body.node_id not in store.nodes_by_id([body.node_id], view):
                raise HTTPException(404, "no such node")
            node_id = body.node_id
        fact = store.attach_fact(fact_id, node_id, pinned=body.pinned)
    log.event("patch_fact", fact=fact.id, node=fact.node_id, pinned=fact.pinned)
    return {
        "id": fact.id,
        "kind": "fact",
        "node_id": fact.node_id,
        "pinned": fact.pinned,
        "scopes": list(fact.scopes),
    }


@router.post("/proposals/{proposal_id}")
def decide_proposal(
    request: Request,
    proposal_id: str,
    body: ProposalDecision,
    principal: Annotated[
        auth.Principal, auth.declare("POST", "/v1/proposals/{proposal_id}", "write")
    ],
) -> dict[str, Any]:
    """Accept (apply: an alias proposal adds the alias, a fact proposal writes an
    explicit fact) or reject a proposal (SPEC 4.4b), by a token bound to one of its
    scopes; a decided one is not decided again. A fact proposal - a fact about the
    owner that was not the owner's words (P6) - is decided only with ``review``:
    an agent fooled by the page it came from must neither apply nor hide it."""
    return decide_proposal_item(request, principal, proposal_id, body.decision == "accept")


def decide_proposal_item(
    request: Request,
    principal: auth.Principal,
    proposal_id: str,
    accept: bool,
    *,
    from_review: bool = False,
) -> dict[str, Any]:
    """The decision on one proposal, shared by ``/v1/proposals`` and the review
    queue (SPEC 8.6). A merge proposal is never accepted there: only its reviewed
    ``merge_nodes`` merges (SPEC 4.4b); a reviewer may reject it outright."""
    store = _store(request)
    if kind_of(proposal_id) != "prop":
        raise HTTPException(404, "no such record")
    with store.lock:
        scopes = _touchable(store, principal, proposal_id)
        try:
            proposal = store.get_proposal(proposal_id)
        except NotFound as exc:
            raise HTTPException(404, "no such record") from exc
        if proposal.kind == "fact" and not principal.can("review"):
            raise HTTPException(403, "a fact proposal needs the 'review' permission")
        if from_review and proposal.kind == "merge" and accept:
            raise HTTPException(409, "a merge is accepted through its dream operation")
        if (
            accept
            and proposal.node_id is not None
            and proposal.node_id
            not in store.nodes_by_id([proposal.node_id], _write_view(list(scopes)))
        ):
            raise HTTPException(404, "no such node")
        try:
            proposal = store.decide_proposal(proposal_id, accept=accept)
        except ProposalDecided as exc:
            raise HTTPException(409, f"proposal already {proposal.status}") from exc
        except ProposalNotApplicable as exc:
            raise HTTPException(
                409,
                "a merge proposal is never accepted directly: a dream's merge_nodes, "
                "approved in review, merges",
            ) from exc
    if proposal.status == "accepted":
        _worker(request).request_embeddings(list(proposal.scopes))  # a new alias, a new name vector
    log.event("proposal", proposal=proposal.id, kind=proposal.kind, status=proposal.status)
    out = {
        "id": proposal.id,
        "kind": proposal.kind,
        "node_id": proposal.node_id,
        "value": proposal.value,
        "status": proposal.status,
        "scopes": list(proposal.scopes),
    }
    if proposal.applied_id is not None:
        out["fact_id"] = proposal.applied_id
    return out


@router.get("/proposals")
def list_proposals(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/proposals", "read")],
    scope: ScopeQ = None,
    scopes: ScopesQ = None,
    subtree: ScopeQ = None,
    kind: Annotated[Literal["alias", "merge", "fact"] | None, Query()] = None,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """The pending proposals visible in the view, oldest first, under budget
    (SPEC 4.4b, 8.3): the only place a fact proposal without a node shows."""
    view = _query_view(principal, scope, scopes, subtree)
    budget = _budget(request, budget_tokens)
    engine = _engine(request)
    items = [proposal_item(p) for p in _store(request).pending_proposals(view, kind=kind)]
    result = pack([engine.cost(i) for i in items], budget)
    return {
        "items": [items[i] for i in result.chosen],
        "used_tokens": result.used_tokens,
        "budget_tokens": budget,
        "omitted": result.omitted,
        "reason": result.reason,
    }


def proposal_item(proposal: Proposal) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": proposal.id,
        "kind": proposal.kind,
        "node_id": proposal.node_id,
        "value": proposal.value,
        "source_id": proposal.source_id,
        "created_at": proposal.created_at,
        "scopes": list(proposal.scopes),
    }
    if proposal.kind == "fact":
        # the statement is ``value``; the fact's own key and value ride beside it
        data = proposal.payload
        item["attribute"] = data.get("attribute")
        item["fact_value"] = data.get("value")
        item["provenance"] = data.get("provenance")
    return item


# --------------------------------------------------------------------------- probe sets (SPEC 8.7)


def _probe_set(store: Store, principal: auth.Principal, set_id: str) -> ProbeSet:
    """The set, or 404 when it does not exist or the token is not bound to its
    scope (it does not exist for that token, P9)."""
    if kind_of(set_id) != "pset":
        raise HTTPException(404, "no such probe set")
    try:
        probe_set = store.get_probe_set(set_id)
    except NotFound as exc:
        raise HTTPException(404, "no such probe set") from exc
    if not principal.may_write([probe_set.scope]):
        raise HTTPException(404, "no such probe set")
    return probe_set


@router.post("/probes", status_code=201)
def create_probe_set(
    request: Request,
    body: ProbeSetCreate,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/probes", "dream")],
) -> dict[str, Any]:
    """Store a probe set (SPEC 11.2). Its expected records must be visible from its
    scope; changing what gates a dream is the ``dream`` permission's."""
    store = _store(request)
    auth.require_write_scopes(principal, [body.scope])
    budget = _budget(request, body.budget_tokens)
    try:
        created = store.create_probe_set(
            scope=body.scope,
            name=body.name,
            budget_tokens=budget,
            probes=[(p.query, p.expected) for p in body.probes],
        )
    except ScopeMismatch as exc:
        raise HTTPException(422, "an expected record is not visible from the set's scope") from exc
    log.event("probe_set", probe_set=created.id, probes=len(created.probes), action="created")
    return {"id": created.id, "scope": created.scope, "probes": len(created.probes)}


@router.get("/probes")
def list_probe_sets(
    request: Request,
    scope: Annotated[str, Query()],
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/probes", "read")],
) -> dict[str, Any]:
    auth.require_write_scopes(principal, [scope])
    sets = _store(request).probe_sets(scope)
    return {
        "sets": [
            {
                "id": s.id,
                "name": s.name,
                "budget_tokens": s.budget_tokens,
                "probes": size,
                "created_at": s.created_at,
            }
            for s, size in sets
        ]
    }


@router.get("/probes/{set_id}")
def get_probe_set(
    request: Request,
    set_id: str,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/probes/{set_id}", "read")],
) -> dict[str, Any]:
    probe_set = _probe_set(_store(request), principal, set_id)
    return {
        "id": probe_set.id,
        "scope": probe_set.scope,
        "name": probe_set.name,
        "budget_tokens": probe_set.budget_tokens,
        "created_at": probe_set.created_at,
        "probes": [
            {"id": p.id, "query": p.query, "expected": list(p.expected)} for p in probe_set.probes
        ],
    }


@router.post("/probes/{set_id}/run")
def run_probes(
    request: Request,
    set_id: str,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/probes/{set_id}/run", "read")],
) -> dict[str, Any]:
    """Run a probe set now: the probe score (SPEC 11.2), ids and numbers only."""
    store = _store(request)
    probe_set = _probe_set(store, principal, set_id)
    score = run_probe_set(_engine(request), store, probe_set)
    log.event("probe_run", probe_set=probe_set.id, scored=score.scored)
    return score.as_dict()


@router.delete("/probes/{set_id}")
def delete_probe_set(
    request: Request,
    set_id: str,
    principal: Annotated[auth.Principal, auth.declare("DELETE", "/v1/probes/{set_id}", "dream")],
) -> dict[str, Any]:
    store = _store(request)
    probe_set = _probe_set(store, principal, set_id)
    deleted = store.delete_probe_set(probe_set.id)
    log.event("probe_set", probe_set=probe_set.id, action="deleted")
    return {"id": probe_set.id, "deleted": deleted}


# --------------------------------------------------------------------------- lifecycle (SPEC 8.1)


def _write_view(scopes: list[str]) -> Visibility:
    return Visibility.of_scope(scopes[0]) if len(scopes) == 1 else Visibility.of_scopes(scopes)


def _queued(job: Job, worker: Worker) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "queue": {"depth": worker.queue.depth(), "mode": worker.mode},
    }


@router.post("/session/start")
def session_start(
    request: Request,
    body: SessionStartRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/session/start", "read")],
) -> dict[str, Any]:
    """The profile block: pinned facts and the most central nodes of the view,
    under budget, deterministic and stable while nothing changes."""
    view = _view(principal, body)
    budget = _budget(request, body.budget_tokens)
    result = profile_block(
        _engine(request), view=view, budget_tokens=budget, exclude_ids=body.exclude_ids
    )
    log.event(
        "session_start",
        items=len(result["items"]),
        omitted=result["omitted"],
        used=result["used_tokens"],
        budget=budget,
    )
    return {**result, "session_id": body.session_id}


@router.post("/turns", status_code=202)
def record_turn(
    request: Request,
    body: TurnRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/turns", "write")],
) -> dict[str, Any]:
    """Record a turn: one insert into the persisted queue, nothing else (P3)."""
    try:
        scopes = auth.require_write_scopes(principal, body.target_scopes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    worker = _worker(request)
    payload = {
        "user": body.user,
        "assistant": body.assistant,
        "tool_summaries": list(body.tool_summaries),
        "external": [part.model_dump() for part in body.external],
        "occurred_at": body.occurred_at or _store(request).now(),
        "external_id": body.external_id,
    }
    job = worker.enqueue(TURN, payload, scopes=scopes, session_id=body.session_id)
    log.event(
        "turn",
        job=job.id,
        chars=len(body.user) + len(body.assistant),
        tools=len(body.tool_summaries),
        external=len(body.external),
    )
    return _queued(job, worker)


@router.post("/precompact")
def precompact(
    request: Request,
    body: PrecompactRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/precompact", "write")],
) -> dict[str, Any]:
    """The harness is about to compact: the messages it will drop are queued for
    extraction, and a short block of what must survive comes back under budget.
    The route is a write; the block needs ``read`` too and is empty without it."""
    try:
        scopes = auth.require_write_scopes(principal, body.target_scopes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    worker = _worker(request)
    payload = {
        "trigger": body.trigger,
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
    }
    job = worker.enqueue(PRECOMPACT, payload, scopes=scopes, session_id=body.session_id)
    budget = _budget(request, body.budget_tokens)
    if principal.can("read"):
        block = profile_block(
            _engine(request),
            view=_write_view(scopes),
            budget_tokens=budget,
            exclude_ids=body.exclude_ids,
        )
    else:
        block = {
            "items": [],
            "used_tokens": 0,
            "budget_tokens": budget,
            "omitted": 0,
            "reason": "read_permission_required",
            "tier": _engine(request).tier,
            "degraded": None,
        }
    log.event("precompact", job=job.id, messages=len(body.messages), items=len(block["items"]))
    return {**block, **_queued(job, worker)}


@router.post("/session/end", status_code=202)
def session_end(
    request: Request,
    body: SessionEndRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/session/end", "write")],
) -> dict[str, Any]:
    """Flush and extract the session: queued, processed in the background."""
    try:
        scopes = auth.require_write_scopes(principal, body.target_scopes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    worker = _worker(request)
    job = worker.enqueue(
        SESSION_END, {"reason": body.reason}, scopes=scopes, session_id=body.session_id
    )
    log.event("session_end", job=job.id)
    return _queued(job, worker)


# --------------------------------------------------------------------------- operations (SPEC 8.4)


@router.get("/background")
def background(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/background", "admin")],
) -> dict[str, Any]:
    return _worker(request).status()


@router.put("/background")
def update_background(
    request: Request,
    body: BackgroundUpdate,
    principal: Annotated[auth.Principal, auth.declare("PUT", "/v1/background", "admin")],
) -> dict[str, Any]:
    """Change the mode (persisted: a paused service comes back paused) and the
    concurrency toward each model endpoint (P11)."""
    worker = _worker(request)
    if body.mode is not None:
        worker.set_mode(body.mode)
    if body.concurrency is not None:
        for kind in ("chat", "embeddings"):
            limit = getattr(body.concurrency, kind)
            if limit is not None:
                worker.set_concurrency(kind, limit)
    return worker.status()


@router.post("/background/run")
def run_background(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/background/run", "admin")],
    body: RunRequest | None = None,
) -> dict[str, Any]:
    """Process the queue now, under an optional budget (calls, tokens, seconds,
    jobs); answers with a run id and what it processed. Refused while paused."""
    body = body or RunRequest()
    limits = body.budget
    budget = Budget(
        max_calls=limits.calls if limits else None,
        max_tokens=limits.tokens if limits else None,
        max_seconds=limits.seconds if limits else None,
    )
    try:
        report = _worker(request).run(budget, max_jobs=body.max_jobs)
    except RunRefused as exc:
        raise HTTPException(409, f"run refused: background is {exc.reason}") from exc
    return report.as_dict()


@router.post("/background/yield")
def yield_background(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/background/yield", "write")],
    body: YieldRequest | None = None,
) -> dict[str, Any]:
    """A foreground turn starts: the worker finishes its current model call and
    starts no other until ``resume`` or ``timeout_seconds`` pass."""
    seconds = _worker(request).yield_for(body.timeout_seconds if body else None)
    return {"yielded": True, "timeout_seconds": seconds}


@router.post("/background/resume")
def resume_background(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/background/resume", "write")],
) -> dict[str, Any]:
    """The foreground turn is done: background model calls may start again."""
    _worker(request).resume()
    return {"yielded": False}
