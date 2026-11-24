"""Routes under ``/v1``. Every route declares its permission; every content-returning
route resolves a view and a budget before touching the store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response

from iifa import __version__, calibration, log, vectors
from iifa.api import auth
from iifa.api.paging import MAX_CURSOR_CHARS, Cursors, read_cursor, view_context
from iifa.api.schemas import (
    AtRequest,
    BackgroundUpdate,
    EdgeDecision,
    ExpandRequest,
    FactPatch,
    ForgetRequest,
    LinkRequest,
    NodeMerge,
    NodePatch,
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
from iifa.ids import kind_of, new_id
from iifa.ingest.jobs import PRECOMPACT, SESSION_END, TURN
from iifa.ingest.keys import canonical_key, rule_twin
from iifa.ingest.queue import RUNNING, WORD_KINDS, Job, JobStarted
from iifa.ingest.schema import COOCCURRENCE
from iifa.ingest.worker import RUN_CANCELLING, Budget, RunRefused, Worker
from iifa.llm import ModelClient
from iifa.names import InvalidName
from iifa.recall import history, paths
from iifa.recall.dense import SuppliedVector
from iifa.recall.engine import RecallEngine
from iifa.recall.paging import take
from iifa.recall.probes import run_probe_set
from iifa.recall.profile import profile_block
from iifa.recall.tokenizer import normalize
from iifa.scopes import Visibility
from iifa.store.corrections import (
    MERGE_NODES,
    Correction,
    CorrectionRefused,
    Corrections,
    default_keep,
)
from iifa.store.models import Edge, Fact, Node, ProbeSet, Proposal
from iifa.store.repo import (
    PROPOSED,
    NotFound,
    ProposalDecided,
    ProposalNotApplicable,
    ScopeMismatch,
    Store,
    checked_name,
)

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


def _cursors(request: Request) -> Cursors:
    cursors: Cursors = request.app.state.cursors
    return cursors


LIST_LIMIT = 100
LIST_LIMIT_MAX = 1000


def _next_cursor(
    request: Request, context: dict[str, Any], position: Sequence[Any] | None
) -> str | None:
    return None if position is None else _cursors(request).encode(context, position)


def _view(principal: auth.Principal, body: Any) -> Visibility:
    return auth.resolve_view(principal, scope=body.scope, scopes=body.scopes, subtree=body.subtree)


ScopeQ = Annotated[str | None, Query()]
ScopesQ = Annotated[list[str] | None, Query()]
CursorQ = Annotated[str | None, Query(max_length=MAX_CURSOR_CHARS)]
LimitQ = Annotated[int, Query(ge=1, le=LIST_LIMIT_MAX)]


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
    paused queue never fills in silence, P11), the background mode and the run in
    progress (ids and counts, so every client sees a run it did not start).
    Answering makes no model call."""
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
            "run": status["run"],
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


@router.get("/nodes")
def list_nodes(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/nodes", "read")],
    scope: ScopeQ = None,
    scopes: ScopesQ = None,
    subtree: ScopeQ = None,
    prefix: Annotated[str | None, Query(min_length=1, max_length=300)] = None,
    sort: Annotated[Literal["name", "changed"], Query()] = "name",
    include_archived: bool = False,
    cursor: CursorQ = None,
    limit: LimitQ = LIST_LIMIT,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """The nodes the view sees, a page at a time (SPEC 8.2): id, name, kind,
    aliases, live facts, pending proposals and last change, by name or by last
    change. ``prefix`` matches the start of the name or of an alias, compared in
    their normalised form (case and Latin diacritics folded). A walk lists the
    nodes that existed at its first page, with their last change as of then."""
    view = _query_view(principal, scope, scopes, subtree)
    budget = _budget(request, budget_tokens)
    norm = None
    if prefix is not None:
        norm = normalize(prefix)
        if not norm:
            raise HTTPException(422, "prefix needs a letter or a digit")
    context = {
        "route": "nodes",
        "view": view_context(view),
        "sort": sort,
        "prefix": norm,
        "include_archived": include_archived,
    }
    position = read_cursor(_cursors(request), cursor, context, str, str, str)
    mark = position[0] if position else new_id("node").split("_", 1)[1]
    rows = _store(request).node_list(
        view,
        sort=sort,
        prefix=norm,
        include_archived=include_archived,
        mark=mark,
        after=(position[1], position[2]) if position else None,
        limit=limit + 1,
    )
    sort_key = "name_norm" if sort == "name" else "changed_at"
    engine = _engine(request)
    pairs = [(row, {k: v for k, v in row.items() if k != "name_norm"}) for row in rows]
    page = take(pairs, limit=limit, budget_tokens=budget, cost=lambda p: engine.cost(p[1]))
    last = page.last[0] if page.more and page.last else None
    return {
        "items": [item for _, item in page.items],
        "sort": sort,
        "next_cursor": _next_cursor(
            request, context, [mark, last[sort_key], last["id"]] if last else None
        ),
        "used_tokens": page.used_tokens,
        "budget_tokens": budget,
        "omitted": page.omitted,
        "reason": page.reason,
    }


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
    """Fact changes of a node, or of the whole view, inside a date range (P7),
    newest first, a page at a time (SPEC 8)."""
    view = _view(principal, body)
    budget = _budget(request, body.budget_tokens)
    context = {
        "route": "timeline",
        "view": view_context(view),
        "node_id": body.node_id,
        "since": body.since,
        "until": body.until,
    }
    position = read_cursor(_cursors(request), body.cursor, context, str, str, str)
    try:
        result = history.timeline(
            _engine(request),
            view=view,
            budget_tokens=budget,
            node_id=body.node_id,
            since=body.since,
            until=body.until,
            limit=body.limit,
            before=(position[0], position[1], position[2]) if position else None,
        )
    except ValueError as exc:
        raise HTTPException(422, f"invalid date: {exc}") from exc
    if result is None:
        raise HTTPException(404, "no such node")
    result["next_cursor"] = _next_cursor(request, context, result.pop("next"))
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
    cursor: CursorQ = None,
    limit: LimitQ = LIST_LIMIT,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """Archived nodes, facts and edges the view sees, most recently archived
    first, a page at a time (SPEC 8)."""
    view = _query_view(principal, scope, scopes, subtree)
    budget = _budget(request, budget_tokens)
    context = {"route": "archive", "view": view_context(view)}
    position = read_cursor(_cursors(request), cursor, context, str, str)
    engine = _engine(request)
    items = _store(request).archived(
        view, before=(position[0], position[1]) if position else None, limit=limit + 1
    )
    page = take(items, limit=limit, budget_tokens=budget, cost=engine.cost)
    last = page.last if page.more else None
    return {
        "items": page.items,
        "next_cursor": _next_cursor(
            request, context, [last["archived_at"], last["id"]] if last else None
        ),
        "used_tokens": page.used_tokens,
        "budget_tokens": budget,
        "omitted": page.omitted,
        "reason": page.reason,
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
    if body.about_self:
        return _remember_self(request, body, scopes)
    if body.node and body.node_id:
        raise HTTPException(422, "give node or node_id, not both")
    if body.node:
        try:  # the one name rule (SPEC 5.7), checked before anything is written
            for name in (body.node.name, *(a for a in body.node.aliases if a.strip())):
                checked_name(name)
        except InvalidName as exc:
            raise HTTPException(422, str(exc)) from exc
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


def _remember_self(request: Request, body: RememberRequest, scopes: list[str]) -> RememberResponse:
    """SPEC 8.3, the agent remembering about itself: on the self's node, origin
    ``extracted``, provenance ``assistant``. No model is called (P3): the same
    thing said again is an echo and writes nothing; another value of a key the
    self holds is written in conflict with it, never closing it. A rule's key
    is also its topic under the other rule family (SPEC 4.2)."""
    if body.node or body.node_id:
        raise HTTPException(422, "about_self names the node: give neither node nor node_id")
    agent = _worker(request).pipeline.agent(scopes)
    if agent is None:
        raise HTTPException(422, "about_self: no self is declared for these scopes")
    store = _store(request)
    view = Visibility.of_scope(scopes[0]) if len(scopes) == 1 else Visibility.of_scopes(scopes)
    key = canonical_key(body.attribute)
    value = normalize(body.value or "")
    with store.lock:
        node = store.find_node(agent.name, Visibility.of_scope(agent.scope))
        created_node = node is None
        if node is None:
            node = store.create_node(
                scopes=[agent.scope], name=agent.name, kind="other", aliases=agent.aliases
            )
        live = store.facts([node.id], view)
        # SPEC 4.2: the key's own facts first; when none holds a value, the
        # same topic under the other rule family stands for the key
        same_key = [f for f in live if key is not None and canonical_key(f.attribute) == key]
        if not any(f.value for f in same_key):
            same_key = [f for f in live if rule_twin(key, canonical_key(f.attribute))]
        echoed = next(
            (
                f
                for f in live
                if normalize(f.statement) == normalize(body.statement)
                or (value and f in same_key and normalize(f.value or "") == value)
            ),
            None,
        )
        held = next((f for f in same_key if f.value), None) if value else None
        if echoed is None:
            source = store.add_source(scopes=scopes, kind="remember", text=body.statement)
            fact = store.add_fact(
                scopes=scopes,
                statement=body.statement,
                origin="extracted",
                node_id=node.id,
                source_id=source.id,
                excerpt=(0, len(body.statement)),
                attribute=key,
                value=body.value,
                valid_from=body.valid_from,
                valid_to=body.valid_to,
                confidence=1.0,
                pinned=body.pinned,
                external_id=body.external_id,
                flags=["conflict"] if held is not None else [],
                conflicts_with=held.id if held is not None else None,
                provenance="assistant",
                part_ref=agent.name,
            )
    if echoed is not None:
        log.event("remember", fact=echoed.id, node=node.id, echo=True, scopes=len(scopes))
        return _remembered(echoed, node, None, created_node, echo=True)
    _worker(request).request_embeddings(scopes)  # tier 3: enqueue only (SPEC 6.1)
    log.event("remember", fact=fact.id, node=node.id, created_node=created_node, scopes=len(scopes))
    return _remembered(fact, node, source.id, created_node)


def _remembered(
    fact: Fact, node: Node, source_id: str | None, created_node: bool, *, echo: bool = False
) -> RememberResponse:
    return RememberResponse(
        fact={
            "id": fact.id,
            "kind": "fact",
            "scopes": list(fact.scopes),
            "external_id": fact.external_id,
        },
        node={"id": node.id, "kind": "node", "scopes": list(node.scopes), "name": node.name},
        source_id=source_id,
        created_node=created_node,
        echo=echo,
    )


def _edge_out(edge: Edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "kind": "edge",
        "scopes": list(edge.scopes),
        "status": edge.status,
        "external_id": edge.external_id,
        "rev": edge.decision_rev,  # an edge's API rev is its decision revision (SPEC 4.4)
    }


@router.post("/link", status_code=201, responses={200: {"description": "Live edge confirmed"}})
def link(
    request: Request,
    response: Response,
    body: LinkRequest,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/link", "write")],
) -> dict[str, Any]:
    """An explicit, confirmed edge (SPEC 5.2). When the pair already has the live
    edge of that type the token touches, that edge is confirmed (journalled as a
    person's decision, SPEC 5.7) and returned rather than a twin added."""
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
        existing = store.live_edge(
            body.src, body.dst, body.type, view, either_way=body.type == COOCCURRENCE
        )
        if (
            existing is not None
            and principal.may_touch(existing.scopes)
            and body.external_id in (None, existing.external_id)
        ):
            correction_id = None
            if existing.status == PROPOSED:
                existing, entry = Corrections(store).decide_edge(
                    existing.id, "confirm", token=principal.name
                )
                correction_id = entry.id
            response.status_code = 200
            log.event("link", edge=existing.id, created=False, correction=correction_id)
            return {**_edge_out(existing), "created": False, "correction_id": correction_id}
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
    log.event("link", edge=edge.id, src=edge.src, dst=edge.dst, created=True)
    return {**_edge_out(edge), "created": True, "correction_id": None}


@router.post("/edges/{edge_id}")
def decide_edge(
    request: Request,
    edge_id: str,
    body: EdgeDecision,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/edges/{edge_id}", "write")],
) -> dict[str, Any]:
    """A person's decision on a proposed edge (SPEC 5.7): ``confirm`` makes it
    confirmed, ``reject`` archives it - journalled with the token's name, undoable."""
    store = _store(request)
    if kind_of(edge_id) != "edge":
        raise HTTPException(404, "no such edge")
    with store.lock:
        _touchable(store, principal, edge_id)
        try:
            edge, entry = Corrections(store).decide_edge(
                edge_id, body.decision, token=principal.name, rev=body.rev
            )
        except CorrectionRefused as exc:
            raise _refused(exc) from exc
    log.event("decide_edge", edge=edge.id, decision=body.decision, correction=entry.id)
    return {
        **_edge_out(edge),
        "src": edge.src,
        "dst": edge.dst,
        "type": edge.type,
        "archived": edge.archived_at is not None,
        "correction_id": entry.id,
    }


def _require_rescope(
    principal: auth.Principal, current: Sequence[str], wanted: Sequence[str]
) -> None:
    """SPEC 4.6: a token grants and takes away only scopes it owns - every scope
    added and every scope removed is at or below one of its bound scopes."""
    changed = sorted(set(current) ^ set(wanted))
    if changed and not principal.may_write(changed):
        raise HTTPException(
            403, "a re-scope may add or remove only scopes inside the token's scopes"
        )


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
    if kind_of(body.id) == "job":
        return _forget_job(request, principal, body.id)
    store = _store(request)
    with store.lock:
        _touchable(store, principal, body.id)
        try:
            removed = store.forget(body.id)
        except NotFound as exc:
            raise HTTPException(404, "no such record") from exc
    log.event("forget", record=body.id, **{f"removed_{k}": v for k, v in removed.items()})
    return {"id": body.id, "removed": removed}


def _forget_job(request: Request, principal: auth.Principal, job_id: str) -> dict[str, Any]:
    """P5: a turn or a compaction dropped before it runs, its payload deleted
    with it. A running or done one is refused (the worker has read it: what it
    wrote is forgotten by its source), and so is a job that carries no words
    (a dream, a session's close: its own route stops it). ``started``: it ran
    before (failed, or interrupted by a crash) and may have written."""
    queue = _worker(request).queue
    gone = "no such job (a job that has run is pruned after a while: forget its source)"
    try:
        job = queue.get(job_id)
    except NotFound as exc:
        raise HTTPException(404, gone) from exc
    auth.require_touch(principal, job.scopes)
    if job.kind not in WORD_KINDS:
        raise HTTPException(409, "only a turn or a compaction waiting to run is forgotten")
    try:
        dropped = queue.drop(job_id)
    except NotFound as exc:
        raise HTTPException(404, gone) from exc
    except JobStarted as exc:
        reason = (
            "the job is running: forget its source once it is done"
            if exc.status == RUNNING
            else "the job has run: forget its source"
        )
        raise HTTPException(409, reason) from exc
    log.event("forget", record=job_id, removed_jobs=1, started=dropped.started)
    return {"id": job_id, "removed": {"jobs": 1}, "started": dropped.started}


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
        if body.scopes is not None:
            _require_rescope(principal, scopes, body.scopes)
            scopes = tuple(body.scopes)
        node_id = fact.node_id
        if body.detach:
            node_id = None
        elif body.node_id:
            view = Visibility.of_scopes(scopes)
            if body.node_id not in store.nodes_by_id([body.node_id], view):
                raise HTTPException(404, "no such node")
            node_id = body.node_id
        elif (
            body.scopes is not None
            and node_id is not None
            and not Visibility.of_scopes(scopes).sees(store.scopes_of(node_id))
        ):
            # SPEC 4.6: archiving or forgetting the node reaches the fact, so the
            # fact may not leave every scope its node is in.
            raise HTTPException(
                409,
                "the fact's node is not visible from the new scopes:"
                " detach the fact or give a node_id visible from them",
            )
        fact = store.attach_fact(fact_id, node_id, pinned=body.pinned, scopes=body.scopes)
    log.event(
        "patch_fact",
        fact=fact.id,
        node=fact.node_id,
        pinned=fact.pinned,
        scopes=len(fact.scopes),
    )
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


MERGE_IN_REVIEW = (
    "a merge proposal is accepted in review only (POST /v1/review/{id}):"
    " a person's decision, with the 'review' permission"
)


def decide_proposal_item(
    request: Request,
    principal: auth.Principal,
    proposal_id: str,
    accept: bool,
    *,
    from_review: bool = False,
    keep: str | None = None,
) -> dict[str, Any]:
    """The decision on one proposal, shared by ``/v1/proposals`` and the review
    queue (SPEC 8.6). A merge proposal is accepted only in the review queue,
    where accepting it merges the two nodes as a person's decision (SPEC 5.7);
    ``/v1/proposals``, an agent's ``write`` route, refuses it."""
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
        if keep is not None and not (proposal.kind == "merge" and accept):
            raise HTTPException(422, "keep is given only to accept a merge proposal")
        if proposal.kind == "merge" and accept:
            if not from_review:
                raise HTTPException(409, MERGE_IN_REVIEW)
            return _accept_merge(request, principal, proposal, keep)
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
        except InvalidName as exc:
            raise HTTPException(422, "this alias is not a name (SPEC 5.1): reject it") from exc
        except ProposalNotApplicable as exc:
            raise HTTPException(409, MERGE_IN_REVIEW) from exc
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


def _accept_merge(
    request: Request, principal: auth.Principal, proposal: Proposal, keep: str | None
) -> dict[str, Any]:
    """A reviewer accepts a merge proposal: the two nodes are merged as a person's
    decision (SPEC 5.7). Called under the store lock."""
    if not principal.can("review"):  # the route says so too; a merge never goes without it
        raise HTTPException(403, "a merge needs the 'review' permission")
    if proposal.status != "pending":
        raise HTTPException(409, f"proposal already {proposal.status}")
    store = _store(request)
    pair = [proposal.node_id or "", proposal.value]
    if keep is not None and keep not in pair:
        raise HTTPException(422, "keep must be one of the two nodes of the proposal")
    for node_id in pair:
        _touchable(store, principal, node_id)
    if keep is None:
        first, second = store.get_node(pair[0]), store.get_node(pair[1])
        if first.kind != second.kind:
            # the kept node keeps its kind (SPEC 5.7): the default must not pick it
            raise HTTPException(422, {"reason": "keep_required", "records": pair})
        kept, absorbed = default_keep(store, first, second)
        keep_id, absorb_id = kept.id, absorbed.id
    else:
        keep_id, absorb_id = keep, pair[1] if keep == pair[0] else pair[0]
    entry = _merge(request, principal, keep_id, absorb_id, proposal=proposal)
    out = proposal_item(store.get_proposal(proposal.id))
    out["status"] = "accepted"
    return {**out, **_merge_out(entry)}


def _refused(exc: CorrectionRefused) -> HTTPException:
    status = 422 if exc.reason in ("no_alias", "no_kind", "same_record") else 409
    detail: dict[str, Any] = {"reason": exc.reason, "records": exc.records}
    if exc.revs:
        detail["revs"] = exc.revs
    return HTTPException(status, detail)


def _merge(
    request: Request,
    principal: auth.Principal,
    keep: str,
    absorb: str,
    *,
    proposal: Proposal | None = None,
    revs: dict[str, int] | None = None,
) -> Correction:
    """The one merge path of a person (SPEC 5.7), under the store lock."""
    store = _store(request)
    try:
        entry = Corrections(store).merge(
            keep, absorb, token=principal.name, proposal=proposal, revs=revs
        )
    except NotFound as exc:
        raise HTTPException(404, "no such node") from exc
    except CorrectionRefused as exc:
        raise _refused(exc) from exc
    _worker(request).request_embeddings(list(store.scopes_of(keep)))
    log.event("merge_nodes", correction=entry.id, keep=keep, absorb=absorb)
    return entry


def _merge_out(entry: Correction) -> dict[str, Any]:
    payload = entry.payload or {}
    return {
        "correction_id": entry.id,
        "keep": payload.get("keep"),
        "absorb": payload.get("absorb"),
        "proposal_id": payload.get("proposal"),
    }


@router.patch("/nodes/{node_id}")
def patch_node(
    request: Request,
    node_id: str,
    body: NodePatch,
    principal: Annotated[auth.Principal, auth.declare("PATCH", "/v1/nodes/{node_id}", "write")],
) -> dict[str, Any]:
    """A person's correction of a node (SPEC 5.7): rename it (the old name stays an
    alias unless ``keep_old_name`` is false), change its kind, add and remove
    aliases - journalled with the token's name, undoable."""
    store = _store(request)
    if kind_of(node_id) != "node":
        raise HTTPException(404, "no such node")
    with store.lock:
        _touchable(store, principal, node_id)
        try:
            node, entry = Corrections(store).patch_node(
                node_id,
                token=principal.name,
                name=body.name,
                keep_old_name=body.keep_old_name,
                kind=body.kind,
                add_aliases=body.add_aliases,
                remove_aliases=body.remove_aliases,
                rev=body.rev,
            )
        except NotFound as exc:
            raise HTTPException(404, "no such node") from exc
        except InvalidName as exc:
            raise HTTPException(422, str(exc)) from exc
        except CorrectionRefused as exc:
            raise _refused(exc) from exc
    if entry is not None:
        _worker(request).request_embeddings(list(node.scopes))  # a new name vector
    log.event("patch_node", node=node.id, correction=entry.id if entry else None)
    return {
        "id": node.id,
        "kind": node.kind,
        "name": node.name,
        "aliases": list(node.aliases),
        "scopes": list(node.scopes),
        "rev": node.rev,
        "correction_id": entry.id if entry else None,
    }


@router.post("/nodes/merge")
def merge_nodes(
    request: Request,
    body: NodeMerge,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/nodes/merge", "review")],
) -> dict[str, Any]:
    """Merge two nodes a person knows are one (SPEC 5.7): their merge proposal -
    the pending one, else one recorded now - is accepted through the review
    path, so the same guard and the same journal apply."""
    store = _store(request)
    for node_id in (body.keep, body.absorb):
        if kind_of(node_id) != "node":
            raise HTTPException(404, "no such node")
    if body.revs is not None and not set(body.revs) <= {body.keep, body.absorb}:
        raise HTTPException(422, "revs names the two nodes only")
    with store.lock:
        for node_id in (body.keep, body.absorb):
            _touchable(store, principal, node_id)
        entry = _merge(request, principal, body.keep, body.absorb, revs=body.revs)
    return _merge_out(entry)


def _correction(store: Store, principal: auth.Principal, correction_id: str) -> Correction:
    """The entry, or 404 when it does not exist or the token does not touch every
    record it names that still exists (one of them at least)."""
    if kind_of(correction_id) != "corr":
        raise HTTPException(404, "no such correction")
    try:
        entry = Corrections(store).get(correction_id)
    except NotFound as exc:
        raise HTTPException(404, "no such correction") from exc
    existing = list(store.scope_sets(entry.records).values())
    if not existing or not all(principal.may_touch(s) for s in existing):
        raise HTTPException(404, "no such correction")
    return entry


def _correction_out(entry: Correction) -> dict[str, Any]:
    return {
        "id": entry.id,
        "op": entry.op,
        "status": entry.status,
        "token": entry.token,
        "created_at": entry.created_at,
        "payload": entry.payload,
        "records": list(entry.records),
        "forgotten": entry.forgotten,
        "undone_at": entry.undone_at,
        "undone_by": entry.undone_by,
    }


@router.get("/corrections/{correction_id}")
def get_correction(
    request: Request,
    correction_id: str,
    principal: Annotated[
        auth.Principal, auth.declare("GET", "/v1/corrections/{correction_id}", "read")
    ],
) -> dict[str, Any]:
    """One entry of the corrections journal (SPEC 5.7): what was changed, by which
    token, when, and whether it was undone."""
    store = _store(request)
    with store.lock:
        return _correction_out(_correction(store, principal, correction_id))


@router.post("/corrections/{correction_id}/undo")
def undo_correction(
    request: Request,
    correction_id: str,
    principal: Annotated[
        auth.Principal, auth.declare("POST", "/v1/corrections/{correction_id}/undo", auth.ANY)
    ],
) -> dict[str, Any]:
    """Revert one correction exactly (SPEC 5.7); refused, naming the records, when
    a later write depends on it. It needs the permission that made the
    correction: ``write`` for a patch, ``review`` for a merge."""
    store = _store(request)
    with store.lock:
        entry = _correction(store, principal, correction_id)
        needed = "review" if entry.op == MERGE_NODES else "write"
        if not principal.can(needed):
            raise HTTPException(403, f"undoing this correction needs the '{needed}' permission")
        try:
            entry = Corrections(store).undo(entry.id, token=principal.name)
        except CorrectionRefused as exc:
            raise _refused(exc) from exc
        scopes = sorted({s for v in store.scope_sets(entry.records).values() for s in v})
    _worker(request).request_embeddings(scopes)
    log.event("undo_correction", correction=entry.id, op=entry.op)
    return _correction_out(entry)


@router.get("/proposals")
def list_proposals(
    request: Request,
    principal: Annotated[auth.Principal, auth.declare("GET", "/v1/proposals", "read")],
    scope: ScopeQ = None,
    scopes: ScopesQ = None,
    subtree: ScopeQ = None,
    kind: Annotated[Literal["alias", "merge", "fact"] | None, Query()] = None,
    cursor: CursorQ = None,
    limit: LimitQ = LIST_LIMIT,
    budget_tokens: Annotated[int | None, Query(ge=0)] = None,
) -> dict[str, Any]:
    """The pending proposals visible in the view, oldest first, a page at a time
    under budget (SPEC 4.4b, 8.3): the only place a fact proposal without a node
    shows."""
    view = _query_view(principal, scope, scopes, subtree)
    budget = _budget(request, budget_tokens)
    context = {"route": "proposals", "view": view_context(view), "kind": kind}
    position = read_cursor(_cursors(request), cursor, context, str, str)
    engine = _engine(request)
    proposals = _store(request).pending_proposals(
        view,
        kind=kind,
        after=(position[0], position[1]) if position else None,
        limit=limit + 1,
    )
    items = [proposal_item(p) for p in proposals]
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
        item["why"] = data.get("why")  # SPEC 4.2: "intent:..." when the intent is in doubt
        item["subject"] = data.get("subject")
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
    asked = body.scopes or [body.scope or body.subtree or ""]
    agent = _worker(request).pipeline.agent(asked) if all(asked) else None
    result = profile_block(
        _engine(request),
        view=view,
        budget_tokens=budget,
        exclude_ids=body.exclude_ids,
        agent=(agent.name, agent.scope) if agent is not None else None,
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
    """Record a turn: one insert into the persisted queue, nothing else (P3).
    With ``record: false`` nothing at all: no queue row, no source, no fact."""
    try:
        scopes = auth.require_write_scopes(principal, body.target_scopes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    worker = _worker(request)
    if not body.record:
        log.event("turn_not_recorded")
        return {
            "job_id": None,
            "kind": TURN,
            "status": "not_recorded",
            "queue": {"depth": worker.queue.depth(), "mode": worker.mode},
        }
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


@router.post("/background/run", responses={202: {"description": "Run started (wait: false)"}})
def run_background(
    request: Request,
    response: Response,
    principal: Annotated[auth.Principal, auth.declare("POST", "/v1/background/run", "admin")],
    body: RunRequest | None = None,
) -> dict[str, Any]:
    """Process the queue now, under an optional budget (calls, tokens, seconds,
    jobs); answers with a run id and what it processed. With ``wait: false`` it
    answers 202 with the run id at once; ``GET /background/run/{run_id}`` follows
    it. Refused while paused, and while another run is in progress (409, naming it)."""
    body = body or RunRequest()
    limits = body.budget
    budget = Budget(
        max_calls=limits.calls if limits else None,
        max_tokens=limits.tokens if limits else None,
        max_seconds=limits.seconds if limits else None,
    )
    worker = _worker(request)
    try:
        if not body.wait:
            started = worker.start_run(budget, max_jobs=body.max_jobs, started_by=principal.name)
            response.status_code = 202
            return {"run_id": started.id, "status": "running"}
        report = worker.run(budget, max_jobs=body.max_jobs, started_by=principal.name)
    except RunRefused as exc:
        detail = f"run refused: background is {exc.reason}"
        if exc.run_id is not None:
            detail += f" with {exc.run_id}"
        raise HTTPException(409, detail) from exc
    return report.as_dict()


@router.get("/background/run/{run_id}")
def background_run(
    request: Request,
    run_id: str,
    principal: Annotated[
        auth.Principal, auth.declare("GET", "/v1/background/run/{run_id}", "admin")
    ],
) -> dict[str, Any]:
    """A run's report, live while it runs (the current run and the last ones are
    kept in memory; a restart forgets them)."""
    report = _worker(request).run_report(run_id) if kind_of(run_id) == "run" else None
    if report is None:
        raise HTTPException(404, "no such run")
    return report


@router.post("/background/run/{run_id}/cancel")
def cancel_background_run(
    request: Request,
    run_id: str,
    principal: Annotated[
        auth.Principal, auth.declare("POST", "/v1/background/run/{run_id}/cancel", "admin")
    ],
) -> dict[str, Any]:
    """Stop a run early, as at a budget limit: no new job starts, a job in flight
    is put back untouched at its next model call, and the run ends with
    ``stopped_by: cancelled``. The mode is unchanged; ``cancelled_by`` names the
    token."""
    worker = _worker(request)
    status = worker.cancel_run(run_id, by=principal.name) if kind_of(run_id) == "run" else None
    if status is None:
        raise HTTPException(404, "no such run")
    if status != RUN_CANCELLING:
        raise HTTPException(409, f"the run is {status}")
    return {"run_id": run_id, "status": status}


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
