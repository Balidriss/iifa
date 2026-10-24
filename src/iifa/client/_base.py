"""What the sync and async clients share: request building and error mapping.

Every operation is a pure function that returns a ``Call`` (method, path, query
parameters, JSON body); the two clients only differ in how they send it. Errors
form one small hierarchy so an adapter can fail open on ``IifaUnavailable``
(the service is down or slow) and treat everything else as a real answer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from iifa.transport import UNREACHABLE, classify_transport_error

DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT = 5.0
API = "/v1"


# --------------------------------------------------------------------------- errors


class IifaError(Exception):
    """Base class of every error the client raises."""


class IifaUnavailable(IifaError):
    """The service could not be reached or did not answer in time. Adapters fail
    open on it. ``reason`` is ``unreachable`` (no connection was made, a timeout
    while connecting included), ``timeout`` (connected, no answer in time) or
    ``broken`` (the exchange failed after the connection), see ``iifa.transport``."""

    def __init__(self, message: str, *, reason: str = UNREACHABLE) -> None:
        super().__init__(message)
        self.reason = reason


class IifaProtocolError(IifaError):
    """The service answered with something that is not the JSON iifa speaks."""


class IifaHTTPError(IifaError):
    """The service answered with an error status. ``detail`` is its JSON ``detail``."""

    def __init__(self, status: int, detail: Any, method: str, path: str) -> None:
        self.status = status
        self.detail = detail
        self.method = method
        self.path = path
        super().__init__(f"{method} {path} -> {status}: {_detail_text(detail)}")


class AuthError(IifaHTTPError):
    """401 (no or bad token) or 403 (permission or scope)."""


class NotFoundError(IifaHTTPError):
    """404: no such record, or none the token may see."""


class InvalidRequest(IifaHTTPError):
    """400 or 422: the request did not validate."""


class ServerError(IifaHTTPError):
    """5xx."""


def _detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, ensure_ascii=False)[:300]


# --------------------------------------------------------------------------- calls


@dataclass(frozen=True)
class Call:
    method: str
    path: str
    params: dict[str, Any] | None = None
    json: dict[str, Any] | None = None


def view(
    scope: str | None,
    scopes: Sequence[str] | None,
    subtree: str | None,
    default_scope: str | None,
) -> dict[str, Any]:
    """The read view as request fields: what was given, else the client's default scope.
    The service checks that exactly one mode is present."""
    out: dict[str, Any] = {}
    if scope is not None:
        out["scope"] = scope
    if scopes is not None:
        out["scopes"] = list(scopes)
    if subtree is not None:
        out["subtree"] = subtree
    if not out and default_scope is not None:
        out["scope"] = default_scope
    return out


def write_scopes(
    scope: str | None, scopes: Sequence[str] | None, default_scope: str | None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if scope is not None:
        out["scope"] = scope
    if scopes is not None:
        out["scopes"] = list(scopes)
    if not out and default_scope is not None:
        out["scope"] = default_scope
    return out


def _clean(body: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if v is not None}


def health_call() -> Call:
    return Call("GET", f"{API}/health")


def stats_call() -> Call:
    return Call("GET", f"{API}/stats")


def recall_call(
    query: str,
    *,
    view_fields: Mapping[str, Any],
    budget_tokens: int | None,
    exclude_ids: Sequence[str],
) -> Call:
    body: dict[str, Any] = {"query": query, **view_fields}
    if budget_tokens is not None:
        body["budget_tokens"] = budget_tokens
    if exclude_ids:
        body["exclude_ids"] = list(exclude_ids)
    return Call("POST", f"{API}/recall", json=body)


def expand_call(
    ids: Sequence[str],
    *,
    view_fields: Mapping[str, Any],
    budget_tokens: int | None,
    level: int,
    cursor: str | None,
) -> Call:
    body: dict[str, Any] = {"ids": list(ids), "level": level, **view_fields}
    if budget_tokens is not None:
        body["budget_tokens"] = budget_tokens
    if cursor is not None:
        body["cursor"] = cursor
    return Call("POST", f"{API}/expand", json=body)


def node_call(node_id: str, *, view_fields: Mapping[str, Any], budget_tokens: int | None) -> Call:
    params: dict[str, Any] = dict(view_fields)
    if budget_tokens is not None:
        params["budget_tokens"] = budget_tokens
    return Call("GET", f"{API}/nodes/{node_id}", params=params)


def lookup_call(
    *, external_id: str | None, name: str | None, view_fields: Mapping[str, Any]
) -> Call:
    params: dict[str, Any] = dict(view_fields)
    if external_id is not None:
        params["external_id"] = external_id
    if name is not None:
        params["name"] = name
    return Call("GET", f"{API}/lookup", params=params)


def archived_call(*, view_fields: Mapping[str, Any], budget_tokens: int | None) -> Call:
    params: dict[str, Any] = dict(view_fields)
    if budget_tokens is not None:
        params["budget_tokens"] = budget_tokens
    return Call("GET", f"{API}/archive", params=params)


def node_ref(node: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if node is None:
        return None
    if isinstance(node, str):
        return {"name": node}
    return _clean(node)


def remember_call(
    statement: str,
    *,
    scope_fields: Mapping[str, Any],
    node: str | Mapping[str, Any] | None,
    node_id: str | None,
    attribute: str | None,
    value: str | None,
    valid_from: str | None,
    valid_to: str | None,
    confidence: float | None,
    pinned: bool,
    external_id: str | None,
    source_reference: str | None,
) -> Call:
    body = _clean(
        {
            "statement": statement,
            **scope_fields,
            "node": node_ref(node),
            "node_id": node_id,
            "attribute": attribute,
            "value": value,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "confidence": confidence,
            "pinned": pinned or None,
            "external_id": external_id,
            "source_reference": source_reference,
        }
    )
    return Call("POST", f"{API}/remember", json=body)


def link_call(
    src: str,
    dst: str,
    type: str,
    *,
    scope_fields: Mapping[str, Any],
    weight: float | None,
    external_id: str | None,
) -> Call:
    body = _clean(
        {
            "src": src,
            "dst": dst,
            "type": type,
            **scope_fields,
            "weight": weight,
            "external_id": external_id,
        }
    )
    return Call("POST", f"{API}/link", json=body)


def archive_call(record_id: str) -> Call:
    return Call("POST", f"{API}/archive", json={"id": record_id})


def restore_call(record_id: str) -> Call:
    return Call("POST", f"{API}/restore", json={"id": record_id})


def forget_call(record_id: str) -> Call:
    return Call("POST", f"{API}/forget", json={"id": record_id})


def patch_fact_call(
    fact_id: str, *, node_id: str | None, detach: bool, pinned: bool | None
) -> Call:
    body = _clean({"node_id": node_id, "detach": detach or None, "pinned": pinned})
    return Call("PATCH", f"{API}/facts/{fact_id}", json=body)


# --------------------------------------------------------------------------- lifecycle (SPEC 8.1)


def session_start_call(
    *,
    view_fields: Mapping[str, Any],
    budget_tokens: int | None,
    session_id: str | None,
    exclude_ids: Sequence[str],
) -> Call:
    body: dict[str, Any] = dict(view_fields)
    if budget_tokens is not None:
        body["budget_tokens"] = budget_tokens
    if session_id is not None:
        body["session_id"] = session_id
    if exclude_ids:
        body["exclude_ids"] = list(exclude_ids)
    return Call("POST", f"{API}/session/start", json=body)


def turn_call(
    *,
    scope_fields: Mapping[str, Any],
    user: str,
    assistant: str,
    tool_summaries: Sequence[str],
    session_id: str | None,
    occurred_at: str | None,
    external_id: str | None,
    external: Sequence[Mapping[str, Any]] = (),
) -> Call:
    body = _clean(
        {
            **scope_fields,
            "user": user,
            "assistant": assistant,
            "tool_summaries": list(tool_summaries) or None,
            "external": [dict(part) for part in external] or None,
            "session_id": session_id,
            "occurred_at": occurred_at,
            "external_id": external_id,
        }
    )
    return Call("POST", f"{API}/turns", json=body)


def precompact_call(
    *,
    scope_fields: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    session_id: str | None,
    trigger: str | None,
    budget_tokens: int | None,
    exclude_ids: Sequence[str],
) -> Call:
    body = _clean(
        {
            **scope_fields,
            "messages": [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in messages
            ],
            "session_id": session_id,
            "trigger": trigger,
            "budget_tokens": budget_tokens,
            "exclude_ids": list(exclude_ids) or None,
        }
    )
    return Call("POST", f"{API}/precompact", json=body)


def session_end_call(
    *, scope_fields: Mapping[str, Any], session_id: str | None, reason: str | None
) -> Call:
    body = _clean({**scope_fields, "session_id": session_id, "reason": reason})
    return Call("POST", f"{API}/session/end", json=body)


# --------------------------------------------------------------------------- reading (SPEC 8.2)


def timeline_call(
    *,
    view_fields: Mapping[str, Any],
    budget_tokens: int | None,
    node_id: str | None,
    since: str | None,
    until: str | None,
    limit: int | None,
) -> Call:
    body = _clean(
        {
            **view_fields,
            "budget_tokens": budget_tokens,
            "node_id": node_id,
            "since": since,
            "until": until,
            "limit": limit,
        }
    )
    return Call("POST", f"{API}/timeline", json=body)


def at_call(
    node_id: str, date: str, *, view_fields: Mapping[str, Any], budget_tokens: int | None
) -> Call:
    body = _clean({**view_fields, "budget_tokens": budget_tokens, "node_id": node_id, "date": date})
    return Call("POST", f"{API}/at", json=body)


def path_call(
    src: str,
    dst: str,
    *,
    view_fields: Mapping[str, Any],
    budget_tokens: int | None,
    max_hops: int | None,
) -> Call:
    body = _clean(
        {
            **view_fields,
            "budget_tokens": budget_tokens,
            "src": src,
            "dst": dst,
            "max_hops": max_hops,
        }
    )
    return Call("POST", f"{API}/path", json=body)


# --------------------------------------------------------------------------- operations (SPEC 8.4)


def background_call() -> Call:
    return Call("GET", f"{API}/background")


def set_background_call(
    *, mode: str | None, chat_concurrency: int | None, embeddings_concurrency: int | None
) -> Call:
    body: dict[str, Any] = {}
    if mode is not None:
        body["mode"] = mode
    concurrency = _clean({"chat": chat_concurrency, "embeddings": embeddings_concurrency})
    if concurrency:
        body["concurrency"] = concurrency
    return Call("PUT", f"{API}/background", json=body)


def run_call(
    *,
    calls: int | None,
    tokens: int | None,
    seconds: float | None,
    max_jobs: int | None,
) -> Call:
    budget = _clean({"calls": calls, "tokens": tokens, "seconds": seconds})
    body = _clean({"budget": budget or None, "max_jobs": max_jobs})
    return Call("POST", f"{API}/background/run", json=body)


def yield_call(*, timeout_seconds: float | None) -> Call:
    return Call(
        "POST", f"{API}/background/yield", json=_clean({"timeout_seconds": timeout_seconds})
    )


def resume_call() -> Call:
    return Call("POST", f"{API}/background/resume")


# --------------------------------------------------------------------------- responses


def auth_headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith(API):
        url = url[: -len(API)]
    if not url:
        raise ValueError("base_url is empty")
    return url


def unavailable(exc: httpx.TransportError, call: Call) -> IifaUnavailable:
    """Classified by ``iifa.transport``: ``unreachable`` when no connection was made
    (a timeout while connecting included: Windows retries a closed loopback port
    for about two seconds before refusing), ``timeout`` when the service accepted
    the connection and did not answer in time, ``broken`` when the exchange failed
    after that."""
    reason = classify_transport_error(exc)
    return IifaUnavailable(
        f"{call.method} {call.path}: service {reason} ({type(exc).__name__})", reason=reason
    )


def handle_response(response: httpx.Response, call: Call) -> Any:
    """The JSON of a successful answer, or the matching ``IifaHTTPError``."""
    status = response.status_code
    if status >= 400:
        detail: Any
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text[:300]
        if status in (401, 403):
            raise AuthError(status, detail, call.method, call.path)
        if status == 404:
            raise NotFoundError(status, detail, call.method, call.path)
        if status in (400, 422):
            raise InvalidRequest(status, detail, call.method, call.path)
        if status >= 500:
            raise ServerError(status, detail, call.method, call.path)
        raise IifaHTTPError(status, detail, call.method, call.path)
    try:
        return response.json()
    except ValueError as exc:
        raise IifaProtocolError(
            f"{call.method} {call.path}: the service answered {status} with a body that is not JSON"
        ) from exc
