"""The asynchronous client (same surface as ``IifaClient``, awaitable)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from iifa.client import _base as b
from iifa.client._base import Call


class AsyncIifaClient:
    """An asyncio client for one iifa service.

    ``scope`` is the default read view and write target used when a call gives
    none. ``transport`` lets tests talk to an in-process app.
    """

    def __init__(
        self,
        base_url: str = b.DEFAULT_URL,
        token: str | None = None,
        *,
        scope: str | None = None,
        timeout: float | httpx.Timeout = b.DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = b.normalize_base_url(base_url)
        self.scope = scope
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=b.auth_headers(token),
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> AsyncIifaClient:
        """``IIFA_URL`` (default ``http://127.0.0.1:8765``), ``IIFA_TOKEN``, ``IIFA_SCOPE``."""
        env = os.environ if env is None else env
        values: dict[str, Any] = {
            "base_url": env.get("IIFA_URL") or b.DEFAULT_URL,
            "token": env.get("IIFA_TOKEN") or None,
            "scope": env.get("IIFA_SCOPE") or None,
        }
        values.update(overrides)
        return cls(**values)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncIifaClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ plumbing

    async def _send(self, call: Call) -> Any:
        try:
            response = await self._http.request(
                call.method, call.path, params=call.params, json=call.json
            )
        except httpx.TransportError as exc:
            raise b.unavailable(exc, call) from exc
        return b.handle_response(response, call)

    def _view(
        self, scope: str | None, scopes: Sequence[str] | None, subtree: str | None
    ) -> dict[str, Any]:
        return b.view(scope, scopes, subtree, self.scope)

    def _write(self, scope: str | None, scopes: Sequence[str] | None) -> dict[str, Any]:
        return b.write_scopes(scope, scopes, self.scope)

    # ------------------------------------------------------------------ operations

    async def health(self) -> dict[str, Any]:
        result: dict[str, Any] = await self._send(b.health_call())
        return result

    async def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = await self._send(b.stats_call())
        return result

    async def recall(
        self,
        query: str,
        *,
        budget_tokens: int | None = None,
        exclude_ids: Sequence[str] = (),
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        """Handles under budget (P1, P2): id, title, one line, why, age, expand_tokens."""
        result: dict[str, Any] = await self._send(
            b.recall_call(
                query,
                view_fields=self._view(scope, scopes, subtree),
                budget_tokens=budget_tokens,
                exclude_ids=exclude_ids,
            )
        )
        return result

    async def expand(
        self,
        ids: Sequence[str],
        *,
        level: int = 1,
        budget_tokens: int | None = None,
        cursor: str | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        """The content behind handles: level 1 = facts, level 2 = facts + source excerpt."""
        result: dict[str, Any] = await self._send(
            b.expand_call(
                ids,
                view_fields=self._view(scope, scopes, subtree),
                budget_tokens=budget_tokens,
                level=level,
                cursor=cursor,
            )
        )
        return result

    async def node(
        self,
        node_id: str,
        *,
        budget_tokens: int | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = await self._send(
            b.node_call(
                node_id, view_fields=self._view(scope, scopes, subtree), budget_tokens=budget_tokens
            )
        )
        return result

    async def lookup(
        self,
        *,
        external_id: str | None = None,
        name: str | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        """Records carrying an ``external_id``, or nodes with a ``name`` or alias."""
        result: dict[str, Any] = await self._send(
            b.lookup_call(
                external_id=external_id, name=name, view_fields=self._view(scope, scopes, subtree)
            )
        )
        return result

    async def archived(
        self,
        *,
        budget_tokens: int | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = await self._send(
            b.archived_call(
                view_fields=self._view(scope, scopes, subtree), budget_tokens=budget_tokens
            )
        )
        return result

    async def timeline(
        self,
        *,
        node_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
        budget_tokens: int | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        """Fact changes of a node (or of the view) inside a date range, newest first."""
        result: dict[str, Any] = await self._send(
            b.timeline_call(
                view_fields=self._view(scope, scopes, subtree),
                budget_tokens=budget_tokens,
                node_id=node_id,
                since=since,
                until=until,
                limit=limit,
            )
        )
        return result

    async def at(
        self,
        node_id: str,
        date: str,
        *,
        budget_tokens: int | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        """What was known about a node at ``date`` (an ISO date or timestamp)."""
        result: dict[str, Any] = await self._send(
            b.at_call(
                node_id,
                date,
                view_fields=self._view(scope, scopes, subtree),
                budget_tokens=budget_tokens,
            )
        )
        return result

    async def path(
        self,
        src: str,
        dst: str,
        *,
        max_hops: int | None = None,
        budget_tokens: int | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        """How two nodes are related: the shortest visible path, at most ``max_hops`` long."""
        result: dict[str, Any] = await self._send(
            b.path_call(
                src,
                dst,
                view_fields=self._view(scope, scopes, subtree),
                budget_tokens=budget_tokens,
                max_hops=max_hops,
            )
        )
        return result

    async def remember(
        self,
        statement: str,
        *,
        node: str | Mapping[str, Any] | None = None,
        node_id: str | None = None,
        attribute: str | None = None,
        value: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float | None = None,
        pinned: bool = False,
        external_id: str | None = None,
        source_reference: str | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """An explicit fact (P6), optionally about a node given by name (created on
        first mention) or by id."""
        result: dict[str, Any] = await self._send(
            b.remember_call(
                statement,
                scope_fields=self._write(scope, scopes),
                node=node,
                node_id=node_id,
                attribute=attribute,
                value=value,
                valid_from=valid_from,
                valid_to=valid_to,
                confidence=confidence,
                pinned=pinned,
                external_id=external_id,
                source_reference=source_reference,
            )
        )
        return result

    async def link(
        self,
        src: str,
        dst: str,
        type: str,
        *,
        weight: float | None = None,
        external_id: str | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = await self._send(
            b.link_call(
                src,
                dst,
                type,
                scope_fields=self._write(scope, scopes),
                weight=weight,
                external_id=external_id,
            )
        )
        return result

    async def archive(self, record_id: str) -> dict[str, Any]:
        """Reversible removal (P5)."""
        result: dict[str, Any] = await self._send(b.archive_call(record_id))
        return result

    async def restore(self, record_id: str) -> dict[str, Any]:
        result: dict[str, Any] = await self._send(b.restore_call(record_id))
        return result

    async def forget(self, record_id: str) -> dict[str, Any]:
        """Irreversible hard delete; needs a token with the ``forget`` permission (P5)."""
        result: dict[str, Any] = await self._send(b.forget_call(record_id))
        return result

    async def patch_fact(
        self,
        fact_id: str,
        *,
        node_id: str | None = None,
        detach: bool = False,
        pinned: bool | None = None,
        scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Attach, detach or pin a fact, or replace its scope set (``scopes``): the
        token must own every scope added and every scope removed (SPEC 4.6)."""
        result: dict[str, Any] = await self._send(
            b.patch_fact_call(fact_id, node_id=node_id, detach=detach, pinned=pinned, scopes=scopes)
        )
        return result

    # ------------------------------------------------------------------ lifecycle (SPEC 8.1)

    async def session_start(
        self,
        *,
        session_id: str | None = None,
        budget_tokens: int | None = None,
        exclude_ids: Sequence[str] = (),
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
        subtree: str | None = None,
    ) -> dict[str, Any]:
        """The profile block: pinned facts and central nodes, under budget, stable
        while nothing changes (SPEC 8.1)."""
        result: dict[str, Any] = await self._send(
            b.session_start_call(
                view_fields=self._view(scope, scopes, subtree),
                budget_tokens=budget_tokens,
                session_id=session_id,
                exclude_ids=exclude_ids,
            )
        )
        return result

    async def turn(
        self,
        user: str = "",
        assistant: str = "",
        *,
        tool_summaries: Sequence[str] = (),
        external: Sequence[Mapping[str, Any]] = (),
        session_id: str | None = None,
        occurred_at: str | None = None,
        external_id: str | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Record a turn: enqueued only, answered at once (P3). ``user`` is the
        owner's words; ``tool_summaries`` and ``external`` parts (``kind``:
        tool, web, document, other; ``text``; ``ref``) are content from outside
        the conversation, never facts about the owner (P6)."""
        result: dict[str, Any] = await self._send(
            b.turn_call(
                scope_fields=self._write(scope, scopes),
                user=user,
                assistant=assistant,
                tool_summaries=tool_summaries,
                external=external,
                session_id=session_id,
                occurred_at=occurred_at,
                external_id=external_id,
            )
        )
        return result

    async def precompact(
        self,
        messages: Sequence[Mapping[str, str]] = (),
        *,
        session_id: str | None = None,
        trigger: str | None = None,
        budget_tokens: int | None = None,
        exclude_ids: Sequence[str] = (),
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """The messages about to be dropped go to the queue; what must survive comes
        back under budget."""
        result: dict[str, Any] = await self._send(
            b.precompact_call(
                scope_fields=self._write(scope, scopes),
                messages=messages,
                session_id=session_id,
                trigger=trigger,
                budget_tokens=budget_tokens,
                exclude_ids=exclude_ids,
            )
        )
        return result

    async def session_end(
        self,
        *,
        session_id: str | None = None,
        reason: str | None = None,
        scope: str | None = None,
        scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Flush and extract the session, in the background."""
        result: dict[str, Any] = await self._send(
            b.session_end_call(
                scope_fields=self._write(scope, scopes), session_id=session_id, reason=reason
            )
        )
        return result

    # ------------------------------------------------------------------ operations (SPEC 8.4)

    async def background(self) -> dict[str, Any]:
        """Mode, concurrency, yield state and queue depth (admin)."""
        result: dict[str, Any] = await self._send(b.background_call())
        return result

    async def set_background(
        self,
        *,
        mode: str | None = None,
        chat_concurrency: int | None = None,
        embeddings_concurrency: int | None = None,
    ) -> dict[str, Any]:
        """Change the background mode (``auto`` | ``paused`` | ``on_demand``, persisted)
        and the per-endpoint concurrency (admin, P11)."""
        result: dict[str, Any] = await self._send(
            b.set_background_call(
                mode=mode,
                chat_concurrency=chat_concurrency,
                embeddings_concurrency=embeddings_concurrency,
            )
        )
        return result

    async def run_background(
        self,
        *,
        calls: int | None = None,
        tokens: int | None = None,
        seconds: float | None = None,
        max_jobs: int | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Process the queue now under a budget; returns the run report (admin).
        With ``wait=False`` it returns the run id at once: follow it with
        ``background_run``."""
        result: dict[str, Any] = await self._send(
            b.run_call(calls=calls, tokens=tokens, seconds=seconds, max_jobs=max_jobs, wait=wait)
        )
        return result

    async def background_run(self, run_id: str) -> dict[str, Any]:
        """A run's report, live while it runs (admin)."""
        result: dict[str, Any] = await self._send(b.background_run_call(run_id))
        return result

    async def yield_background(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        """A foreground turn starts: no new background model call until ``resume``."""
        result: dict[str, Any] = await self._send(b.yield_call(timeout_seconds=timeout_seconds))
        return result

    async def resume_background(self) -> dict[str, Any]:
        result: dict[str, Any] = await self._send(b.resume_call())
        return result
