"""A scripted fake of the iifa HTTP API, for adapter tests (SPEC 13.1).

Like the fake model server it listens on the loopback interface only, records
every call, and answers according to a script: a canned body, a status code, a
body that is not JSON, or nothing for a while (long enough for the adapter's
deadline to pass). The adapter under test must fail open on every one of those.

    with FakeIifaService() as service:
        service.script(recall=[Hang(3.0)])
        ...
        assert service.calls[-1].path == "/v1/recall"

Tests in another language (the opencode plugin's, under ``node --test``) drive
the same fake over HTTP: ``python -m tests.fakes.iifa_service`` prints
``{"url": ...}`` on one line and serves until its stdin closes; ``POST
/_fake/script`` queues behaviours (``{"path", "behaviours": [{"type": "reply",
"body", "status", "delay"} | {"type": "fail", "status", "detail"} | {"type":
"malformed"} | {"type": "hang", "seconds"}]}``), ``GET /_fake/calls`` lists the
recorded calls, ``POST /_fake/reset`` clears both. Control calls are not
recorded.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import uvicorn

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

HEALTH = "/v1/health"
RECALL = "/v1/recall"
REMEMBER = "/v1/remember"
SESSION_START = "/v1/session/start"
TURNS = "/v1/turns"
PRECOMPACT = "/v1/precompact"
SESSION_END = "/v1/session/end"
EXPAND = "/v1/expand"
ARCHIVE = "/v1/archive"
CONTROL = "/_fake/"

HANDLE = {
    "id": "fact_01J8Z3M4Q5R6S7T8V9W0X1Y2Z3",
    "kind": "fact",
    "title": "Karim — leads the solar installation planning",
    "line": "Since 2026-03, leads installation planning.",
    "why": "query term 'solar'",
    "age_days": 12,
    "expand_tokens": 140,
    "flags": [],
}


def recall_body(items: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    items = [HANDLE] if items is None else items
    body: dict[str, Any] = {
        "items": items,
        "used_tokens": 60 * len(items),
        "budget_tokens": 400,
        "omitted": 0,
        "reason": None if items else "no_match",
        "tier": 0,
        "degraded": None,
    }
    body.update(extra)
    return body


PROFILE_FACT = {
    "id": "fact_01J8Z3M4Q5R6S7T8V9W0X1Y2Z4",
    "kind": "fact",
    "node_id": None,
    "node_name": None,
    "statement": "Always answer in French",
    "attribute": None,
    "value": None,
    "valid_from": None,
    "flags": ["pinned"],
}

PROFILE_NODE = {
    "id": "node_01J8Z3M4Q5R6S7T8V9W0X1Y2Z5",
    "kind": "node",
    "name": "Karim",
    "node_kind": "person",
    "summary": "",
    "aliases": [],
    "facts": 3,
    "edges": 1,
}


def profile_body(items: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    items = [PROFILE_FACT, PROFILE_NODE] if items is None else items
    body: dict[str, Any] = {
        "items": items,
        "used_tokens": 70 * len(items),
        "budget_tokens": 400,
        "omitted": 0,
        "reason": None if items else "empty_profile",
        "tier": 0,
        "degraded": None,
        "session_id": None,
    }
    body.update(extra)
    return body


QUEUED = {"job_id": "job_01J8Z3M4Q5R6S7T8V9W0X1Y2Z6", "status": "pending"}
QUEUE = {"depth": 1, "mode": "auto"}

HEALTH_BODY = {
    "status": "ok",
    "version": "0.1.0a1",
    "tier": 0,
    "backends": {"chat": None, "embeddings": None},
    "queue": {
        "depth": 0,
        "pending": 0,
        "running": 0,
        "failed": 0,
        "done": 0,
        "oldest_pending": None,
    },
    "background": {
        "mode": "auto",
        "yielded": False,
        "concurrency": {"chat": 1, "embeddings": 1},
        "worker": "running",
    },
    "stale_vectors": 0,
    "tokenizer": "conservative",
}


@dataclass(frozen=True)
class Reply:
    body: Any = None
    status: int = 200
    delay: float = 0.0


@dataclass(frozen=True)
class Fail:
    status: int = 500
    detail: str = "internal error (scripted)"
    delay: float = 0.0


@dataclass(frozen=True)
class Malformed:
    body: bytes = b'{"items": [{"id": "fact_'
    delay: float = 0.0


@dataclass(frozen=True)
class Hang:
    seconds: float = 5.0


Behaviour = Reply | Fail | Malformed | Hang


@dataclass
class RecordedCall:
    method: str
    path: str
    query: str
    body: Any
    headers: dict[str, str]
    started_at: float
    behaviour: Behaviour
    status: int | None = None
    _: None = field(default=None, repr=False)


class FakeIifaService:
    def __init__(self) -> None:
        self.calls: list[RecordedCall] = []
        self._scripts: dict[str, deque[Behaviour]] = {}
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.host = "127.0.0.1"
        self.port = 0

    # ---- lifecycle

    def start(self) -> FakeIifaService:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, 0))
        sock.listen(64)
        self.port = sock.getsockname()[1]
        config = uvicorn.Config(
            self._asgi,
            host=self.host,
            port=self.port,
            log_level="error",
            lifespan="off",
            interface="asgi3",
        )
        self._server = uvicorn.Server(config)
        server = self._server

        def run() -> None:
            asyncio.run(server.serve(sockets=[sock]))

        self._thread = threading.Thread(target=run, name="fake-iifa-service", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:  # pragma: no cover
                raise RuntimeError("fake iifa service did not start")
            time.sleep(0.005)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    def __enter__(self) -> FakeIifaService:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ---- scripting

    def script(self, path: str, behaviours: Iterable[Behaviour]) -> None:
        """Queue behaviours for ``path`` (``/v1/recall``...), consumed one per call."""
        with self._lock:
            self._scripts.setdefault(path, deque()).extend(behaviours)

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self._scripts.clear()

    def calls_to(self, path: str) -> list[RecordedCall]:
        return [c for c in self.calls if c.path == path]

    # ---- ASGI

    async def _asgi(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # pragma: no cover
            return
        path, method = scope["path"], scope["method"]
        body = await _read_body(receive)
        if path.startswith(CONTROL):
            status, payload = self._control(path, method, body)
            await _send_raw(send, status, payload)
            return
        headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
        with self._lock:
            queue = self._scripts.get(path)
            behaviour = queue.popleft() if queue else self._default(path, method)
            call = RecordedCall(
                method,
                path,
                scope.get("query_string", b"").decode(),
                _parse_json(body),
                headers,
                time.time(),
                behaviour,
            )
            self.calls.append(call)
        status, payload = await self._answer(behaviour)
        call.status = status
        await _send_raw(send, status, payload)

    def _control(self, path: str, method: str, body: bytes) -> tuple[int, bytes]:
        """The control plane for tests that cannot call ``script`` in-process."""
        if path == CONTROL + "reset" and method == "POST":
            self.reset()
            return 200, _encode({"ok": True})
        if path == CONTROL + "calls" and method == "GET":
            with self._lock:
                calls = [
                    {
                        "method": c.method,
                        "path": c.path,
                        "body": c.body,
                        "authorization": c.headers.get("authorization"),
                        "status": c.status,
                    }
                    for c in self.calls
                ]
            return 200, _encode(calls)
        if path == CONTROL + "script" and method == "POST":
            try:
                request = json.loads(body)
                behaviours = [_behaviour(b) for b in request["behaviours"]]
                self.script(str(request["path"]), behaviours)
            except (ValueError, KeyError, TypeError) as exc:
                return 400, _encode({"detail": f"bad script: {exc}"})
            return 200, _encode({"ok": True, "queued": len(behaviours)})
        return 404, _encode({"detail": f"unknown control route {method} {path}"})

    @staticmethod
    def _default(path: str, method: str) -> Behaviour:
        if path == HEALTH and method == "GET":
            return Reply(HEALTH_BODY)
        if path == RECALL and method == "POST":
            return Reply(recall_body())
        if path == REMEMBER and method == "POST":
            return Reply(
                {
                    "fact": {"id": HANDLE["id"], "kind": "fact", "scopes": ["user:alice"]},
                    "node": None,
                    "source_id": "src_01J8Z3M4Q5R6S7T8V9W0X1Y2Z3",
                    "created_node": False,
                },
                status=201,
            )
        if path == SESSION_START and method == "POST":
            return Reply(profile_body())
        if path == TURNS and method == "POST":
            return Reply({**QUEUED, "kind": "turn", "queue": QUEUE}, status=202)
        if path == PRECOMPACT and method == "POST":
            return Reply({**profile_body(), **QUEUED, "kind": "precompact", "queue": QUEUE})
        if path == SESSION_END and method == "POST":
            return Reply({**QUEUED, "kind": "session_end", "queue": QUEUE}, status=202)
        if path == EXPAND and method == "POST":
            return Reply(
                {
                    "items": [{**PROFILE_FACT, "statement": "Karim leads the solar planning"}],
                    "used_tokens": 40,
                    "budget_tokens": 400,
                    "omitted": 0,
                    "cursor": None,
                }
            )
        if path == ARCHIVE and method == "POST":
            return Reply({"id": HANDLE["id"], "kind": "fact", "archived": True})
        return Fail(404, f"unknown route {method} {path}")

    @staticmethod
    async def _answer(behaviour: Behaviour) -> tuple[int, bytes]:
        if isinstance(behaviour, Hang):
            await asyncio.sleep(behaviour.seconds)
            return 200, _encode({"detail": "answered after hanging"})
        if behaviour.delay:
            await asyncio.sleep(behaviour.delay)
        if isinstance(behaviour, Fail):
            return behaviour.status, _encode({"detail": behaviour.detail})
        if isinstance(behaviour, Malformed):
            return 200, behaviour.body
        return behaviour.status, _encode(behaviour.body if behaviour.body is not None else {})


def _behaviour(spec: dict[str, Any]) -> Behaviour:
    kind = spec["type"]
    if kind == "reply":
        return Reply(spec.get("body"), int(spec.get("status", 200)), float(spec.get("delay", 0)))
    if kind == "fail":
        return Fail(int(spec.get("status", 500)), str(spec.get("detail", "scripted")))
    if kind == "malformed":
        return Malformed()
    if kind == "hang":
        return Hang(float(spec.get("seconds", 5.0)))
    raise ValueError(f"unknown behaviour {kind!r}")


def main() -> None:
    """Serve until stdin closes; the first line on stdout says where."""
    with FakeIifaService() as service:
        print(json.dumps({"url": service.url}), flush=True)
        sys.stdin.read()


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            return b"".join(chunks)


def _parse_json(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return {"_unparseable": body.decode("utf-8", "replace")}


def _encode(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


async def _send_raw(send: Send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    main()
