"""A scripted fake of an OpenAI-compatible model server (chat + embeddings).

It listens on the loopback interface only, records every call it receives and
answers according to a script. Anything the real thing can do wrong, this fake
can be told to do on purpose: answer slowly, hang past a timeout, fail with a
status code, return a malformed body, or return whatever text the test wants
(an answer in the wrong language, hostile content, operations outside a
catalogue).

The call log is the point: "paused means zero model calls" (P11) becomes an
assertion on ``server.calls``, not a hope.

Usage::

    with FakeOpenAIServer() as server:
        server.script(chat=[Reply("Bonjour"), Fail(500)])
        ...
        assert len(server.calls) == 2
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import uvicorn

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

CHAT_PATH = "/v1/chat/completions"
EMBEDDINGS_PATH = "/v1/embeddings"
MODELS_PATH = "/v1/models"


# --------------------------------------------------------------------------- behaviours


@dataclass(frozen=True)
class Reply:
    """Answer normally. ``content`` is the assistant text for chat calls.

    For embeddings, ``vectors`` overrides the deterministic default vectors.
    """

    content: str = "OK"
    vectors: Sequence[Sequence[float]] | None = None
    finish_reason: str = "stop"
    delay: float = 0.0


@dataclass(frozen=True)
class Fail:
    """Answer with an HTTP error status and a JSON error body."""

    status: int = 500
    message: str = "internal error (scripted)"
    delay: float = 0.0


@dataclass(frozen=True)
class Malformed:
    """Answer 200 with a body that is not valid JSON."""

    body: bytes = b'{"choices": [{"message": {"content": "truncated'
    delay: float = 0.0


@dataclass(frozen=True)
class Hang:
    """Do not answer for ``seconds`` (long enough for the client to time out)."""

    seconds: float = 5.0


@dataclass(frozen=True)
class Respond:
    """Decide the behaviour from the request body: a scripted model that reads its
    input (the benchmark's oracle behind a real HTTP endpoint)."""

    fn: Callable[[Any], Reply | Fail | Malformed | Hang]


Behaviour = Reply | Fail | Malformed | Hang | Respond


def chat_json(payload: Any, **kwargs: Any) -> Reply:
    """A chat reply whose content is ``payload`` serialised as JSON."""
    return Reply(content=json.dumps(payload, ensure_ascii=False), **kwargs)


# --------------------------------------------------------------------------- call log


@dataclass
class RecordedCall:
    """One request seen by the fake, as the service sent it."""

    method: str
    path: str
    body: Any
    headers: dict[str, str]
    started_at: float
    behaviour: Behaviour
    status: int | None = None

    @property
    def model(self) -> str | None:
        if isinstance(self.body, dict):
            model = self.body.get("model")
            return str(model) if model is not None else None
        return None

    @property
    def messages(self) -> list[dict[str, Any]]:
        if isinstance(self.body, dict):
            messages = self.body.get("messages")
            if isinstance(messages, list):
                return messages
        return []


# --------------------------------------------------------------------------- server


def deterministic_vector(text: str, dimension: int) -> list[float]:
    """A unit vector derived from the text alone, stable across runs and platforms."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    seed = digest
    while len(values) < dimension:
        seed = hashlib.sha256(seed).digest()
        values.extend(struct.unpack("<8f", seed) if dimension - len(values) >= 8 else ())
        if dimension - len(values) < 8:
            values.extend(struct.unpack("<f", seed[:4]))
    values = [v if math.isfinite(v) else 0.0 for v in values[:dimension]]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class FakeOpenAIServer:
    """Scripted OpenAI-compatible server on 127.0.0.1 with a call log."""

    def __init__(self, *, embedding_dimension: int = 8, model_names: Iterable[str] = ()) -> None:
        self.embedding_dimension = embedding_dimension
        self.model_names = list(model_names) or ["fake-chat", "fake-embed"]
        self.calls: list[RecordedCall] = []
        self._scripts: dict[str, deque[Behaviour]] = {CHAT_PATH: deque(), EMBEDDINGS_PATH: deque()}
        self._defaults: dict[str, Behaviour] = {CHAT_PATH: Reply(), EMBEDDINGS_PATH: Reply()}
        self._lock = threading.Lock()
        self._vectors: dict[str, list[float]] = {}
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.host = "127.0.0.1"
        self.port = 0

    # ---- lifecycle

    def start(self) -> FakeOpenAIServer:
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

        self._thread = threading.Thread(target=run, name="fake-openai-server", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:  # pragma: no cover - only on a broken machine
                raise RuntimeError("fake OpenAI server did not start")
            time.sleep(0.005)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    def __enter__(self) -> FakeOpenAIServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    # ---- scripting

    def script(
        self,
        *,
        chat: Iterable[Behaviour] = (),
        embeddings: Iterable[Behaviour] = (),
    ) -> None:
        """Queue behaviours, consumed one per call. After the queue, the default applies."""
        with self._lock:
            self._scripts[CHAT_PATH].extend(chat)
            self._scripts[EMBEDDINGS_PATH].extend(embeddings)

    def set_default(
        self, *, chat: Behaviour | None = None, embeddings: Behaviour | None = None
    ) -> None:
        with self._lock:
            if chat is not None:
                self._defaults[CHAT_PATH] = chat
            if embeddings is not None:
                self._defaults[EMBEDDINGS_PATH] = embeddings

    def set_vectors(self, vectors: Mapping[str, Sequence[float]]) -> None:
        """Fix the embedding of given input texts (unit-normalised by the service
        anyway), so a test decides which texts are close; every other text keeps
        its deterministic vector."""
        with self._lock:
            self._vectors.update({text: list(v) for text, v in vectors.items()})

    def reset(self) -> None:
        """Forget the call log, every queued behaviour and every fixed vector."""
        with self._lock:
            self._vectors.clear()
            self.calls.clear()
            for queue in self._scripts.values():
                queue.clear()
            self._defaults = {CHAT_PATH: Reply(), EMBEDDINGS_PATH: Reply()}

    def calls_to(self, path: str) -> list[RecordedCall]:
        return [c for c in self.calls if c.path == path]

    @property
    def chat_calls(self) -> list[RecordedCall]:
        return self.calls_to(CHAT_PATH)

    @property
    def embedding_calls(self) -> list[RecordedCall]:
        return self.calls_to(EMBEDDINGS_PATH)

    # ---- ASGI

    async def _asgi(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # pragma: no cover - uvicorn only sends http here
            return
        path = scope["path"]
        method = scope["method"]
        body = await _read_body(receive)
        headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
        if path == MODELS_PATH and method == "GET":
            payload = {
                "object": "list",
                "data": [{"id": m, "object": "model"} for m in self.model_names],
            }
            await _send_json(send, 200, payload)
            return
        if path not in self._scripts or method != "POST":
            await _send_json(send, 404, {"error": {"message": f"unknown route {method} {path}"}})
            return
        parsed = _parse_json(body)
        with self._lock:
            queue = self._scripts[path]
            behaviour = queue.popleft() if queue else self._defaults[path]
            call = RecordedCall(method, path, parsed, headers, time.time(), behaviour)
            self.calls.append(call)
        status, response_body = await self._answer(path, parsed, behaviour)
        call.status = status
        await _send_raw(send, status, response_body)

    async def _answer(self, path: str, request: Any, behaviour: Behaviour) -> tuple[int, bytes]:
        if isinstance(behaviour, Respond):
            behaviour = behaviour.fn(request)
        if isinstance(behaviour, Hang):
            await asyncio.sleep(behaviour.seconds)
            return 200, _encode({"error": {"message": "answered after hanging"}})
        if behaviour.delay:
            await asyncio.sleep(behaviour.delay)
        if isinstance(behaviour, Fail):
            return behaviour.status, _encode(
                {"error": {"message": behaviour.message, "type": "scripted_error"}}
            )
        if isinstance(behaviour, Malformed):
            return 200, behaviour.body
        if not isinstance(request, dict):
            return 400, _encode({"error": {"message": "request body must be a JSON object"}})
        if request.get("stream"):
            return 400, _encode({"error": {"message": "the fake does not stream"}})
        model = request.get("model")
        if not isinstance(model, str) or not model:
            return 400, _encode({"error": {"message": "model is required on every call"}})
        if path == CHAT_PATH:
            return 200, _encode(self._chat_response(request, behaviour, model))
        return 200, _encode(self._embeddings_response(request, behaviour, model))

    def _chat_response(
        self, request: dict[str, Any], behaviour: Reply, model: str
    ) -> dict[str, Any]:
        prompt_chars = sum(len(str(m.get("content", ""))) for m in request.get("messages", []))
        return {
            "id": f"chatcmpl-fake-{len(self.calls)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": behaviour.content},
                    "finish_reason": behaviour.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": math.ceil(prompt_chars / 4),
                "completion_tokens": math.ceil(len(behaviour.content) / 4),
                "total_tokens": math.ceil(prompt_chars / 4) + math.ceil(len(behaviour.content) / 4),
            },
        }

    def _embeddings_response(
        self, request: dict[str, Any], behaviour: Reply, model: str
    ) -> dict[str, Any]:
        raw_input = request.get("input", [])
        inputs: list[str] = (
            [raw_input] if isinstance(raw_input, str) else [str(x) for x in raw_input]
        )
        if behaviour.vectors is not None:
            vectors = [list(v) for v in behaviour.vectors]
        else:
            vectors = [
                self._vectors.get(text) or deterministic_vector(text, self.embedding_dimension)
                for text in inputs
            ]
        return {
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": vec}
                for i, vec in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": sum(math.ceil(len(t) / 4) for t in inputs),
                "total_tokens": 0,
            },
        }


# --------------------------------------------------------------------------- ASGI helpers


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


async def _send_json(send: Send, status: int, payload: Any) -> None:
    await _send_raw(send, status, _encode(payload))


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


__all__ = [
    "CHAT_PATH",
    "EMBEDDINGS_PATH",
    "MODELS_PATH",
    "Behaviour",
    "Fail",
    "FakeOpenAIServer",
    "Hang",
    "Malformed",
    "RecordedCall",
    "Reply",
    "chat_json",
    "deterministic_vector",
]
_ = field  # keep dataclasses.field importable for extensions of the fake
