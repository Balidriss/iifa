"""OpenAI-compatible chat and embeddings clients (SPEC 7.5).

The model is **explicit on every call**: ``chat`` and ``embed`` take ``model``
as a required keyword and put it in the request body, so nothing ever relies on
a server-side default. The only hosts contacted are the two the operator
configured (P4): the client ignores proxy variables and never discovers
anything. Nothing on the recall path holds a reference to this module (P3).

Every call passes through the background gate (P11) before a byte leaves: the
per-endpoint concurrency limit and the harness's ``yield``. A call that is not
allowed to start within ``wait_timeout`` raises ``ModelWaitTimeout`` without
contacting the endpoint.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from iifa.tokens import ConservativeTokenizer
from iifa.transport import classify_transport_error

CHAT = "chat"
EMBEDDINGS = "embeddings"
ENDPOINT_KINDS = (CHAT, EMBEDDINGS)
CHAT_PATH = "/chat/completions"
EMBEDDINGS_PATH = "/embeddings"
DEFAULT_TIMEOUT = 120.0
TRUNCATED = "length"

# What iifa puts in a request body itself: an operator's extra parameters
# (SPEC 7.4b) may add to a request, never replace any of these.
RESERVED_REQUEST_KEYS: frozenset[str] = frozenset(
    {"model", "messages", "input", "stream", "max_tokens", "temperature", "response_format"}
)


class ModelError(RuntimeError):
    """A call failed. ``reason`` is a short word for the log (never content):
    ``unreachable`` / ``timeout`` / ``broken`` from ``iifa.transport`` for a
    transport failure, ``http_<status>``, ``bad_json`` or ``bad_shape`` for an
    answer; ``retryable`` says whether a later attempt could succeed."""

    def __init__(self, endpoint: str, reason: str, *, retryable: bool) -> None:
        super().__init__(f"{endpoint}: {reason}")
        self.endpoint = endpoint
        self.reason = reason
        self.retryable = retryable


class ModelWaitTimeout(ModelError):
    """The gate did not let the call start in time (yielded, or every slot busy)."""

    def __init__(self, endpoint: str) -> None:
        super().__init__(endpoint, "wait_timeout", retryable=True)


class NoEndpoint(ModelError):
    """No endpoint of that kind is configured (tier 0 for chat, tier < 3 for embeddings)."""

    def __init__(self, endpoint: str) -> None:
        super().__init__(endpoint, "no_endpoint", retryable=False)


class Gate(Protocol):
    """What the client needs from the background controller (P11)."""

    def acquire(self, kind: str, *, timeout: float | None) -> bool: ...

    def release(self, kind: str) -> None: ...

    def foreground(self) -> AbstractContextManager[None]: ...


class OpenGate:
    """A gate that always lets a call through (tests and tools without a worker)."""

    def acquire(self, kind: str, *, timeout: float | None) -> bool:
        return True

    def release(self, kind: str) -> None:
        return None

    def foreground(self) -> AbstractContextManager[None]:
        return nullcontext()


@dataclass(frozen=True)
class Endpoint:
    kind: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout: float = DEFAULT_TIMEOUT
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clash = sorted(set(self.extra) & RESERVED_REQUEST_KEYS)
        if clash:
            raise ValueError(
                f"{self.kind} endpoint: extra parameters may not replace {', '.join(clash)}"
            )
        object.__setattr__(self, "extra", dict(self.extra))
        if self.kind not in ENDPOINT_KINDS:
            raise ValueError(f"unknown endpoint kind {self.kind!r}")
        if not self.model:
            raise ValueError(f"{self.kind} endpoint: the model must be explicit")
        url = self.base_url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"{self.kind} endpoint: base_url must be http(s)")
        object.__setattr__(self, "base_url", url)
        if self.timeout <= 0:
            raise ValueError(f"{self.kind} endpoint: timeout must be positive")


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    estimated: bool = False

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    finish_reason: str | None
    usage: Usage
    seconds: float


@dataclass(frozen=True)
class EmbeddingsResult:
    vectors: list[list[float]]
    model: str
    dimension: int
    usage: Usage
    seconds: float


@dataclass
class EndpointStatus:
    calls: int = 0
    failures: int = 0
    last_error: str | None = None


class ModelClient:
    """One HTTP client toward the configured chat and embeddings endpoints."""

    def __init__(
        self,
        chat: Endpoint | None = None,
        embeddings: Endpoint | None = None,
        *,
        gate: Gate | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if chat is not None and chat.kind != CHAT:
            raise ValueError("chat endpoint has the wrong kind")
        if embeddings is not None and embeddings.kind != EMBEDDINGS:
            raise ValueError("embeddings endpoint has the wrong kind")
        self._endpoints: dict[str, Endpoint | None] = {CHAT: chat, EMBEDDINGS: embeddings}
        self._gate: Gate = gate or OpenGate()
        self._http = httpx.Client(trust_env=False, transport=transport)
        self._status = {CHAT: EndpointStatus(), EMBEDDINGS: EndpointStatus()}
        self._lock = threading.Lock()
        self._estimator = ConservativeTokenizer()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def chat_endpoint(self) -> Endpoint | None:
        return self._endpoints[CHAT]

    @property
    def embeddings_endpoint(self) -> Endpoint | None:
        return self._endpoints[EMBEDDINGS]

    def endpoint(self, kind: str) -> Endpoint | None:
        return self._endpoints[kind]

    def status(self) -> dict[str, dict[str, Any] | None]:
        """What ``/v1/health`` shows per endpoint: configured model and URL, call and
        failure counts, the last failure's reason. Never a probe: reporting health
        makes no model call."""
        out: dict[str, dict[str, Any] | None] = {}
        with self._lock:
            for kind, endpoint in self._endpoints.items():
                if endpoint is None:
                    out[kind] = None
                    continue
                status = self._status[kind]
                out[kind] = {
                    "model": endpoint.model,
                    "url": endpoint.base_url,
                    "calls": status.calls,
                    "failures": status.failures,
                    "last_error": status.last_error,
                }
        return out

    # ------------------------------------------------------------------ calls

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        wait_timeout: float | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_object: bool = False,
        accept_truncated: bool = False,
    ) -> ChatResult:
        """One chat completion. ``model`` is required: it goes in the request body
        on every call. ``wait_timeout`` bounds the wait at the gate, not the call.

        An answer cut by ``max_tokens`` (``finish_reason: length``) raises
        ``ModelError("length")`` so that no caller ever parses it (SPEC 7.4b); a
        caller that wants nothing but the call itself says ``accept_truncated``."""
        endpoint = self._require(CHAT, model)
        body: dict[str, Any] = {
            **endpoint.extra,
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_object:
            body["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        data = self._post(endpoint, CHAT_PATH, body, wait_timeout)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise self._fail(CHAT, "bad_shape", retryable=True, counted=True) from exc
        if not isinstance(content, str):
            raise self._fail(CHAT, "bad_shape", retryable=True, counted=True)
        if finish == TRUNCATED and not accept_truncated:
            # the same call truncates the same way: retrying it as is cannot help
            raise self._fail(CHAT, TRUNCATED, retryable=False, counted=True)
        usage = self._usage(
            data,
            prompt_text="".join(str(m.get("content", "")) for m in messages),
            completion_text=content,
        )
        return ChatResult(
            content=content,
            model=str(data.get("model") or model),
            finish_reason=str(finish) if finish is not None else None,
            usage=usage,
            seconds=time.monotonic() - started,
        )

    def embed(
        self,
        inputs: Sequence[str],
        *,
        model: str,
        wait_timeout: float | None = None,
        foreground: bool = False,
        timeout: float | None = None,
    ) -> EmbeddingsResult:
        """Embeddings for ``inputs``, in input order. ``model`` is required.

        ``foreground`` is the query embedding of a recall (SPEC 6.1): it does not
        wait at the background gate; while it is in flight the gate starts no
        background call. ``timeout`` bounds the call itself (at most the
        endpoint's own timeout)."""
        endpoint = self._require(EMBEDDINGS, model)
        if not inputs:
            raise ValueError("nothing to embed")
        body = {**endpoint.extra, "model": model, "input": list(inputs)}
        started = time.monotonic()
        if foreground:
            limit = endpoint.timeout if timeout is None else min(timeout, endpoint.timeout)
            with self._gate.foreground():
                data = self._send(endpoint, EMBEDDINGS_PATH, body, timeout=limit)
        else:
            data = self._post(endpoint, EMBEDDINGS_PATH, body, wait_timeout)
        try:
            rows = sorted(data["data"], key=lambda d: int(d["index"]))
            vectors = [[float(x) for x in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise self._fail(EMBEDDINGS, "bad_shape", retryable=True, counted=True) from exc
        if len(vectors) != len(inputs) or not vectors[0]:
            raise self._fail(EMBEDDINGS, "bad_shape", retryable=True, counted=True)
        dimension = len(vectors[0])
        if any(len(v) != dimension for v in vectors):
            raise self._fail(EMBEDDINGS, "bad_shape", retryable=True, counted=True)
        usage = self._usage(data, prompt_text="".join(inputs), completion_text="")
        return EmbeddingsResult(
            vectors=vectors,
            model=str(data.get("model") or model),
            dimension=dimension,
            usage=usage,
            seconds=time.monotonic() - started,
        )

    # ------------------------------------------------------------------ plumbing

    def _require(self, kind: str, model: str) -> Endpoint:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"{kind}: the model must be explicit on every call")
        endpoint = self._endpoints[kind]
        if endpoint is None:
            raise NoEndpoint(kind)
        return endpoint

    def _post(
        self, endpoint: Endpoint, path: str, body: dict[str, Any], wait_timeout: float | None
    ) -> Any:
        if not self._gate.acquire(endpoint.kind, timeout=wait_timeout):
            raise ModelWaitTimeout(endpoint.kind)
        try:
            return self._send(endpoint, path, body)
        finally:
            self._gate.release(endpoint.kind)

    def _send(
        self, endpoint: Endpoint, path: str, body: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        headers = {"Accept": "application/json"}
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        try:
            response = self._http.post(
                endpoint.base_url + path,
                json=body,
                headers=headers,
                timeout=endpoint.timeout if timeout is None else timeout,
            )
        except httpx.TransportError as exc:  # unreachable, timeout or broken (iifa.transport)
            raise self._fail(endpoint.kind, classify_transport_error(exc), retryable=True) from exc
        status = response.status_code
        if status >= 400:
            retryable = status >= 500 or status in (408, 409, 425, 429)
            raise self._fail(endpoint.kind, f"http_{status}", retryable=retryable)
        try:
            data = response.json()
        except ValueError as exc:
            raise self._fail(endpoint.kind, "bad_json", retryable=True) from exc
        if not isinstance(data, dict):
            raise self._fail(endpoint.kind, "bad_shape", retryable=True)
        with self._lock:
            self._status[endpoint.kind].calls += 1
        return data

    def _fail(
        self, kind: str, reason: str, *, retryable: bool, counted: bool = False
    ) -> ModelError:
        """Record a failure; ``counted`` when ``_send`` already counted the call."""
        with self._lock:
            status = self._status[kind]
            if not counted:
                status.calls += 1
            status.failures += 1
            status.last_error = reason
        return ModelError(kind, reason, retryable=retryable)

    def _usage(self, data: Mapping[str, Any], *, prompt_text: str, completion_text: str) -> Usage:
        raw = data.get("usage")
        if isinstance(raw, dict):
            try:
                prompt = int(raw.get("prompt_tokens", 0))
                completion = int(raw.get("completion_tokens", 0))
                if prompt >= 0 and completion >= 0 and (prompt or completion):
                    return Usage(prompt, completion)
            except (TypeError, ValueError):
                pass
        return Usage(
            self._estimator.count(prompt_text), self._estimator.count(completion_text), True
        )


class QueryEmbedder:
    """What the recall engine holds at tier 3 (P3, SPEC 6.1): it embeds one query,
    in the foreground, with a bounded timeout, and can do nothing else - no chat,
    no background call."""

    def __init__(self, client: ModelClient) -> None:
        if client.embeddings_endpoint is None:
            raise ValueError("a query embedder needs an embeddings endpoint")
        self.__client = client

    @property
    def model(self) -> str:
        endpoint = self.__client.embeddings_endpoint
        assert endpoint is not None
        return endpoint.model

    def embed_query(self, text: str, *, timeout: float) -> list[float]:
        """The query's vector; raises ``ModelError`` (the caller degrades, P10)."""
        result = self.__client.embed([text], model=self.model, foreground=True, timeout=timeout)
        return result.vectors[0]


__all__ = [
    "CHAT",
    "CHAT_PATH",
    "EMBEDDINGS",
    "EMBEDDINGS_PATH",
    "ENDPOINT_KINDS",
    "RESERVED_REQUEST_KEYS",
    "TRUNCATED",
    "ChatResult",
    "EmbeddingsResult",
    "Endpoint",
    "Gate",
    "ModelClient",
    "ModelError",
    "ModelWaitTimeout",
    "NoEndpoint",
    "OpenGate",
    "QueryEmbedder",
    "Usage",
]
