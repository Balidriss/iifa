"""The model client (SPEC 7.5): OpenAI-compatible chat and embeddings, the model
explicit in every request body, failures mapped to short retryable-or-not
reasons, and every call counted by the fake server."""

from __future__ import annotations

import threading
import time

import pytest

from iifa.llm import (
    CHAT,
    EMBEDDINGS,
    Endpoint,
    ModelClient,
    ModelError,
    ModelWaitTimeout,
    NoEndpoint,
)
from tests.fakes.openai_server import (
    Fail,
    FakeOpenAIServer,
    Hang,
    Malformed,
    Reply,
    chat_json,
)


def client_for(fake: FakeOpenAIServer, *, timeout: float = 5.0, **kwargs: object) -> ModelClient:
    return ModelClient(
        Endpoint(CHAT, fake.base_url, "fake-chat", "sk-test", timeout),
        Endpoint(EMBEDDINGS, fake.base_url + "/", "fake-embed", "", timeout),
        **kwargs,  # type: ignore[arg-type]
    )


def test_chat_sends_the_explicit_model_and_reads_usage(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[chat_json({"facts": ["Karim habite à Lyon"]})])
    with client_for(fake_openai) as client:
        result = client.chat(
            [
                {"role": "system", "content": "Réponds en JSON"},
                {"role": "user", "content": "Karim"},
            ],
            model="fake-chat",
            json_object=True,
            max_tokens=50,
        )
    assert result.content == '{"facts": ["Karim habite à Lyon"]}'
    assert result.model == "fake-chat" and result.finish_reason == "stop"
    assert result.usage.prompt_tokens > 0 and not result.usage.estimated
    assert result.usage.total == result.usage.prompt_tokens + result.usage.completion_tokens
    call = fake_openai.chat_calls[0]
    assert call.body["model"] == "fake-chat" and call.body["stream"] is False
    assert call.body["response_format"] == {"type": "json_object"}
    assert call.body["max_tokens"] == 50 and call.body["temperature"] == 0.0
    assert call.headers["authorization"] == "Bearer sk-test"
    assert client.status()[CHAT] == {
        "model": "fake-chat",
        "url": fake_openai.base_url,
        "calls": 1,
        "failures": 0,
        "last_error": None,
    }


def test_embeddings_keep_input_order_and_need_no_api_key(fake_openai: FakeOpenAIServer) -> None:
    with client_for(fake_openai) as client:
        result = client.embed(["projet solaire", "solar project"], model="fake-embed")
    assert result.dimension == fake_openai.embedding_dimension == len(result.vectors[0])
    assert len(result.vectors) == 2 and result.vectors[0] != result.vectors[1]
    call = fake_openai.embedding_calls[0]
    assert call.body == {"model": "fake-embed", "input": ["projet solaire", "solar project"]}
    assert "authorization" not in call.headers
    with pytest.raises(ValueError, match="nothing to embed"), client_for(fake_openai) as client:
        client.embed([], model="fake-embed")


def test_the_model_is_required_on_every_call(fake_openai: FakeOpenAIServer) -> None:
    with client_for(fake_openai) as client:
        with pytest.raises(ValueError, match="explicit"):
            client.chat([{"role": "user", "content": "x"}], model="")
        with pytest.raises(ValueError, match="explicit"):
            client.embed(["x"], model="   ")
    assert fake_openai.calls == []
    with pytest.raises(ValueError, match="explicit"):
        Endpoint(CHAT, fake_openai.base_url, "")
    with pytest.raises(ValueError, match="http"):
        Endpoint(CHAT, "ftp://models", "m")
    with pytest.raises(ValueError, match="kind"):
        Endpoint("vision", fake_openai.base_url, "m")


def test_an_unconfigured_endpoint_is_refused_without_a_call(fake_openai: FakeOpenAIServer) -> None:
    with ModelClient(Endpoint(CHAT, fake_openai.base_url, "fake-chat")) as client:
        assert client.embeddings_endpoint is None
        with pytest.raises(NoEndpoint) as refused:
            client.embed(["x"], model="fake-embed")
        assert refused.value.reason == "no_endpoint" and not refused.value.retryable
        assert client.status() == {
            CHAT: {
                "model": "fake-chat",
                "url": fake_openai.base_url,
                "calls": 0,
                "failures": 0,
                "last_error": None,
            },
            EMBEDDINGS: None,
        }
    assert fake_openai.calls == []


@pytest.mark.parametrize(
    ("behaviour", "reason", "retryable"),
    [
        (Fail(500), "http_500", True),
        (Fail(429, "slow down"), "http_429", True),
        (Fail(401, "bad key"), "http_401", False),
        (Fail(400, "bad request"), "http_400", False),
        (Malformed(), "bad_json", True),
        (Malformed(body=b'{"choices": []}'), "bad_shape", True),
        (Reply(content=""), None, True),
    ],
)
def test_failures_are_short_reasons(
    fake_openai: FakeOpenAIServer, behaviour: object, reason: str | None, retryable: bool
) -> None:
    fake_openai.script(chat=[behaviour])  # type: ignore[list-item]
    with client_for(fake_openai) as client:
        if reason is None:
            assert client.chat([{"role": "user", "content": "x"}], model="fake-chat").content == ""
            return
        with pytest.raises(ModelError) as failed:
            client.chat([{"role": "user", "content": "x"}], model="fake-chat")
        assert failed.value.reason == reason and failed.value.retryable is retryable
        assert failed.value.endpoint == CHAT
        status = client.status()[CHAT]
        assert status == {
            "model": "fake-chat",
            "url": fake_openai.base_url,
            "calls": 1,
            "failures": 1,
            "last_error": reason,
        }


def test_timeouts_and_unreachable_endpoints(fake_openai: FakeOpenAIServer) -> None:
    """A hang after the connection is a ``timeout``; a closed port is ``unreachable``
    on every platform: Windows turns it into a connect timeout, which
    ``iifa.transport`` classifies as unreachable (the deterministic version of
    that case is in tests/unit/test_transport.py)."""
    fake_openai.script(chat=[Hang(seconds=2.0)])
    with client_for(fake_openai, timeout=0.3) as client:
        started = time.monotonic()
        with pytest.raises(ModelError) as timed_out:
            client.chat([{"role": "user", "content": "x"}], model="fake-chat")
        assert timed_out.value.reason == "timeout" and timed_out.value.retryable
        assert time.monotonic() - started < 1.5
    with ModelClient(Endpoint(CHAT, "http://127.0.0.1:9", "m", timeout=1.0)) as client:
        with pytest.raises(ModelError) as unreachable:
            client.chat([{"role": "user", "content": "x"}], model="m")
        assert unreachable.value.reason == "unreachable" and unreachable.value.retryable


def test_usage_is_estimated_when_the_server_sends_none(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(
        chat=[
            chat_json(
                {
                    "choices": [{"message": {"content": "Bonjour"}, "finish_reason": None}],
                    "model": "served-model",
                }
            )
        ]
    )
    # the scripted content is JSON *text*; make the fake return it as the whole body
    fake_openai.reset()
    fake_openai.script(
        chat=[
            Malformed(
                body=(
                    b'{"choices": [{"message": {"content": "Bonjour"}}], "model": "served-model"}'
                )
            )
        ]
    )
    with client_for(fake_openai) as client:
        result = client.chat([{"role": "user", "content": "Salut, ça va ?"}], model="fake-chat")
    assert result.content == "Bonjour" and result.model == "served-model"
    assert result.finish_reason is None
    assert result.usage.estimated and result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0


class _BlockingGate:
    """A gate that never opens: what a yield looks like from the client's side."""

    def __init__(self) -> None:
        self.waits: list[float | None] = []
        self.released = 0

    def acquire(self, kind: str, *, timeout: float | None) -> bool:
        self.waits.append(timeout)
        if timeout is not None:
            time.sleep(min(timeout, 0.05))
        return False

    def release(self, kind: str) -> None:
        self.released += 1


def test_a_closed_gate_means_no_request_at_all(fake_openai: FakeOpenAIServer) -> None:
    gate = _BlockingGate()
    with client_for(fake_openai, gate=gate) as client:
        with pytest.raises(ModelWaitTimeout) as waited:
            client.chat([{"role": "user", "content": "x"}], model="fake-chat", wait_timeout=0.01)
        assert waited.value.reason == "wait_timeout" and waited.value.retryable
        with pytest.raises(ModelWaitTimeout):
            client.embed(["x"], model="fake-embed")
    assert gate.waits == [0.01, None] and gate.released == 0
    assert fake_openai.calls == []  # nothing left the service


class _CountingGate:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def acquire(self, kind: str, *, timeout: float | None) -> bool:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        return True

    def release(self, kind: str) -> None:
        with self.lock:
            self.active -= 1


def test_the_gate_is_released_after_every_call_even_a_failed_one(
    fake_openai: FakeOpenAIServer,
) -> None:
    gate = _CountingGate()
    fake_openai.script(chat=[Fail(500), Reply("ok")])
    with client_for(fake_openai, gate=gate) as client:
        with pytest.raises(ModelError):
            client.chat([{"role": "user", "content": "x"}], model="fake-chat")
        client.chat([{"role": "user", "content": "x"}], model="fake-chat")
    assert gate.active == 0 and gate.peak == 1
