"""The model client and real models (SPEC 7.4b): a truncated answer is a failure
the caller never parses, extra request parameters reach every request body of
their endpoint and cannot replace what iifa sets, and JSON mode is optional."""

from __future__ import annotations

import pytest

from iifa.llm import CHAT, EMBEDDINGS, Endpoint, ModelClient, ModelError
from tests.fakes.openai_server import FakeOpenAIServer, Reply


def client_for(
    fake: FakeOpenAIServer,
    *,
    chat_extra: dict[str, object] | None = None,
    embeddings_extra: dict[str, object] | None = None,
) -> ModelClient:
    return ModelClient(
        Endpoint(CHAT, fake.base_url, "fake-chat", extra=chat_extra or {}),
        Endpoint(EMBEDDINGS, fake.base_url, "fake-embed", extra=embeddings_extra or {}),
    )


def test_a_truncated_answer_is_a_failure(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[Reply('{"nodes": [{"name": "Kar', finish_reason="length")])
    with client_for(fake_openai) as client:
        with pytest.raises(ModelError) as failed:
            client.chat([{"role": "user", "content": "x"}], model="fake-chat", max_tokens=8)
        assert failed.value.reason == "length" and not failed.value.retryable
        status = client.status()[CHAT]
        assert status is not None and status["calls"] == 1 and status["failures"] == 1
        assert status["last_error"] == "length"


def test_a_caller_may_accept_a_truncated_answer_explicitly(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[Reply("po", finish_reason="length")])
    with client_for(fake_openai) as client:
        result = client.chat(
            [{"role": "user", "content": "ping"}],
            model="fake-chat",
            max_tokens=1,
            accept_truncated=True,
        )
    assert result.content == "po" and result.finish_reason == "length"


def test_extras_reach_every_request_body_of_their_endpoint(fake_openai: FakeOpenAIServer) -> None:
    with client_for(
        fake_openai,
        chat_extra={"reasoning_effort": "none", "options": {"num_ctx": 8192}},
        embeddings_extra={"dimensions": 8},
    ) as client:
        client.chat([{"role": "user", "content": "x"}], model="fake-chat", max_tokens=5)
        client.chat([{"role": "user", "content": "y"}], model="fake-chat", json_object=True)
        client.embed(["x"], model="fake-embed")
    for call in fake_openai.chat_calls:
        assert call.body["reasoning_effort"] == "none"
        assert call.body["options"] == {"num_ctx": 8192}
        assert "dimensions" not in call.body
    assert fake_openai.embedding_calls[0].body["dimensions"] == 8
    assert "reasoning_effort" not in fake_openai.embedding_calls[0].body


@pytest.mark.parametrize("key", ["model", "messages", "max_tokens", "response_format", "input"])
def test_an_endpoint_refuses_an_extra_that_replaces_what_iifa_sets(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        Endpoint(CHAT, "http://127.0.0.1:1/v1", "m", extra={key: "x"})


def test_json_mode_off_sends_no_response_format(fake_openai: FakeOpenAIServer) -> None:
    with client_for(fake_openai) as client:
        client.chat([{"role": "user", "content": "x"}], model="fake-chat", json_object=False)
    assert "response_format" not in fake_openai.chat_calls[0].body
