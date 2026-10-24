"""The fake model server must itself be trustworthy: every failure mode it is
scripted to produce has to happen, and every call has to be recorded."""

from __future__ import annotations

import json
import time

import httpx
import pytest

from tests.fakes.openai_server import (
    CHAT_PATH,
    EMBEDDINGS_PATH,
    Fail,
    FakeOpenAIServer,
    Hang,
    Malformed,
    Reply,
    chat_json,
    deterministic_vector,
)


def _chat(server: FakeOpenAIServer, text: str = "hello", **extra: object) -> httpx.Response:
    payload: dict[str, object] = {
        "model": "fake-chat",
        "messages": [{"role": "user", "content": text}],
        **extra,
    }
    return httpx.post(server.base_url + "/chat/completions", json=payload, timeout=10)


def test_default_chat_reply_and_call_log(fake_openai: FakeOpenAIServer) -> None:
    response = _chat(fake_openai, "Bonjour")
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "OK"
    assert body["model"] == "fake-chat"
    assert len(fake_openai.calls) == 1
    call = fake_openai.calls[0]
    assert call.path == CHAT_PATH
    assert call.model == "fake-chat"
    assert call.messages == [{"role": "user", "content": "Bonjour"}]
    assert call.status == 200


def test_scripted_behaviours_are_consumed_in_order(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[Reply("première"), Fail(503), Malformed()])
    assert _chat(fake_openai).json()["choices"][0]["message"]["content"] == "première"
    second = _chat(fake_openai)
    assert second.status_code == 503
    assert second.json()["error"]["type"] == "scripted_error"
    third = _chat(fake_openai)
    assert third.status_code == 200
    with pytest.raises(json.JSONDecodeError):
        third.json()
    # After the script, the default applies again.
    assert _chat(fake_openai).json()["choices"][0]["message"]["content"] == "OK"
    assert [c.status for c in fake_openai.calls] == [200, 503, 200, 200]


def test_chat_json_helper_round_trips_unicode(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[chat_json({"facts": ["Karim habite à Lyon", "كريم"]})])
    content = _chat(fake_openai).json()["choices"][0]["message"]["content"]
    assert json.loads(content) == {"facts": ["Karim habite à Lyon", "كريم"]}


def test_hang_exceeds_client_timeout_and_is_still_recorded(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[Hang(seconds=0.6)])
    started = time.monotonic()
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(
            fake_openai.base_url + "/chat/completions",
            json={"model": "m", "messages": []},
            timeout=httpx.Timeout(5, read=0.2),
        )
    assert time.monotonic() - started < 2
    assert len(fake_openai.calls) == 1


def test_delay_is_observable(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[Reply("slow", delay=0.15)])
    started = time.monotonic()
    assert _chat(fake_openai).status_code == 200
    assert time.monotonic() - started >= 0.15


def test_model_is_required_on_every_call(fake_openai: FakeOpenAIServer) -> None:
    response = httpx.post(
        fake_openai.base_url + "/chat/completions", json={"messages": []}, timeout=10
    )
    assert response.status_code == 400
    assert "model" in response.json()["error"]["message"]


def test_streaming_is_refused_loudly(fake_openai: FakeOpenAIServer) -> None:
    assert _chat(fake_openai, stream=True).status_code == 400


def test_embeddings_are_deterministic_and_unit_length(fake_openai: FakeOpenAIServer) -> None:
    payload = {"model": "fake-embed", "input": ["solar project", "projet solaire"]}
    first = httpx.post(fake_openai.base_url + "/embeddings", json=payload, timeout=10).json()
    second = httpx.post(fake_openai.base_url + "/embeddings", json=payload, timeout=10).json()
    assert first["data"] == second["data"]
    assert [d["index"] for d in first["data"]] == [0, 1]
    vector = first["data"][0]["embedding"]
    assert len(vector) == fake_openai.embedding_dimension
    assert abs(sum(v * v for v in vector) - 1.0) < 1e-5
    assert first["data"][0]["embedding"] != first["data"][1]["embedding"]
    assert fake_openai.calls[0].path == EMBEDDINGS_PATH
    assert len(fake_openai.embedding_calls) == 2
    assert fake_openai.chat_calls == []


def test_embeddings_accept_a_single_string_and_scripted_vectors(
    fake_openai: FakeOpenAIServer,
) -> None:
    fake_openai.script(embeddings=[Reply(vectors=[[1.0, 0.0, 0.0]])])
    body = httpx.post(
        fake_openai.base_url + "/embeddings",
        json={"model": "fake-embed", "input": "one"},
        timeout=10,
    ).json()
    assert body["data"][0]["embedding"] == [1.0, 0.0, 0.0]


def test_deterministic_vector_handles_any_dimension() -> None:
    for dimension in (1, 3, 8, 13, 64):
        vector = deterministic_vector("x", dimension)
        assert len(vector) == dimension
        assert abs(sum(v * v for v in vector) - 1.0) < 1e-5


def test_models_route_and_unknown_routes(fake_openai: FakeOpenAIServer) -> None:
    models = httpx.get(fake_openai.base_url + "/models", timeout=10).json()
    assert {m["id"] for m in models["data"]} == {"fake-chat", "fake-embed"}
    assert httpx.get(fake_openai.base_url + "/nope", timeout=10).status_code == 404
    assert fake_openai.calls == []  # only model routes are recorded


def test_reset_clears_log_and_scripts(fake_openai: FakeOpenAIServer) -> None:
    fake_openai.script(chat=[Fail(500)])
    fake_openai.reset()
    assert _chat(fake_openai).status_code == 200
    assert len(fake_openai.calls) == 1


def test_a_second_server_gets_its_own_port() -> None:
    with FakeOpenAIServer(embedding_dimension=3) as other:
        assert other.port != 0
        body = httpx.post(
            other.base_url + "/embeddings", json={"model": "e", "input": "x"}, timeout=10
        ).json()
        assert len(body["data"][0]["embedding"]) == 3
