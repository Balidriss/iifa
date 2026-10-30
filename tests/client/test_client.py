"""The Python SDK against the real service: the async client in-process (ASGI
transport), the blocking client over a loopback socket, and the error
hierarchy against the scripted fake service."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from iifa.client import (
    AsyncIifaClient,
    AuthError,
    IifaClient,
    IifaHTTPError,
    IifaProtocolError,
    IifaUnavailable,
    InvalidRequest,
    NotFoundError,
    ServerError,
)
from iifa.client._base import Call, normalize_base_url, unavailable
from tests.fakes.iifa_service import Fail, FakeIifaService, Hang, Malformed, Reply
from tests.fakes.live_service import LiveService
from tests.helpers import TOKENS


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def aclient(app: FastAPI, token: str = "all", **kw: Any) -> AsyncIifaClient:
    return AsyncIifaClient(
        "http://iifa.test", TOKENS[token][0], transport=httpx.ASGITransport(app=app), **kw
    )


# --------------------------------------------------------------------------- async, in-process


def test_async_client_round_trip(app: FastAPI) -> None:
    async def scenario() -> None:
        async with aclient(app, scope="user:alice") as memory:
            health = await memory.health()
            assert health["status"] == "ok" and health["tier"] == 0
            written = await memory.remember(
                "Karim leads the solar installation planning",
                node={"name": "Karim", "kind": "person", "aliases": ["K."]},
                attribute="role",
                value="lead",
                pinned=True,
                external_id="crm-42",
            )
            assert written["created_node"] is True
            fact_id, node_id = written["fact"]["id"], written["node"]["id"]
            recalled = await memory.recall("solar", budget_tokens=500)
            assert next(h["id"] for h in recalled["items"]) in (fact_id, node_id)
            assert recalled["used_tokens"] <= 500
            expanded = await memory.expand([fact_id], level=2, budget_tokens=800)
            assert expanded["items"][0]["statement"].startswith("Karim leads")
            assert expanded["items"][0]["excerpt"].startswith("Karim leads")
            node = await memory.node(node_id, budget_tokens=800)
            assert node["name"] == "Karim" and node["facts"][0]["id"] == fact_id
            by_ext = await memory.lookup(external_id="crm-42")
            assert by_ext["records"] == [{"id": fact_id, "kind": "fact"}]
            by_name = await memory.lookup(name="k.")
            assert by_name["records"] == [{"id": node_id, "kind": "node", "name": "Karim"}]
            other = await memory.remember("Nadia moved to Lyon", node="Nadia")
            linked = await memory.link(node_id, other["node"]["id"], "knows", weight=2.0)
            assert linked["kind"] == "edge" and linked["status"] == "confirmed"
            archived = await memory.archive(fact_id)
            assert archived == {"id": fact_id, "kind": "fact", "archived": True}
            listed = await memory.archived(budget_tokens=500)
            assert [a["id"] for a in listed["items"]] == [fact_id]
            restored = await memory.restore(fact_id)
            assert restored["archived"] is False
            patched = await memory.patch_fact(fact_id, detach=True, pinned=False)
            assert patched["node_id"] is None and patched["pinned"] is False
            shared = await memory.patch_fact(fact_id, scopes=["user:alice", "user:alice/x:y"])
            assert shared["scopes"] == ["user:alice", "user:alice/x:y"]
            gone = await memory.forget(other["node"]["id"])
            assert gone["removed"]["nodes"] == 1
            stats = await memory.stats()
            assert stats["counts"]["nodes"] == 1

    run(scenario())


def test_async_client_maps_statuses_to_errors(app: FastAPI) -> None:
    async def scenario() -> None:
        async with aclient(app, "read", scope="user:alice") as reader:
            with pytest.raises(AuthError) as denied:
                await reader.remember("x")
            assert denied.value.status == 403 and "permission" in str(denied.value)
            with pytest.raises(NotFoundError):
                await reader.node("node_" + "0" * 26)
            with pytest.raises(InvalidRequest) as invalid:
                await reader.recall("")
            assert invalid.value.status == 422
            with pytest.raises(InvalidRequest):
                await reader.lookup()  # neither external_id nor name
        async with AsyncIifaClient(
            "http://iifa.test", "nope", transport=httpx.ASGITransport(app=app)
        ) as stranger:
            with pytest.raises(AuthError) as unauth:
                await stranger.health()
            assert unauth.value.status == 401
        async with aclient(app, "alice") as alice:
            with pytest.raises(InvalidRequest):  # no default scope, none given
                await alice.recall("solar")
            with pytest.raises(AuthError):  # a scope outside the token's bounds
                await alice.recall("solar", scope="user:bob")

    run(scenario())


# --------------------------------------------------------------------------- blocking, over a socket


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveService]:
    with LiveService(tmp_path_factory.mktemp("live")) as service:
        yield service


def test_sync_client_round_trip_over_loopback(live: LiveService) -> None:
    with IifaClient(live.url, live.token(), scope="user:sync") as memory:
        assert memory.health()["status"] == "ok"
        written = memory.remember("Émilie préfère les réponses courtes", node="Émilie")
        recalled = memory.recall("réponses courtes", budget_tokens=400)
        assert recalled["items"][0]["id"] == written["fact"]["id"]
        assert memory.expand([written["fact"]["id"]])["items"][0]["node_name"] == "Émilie"
        assert memory.lookup(name="émilie")["records"][0]["id"] == written["node"]["id"]
        assert memory.archive(written["fact"]["id"])["archived"] is True
        assert memory.recall("réponses courtes")["items"] == []
        assert memory.restore(written["fact"]["id"])["archived"] is False
        moved = memory.patch_fact(written["fact"]["id"], scopes=["user:sync/mode:a"], detach=True)
        assert moved["scopes"] == ["user:sync/mode:a"] and moved["node_id"] is None
        assert memory.recall("réponses courtes")["items"] == []  # user:sync no longer sees it


def test_sync_client_from_env_and_url_forms(live: LiveService) -> None:
    env = {"IIFA_URL": live.url + "/v1/", "IIFA_TOKEN": live.token(), "IIFA_SCOPE": "user:env"}
    with IifaClient.from_env(env) as memory:
        assert memory.base_url == live.url and memory.scope == "user:env"
        assert memory.health()["status"] == "ok"
    assert normalize_base_url("http://h:1/v1") == "http://h:1"
    with pytest.raises(ValueError):
        normalize_base_url("  ")
    with IifaClient.from_env({}) as blank:
        assert blank.base_url == "http://127.0.0.1:8765" and blank.scope is None


# --------------------------------------------------------------------------- the fake service


@pytest.fixture(scope="module")
def fake() -> Iterator[FakeIifaService]:
    with FakeIifaService() as service:
        yield service


def test_transport_failures_are_unavailable(fake: FakeIifaService, tmp_path: Path) -> None:
    fake.reset()
    fake.script("/v1/recall", [Hang(2.0)])
    with (
        IifaClient(fake.url, "t", scope="user:a", timeout=0.2) as memory,
        pytest.raises(IifaUnavailable, match="timeout"),
    ):
        memory.recall("x")
    with (
        IifaClient("http://127.0.0.1:9", "t", scope="user:a", timeout=0.5) as memory,
        pytest.raises(IifaUnavailable, match="unreachable"),
    ):
        memory.health()


def test_status_and_body_errors_from_the_fake(fake: FakeIifaService) -> None:
    fake.reset()
    fake.script(
        "/v1/recall",
        [Fail(500), Malformed(), Fail(418, "teapot"), Reply({"items": []}, status=200)],
    )
    with IifaClient(fake.url, "t", scope="user:a") as memory:
        with pytest.raises(ServerError) as boom:
            memory.recall("x")
        assert boom.value.status == 500 and boom.value.detail == "internal error (scripted)"
        with pytest.raises(IifaProtocolError):
            memory.recall("x")
        with pytest.raises(IifaHTTPError) as odd:
            memory.recall("x")
        assert odd.value.status == 418 and type(odd.value) is IifaHTTPError
        assert memory.recall("x") == {"items": []}
    sent = fake.calls_to("/v1/recall")
    assert len(sent) == 4
    assert sent[0].headers["authorization"] == "Bearer t"
    assert sent[0].body == {"query": "x", "scope": "user:a"}


@pytest.mark.parametrize(
    ("exc", "label"),
    [
        (httpx.ConnectError("refused"), "unreachable"),
        (httpx.ConnectTimeout("connect"), "unreachable"),  # Windows: a closed port looks like this
        (httpx.ReadTimeout("read"), "timeout"),
        (httpx.WriteTimeout("write"), "timeout"),
        (httpx.PoolTimeout("pool"), "timeout"),
        (httpx.RemoteProtocolError("reset"), "broken"),
    ],
)
def test_transport_errors_are_classified_by_phase(exc: httpx.TransportError, label: str) -> None:
    """The one classification of ``iifa.transport`` (its full matrix lives in
    tests/unit/test_transport.py), as the SDK exposes it."""
    error = unavailable(exc, Call("GET", "/v1/health"))
    assert isinstance(error, IifaUnavailable) and error.reason == label
    assert f"service {label}" in str(error) and type(exc).__name__ in str(error)
