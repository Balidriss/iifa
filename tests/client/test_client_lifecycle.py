"""The SDK methods for the lifecycle and operations routes, in-process (async)
and over a socket (blocking), against the real service."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from iifa.client import AsyncIifaClient, AuthError, IifaClient, InvalidRequest
from tests.fakes.live_service import LiveService
from tests.helpers import TOKENS


def aclient(app: FastAPI, token: str = "all", **kw: Any) -> AsyncIifaClient:
    return AsyncIifaClient(
        "http://iifa.test", TOKENS[token][0], transport=httpx.ASGITransport(app=app), **kw
    )


def test_async_lifecycle_and_operations(app: FastAPI) -> None:
    async def scenario() -> None:
        async with aclient(app, scope="user:alice") as memory:
            assert (await memory.set_background(mode="on_demand"))["mode"] == "on_demand"
            written = await memory.remember("Always answer in French", pinned=True)
            block = await memory.session_start(session_id="s1", budget_tokens=300)
            assert [i["id"] for i in block["items"]] == [written["fact"]["id"]]
            assert block["session_id"] == "s1" and block["used_tokens"] <= 300
            again = await memory.session_start(exclude_ids=[written["fact"]["id"]])
            assert again["items"] == []
            turn = await memory.turn(
                "who leads solar?", "Karim.", tool_summaries=["crm"], session_id="s1"
            )
            assert turn["kind"] == "turn" and turn["queue"]["depth"] == 1
            compact = await memory.precompact(
                [{"role": "user", "content": "old"}],
                session_id="s1",
                trigger="auto",
                budget_tokens=200,
            )
            assert compact["kind"] == "precompact" and compact["items"]
            ended = await memory.session_end(session_id="s1", reason="clear")
            assert ended["kind"] == "session_end"
            status = await memory.background()
            assert status["queue"]["depth"] == 3 and status["mode"] == "on_demand"
            yielded = await memory.yield_background(timeout_seconds=2)
            assert yielded == {"yielded": True, "timeout_seconds": 2.0}
            assert (await memory.resume_background()) == {"yielded": False}
            report = await memory.run_background(calls=10, seconds=30, max_jobs=2)
            assert report["stopped_by"] == "max_jobs" and report["processed"]["done"] == 2
            held: list[Any] = []
            app.state.worker._spawn = held.append
            started = await memory.run_background(wait=False)
            assert started["status"] == "running"
            live = await memory.background_run(started["run_id"])
            assert live["status"] == "running" and live["jobs_left"] == 1
            held[0]()
            report = await memory.background_run(started["run_id"])
            assert report["status"] == "done" and report["stopped_by"] == "queue_empty"
            changed = await memory.set_background(mode="auto", chat_concurrency=2)
            assert changed["mode"] == "auto" and changed["concurrency"]["chat"] == 2
            with pytest.raises(InvalidRequest):
                await memory.turn()  # nothing to record
            with pytest.raises(InvalidRequest):
                await memory.set_background(mode="sometimes")
        async with aclient(app, "read", scope="user:alice") as reader:
            assert (await reader.session_start())["reason"] in (None, "empty_profile")
            with pytest.raises(AuthError):
                await reader.turn("x")
            with pytest.raises(AuthError):
                await reader.background()
            with pytest.raises(AuthError):
                await reader.yield_background()

    asyncio.run(scenario())


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveService]:
    with LiveService(tmp_path_factory.mktemp("client-lifecycle"), worker_poll=0.02) as service:
        yield service


def test_blocking_client_over_a_socket(live: LiveService) -> None:
    with IifaClient(live.url, live.token(), scope="user:alice/agent:demo") as memory:
        memory.remember("Nadia moved to Lyon", node="Nadia", pinned=True)
        block = memory.session_start(budget_tokens=500)
        assert block["items"][0]["statement"] == "Nadia moved to Lyon"
        assert memory.turn("hello", "hi")["status"] == "pending"
        assert memory.precompact()["kind"] == "precompact"
        assert memory.session_end()["kind"] == "session_end"
        assert memory.background()["mode"] == "auto"
        assert memory.yield_background()["yielded"] is True
        assert memory.resume_background()["yielded"] is False
        ran = memory.run_background(max_jobs=1)
        assert memory.background_run(ran["run_id"])["status"] == "done"
