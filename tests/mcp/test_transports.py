"""Both transports of SPEC 9 with the official client: stdio (``python -m iifa.mcp``
as a subprocess toward a live service) and streamable HTTP (mounted on the
service at ``/mcp``, behind the same bearer tokens as the API)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from iifa.client import IifaClient
from iifa.mcp import CORE_TOOLS
from iifa.recalled import NOTICE
from mcp import Client, StdioServerParameters
from tests.fakes.live_service import LiveService
from tests.mcp.conftest import run, text_of


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveService]:
    with LiveService(tmp_path_factory.mktemp("mcp-live")) as service:
        yield service


def payload(result: Any) -> dict[str, Any]:
    lines = text_of(result).split("\n")
    assert lines[1] == NOTICE
    body: dict[str, Any] = json.loads(lines[2])
    return body


def stdio_params(live: LiveService, **env: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "iifa.mcp"],
        env={
            **os.environ,
            "IIFA_URL": live.url,
            "IIFA_TOKEN": live.token("alice"),
            "IIFA_SCOPE": "user:alice/agent:coder",
            **env,
        },
    )


# --------------------------------------------------------------------------- stdio


@pytest.mark.security
def test_stdio_server_lists_tools_remembers_and_recalls(live: LiveService) -> None:
    async def go() -> tuple[set[str], Any, Any]:
        async with Client(stdio_params(live)) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            written = await client.call_tool(
                "memory_remember",
                {"statement": "Youssef studies physics in Rabat", "node_name": "Youssef"},
            )
            recalled = await client.call_tool(
                "memory_recall", {"query": "physics Rabat", "budget_tokens": 300}
            )
            return names, written, recalled

    names, written, recalled = run(go())
    assert names == set(CORE_TOOLS)
    assert not written.is_error and payload(written)["created_node"] is True
    assert payload(recalled)["items"][0]["id"] == payload(written)["fact"]["id"]
    # what the tools wrote is real: the HTTP API sees it
    with IifaClient(live.url, live.token(), scope="user:alice/agent:coder") as api:
        assert api.lookup(name="Youssef")["records"][0]["id"] == payload(written)["node"]["id"]


def test_stdio_server_exposes_forget_only_when_asked(live: LiveService) -> None:
    async def go() -> set[str]:
        async with Client(stdio_params(live, IIFA_MCP_EXPOSE_FORGET="1")) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert run(go()) == {*CORE_TOOLS, "memory_forget"}


def test_stdio_server_refuses_to_start_without_a_token() -> None:
    env = {k: v for k, v in os.environ.items() if k != "IIFA_TOKEN"}
    done = subprocess.run(
        [sys.executable, "-m", "iifa.mcp"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 2 and "IIFA_TOKEN" in done.stderr
    placeholder = subprocess.run(
        [sys.executable, "-m", "iifa.mcp"],
        env={**env, "IIFA_TOKEN": "${IIFA_TOKEN}"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert placeholder.returncode == 2 and "placeholder" in placeholder.stderr


def test_iifa_cli_has_an_mcp_subcommand() -> None:
    env = {k: v for k, v in os.environ.items() if k != "IIFA_TOKEN"}
    done = subprocess.run(
        [sys.executable, "-m", "iifa", "mcp", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0 and "--expose-forget" in done.stdout


@pytest.mark.security
def test_p3_stdio_server_stays_up_when_the_service_is_down(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    with LiveService(tmp_path_factory.mktemp("gone")) as service:
        params = stdio_params(service)
    # the service is stopped; its port is closed

    async def go() -> tuple[set[str], Any]:
        async with Client(params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            return names, await client.call_tool("memory_recall", {"query": "anything"})

    names, result = run(go())
    assert names == set(CORE_TOOLS)
    assert result.is_error and "unavailable" in text_of(result)


# --------------------------------------------------------------------------- streamable HTTP


def http_transport(live: LiveService, token: str | None) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return streamable_http_client(
        live.url + "/mcp", http_client=create_mcp_http_client(headers=headers)
    )


@pytest.mark.security
def test_http_endpoint_requires_a_configured_bearer_token(live: LiveService) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    accept = {"Accept": "application/json, text/event-stream"}
    anonymous = httpx.post(live.url + "/mcp", json=body, headers=accept)
    assert anonymous.status_code == 401
    assert anonymous.headers["WWW-Authenticate"] == "Bearer"
    stranger = httpx.post(
        live.url + "/mcp", json=body, headers={**accept, "Authorization": "Bearer nope"}
    )
    assert stranger.status_code == 401


def test_http_transport_uses_the_callers_token_for_every_tool_call(live: LiveService) -> None:
    async def go() -> tuple[set[str], Any, Any]:
        async with Client(http_transport(live, live.token("alice"))) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            written = await client.call_tool(
                "memory_remember",
                {"statement": "Léa manages the budget", "node_name": "Léa", "scope": "user:alice"},
            )
            forbidden = await client.call_tool(
                "memory_recall", {"query": "budget", "scope": "user:bob"}
            )
            return names, written, forbidden

    names, written, forbidden = run(go())
    assert names == set(CORE_TOOLS)
    assert not written.is_error and payload(written)["fact"]["scopes"] == ["user:alice"]
    assert forbidden.is_error and "403" in text_of(forbidden)  # alice's token, bob's scope

    async def as_reader() -> Any:
        async with Client(http_transport(live, live.token("read"))) as client:
            return await client.call_tool(
                "memory_remember", {"statement": "x", "scope": "user:alice"}
            )

    denied = run(as_reader())
    assert denied.is_error and "403" in text_of(denied) and "write" in text_of(denied)
