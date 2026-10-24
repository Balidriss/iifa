"""MCP tests: the server built in memory over the in-process app, driven by the
official client; transports get their own module."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from mcp.server.mcpserver import MCPServer

from iifa.client import AsyncIifaClient
from iifa.mcp import MCPConfig, build_server
from mcp import Client
from tests.helpers import TOKENS


def run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)  # type: ignore[arg-type]


def make_server(
    app: FastAPI,
    *,
    token: str = "alice",
    base_url: str = "http://iifa.test",
    forward_headers: bool = False,
    **config: Any,
) -> MCPServer:
    """A server whose tools call ``app`` in-process as ``token`` (or as the bearer
    the MCP request carried, when ``forward_headers``)."""

    @asynccontextmanager
    async def client_for(headers: Mapping[str, str] | None) -> AsyncIterator[AsyncIifaClient]:
        bearer = TOKENS[token][0]
        if forward_headers and headers and headers.get("authorization"):
            bearer = headers["authorization"].split(" ", 1)[1]
        async with AsyncIifaClient(
            base_url, bearer, transport=httpx.ASGITransport(app=app), timeout=5.0
        ) as client:
            yield client

    return build_server(client_for, MCPConfig(**config))


ToolCall = Callable[[str, dict[str, Any]], Awaitable[Any]]


@pytest.fixture
def server(app: FastAPI) -> MCPServer:
    return make_server(app, default_scope="user:alice/agent:coder")


@pytest.fixture
def call(server: MCPServer) -> Callable[[str, dict[str, Any]], Any]:
    """``call(tool, arguments)`` -> the CallToolResult, through the official client."""

    def _call(name: str, arguments: dict[str, Any]) -> Any:
        async def go() -> Any:
            async with Client(server) as client:
                return await client.call_tool(name, arguments)

        return run(go())

    return _call


def text_of(result: Any) -> str:
    return "".join(block.text for block in result.content if getattr(block, "text", None))
