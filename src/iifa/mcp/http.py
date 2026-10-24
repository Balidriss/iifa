"""Streamable HTTP transport: the MCP server mounted on the service's own app.

One service process serves the API and MCP over HTTP (SPEC 7.2). The endpoint
lives at ``/mcp``, requires the same bearer token as the API (a request without
a valid token is refused before the MCP layer sees it), and each tool call runs
against the API in-process with the caller's token, so permissions and scopes
apply exactly as they do over HTTP.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.security.utils import get_authorization_scheme_param
from mcp.server.mcpserver import MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from iifa.api.auth import TokenRegistry
from iifa.client import AsyncIifaClient
from iifa.config import Settings
from iifa.mcp.server import MCPConfig, build_server

MCP_PATH = "/mcp"
_INTERNAL_URL = "http://iifa.internal"
_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def bearer_token(headers: Mapping[str, str] | None) -> str | None:
    if not headers:
        return None
    scheme, credentials = get_authorization_scheme_param(headers.get("authorization"))
    return credentials if scheme.lower() == "bearer" and credentials else None


class BearerGuard:
    """Refuses any request to the MCP endpoint that does not carry a configured token."""

    def __init__(self, app: ASGIApp, registry: TokenRegistry) -> None:
        self._app = app
        self._registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            token = bearer_token(Headers(scope=scope))
            if token is None or self._registry.authenticate(token) is None:
                response = JSONResponse(
                    {"detail": "bearer token required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


def _security(host: str) -> TransportSecuritySettings:
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
    if host not in _LOOPBACK:
        hosts.append(f"{host}:*")
        origins.extend([f"http://{host}:*", f"https://{host}:*"])
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


def mount_mcp(app: FastAPI, settings: Settings) -> MCPServer:
    """Add ``/mcp`` to ``app``. Tool calls go through the API in-process with the caller's token."""
    registry: TokenRegistry = app.state.registry

    @asynccontextmanager
    async def client_for(headers: Mapping[str, str] | None) -> AsyncIterator[AsyncIifaClient]:
        async with AsyncIifaClient(
            _INTERNAL_URL,
            bearer_token(headers),
            transport=httpx.ASGITransport(app=app),
            timeout=60.0,
        ) as client:
            yield client

    server = build_server(
        client_for,
        MCPConfig(
            default_budget_tokens=settings.default_budget_tokens,
            max_budget_tokens=settings.max_budget_tokens,
            default_scope=settings.mcp_default_scope,
            expose_forget=settings.mcp_expose_forget,
            tokenizer=app.state.engine.tokenizer,
        ),
    )
    # Build the SDK's transport (this creates the session manager), then serve
    # its ASGI handler from the API app behind the bearer guard.
    server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=True,
        transport_security=_security(settings.host),
        host=settings.host,
    )
    app.state.mcp_server = server
    guard = BearerGuard(StreamableHTTPASGIApp(server.session_manager), registry)
    app.router.routes.append(Route(MCP_PATH, endpoint=guard, include_in_schema=False))
    return server


@asynccontextmanager
async def mcp_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run the MCP session manager for the lifetime of the app (the mounted route
    has no lifespan of its own)."""
    server: MCPServer | None = getattr(app.state, "mcp_server", None)
    if server is None:
        yield
        return
    async with server.session_manager.run():
        yield
