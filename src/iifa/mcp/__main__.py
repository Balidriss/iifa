"""``python -m iifa.mcp``: the MCP server over stdio, for harnesses that launch
their MCP servers as subprocesses (Claude Code, Goose, opencode, Cursor...).

It talks to a running iifa service:

    IIFA_URL     the service (default http://127.0.0.1:8765)
    IIFA_TOKEN   the bearer token to use (required)
    IIFA_SCOPE   the default scope of every call (else tools need `scope`)
    IIFA_DEFAULT_BUDGET_TOKENS, IIFA_MCP_EXPOSE_FORGET, IIFA_TIMEOUT

A service that is down when the server starts is not an error: tools answer
with an error result until it comes back (P3).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from iifa.client import DEFAULT_TIMEOUT, DEFAULT_URL, AsyncIifaClient
from iifa.mcp.server import MCPConfig, build_server

_TRUE = ("1", "true", "yes", "on")


def parse_args(argv: list[str] | None, env: Mapping[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m iifa.mcp", description="iifa memory tools over MCP stdio"
    )
    parser.add_argument("--url", default=env.get("IIFA_URL") or DEFAULT_URL, help="the service")
    parser.add_argument(
        "--scope", default=env.get("IIFA_SCOPE") or None, help="default scope of every call"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=int(env.get("IIFA_DEFAULT_BUDGET_TOKENS") or 400),
        help="default budget_tokens (400)",
    )
    parser.add_argument(
        "--expose-forget",
        action="store_true",
        default=(env.get("IIFA_MCP_EXPOSE_FORGET") or "").strip().lower() in _TRUE,
        help="expose memory_forget (still needs a token with the forget permission)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(env.get("IIFA_TIMEOUT") or DEFAULT_TIMEOUT),
        help="HTTP timeout toward the service, seconds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    env = os.environ
    args = parse_args(argv, env)
    token = env.get("IIFA_TOKEN") or ""
    if token.startswith("${"):
        sys.stderr.write(
            f"iifa.mcp: IIFA_TOKEN is an unexpanded placeholder ({token}); configure the "
            "plugin option or set IIFA_TOKEN\n"
        )
        return 2
    if not token:
        sys.stderr.write(
            "iifa.mcp: IIFA_TOKEN is not set (a token the service is configured with)\n"
        )
        return 2
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(name)s %(message)s")

    shared: list[AsyncIifaClient] = []

    @asynccontextmanager
    async def client_for(headers: Mapping[str, str] | None) -> AsyncIterator[AsyncIifaClient]:
        if not shared:
            shared.append(AsyncIifaClient(args.url, token, timeout=args.timeout))
        yield shared[0]

    server = build_server(
        client_for,
        MCPConfig(
            default_budget_tokens=args.budget,
            default_scope=args.scope,
            expose_forget=args.expose_forget,
        ),
    )
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
