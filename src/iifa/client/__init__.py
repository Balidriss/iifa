"""The Python SDK: ``IifaClient`` (blocking) and ``AsyncIifaClient`` (asyncio),
one method per route of the HTTP API, and a small error hierarchy.

    from iifa.client import IifaClient

    with IifaClient("http://127.0.0.1:8765", token, scope="user:alice") as memory:
        handles = memory.recall("who works on the solar project?", budget_tokens=300)
        content = memory.expand([h["id"] for h in handles["items"][:2]], level=2)
        memory.remember("Karim leads the solar installation planning", node="Karim")

An adapter catches ``IifaUnavailable`` to fail open (P3) and wraps what it
injects with ``iifa.recalled.as_recalled_data`` (P8).
"""

from iifa.client._base import (
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    AuthError,
    IifaError,
    IifaHTTPError,
    IifaProtocolError,
    IifaUnavailable,
    InvalidRequest,
    NotFoundError,
    ServerError,
)
from iifa.client.aio import AsyncIifaClient
from iifa.client.sync import IifaClient

__all__ = [
    "DEFAULT_TIMEOUT",
    "DEFAULT_URL",
    "AsyncIifaClient",
    "AuthError",
    "IifaClient",
    "IifaError",
    "IifaHTTPError",
    "IifaProtocolError",
    "IifaUnavailable",
    "InvalidRequest",
    "NotFoundError",
    "ServerError",
]
