"""Helpers shared by API, security and perf tests: tokens, settings, a remember shortcut."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from iifa.config import PERMISSIONS, Settings, TokenConfig

KEY = "test-key-not-a-secret"

# name -> (plaintext token, scopes, permissions)
TOKENS: dict[str, tuple[str, tuple[str, ...], frozenset[str]]] = {
    "all": ("tok-all", ("*",), PERMISSIONS),
    "read": ("tok-read", ("*",), frozenset({"read"})),
    "write": ("tok-write", ("*",), frozenset({"write"})),
    "forget": ("tok-forget", ("*",), frozenset({"forget"})),
    "dream": ("tok-dream", ("*",), frozenset({"dream"})),
    "admin": ("tok-admin", ("*",), frozenset({"admin"})),
    "review": ("tok-review", ("*",), frozenset({"review"})),
    "alice": ("tok-alice", ("user:alice",), frozenset({"read", "write"})),
    "alice-coder": ("tok-alice-coder", ("user:alice/agent:coder",), frozenset({"read", "write"})),
    "bob": ("tok-bob", ("user:bob",), frozenset({"read", "write", "forget"})),
    "alice-admin": ("tok-alice-admin", ("user:alice",), frozenset({"read", "admin"})),
    "alice-dream": ("tok-alice-dream", ("user:alice",), frozenset({"read", "dream"})),
    "alice-review": (
        "tok-alice-review",
        ("user:alice",),
        frozenset({"read", "write", "review"}),
    ),
    "bob-review": ("tok-bob-review", ("user:bob",), frozenset({"read", "review"})),
    "bob-dream": ("tok-bob-dream", ("user:bob",), frozenset({"read", "dream"})),
}


def auth(name: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[name][0]}"}


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    tokens = tuple(
        TokenConfig.from_plaintext(token, name=name, scopes=scopes, permissions=perms)
        for name, (token, scopes, perms) in TOKENS.items()
    )
    values: dict[str, Any] = {"db_path": tmp_path / "memory.db", "db_key": KEY, "tokens": tokens}
    values.update(overrides)
    return Settings(**values)


def remember(
    client: TestClient, statement: str, *, scope: str = "user:alice", **extra: Any
) -> dict[str, Any]:
    body_in: dict[str, Any] = {"statement": statement, **extra}
    if "scopes" not in extra:
        body_in["scope"] = scope
    response = client.post("/v1/remember", json=body_in)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def wait_until(
    predicate: Callable[[], bool], *, timeout: float = 5.0, interval: float = 0.01
) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` passes (a bounded wait on a
    background thread, never a fixed sleep). Returns whether it held."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
