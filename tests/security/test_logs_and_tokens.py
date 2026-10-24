"""SPEC 12: logs carry no memory content; tokens are hashed and compared in
constant time; the default bind address is loopback."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import iifa.api.auth as auth_module
from iifa import log
from iifa.api.app import create_app
from iifa.api.auth import TokenRegistry
from iifa.config import Settings, TokenConfig, load_settings
from tests.api.conftest import auth, remember
from tests.fixtures.corpus import CANARY

pytestmark = pytest.mark.security


def test_logs_contain_no_memory_content(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    fact = remember(
        client,
        f"Karim said {CANARY}",
        node={"name": f"Node {CANARY}", "aliases": [f"alias {CANARY}"]},
    )
    client.post("/v1/recall", json={"scope": "user:alice", "query": CANARY})
    client.post("/v1/expand", json={"scope": "user:alice", "ids": [fact["fact"]["id"]], "level": 2})
    client.get(f"/v1/nodes/{fact['node']['id']}", params={"scope": "user:alice"})
    client.post("/v1/archive", json={"id": fact["fact"]["id"]})
    client.post("/v1/restore", json={"id": fact["fact"]["id"]})
    client.patch(f"/v1/facts/{fact['fact']['id']}", json={"pinned": True})
    client.post("/v1/forget", json={"id": fact["node"]["id"]})
    assert caplog.records, "the service logged nothing at all?"
    assert any(r.name == "iifa" for r in caplog.records)
    assert CANARY not in caplog.text
    assert "Karim" not in caplog.text


def test_log_helper_refuses_long_values() -> None:
    with pytest.raises(ValueError, match="too long"):
        log.event("x", text="a" * 65)
    log.event("x", count=3, record="fact_" + "0" * 26, mode=None)


def test_tokens_are_hashed_and_compared_in_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = TokenRegistry(
        [
            TokenConfig.from_plaintext(
                "alpha", name="a", scopes=("*",), permissions=frozenset({"read"})
            ),
            TokenConfig.from_plaintext(
                "beta", name="b", scopes=("*",), permissions=frozenset({"write"})
            ),
        ]
    )
    calls: list[tuple[str, str]] = []
    real = auth_module.hmac.compare_digest

    def spy(a: str, b: str) -> bool:
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(auth_module.hmac, "compare_digest", spy)
    assert registry.authenticate("alpha") is not None
    assert registry.authenticate("beta") is not None
    assert registry.authenticate("gamma") is None
    assert registry.authenticate("") is None
    # every stored digest is compared on every attempt, whichever matched (no early exit)
    assert [len(calls[i : i + 2]) for i in range(0, len(calls), 2)] == [2, 2, 2, 2]
    assert all(len(a) == 64 and len(b) == 64 for a, b in calls)  # digests only, never plaintext
    assert all("alpha" not in a + b for a, b in calls)


def test_a_valid_token_with_no_permissions_is_forbidden_everywhere(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "m.db",
        db_key="k",
        tokens=(
            TokenConfig.from_plaintext("nothing", name="n", scopes=("*",), permissions=frozenset()),
        ),
    )
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": "Bearer nothing"}
        assert client.get("/v1/health", headers=headers).status_code == 200  # any valid token
        assert (
            client.post(
                "/v1/recall", json={"scope": "user:alice", "query": "x"}, headers=headers
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/v1/remember", json={"scope": "user:alice", "statement": "x"}, headers=headers
            ).status_code
            == 403
        )


def test_default_bind_address_is_loopback(tmp_path: Path) -> None:
    settings = load_settings(env={"IIFA_DB_KEY": "k"})
    assert settings.host == "127.0.0.1"
    assert (
        load_settings(env={"IIFA_DB_KEY": "k", "IIFA_HOST": "0.0.0.0"}).host == "0.0.0.0"
    )  # explicit opt-in


def test_scope_bound_token_cannot_widen_its_own_view(client: TestClient) -> None:
    assert (
        client.post(
            "/v1/recall",
            json={"scopes": ["user:alice", "user:carol"], "query": "x"},
            headers=auth("alice"),
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/v1/remember",
            json={"scopes": ["user:alice", "user:carol"], "statement": "x"},
            headers=auth("alice"),
        ).status_code
        == 403
    )


def test_routes_cannot_declare_an_unknown_permission() -> None:
    with pytest.raises(ValueError, match="unknown permission"):
        auth_module.require("root")
