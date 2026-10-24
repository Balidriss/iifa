"""Suite-wide fixtures.

Two rules from SPEC section 13.2 live here and apply to every test:

* **No network.** An autouse fixture blocks every socket connection and every
  name resolution that is not the loopback interface. It also enforces P4: a
  test that phones home fails.
* **Deterministic time.** Tests that care about time use the ``clock`` fixture,
  never ``sleep``.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import Settings
from iifa.store.repo import Store
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import KEY, auth, make_settings

# --------------------------------------------------------------------------- socket guard


class NetworkBlocked(RuntimeError):
    """Raised when a test tries to reach anything but the loopback interface (P4)."""


_LOCAL_NAMES = frozenset({"", "localhost", "127.0.0.1", "::1", "0.0.0.0", "::"})


def _host_is_local(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    bare = host.split("%", 1)[0].strip("[]").lower()
    if bare in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(bare).is_loopback
    except ValueError:
        return False


def _check_address(address: object) -> None:
    if isinstance(address, (str, bytes)):  # AF_UNIX: a path on this machine
        return
    if isinstance(address, tuple) and address:
        if _host_is_local(address[0]):
            return
        raise NetworkBlocked(f"test attempted to connect to {address[0]!r}: no network in tests")
    raise NetworkBlocked(f"test attempted to connect to {address!r}: no network in tests")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block every outbound connection except loopback, at the socket level."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> None:
        _check_address(address)
        real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        _check_address(address)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _host_is_local(host):
            raise NetworkBlocked(f"test attempted to resolve {host!r}: no network in tests")
        return real_getaddrinfo(host, *args, **kwargs)

    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


# --------------------------------------------------------------------------- clock


class FakeClock:
    """An injectable clock. Tests advance it; nothing ever sleeps."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now_utc = start or datetime(2026, 9, 26, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now_utc

    def advance(self, **kwargs: float) -> datetime:
        self.now_utc += timedelta(**kwargs)
        return self.now_utc


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# --------------------------------------------------------------------------- fake model server


@pytest.fixture(scope="session")
def fake_openai_session() -> Iterator[FakeOpenAIServer]:
    """One fake model server per test session (it is cheap, but starting it is not free)."""
    with FakeOpenAIServer() as server:
        yield server


@pytest.fixture
def fake_openai(fake_openai_session: FakeOpenAIServer) -> Iterator[FakeOpenAIServer]:
    """The session fake, reset before and after each test so scripts never leak."""
    fake_openai_session.reset()
    yield fake_openai_session
    fake_openai_session.reset()


# --------------------------------------------------------------------------- store and app


@pytest.fixture
def store(tmp_path: Path, clock: FakeClock) -> Iterator[Store]:
    """A real encrypted store in tmp_path with the injectable clock."""
    with Store(tmp_path / "memory.db", KEY, clock=clock) as store:
        yield store


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """An in-process client holding the all-permissions token; pass ``headers=auth(...)`` to switch."""
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client
