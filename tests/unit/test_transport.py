"""One classification of transport failures for the model client, the SDK and
the Claude Code hook: a timeout during the connect phase is ``unreachable``,
whatever the platform does with a closed port. The slow connect is simulated
at the socket, deterministically, on every OS."""

from __future__ import annotations

import importlib.util
import socket
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from iifa import transport
from iifa.client import IifaClient, IifaUnavailable
from iifa.client._base import Call, unavailable
from iifa.llm import CHAT, Endpoint, ModelClient, ModelError

HOOK = Path(__file__).resolve().parents[2] / "adapters" / "claude-code" / "hooks" / "iifa_hook.py"


@pytest.fixture(scope="module")
def hook() -> Any:
    spec = importlib.util.spec_from_file_location("iifa_hook_transport", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- the two facts


TRUTH_TABLE = [
    (False, False, "unreachable"),  # refused, reset while connecting, no route
    (False, True, "unreachable"),  # the connect itself timed out (Windows on a closed port)
    (True, True, "timeout"),  # connected, no answer in time
    (True, False, "broken"),  # connected, then reset or a reply that is not HTTP
]


@pytest.mark.parametrize(("connected", "timed_out", "expected"), TRUTH_TABLE)
def test_classify(connected: bool, timed_out: bool, expected: str) -> None:
    assert transport.classify(connected=connected, timed_out=timed_out) == expected
    assert expected in transport.REASONS


@pytest.mark.parametrize(("connected", "timed_out", "expected"), TRUTH_TABLE)
def test_the_hook_s_copy_agrees_with_iifa(
    hook: Any, connected: bool, timed_out: bool, expected: str
) -> None:
    assert hook.classify(connected=connected, timed_out=timed_out) == expected
    assert hook.classify(connected=connected, timed_out=timed_out) == transport.classify(
        connected=connected, timed_out=timed_out
    )
    assert (hook.UNREACHABLE, hook.TIMEOUT, hook.BROKEN) == transport.REASONS


# --------------------------------------------------------------------------- httpx exceptions


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectError("refused"), "unreachable"),
        (httpx.ConnectTimeout("connect"), "unreachable"),  # Windows: a closed port looks like this
        (httpx.ProxyError("proxy"), "unreachable"),
        (httpx.UnsupportedProtocol("scheme"), "unreachable"),
        (httpx.ReadTimeout("read"), "timeout"),
        (httpx.WriteTimeout("write"), "timeout"),
        (httpx.PoolTimeout("pool"), "timeout"),
        (httpx.ReadError("reset"), "broken"),
        (httpx.WriteError("reset"), "broken"),
        (httpx.RemoteProtocolError("not http"), "broken"),
        (httpx.LocalProtocolError("bad request"), "broken"),
        (httpx.CloseError("close"), "broken"),
    ],
)
def test_httpx_errors_map_to_the_three_words(exc: httpx.TransportError, expected: str) -> None:
    assert transport.classify_transport_error(exc) == expected
    error = unavailable(exc, Call("GET", "/v1/health"))
    assert isinstance(error, IifaUnavailable) and error.reason == expected
    assert f"service {expected}" in str(error) and type(exc).__name__ in str(error)


def test_the_hook_maps_the_phases_the_same_way(hook: Any) -> None:
    """The hook sees phases, not httpx classes: connect vs after; the result matches."""
    for exc in (ConnectionRefusedError(), TimeoutError("connect"), OSError("no route")):
        assert hook.classify(connected=False, timed_out=isinstance(exc, TimeoutError)) == (
            transport.classify_transport_error(httpx.ConnectTimeout("x"))
            if isinstance(exc, TimeoutError)
            else transport.classify_transport_error(httpx.ConnectError("x"))
        )
    assert hook.classify(connected=True, timed_out=True) == transport.classify_transport_error(
        httpx.ReadTimeout("x")
    )
    assert hook.classify(connected=True, timed_out=False) == transport.classify_transport_error(
        httpx.RemoteProtocolError("x")
    )


# --------------------------------------------------------------------------- a slow connect, simulated


@pytest.fixture
def slow_connect(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every TCP connect exceeds its timeout, as Windows does for a closed
    loopback port for about two seconds: deterministic, no port involved."""
    attempts: list[Any] = []

    def connect(self: socket.socket, address: Any) -> None:
        attempts.append(address)
        raise TimeoutError("simulated: the connect exceeded its timeout")

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def test_a_slow_connect_is_unreachable_for_the_model_client(slow_connect: list[Any]) -> None:
    with (
        ModelClient(Endpoint(CHAT, "http://127.0.0.1:1/v1", "m", timeout=0.2)) as client,
        pytest.raises(ModelError) as failed,
    ):
        client.chat([{"role": "user", "content": "x"}], model="m")
    assert failed.value.reason == "unreachable" and failed.value.retryable
    assert client.status()[CHAT]["last_error"] == "unreachable"
    assert slow_connect and slow_connect[0][0] == "127.0.0.1"


def test_a_slow_connect_is_unreachable_for_the_sdk(slow_connect: list[Any]) -> None:
    with (
        IifaClient("http://127.0.0.1:1", "t", timeout=0.2) as memory,
        pytest.raises(IifaUnavailable) as failed,
    ):
        memory.health()
    assert failed.value.reason == "unreachable" and "ConnectTimeout" in str(failed.value)
    assert slow_connect


def test_a_refused_connect_is_unreachable_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def connect(self: socket.socket, address: Any) -> None:
        raise ConnectionRefusedError("simulated: nothing listens")

    monkeypatch.setattr(socket.socket, "connect", connect)
    with (
        ModelClient(Endpoint(CHAT, "http://127.0.0.1:1/v1", "m", timeout=0.2)) as client,
        pytest.raises(ModelError) as failed,
    ):
        client.chat([{"role": "user", "content": "x"}], model="m")
    assert failed.value.reason == "unreachable"
    with (
        IifaClient("http://127.0.0.1:1", "t", timeout=0.2) as memory,
        pytest.raises(IifaUnavailable) as unreachable,
    ):
        memory.health()
    assert unreachable.value.reason == "unreachable" and "ConnectError" in str(unreachable.value)


def test_the_hook_reports_a_slow_connect_as_unreachable(
    hook: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook's own HTTP client, with the real http.client stack: the connect
    raises a timeout, as it does on Windows past the hook's deadline."""

    def connect(self: socket.socket, address: Any) -> None:
        raise TimeoutError("simulated: the connect exceeded its timeout")

    monkeypatch.setattr(socket.socket, "connect", connect)
    settings = hook.Settings({"IIFA_URL": "http://127.0.0.1:1", "IIFA_TOKEN": "t"})
    with pytest.raises(hook.Unavailable) as failed:
        hook.call(settings, "GET", "/v1/health", None, time.monotonic() + 1)
    assert str(failed.value) == "unreachable"
