"""P4 - no egress. The guard in conftest must block every non-loopback
connection and name resolution, and must let the loopback fakes through."""

from __future__ import annotations

import socket

import httpx
import pytest

from tests.conftest import NetworkBlocked
from tests.fakes.openai_server import FakeOpenAIServer

pytestmark = pytest.mark.security


def test_p4_connect_to_public_ip_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(NetworkBlocked):
        sock.connect(("93.184.216.34", 443))


def test_p4_connect_ex_to_public_ip_is_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, pytest.raises(NetworkBlocked):
        sock.connect_ex(("10.0.0.1", 80))


def test_p4_name_resolution_is_blocked() -> None:
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("example.com", 443)


def test_p4_httpx_to_the_outside_fails_before_any_packet_leaves() -> None:
    with pytest.raises((NetworkBlocked, httpx.TransportError)):
        httpx.get("https://example.com/", timeout=2)


def test_p4_ipv6_and_localhost_forms_are_local() -> None:
    for host in ("localhost", "127.0.0.1", "::1", "127.0.0.2"):
        socket.getaddrinfo(host, 0)  # must not raise


def test_p4_loopback_fake_is_reachable(fake_openai: FakeOpenAIServer) -> None:
    response = httpx.get(fake_openai.base_url + "/models", timeout=5)
    assert response.status_code == 200
