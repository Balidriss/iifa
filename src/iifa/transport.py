"""Classifying transport failures, once for the whole codebase.

Three words, shared by the model client (``iifa.llm``), the Python SDK
(``iifa.client``) and the Claude Code hook, which keeps a standard-library copy
of ``classify`` that a test holds identical:

* ``unreachable`` — no connection was made: refused, reset while connecting,
  no route, a proxy or URL problem, or a **timeout during the connect phase**;
* ``timeout`` — the peer accepted the connection and then did not answer in
  time;
* ``broken`` — the peer accepted the connection and the exchange then failed
  for another reason (reset mid-way, a reply that is not HTTP).

The connect-phase timeout is the case that differs across platforms. On
Windows a connection to a closed loopback port is not refused at once: the
stack retries for about two seconds, so a short connect timeout is exactly how
"nothing listens there" shows up. Reporting it as a timeout would make an
adapter believe the service is slow when it is down, and a worker retry a
model endpoint that is not there.
"""

from __future__ import annotations

import httpx

UNREACHABLE = "unreachable"
TIMEOUT = "timeout"
BROKEN = "broken"
REASONS = (UNREACHABLE, TIMEOUT, BROKEN)


def classify(*, connected: bool, timed_out: bool) -> str:
    """The reason for a failed exchange, from two facts: was a connection made,
    and did a timeout end the attempt. Identical to the hook's copy."""
    if not connected:
        return UNREACHABLE
    return TIMEOUT if timed_out else BROKEN


# Raised before any connection exists: the peer was never reached.
_NOT_CONNECTED: tuple[type[httpx.TransportError], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ProxyError,
    httpx.UnsupportedProtocol,
)


def classify_transport_error(exc: httpx.TransportError) -> str:
    """``unreachable``, ``timeout`` or ``broken`` for an httpx transport error."""
    return classify(
        connected=not isinstance(exc, _NOT_CONNECTED),
        timed_out=isinstance(exc, httpx.TimeoutException),
    )


__all__ = ["BROKEN", "REASONS", "TIMEOUT", "UNREACHABLE", "classify", "classify_transport_error"]
