"""Identifiers: a type prefix plus a ULID, e.g. ``fact_01J8Z3M4Q5R6S7T8V9W0X1Y2Z3``.

ULIDs sort by creation time and need no coordination. Within one process they
are **monotonic**: an id created in the same millisecond as the previous one
(or after the wall clock stepped back) takes the previous random part plus
one, so ids sort in creation order even when a burst of records is written
inside one clock tick, which extraction does constantly and which Windows,
whose clock ticks every 15 ms, makes routine. Every "newest first" query that
breaks ties on the id relies on it. Only the standard library is used
(``os.urandom`` for the 80 random bits, a lock around the shared state).
"""

from __future__ import annotations

import os
import threading
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_LENGTH = 26
_TIME_CHARS = 10
_RANDOM_CHARS = 16
_RANDOM_BITS = 80
_MAX_RANDOM = (1 << _RANDOM_BITS) - 1
_WAIT_SECONDS = 0.0002

PREFIXES = frozenset(
    {"node", "fact", "edge", "src", "job", "run", "prop", "pset", "probe", "dream", "dop", "dadj"}
)


class _Sequence:
    """The process-wide monotonic state: the last (timestamp, randomness) handed out."""

    __slots__ = ("lock", "randomness", "timestamp")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.timestamp = -1
        self.randomness = 0


_sequence = _Sequence()


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def _decode(text: str) -> int:
    value = 0
    for char in text:
        value = (value << 5) | _CROCKFORD.index(char)
    return value


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _random_bits() -> int:
    return int.from_bytes(os.urandom(_RANDOM_BITS // 8), "big")


def _next() -> tuple[int, int]:
    """The next (timestamp, randomness) of the process's monotonic sequence."""
    seq = _sequence
    with seq.lock:
        now = _now_ms()
        if now > seq.timestamp:
            seq.timestamp, seq.randomness = now, _random_bits()
        elif seq.randomness < _MAX_RANDOM:  # same millisecond, or a clock that stepped back
            seq.randomness += 1
        else:  # the random part cannot grow any more: wait for the next millisecond
            while now <= seq.timestamp:
                time.sleep(_WAIT_SECONDS)
                now = _now_ms()
            seq.timestamp, seq.randomness = now, _random_bits()
        return seq.timestamp, seq.randomness


def new_ulid(timestamp_ms: int | None = None) -> str:
    """A 26-character ULID (48-bit millisecond timestamp + 80 random bits).

    Without ``timestamp_ms`` it is the next id of the process's monotonic
    sequence: strictly greater than every id created before it in this
    process. With an explicit ``timestamp_ms`` it encodes that time with fresh
    random bits and leaves the sequence alone (an encoding affordance, not a
    creation-ordered id).
    """
    if timestamp_ms is not None:
        return _encode(timestamp_ms, _TIME_CHARS) + _encode(_random_bits(), _RANDOM_CHARS)
    timestamp, randomness = _next()
    return _encode(timestamp, _TIME_CHARS) + _encode(randomness, _RANDOM_CHARS)


def decode_ulid(ulid: str) -> tuple[int, int]:
    """``(timestamp_ms, randomness)`` of a ULID (tests and debugging)."""
    if len(ulid) != _ULID_LENGTH or any(c not in _CROCKFORD for c in ulid):
        raise ValueError("not a ULID")
    return _decode(ulid[:_TIME_CHARS]), _decode(ulid[_TIME_CHARS:])


def new_id(prefix: str) -> str:
    if prefix not in PREFIXES:
        raise ValueError(f"unknown id prefix {prefix!r}")
    return f"{prefix}_{new_ulid()}"


def kind_of(identifier: str) -> str | None:
    """The kind an id refers to (a prefix of ``PREFIXES``), or None for anything else."""
    prefix, _, rest = identifier.partition("_")
    if prefix in PREFIXES and len(rest) == _ULID_LENGTH and all(c in _CROCKFORD for c in rest):
        return prefix
    return None
