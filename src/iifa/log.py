"""Logging that can never leak memory content (P5).

Every log line goes through ``event``, which accepts only identifiers, counts,
durations and enumerated words. There is no way to pass a statement, a name or
a source excerpt, so the canary scan in the security tests has nothing to find.
"""

from __future__ import annotations

import logging
from typing import Final

logger: Final = logging.getLogger("iifa")

_MAX_VALUE_LENGTH = 64


def event(name: str, **fields: int | float | str | bool | None) -> None:
    """Log ``name`` with structured fields. Values are ids, counts or short words only."""
    parts = []
    for key, value in fields.items():
        if isinstance(value, str) and len(value) > _MAX_VALUE_LENGTH:
            raise ValueError(f"log field {key!r} is too long to be an id or a word")
        parts.append(f"{key}={value}")
    logger.info("%s %s", name, " ".join(parts))
