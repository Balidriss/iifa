"""Versioned migrations: numbered SQL files applied forward, tracked in ``user_version``."""

from __future__ import annotations

import re
from importlib import resources
from typing import Any

_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def _migrations() -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for entry in resources.files("iifa.store.migrations").iterdir():
        match = _NAME.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.read_text(encoding="utf-8")))
    found.sort()
    for expected, (number, _) in enumerate(found, start=1):
        if number != expected:
            raise RuntimeError(
                f"migration numbering gap: expected {expected:04d}, found {number:04d}"
            )
    return found


SCHEMA_VERSION: int = len(_migrations())


def current_version(conn: Any) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: Any) -> int:
    """Apply every migration newer than the file's version. Returns the new version."""
    version = current_version(conn)
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {version} is newer than this iifa ({SCHEMA_VERSION})")
    for number, sql in _migrations():
        if number <= version:
            continue
        # executescript commits any open transaction first, so the transaction
        # lives inside the script: the migration and its version stamp are atomic.
        try:
            conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version = {number};\nCOMMIT;")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        version = number
    return version
