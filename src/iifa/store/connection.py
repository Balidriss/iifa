"""Opening the SQLCipher database with the operator's key."""

from __future__ import annotations

import os
from typing import Any

from sqlcipher3 import dbapi2 as sqlcipher

Connection = Any  # sqlcipher3's Connection has no typing stubs


# In WAL mode with ``synchronous = NORMAL`` a commit appends to the WAL and never
# fsyncs; the two fsyncs happen at each checkpoint. SQLite checkpoints on the
# writing connection every 1000 pages by default, which at ~15 pages per fact is
# every ~65 writes: about 8 ms each on Linux, hundreds of milliseconds on Windows
# where FlushFileBuffers is slow (the 10k-fact benchmark took 144 s there with a
# per-write p95 of 1.2 ms). The store therefore checkpoints from a background
# thread on its own connection (``iifa.store.checkpoint``) and only keeps the
# automatic checkpoint as a backstop, far enough away that it never fires while
# the thread keeps up.
BACKSTOP_AUTOCHECKPOINT_PAGES = 20000
BUSY_TIMEOUT_MS = 5000


class WrongKeyError(RuntimeError):
    """The file exists but the key does not open it."""


def _quote(key: str) -> str:
    return "'" + key.replace("'", "''") + "'"


def open_database(
    path: str | os.PathLike[str],
    key: str,
    *,
    plaintext_for_tests: bool = False,
    autocheckpoint_pages: int | None = None,
) -> Connection:
    """Open (or create) the database at ``path`` encrypted with ``key``.

    ``plaintext_for_tests`` opens an unencrypted file so a test can scan the
    raw bytes of the file for leftovers after ``forget``. Nothing in the
    service ever sets it. ``autocheckpoint_pages`` overrides SQLite's automatic
    checkpoint threshold (1000 pages) on this connection.
    """
    if not key and not plaintext_for_tests:
        raise ValueError("a database key is required")
    conn = sqlcipher.connect(str(path), check_same_thread=False, isolation_level=None)
    if not plaintext_for_tests:
        conn.execute(f"PRAGMA key = {_quote(key)}")
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlcipher.DatabaseError as exc:
        conn.close()
        raise WrongKeyError(f"cannot open {os.fspath(path)}: wrong key or not a database") from exc
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA secure_delete = ON")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    if autocheckpoint_pages is not None:
        conn.execute(f"PRAGMA wal_autocheckpoint = {int(autocheckpoint_pages)}")
    conn.row_factory = sqlcipher.Row
    return conn
