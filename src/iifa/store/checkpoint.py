"""Background WAL checkpoints: the write path never waits for an fsync.

A commit in WAL mode with ``synchronous = NORMAL`` only appends to the WAL. The
expensive part, copying frames into the database file and syncing both files,
is the checkpoint, and SQLite runs it on the *writing* connection once the WAL
passes a threshold, so one write in sixty-five paid for all the others: 8 ms on
Linux, up to a second on Windows. ``Checkpointer`` runs ``PASSIVE`` checkpoints
from its own thread and its own connection every ``interval`` seconds instead.
A passive checkpoint never blocks the writer or readers; the WAL is reset by
the writer's next transaction once it has been fully checkpointed.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from iifa.store.connection import open_database

DEFAULT_INTERVAL = 0.5


@dataclass
class CheckpointStats:
    runs: int = 0
    frames: int = 0
    busy: int = 0
    seconds: float = 0.0
    slowest: float = 0.0


class Checkpointer:
    """Runs ``PRAGMA wal_checkpoint(PASSIVE)`` every ``interval`` seconds on a
    second connection. ``lock`` serialises it with the store's own explicit
    checkpoints (``forget`` truncates the WAL after scrubbing)."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        key: str,
        *,
        interval: float = DEFAULT_INTERVAL,
        lock: threading.Lock,
        plaintext_for_tests: bool = False,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval
        self.stats = CheckpointStats()
        self._lock = lock
        self._conn = open_database(path, key, plaintext_for_tests=plaintext_for_tests)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="iifa-checkpoint", daemon=True)

    def start(self) -> Checkpointer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
        with self._lock:
            self._conn.close()

    def run_once(self) -> CheckpointStats:
        """One passive checkpoint, now (the thread calls this; tests may too)."""
        started = time.perf_counter()
        with self._lock:
            row = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        elapsed = time.perf_counter() - started
        stats = self.stats
        stats.runs += 1
        stats.busy += int(row[0])
        stats.frames += max(0, int(row[2]))
        stats.seconds += elapsed
        stats.slowest = max(stats.slowest, elapsed)
        return stats

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.run_once()
            except Exception:  # a failed checkpoint is retried on the next tick
                if self._stop.is_set():
                    return
