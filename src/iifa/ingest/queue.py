"""The persisted ingestion queue (SPEC 7.2).

Every piece of background work is a row in the encrypted database *before* the
request that created it answers, so a crash or a restart loses nothing. The
worker claims rows one at a time in creation order; a row still ``running``
when the service starts was interrupted and goes back to ``pending``.

A job's payload (a turn's text, the messages dropped by a compaction) is memory
content: it is cleared the moment the job is done and the row is pruned soon
after, so the queue never becomes a second, unindexed copy of what was said
(P5). Failed jobs keep their payload so that a later attempt can process them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from iifa.ids import new_id
from iifa.scopes import validate_scope
from iifa.store.repo import NotFound, Store

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
STATUSES = (PENDING, RUNNING, DONE, FAILED)

KEEP_DONE = 200  # done rows kept for inspection (payload already cleared)


@dataclass(frozen=True)
class Job:
    id: str
    seq: int
    kind: str
    payload: dict[str, Any]
    scopes: tuple[str, ...]
    session_id: str | None
    status: str
    attempts: int
    error: str | None
    created_at: str
    not_before: str | None
    started_at: str | None
    finished_at: str | None


class Queue:
    """Rows of the ``queue`` table, through the store's single connection and lock."""

    def __init__(self, store: Store) -> None:
        self._store = store

    # ------------------------------------------------------------------ producers

    def enqueue(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        scopes: Iterable[str],
        session_id: str | None = None,
    ) -> Job:
        """One insert; this is all a hot-path route does (SPEC 7.4: < 10 ms)."""
        unique = tuple(dict.fromkeys(validate_scope(s) for s in scopes))
        if not unique:
            raise ValueError("a job needs at least one scope")
        if not kind:
            raise ValueError("a job needs a kind")
        job_id = new_id("job")
        store = self._store
        now = store.now()
        with store.lock, store.transaction():
            cursor = store.execute(
                "INSERT INTO queue (id, kind, payload, scopes, session_id, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    kind,
                    json.dumps(dict(payload), ensure_ascii=False),
                    json.dumps(list(unique)),
                    session_id,
                    PENDING,
                    now,
                ),
            )
            seq = int(cursor.lastrowid)
        return Job(
            id=job_id,
            seq=seq,
            kind=kind,
            payload=dict(payload),
            scopes=unique,
            session_id=session_id,
            status=PENDING,
            attempts=0,
            error=None,
            created_at=now,
            not_before=None,
            started_at=None,
            finished_at=None,
        )

    # ------------------------------------------------------------------ the worker

    def claim(self) -> Job | None:
        """The oldest pending job whose retry time has come, marked ``running``."""
        store = self._store
        now = store.now()
        with store.lock, store.transaction():
            row = store.execute(
                "SELECT * FROM queue WHERE status = ? AND (not_before IS NULL OR not_before <= ?)"
                " ORDER BY seq LIMIT 1",
                (PENDING, now),
            ).fetchone()
            if row is None:
                return None
            store.execute(
                "UPDATE queue SET status = ?, started_at = ?, not_before = NULL WHERE id = ?",
                (RUNNING, now, row["id"]),
            )
            return self.get(row["id"])

    def complete(self, job_id: str) -> Job:
        """Done: the payload is cleared at once and old done rows are pruned."""
        store = self._store
        with store.lock, store.transaction():
            self._require(job_id)
            store.execute(
                "UPDATE queue SET status = ?, finished_at = ?, payload = NULL, error = NULL"
                " WHERE id = ?",
                (DONE, store.now(), job_id),
            )
            store.execute(
                "DELETE FROM queue WHERE status = ? AND seq NOT IN"
                " (SELECT seq FROM queue WHERE status = ? ORDER BY seq DESC LIMIT ?)",
                (DONE, DONE, KEEP_DONE),
            )
            return self.get(job_id)

    def fail(self, job_id: str, error: str, *, retry_at: str | None) -> Job:
        """One attempt failed. With ``retry_at`` the job waits for another attempt;
        without it the job is ``failed`` and keeps its payload for an operator."""
        store = self._store
        if len(error) > 120:
            raise ValueError("a job error is a short reason, never content")
        with store.lock, store.transaction():
            self._require(job_id)
            if retry_at is None:
                store.execute(
                    "UPDATE queue SET status = ?, attempts = attempts + 1, error = ?,"
                    " finished_at = ?, started_at = NULL WHERE id = ?",
                    (FAILED, error, store.now(), job_id),
                )
            else:
                store.execute(
                    "UPDATE queue SET status = ?, attempts = attempts + 1, error = ?,"
                    " not_before = ?, started_at = NULL WHERE id = ?",
                    (PENDING, error, retry_at, job_id),
                )
            return self.get(job_id)

    def release(self, job_id: str) -> Job:
        """Back to ``pending`` untouched: the worker stopped (budget, shutdown)
        before the job ran, so this is not an attempt."""
        store = self._store
        with store.lock, store.transaction():
            self._require(job_id)
            store.execute(
                "UPDATE queue SET status = ?, started_at = NULL WHERE id = ? AND status = ?",
                (PENDING, job_id, RUNNING),
            )
            return self.get(job_id)

    def recover(self) -> int:
        """At startup: whatever was ``running`` when the process died runs again."""
        store = self._store
        with store.lock, store.transaction():
            cursor = store.execute(
                "UPDATE queue SET status = ?, started_at = NULL WHERE status = ?",
                (PENDING, RUNNING),
            )
            return int(cursor.rowcount)

    def retry(self, job_id: str) -> Job:
        """An operator's second chance for a ``failed`` job."""
        store = self._store
        with store.lock, store.transaction():
            job = self._require(job_id)
            if job.status != FAILED:
                raise ValueError("only a failed job can be retried")
            store.execute(
                "UPDATE queue SET status = ?, error = NULL, finished_at = NULL WHERE id = ?",
                (PENDING, job_id),
            )
            return self.get(job_id)

    # ------------------------------------------------------------------ reading

    def get(self, job_id: str) -> Job:
        with self._store.lock:
            return self._require(job_id)

    def _require(self, job_id: str) -> Job:
        row = self._store.execute("SELECT * FROM queue WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFound(job_id)
        return _job_from_row(row)

    def jobs(self, *, status: str | None = None, limit: int = 100) -> list[Job]:
        """Jobs in creation order, optionally of one status (inspection and tests)."""
        with self._store.lock:
            if status is None:
                rows = self._store.execute(
                    "SELECT * FROM queue ORDER BY seq LIMIT ?", (int(limit),)
                ).fetchall()
            else:
                rows = self._store.execute(
                    "SELECT * FROM queue WHERE status = ? ORDER BY seq LIMIT ?",
                    (status, int(limit)),
                ).fetchall()
            return [_job_from_row(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Counts per status, the depth (pending + running) and the oldest pending
        item's creation time: what ``/v1/health`` shows so a paused queue never
        fills in silence (P11)."""
        with self._store.lock:
            counts = dict.fromkeys(STATUSES, 0)
            for row in self._store.execute(
                "SELECT status, count(*) AS n FROM queue GROUP BY status"
            ).fetchall():
                counts[row["status"]] = int(row["n"])
            oldest = self._store.execute(
                "SELECT created_at FROM queue WHERE status = ? ORDER BY seq LIMIT 1", (PENDING,)
            ).fetchone()
            return {
                "depth": counts[PENDING] + counts[RUNNING],
                "pending": counts[PENDING],
                "running": counts[RUNNING],
                "failed": counts[FAILED],
                "done": counts[DONE],
                "oldest_pending": oldest["created_at"] if oldest else None,
            }

    def depth(self) -> int:
        return int(self.stats()["depth"])

    def has_open(self, kind: str) -> bool:
        """Is a job of ``kind`` pending or running (coalescing background work)?"""
        with self._store.lock:
            row = self._store.execute(
                "SELECT 1 FROM queue WHERE kind = ? AND status IN (?, ?) LIMIT 1",
                (kind, PENDING, RUNNING),
            ).fetchone()
            return row is not None


def _job_from_row(row: Any) -> Job:
    payload = row["payload"]
    return Job(
        id=row["id"],
        seq=int(row["seq"]),
        kind=row["kind"],
        payload=json.loads(payload) if payload else {},
        scopes=tuple(json.loads(row["scopes"])),
        session_id=row["session_id"],
        status=row["status"],
        attempts=int(row["attempts"]),
        error=row["error"],
        created_at=row["created_at"],
        not_before=row["not_before"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )
