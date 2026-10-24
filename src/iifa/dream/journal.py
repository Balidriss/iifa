"""The dream's runs and their journal (SPEC 5.6 rule 4), in the encrypted file.

A run is a row with its scope, status, budget, policy, the state it resumes
from and, once finished, its report (counts and ids only). Every operation it
handled is a journal entry: what the model proposed (``payload``) and why
(``reason_given``), what became of it (``status``, ``reason``), the records it
touches, the revisions it read (``snapshot``), how to revert it (``changes``)
and the revisions right after it (``after_revs``). Entries hold content, so
``Store.forget`` wipes the ones that mention a forgotten record (P5).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from iifa.dream.catalogue import OPERATIONS
from iifa.dream.policy import ReviewPolicy
from iifa.ids import new_id
from iifa.store.repo import NotFound, Store

PENDING, RUNNING, FINISHED, CANCELLED, ROLLED_BACK, FAILED = (
    "pending",
    "running",
    "finished",
    "cancelled",
    "rolled_back",
    "failed",
)
ACTIVE = (PENDING, RUNNING)
# the status of a journal entry
PLANNED, APPLIED, QUEUED, REJECTED, DROPPED, UNDONE = (
    "planned",
    "applied",
    "queued",
    "rejected",
    "dropped",
    "undone",
)


@dataclass(frozen=True)
class Run:
    id: str
    scope: str
    status: str
    dry_run: bool
    operations: tuple[str, ...]
    budget: dict[str, Any]
    policy: ReviewPolicy
    cancel_requested: bool
    job_id: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    probes_before: list[dict[str, Any]] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] | None = None


@dataclass(frozen=True)
class Adjudication:
    """What the model said about a pair (SPEC 5.6), kept for the reviewer."""

    id: str
    run_id: str
    a: str
    b: str
    verdict: str
    reason: str | None
    fingerprint: str
    proposal_id: str | None
    op_id: str | None
    created_at: str


def pair_key(a: str, b: str) -> tuple[str, str]:
    """A pair is unordered; it is stored and compared in id order."""
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True)
class Entry:
    id: str
    run_id: str
    seq: int
    op: str
    risk: str
    route: str | None
    status: str
    reason: str | None
    payload: dict[str, Any] | None
    reason_given: str | None
    snapshot: dict[str, int]
    changes: dict[str, Any] | None
    after_revs: dict[str, int]
    records: tuple[str, ...]
    forgotten: bool
    created_at: str
    decided_at: str | None
    undone_at: str | None


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class Journal:
    """Rows of ``dream_runs``, ``dream_ops`` and ``dream_cursors``, through the
    store's connection and lock (one writer)."""

    def __init__(self, store: Store) -> None:
        self.store = store

    # ------------------------------------------------------------------ runs

    def create_run(
        self,
        *,
        scope: str,
        dry_run: bool,
        operations: Sequence[str] | None,
        budget: Mapping[str, Any],
        policy: ReviewPolicy,
    ) -> Run:
        run_id = new_id("dream")
        chosen = [o for o in OPERATIONS if operations is None or o in operations]
        store = self.store
        with store.lock, store.transaction():
            store.execute(
                "INSERT INTO dream_runs (id, scope, status, dry_run, operations, budget, policy,"
                " created_at, state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    scope,
                    PENDING,
                    int(dry_run),
                    _dumps(chosen),
                    _dumps(dict(budget)),
                    _dumps(policy.as_dict()),
                    store.now(),
                    _dumps({}),
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> Run:
        with self.store.lock:
            row = self.store.execute("SELECT * FROM dream_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFound(run_id)
        return _run_from_row(row)

    def active_runs(self) -> list[Run]:
        with self.store.lock:
            rows = self.store.execute(
                "SELECT * FROM dream_runs WHERE status IN (?, ?) ORDER BY id", ACTIVE
            ).fetchall()
        return [_run_from_row(r) for r in rows]

    def set_job(self, run_id: str, job_id: str) -> None:
        self._update_run(run_id, job_id=job_id)

    def start_run(self, run_id: str) -> Run:
        self._update_run(run_id, status=RUNNING, started_at=self.store.now())
        return self.get_run(run_id)

    def save_state(self, run_id: str, state: Mapping[str, Any]) -> None:
        self._update_run(run_id, state=_dumps(dict(state)))

    def save_probes_before(self, run_id: str, scores: list[dict[str, Any]]) -> None:
        self._update_run(run_id, probes_before=_dumps(scores))

    def finish_run(self, run_id: str, status: str, report: Mapping[str, Any]) -> Run:
        self._update_run(
            run_id, status=status, finished_at=self.store.now(), report=_dumps(dict(report))
        )
        return self.get_run(run_id)

    def request_cancel(self, run_id: str) -> Run:
        """A pending run is cancelled at once; a running one stops at its next unit."""
        store = self.store
        with store.lock, store.transaction():
            run = self.get_run(run_id)
            if run.status == PENDING:
                report = {"stopped_by": "cancelled", "units": 0}
                store.execute(
                    "UPDATE dream_runs SET status = ?, cancel_requested = 1, finished_at = ?,"
                    " report = ? WHERE id = ?",
                    (CANCELLED, store.now(), _dumps(report), run_id),
                )
            elif run.status == RUNNING:
                store.execute("UPDATE dream_runs SET cancel_requested = 1 WHERE id = ?", (run_id,))
            return self.get_run(run_id)

    def _update_run(self, run_id: str, **columns: Any) -> None:
        store = self.store
        assignments = ", ".join(f"{name} = ?" for name in columns)
        with store.lock, store.transaction():
            cursor = store.execute(
                f"UPDATE dream_runs SET {assignments} WHERE id = ?",  # noqa: S608 - fixed names
                (*columns.values(), run_id),
            )
            if cursor.rowcount == 0:
                raise NotFound(run_id)

    # ------------------------------------------------------------------ cursors

    def cursor(self, scope: str) -> str:
        """Where the next run of ``scope`` starts: after this node id ('' = the start)."""
        with self.store.lock:
            row = self.store.execute(
                "SELECT after FROM dream_cursors WHERE scope = ?", (scope,)
            ).fetchone()
        return "" if row is None else str(row["after"])

    def set_cursor(self, scope: str, after: str) -> None:
        store = self.store
        with store.lock, store.transaction():
            store.execute(
                "INSERT INTO dream_cursors (scope, after, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT (scope) DO UPDATE SET after = excluded.after,"
                " updated_at = excluded.updated_at",
                (scope, after, store.now()),
            )

    # ------------------------------------------------------------------ entries

    def add_op(
        self,
        run_id: str,
        *,
        op: str,
        risk: str,
        route: str | None,
        status: str,
        reason: str | None,
        payload: Mapping[str, Any] | None,
        reason_given: str | None,
        snapshot: Mapping[str, int],
        changes: Mapping[str, Any] | None,
        after_revs: Mapping[str, int],
        records: Iterable[str],
    ) -> Entry:
        store = self.store
        op_id = new_id("dop")
        with store.lock, store.transaction():
            row = store.execute(
                "SELECT coalesce(max(seq), 0) + 1 AS next FROM dream_ops WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            store.execute(
                "INSERT INTO dream_ops (id, run_id, seq, op, risk, route, status, reason, payload,"
                " reason_given, snapshot, changes, after_revs, created_at, decided_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    op_id,
                    run_id,
                    int(row["next"]),
                    op,
                    risk,
                    route,
                    status,
                    reason,
                    _dumps(dict(payload) if payload is not None else None),
                    reason_given or None,
                    _dumps(dict(snapshot)),
                    _dumps(dict(changes) if changes is not None else None),
                    _dumps(dict(after_revs)),
                    store.now(),
                    store.now() if status == APPLIED else None,
                ),
            )
            for record_id in dict.fromkeys(records):
                store.execute(
                    "INSERT OR IGNORE INTO dream_op_records (op_id, record_id) VALUES (?, ?)",
                    (op_id, record_id),
                )
        return self.get_op(op_id)

    def update_op(self, op_id: str, **columns: Any) -> Entry:
        store = self.store
        values = {
            name: _dumps(value) if name in ("changes", "after_revs", "snapshot") else value
            for name, value in columns.items()
        }
        assignments = ", ".join(f"{name} = ?" for name in values)
        with store.lock, store.transaction():
            cursor = store.execute(
                f"UPDATE dream_ops SET {assignments} WHERE id = ?",  # noqa: S608 - fixed names
                (*values.values(), op_id),
            )
            if cursor.rowcount == 0:
                raise NotFound(op_id)
        return self.get_op(op_id)

    def add_records(self, op_id: str, records: Iterable[str]) -> None:
        store = self.store
        with store.lock, store.transaction():
            for record_id in dict.fromkeys(records):
                store.execute(
                    "INSERT OR IGNORE INTO dream_op_records (op_id, record_id) VALUES (?, ?)",
                    (op_id, record_id),
                )

    def get_op(self, op_id: str) -> Entry:
        with self.store.lock:
            row = self.store.execute("SELECT * FROM dream_ops WHERE id = ?", (op_id,)).fetchone()
            if row is None:
                raise NotFound(op_id)
            return self._entry(row)

    def ops_of_run(self, run_id: str, *, after: int = 0, limit: int | None = None) -> list[Entry]:
        with self.store.lock:
            sql = "SELECT * FROM dream_ops WHERE run_id = ? AND seq > ? ORDER BY seq"
            params: tuple[Any, ...] = (run_id, after)
            if limit is not None:
                sql += " LIMIT ?"
                params = (*params, int(limit))
            rows = self.store.execute(sql, params).fetchall()
            return [self._entry(r) for r in rows]

    def queued(self) -> list[Entry]:
        """Every queued operation, oldest first (the review queue filters by view)."""
        with self.store.lock:
            rows = self.store.execute(
                "SELECT * FROM dream_ops WHERE status = ? ORDER BY created_at, id", (QUEUED,)
            ).fetchall()
            return [self._entry(r) for r in rows]

    def applied_touching(self, record_id: str) -> list[Entry]:
        """Applied entries that recorded a revision for ``record_id``."""
        with self.store.lock:
            rows = self.store.execute(
                "SELECT o.* FROM dream_ops o JOIN dream_op_records r ON r.op_id = o.id"
                " WHERE r.record_id = ? AND o.status = ? ORDER BY o.created_at, o.id",
                (record_id, APPLIED),
            ).fetchall()
            return [self._entry(r) for r in rows]

    def merge_ops(self, a: str, b: str, *, status: str) -> list[Entry]:
        """The ``merge_nodes`` entries of a pair (either way round) with ``status``."""
        pair = set(pair_key(a, b))
        with self.store.lock:
            rows = self.store.execute(
                "SELECT o.* FROM dream_ops o JOIN dream_op_records r ON r.op_id = o.id"
                " WHERE r.record_id = ? AND o.op = 'merge_nodes' AND o.status = ?"
                " ORDER BY o.created_at, o.id",
                (min(pair), status),
            ).fetchall()
            entries = [self._entry(r) for r in rows]
        return [
            e
            for e in entries
            if e.payload is not None and {e.payload.get("keep"), e.payload.get("absorb")} == pair
        ]

    # ------------------------------------------------------------------ adjudications

    def add_adjudication(
        self,
        run_id: str,
        *,
        a: str,
        b: str,
        verdict: str,
        reason: str,
        fingerprint: str,
        records: Iterable[str],
        proposal_id: str | None = None,
        op_id: str | None = None,
    ) -> Adjudication:
        first, second = pair_key(a, b)
        adjudication_id = new_id("dadj")
        store = self.store
        with store.lock, store.transaction():
            store.execute(
                "INSERT INTO dream_adjudications (id, run_id, a, b, verdict, reason, fingerprint,"
                " proposal_id, op_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    adjudication_id,
                    run_id,
                    first,
                    second,
                    verdict,
                    reason or None,
                    fingerprint,
                    proposal_id,
                    op_id,
                    store.now(),
                ),
            )
            for record_id in dict.fromkeys(records):
                store.execute(
                    "INSERT OR IGNORE INTO dream_adjudication_records (adjudication_id, record_id)"
                    " VALUES (?, ?)",
                    (adjudication_id, record_id),
                )
        return self._adjudication(adjudication_id)

    def latest_adjudication(self, a: str, b: str) -> Adjudication | None:
        first, second = pair_key(a, b)
        with self.store.lock:
            row = self.store.execute(
                "SELECT * FROM dream_adjudications WHERE a = ? AND b = ?"
                " ORDER BY created_at DESC, id DESC LIMIT 1",
                (first, second),
            ).fetchone()
        return None if row is None else _adjudication_from_row(row)

    def proposals_created(self, run_id: str) -> list[str]:
        """The merge proposals the run's ``unsure`` verdicts created."""
        with self.store.lock:
            rows = self.store.execute(
                "SELECT proposal_id FROM dream_adjudications"
                " WHERE run_id = ? AND proposal_id IS NOT NULL ORDER BY created_at, id",
                (run_id,),
            ).fetchall()
        return [r["proposal_id"] for r in rows]

    def _adjudication(self, adjudication_id: str) -> Adjudication:
        with self.store.lock:
            row = self.store.execute(
                "SELECT * FROM dream_adjudications WHERE id = ?", (adjudication_id,)
            ).fetchone()
        if row is None:
            raise NotFound(adjudication_id)
        return _adjudication_from_row(row)

    def _entry(self, row: Any) -> Entry:
        records = tuple(
            r["record_id"]
            for r in self.store.execute(
                "SELECT record_id FROM dream_op_records WHERE op_id = ? ORDER BY record_id",
                (row["id"],),
            )
        )
        return Entry(
            id=row["id"],
            run_id=row["run_id"],
            seq=int(row["seq"]),
            op=row["op"],
            risk=row["risk"],
            route=row["route"],
            status=row["status"],
            reason=row["reason"],
            payload=_loads(row["payload"]),
            reason_given=row["reason_given"],
            snapshot=_loads(row["snapshot"]) or {},
            changes=_loads(row["changes"]),
            after_revs=_loads(row["after_revs"]) or {},
            records=records,
            forgotten=bool(row["forgotten"]),
            created_at=row["created_at"],
            decided_at=row["decided_at"],
            undone_at=row["undone_at"],
        )


def _adjudication_from_row(row: Any) -> Adjudication:
    return Adjudication(
        id=row["id"],
        run_id=row["run_id"],
        a=row["a"],
        b=row["b"],
        verdict=row["verdict"],
        reason=row["reason"],
        fingerprint=row["fingerprint"],
        proposal_id=row["proposal_id"],
        op_id=row["op_id"],
        created_at=row["created_at"],
    )


def _run_from_row(row: Any) -> Run:
    return Run(
        id=row["id"],
        scope=row["scope"],
        status=row["status"],
        dry_run=bool(row["dry_run"]),
        operations=tuple(_loads(row["operations"]) or ()),
        budget=_loads(row["budget"]) or {},
        policy=ReviewPolicy.from_dict(_loads(row["policy"]) or {}),
        cancel_requested=bool(row["cancel_requested"]),
        job_id=row["job_id"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        probes_before=_loads(row["probes_before"]),
        state=_loads(row["state"]) or {},
        report=_loads(row["report"]),
    )


__all__ = [
    "ACTIVE",
    "APPLIED",
    "CANCELLED",
    "DROPPED",
    "FAILED",
    "FINISHED",
    "PENDING",
    "PLANNED",
    "QUEUED",
    "REJECTED",
    "ROLLED_BACK",
    "RUNNING",
    "UNDONE",
    "Adjudication",
    "Entry",
    "Journal",
    "Run",
    "pair_key",
]
