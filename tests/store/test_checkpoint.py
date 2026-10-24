"""WAL checkpoints run off the write path (the fix behind the 10k-write tail seen
on Windows): the writer's own automatic checkpoint is pushed far away, a
background thread moves frames instead, and ``forget`` still truncates the WAL."""

from __future__ import annotations

import os
from pathlib import Path

from iifa.store.checkpoint import Checkpointer
from iifa.store.connection import BACKSTOP_AUTOCHECKPOINT_PAGES
from iifa.store.repo import Store
from tests.fixtures.corpus import CANARY


def _wal(store: Store) -> Path:
    return Path(store.path + "-wal")


def test_writer_connection_only_checkpoints_as_a_backstop(store: Store) -> None:
    pages = store._conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
    assert pages == BACKSTOP_AUTOCHECKPOINT_PAGES > 1000  # SQLite's default is 1000
    assert isinstance(store.checkpointer, Checkpointer)
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0


def test_background_thread_moves_frames_and_the_next_write_resets_the_wal(tmp_path: Path) -> None:
    with Store(tmp_path / "c.db", "k", checkpoint_interval=3600) as store:
        for i in range(50):
            store.add_fact(scopes=["user:a"], statement=f"fact number {i}", origin="explicit")
        assert store.checkpointer is not None
        before = store.checkpointer.stats.frames
        stats = store.checkpointer.run_once()  # what the thread does every interval
        assert stats.runs == 1 and stats.frames > before and stats.busy == 0
        # the WAL is fully checkpointed: the writer restarts it on its next transaction
        wal_before = os.path.getsize(_wal(store))
        store.add_fact(scopes=["user:a"], statement="one more", origin="explicit")
        assert os.path.getsize(_wal(store)) <= wal_before
        assert store.checkpoint()["log_frames"] < 100


def test_zero_interval_disables_the_thread_and_keeps_sqlite_defaults(tmp_path: Path) -> None:
    with Store(tmp_path / "d.db", "k", checkpoint_interval=0) as store:
        assert store.checkpointer is None
        assert store._conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1000


def test_forget_still_truncates_the_wal_with_the_thread_running(tmp_path: Path) -> None:
    with Store(tmp_path / "e.db", "k", plaintext_for_tests=True, checkpoint_interval=0.01) as store:
        fact = store.add_fact(scopes=["user:a"], statement=f"secret {CANARY}", origin="explicit")
        for i in range(20):
            store.add_fact(scopes=["user:a"], statement=f"filler {i}", origin="explicit")
        store.forget(fact.id)
        assert os.path.getsize(_wal(store)) == 0
        assert CANARY.encode() not in Path(store.path).read_bytes()


def test_manual_checkpoint_modes(store: Store) -> None:
    store.add_fact(scopes=["user:a"], statement="x", origin="explicit")
    assert set(store.checkpoint("PASSIVE")) == {"busy", "log_frames", "checkpointed"}
    assert store.checkpoint("TRUNCATE")["log_frames"] == 0
    try:
        store.checkpoint("NOPE")
    except ValueError as exc:
        assert "NOPE" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown mode must be refused")
