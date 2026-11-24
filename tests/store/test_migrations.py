"""Versioned migrations: a fixture database per released schema version is
migrated forward and read back."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from iifa.scopes import Visibility
from iifa.store.migrate import SCHEMA_VERSION, current_version, migrate
from iifa.store.repo import Store
from tests.fixtures.make_fixture_db import FIXTURE_DIR, FIXTURE_KEY, expected_contents

FIXTURES = sorted(FIXTURE_DIR.glob("v*.db"), key=lambda p: int(p.stem[1:]))  # v10 after v9


def test_fresh_database_is_at_the_current_version(tmp_path: Path) -> None:
    with Store(tmp_path / "m.db", "k") as store:
        assert current_version(store._conn) == SCHEMA_VERSION
        assert migrate(store._conn) == SCHEMA_VERSION  # idempotent


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    with Store(tmp_path / "m.db", "k") as store:
        store._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer"):
        Store(tmp_path / "m.db", "k")


def test_there_is_a_fixture_for_every_released_version() -> None:
    assert [int(p.stem[1:]) for p in FIXTURES] == list(range(1, SCHEMA_VERSION + 1))


@pytest.mark.parametrize("fixture", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_fixture_migrates_forward_and_keeps_its_contents(fixture: Path, tmp_path: Path) -> None:
    copy = tmp_path / fixture.name
    shutil.copy(fixture, copy)
    with Store(copy, FIXTURE_KEY) as store:
        assert current_version(store._conn) == SCHEMA_VERSION
        view = Visibility.of_scope("user:fixture")
        node = store.find_node("Karim", view)
        assert node is not None
        statements = {f.statement for f in store.facts([node.id], view)}
        assert statements == expected_contents()


def test_migration_11_drops_summaries_from_the_lexical_index(tmp_path: Path) -> None:
    """A store indexed before version 11 (a node's document held its
    summary) is reindexed once at open: names and aliases only, lengths kept."""
    view = Visibility.of_scope("user:fixture")
    path = tmp_path / "m.db"
    with Store(path, "k") as store:
        node = store.create_node(scopes=["user:fixture"], name="Karim", aliases=["Kar"])
        store.update_node(node.id, summary="An electrician in Lyon")
        indexed = store.corpus_stats(view)
        # the document as versions 1-10 wrote it
        store._unindex(node.id)
        store._index(node.id, "node", "Karim Kar An electrician in Lyon", ["user:fixture"])
        store._conn.execute("ALTER TABLE facts DROP COLUMN value_norm")  # version 13's
        store._conn.execute("DROP INDEX dream_runs_created")  # version 16's
        store._conn.execute("ALTER TABLE dream_runs DROP COLUMN started_by")
        for trigger in ("edges_rev", "record_scopes_rev_insert", "record_scopes_rev_delete"):
            store._conn.execute(f"DROP TRIGGER {trigger}")  # replayed by version 16
        store._conn.execute("ALTER TABLE edges DROP COLUMN decision_rev")
        for table in ("facts", "session_facts"):  # version 17's
            for column in ("part_kind", "part_ref", "read_at"):
                store._conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        store._conn.execute("ALTER TABLE facts DROP COLUMN observed_at")  # version 18's
        store._conn.execute("ALTER TABLE sources DROP COLUMN external")  # version 19's
        for table in ("fact_activities", "session_activities"):  # version 19's too
            store._conn.execute(f"DROP TABLE {table}")
        store._conn.execute("PRAGMA user_version = 10")
        assert [p.doc_id for p in store.postings(["electrician"], view)] == [node.id]
    with Store(path, "k") as store:
        assert current_version(store._conn) == SCHEMA_VERSION
        assert store.postings(["electrician"], view) == []
        assert [p.doc_id for p in store.postings(["kar"], view)] == [node.id]
        assert store.corpus_stats(view) == indexed
        assert store.get_state("reindex_nodes") is None  # once
