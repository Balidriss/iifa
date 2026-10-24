"""Build ``tests/fixtures/db/v<N>.db`` for the current schema version.

Run once per released schema version and commit the file::

    python -m tests.fixtures.make_fixture_db
"""

from __future__ import annotations

from pathlib import Path

from iifa.store.migrate import SCHEMA_VERSION
from iifa.store.repo import Store

FIXTURE_DIR = Path(__file__).parent / "db"
FIXTURE_KEY = "fixture-key"


def expected_contents() -> set[str]:
    return {"Karim works on the solar project", "Karim préfère le thé"}


def build(path: Path) -> None:
    if path.exists():
        path.unlink()
    # No background checkpoint thread: switching the journal mode needs one connection.
    with Store(path, FIXTURE_KEY, checkpoint_interval=0) as store:
        node = store.create_node(
            scopes=["user:fixture"], name="Karim", kind="person", aliases=["K"]
        )
        for statement in sorted(expected_contents()):
            store.add_fact(
                scopes=["user:fixture"], node_id=node.id, statement=statement, origin="explicit"
            )
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        store._conn.execute("PRAGMA journal_mode = DELETE")


if __name__ == "__main__":
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    target = FIXTURE_DIR / f"v{SCHEMA_VERSION}.db"
    build(target)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
