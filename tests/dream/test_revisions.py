"""Revisions (SPEC 4.4): every change to a node, fact or edge moves its ``rev``,
whoever makes it - content, life, aliases, scope set - and a forgotten record
has none. The dream's optimistic checks (rule 8) and undo (rule 4) rest on it,
so each kind of change gets its assertion."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from iifa.scopes import Visibility
from iifa.store.repo import Store
from tests.dream.conftest import SCOPE, Memory
from tests.fixtures.make_fixture_db import FIXTURE_DIR, FIXTURE_KEY


def _moves(store: Store, record_id: str, change: Callable[[], object]) -> None:
    before = store.revisions([record_id])[record_id]
    change()
    after = store.revisions([record_id])
    assert record_id in after, "the record is gone"
    assert after[record_id] > before, "the change did not move the revision"


def test_new_records_carry_a_revision(store: Store, memory: Memory) -> None:
    revs = store.revisions([memory.karim.id, memory.home.id])
    assert set(revs) == {memory.karim.id, memory.home.id}
    assert store.get_node(memory.karim.id).rev == revs[memory.karim.id]
    assert store.get_fact(memory.home.id).rev == revs[memory.home.id]


def test_unknown_and_non_record_ids_have_no_revision(store: Store, memory: Memory) -> None:
    assert store.revisions(["fact_" + "0" * 26, "nonsense"]) == {}


@pytest.mark.parametrize(
    "change",
    ["summary", "alias_added", "alias_removed", "archive", "restore", "flags", "new_fact"],
)
def test_every_change_to_a_node_moves_its_revision(
    store: Store, memory: Memory, change: str
) -> None:
    node = memory.karim.id
    if change == "restore":
        store.archive(node)
    actions: dict[str, Callable[[], object]] = {
        "summary": lambda: store.update_node(node, summary="Karim, from Lyon."),
        "alias_added": lambda: store.update_node(node, add_aliases=["Kaka"]),
        "alias_removed": lambda: store.remove_aliases(node, ["K."]),
        "archive": lambda: store.archive(node),
        "restore": lambda: store.restore(node),
        "flags": lambda: store.update_node(node, flags=["language_mismatch"]),
        "new_fact": lambda: store.add_fact(
            scopes=[SCOPE], node_id=node, statement="Karim plays chess.", origin="extracted"
        ),
    }
    _moves(store, node, actions[change])


@pytest.mark.parametrize(
    "change", ["close", "flag", "attach", "pin", "archive", "restore", "superseded_by"]
)
def test_every_change_to_a_fact_moves_its_revision(
    store: Store, memory: Memory, change: str
) -> None:
    fact = memory.tea.id
    if change == "restore":
        store.archive(fact)
    actions: dict[str, Callable[[], object]] = {
        "close": lambda: store.close_fact(fact),
        "flag": lambda: store.flag_fact(fact, "conflict", conflicts_with=memory.green_tea.id),
        "attach": lambda: store.attach_fact(fact, memory.kiki.id),
        "pin": lambda: store.attach_fact(fact, memory.karim.id, pinned=True),
        "archive": lambda: store.archive(fact),
        "restore": lambda: store.restore(fact),
        "superseded_by": lambda: store.close_fact(fact, superseded_by=memory.green_tea.id),
    }
    _moves(store, fact, actions[change])


@pytest.mark.parametrize("change", ["reinforce", "confirm", "close", "archive", "evidence"])
def test_every_change_to_an_edge_moves_its_revision(
    store: Store, memory: Memory, change: str
) -> None:
    edge = store.add_edge(
        scopes=[SCOPE],
        src=memory.karim.id,
        dst=memory.lyon.id,
        type="lives_in",
        origin="test",
        status="proposed",
    ).id
    source = store.add_source(scopes=[SCOPE], kind="turn", text="Karim lives in Lyon.")
    actions: dict[str, Callable[[], object]] = {
        "reinforce": lambda: store.reinforce_edge(edge),
        "confirm": lambda: store.confirm_edge(edge),
        "close": lambda: store.close_edge(edge),
        "archive": lambda: store.archive(edge),
        "evidence": lambda: store.add_evidence(edge, source.id),
    }
    _moves(store, edge, actions[change])


def test_decay_moves_the_revision_of_the_edges_it_changes(
    store: Store, memory: Memory, clock: object
) -> None:
    edge = store.add_edge(
        scopes=[SCOPE],
        src=memory.karim.id,
        dst=memory.lyon.id,
        type="knows",
        origin="test",
        status="proposed",
    ).id
    clock.advance(days=10)  # type: ignore[attr-defined]
    _moves(store, edge, lambda: store.decay_edges(half_life_days=30, prune_below=0.01))


def test_a_scope_set_change_moves_the_revision(store: Store, memory: Memory) -> None:
    """No route changes a record's scopes today; the trigger still covers any
    future path, which is the point of keeping revisions in the store."""
    fact = memory.tea.id

    def add_scope() -> None:
        with store.lock:
            store.execute(
                "INSERT INTO record_scopes (record_id, scope) VALUES (?, ?)",
                (fact, "user:alice/project:solar"),
            )

    def drop_scope() -> None:
        with store.lock:
            store.execute(
                "DELETE FROM record_scopes WHERE record_id = ? AND scope = ?",
                (fact, "user:alice/project:solar"),
            )

    _moves(store, fact, add_scope)
    _moves(store, fact, drop_scope)


def test_a_direct_update_cannot_skip_the_revision(store: Store, memory: Memory) -> None:
    """Whatever writes the row - even raw SQL that forgets the column - moves it."""
    fact = memory.tea.id

    def raw() -> None:
        with store.lock:
            store.execute("UPDATE facts SET statement = ? WHERE id = ?", ("Karim likes tea.", fact))

    _moves(store, fact, raw)


def test_a_forgotten_record_has_no_revision(store: Store, memory: Memory) -> None:
    store.forget(memory.tea.id)
    assert store.revisions([memory.tea.id]) == {}


def test_reads_do_not_move_revisions(store: Store, memory: Memory) -> None:
    before = store.revisions([memory.karim.id, memory.home.id])
    store.get_node(memory.karim.id)
    store.get_fact(memory.home.id)
    assert store.revisions([memory.karim.id, memory.home.id]) == before


def test_a_store_migrated_from_before_revisions_keeps_them(tmp_path: Path) -> None:
    """Schema 9 adds revisions to every existing record and the triggers that
    move them, so a store of schema 8 is protected like a new one."""
    copy = tmp_path / "v8.db"
    shutil.copy(FIXTURE_DIR / "v8.db", copy)
    with Store(copy, FIXTURE_KEY) as store:
        node = store.find_node("Karim", Visibility.of_scope("user:fixture"))
        assert node is not None
        before = store.revisions([node.id])[node.id]
        store.update_node(node.id, add_aliases=["Kaka"])
        assert store.revisions([node.id])[node.id] > before
