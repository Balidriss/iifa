"""``group_activities`` (SPEC 5.6, 5.5): a model names each occurrence of one
activity with its object; the dream groups them under one generic task node by
``instance_of`` edges, a name made only of the words every occurrence shares.
Medium risk: reviewed by default. A task unit shows the alike activities, the
only ones the operation may name."""

from __future__ import annotations

from typing import Any

import pytest

from iifa.dream.apply import Applier
from iifa.dream.journal import Journal
from iifa.dream.policy import ReviewPolicy
from iifa.dream.units import read_unit
from iifa.scopes import Visibility
from iifa.store.models import Node
from iifa.store.repo import Store
from tests.dream.conftest import SCOPE, new_run, submit

MEDIUM_AUTO = ReviewPolicy(medium="auto")
VIEW = Visibility.of_scope(SCOPE)
PORTRAITS = (
    "painting a portrait of the baker",
    "painting a portrait of the mayor",
    "painting a portrait of the gardener",
)


def _tasks(store: Store, *names: str) -> list[Node]:
    return [store.create_node(scopes=[SCOPE], name=n, kind="task") for n in names]


def _group(name: str, members: list[Node]) -> dict[str, Any]:
    return {"op": "group_activities", "name": name, "members": [m.id for m in members]}


def _instances(store: Store, parent: str) -> set[str]:
    return {
        e.src
        for e in store.edges_of([parent], VIEW)
        if e.dst == parent and e.type == "instance_of" and e.status == "confirmed"
    }


def test_the_occurrences_are_grouped_under_a_new_generic_activity_and_undo_removes_it(
    store: Store, applier: Applier, journal: Journal
) -> None:
    portraits = _tasks(store, *PORTRAITS)
    entry = submit(
        store,
        applier,
        new_run(journal, policy=MEDIUM_AUTO),
        _group("painting a portrait", portraits),
    )
    assert (entry.status, entry.reason) == ("applied", None)
    generic = store.find_node("painting a portrait", VIEW)
    assert generic is not None and (generic.kind, generic.scopes) == ("task", (SCOPE,))
    assert _instances(store, generic.id) == {v.id for v in portraits}
    applier.undo_op(entry.id)
    assert store.find_node("painting a portrait", VIEW) is None
    assert all(store.edges_of([v.id], VIEW) == [] for v in portraits)


def test_grouping_is_reviewed_by_default(store: Store, applier: Applier, journal: Journal) -> None:
    portraits = _tasks(store, *PORTRAITS)
    entry = submit(store, applier, new_run(journal), _group("painting a portrait", portraits))
    assert entry.status == "queued"
    assert store.find_node("painting a portrait", VIEW) is None


def test_a_member_named_like_the_generic_activity_becomes_its_parent(
    store: Store, applier: Applier, journal: Journal
) -> None:
    short, long = _tasks(store, "restoring a backup", "restoring a database backup")
    entry = submit(
        store,
        applier,
        new_run(journal, policy=MEDIUM_AUTO),
        _group("Restoring a backup", [short, long]),
    )
    assert entry.status == "applied"
    assert len(store.nodes_of_kind("task", VIEW)) == 2  # no new node
    assert _instances(store, short.id) == {long.id}


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("painting a mural", "not_shared"),  # "mural" is in no name
        ("cooking", "one_word"),
        ("painting", "one_word"),  # shared by every name, yet only the verb kept
        ("a of the", "one_word"),  # no word that counts
        ("painting a portrаit", "mixed_script"),  # a Cyrillic letter
        ("User", "speaker"),
    ],
)
def test_the_generic_name_is_only_what_the_names_share(
    store: Store, applier: Applier, journal: Journal, name: str, reason: str
) -> None:
    portraits = _tasks(store, *PORTRAITS)
    entry = submit(store, applier, new_run(journal, policy=MEDIUM_AUTO), _group(name, portraits))
    assert (entry.status, entry.reason) == ("dropped", reason)


def test_only_live_task_nodes_of_one_scope_set_are_grouped(
    store: Store, applier: Applier, journal: Journal
) -> None:
    (portrait,) = _tasks(store, PORTRAITS[0])
    person = store.create_node(scopes=[SCOPE], name="painting a portrait Sam", kind="person")
    other = store.create_node(
        scopes=[f"{SCOPE}/project:x"], name="painting a portrait", kind="task"
    )
    run = new_run(journal, policy=MEDIUM_AUTO)
    assert (
        submit(store, applier, run, _group("painting a portrait", [portrait, person])).reason
        == "not_task"
    )
    assert submit(store, applier, run, _group("painting a portrait", [portrait, other])).reason == (
        "scope_mismatch"
    )


def test_a_name_held_by_another_kind_of_node_is_refused(
    store: Store, applier: Applier, journal: Journal
) -> None:
    portraits = _tasks(store, *PORTRAITS)
    store.create_node(scopes=[SCOPE], name="painting a portrait", kind="object")
    entry = submit(
        store,
        applier,
        new_run(journal, policy=MEDIUM_AUTO),
        _group("painting a portrait", portraits),
    )
    assert (entry.status, entry.reason) == ("dropped", "name_taken")


def test_an_occurrence_already_grouped_elsewhere_or_twice_is_refused(
    store: Store, applier: Applier, journal: Journal
) -> None:
    portraits = _tasks(store, *PORTRAITS)
    run = new_run(journal, policy=MEDIUM_AUTO)
    assert submit(store, applier, run, _group("painting a portrait", portraits)).status == "applied"
    again = submit(store, applier, run, _group("painting a portrait", portraits))
    assert (again.status, again.reason) == ("dropped", "duplicate")
    elsewhere = submit(store, applier, run, _group("painting portrait", portraits[:2]))
    assert (elsewhere.status, elsewhere.reason) == ("dropped", "grouped")


def test_a_task_unit_shows_its_alike_activities_and_no_other(store: Store) -> None:
    portraits = _tasks(store, *PORTRAITS)
    (unrelated,) = _tasks(store, "writing a song")
    person = store.create_node(scopes=[SCOPE], name="portrait Sam", kind="person")
    unit = read_unit(store, SCOPE, portraits[0], lambda scopes: None)
    assert {n.id for n in unit.alike} == {portraits[1].id, portraits[2].id}
    assert unrelated.id not in unit.ids and person.id not in unit.ids
    assert {portraits[1].id, portraits[2].id} <= unit.ids  # the answer may name them
    shown = unit.as_prompt()["alike_activities"]
    assert {a["name"] for a in shown} == {PORTRAITS[1], PORTRAITS[2]}
    plain = read_unit(store, SCOPE, person, lambda scopes: None)
    assert plain.alike == () and "alike_activities" not in plain.as_prompt()
