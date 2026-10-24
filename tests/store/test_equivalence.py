"""Equivalence groups (SPEC 5.4): facts stating one value in other forms or
languages, linked by ``same_as`` to the group's first fact; read through the
view (P9); ``forget`` of the first re-roots the group (P5)."""

from __future__ import annotations

import pytest

from iifa.recall.engine import RecallEngine, preferred
from iifa.scopes import Visibility
from iifa.store.models import Fact
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

ALICE = "user:alice"
VIEW = Visibility.of_scope(ALICE)


def karim(store: Store) -> str:
    return store.create_node(scopes=[ALICE], name="Karim", kind="person").id


def fact(store: Store, node: str, statement: str, value: str, **extra: object) -> str:
    return store.add_fact(
        scopes=extra.pop("scopes", [ALICE]),  # type: ignore[arg-type]
        node_id=node,
        statement=statement,
        origin="extracted",
        attribute="home",
        value=value,
        **extra,  # type: ignore[arg-type]
    ).id


def test_a_fact_records_the_group_it_joins(store: Store) -> None:
    node = karim(store)
    root = fact(store, node, "Karim habite à Lyon.", "Lyon", language="fr")
    member = fact(store, node, "Карим живёт в Лионе.", "Лион", language="ru", same_as=root)
    assert store.get_fact(member).same_as == root and store.get_fact(root).same_as is None
    assert store.fact_roots([root, member]) == {root: root, member: root}
    groups = store.equivalents([root], VIEW)
    assert sorted(f.id for f in groups[root]) == sorted([root, member])


def test_a_group_is_read_through_the_view(store: Store) -> None:
    """P9: a member the view cannot see, an archived or a closed one is not offered."""
    node = karim(store)
    root = fact(store, node, "Karim habite à Lyon.", "Lyon")
    hidden = fact(store, node, "Karim lives in Lyon.", "Lyon", same_as=root, scopes=["user:bob"])
    archived = fact(store, node, "Карим живёт в Лионе.", "Лион", same_as=root)
    closed = fact(store, node, "كريم يسكن في ليون.", "ليون", same_as=root)
    store.archive(archived)
    store.close_fact(closed)
    assert [f.id for f in store.equivalents([root], VIEW)[root]] == [root]
    assert hidden not in {f.id for f in store.equivalents([root], VIEW)[root]}


@pytest.mark.security
def test_p5_forgetting_the_first_fact_re_roots_the_group(store: Store) -> None:
    node = karim(store)
    root = fact(store, node, "Karim habite à Lyon.", "Lyon")
    second = fact(store, node, "Karim lives in Lyon.", "Lyon", same_as=root)
    third = fact(store, node, "Карим живёт в Лионе.", "Лион", same_as=root)
    store.forget(root)
    assert store.get_fact(second).same_as is None  # the next one leads the group now
    assert store.get_fact(third).same_as == second
    rows = store.execute("SELECT count(*) FROM facts WHERE same_as = ?", (root,)).fetchone()
    assert rows[0] == 0  # no reference to a forgotten fact remains
    store.forget(third)
    assert store.equivalents([second], VIEW)[second][0].id == second


# --------------------------------------------------------------------------- recall: one per group


def engine(store: Store) -> RecallEngine:
    return RecallEngine(store, ConservativeTokenizer(), tier=2)


def lyon_group(store: Store) -> tuple[str, str]:
    node = karim(store)
    root = fact(store, node, "Karim habite à Lyon avec toute sa famille.", "Lyon", language="fr")
    member = fact(
        store,
        node,
        "Карим живёт в Лионе со всей своей семьёй.",
        "Лион",
        language="ru",
        same_as=root,
    )
    return root, member


def fact_items(result: dict[str, object]) -> list[str]:
    return [i["id"] for i in result["items"] if i["kind"] == "fact"]  # type: ignore[attr-defined,index]


def test_recall_injects_one_fact_per_group_in_the_querys_language(store: Store) -> None:
    root, member = lyon_group(store)
    french = engine(store).recall(
        "Où habite Karim avec sa famille ?", view=VIEW, budget_tokens=2000
    )
    assert fact_items(french) == [root]
    russian = engine(store).recall(
        "Где живёт Карим со своей семьёй?", view=VIEW, budget_tokens=2000
    )
    assert fact_items(russian) == [member]  # the Russian one, found or not by the words


def test_the_budget_does_not_pay_twice_for_one_value(store: Store) -> None:
    root, member = lyon_group(store)
    both = engine(store).recall("Karim Lyon Карим Лионе", view=VIEW, budget_tokens=2000)
    assert len([i for i in fact_items(both) if i in (root, member)]) == 1


def test_an_excluded_group_is_not_injected_again(store: Store) -> None:
    root, _ = lyon_group(store)
    result = engine(store).recall(
        "Где живёт Карим со своей семьёй?", view=VIEW, budget_tokens=2000, exclude_ids=[root]
    )
    assert fact_items(result) == []  # the value is already in the agent's context


def test_the_more_precise_member_is_preferred() -> None:
    def member(fact_id: str, value: str, language: str) -> Fact:
        return Fact(
            id=fact_id,
            node_id="n",
            statement="s",
            attribute="date.deadline",
            value=value,
            valid_from=None,
            valid_to=None,
            origin="extracted",
            confidence=0.7,
            source_id=None,
            excerpt_start=None,
            excerpt_end=None,
            superseded_by=None,
            pinned=False,
            flags=(),
            scopes=(ALICE,),
            external_id=None,
            created_at="t",
            updated_at="t",
            archived_at=None,
            language=language,
        )

    group = [member("a", "mars 2026", "fr"), member("b", "15 March 2026", "en")]
    assert preferred(group, "de").id == "b"  # no member in German: the fuller date
    assert preferred(group, "fr").id == "a"  # the query's language first
    assert preferred([member("a", "Lyon", "fr"), member("b", "Лион", "ru")], "und").id == "a"


def test_the_preferred_member_replaces_the_one_the_words_found(store: Store) -> None:
    """Only the French member matches "Karim" by its words; the group's first
    fact (Russian) is preferred when the query's language says nothing: it is
    injected in the found one's place, at its rank."""
    node = karim(store)
    root = fact(store, node, "Карим живёт в Лионе со всей своей семьёй.", "Лион", language="ru")
    fact(store, node, "Karim habite à Lyon avec sa famille.", "Lyon", language="fr", same_as=root)
    result = engine(store).recall("Karim", view=VIEW, budget_tokens=2000)
    assert fact_items(result) == [root]
