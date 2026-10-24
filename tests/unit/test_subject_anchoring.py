"""Subject anchoring (SPEC 5.5): a question naming exactly one visible subject
puts that subject's facts first - a reordering, nothing added or removed; the
node's own handle keeps its place; a question naming none, two, or two
homonyms is ranked as before."""

from __future__ import annotations

from typing import Any

from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

ALICE = Visibility.of_scope("user:alice")
QUESTION = "Où habite Karim maintenant ?"


def _store(store: Store) -> dict[str, str]:
    """The measured case: other facts match "habite" and "maintenant", the
    current one only the subject's name."""
    karim = store.create_node(scopes=["user:alice"], name="Karim", kind="person")
    ivan = store.create_node(scopes=["user:alice"], name="Ivan", kind="person")
    emilie = store.create_node(scopes=["user:alice"], name="Émilie", kind="person")
    ids: dict[str, str] = {"karim": karim.id, "ivan": ivan.id}
    for key, node, statement in (
        ("moved", karim.id, "Karim a déménagé à Osaka."),
        ("tea", karim.id, "Karim préfère maintenant le thé."),
        ("ivan", ivan.id, "Ivan habite à Kazan depuis maintenant deux ans."),
        ("emilie", emilie.id, "Émilie habite à Casablanca."),
    ):
        ids[key] = store.add_fact(
            scopes=["user:alice"], node_id=node, statement=statement, origin="explicit"
        ).id
    return ids


def _ids(store: Store, query: str, *, anchoring: bool, view: Visibility = ALICE) -> list[str]:
    engine = RecallEngine(store, ConservativeTokenizer(), subject_anchoring=anchoring)
    result: dict[str, Any] = engine.recall(query, view=view, budget_tokens=4000)
    return [item["id"] for item in result["items"]]


def test_a_question_naming_one_subject_puts_its_facts_first(store: Store) -> None:
    ids = _store(store)
    before = _ids(store, QUESTION, anchoring=False)
    after = _ids(store, QUESTION, anchoring=True)
    assert sorted(before) == sorted(after)  # a reordering: nothing added, nothing removed
    facts = {ids["moved"], ids["tea"]}
    assert set(after[: len(facts)]) == facts  # the subject's facts come first
    assert before.index(ids["moved"]) > before.index(ids["ivan"])  # the case it fixes
    # each side keeps its own order
    assert [i for i in after if i in facts] == [i for i in before if i in facts]
    assert [i for i in after if i not in facts] == [i for i in before if i not in facts]


def test_the_subject_node_handle_keeps_its_place(store: Store) -> None:
    """Only the facts move: the node's handle (175 to 358 tokens on the bench)
    moved ahead pushed the answer out of small budgets."""
    ids = _store(store)
    facts = {ids["moved"], ids["tea"]}
    before = [i for i in _ids(store, QUESTION, anchoring=False) if i not in facts]
    after = [i for i in _ids(store, QUESTION, anchoring=True) if i not in facts]
    assert ids["karim"] in after
    assert after.index(ids["karim"]) == before.index(ids["karim"])


def test_a_question_naming_no_subject_is_ranked_as_before(store: Store) -> None:
    _store(store)
    query = "Qui habite à Kazan maintenant ?"
    assert _ids(store, query, anchoring=True) == _ids(store, query, anchoring=False)


def test_a_question_naming_two_subjects_is_ranked_as_before(store: Store) -> None:
    _store(store)
    query = "Où habitent Karim et Ivan maintenant ?"
    assert _ids(store, query, anchoring=True) == _ids(store, query, anchoring=False)


def test_two_homonyms_in_the_view_are_not_anchored(store: Store) -> None:
    _store(store)
    other = store.create_node(scopes=["user:alice/project:x"], name="Karim", kind="person")
    store.add_fact(
        scopes=["user:alice/project:x"],
        node_id=other.id,
        statement="Karim habite maintenant à Lyon.",
        origin="explicit",
    )
    both = Visibility.of_subtree("user:alice")
    assert _ids(store, QUESTION, anchoring=True, view=both) == _ids(
        store, QUESTION, anchoring=False, view=both
    )


def test_a_subject_the_view_cannot_see_anchors_nothing(store: Store) -> None:
    """P9: only a visible node can be the named subject."""
    _store(store)
    bob = Visibility.of_scope("user:bob")
    hidden = store.create_node(scopes=["user:bob"], name="Ivan", kind="person")
    store.add_fact(
        scopes=["user:bob"], node_id=hidden.id, statement="Ivan habite à Paris.", origin="explicit"
    )
    assert _ids(store, QUESTION, anchoring=True, view=bob) == _ids(
        store, QUESTION, anchoring=False, view=bob
    )


def test_an_alias_names_the_subject_and_nothing_is_added(store: Store) -> None:
    """The alias names Karim: his one candidate fact ("maintenant") moves ahead;
    the move, which shares no word with the question, is not added (Next: a
    named subject's facts could join the candidates)."""
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    query = "Où habite Kiki maintenant ?"
    before = _ids(store, query, anchoring=False)
    after = _ids(store, query, anchoring=True)
    assert sorted(after) == sorted(before) and ids["moved"] not in after
    assert after[0] == ids["tea"]


# --------------------------------------------------------------------------- the naming rule (SPEC 4.2)


def _people(store: Store, name: str, own: str, other: str) -> tuple[str, str]:
    """A subject and a distractor: the distractor shares the question's common
    word, the subject's fact only its name."""
    subject = store.create_node(scopes=["user:alice"], name=name, kind="person")
    third = store.create_node(scopes=["user:alice"], name="Ivan", kind="person")
    mine = store.add_fact(
        scopes=["user:alice"], node_id=subject.id, statement=own, origin="explicit"
    )
    store.add_fact(scopes=["user:alice"], node_id=third.id, statement=other, origin="explicit")
    return subject.id, mine.id


def _discriminates(store: Store, query: str, subject: str, fact: str) -> None:
    """Without anchoring the distractor comes first; with it, the subject."""
    before = _ids(store, query, anchoring=False)
    after = _ids(store, query, anchoring=True)
    assert before[0] not in {subject, fact}, before  # the test would prove nothing
    assert after[0] == fact  # the subject's fact comes first; its node keeps its place
    assert sorted(before) == sorted(after)


def test_a_name_without_its_accent_names_the_subject(store: Store) -> None:
    subject, fact = _people(
        store, "Émilie", "Émilie a déménagé à Lyon.", "Ivan habite maintenant à Kazan."
    )
    _discriminates(store, "Où habite Emilie maintenant ?", subject, fact)


def test_an_elided_name_names_the_subject(store: Store) -> None:
    subject, fact = _people(
        store, "Émilie", "Émilie travaille à Lyon.", "Le bureau de Ivan est à Kazan, au bureau."
    )
    _discriminates(store, "Où est le bureau d'Émilie ?", subject, fact)


def test_an_inflected_name_names_the_subject(store: Store) -> None:
    """ "Карима" (genitive) names Карим, by the naming rule of SPEC 4.2."""
    subject, fact = _people(
        store, "Карим", "Карима перевели в Казань.", "Сестра Ивана живёт в Москве."
    )
    _discriminates(store, "Где живёт сестра Карима?", subject, fact)
