"""Subject anchoring (SPEC 5.5): a question naming exactly one visible subject
puts that subject's facts first; the node's own handle keeps its place; when
the question names it exactly (whole name or whole alias), the subject's live
facts also join the candidates, within the budget. A question naming none,
two, or two homonyms is ranked as before."""

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


# --------------------------------------------------------------------------- the named subject's facts join


def _recall(
    store: Store, query: str, *, budget: int = 4000, view: Visibility = ALICE, **kw: Any
) -> dict[str, Any]:
    engine = RecallEngine(store, ConservativeTokenizer(), **kw)
    result: dict[str, Any] = engine.recall(query, view=view, budget_tokens=budget)
    return result


def test_an_alias_brings_the_subjects_facts_that_share_no_word_with_the_question(
    store: Store,
) -> None:
    """ "Où habite Kiki ?", Kiki an alias of Karim: the move shares no word with
    the question; it joins, after the subject's searched fact, ahead of the rest."""
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    query = "Où habite Kiki maintenant ?"
    before = _ids(store, query, anchoring=False)
    after = _ids(store, query, anchoring=True)
    assert ids["moved"] not in before  # the search alone never finds it
    assert after[:2] == [ids["tea"], ids["moved"]]
    assert set(before) <= set(after)  # nothing the search found is lost
    joined = next(i for i in _recall(store, query)["items"] if i["id"] == ids["moved"])
    assert joined["why"] == "subject named in the question: Karim"


def test_the_name_itself_brings_the_facts_too(store: Store) -> None:
    ids = _store(store)
    told = store.add_fact(
        scopes=["user:alice"],
        node_id=ids["karim"],
        statement="Il a vendu sa voiture.",
        origin="explicit",
    )
    assert told.id not in _ids(store, "Que sait-on de Karim ?", anchoring=False)
    assert told.id in _ids(store, "Que sait-on de Karim ?", anchoring=True)


def test_without_anchoring_nothing_joins(store: Store) -> None:
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    assert ids["moved"] not in _ids(store, "Où habite Kiki ?", anchoring=False)


def test_only_live_facts_join(store: Store) -> None:
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    old = store.add_fact(
        scopes=["user:alice"],
        node_id=ids["karim"],
        statement="Karim vit à Rabat.",
        origin="explicit",
    )
    store.close_fact(old.id, superseded_by=ids["moved"])
    after = _ids(store, "Où habite Kiki ?", anchoring=True)
    assert ids["moved"] in after and old.id not in after


def test_an_excluded_fact_does_not_join(store: Store) -> None:
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    engine = RecallEngine(store, ConservativeTokenizer())
    result = engine.recall(
        "Où habite Kiki ?", view=ALICE, budget_tokens=4000, exclude_ids=[ids["moved"]]
    )
    assert ids["moved"] not in [item["id"] for item in result["items"]]


def test_two_named_subjects_join_nothing(store: Store) -> None:
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    query = "Où habitent Kiki et Ivan ?"
    assert _ids(store, query, anchoring=True) == _ids(store, query, anchoring=False)


def _many(store: Store, count: int) -> tuple[str, list[str]]:
    karim = store.create_node(scopes=["user:alice"], name="Karim", kind="person")
    store.update_node(karim.id, add_aliases=["Kiki"])
    facts = [
        store.add_fact(
            scopes=["user:alice"],
            node_id=karim.id,
            statement=f"Karim a noté la référence numéro {n} du dossier.",
            origin="explicit",
        ).id
        for n in range(count)
    ]
    return karim.id, facts


def test_the_budget_holds_whatever_the_subjects_size(store: Store) -> None:
    """P1: hundreds of joined facts, every budget from nothing to large, never
    a token over; the join is bounded by the candidate bound."""
    _, facts = _many(store, 260)
    for budget in (0, 1, 30, 60, 61, 150, 300, 600, 1200, 5000):
        result = _recall(store, "Où habite Kiki ?", budget=budget)
        assert result["used_tokens"] <= budget
        engine = RecallEngine(store, ConservativeTokenizer())
        assert sum(engine.cost(item) for item in result["items"]) == result["used_tokens"]
    wide = _recall(store, "Où habite Kiki ?", budget=10**6, max_candidates=40)
    assert len([i for i in wide["items"] if i["id"] in set(facts)]) == 40
    newest = _recall(store, "Où habite Kiki ?", budget=300)["items"]
    assert newest[0]["id"] == facts[-1]  # most recent first


def test_a_named_node_the_view_cannot_see_adds_nothing(store: Store) -> None:
    """P9: Bob asks about Alice's Kiki: nothing of hers, joined or found."""
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    bob = Visibility.of_scope("user:bob")
    assert _recall(store, "Où habite Kiki ?", view=bob)["items"] == []


def test_a_fact_in_a_scope_the_view_cannot_see_does_not_join(store: Store) -> None:
    """P9: the node is visible, one of its facts is in a narrower scope."""
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    hidden = store.add_fact(
        scopes=["user:alice/project:x"],
        node_id=ids["karim"],
        statement="Karim garde la clé du local.",
        origin="explicit",
    )
    after = _ids(store, "Où habite Kiki ?", anchoring=True)
    assert ids["moved"] in after and hidden.id not in after
    wide = _ids(store, "Où habite Kiki ?", anchoring=True, view=Visibility.of_subtree("user:alice"))
    assert hidden.id in wide


def test_a_near_form_does_not_bring_another_persons_facts(store: Store) -> None:
    """ "Karima" is not Karim: an inflected ending anchors (SPEC 4.2) but never
    joins - a wrong join injects another person's facts."""
    ids = _store(store)
    after = _ids(store, "Où habite Karima ?", anchoring=True)
    assert ids["moved"] not in after and ids["tea"] not in after


def test_karim_and_karima_each_bring_only_their_own_facts(store: Store) -> None:
    ids = _store(store)
    karima = store.create_node(scopes=["user:alice"], name="Karima", kind="person")
    hers = store.add_fact(
        scopes=["user:alice"],
        node_id=karima.id,
        statement="Elle vit à Nantes.",
        origin="explicit",
    )
    about_her = _ids(store, "Où habite Karima ?", anchoring=True)
    assert hers.id in about_her
    assert ids["moved"] not in about_her and ids["tea"] not in about_her
    about_him = _ids(store, "Où habite Karim ?", anchoring=True)
    assert ids["moved"] in about_him and hers.id not in about_him


def test_a_piece_of_a_word_never_names_the_subject(store: Store) -> None:
    ids = _store(store)
    store.update_node(ids["karim"], add_aliases=["Kiki"])
    assert ids["moved"] not in _ids(store, "Où habite Kikio ?", anchoring=True)


def test_an_inflected_name_anchors_but_adds_nothing(store: Store) -> None:
    subject = store.create_node(scopes=["user:alice"], name="Карим", kind="person")
    moved = store.add_fact(
        scopes=["user:alice"],
        node_id=subject.id,
        statement="Он переехал в Казань.",
        origin="explicit",
    )
    assert moved.id not in _ids(store, "Где живёт сестра Карима?", anchoring=True)
    assert moved.id in _ids(store, "Где живёт Карим?", anchoring=True)


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


def test_a_homonym_the_search_did_not_return_stops_the_join(store: Store) -> None:
    """Two visible Karims, one node cut by the candidate bound: which one is
    meant cannot be told, so neither one's facts join."""
    first = store.create_node(scopes=["user:alice"], name="Karim", kind="person")
    second = store.create_node(scopes=["user:alice"], name="Karim", kind="person")
    unnamed = {
        store.add_fact(
            scopes=["user:alice"], node_id=node.id, statement=statement, origin="explicit"
        ).id
        for node, statement in ((first, "Il a vendu sa voiture."), (second, "Il aime le vélo."))
    }
    query = "Que sait-on de Karim ?"
    narrow = _recall(store, query, max_candidates=1)  # one Karim's node only is found
    assert len(narrow["items"]) >= 1
    assert not unnamed & {i["id"] for i in narrow["items"]}
    assert not unnamed & set(_ids(store, query, anchoring=True))


def test_a_name_in_a_script_without_spaces_never_joins(store: Store) -> None:
    tanaka = store.create_node(scopes=["user:alice"], name="田中", kind="person")
    moved = store.add_fact(
        scopes=["user:alice"], node_id=tanaka.id, statement="大阪に引っ越した。", origin="explicit"
    )
    assert moved.id not in _ids(store, "田中、どこに住んでいますか？", anchoring=True)
