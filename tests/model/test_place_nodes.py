"""Place nodes the model proposes beside a value: a node named
after another subject's value ("bureau de Osaka" for where Karim works) is
still written, but with no fact of its own its handle never ranks ahead of a
fact that matched every word it matched; a place with facts of its own is
ranked as before, and a place the model links or describes is a node like
any other."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.recall import engine as recall_engine
from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.tokens import ConservativeTokenizer
from tests.fakes.openai_server import FakeOpenAIServer
from tests.tier2 import SCOPE, answer, post_turn, run_all

VIEW = Visibility.of_scope(SCOPE)
QUESTION = "Où se trouve le bureau de Karim ?"


def _fact(subject: str, statement: str, attribute: str | None, value: str | None) -> Any:
    return {
        "subject": subject,
        "statement": statement,
        "attribute": attribute,
        "value": value,
        "quote": statement,
        "confidence": 0.9,
    }


def _works_at(person: str, city: str) -> Any:
    """What a real model answered: the workplace as the person's value, plus a
    place node named after it, a summary naming the person, and an edge."""
    statement = f"{person} travaille au bureau de {city}."
    place = f"bureau de {city}"
    return answer(
        nodes=[
            {"name": person, "kind": "person"},
            {"name": place, "kind": "place", "summary": f"Le bureau où travaille {person}."},
        ],
        facts=[_fact(person, statement, "workplace", city)],
        edges=[{"src": person, "dst": place, "type": "works_at", "quote": statement}],
    )


def _store(
    client: TestClient, app: FastAPI, fake: FakeOpenAIServer, *, venue: bool = False
) -> dict[str, str]:
    turns = [
        (
            "Karim habite à Lyon.",
            answer(facts=[_fact("Karim", "Karim habite à Lyon.", "home", "Lyon")]),
        ),
        ("Karim travaille au bureau de Osaka.", _works_at("Karim", "Osaka")),
        ("Ivan travaille au bureau de Montréal.", _works_at("Ivan", "Montréal")),
        (
            "Karim préfère le thé.",
            answer(facts=[_fact("Karim", "Karim préfère le thé.", "preference.drink", "thé")]),
        ),
    ]
    if venue:  # the place gains a fact of its own
        own = "Le bureau de Osaka ferme le vendredi."
        turns.append((own, answer(facts=[_fact("bureau de Osaka", own, None, None)])))
    fake.script(chat=[reply for _, reply in turns])
    for text, _ in turns:
        post_turn(client, text)
    run_all(client, app)
    store = app.state.store
    ids: dict[str, str] = {}
    for name in ("Karim", "Ivan", "bureau de Osaka", "bureau de Montréal"):
        node = store.find_node(name, VIEW)
        assert node is not None, name
        ids[name] = node.id
    (office,) = (f for f in store.facts([ids["Karim"]], VIEW) if f.attribute == "workplace")
    ids["office"] = office.id
    return ids


def _ranked(app: FastAPI, query: str, budget: int = 4000, *, anchoring: bool = True) -> list[str]:
    engine = RecallEngine(app.state.store, ConservativeTokenizer(), subject_anchoring=anchoring)
    result: dict[str, Any] = engine.recall(query, view=VIEW, budget_tokens=budget)
    return [item["id"] for item in result["items"]]


def _unmoved(monkeypatch: Any, app: FastAPI, query: str, *, anchoring: bool) -> list[str]:
    """The ranking the rule changes: the same recall with the rule taken out."""
    with monkeypatch.context() as patch:
        patch.setattr(
            recall_engine.RecallEngine, "_behind_their_facts", lambda _self, out, _view: out
        )
        return _ranked(app, query, anchoring=anchoring)


def test_the_model_s_place_nodes_are_written(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Nothing is dropped by a word list: the places stay nodes, linked."""
    ids = _store(client, app, fake_openai)
    store = app.state.store
    for name in ("bureau de Osaka", "bureau de Montréal"):
        assert store.get_node(ids[name]).kind == "place"
        assert store.facts([ids[name]], VIEW) == []
    assert any(e.dst == ids["bureau de Osaka"] for e in store.edges_of([ids["Karim"]], VIEW))


def test_a_place_named_after_a_value_ranks_behind_the_facts_that_say_it(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, monkeypatch: Any
) -> None:
    """Ranked without anchoring, as for a question naming two subjects: by
    score alone the places' short names outscore Karim's workplace fact on
    "bureau" and "de"; each comes after every fact that matched those words
    too. Its summary, which names Karim, is never ranked."""
    ids = _store(client, app, fake_openai)
    office, places = ids["office"], [ids["bureau de Osaka"], ids["bureau de Montréal"]]
    before = _unmoved(monkeypatch, app, QUESTION, anchoring=False)
    assert all(before.index(p) < before.index(office) for p in places)  # by score alone
    after = _ranked(app, QUESTION, anchoring=False)
    assert sorted(after) == sorted(before)  # a reordering: nothing added, nothing removed
    store = app.state.store
    for place in places:
        holders = [
            i for i in after if i.startswith("fact_") and "bureau" in store.get_fact(i).statement
        ]
        assert all(after.index(h) < after.index(place) for h in holders)
    rest = [i for i in after if i not in places]
    assert rest == [i for i in before if i not in places]  # the others keep their order


def test_with_anchoring_the_named_subject_s_workplace_fits_a_small_budget(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ids = _store(client, app, fake_openai)
    ranked = _ranked(app, QUESTION)
    assert ranked.index(ids["office"]) < ranked.index(ids["bureau de Osaka"])
    assert ids["office"] in _ranked(app, QUESTION, 300)


def test_a_place_with_a_fact_of_its_own_keeps_its_rank(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, monkeypatch: Any
) -> None:
    ids = _store(client, app, fake_openai, venue=True)
    before = _unmoved(monkeypatch, app, QUESTION, anchoring=False)
    after = _ranked(app, QUESTION, anchoring=False)
    osaka = ids["bureau de Osaka"]
    assert after.index(osaka) == before.index(osaka)
    assert after.index(ids["bureau de Montréal"]) > after.index(ids["office"])


def test_a_city_someone_lives_in_and_a_venue_with_its_own_facts_are_nodes(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Controls: a place the model links or describes stays a node; a question
    about the venue gets the venue's own fact first."""
    lives = "Alice habite à Lyon."
    venue = "Le Comptoir ouvre à 18 heures et sert des tapas."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Alice", "kind": "person"}, {"name": "Lyon", "kind": "place"}],
                facts=[_fact("Alice", lives, "home", "Lyon")],
                edges=[{"src": "Alice", "dst": "Lyon", "type": "lives_in", "quote": lives}],
            ),
            answer(
                nodes=[{"name": "Le Comptoir", "kind": "place"}],
                facts=[_fact("Le Comptoir", venue, None, None)],
            ),
        ]
    )
    post_turn(client, lives)
    post_turn(client, venue)
    run_all(client, app)
    store = app.state.store
    alice, lyon = store.find_node("Alice", VIEW), store.find_node("Lyon", VIEW)
    comptoir = store.find_node("Le Comptoir", VIEW)
    assert alice is not None and lyon is not None and comptoir is not None
    assert lyon.kind == "place" and comptoir.kind == "place"
    assert any(
        (e.src, e.dst, e.type) == (alice.id, lyon.id, "lives_in")
        for e in store.edges_of([alice.id], VIEW)
    )
    (own,) = store.facts([comptoir.id], VIEW)
    assert _ranked(app, "À quelle heure ouvre Le Comptoir ?", 300)[0] == own.id
    # Lyon holds no fact of its own: a question naming only the city gets
    # Alice's fact, which says it, ahead of the city's handle
    ranked = _ranked(app, "Lyon ?")
    assert lyon.id in ranked and ranked.index(lyon.id) > ranked.index(alice_home(store, alice.id))


def alice_home(store: Any, alice_id: str) -> str:
    (home,) = store.facts([alice_id], VIEW)
    return str(home.id)
