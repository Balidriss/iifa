"""Attribute keys through the pipeline (SPEC 5.4): the prompt steers them (base
vocabulary, the subject's existing keys), the write path aligns them (a table
of unambiguous variants without a model, near free-form keys by embeddings at
tier 3), so that a change of value supersedes the old one - and, when we cannot
tell, nothing is closed."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.helpers import auth
from tests.model.test_tier3 import FAR, NEAR, tier3_settings
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)
LYON = "Karim habite à Lyon depuis le printemps avec sa famille."
PARIS = "Karim habite maintenant à Paris avec sa famille et son chat."
OFFICE = "Karim travaille maintenant au bureau de Paris tous les jours."


def make_client(app: FastAPI) -> TestClient:
    return TestClient(app, headers=auth("all"))


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


def karim_with(app: FastAPI, attribute: str, value: str, statement: str = LYON) -> tuple[str, str]:
    store = app.state.store
    node = store.find_node("Karim", VIEW) or store.create_node(
        scopes=[SCOPE], name="Karim", kind="person"
    )
    fact = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement=statement,
        origin="extracted",
        attribute=attribute,
        value=value,
        language="fr",
    )
    return node.id, fact.id


def karim_answer(statement: str, attribute: str, value: str) -> Any:
    return answer(
        nodes=[{"name": "Karim", "kind": "person"}],
        facts=[
            {
                "subject": "Karim",
                "statement": statement,
                "attribute": attribute,
                "value": value,
                "quote": statement,
            }
        ],
    )


def ingest(client: TestClient, app: FastAPI, text: str) -> dict[str, Any]:
    post_turn(client, text)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def live_keys(app: FastAPI, node: str) -> list[tuple[str | None, str | None]]:
    return sorted((f.attribute, f.value) for f in app.state.store.facts([node], VIEW))


def test_the_prompt_gives_the_vocabulary_and_the_subjects_existing_keys(
    app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    karim_with(app, "home", "Lyon")
    karim_with(app, "preference", "réponses courtes", "Karim préfère les réponses courtes.")
    store = app.state.store
    sofia = store.create_node(scopes=["user:bob"], name="Sofia", kind="person")
    store.add_fact(
        scopes=["user:bob"],
        node_id=sofia.id,
        statement="Sofia is a secret agent.",
        origin="extracted",
        attribute="role",
        value="agent",
    )
    fake_openai.script(chat=[answer()])
    with make_client(app) as client:
        ingest(client, app, "Karim et Sofia ont parlé du projet solaire hier soir.")
    system = fake_openai.chat_calls[0].messages[0]["content"]
    for word in ("home (", "workplace (", "preference.<topic>", "status.<topic>"):
        assert word in system
    assert "Karim: home" in system
    assert "Karim: home, preference" not in system  # a bare family is never offered for reuse
    assert "Sofia" not in system  # P9: only the subjects the source's scopes see


def test_an_unambiguous_variant_supersedes_and_the_canonical_key_is_stored(
    app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    node, old = karim_with(app, "lives_in", "Lyon")
    fake_openai.script(chat=[karim_answer(PARIS, "residence", "Paris")])
    with make_client(app) as client:
        result = ingest(client, app, PARIS)
        assert result["facts"]["superseded"] == 1 and result["facts"]["conflict"] == 0
        assert result["keys"]["renamed"] == 1
        assert app.state.store.get_fact(old).valid_to is not None
        assert live_keys(app, node) == [("home", "Paris")]


@pytest.mark.parametrize("ambiguous", ["location", "city", "work", "works_at"])
def test_an_ambiguous_key_never_closes_a_home(
    app: FastAPI, fake_openai: FakeOpenAIServer, ambiguous: str
) -> None:
    """A work city keyed "location" or "city": we cannot tell, we do not supersede."""
    node, old = karim_with(app, "home", "Lyon")
    fake_openai.script(chat=[karim_answer(OFFICE, ambiguous, "Paris")])
    with make_client(app) as client:
        result = ingest(client, app, OFFICE)
        assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
        assert app.state.store.get_fact(old).valid_to is None
        assert ("home", "Lyon") in live_keys(app, node)


def test_preferences_on_different_topics_do_not_replace_each_other(
    app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    node, _ = karim_with(
        app,
        "preference.answers",
        "réponses courtes",
        "Karim préfère les réponses courtes et précises le matin.",
    )
    statement = "Karim préfère le thé vert au café le matin."
    fake_openai.script(chat=[karim_answer(statement, "preferred_drink", "thé vert")])
    with make_client(app) as client:
        result = ingest(client, app, statement)
        assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
        keys = [k for k, _ in live_keys(app, node)]
        assert keys == ["preference.answers", "preference.drink"]


@pytest.mark.parametrize(("old_key", "new_key"), [("preference", "prefers"), ("status", "status")])
def test_a_bare_family_key_never_supersedes(
    app: FastAPI, fake_openai: FakeOpenAIServer, old_key: str, new_key: str
) -> None:
    """ "prefers tea" must not close "prefers short answers" because a model left
    out the topic: a bare family key drives neither a supersession nor a conflict."""
    node, old = karim_with(
        app, old_key, "réponses courtes", "Karim préfère les réponses courtes le matin."
    )
    statement = "Karim préfère le thé vert au café le matin."
    fake_openai.script(chat=[karim_answer(statement, new_key, "thé vert")])
    with make_client(app) as client:
        result = ingest(client, app, statement)
        assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
        assert app.state.store.get_fact(old).valid_to is None
        assert len(live_keys(app, node)) == 2


# --------------------------------------------------------------------------- tier 3 (step 4)


def tier3_app(
    tmp_path: Path, fake: FakeOpenAIServer, clock: FakeClock, **overrides: Any
) -> FastAPI:
    return create_app(tier3_settings(tmp_path, fake, **overrides), clock=clock)


COMMUTE = "Karim va au travail à vélo tous les matins depuis janvier."
TRAIN = "Karim va maintenant au travail en train tous les matins."


def key_inputs(fake: FakeOpenAIServer, word: str) -> list[list[str]]:
    return [c.body["input"] for c in fake.embedding_calls if word in c.body["input"]]


def test_tier3_a_near_free_form_key_is_aligned_by_embeddings(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    app = tier3_app(tmp_path, fake_openai, clock, key_threshold=0.9)
    try:
        fake_openai.set_vectors({"commute mode": NEAR, "transport mode": NEAR, "Karim": FAR})
        node, old = karim_with(app, "commute_mode", "vélo", COMMUTE)
        fake_openai.script(chat=[karim_answer(TRAIN, "transport_mode", "train")])
        with make_client(app) as client:
            result = ingest(client, app, TRAIN)
            assert result["keys"]["matched"] == 1
            assert result["facts"]["superseded"] == 1
            assert app.state.store.get_fact(old).valid_to is not None
            assert live_keys(app, node) == [("commute_mode", "train")]
            (key_call,) = key_inputs(fake_openai, "transport mode")
            assert set(key_call) == {"transport mode", "commute mode"}  # keys only, one call
    finally:
        app.state.store.close()


def test_tier3_a_vocabulary_key_is_never_renamed_nor_a_target(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """A new "employer" must not be matched to "role" and close it; a free-form
    "dwelling" must not be matched to "home" either: step 4 is for free-form keys."""
    app = tier3_app(tmp_path, fake_openai, clock, key_threshold=0.5)
    try:
        fake_openai.set_vectors({"role": NEAR, "employer": NEAR, "home": NEAR, "dwelling": NEAR})
        _, role = karim_with(app, "role", "journaliste", "Karim travaille comme journaliste.")
        _, home = karim_with(app, "home", "Lyon")
        employer = "Karim travaille désormais pour le journal Le Monde à Paris."
        dwelling = "Karim loge désormais dans un petit appartement à Paris."
        fake_openai.script(
            chat=[
                karim_answer(employer, "employer", "Le Monde"),
                karim_answer(dwelling, "dwelling", "appartement à Paris"),
            ]
        )
        with make_client(app) as client:
            for text in (employer, dwelling):
                result = ingest(client, app, text)
                assert result["keys"]["matched"] == 0 and result["facts"]["superseded"] == 0
            store = app.state.store
            assert store.get_fact(role).valid_to is None and store.get_fact(home).valid_to is None
            assert not key_inputs(fake_openai, "employer")
            assert not key_inputs(fake_openai, "dwelling")  # no free-form key to compare with
    finally:
        app.state.store.close()


def test_tier3_family_keys_compare_their_topics_only(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """ "preference music" must not match "preference drink" on the shared word:
    the topics are compared, within the family, and "beverage" does match "drink"."""
    app = tier3_app(tmp_path, fake_openai, clock, key_threshold=0.9)
    try:
        fake_openai.set_vectors({"drink": NEAR, "beverage": NEAR, "music": FAR})
        node, drink = karim_with(app, "preference.drink", "thé", "Karim préfère le thé le matin.")
        music = "Karim préfère le jazz quand il travaille le soir."
        beverage = "Karim préfère maintenant le café le matin au bureau."
        fake_openai.script(
            chat=[
                karim_answer(music, "preference.music", "jazz"),
                karim_answer(beverage, "preference.beverage", "café"),
                # a preference never closes on its own (SPEC 5.4): it is asked
                chat_json({"replacements": [{"pair": 0, "verdict": "replaces"}]}),
            ]
        )
        with make_client(app) as client:
            first = ingest(client, app, music)
            assert first["keys"]["matched"] == 0 and first["facts"]["superseded"] == 0
            (topics,) = key_inputs(fake_openai, "music")
            assert set(topics) == {"music", "drink"}  # topics only, never "preference"
            second = ingest(client, app, beverage)
            assert second["keys"]["matched"] == 1 and second["facts"]["superseded"] == 1
            assert app.state.store.get_fact(drink).valid_to is not None
            assert live_keys(app, node) == [
                ("preference.drink", "café"),
                ("preference.music", "jazz"),
            ]
    finally:
        app.state.store.close()


def test_tier3_uncalibrated_keys_are_not_embedded(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    app = tier3_app(tmp_path, fake_openai, clock)  # no key_threshold: step 4 is off
    try:
        fake_openai.set_vectors({"commute mode": NEAR, "transport mode": NEAR})
        _, old = karim_with(app, "commute_mode", "vélo", COMMUTE)
        fake_openai.script(chat=[karim_answer(TRAIN, "transport_mode", "train")])
        with make_client(app) as client:
            result = ingest(client, app, TRAIN)
            assert result["keys"]["matched"] == 0 and result["facts"]["superseded"] == 0
            assert app.state.store.get_fact(old).valid_to is None
            assert not key_inputs(fake_openai, "transport mode")
    finally:
        app.state.store.close()


def test_tier3_a_far_key_is_kept(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    app = tier3_app(tmp_path, fake_openai, clock, key_threshold=0.9)
    try:
        fake_openai.set_vectors({"commute mode": NEAR, "shoe size": FAR})
        node, _ = karim_with(app, "commute_mode", "vélo", COMMUTE)
        statement = "Karim porte des chaussures de pointure quarante-deux."
        fake_openai.script(chat=[karim_answer(statement, "shoe_size", "42")])
        with make_client(app) as client:
            result = ingest(client, app, statement)
            assert result["keys"]["matched"] == 0 and result["facts"]["superseded"] == 0
            assert [k for k, _ in live_keys(app, node)] == ["commute_mode", "shoe_size"]
    finally:
        app.state.store.close()
