"""Role, team and preferences (SPEC 5.4): they may hold several values at
once, so with the replacement question switched on the adjudication asks
whether the new value replaces the old one or both are true. replaces
supersedes, same links, both keeps both live without a conflict, unsure (or
anything invalid) keeps the conflict. The switch is on by default (with the
source's quotes, measured with qwen3.5:9b, the headline metric is no lower and
15 false replaces are stopped); off, these keys close within one language."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest.extract import PAIR_QUOTE_CHARS
from iifa.ingest.schema import parse_replacement_verdicts
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.model.test_values_pipeline import VIEW, ingest, karim_fact, newest, said
from tests.tier2 import answer, post_turn, run_all, tier2_settings

ARCHITECT_FR = "Karim travaille comme architecte depuis des années."
ACCOUNTANT_RU = "Карим теперь работает бухгалтером в Лионе."
ACCOUNTANT_FR = "Karim travaille comme comptable depuis des années."
PHOTOGRAPHER_FR = "Karim est aussi photographe le week-end."


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    settings = tier2_settings(tmp_path, fake_openai, replacement_question=True)
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def default_app(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> Iterator[FastAPI]:
    """The switch turned off: the behaviour before the replacement question."""
    settings = tier2_settings(tmp_path, fake_openai, replacement_question=False)
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def default_client(default_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(default_app, headers=auth("all")) as test_client:
        yield test_client


def test_switched_off_a_role_in_one_language_closes_as_before(
    default_client: TestClient, default_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(default_app, ACCOUNTANT_FR, "role", "comptable", "fr")
    fake_openai.script(chat=[said(PHOTOGRAPHER_FR, "role", "photographe")])
    result = ingest(default_client, default_app, PHOTOGRAPHER_FR)
    assert len(fake_openai.chat_calls) == 1  # no question asked
    assert result["facts"]["superseded"] == 1
    assert default_app.state.store.get_fact(old).valid_to is not None


def test_switched_off_a_role_across_languages_keeps_its_conflict(
    default_client: TestClient, default_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(default_app, ARCHITECT_FR, "role", "architecte", "fr")
    fake_openai.script(chat=[said(ACCOUNTANT_RU, "role", "бухгалтер")])
    result = ingest(default_client, default_app, ACCOUNTANT_RU)
    assert len(fake_openai.chat_calls) == 1
    assert result["facts"]["conflict"] == 1 and result["facts"]["superseded"] == 0
    assert result["values"]["replacement"] == {"same": 0, "replaces": 0, "both": 0, "unsure": 0}
    assert default_app.state.store.get_fact(old).valid_to is None


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def replacement(word: str) -> Reply:
    return chat_json({"replacements": [{"pair": 0, "verdict": word}]})


def _role_change(
    client: TestClient, app: FastAPI, fake: FakeOpenAIServer, verdict: Reply
) -> tuple[str, dict[str, Any]]:
    old = karim_fact(app, ARCHITECT_FR, "role", "architecte", "fr")
    fake.script(chat=[said(ACCOUNTANT_RU, "role", "бухгалтер"), verdict])
    return old, ingest(client, app, ACCOUNTANT_RU)


def test_a_role_that_replaces_supersedes(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old, result = _role_change(client, app, fake_openai, replacement("replaces"))
    assert len(fake_openai.chat_calls) == 2
    system = fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert "replace" in system and "both" in system
    assert '"quote", the exact words of its source' in system
    assert result["values"]["replacement"]["replaces"] == 1
    assert result["facts"]["superseded"] == 1 and result["facts"]["conflict"] == 0
    assert app.state.store.get_fact(old).superseded_by == newest(app).id


def test_a_role_held_alongside_keeps_both_without_a_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old, result = _role_change(client, app, fake_openai, replacement("both"))
    assert result["values"]["replacement"]["both"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    assert app.state.store.get_fact(old).valid_to is None
    new = newest(app)
    assert "conflict" not in new.flags and new.same_as is None


def test_the_same_role_in_another_language_is_linked(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old, result = _role_change(client, app, fake_openai, replacement("same"))
    assert result["values"]["replacement"]["same"] == 1
    assert newest(app).same_as == old and result["facts"]["conflict"] == 0


@pytest.mark.parametrize(
    "verdict", [replacement("unsure"), replacement("different"), Reply("not json at all")]
)
def test_unsure_or_anything_else_keeps_the_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, verdict: Reply
) -> None:
    old, result = _role_change(client, app, fake_openai, verdict)
    assert result["values"]["replacement"]["unsure"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert app.state.store.get_fact(old).valid_to is None


def test_a_preference_is_asked_the_replacement_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, "Karim préfère le thé le matin.", "preference.drink", "thé", "fr")
    statement = "Карим теперь предпочитает кофе по утрам."
    fake_openai.script(chat=[said(statement, "preference.drink", "кофе"), replacement("replaces")])
    result = ingest(client, app, statement)
    assert result["facts"]["superseded"] == 1
    assert app.state.store.get_fact(old).valid_to is not None


def test_a_single_valued_key_keeps_its_own_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    karim_fact(app, "Karim habite à Lyon depuis toujours.", "home", "Lyon", "fr")
    statement = "Карим теперь живёт в Париже."
    fake_openai.script(
        chat=[
            said(statement, "home", "Париж"),
            chat_json({"verdicts": [{"pair": 0, "verdict": "different"}]}),
        ]
    )
    result = ingest(client, app, statement)
    system = fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert "name the same thing" in system and "both" not in system
    assert result["facts"]["superseded"] == 1


def test_never_asked_against_an_explicit_fact(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    karim_fact(app, ARCHITECT_FR, "role", "architecte", "fr", origin="explicit")
    fake_openai.script(chat=[said(ACCOUNTANT_RU, "role", "бухгалтер")])
    result = ingest(client, app, ACCOUNTANT_RU)
    assert len(fake_openai.chat_calls) == 1 and result["facts"]["conflict"] == 1


def _same_language(
    client: TestClient, app: FastAPI, fake: FakeOpenAIServer, *answers: Reply
) -> tuple[str, dict[str, Any]]:
    old = karim_fact(app, ACCOUNTANT_FR, "role", "comptable", "fr")
    fake.script(chat=[said(PHOTOGRAPHER_FR, "role", "photographe"), *answers])
    return old, ingest(client, app, PHOTOGRAPHER_FR)


def test_within_one_language_an_added_role_keeps_both(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """ "je suis aussi photographe" after "je suis comptable": the accountant role
    is still true, so it is not closed (SPEC 5.4, single-valued assumption)."""
    old, result = _same_language(client, app, fake_openai, replacement("both"))
    assert len(fake_openai.chat_calls) == 2
    assert '"replacements"' in fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert result["values"]["replacement"]["both"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    assert app.state.store.get_fact(old).valid_to is None


def test_within_one_language_a_replacing_role_supersedes(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old, result = _same_language(client, app, fake_openai, replacement("replaces"))
    assert result["facts"]["superseded"] == 1
    assert app.state.store.get_fact(old).valid_to is not None


def test_within_one_language_without_an_answer_the_conflict_is_visible(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old, result = _same_language(client, app, fake_openai, Reply("no idea"))
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert app.state.store.get_fact(old).valid_to is None
    assert "conflict" in newest(app).flags


def test_within_one_language_a_preference_is_asked_too(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, "Karim préfère le thé le matin.", "preference.drink", "thé", "fr")
    statement = "Karim aime aussi le café après le déjeuner."
    fake_openai.script(chat=[said(statement, "preference.drink", "café"), replacement("both")])
    result = ingest(client, app, statement)
    assert result["values"]["replacement"]["both"] == 1
    assert app.state.store.get_fact(old).valid_to is None


def test_within_one_language_a_single_valued_key_still_closes_without_a_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, "Karim habite à Lyon depuis toujours.", "home", "Lyon", "fr")
    statement = "Karim habite maintenant à Paris."
    fake_openai.script(chat=[said(statement, "home", "Paris")])
    result = ingest(client, app, statement)
    assert len(fake_openai.chat_calls) == 1 and result["facts"]["superseded"] == 1
    assert app.state.store.get_fact(old).valid_to is not None


def test_the_replacement_verdicts_parse_strictly() -> None:
    content = (
        '{"replacements": [{"pair": 0, "verdict": "replaces"}, {"pair": 0, "verdict": "both"},'
        ' {"pair": 1, "verdict": "different"}, {"pair": 5, "verdict": "same"},'
        ' {"pair": 2, "verdict": "both"}], "extra": 1}'
    )
    verdicts, rejected = parse_replacement_verdicts(content, 4)
    assert verdicts == {0: "replaces", 1: "unsure", 2: "both", 3: "unsure"}
    assert rejected == 4  # the duplicate, the unknown word, the pair out of range, the extra key
    assert parse_replacement_verdicts("nope", 2) == ({0: "unsure", 1: "unsure"}, 1)


# --------------------------------------------------------------------------- the source's own words


def _sourced_fact(app: FastAPI, sentence: str, key: str, value: str, language: str) -> str:
    """A stored fact with its source and offsets, as the write path leaves it."""
    store = app.state.store
    node = store.find_node("Karim", VIEW) or store.create_node(
        scopes=["user:alice"], name="Karim", kind="person"
    )
    text = f"User: {sentence}"
    source = store.add_source(scopes=["user:alice"], kind="turn", text=text)
    start = text.index(sentence)
    return str(
        store.add_fact(
            scopes=["user:alice"],
            node_id=node.id,
            statement=sentence.replace(" maintenant", ""),  # what a model writes
            origin="extracted",
            source_id=source.id,
            excerpt=(start, start + len(sentence)),
            attribute=key,
            value=value,
            language=language,
        ).id
    )


def _pairs(fake: FakeOpenAIServer) -> list[dict[str, Any]]:
    content = fake.chat_calls[1].body["messages"][-1]["content"]
    body = content.split("<<<PAIRS>>>\n", 1)[1].rsplit("\n<<<END>>>", 1)[0]
    pairs: list[dict[str, Any]] = json.loads(body)
    return pairs


def test_each_fact_of_a_pair_carries_its_source_words(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The model drops "maintenant" when it writes a statement; the quote keeps it."""
    _sourced_fact(app, "Karim travaille comme architecte.", "role", "architecte", "fr")
    sentence = "Karim travaille maintenant comme comptable."
    reply = said("Karim travaille comme comptable.", "role", "comptable")
    fake_openai.script(chat=[reply, replacement("replaces")])
    # the model quotes the service's label too: it is left out of the pair
    fact = json.loads(reply.content)["facts"][0]
    assert fact["quote"] == "Karim travaille comme comptable."
    post_turn(client, sentence)
    run_all(client, app)
    (pair,) = _pairs(fake_openai)
    assert pair["a"]["quote"] == "Karim travaille comme architecte."
    assert pair["b"]["statement"] == "Karim travaille comme comptable."


def test_the_new_quote_is_the_source_sentence_without_the_label(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    _sourced_fact(app, "Karim travaille comme architecte.", "role", "architecte", "fr")
    sentence = "Karim travaille maintenant comme comptable."
    fact = {
        "subject": "Karim",
        "statement": "Karim travaille comme comptable.",
        "attribute": "role",
        "value": "comptable",
        "quote": f"User: {sentence}",
    }
    fake_openai.script(
        chat=[
            answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact]),
            replacement("replaces"),
        ]
    )
    post_turn(client, sentence)
    run_all(client, app)
    (pair,) = _pairs(fake_openai)
    assert pair["b"]["quote"] == sentence  # "maintenant" is there, "User: " is not


def test_without_retained_sources_the_old_quote_is_absent(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = tier2_settings(
        tmp_path, fake_openai, replacement_question=True, retain_sources=False
    )
    application = create_app(settings, clock=clock)
    try:
        with TestClient(application, headers=auth("all")) as client:
            _sourced_fact(
                application, "Karim travaille comme architecte.", "role", "architecte", "fr"
            )
            sentence = "Karim travaille maintenant comme comptable."
            fake_openai.script(chat=[said(sentence, "role", "comptable"), replacement("replaces")])
            post_turn(client, sentence)
            run_all(client, application)
            (pair,) = _pairs(fake_openai)
            assert "quote" not in pair["a"]
            assert pair["b"]["quote"] == sentence
    finally:
        application.state.store.close()


def test_a_quote_is_bounded() -> None:
    assert PAIR_QUOTE_CHARS == 300
