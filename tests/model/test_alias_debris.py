"""Formatting debris and placeholders are no names (SPEC 5.1): a reply written in
Markdown leaves bold, brackets or quotes around the words a model copies; a
sentence saying a name is not known yet is no alias. Markdown is stripped, an
unbalanced bracket is refused, a name the source does not give is no alias and
no proposal - whatever the language."""

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
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def ingest(client: TestClient, app: FastAPI, text: str) -> dict[str, Any]:
    post_turn(client, text)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def node(app: FastAPI, name: str) -> Any:
    found = app.state.store.find_node(name, VIEW)
    assert found is not None, name
    return found


def names(app: FastAPI) -> list[str]:
    return [r["name"] for r in app.state.store.execute("SELECT name FROM nodes ORDER BY name")]


def proposals(app: FastAPI) -> list[str]:
    return [p.value for p in app.state.store.pending_proposals(VIEW, kind="alias")]


def every_alias(app: FastAPI) -> list[str]:
    return [r["alias"] for r in app.state.store.execute("SELECT alias FROM node_aliases")]


# --------------------------------------------------------------------------- Markdown


def test_markdown_around_a_name_and_an_alias_is_stripped(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "**Émilie**, que tout le monde appelle **Mimi**, aime le thé."
    fake_openai.script(
        chat=[answer(nodes=[{"name": "**Émilie**", "kind": "person", "aliases": ["*Mimi*"]}])]
    )
    result = ingest(client, app, text)
    assert names(app) == ["Émilie"]
    assert node(app, "Émilie").aliases == ("Mimi",)
    assert result["aliases"]["kept"] == 1 and result["aliases"]["malformed"] == 0


def test_markdown_inside_the_source_span_is_not_stored(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The model's copy is plain, the source's span is not: the stored spelling is
    the source's, without its formatting."""
    text = "La réunion du **mardi** 3 mars aura lieu au Orme."
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {
                        "name": "Réunion du mardi",
                        "kind": "event",
                        "aliases": ["mardi 3 mars"],
                    }
                ]
            )
        ]
    )
    ingest(client, app, text)
    assert node(app, "Réunion du mardi").aliases == ("mardi 3 mars",)


def test_the_bold_left_open_in_an_alias_never_reaches_memory(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A shape: ``réunion du **mardi 3 mars``. A description, it
    is proposed - without its debris."""
    text = "La réunion du **mardi 3 mars** aura lieu au Orme."
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {
                        "name": "Réunion",
                        "kind": "event",
                        "aliases": ["réunion du **mardi 3 mars"],
                    }
                ]
            )
        ]
    )
    ingest(client, app, text)
    assert proposals(app) == ["réunion du mardi 3 mars"]
    assert not any("*" in a for a in every_alias(app))


# --------------------------------------------------------------------------- brackets


def test_an_alias_with_an_unclosed_parenthesis_is_refused(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A shape: ``Le Orme (Orme Vallon``. Two repairs are possible ("Le
    Orme (Orme Vallon)", "Le Orme"): none is made, nothing is proposed."""
    text = "On répète au Le Orme (Orme Vallon) mardi."
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {
                        "name": "Le Orme",
                        "kind": "place",
                        "aliases": ["Le Orme (Orme Vallon", "Orme Vallon"],
                    }
                ]
            )
        ]
    )
    result = ingest(client, app, text)
    assert node(app, "Le Orme").aliases == ("Orme Vallon",)
    assert proposals(app) == []
    assert result["aliases"]["malformed"] == 1 and result["aliases"]["kept"] == 1


def test_a_source_span_left_unbalanced_is_refused(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The model dropped the bracket, the source holds it and never closes it: the
    span the alias would be stored as is debris."""
    text = "Rendez-vous au Le Orme, ou Orme (Vallon, mardi soir."
    fake_openai.script(
        chat=[answer(nodes=[{"name": "Le Orme", "kind": "place", "aliases": ["Orme Vallon"]}])]
    )
    result = ingest(client, app, text)
    assert node(app, "Le Orme").aliases == ()
    assert proposals(app) == []
    assert result["aliases"]["malformed"] == 1


def test_a_balanced_bracket_in_a_name_is_kept_whole(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "L'agence Lyon (bureau) ouvre lundi, dit O'Brien."
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {"name": "Agence de Lyon", "kind": "place", "aliases": ["Lyon (bureau)"]},
                    {"name": "O'Brien", "kind": "person"},
                ]
            )
        ]
    )
    result = ingest(client, app, text)
    assert node(app, "Agence de Lyon").aliases == ("Lyon (bureau)",)
    assert "O'Brien" in names(app)
    assert result["aliases"]["malformed"] == 0


def test_a_node_name_a_subject_or_an_edge_end_with_an_unclosed_bracket_is_refused(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Karim dirige Le Orme (Orme Vallon) depuis mars."
    debris = "Le Orme (Orme Vallon"
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Karim", "kind": "person"}, {"name": debris, "kind": "place"}],
                facts=[
                    {
                        "subject": debris,
                        "statement": "Le Orme (Orme Vallon) ouvre depuis mars.",
                        "quote": text,
                    }
                ],
                edges=[{"src": "Karim", "dst": debris, "type": "manages", "quote": text}],
            )
        ]
    )
    result = ingest(client, app, text)
    assert names(app) == ["Karim"]
    assert result["rejected"]["nodes"] == 1
    assert result["rejected"]["facts"] == 1 and result["rejected"]["edges"] == 1


# --------------------------------------------------------------------------- placeholders


def test_a_name_the_model_says_is_unknown_is_no_alias_and_no_proposal(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The structural signal: ``null`` in ``aliases`` - the source says a name
    exists and does not give it."""
    text = "Marie-Anne a un nom de plume ; il n'est pas encore choisi."
    fake_openai.script(
        chat=[answer(nodes=[{"name": "Marie-Anne", "kind": "person", "aliases": [None]}])]
    )
    result = ingest(client, app, text)
    assert node(app, "Marie-Anne").aliases == ()
    assert proposals(app) == []
    assert result["aliases"]["unknown"] == 1 and result["aliases"]["proposed"] == 0


@pytest.mark.parametrize(
    ("text", "name", "placeholder"),
    [
        (
            "Marie-Anne a un nom de plume, il reste à choisir.",
            "Marie-Anne",
            "nom de plume à choisir plus tard",
        ),
        (
            "У Ивана есть псевдоним.",
            "Иван",
            "псевдоним пока не выбран",
        ),
    ],
)
def test_a_description_in_the_models_own_words_is_no_proposal(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    text: str,
    name: str,
    placeholder: str,
) -> None:
    """A placeholder the model wrote in its own words: a description the source
    does not hold. Read by its shape and its absence from the source, never by
    a list of words of one language."""
    fake_openai.script(
        chat=[answer(nodes=[{"name": name, "kind": "person", "aliases": [placeholder]}])]
    )
    result = ingest(client, app, text)
    assert node(app, name).aliases == ()
    assert proposals(app) == []
    assert result["aliases"]["descriptive"] == 1 and result["aliases"]["proposed"] == 0


def test_the_extraction_prompt_offers_null_for_a_name_not_given(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[answer()])
    ingest(client, app, "Karim prépare le café ce matin pour tout le monde.")
    system = fake_openai.chat_calls[0].body["messages"][0]["content"]
    assert "without Markdown" in system and "write null for it in aliases" in system
