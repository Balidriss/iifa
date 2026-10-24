"""Names the write path will not take as they come (SPEC 4.2, 5.1): a word that
mixes two spaced scripts never becomes a node, a subject or an alias ("\u0643azan"
is a node qwen3.5:9b proposes on the synthetic corpus); an alias that is a description
rather than a name becomes a proposal, even when the source holds it."""

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

MIXED_CITY = "\u0643azan"  # Arabic kaf, then Latin letters: one word, two scripts
MIXED_NAME = "Yo\u0443ssef"  # a Cyrillic u inside a Latin name
MIXED_ALIAS = "K\u0430rim"  # a Cyrillic a inside a Latin name


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


def names(app: FastAPI) -> list[str]:
    return [r["name"] for r in app.state.store.execute("SELECT name FROM nodes ORDER BY name")]


def node(app: FastAPI, name: str) -> Any:
    found = app.state.store.find_node(name, Visibility.of_scope(SCOPE))
    assert found is not None, name
    return found


# --------------------------------------------------------------------------- mixed scripts (SPEC 4.2)


def test_a_node_name_mixing_two_scripts_never_becomes_a_node(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Youssef a déménagé à Kazan au printemps."
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {"name": "Youssef", "kind": "person"},
                    {"name": MIXED_CITY, "kind": "place"},
                ],
                facts=[
                    {
                        "subject": "Youssef",
                        "statement": "Youssef a déménagé à Kazan au printemps.",
                        "attribute": "home",
                        "value": "Kazan",
                        "quote": text,
                    }
                ],
                edges=[{"src": "Youssef", "dst": MIXED_CITY, "type": "lives_in", "quote": text}],
            )
        ]
    )
    result = ingest(client, app, text)
    assert names(app) == ["Youssef"]
    assert result["nodes"]["mixed_script"] == 1
    assert result["facts"]["written"] == 1
    assert app.state.store.execute("SELECT count(*) FROM edges").fetchone()[0] == 0


def test_a_subject_mixing_two_scripts_is_asked_again_then_dropped(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fact = {
        "subject": MIXED_NAME,
        "statement": "Youssef habite à Lyon depuis mars.",
        "attribute": "home",
        "value": "Lyon",
        "quote": "Youssef habite à Lyon depuis mars.",
    }
    fake_openai.script(chat=[answer(facts=[fact]), answer(facts=[fact])])
    result = ingest(client, app, "Youssef habite à Lyon depuis mars.")
    assert len(fake_openai.chat_calls) == 2 and result["retries"] == 1
    assert result["facts"]["written"] == 0 and result["facts"]["mixed_script"] == 1
    assert MIXED_NAME in fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert names(app) == []


def test_an_alias_mixing_two_scripts_is_dropped(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Karim prépare le café."
    fake_openai.script(
        chat=[answer(nodes=[{"name": "Karim", "kind": "person", "aliases": [MIXED_ALIAS]}])]
    )
    result = ingest(client, app, text)
    assert node(app, "Karim").aliases == ()
    assert result["aliases"]["proposed"] == 0 and result["aliases"]["invented"] == 1


# --------------------------------------------------------------------------- descriptive aliases (SPEC 5.1)


def test_a_grounded_alias_that_is_a_description_becomes_a_proposal(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    description = "la planification de l'installation du projet solaire"
    text = f"Le projet solaire, c'est {description}, avance bien."
    fake_openai.script(
        chat=[
            answer(nodes=[{"name": "Projet solaire", "kind": "project", "aliases": [description]}])
        ]
    )
    result = ingest(client, app, text)
    assert node(app, "Projet solaire").aliases == ()
    assert result["aliases"]["descriptive"] == 1 and result["aliases"]["proposed"] == 1
    store = app.state.store
    (proposal,) = store.proposals([node(app, "Projet solaire").id], Visibility.of_scope(SCOPE))
    assert proposal.kind == "alias" and proposal.value == description


def test_a_long_full_name_for_a_short_name_is_proposed_too(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Ivan Petrovich Sidorov dirige le projet."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Ivan", "kind": "person", "aliases": ["Ivan Petrovich Sidorov"]}]
            )
        ]
    )
    result = ingest(client, app, text)
    assert node(app, "Ivan").aliases == ()
    assert result["aliases"]["descriptive"] == 1


def test_a_short_grounded_alias_is_still_kept(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Émilie, que tout le monde appelle Mimi, aime le thé."
    fake_openai.script(
        chat=[answer(nodes=[{"name": "Émilie", "kind": "person", "aliases": ["Mimi"]}])]
    )
    result = ingest(client, app, text)
    assert node(app, "Émilie").aliases == ("Mimi",)
    assert result["aliases"]["descriptive"] == 0
