"""Undetermined languages through the write path (P12, SPEC 5.4): when either
statement is undetermined, the scripts decide - another script is another
language (the path across languages), one script is one language, as before;
a Latin statement for a Japanese source is a mismatch, asked again, then
stored flagged."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply
from tests.helpers import auth
from tests.model.test_values_pipeline import ingest, karim_fact, newest, said, verdict
from tests.tier2 import answer, tier2_settings


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def test_an_undetermined_statement_in_another_script_does_not_close_the_value(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A Latin statement against a Russian one is surely another language: the
    path across languages, where a preference gets the replacement question
    (an invalid answer here: the conflict stays, nothing is closed)."""
    old = karim_fact(app, "Карим предпочитает чай.", "preference.drink", "чай", "ru")
    fake_openai.script(chat=[said("Karim prefers tea.", "preference.drink", "tea")])
    result = ingest(client, app, "Karim prefers tea.")
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert app.state.store.get_fact(old).valid_to is None
    assert len(fake_openai.chat_calls) == 2 and result["values"]["replacement"]["unsure"] == 1


def test_within_one_script_an_undetermined_statement_is_one_language_as_before(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Measured, not convenient (SPEC 5.4): an undetermined Latin statement and a
    French one are one language, so a single-valued key closes without a
    question, as before."""
    old = karim_fact(app, "Karim habite à Paris.", "home", "Paris", "fr")
    fake_openai.script(chat=[said("Karim lives in Lyon.", "home", "Lyon")])
    result = ingest(client, app, "Karim lives in Lyon.")
    assert len(fake_openai.chat_calls) == 1 and result["facts"]["superseded"] == 1
    assert app.state.store.get_fact(old).superseded_by == newest(app).id


def test_two_undetermined_statements_in_one_script_still_supersede(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, "Karim lives in Paris.", "home", "Paris", None)  # type: ignore[arg-type]
    fake_openai.script(chat=[said("Karim lives in Lyon.", "home", "Lyon")])
    result = ingest(client, app, "Karim lives in Lyon.")
    assert result["facts"]["superseded"] == 1
    assert app.state.store.get_fact(old).superseded_by == newest(app).id
    assert newest(app).language is None  # D5: no borrowed language


def test_undetermined_against_another_script_takes_the_path_across_languages(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, "Карим живёт в Париже.", "home", "Париж", "ru")
    fake_openai.script(chat=[said("Karim lives in Lyon.", "home", "Lyon"), verdict("different")])
    result = ingest(client, app, "Karim lives in Lyon.")
    assert len(fake_openai.chat_calls) == 2  # home is single-valued: adjudicated
    assert result["values"]["adjudicated"]["different"] == 1
    assert result["facts"]["superseded"] == 1
    assert app.state.store.get_fact(old).valid_to is not None


def _kenji(statement: str) -> Reply:
    fact: dict[str, Any] = {
        "subject": "Kenji",
        "statement": statement,
        "attribute": "role",
        "value": "journalist",
        "quote": "健二は記者として働いている",
    }
    return answer(nodes=[{"name": "Kenji", "kind": "person"}], facts=[fact])


def test_p12_a_latin_statement_for_a_japanese_source_is_asked_again_then_flagged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[_kenji("Kenji works as a journalist."), _kenji("Kenji works as a journalist.")]
    )
    result = ingest(client, app, "健二は記者として働いている。")
    assert len(fake_openai.chat_calls) == 2 and result["retries"] == 1
    assert result["facts"]["written"] == 1 and result["facts"]["language_mismatch"] == 1
    assert "language_mismatch" in newest(app).flags


def test_p12_the_known_hole_a_same_script_translation_is_not_judged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Documented, not fixed: "Karim prefers tea." for a French source is
    undetermined and in the same script; nothing flags it (SPEC P12)."""
    fake_openai.script(chat=[said("Karim prefers tea.", "preference.drink", "tea")])
    result = ingest(client, app, "Karim préfère le thé depuis toujours.")
    assert result["facts"]["language_mismatch"] == 0 and result["retries"] == 0
