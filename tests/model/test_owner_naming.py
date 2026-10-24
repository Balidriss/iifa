"""The owner's name across scripts (SPEC 4.2): for the naming check only, an
owner fact names its subject when the statement holds a form of the owner's
declared name that transliterates to it - provided the subject is exactly the
declared spelling and the quote lies in the owner's part. It confirms a subject
already fixed; it never fixes one, and an adapted form ("Алиса" for Alice) is
not helped."""

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
from tests.model.test_speaker import facts, ingest, node_names, speaker_says
from tests.tier2 import SCOPE, post_turn, run_all, tier2_settings


def _client(
    tmp_path: Path, fake: FakeOpenAIServer, clock: FakeClock, name: str
) -> Iterator[tuple[TestClient, FastAPI]]:
    # no alias in another script: the owner did not foresee every one
    settings = tier2_settings(tmp_path, fake, owners=((SCOPE, name, ()),))
    app = create_app(settings, clock=clock)
    try:
        with TestClient(app, headers=auth("all")) as client:
            yield client, app
    finally:
        app.state.store.close()


@pytest.fixture
def samir(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> Iterator[tuple[TestClient, FastAPI]]:
    yield from _client(tmp_path, fake_openai, clock, "Samir")


@pytest.fixture
def alice(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> Iterator[tuple[TestClient, FastAPI]]:
    yield from _client(tmp_path, fake_openai, clock, "Alice")


def _owner_id(app: FastAPI, name: str) -> Any:
    node = app.state.store.find_node(name, Visibility.of_scope(SCOPE))
    return node.id if node is not None else None


def test_a_transliterated_owner_name_names_the_owner(
    samir: tuple[TestClient, FastAPI], fake_openai: FakeOpenAIServer
) -> None:
    client, app = samir
    fake_openai.script(
        chat=[speaker_says("Samir", "Самир живёт в Лионе.", "Лион", "Я живу в Лионе")]
    )
    result = ingest(client, app, "Я живу в Лионе.")
    assert result["facts"]["written"] == 1, result["facts"]
    (fact,) = facts(app)
    assert fact.node_id == _owner_id(app, "Samir") and fact.provenance == "owner"


def test_the_declared_spelling_inside_a_statement_in_another_script_names_the_owner(
    alice: tuple[TestClient, FastAPI], fake_openai: FakeOpenAIServer
) -> None:
    client, app = alice
    fake_openai.script(
        chat=[
            speaker_says(
                "Alice", "Alice переехала в Монреаль.", "Монреаль", "Я переехала в Монреаль"
            )
        ]
    )
    result = ingest(client, app, "Я переехала в Монреаль.")
    assert result["facts"]["written"] == 1, result["facts"]
    assert result["facts"]["language_mismatch"] == 0  # still a Russian statement (P12)
    assert facts(app)[0].node_id == _owner_id(app, "Alice")


def test_an_adapted_form_that_does_not_transliterate_is_not_helped(
    alice: tuple[TestClient, FastAPI], fake_openai: FakeOpenAIServer
) -> None:
    """ "Алиса" is the Russian form of Alice, not a transliteration (alisa / alice):
    the comparison says unknown, so the statement does not name the owner."""
    client, app = alice
    statement = "Алиса живёт в Монреале."
    fake_openai.script(chat=[speaker_says("Alice", statement, "Монреаль", "Я живу в Монреале")] * 2)
    result = ingest(client, app, "Я живу в Монреале.")
    assert result["facts"]["written"] == 0
    assert facts(app) == []


def test_transliteration_never_chooses_the_subject(
    alice: tuple[TestClient, FastAPI], fake_openai: FakeOpenAIServer
) -> None:
    """A third party called Alisa, spoken of in Russian, stays Alisa's."""
    client, app = alice
    fake_openai.script(
        chat=[speaker_says("Alisa", "Алиса живёт в Казани.", "Казань", "Алиса живёт в Казани")]
    )
    result = ingest(client, app, "Алиса живёт в Казани.")
    assert result["facts"]["speaker_owner"] == 0
    for fact in facts(app):
        assert fact.node_id != _owner_id(app, "Alice")
    assert "Alice" not in node_names(app)


def test_only_the_exact_declared_spelling_as_subject_is_helped(
    samir: tuple[TestClient, FastAPI], fake_openai: FakeOpenAIServer
) -> None:
    client, app = samir
    fake_openai.script(
        chat=[speaker_says("Самир", "Самир живёт в Лионе.", "Лион", "Я живу в Лионе")]
    )
    ingest(client, app, "Я живу в Лионе.")
    assert all(f.node_id != _owner_id(app, "Samir") for f in facts(app))


def test_a_quote_outside_the_owner_part_is_not_helped(
    samir: tuple[TestClient, FastAPI], fake_openai: FakeOpenAIServer
) -> None:
    client, app = samir
    fake_openai.script(
        chat=[speaker_says("Samir", "Самир живёт в Лионе.", "Лион", "Самир живёт в Лионе")] * 2
    )
    post_turn(client, "Read this.", external=[{"kind": "web", "text": "Самир живёт в Лионе."}])
    jobs = run_all(client, app)["jobs"]
    result = next(j["result"] for j in jobs if j["kind"] == "turn")
    assert result["facts"]["written"] == 0 and result["facts"]["proposed"] == 0
    assert result["facts"]["cross_script"] + result["facts"]["incomplete"] == 1
