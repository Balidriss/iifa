"""A free-form key that holds one value: a key is a slug the model
chose, and a generic one (``content``, ``info``) may carry two unrelated
statements about one subject. A shared key alone does not prove a
replacement, so with the replacement question on a changed free-form value
is asked "replaces, or both true?" - within one language too. ``both`` keeps
both live without a conflict, ``unsure`` or an invalid answer keeps both with
a conflict, ``replaces`` supersedes, ``same`` keeps the fact held. A value the
comparison proves different (another number in the same form) supersedes
without a question; off, a change within one language supersedes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import answer, post_turn, run_all, tier2_settings

SHELF = "Pantry"
HOLDS = "The Pantry shelf holds 12 jars, about 9 kg."
SPARE = "Spare jars are kept on the Pantry shelf."
HOLDS_MORE = "The Pantry shelf holds 13 jars, about 9 kg."
HOLDS_FR = "L'étagère Pantry contient des bocaux de réserve pour la cuisine."


def _app(tmp_path: Path, fake: FakeOpenAIServer, clock: FakeClock, **switches: Any) -> FastAPI:
    return create_app(tier2_settings(tmp_path, fake, **switches), clock=clock)


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = _app(tmp_path, fake_openai, clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def told(statement: str, value: str, key: str = "content") -> Reply:
    fact = {
        "subject": SHELF,
        "statement": statement,
        "attribute": key,
        "value": value,
        "cardinality": "one",
        "quote": statement,
    }
    return answer(nodes=[{"name": SHELF, "kind": "object"}], facts=[fact])


def replacement(word: str) -> Reply:
    return chat_json({"replacements": [{"pair": 0, "verdict": word}]})


def ingest(client: TestClient, app: FastAPI, fake: FakeOpenAIServer, *replies: Reply) -> Any:
    fake.script(chat=list(replies))
    # the turn says what the scripted extraction answers (its subject is named)
    post_turn(client, json.loads(replies[0].content)["facts"][0]["quote"])
    jobs = run_all(client, app)["jobs"]
    return next(j["result"] for j in jobs if j["kind"] == "turn")


def facts_of(app: FastAPI, key: str = "content") -> list[Any]:
    rows = app.state.store.execute(
        "SELECT id FROM facts WHERE attribute = ? ORDER BY created_at, id", (key,)
    ).fetchall()
    return [app.state.store.get_fact(r["id"]) for r in rows]


def _second(client: TestClient, app: FastAPI, fake: FakeOpenAIServer, verdict: Reply | None) -> Any:
    ingest(client, app, fake, told(HOLDS, "12 jars, about 9 kg"))
    replies = [told(SPARE, "spare jars")] + ([verdict] if verdict else [])
    return ingest(client, app, fake, *replies)


def _both_live(app: FastAPI) -> tuple[Any, Any]:
    old, new = facts_of(app)
    assert old.valid_to is None and old.superseded_by is None
    assert new.valid_to is None and new.superseded_by is None
    return old, new


def test_two_unrelated_statements_under_a_generic_key_both_stay(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A count is not closed by an unrelated statement under the same key."""
    result = _second(client, app, fake_openai, replacement("both"))
    assert len(fake_openai.chat_calls) == 3  # two extractions, one question
    system = fake_openai.chat_calls[2].body["messages"][0]["content"]
    assert "generic key" in system
    assert result["values"]["replacement"]["both"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    _old, new = _both_live(app)
    assert "conflict" not in new.flags


@pytest.mark.parametrize("verdict", [replacement("unsure"), Reply(content="not json"), None])
def test_without_a_clear_answer_both_stay_in_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, verdict: Reply | None
) -> None:
    """When we cannot tell, we do not supersede: unsure, an invalid answer, or
    no answer at all (the fake has nothing scripted) keep both live."""
    result = _second(client, app, fake_openai, verdict)
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    _old, new = _both_live(app)
    assert "conflict" in new.flags


def test_a_new_value_that_replaces_supersedes(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    result = _second(client, app, fake_openai, replacement("replaces"))
    assert result["values"]["replacement"]["replaces"] == 1
    assert result["facts"]["superseded"] == 1
    old, new = facts_of(app)
    assert old.superseded_by == new.id and new.valid_to is None


def test_the_same_value_told_again_keeps_the_fact_held(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    result = _second(client, app, fake_openai, replacement("same"))
    assert result["facts"]["kept_same"] == 1 and result["facts"]["written"] == 0
    (only,) = facts_of(app)
    assert only.statement == HOLDS and only.valid_to is None


def test_a_free_form_change_is_not_asked_the_restatement_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Values sharing words ("jars") are not asked "the same value told again?";
    the replacement question already offers ``same``: one question, not two."""
    _second(client, app, fake_openai, replacement("both"))
    asked = [c.body["messages"][0]["content"] for c in fake_openai.chat_calls[2:]]
    assert len(asked) == 1 and '"replacements"' in asked[0]


def test_another_number_in_the_same_form_supersedes_without_a_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ingest(client, app, fake_openai, told(HOLDS, "12 jars"))
    result = ingest(client, app, fake_openai, told(HOLDS_MORE, "13 jars"))
    assert len(fake_openai.chat_calls) == 2  # proved different: nothing asked
    assert result["facts"]["superseded"] == 1
    old, new = facts_of(app)
    assert old.superseded_by == new.id


def test_an_earlier_value_is_not_asked_and_closes_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ingest(client, app, fake_openai, _dated(HOLDS, "12 jars, about 9 kg", "2026-05-01"))
    result = ingest(client, app, fake_openai, _dated(SPARE, "spare jars", "2026-03-01"))
    assert len(fake_openai.chat_calls) == 2
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    _both_live(app)


def _dated(statement: str, value: str, valid_from: str) -> Reply:
    fact = {
        "subject": SHELF,
        "statement": statement,
        "attribute": "content",
        "value": value,
        "valid_from": valid_from,
        "cardinality": "one",
        "quote": statement,
    }
    return answer(nodes=[{"name": SHELF, "kind": "object"}], facts=[fact])


def test_across_languages_a_free_form_change_is_asked_too(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ingest(client, app, fake_openai, told(HOLDS, "12 jars, about 9 kg"))
    result = ingest(
        client, app, fake_openai, told(HOLDS_FR, "bocaux de réserve"), replacement("both")
    )
    assert len(fake_openai.chat_calls) == 3
    assert result["values"]["replacement"]["both"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    _both_live(app)


def test_a_vocabulary_key_is_unchanged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """``home`` holds one value by definition: a change within one language
    closes without the replacement question."""
    ingest(client, app, fake_openai, told("The Pantry shelf is kept in Lyon.", "Lyon", "home"))
    result = ingest(
        client, app, fake_openai, told("The Pantry shelf is kept in Paris.", "Paris", "home")
    )
    assert len(fake_openai.chat_calls) == 2
    assert result["facts"]["superseded"] == 1


def test_switched_off_a_free_form_change_supersedes_as_before(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = _app(
        tmp_path, fake_openai, clock, replacement_question=False, restatement_question=False
    )
    try:
        with TestClient(application, headers=auth("all")) as off:
            ingest(off, application, fake_openai, told(HOLDS, "12 jars, about 9 kg"))
            result = ingest(off, application, fake_openai, told(SPARE, "spare jars"))
        assert len(fake_openai.chat_calls) == 2
        assert result["facts"]["superseded"] == 1
    finally:
        application.state.store.close()
