"""A value's comparison form through the write path: switched on,
the extraction prompt asks for a ``value_norm`` and the service keeps it beside
the fact; across languages, two values whose forms compare the same are
linked without asking the model. A form only ever links: forms that differ
prove nothing, a value proved different is not overruled, and within one
language nothing changes."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest.prompts import extraction_messages
from iifa.ingest.schema import parse_answer
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)


def _app(tmp_path: Path, fake: FakeOpenAIServer, clock: FakeClock, **overrides: Any) -> FastAPI:
    return create_app(tier2_settings(tmp_path, fake, **overrides), clock=clock)


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = _app(tmp_path, fake_openai, clock, value_norm=True)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def karim_fact(
    app: FastAPI, statement: str, attribute: str, value: str, language: str, **extra: Any
) -> str:
    store = app.state.store
    node = store.find_node("Karim", VIEW) or store.create_node(
        scopes=[SCOPE], name="Karim", kind="person", aliases=["Карим"]
    )
    return str(
        store.add_fact(
            scopes=[SCOPE],
            node_id=node.id,
            statement=statement,
            origin=extra.pop("origin", "extracted"),
            attribute=attribute,
            value=value,
            language=language,
            **extra,
        ).id
    )


def said(
    statement: str, attribute: str, value: str, norm: str | None, valid_from: str | None = None
) -> Reply:
    fact: dict[str, Any] = {
        "subject": "Karim",
        "statement": statement,
        "attribute": attribute,
        "value": value,
        "quote": statement,
    }
    if norm is not None:
        fact["value_norm"] = norm
    if valid_from:
        fact["valid_from"] = valid_from
    return answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact])


def verdict(word: str, key: str = "verdicts") -> Reply:
    return chat_json({key: [{"pair": 0, "verdict": word}]})


def ingest(client: TestClient, app: FastAPI, text: str) -> dict[str, Any]:
    post_turn(client, text)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def newest(app: FastAPI) -> Any:
    rows = app.state.store.execute(
        "SELECT id FROM facts ORDER BY created_at DESC, id DESC"
    ).fetchall()
    return app.state.store.get_fact(rows[0]["id"])


LONDON_FR = "Karim habite à Londres avec toute sa famille."
LONDON_EN = "Karim lives in London with his whole family these days."
PARIS_RU = "Карим переехал в Париж вместе со всей своей семьёй."


# --------------------------------------------------------------------------- the prompt and the schema


def test_the_prompt_asks_for_the_form_only_when_switched_on() -> None:
    on = extraction_messages("Karim habite à Lyon.", languages=["fr"], value_norm=True)
    off = extraction_messages("Karim habite à Lyon.", languages=["fr"])
    assert "value_norm" in on[0]["content"] and "value_norm" not in off[0]["content"]
    # P12: the value itself stays in the language of the source
    assert "in French" in on[0]["content"]


def test_the_schema_bounds_the_form_and_still_refuses_unknown_fields() -> None:
    fact = {"subject": "Karim", "statement": "Karim habite à Lyon.", "value": "Lyon"}
    ok = parse_answer(json.dumps({"facts": [{**fact, "value_norm": "  lyon  "}]}))
    assert ok.facts[0].value_norm == "lyon"
    long = parse_answer(json.dumps({"facts": [{**fact, "value_norm": "x" * 201}]}))
    assert long.facts[0].value_norm is None and long.rejected.facts == 0  # the fact stays
    other = parse_answer(json.dumps({"facts": [{**fact, "value_canon": "lyon"}]}))
    assert other.facts == [] and other.rejected.facts == 1


# --------------------------------------------------------------------------- linked by the form


def test_an_exonym_with_the_same_form_is_linked_without_asking(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[said(LONDON_FR, "home", "Londres", "London")])
    ingest(client, app, LONDON_FR)
    old = newest(app).id
    fake_openai.script(chat=[said(LONDON_EN, "home", "London", "london")])
    result = ingest(client, app, LONDON_EN)
    assert len(fake_openai.chat_calls) == 2  # no adjudication
    assert result["values"]["norm"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    fact = newest(app)
    assert fact.same_as == old and "conflict" not in fact.flags
    assert app.state.store.get_fact(old).valid_to is None
    assert "value_norm" in fake_openai.chat_calls[0].messages[0]["content"]


def test_a_role_in_another_language_with_the_same_form_is_not_asked(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Role may hold several values: the form still links a restatement, so
    the replacement question is not needed."""
    old = karim_fact(
        app, "Karim travaille comme comptable.", "role", "comptable", "fr", value_norm="accountant"
    )
    english = "Karim works as an accountant at the firm."
    fake_openai.script(chat=[said(english, "role", "accountant", "Accountant")])
    result = ingest(client, app, english)
    assert len(fake_openai.chat_calls) == 1
    assert result["values"]["norm"] == 1 and result["values"]["replacement"]["unsure"] == 0
    assert newest(app).same_as == old


def test_an_earlier_value_with_the_same_form_is_linked_not_a_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(
        app, LONDON_FR, "home", "Londres", "fr", value_norm="london", valid_from="2026-05"
    )
    fake_openai.script(chat=[said(LONDON_EN, "home", "London", "london", valid_from="2026-01")])
    result = ingest(client, app, LONDON_EN)
    assert result["values"]["norm"] == 1 and result["facts"]["conflict"] == 0
    assert newest(app).same_as == old


# --------------------------------------------------------------------------- what a form never does


def test_forms_that_differ_prove_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The pair goes to the adjudication as without forms; unsure keeps both."""
    old = karim_fact(app, LONDON_FR, "home", "Londres", "fr", value_norm="london")
    fake_openai.script(chat=[said(PARIS_RU, "home", "Париж", "paris"), verdict("unsure")])
    result = ingest(client, app, PARIS_RU)
    assert len(fake_openai.chat_calls) == 2
    assert result["values"]["norm"] == 0 and result["values"]["adjudicated"]["unsure"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert app.state.store.get_fact(old).valid_to is None


def test_forms_one_letter_apart_are_not_the_same(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(
        app, "Karim travaille comme peintre.", "role", "peintre", "fr", value_norm="painter"
    )
    russian = "Карим теперь работает печатником."
    fake_openai.script(
        chat=[said(russian, "role", "печатник", "Printer"), verdict("replaces", "replacements")]
    )
    result = ingest(client, app, russian)
    assert result["values"]["norm"] == 0 and result["values"]["replacement"]["replaces"] == 1
    assert app.state.store.get_fact(old).valid_to is not None


def test_a_form_never_overrules_a_value_proved_different(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(
        app,
        "La date limite du projet de Karim est fixée à mars 2026.",
        "date.deadline",
        "mars 2026",
        "fr",
        value_norm="2026-03",
        valid_from="2026-01",
    )
    english = "Karim's project deadline has moved to April 2026 after the review."
    fake_openai.script(chat=[said(english, "date.deadline", "April 2026", "2026-03", "2026-02")])
    result = ingest(client, app, english)
    assert result["values"]["norm"] == 0 and result["values"]["different"] == 1
    assert app.state.store.get_fact(old).valid_to is not None


def test_within_one_language_the_form_changes_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, "Karim habite à Lyon.", "home", "Lyon", "fr", value_norm="lyon")
    french = "Karim a déménagé à Paris le mois dernier."
    fake_openai.script(chat=[said(french, "home", "Paris", "lyon")])
    result = ingest(client, app, french)
    assert result["values"]["norm"] == 0 and result["facts"]["superseded"] == 1
    assert app.state.store.get_fact(old).superseded_by == newest(app).id


def test_an_explicit_fact_keeps_its_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """P6: an explicit fact carries no form, and none is read against it."""
    karim_fact(app, LONDON_FR, "home", "Londres", "fr", origin="explicit")
    fake_openai.script(chat=[said(LONDON_EN, "home", "London", "london")])
    result = ingest(client, app, LONDON_EN)
    assert result["values"]["norm"] == 0 and result["facts"]["conflict"] == 1


def test_one_side_without_a_form_proves_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    karim_fact(app, LONDON_FR, "home", "Londres", "fr")
    fake_openai.script(chat=[said(LONDON_EN, "home", "London", "london"), verdict("unsure")])
    result = ingest(client, app, LONDON_EN)
    assert result["values"]["norm"] == 0 and result["facts"]["conflict"] == 1


# --------------------------------------------------------------------------- switched off


def test_switched_off_the_form_is_neither_asked_nor_kept_nor_read(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = _app(tmp_path, fake_openai, clock)
    try:
        with TestClient(application, headers=auth("all")) as off:
            karim_fact(application, LONDON_FR, "home", "Londres", "fr", value_norm="london")
            fake_openai.script(
                chat=[said(LONDON_EN, "home", "London", "london"), verdict("unsure")]
            )
            result = ingest(off, application, LONDON_EN)
            assert "value_norm" not in fake_openai.chat_calls[0].messages[0]["content"]
            assert len(fake_openai.chat_calls) == 2  # asked, as without forms
            assert result["values"]["norm"] == 0 and result["facts"]["conflict"] == 1
            fact = newest(application)
            assert application.state.store.value_norms([fact.id]) == {}
    finally:
        application.state.store.close()
