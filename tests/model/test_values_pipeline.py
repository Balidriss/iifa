"""Values across languages through the write path (SPEC 5.4): deterministic
same / different first, the chat model asked only for the unknown ones, on
single-valued keys, when the new fact is later; an equivalence links, a
supersession closes the whole group, and the explicit is never closed (P6)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)
LYON_FR = "Karim habite à Lyon avec toute sa famille depuis le printemps."


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def karim_fact(
    app: FastAPI,
    statement: str,
    attribute: str,
    value: str,
    language: str,
    **extra: Any,
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


def said(statement: str, attribute: str, value: str, valid_from: str | None = None) -> Reply:
    fact: dict[str, Any] = {
        "subject": "Karim",
        "statement": statement,
        "attribute": attribute,
        "value": value,
        "quote": statement,
    }
    if valid_from:
        fact["valid_from"] = valid_from
    return answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact])


def verdict(word: str) -> Reply:
    return chat_json({"verdicts": [{"pair": 0, "verdict": word}]})


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


# --------------------------------------------------------------------------- deterministic


def test_the_same_value_in_another_language_is_linked_not_a_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, LYON_FR, "home", "Lyon", "fr")
    russian = "Карим живёт в Лионе со всей своей семьёй с весны."
    fake_openai.script(chat=[said(russian, "home", "Лион")])
    result = ingest(client, app, russian)
    assert result["facts"]["written"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    assert result["values"]["same"] == 1
    assert len(fake_openai.chat_calls) == 1  # no adjudication for a proved value
    fact = newest(app)
    assert fact.same_as == old and "conflict" not in fact.flags
    assert app.state.store.get_fact(old).valid_to is None


def test_a_later_date_proved_different_supersedes_across_languages(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(
        app,
        "La date limite du projet de Karim est fixée à mars 2026.",
        "date.deadline",
        "mars 2026",
        "fr",
        valid_from="2026-01",
    )
    english = "Karim's project deadline has moved to April 2026 after the review."
    fake_openai.script(chat=[said(english, "date.deadline", "April 2026", "2026-02")])
    result = ingest(client, app, english)
    assert result["facts"]["superseded"] == 1 and result["values"]["different"] == 1
    assert len(fake_openai.chat_calls) == 1
    assert app.state.store.get_fact(old).valid_to is not None


# --------------------------------------------------------------------------- adjudicated


PARIS_RU = "Карим переехал в Париж вместе со всей своей семьёй."


def test_an_unknown_value_is_adjudicated_different_and_supersedes(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, LYON_FR, "home", "Lyon", "fr")
    fake_openai.script(chat=[said(PARIS_RU, "home", "Париж"), verdict("different")])
    result = ingest(client, app, PARIS_RU)
    assert result["values"]["adjudicated"] == {"same": 0, "different": 1, "unsure": 0}
    assert result["facts"]["superseded"] == 1 and result["facts"]["conflict"] == 0
    assert app.state.store.get_fact(old).superseded_by == newest(app).id
    prompt = fake_openai.chat_calls[1].messages
    system, user = prompt[0]["content"], prompt[-1]["content"]
    assert "DATA" in system and '"verdicts"' in system
    pairs = json.loads(user.split("<<<PAIRS>>>")[1].split("<<<END>>>")[0])
    assert pairs == [
        {
            "pair": 0,
            "subject": "Karim",
            "key": "home",
            "a": {"value": "Lyon", "statement": LYON_FR, "language": "fr"},
            "b": {"value": "Париж", "statement": PARIS_RU, "language": "ru"},
        }
    ]


def test_an_exonym_adjudicated_same_is_linked(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, "Karim habite à Londres avec toute sa famille.", "home", "Londres", "fr")
    english = "Karim lives in London with his whole family these days."
    fake_openai.script(chat=[said(english, "home", "London"), verdict("same")])
    result = ingest(client, app, english)
    assert result["values"]["adjudicated"]["same"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    assert newest(app).same_as == old


@pytest.mark.parametrize("reply", [verdict("unsure"), verdict("maybe"), Reply("no json at all")])
def test_unsure_or_invalid_leaves_the_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, reply: Reply
) -> None:
    old = karim_fact(app, LYON_FR, "home", "Lyon", "fr")
    fake_openai.script(chat=[said(PARIS_RU, "home", "Париж"), reply])
    result = ingest(client, app, PARIS_RU)
    assert result["values"]["adjudicated"]["unsure"] == 1
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert app.state.store.get_fact(old).valid_to is None
    assert "conflict" in newest(app).flags


# --------------------------------------------------------------------------- never asked


def test_a_multi_valued_key_gets_the_replacement_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Role may hold several values: it is asked "replaces, or both true?", not
    "the same thing?"; an answer the schema refuses keeps the conflict."""
    karim_fact(app, "Karim travaille comme architecte à plein temps.", "role", "architecte", "fr")
    russian = "Карим также работает преподавателем по вечерам."
    fake_openai.script(chat=[said(russian, "role", "преподаватель")])
    result = ingest(client, app, russian)
    assert len(fake_openai.chat_calls) == 2
    assert '"replacements"' in fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert result["values"]["replacement"]["unsure"] == 1
    assert result["facts"]["conflict"] == 1 and result["facts"]["superseded"] == 0


def test_an_earlier_value_is_never_adjudicated(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    karim_fact(app, LYON_FR, "home", "Lyon", "fr", valid_from="2026-05")
    fake_openai.script(chat=[said(PARIS_RU, "home", "Париж", "2025-01")])
    result = ingest(client, app, PARIS_RU)
    assert len(fake_openai.chat_calls) == 1
    assert result["facts"]["conflict"] == 1 and result["facts"]["superseded"] == 0


@pytest.mark.security
def test_p6_an_explicit_value_is_never_adjudicated_nor_closed(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    old = karim_fact(app, LYON_FR, "home", "Lyon", "fr", origin="explicit")
    fake_openai.script(chat=[said(PARIS_RU, "home", "Париж")])
    result = ingest(client, app, PARIS_RU)
    assert len(fake_openai.chat_calls) == 1
    assert result["facts"]["conflict"] == 1
    assert app.state.store.get_fact(old).valid_to is None


# --------------------------------------------------------------------------- groups


def test_a_supersession_closes_the_whole_group(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """ "Лион", linked to "Lyon", states the same outdated value: a later "Paris"
    closes both, and nothing is asked about the Russian one."""
    root = karim_fact(app, LYON_FR, "home", "Lyon", "fr")
    member = karim_fact(
        app, "Карим живёт в Лионе со всей своей семьёй.", "home", "Лион", "ru", same_as=root
    )
    french = "Karim habite maintenant à Paris avec toute sa famille."
    fake_openai.script(chat=[said(french, "home", "Paris")])
    result = ingest(client, app, french)
    assert len(fake_openai.chat_calls) == 1
    assert result["facts"]["superseded"] == 2
    store = app.state.store
    assert store.get_fact(root).valid_to is not None and store.get_fact(member).valid_to is not None
    assert [f.value for f in store.facts([store.get_fact(root).node_id], VIEW)] == ["Paris"]


def test_a_preference_gets_the_replacement_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Preferences are often multi-valued (tea AND coffee): like role and team
    they are asked "replaces, or both true?"; an invalid answer keeps a visible
    conflict."""
    karim_fact(app, "Karim préfère le thé le matin au bureau.", "preference.drink", "thé", "fr")
    russian = "Карим предпочитает кофе по утрам в офисе."
    fake_openai.script(chat=[said(russian, "preference.drink", "кофе")])
    result = ingest(client, app, russian)
    assert len(fake_openai.chat_calls) == 2
    assert result["values"]["replacement"]["unsure"] == 1
    assert result["facts"]["conflict"] == 1 and result["facts"]["superseded"] == 0


def test_a_duplicate_of_one_group_does_not_spare_another_group_to_close(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The --rephrase case: Paris (ru) and Lyon (fr) both live, left in conflict
    by an unsure adjudication; Lyon is told again. The new fact duplicates Lyon,
    and the adjudication says Paris is a different value: Paris must still be
    closed, or a stale value stays injected."""
    paris = karim_fact(app, "Карим живёт в Париже.", "home", "Париж", "ru")
    lyon = karim_fact(app, LYON_FR, "home", "Lyon", "fr", conflicts_with=paris)
    fake_openai.script(
        chat=[said("Karim habite toujours à Lyon.", "home", "Lyon"), verdict("different")]
    )
    result = ingest(client, app, "Karim habite toujours à Lyon.")
    assert len(fake_openai.chat_calls) == 2  # Paris was adjudicated against the new value
    assert result["facts"]["written"] == 0 and result["facts"]["duplicate"] == 1
    assert result["facts"]["superseded"] == 1
    closed = app.state.store.get_fact(paris)
    assert closed.valid_to is not None and closed.superseded_by == lyon
    assert app.state.store.get_fact(lyon).valid_to is None
