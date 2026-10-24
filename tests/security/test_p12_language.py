"""P12 - memory keeps the language it was told in. The target language is stated
in every writing prompt; a fact or a summary the fake model writes in another
language is rejected and asked for again once, then stored with a
``language_mismatch`` flag rather than silently. French and a non-Latin script
(Russian) are the sources here, plus a mixed one (SPEC 13.2)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import Settings
from iifa.ingest.prompts import extraction_messages, summary_messages
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply
from tests.fixtures.corpus import SAMPLES
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, end_session, post_turn, run_all, tier2_settings

pytestmark = pytest.mark.security

VIEW = Visibility.of_scope(SCOPE)


@pytest.fixture
def settings(tmp_path: Path, fake_openai: FakeOpenAIServer) -> Settings:
    return tier2_settings(tmp_path, fake_openai)


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def system_prompt(fake: FakeOpenAIServer, index: int) -> str:
    return str(fake.chat_calls[index].messages[0]["content"])


FRENCH_QUOTE = "dirige la planification de l'installation du projet solaire"
ENGLISH_DRIFT = "Karim leads the installation planning of the solar project."
FRENCH_FACT = "Karim dirige la planification de l'installation du projet solaire."


def fact(statement: str, quote: str = FRENCH_QUOTE) -> dict[str, Any]:
    return {"subject": "Karim", "statement": statement, "quote": quote}


def test_p12_the_target_language_is_stated_and_a_drift_is_retried_then_flagged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact(ENGLISH_DRIFT)]),
            answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact(ENGLISH_DRIFT)]),
        ]
    )
    post_turn(client, SAMPLES["fr"], "Noté.")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["calls"] == 2 and result["retries"] == 1
    assert result["languages"] == ["fr"] and result["facts"]["language_mismatch"] == 1
    assert "in French" in system_prompt(fake_openai, 0)
    assert "Never translate" in system_prompt(fake_openai, 0)
    assert "wrong language" in system_prompt(fake_openai, 1) and "fr" in system_prompt(
        fake_openai, 1
    )
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None
    stored = store.facts([karim.id], VIEW)[0]
    assert stored.statement == ENGLISH_DRIFT and stored.language == "en"
    assert stored.flags == ("language_mismatch",)
    assert store.get_source(stored.source_id or "").language == "fr"
    recalled = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "solar project", "budget_tokens": 800}
    ).json()
    assert recalled["items"][0]["flags"] == ["language_mismatch"]


def test_p12_a_correct_retry_is_stored_unflagged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact(ENGLISH_DRIFT)]),
            answer(nodes=[{"name": "Karim", "kind": "person"}], facts=[fact(FRENCH_FACT)]),
        ]
    )
    post_turn(client, SAMPLES["fr"], "Noté.")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["calls"] == 2 and result["facts"]["language_mismatch"] == 0
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None
    stored = store.facts([karim.id], VIEW)[0]
    assert stored.statement == FRENCH_FACT and stored.language == "fr" and stored.flags == ()


def test_p12_a_non_latin_source_keeps_its_script(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    russian = "Карим руководит планированием установки солнечного проекта."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Карим", "kind": "person"}],
                facts=[
                    {"subject": "Карим", "statement": russian, "quote": "руководит планированием"},
                    {
                        "subject": "Карим",
                        "statement": "Karim leads the planning.",
                        "quote": "руководит",
                    },
                ],
            ),
            answer(
                nodes=[{"name": "Карим", "kind": "person"}],
                facts=[
                    {"subject": "Карим", "statement": russian, "quote": "руководит планированием"},
                    {
                        "subject": "Карим",
                        "statement": "Karim leads the planning.",
                        "quote": "руководит",
                    },
                ],
            ),
        ]
    )
    post_turn(client, SAMPLES["ru"], "Принято.")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["languages"] == ["ru"] and "in Russian" in system_prompt(fake_openai, 0)
    assert result["facts"]["written"] == 2 and result["facts"]["language_mismatch"] == 1
    store = app.state.store
    karim = store.find_node("Карим", VIEW)
    assert karim is not None
    by_statement = {f.statement: f for f in store.facts([karim.id], VIEW)}
    assert by_statement[russian].language == "ru" and by_statement[russian].flags == ()
    drifted = by_statement["Karim leads the planning."]
    assert drifted.language == "en" and drifted.flags == ("language_mismatch",)
    # lexical search in the user's script still finds the kept one
    found = client.post("/v1/recall", json={"scope": SCOPE, "query": "планированием"}).json()
    assert [i["id"] for i in found["items"]] == [by_statement[russian].id]


def test_p12_a_mixed_source_keeps_each_fact_in_the_language_of_its_sentence(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = (
        "Karim leads the solar project since March. "
        "Émilie préfère les réponses courtes. "
        "Карим переехал в Лион."
    )
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    # the source spells Karim in Cyrillic too: the answer lists that
                    # spelling, so "Карим переехал в Лион." names its subject (SPEC 4.2)
                    {"name": "Karim", "kind": "person", "aliases": ["Карим"]},
                    {"name": "Émilie", "kind": "person"},
                ],
                facts=[
                    {
                        "subject": "Karim",
                        "statement": "Karim leads the solar project.",
                        "quote": "leads the solar project",
                    },
                    {
                        "subject": "Émilie",
                        "statement": "Émilie préfère les réponses courtes.",
                        "quote": "préfère les réponses courtes",
                    },
                    {
                        "subject": "Karim",
                        "statement": "Карим переехал в Лион.",
                        "quote": "переехал в Лион",
                    },
                    {
                        "subject": "Karim",
                        "statement": "Karim moved to Lyon.",
                        "quote": "переехал в Лион",
                    },
                ],
            )
        ]
        * 2
    )
    post_turn(client, text, "ok")
    result = run_all(client, app)["jobs"][0]["result"]
    prompt = system_prompt(fake_openai, 0)
    assert "mixes" in prompt and "English" in prompt and "French" in prompt and "Russian" in prompt
    assert result["facts"]["written"] == 4 and result["facts"]["language_mismatch"] == 1
    store = app.state.store
    flagged = [f for f in store.facts_by_id([], VIEW)]  # placeholder for typing
    assert flagged == []
    all_facts = {
        f.statement: f
        for node in (store.find_node("Karim", VIEW), store.find_node("Émilie", VIEW))
        if node is not None
        for f in store.facts([node.id], VIEW)
    }
    assert all_facts["Karim leads the solar project."].flags == ()
    assert all_facts["Émilie préfère les réponses courtes."].language == "fr"
    assert all_facts["Карим переехал в Лион."].flags == ()
    assert all_facts["Karim moved to Lyon."].flags == ("language_mismatch",)


def test_p12_a_non_latin_source_gets_its_summary_in_its_script(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The summary half of P12 for a non-Latin script: an English summary of a
    Russian subject is refused and asked for again; the Russian one is kept."""
    russian = "Карим руководит планированием установки солнечного проекта."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Карим", "kind": "person"}],
                facts=[
                    {"subject": "Карим", "statement": russian, "quote": "руководит планированием"}
                ],
            ),
            Reply("Karim leads the planning of the solar installation."),
            Reply("Карим руководит планированием солнечного проекта."),
        ]
    )
    post_turn(client, SAMPLES["ru"], "Принято.", session_id="s1")
    run_all(client, app)
    end_session(client, "s1")
    summaries = run_all(client, app)["jobs"][0]["result"]["summaries"]
    assert summaries["refreshed"] == 1 and summaries["language_mismatch"] == 0
    assert summaries["calls"] == 2
    assert "not in the required language" in system_prompt(fake_openai, 2)
    karim = app.state.store.find_node("Карим", VIEW)
    assert karim is not None
    assert karim.summary == "Карим руководит планированием солнечного проекта."
    assert karim.language == "ru" and karim.flags == ()


def test_p12_summaries_are_written_in_the_language_of_the_facts(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(
                nodes=[
                    {"name": "Karim", "kind": "person"},
                    {"name": "Projet solaire", "kind": "project"},
                ],
                facts=[
                    fact(FRENCH_FACT),
                    {
                        "subject": "Projet solaire",
                        "statement": "Le projet solaire est dirigé par Karim.",
                        "quote": "projet solaire",
                    },
                ],
            ),
            # Karim: English first, French on retry; the project: English twice
            Reply("Karim leads the installation planning of the solar project."),
            Reply("Karim dirige la planification de l'installation du projet solaire."),
            Reply("The solar project is led by Karim."),
            Reply("Still an English sentence about the project."),
        ]
    )
    post_turn(client, SAMPLES["fr"], "Noté.", session_id="s1")
    run_all(client, app)
    end_session(client, "s1")
    report = run_all(client, app)
    summaries = report["jobs"][0]["result"]["summaries"]
    assert summaries == {
        "refreshed": 2,
        "language_mismatch": 1,
        "truncated": 0,
        "skipped": 0,
        "calls": 4,
    }
    assert "in French" in system_prompt(fake_openai, 1)
    assert "not in the required language" in system_prompt(fake_openai, 2)
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    project = store.find_node("Projet solaire", VIEW)
    assert karim is not None and project is not None
    assert karim.summary.startswith("Karim dirige") and karim.language == "fr" and karim.flags == ()
    assert project.summary == "Still an English sentence about the project."
    assert project.language == "en" and project.flags == ("language_mismatch",)
    view = client.get(f"/v1/nodes/{project.id}", params={"scope": SCOPE}).json()
    assert view["flags"] == ["language_mismatch"] and view["language"] == "en"
    # the profile block carries the (flagged) summary as data, one line
    block = client.post("/v1/session/start", json={"scope": SCOPE, "budget_tokens": 2000}).json()
    names = {i["name"]: i for i in block["items"] if i["kind"] == "node"}
    assert names["Karim"]["summary"].startswith("Karim dirige")


def test_p12_a_scope_may_force_its_language(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = tier2_settings(tmp_path, fake_openai, language_overrides=(("user:alice", "fr"),))
    app = create_app(settings, clock=clock)
    with TestClient(app, headers=auth("all")) as client:
        fake_openai.script(
            chat=[
                answer(
                    nodes=[{"name": "Karim", "kind": "person"}],
                    facts=[fact("Karim leads the planning.", "leads")],
                ),
                answer(
                    nodes=[{"name": "Karim", "kind": "person"}],
                    facts=[fact("Karim leads the planning.", "leads")],
                ),
            ]
        )
        post_turn(
            client,
            "Karim leads the planning of the solar project.",
            "ok",
            scope="user:alice/agent:coder",
        )
        result = run_all(client, app)["jobs"][0]["result"]
        assert result["languages"] == ["fr"] and result["facts"]["language_mismatch"] == 1
        prompt = system_prompt(fake_openai, 0)
        assert "in French" in prompt and "this scope requires it" in prompt
    app.state.store.close()


@pytest.mark.parametrize("code", ["en", "fr", "ru"])
def test_p12_the_language_line_never_contradicts_an_english_target(code: str) -> None:
    """The drift warning "not even into English" is for every target but English:
    an English source told not to write English would blur what is measured."""
    extraction = extraction_messages("x", languages=[code])[0]["content"]
    summary = summary_messages("Karim", "person", ["x"], language=code)[0]["content"]
    for prompt in (extraction, summary):
        assert "Never translate into another language" in prompt
        assert ("not even into English" in prompt) is (code != "en")


def test_the_extraction_schema_asks_for_statements_that_name_their_subject() -> None:
    """SPEC 4.2: a statement names its subject. Asking only for "one short
    sentence" lets qwen2.5:7b write "leads the installation planning" for Karim,
    and a retry note alone does not correct it."""
    system = extraction_messages("x", languages=["fr"])[0]["content"]
    assert '"statement": "one complete sentence that names its subject"' in system
    assert 'Each statement names its subject (e.g. "Karim leads…", never "leads…").' in system


def test_the_extraction_schema_asks_for_the_source_spelling_of_a_subject() -> None:
    """SPEC 4.2: a subject is named in the statement's script only; the base prompt
    asks for the source's spelling, never for spellings among the aliases (that
    makes models stuff aliases: only the retry note asks it)."""
    system = extraction_messages("x", languages=["fr"])[0]["content"]
    assert "In the statement, write the subject as the source writes it." in system
    assert "list that spelling" not in system
