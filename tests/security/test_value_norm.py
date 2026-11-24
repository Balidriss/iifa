"""A value's comparison form is for comparison only: no route
shows it, no search finds it, no prompt carries it, and it only ever links -
it closes nothing and never lets external content join what the owner said
(P6). Proven with a canary form written by the fake model."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from iifa.store.models import Fact
from tests.conftest import FakeClock
from tests.dream.conftest import dream, dream_settings, quiet_model
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, end_session, post_turn, run_all, tier2_settings

pytestmark = pytest.mark.security

VIEW = Visibility.of_scope(SCOPE)
CANARY = "zqxnormcanary"
LYON_FR = "Karim habite à Lyon avec toute sa famille."
PARIS_RU = "Карим переехал в Париж вместе со всей своей семьёй."


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai, value_norm=True), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def said(statement: str, value: str, norm: str) -> dict[str, Any]:
    return {
        "subject": "Karim",
        "statement": statement,
        "attribute": "home",
        "value": value,
        "value_norm": norm,
        "quote": statement,
    }


def karim(*facts: dict[str, Any]) -> Reply:
    return answer(
        nodes=[{"name": "Karim", "kind": "person", "aliases": ["Карим"]}], facts=list(facts)
    )


def ingest(client: TestClient, app: FastAPI, text: str, **extra: Any) -> dict[str, Any]:
    post_turn(client, text, **extra)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def facts_of(app: FastAPI) -> list[Fact]:
    rows = app.state.store.execute("SELECT id FROM facts ORDER BY created_at, id").fetchall()
    return [app.state.store.get_fact(r["id"]) for r in rows]


def _canary_memory(client: TestClient, app: FastAPI, fake: FakeOpenAIServer) -> dict[str, Any]:
    """Karim's home in French with the canary form, then a move told in Russian
    (forms differ: adjudicated different, superseded): every reading route has
    something to show."""
    fake.script(chat=[karim(said(LYON_FR, "Lyon", CANARY))])
    first = ingest(client, app, LYON_FR, session_id="s1")
    fake.script(
        chat=[
            karim(said(PARIS_RU, "Париж", f"{CANARY} paris")),
            chat_json({"verdicts": [{"pair": 0, "verdict": "different"}]}),
        ]
    )
    second = ingest(client, app, PARIS_RU, session_id="s1")
    store = app.state.store
    lyon, paris = facts_of(app)
    assert store.value_norms([lyon.id, paris.id]) == {
        lyon.id: CANARY,
        paris.id: f"{CANARY} paris",
    }
    assert lyon.superseded_by == paris.id
    return {"first": first, "second": second, "lyon": lyon, "paris": paris}


def test_the_comparison_form_is_never_shown_by_any_route(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = _canary_memory(client, app, fake_openai)
    lyon, paris = memory["lyon"], memory["paris"]
    node = paris.node_id
    responses = [
        json.dumps(memory["first"]),
        json.dumps(memory["second"]),
        client.post("/v1/recall", json={"scope": SCOPE, "query": "Karim Lyon Париж"}).text,
        client.post(
            "/v1/expand", json={"scope": SCOPE, "ids": [lyon.id, paris.id, node], "level": 2}
        ).text,
        client.get(f"/v1/nodes/{node}", params={"scope": SCOPE}).text,
        client.post("/v1/timeline", json={"scope": SCOPE, "budget_tokens": 4000}).text,
        client.post("/v1/at", json={"scope": SCOPE, "node_id": node, "date": "2030-01-01"}).text,
        client.get("/v1/lookup", params={"scope": SCOPE, "name": "Karim"}).text,
        client.get("/v1/proposals", params={"scope": SCOPE}).text,
        client.post("/v1/session/start", json={"scope": SCOPE, "budget_tokens": 4000}).text,
    ]
    assert all(CANARY not in text for text in responses)
    assert "Lyon" in responses[5]  # the timeline did show the facts
    # the record itself has no field for it: no serialiser, today's or an
    # export's, can carry it
    assert all("norm" not in f.name for f in dataclasses.fields(Fact))
    assert CANARY not in repr(app.state.store.get_fact(lyon.id))


def test_the_comparison_form_is_never_searched(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    _canary_memory(client, app, fake_openai)
    store = app.state.store
    assert store.postings([CANARY], VIEW) == []
    found = client.post("/v1/recall", json={"scope": SCOPE, "query": CANARY}).json()
    assert found["items"] == []


def test_the_comparison_form_is_never_sent_to_a_model(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Extraction (known keys), the value adjudication and the summaries all read
    the facts: none of them carries the form."""
    _canary_memory(client, app, fake_openai)
    fake_openai.script(chat=[Reply("Karim vit à Paris."), Reply("Karim vit à Paris.")])
    end_session(client, "s1")
    run_all(client, app)
    calls = fake_openai.chat_calls
    adjudication = [c for c in calls if '"verdicts"' in c.messages[0]["content"]]
    assert len(adjudication) == 1 and "Lyon" in adjudication[0].messages[-1]["content"]
    assert len(calls) >= 4  # two extractions, one adjudication, the summary
    for call in calls:
        assert CANARY not in json.dumps(call.messages, ensure_ascii=False)


def test_the_comparison_form_is_never_sent_to_the_dream_model(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = create_app(dream_settings(tmp_path, fake_openai), clock=clock)
    try:
        store = application.state.store
        node = store.create_node(scopes=[SCOPE], name="Karim", kind="person")
        for statement, value in ((LYON_FR, "Lyon"), ("Karim lives in Lyon.", "Lyon")):
            store.add_fact(
                scopes=[SCOPE],
                node_id=node.id,
                statement=statement,
                origin="extracted",
                attribute="home",
                value=value,
                value_norm=CANARY,
            )
        fake_openai.set_default(chat=quiet_model())
        with TestClient(application, headers=auth("all")) as dream_client:
            run = dream(dream_client)
        assert run["status"] == "finished"
        assert fake_openai.chat_calls
        for call in fake_openai.chat_calls:
            assert CANARY not in json.dumps(call.messages, ensure_ascii=False)
    finally:
        application.state.store.close()


def test_the_comparison_form_never_closes_a_fact(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Forms that differ are no proof of a change: without a model answer,
    both values stay live, in conflict."""
    fake_openai.script(chat=[karim(said(LYON_FR, "Lyon", "lyon"))])
    ingest(client, app, LYON_FR)
    fake_openai.script(
        chat=[
            karim(said(PARIS_RU, "Париж", "paris")),
            chat_json({"verdicts": [{"pair": 0, "verdict": "unsure"}]}),
        ]
    )
    result = ingest(client, app, PARIS_RU)
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert all(f.valid_to is None for f in facts_of(app))


def test_the_owner_never_joins_external_content_by_its_form(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """P6 the other way round: a page read first plants the form the owner's
    later words will carry; the owner's fact is never linked into the page's
    group (recall would inject the page's value as the owner's)."""
    page = "Карим переехал в Париж."
    fake_openai.script(chat=[karim(said(page, "Париж", "lyon"))])
    ingest(client, app, "Lis ceci.", external=[{"kind": "web", "text": page}])
    fake_openai.script(
        chat=[
            karim(said(LYON_FR, "Lyon", "lyon")),
            chat_json({"verdicts": [{"pair": 0, "verdict": "unsure"}]}),
        ]
    )
    result = ingest(client, app, LYON_FR)
    assert result["values"]["norm"] == 0
    paris, lyon = facts_of(app)
    assert paris.provenance == "external" and lyon.provenance == "owner"
    assert lyon.same_as is None and paris.valid_to is None


def test_external_content_never_joins_the_owner_by_its_form(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """P6: a page whose value claims the owner's form (a planted "lyon") stays a
    visible conflict, never linked into the owner's group (recall would then
    inject one of the two as the other)."""
    fake_openai.script(chat=[karim(said(LYON_FR, "Lyon", "lyon"))])
    ingest(client, app, LYON_FR)
    page = "Карим переехал в Париж."
    fake_openai.script(chat=[karim(said(page, "Париж", "lyon"))])
    result = ingest(client, app, "Lis ceci.", external=[{"kind": "web", "text": page}])
    assert result["values"]["norm"] == 0 and result["facts"]["conflict"] == 1
    lyon, paris = facts_of(app)
    assert paris.provenance == "external" and paris.same_as is None
    assert "conflict" in paris.flags and lyon.valid_to is None
