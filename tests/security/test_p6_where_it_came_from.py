"""Where a fact came from (P6): every read route tells the reader
whether a fact is the owner's own words, the assistant's, or content read from
outside - with the part's kind, the short form of its ref and the day it was
read - and says nothing more for the owner's own words, so a recall of the
owner's facts costs what it costs without it (P1)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest import sessions
from iifa.store.models import Fact
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, end_session, post_turn, run_all, tier2_settings

pytestmark = pytest.mark.security

OWNER_SAYS = "Karim's phone is 555-0101."
ASSISTANT_SAYS = "Karim's e-mail is karim@mail.test, as far as I know."
PAGE = "Contact desk: Karim, office 12 on Main Street."
REF = "https://www.example.com/post/9?token=s3cret"
READ = "2026-09-20"
HANDLE_KEYS = {"id", "kind", "title", "line", "why", "age_days", "expand_tokens", "flags"}


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def fact(subject: str, statement: str, key: str, value: str, quote: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "statement": statement,
        "attribute": key,
        "value": value,
        "quote": quote,
    }


def all_facts(app: FastAPI) -> list[Fact]:
    store: Store = app.state.store
    rows = store.execute("SELECT id FROM facts ORDER BY created_at, id").fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def three_facts(client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer) -> dict[str, Fact]:
    """The shape: one node, three facts of one key family, one per part."""
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Karim", "kind": "person"}],
                facts=[
                    fact(
                        "Karim",
                        "Karim's phone is 555-0101.",
                        "contact.phone",
                        "555-0101",
                        "Karim's phone is 555-0101",
                    ),
                    fact(
                        "Karim",
                        "Karim's e-mail is karim@mail.test.",
                        "contact.email",
                        "karim@mail.test",
                        "Karim's e-mail is karim@mail.test",
                    ),
                    fact(
                        "Karim",
                        "Karim's office is office 12 on Main Street.",
                        "contact.office",
                        "office 12",
                        "Karim, office 12 on Main Street",
                    ),
                ],
            )
        ]
    )
    post_turn(
        client,
        OWNER_SAYS,
        ASSISTANT_SAYS,
        occurred_at=f"{READ}T10:00:00Z",
        external=[{"kind": "web", "text": PAGE, "ref": REF}],
    )
    run_all(client, app)
    facts = {f.attribute or "": f for f in all_facts(app)}
    assert set(facts) == {"contact.phone", "contact.email", "contact.office"}
    return facts


EXPECTED_FROM = {
    "contact.phone": None,
    "contact.email": "assistant",
    "contact.office": f"web example.com {READ}",
}


def _by_key(facts: dict[str, Fact], items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {f.id: key for key, f in facts.items()}
    return {keys[i["id"]]: i.get("from") for i in items if i["id"] in keys}


def test_a_fact_keeps_its_part_kind_the_short_form_of_its_ref_and_the_day_read(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    facts = three_facts(client, app, fake_openai)
    office = facts["contact.office"]
    assert (office.provenance, office.part_kind, office.part_ref, office.read_at) == (
        "external",
        "web",
        "example.com",
        READ,
    )
    for key in ("contact.phone", "contact.email"):
        assert (facts[key].part_kind, facts[key].part_ref, facts[key].read_at) == (None, None, None)
    # the whole ref, with its token, is kept nowhere in the store
    store: Store = app.state.store
    for table in ("facts", "sources", "proposals"):
        rows = store.execute(f"SELECT * FROM {table}").fetchall()
        assert "s3cret" not in json.dumps([dict(r) for r in rows])


def test_every_read_route_says_where_a_fact_came_from(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    facts = three_facts(client, app, fake_openai)
    ids = [f.id for f in facts.values()]
    karim = facts["contact.phone"].node_id

    recalled = client.post(
        "/v1/recall",
        json={"scope": SCOPE, "query": "Karim phone e-mail office", "budget_tokens": 4000},
    ).json()["items"]
    assert _by_key(facts, recalled) == EXPECTED_FROM
    owner_handle = next(i for i in recalled if i["id"] == facts["contact.phone"].id)
    assert set(owner_handle) == HANDLE_KEYS  # the owner's own words: nothing added

    expanded = client.post(
        "/v1/expand", json={"scope": SCOPE, "ids": ids, "level": 1, "budget_tokens": 4000}
    ).json()["items"]
    assert _by_key(facts, expanded) == EXPECTED_FROM
    office = next(i for i in expanded if i["id"] == facts["contact.office"].id)
    assert (office["provenance"], office["part_kind"], office["part_ref"], office["read_at"]) == (
        "external",
        "web",
        "example.com",
        READ,
    )
    email = next(i for i in expanded if i["id"] == facts["contact.email"].id)
    assert email["provenance"] == "assistant" and "part_kind" not in email
    phone = next(i for i in expanded if i["id"] == facts["contact.phone"].id)
    assert "from" not in phone and "provenance" not in phone

    node = client.post(
        "/v1/expand", json={"scope": SCOPE, "ids": [karim], "level": 1, "budget_tokens": 4000}
    ).json()["items"][0]
    assert _by_key(facts, node["facts"]) == EXPECTED_FROM

    viewed = client.get(f"/v1/nodes/{karim}", params={"scope": SCOPE, "budget_tokens": 4000}).json()
    assert _by_key(facts, viewed["facts"]) == EXPECTED_FROM

    timeline = client.post(
        "/v1/timeline", json={"scope": SCOPE, "node_id": karim, "budget_tokens": 4000}
    ).json()["items"]
    assert _by_key(facts, timeline) == EXPECTED_FROM

    known = client.post(
        "/v1/at",
        json={
            "scope": SCOPE,
            "node_id": karim,
            "date": app.state.store.now(),
            "budget_tokens": 4000,
        },
    ).json()
    assert _by_key(facts, known["items"]) == EXPECTED_FROM

    for fact_id in ids:
        assert client.patch(f"/v1/facts/{fact_id}", json={"pinned": True}).status_code == 200
    profile = client.post("/v1/session/start", json={"scope": SCOPE, "budget_tokens": 4000})
    assert _by_key(facts, profile.json()["items"]) == EXPECTED_FROM


def test_the_day_read_is_the_day_the_turn_was_recorded_when_the_host_gives_none(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Karim", "kind": "person"}],
                facts=[
                    fact(
                        "Karim",
                        "Karim's office is office 12.",
                        "contact.office",
                        "office 12",
                        "Karim, office 12 on Main Street",
                    )
                ],
            )
        ]
    )
    day = clock.now_utc.date().isoformat()
    post_turn(
        client,
        "Read this.",
        external=[{"kind": "document", "text": PAGE, "ref": "C:\\docs\\desk list.pdf"}],
    )
    clock.advance(days=2)  # the background extracts it later: the day read stays the turn's
    run_all(client, app)
    (office,) = all_facts(app)
    assert (office.part_kind, office.part_ref, office.read_at) == ("document", "desk_list.pdf", day)
    recalled = client.post("/v1/recall", json={"scope": SCOPE, "query": "Karim office"}).json()
    handle = next(i for i in recalled["items"] if i["id"] == office.id)
    assert handle["from"] == f"document desk_list.pdf {day}"


def test_a_node_known_only_from_what_was_read_says_so(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    facts = three_facts(client, app, fake_openai)
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Ines", "kind": "person"}],
                facts=[
                    fact(
                        "Ines",
                        "Ines's office is office 14.",
                        "contact.office",
                        "office 14",
                        "Ines, office 14",
                    )
                ],
            )
        ]
    )
    post_turn(client, "And this one?", tool_summaries=["Desk: Ines, office 14."])
    run_all(client, app)
    recalled = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "Ines Karim", "budget_tokens": 4000}
    ).json()["items"]
    nodes = {i["title"]: i for i in recalled if i["kind"] == "node"}
    assert nodes["Ines"]["flags"] == ["external_only"]
    assert nodes["Karim"]["flags"] == []  # one fact is the owner's own
    ines = next(
        f
        for f in all_facts(app)
        if f.attribute == "contact.office" and f.id != facts["contact.office"].id
    )
    handle = next(i for i in recalled if i["id"] == ines.id)
    assert handle["from"].startswith("tool ") and handle["from"].count(" ") == 1  # no ref


def test_an_owner_only_recall_costs_what_it_costs_without_the_field(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Karim", "kind": "person"}],
                facts=[
                    fact(
                        "Karim",
                        "Karim's phone is 555-0101.",
                        "contact.phone",
                        "555-0101",
                        "Karim's phone is 555-0101",
                    )
                ],
            )
        ]
    )
    post_turn(client, OWNER_SAYS)
    run_all(client, app)
    body = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "Karim phone", "budget_tokens": 4000}
    ).json()
    assert body["items"]
    for item in body["items"]:
        assert set(item) == HANDLE_KEYS
        assert "external_only" not in item["flags"]


def test_an_accepted_fact_proposal_keeps_where_it_came_from(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = tier2_settings(tmp_path, fake_openai, owners=(("user:alice", "Alice", ()),))
    application = create_app(settings, clock=clock)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    answer(
                        nodes=[{"name": "Alice", "kind": "person"}],
                        facts=[
                            fact(
                                "Alice",
                                "Alice lives in Moscow.",
                                "home",
                                "Moscow",
                                "The user lives in Moscow",
                            )
                        ],
                    )
                ]
            )
            post_turn(
                client,
                "Summarise this page.",
                occurred_at=READ,
                external=[{"kind": "web", "text": "The user lives in Moscow.", "ref": REF}],
            )
            run_all(client, application)
            store: Store = application.state.store
            (row,) = store.execute("SELECT id FROM proposals WHERE kind = 'fact'").fetchall()
            proposal = store.get_proposal(row["id"])
            assert (proposal.payload["part_kind"], proposal.payload["part_ref"]) == (
                "web",
                "example.com",
            )
            decided = client.post(f"/v1/proposals/{proposal.id}", json={"decision": "accept"})
            assert decided.status_code == 200, decided.text
            accepted = store.get_fact(decided.json()["fact_id"])
            assert (accepted.origin, accepted.provenance) == ("explicit", "external")
            assert (accepted.part_kind, accepted.part_ref, accepted.read_at) == (
                "web",
                "example.com",
                READ,
            )
    finally:
        application.state.store.close()


def test_a_session_note_kept_at_the_close_keeps_where_it_came_from(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = tier2_settings(
        tmp_path, fake_openai, extraction_intent=True, owners=(("user:alice", "Alice", ()),)
    )
    application = create_app(settings, clock=clock)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    chat_json(
                        {
                            "nodes": [],
                            "edges": [],
                            "frame": "sincere",
                            "facts": [
                                {
                                    "subject": "Weather Station",
                                    "statement": "Weather Station records 60 readings per hour.",
                                    "attribute": "frame_rate",
                                    "value": "60 frames per second",
                                    "quote": "Weather Station: 60 readings per hour",
                                    "modality": "fact",
                                    "duration": "unsure",
                                }
                            ],
                        }
                    ),
                    chat_json({"episode": None, "keep": [0]}),
                ]
            )
            post_turn(
                client,
                "What does the profiler say?",
                session_id="station",
                occurred_at=READ,
                external=[
                    {
                        "kind": "tool",
                        "text": "Weather Station: 60 readings per hour",
                        "ref": "profiler",
                    }
                ],
            )
            run_all(client, application)
            state = sessions.load(application.state.store, "station", [SCOPE])
            assert [(n.part_kind, n.part_ref, n.read_at) for n in state.facts] == [
                ("tool", "profiler", READ)
            ]
            end_session(client, "station")
            run_all(client, application)
            (kept,) = all_facts(application)
            assert (kept.provenance, kept.part_kind, kept.part_ref, kept.read_at) == (
                "external",
                "tool",
                "profiler",
                READ,
            )
    finally:
        application.state.store.close()


def test_a_declared_assistant_name_labels_its_part_and_its_facts(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """The name the operator gives the assistant; the provenance and
    every rule that reads it stay `assistant`."""
    settings = tier2_settings(tmp_path, fake_openai, assistants=((SCOPE, "NOVA"),))
    application = create_app(settings, clock=clock)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    answer(
                        nodes=[{"name": "Karim", "kind": "person"}],
                        facts=[
                            fact(
                                "Karim",
                                "Karim's e-mail is karim@mail.test.",
                                "contact.email",
                                "karim@mail.test",
                                "NOVA: Karim's e-mail is karim@mail.test",
                            )
                        ],
                    )
                ]
            )
            post_turn(client, "What is Karim's e-mail?", ASSISTANT_SAYS)
            run_all(client, application)
            source = fake_openai.chat_calls[0].body["messages"]
            assert "\n\nNOVA: Karim's e-mail is" in source[-1]["content"]
            assert 'labelled "NOVA:"' in source[0]["content"]
            (email,) = all_facts(application)
            assert (email.provenance, email.part_ref, email.part_kind) == (
                "assistant",
                "NOVA",
                None,
            )
            recalled = client.post(
                "/v1/recall", json={"scope": SCOPE, "query": "Karim e-mail"}
            ).json()["items"]
            assert next(i for i in recalled if i["id"] == email.id)["from"] == "NOVA"
            expanded = client.post(
                "/v1/expand", json={"scope": SCOPE, "ids": [email.id], "budget_tokens": 4000}
            ).json()["items"][0]
            assert (expanded["from"], expanded["provenance"]) == ("NOVA", "assistant")
    finally:
        application.state.store.close()
