"""P6 - provenance on everything, and an explicit fact is never overridden by an
extracted one; P7 - contradictions supersede, never overwrite, and history stays
queryable. Proven end to end: a turn, the fake model's answer, the store, the
API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import Settings
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth, remember
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

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


def city_fact(value: str, quote: str, **extra: object) -> dict[str, object]:
    return {
        "subject": "Karim",
        "statement": f"Karim lives in {value}.",
        "attribute": "city",
        "value": value,
        "quote": quote,
        **extra,
    }


def test_p6_every_extracted_record_carries_its_provenance(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Karim", "kind": "person"}, {"name": "Lyon", "kind": "place"}],
                facts=[city_fact("Lyon", "Karim lives in Lyon", confidence=0.85)],
                edges=[{"src": "Karim", "dst": "Lyon", "type": "lives in", "confidence": 0.9}],
            )
        ]
    )
    post_turn(client, "Karim lives in Lyon since May.", "Noted.", session_id="s1")
    report = run_all(client, app)
    result = report["jobs"][0]["result"]
    assert result["facts"]["written"] == 1 and result["edges"]["proposed"] == 2
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None
    fact = store.facts([karim.id], VIEW)[0]
    # who wrote it, from where, which span, with what confidence, when
    assert fact.origin == "extracted" and fact.confidence == 0.85
    assert fact.source_id is not None
    source = store.get_source(fact.source_id)
    assert source.kind == "turn" and source.session_id == "s1" and source.text is not None
    assert source.text[fact.excerpt_start : fact.excerpt_end] == "Karim lives in Lyon"
    assert fact.created_at == store.now()
    for edge in store.edges_of([karim.id], VIEW):
        assert edge.origin == "extracted" and edge.evidence == (source.id,)
        assert edge.source_id == source.id
    # and the API shows it: origin and confidence on expand, the excerpt at level 2
    expanded = client.post(
        "/v1/expand", json={"scope": SCOPE, "ids": [fact.id], "level": 2, "budget_tokens": 800}
    ).json()["items"][0]
    assert expanded["origin"] == "extracted" and expanded["confidence"] == 0.85
    assert expanded["source_id"] == source.id and expanded["excerpt"] == "Karim lives in Lyon"
    # what a remember writes is explicit, with the statement itself as its source
    explicit = remember(client, "Karim prefers tea", node_id=karim.id)
    stored = store.get_fact(explicit["fact"]["id"])
    assert stored.origin == "explicit" and stored.source_id == explicit["source_id"]


def test_p6_an_extracted_fact_never_overrides_an_explicit_one(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Both stay; the new one is flagged ``conflict`` with what it stands against,
    and recall reports it (SPEC 5.4)."""
    explicit = remember(
        client,
        "Karim lives in Paris",
        node={"name": "Karim", "kind": "person"},
        attribute="city",
        value="Paris",
    )
    fake_openai.script(chat=[answer(facts=[city_fact("Lyon", "Karim lives in Lyon")])])
    post_turn(client, "Karim lives in Lyon now.", "ok")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["facts"] == {
        "written": 1,
        "duplicate": 0,
        "superseded": 0,
        "conflict": 1,
        "language_mismatch": 0,
        "incomplete": 0,
        "cross_script": 0,
        "mixed_script": 0,
        "speaker_owner": 0,
        "speaker_free": 0,
        "proposed": 0,
    }
    store = app.state.store
    live = store.facts([explicit["node"]["id"]], VIEW)
    assert {f.statement for f in live} == {"Karim lives in Paris", "Karim lives in Lyon."}
    old = store.get_fact(explicit["fact"]["id"])
    assert old.valid_to is None and old.flags == () and old.superseded_by is None
    new = next(f for f in live if f.origin == "extracted")
    assert new.flags == ("conflict",) and new.conflicts_with == old.id
    recalled = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "Karim lives", "budget_tokens": 800}
    ).json()
    flags = {i["id"]: i["flags"] for i in recalled["items"]}
    assert flags[new.id] == ["conflict"] and flags[old.id] == []
    expanded = client.post("/v1/expand", json={"scope": SCOPE, "ids": [new.id]}).json()
    assert expanded["items"][0]["conflicts_with"] == old.id
    # the explicit fact is untouched by a second extraction saying the same thing
    fake_openai.script(chat=[answer(facts=[city_fact("Lyon", "Lyon")])])
    post_turn(client, "Still Lyon.", "ok")
    assert run_all(client, app)["jobs"][0]["result"]["facts"]["duplicate"] == 1
    assert store.get_fact(old.id).valid_to is None


def test_p7_a_contradiction_closes_the_old_fact_and_history_stays_queryable(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Karim", "kind": "person"}],
                facts=[city_fact("Paris", "Karim lives in Paris")],
            ),
            answer(facts=[city_fact("Lyon", "moved to Lyon")]),
        ]
    )
    post_turn(client, "Karim lives in Paris.", "ok")
    run_all(client, app)
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None
    day_one = store.now()
    clock.advance(days=30)
    post_turn(client, "Karim moved to Lyon.", "ok")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["facts"]["written"] == 1 and result["facts"]["superseded"] == 1
    live = store.facts([karim.id], VIEW)
    assert [f.value for f in live] == ["Lyon"]
    old = next(f for f in store.facts([karim.id], VIEW, live_only=False) if f.value == "Paris")
    assert old.valid_to == store.now() and old.superseded_by == live[0].id
    assert old.archived_at is None  # closed, never overwritten nor deleted
    # recall no longer returns it; history does
    recalled = client.post("/v1/recall", json={"scope": SCOPE, "query": "Paris"}).json()
    assert recalled["items"] == []
    then = client.post(
        "/v1/at", json={"scope": SCOPE, "node_id": karim.id, "date": day_one, "budget_tokens": 800}
    ).json()
    assert [i["value"] for i in then["items"]] == ["Paris"]
    now = client.post(
        "/v1/at", json={"scope": SCOPE, "node_id": karim.id, "date": store.now()}
    ).json()
    assert [i["value"] for i in now["items"]] == ["Lyon"]
    line = client.post(
        "/v1/timeline", json={"scope": SCOPE, "node_id": karim.id, "budget_tokens": 2000}
    ).json()
    assert [(i["change"], i["value"]) for i in line["items"]] == [
        ("recorded", "Lyon"),
        ("closed", "Paris"),
        ("recorded", "Paris"),
    ]
    # a dated contradiction closes the old fact at the new one's own start
    fake_openai.script(
        chat=[answer(facts=[city_fact("Berlin", "Berlin", valid_from="2026-12-01")])]
    )
    post_turn(client, "Karim will move to Berlin on 2026-12-01.", "ok")
    run_all(client, app)
    lyon = store.get_fact(live[0].id)
    assert lyon.valid_to == "2026-12-01" and lyon.superseded_by is not None


def since_fact(statement: str, value: str, quote: str) -> dict[str, object]:
    return {
        "subject": "Karim",
        "statement": statement,
        "attribute": "since",
        "value": value,
        "quote": quote,
    }


def test_p7_the_same_value_in_another_language_is_linked_never_closed(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """SPEC 5.4: "mars 2026" and "March 2026" are the same value in two languages.
    The English restatement must not close the French fact.
    Proved the same, both stay live, linked (no conflict); a later change in one
    language closes the whole group, which states the outdated value."""
    french = "Karim dirige la planification du projet solaire depuis mars 2026."
    english = "Karim leads the planning of the solar project since March 2026."
    fake_openai.script(
        chat=[
            answer(facts=[since_fact(french, "mars 2026", "depuis mars 2026")]),
            answer(facts=[since_fact(english, "March 2026", "since March 2026")]),
            answer(
                facts=[
                    since_fact(
                        "Karim dirige la planification du projet solaire depuis avril 2026.",
                        "avril 2026",
                        "depuis avril 2026",
                    )
                ]
            ),
        ]
    )
    post_turn(client, french, "Noté.")
    post_turn(client, english, "Noted.")
    reports = run_all(client, app)["jobs"]
    second = reports[1]["result"]
    assert second["facts"]["superseded"] == 0 and second["facts"]["conflict"] == 0
    assert second["values"]["same"] == 1
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None
    live = {f.language: f for f in store.facts([karim.id], VIEW)}
    assert set(live) == {"fr", "en"}
    assert live["fr"].valid_to is None and live["fr"].flags == ()
    assert live["en"].same_as == live["fr"].id and live["en"].flags == ()

    # a changed value in one language closes the old fact and its equivalent (P7)
    post_turn(client, "Karim dirige la planification du projet solaire depuis avril 2026.", "ok")
    third = run_all(client, app)["jobs"][0]["result"]["facts"]
    assert third["superseded"] == 2
    assert store.get_fact(live["fr"].id).valid_to is not None
    assert store.get_fact(live["en"].id).valid_to is not None
