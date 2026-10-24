"""Semantic resolution, merge proposals and semantic links (SPEC 5.1 step 3,
5.2, 6.1): a near mention in another script becomes a candidate that the
batched adjudication may attach; existing near nodes are proposed for a merge,
never merged; close summaries get a proposed `similar_to` edge."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.helpers import auth, remember
from tests.model.test_tier3 import FAR, NEAR, tier3_settings
from tests.tier2 import SCOPE, answer, post_turn, run_all

VIEW = Visibility.of_scope(SCOPE)
RUSSIAN = "Карим переехал в Лион."


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier3_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def russian_answer() -> object:
    return answer(
        nodes=[{"name": "Карим", "kind": "person"}],
        facts=[{"subject": "Карим", "statement": RUSSIAN, "quote": "Карим переехал в Лион"}],
    )


def known_karim(client: TestClient, app: FastAPI, fake: FakeOpenAIServer) -> str:
    fake.set_vectors({"Karim": NEAR, "Карим": NEAR, "Лион": FAR})
    karim = remember(client, "Karim leads Solar.", node={"name": "Karim", "kind": "person"})
    run_all(client, app)  # Karim's name vector
    node_id: str = karim["node"]["id"]
    return node_id


def test_a_cross_script_mention_is_a_candidate_the_adjudication_attaches(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    karim = known_karim(client, app, fake_openai)
    fake_openai.script(
        chat=[
            russian_answer(),
            chat_json({"decisions": [{"mention": 0, "choice": karim}]}),
        ]
    )
    post_turn(client, RUSSIAN)
    report = run_all(client, app)
    turn = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert turn["mentions_adjudicated"] == 1 and turn["nodes"]["created"] == 0
    # SPEC 11.1: the report says what was asked and what the model answered
    assert turn["adjudication_calls"] == 1
    [asked] = turn["adjudications"]
    assert asked["mention"] == "Карим" and asked["chosen"] == "Karim"
    [candidate] = asked["candidates"]
    assert candidate["name"] == "Karim" and candidate["semantic"] is True
    assert 0 < candidate["score"] <= 1
    prompt = fake_openai.chat_calls[1].messages[1]["content"]
    assert "Mention 0: 'Карим'" in prompt and karim in prompt
    store = app.state.store
    node = store.get_node(karim)
    assert "Карим" in node.aliases  # in the source, now Karim's (SPEC 5.1)
    assert [f.statement for f in store.facts([karim], VIEW) if f.origin == "extracted"] == [RUSSIAN]


def test_a_semantic_neighbour_never_resolves_on_its_own(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The adjudication says "new": a new node, and then, both existing and close,
    a merge proposal (never a merge)."""
    karim = known_karim(client, app, fake_openai)
    fake_openai.script(
        chat=[russian_answer(), chat_json({"decisions": [{"mention": 0, "choice": "new"}]})]
    )
    post_turn(client, RUSSIAN)
    report = run_all(client, app)
    turn = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert [(a["mention"], a["chosen"]) for a in turn["adjudications"]] == [("Карим", None)]
    store = app.state.store
    cyrillic = store.find_node("Карим", VIEW)
    assert cyrillic is not None and cyrillic.id != karim
    merges = [p for p in store.proposals([karim, cyrillic.id], VIEW) if p.kind == "merge"]
    assert len(merges) == 1
    assert {merges[0].node_id, merges[0].value} == {karim, cyrillic.id}
    assert store.get_node(karim).archived_at is None and cyrillic.archived_at is None


def test_far_mentions_are_not_candidates(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    known_karim(client, app, fake_openai)
    fake_openai.set_vectors({"Лион": FAR, "Иван": FAR})
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Иван", "kind": "person"}],
                facts=[{"subject": "Иван", "statement": "Иван живёт в Лионе.", "quote": "Иван"}],
            )
        ]
    )
    post_turn(client, "Иван живёт в Лионе.")
    report = run_all(client, app)
    turn = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert turn["mentions_adjudicated"] == 0 and turn["nodes"]["created"] == 1
    assert turn["adjudications"] == [] and turn["adjudication_calls"] == 0


def test_close_summaries_get_a_proposed_similar_to_edge(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.set_vectors(
        {
            "Solar": FAR,
            "Projet solaire": [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0],
            "Installing panels on the roof": NEAR,
            "Installation de panneaux sur le toit": [1.0, 0.12, 0, 0, 0, 0, 0, 0],
        }
    )
    solar = remember(
        client,
        "Solar starts in March.",
        node={"name": "Solar", "kind": "project", "summary": "Installing panels on the roof"},
    )
    projet = remember(
        client,
        "Le projet démarre en mars.",
        node={
            "name": "Projet solaire",
            "kind": "project",
            "summary": "Installation de panneaux sur le toit",
        },
    )
    run_all(client, app)
    store = app.state.store
    ids = {solar["node"]["id"], projet["node"]["id"]}
    similar = [e for e in store.edges_between(list(ids), VIEW) if e.type == "similar_to"]
    assert len(similar) == 1
    assert {similar[0].src, similar[0].dst} == ids and similar[0].status == "proposed"
    assert similar[0].weight < 1.0
    # names are far apart: no merge proposal
    assert [p for p in store.proposals(list(ids), VIEW) if p.kind == "merge"] == []


@pytest.fixture
def uncalibrated(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> Iterator[tuple[FastAPI, TestClient]]:
    settings = tier3_settings(
        tmp_path, fake_openai, resolve_threshold=None, merge_threshold=None, link_threshold=None
    )
    application = create_app(settings, clock=clock)
    with TestClient(application, headers=auth("all")) as test_client:
        yield application, test_client
    application.state.store.close()


def test_an_uncalibrated_model_proposes_nothing_semantic(
    uncalibrated: tuple[FastAPI, TestClient], fake_openai: FakeOpenAIServer
) -> None:
    """SPEC 6.1: with no profile and no configured thresholds, identical vectors
    still make no candidate, no merge proposal and no similar_to edge."""
    app, client = uncalibrated
    fake_openai.set_vectors({"Karim": NEAR, "Карим": NEAR, "Kar": NEAR, "Kar summary": NEAR})
    karim = remember(client, "Karim leads Solar.", node={"name": "Karim", "kind": "person"})
    remember(client, "Карим переехал в Лион.", node={"name": "Карим", "kind": "person"})
    run_all(client, app)
    fake_openai.script(chat=[russian_answer()])
    post_turn(client, RUSSIAN)
    report = run_all(client, app)
    turn = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert turn["mentions_adjudicated"] == 0
    store = app.state.store
    assert store.proposals([karim["node"]["id"]], VIEW) == []
    assert [e for e in store.edges_of([karim["node"]["id"]], VIEW) if e.type == "similar_to"] == []
