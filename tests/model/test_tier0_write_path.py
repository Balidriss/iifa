"""SPEC 6, tier 0: with no model at all, the write path still links known
subjects. A recorded turn that mentions two known names yields a proposed
co-occurrence edge, no model is called, and recall reaches the neighbour
through the graph."""

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
from tests.helpers import auth, make_settings, remember
from tests.tier2 import SCOPE, end_session, post_turn, run_all

VIEW = Visibility.of_scope(SCOPE)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path, background_mode="on_demand", worker_poll=0.02)


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def test_tier0_turns_link_mentioned_known_names_without_a_model(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    assert client.get("/v1/health").json()["tier"] == 0
    karim = remember(client, "Karim is an electrician", node={"name": "Karim", "aliases": ["K"]})
    solar = remember(client, "The Solar project starts in March", node={"name": "Solar project"})
    remember(client, "Nadia is an accountant", node={"name": "Nadia"})
    post_turn(client, "Did Karim finish the solar project plan?", "K did, yesterday.")
    report = run_all(client, app)
    result = report["jobs"][0]["result"]
    assert report["calls"] == 0 and fake_openai.calls == []
    assert result["tier"] == 0 and result["calls"] == 0
    assert result["nodes"] == {"created": 0, "matched": 2, "mixed_script": 0}
    assert result["facts"]["written"] == 0 and result["edges"]["proposed"] == 1
    store = app.state.store
    edges = store.edges_of([karim["node"]["id"]], VIEW)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.type == "co_occurs" and edge.status == "proposed" and edge.origin == "extracted"
    assert {edge.src, edge.dst} == {karim["node"]["id"], solar["node"]["id"]}
    assert edge.weight == 0.5 and len(edge.evidence) == 1
    source = store.get_source(edge.evidence[0])
    assert source.kind == "turn" and source.text is not None and "K did" in source.text
    # the graph hop reaches the neighbour, at reduced weight, and says why
    recalled = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "electrician", "budget_tokens": 2000}
    ).json()
    why = {i["id"]: i["why"] for i in recalled["items"]}
    assert any("co_occurs" in w and "Solar project" in w for w in why.values())
    # a session end at tier 0 only decays: no summary, no call
    end_session(client, "s1")
    result = run_all(client, app)["jobs"][0]["result"]
    assert result["summaries"]["calls"] == 0 and fake_openai.calls == []
    assert client.get("/v1/health").json()["backends"]["chat"] is None
