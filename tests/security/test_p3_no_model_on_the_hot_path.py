"""P3 - a recall on the hot path does no LLM call, and neither does any other
hot-path route, even with chat and embeddings configured and the worker in
``auto`` mode. The fake model server's call log is the proof."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import iifa.recall.dense as dense_module
import iifa.recall.engine as engine_module
import iifa.recall.profile as profile_module
from iifa.api.app import create_app
from iifa.config import Settings
from iifa.ingest.queue import DONE
from iifa.llm import QueryEmbedder
from iifa.recall.engine import RecallEngine
from tests.api.conftest import remember
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth, make_settings

pytestmark = pytest.mark.security


@pytest.fixture
def settings(tmp_path: Path, fake_openai: FakeOpenAIServer) -> Settings:
    return make_settings(
        tmp_path,
        chat_url=fake_openai.base_url,
        chat_model="fake-chat",
        embeddings_url=fake_openai.base_url,
        embeddings_model="fake-embed",
        worker_poll=0.02,
    )


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


def test_p3_no_model_call_on_any_hot_path_route(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Every hot-path route answers without a model call, with chat and
    embeddings configured. The lifecycle writes only enqueue: the calls they
    lead to happen in the background, here held back by ``paused`` so the
    log proves the routes themselves made none."""
    assert client.get("/v1/health").json()["backends"]["chat"] is not None
    assert client.put("/v1/background", json={"mode": "paused"}).status_code == 200
    karim = remember(client, "Karim leads the solar project", node={"name": "Karim"}, pinned=True)
    solar = remember(client, "The solar project starts in March", node={"name": "Solar"})
    client.post(
        "/v1/link",
        json={"scope": "user:alice", "src": karim["node"]["id"], "dst": solar["node"]["id"]}
        | {"type": "works_on"},
    )
    client.post("/v1/recall", json={"scope": "user:alice", "query": "solar project"})
    client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": [karim["fact"]["id"]], "level": 2}
    )
    client.get(f"/v1/nodes/{karim['node']['id']}", params={"scope": "user:alice"})
    client.get("/v1/lookup", params={"scope": "user:alice", "name": "karim"})
    client.post("/v1/session/start", json={"scope": "user:alice"})
    client.post("/v1/timeline", json={"scope": "user:alice", "node_id": karim["node"]["id"]})
    client.post(
        "/v1/at",
        json={"scope": "user:alice"} | {"node_id": karim["node"]["id"]} | {"date": "2030-01-01"},
    )
    client.post(
        "/v1/path",
        json={"scope": "user:alice", "src": karim["node"]["id"], "dst": solar["node"]["id"]},
    )
    turn = client.post(
        "/v1/turns", json={"scope": "user:alice", "user": "solar?", "assistant": "Karim"}
    )
    compact = client.post(
        "/v1/precompact",
        json={"scope": "user:alice", "messages": [{"role": "user", "content": "solar"}]},
    )
    end = client.post("/v1/session/end", json={"scope": "user:alice"})
    assert turn.status_code == 202 and compact.status_code == 200 and end.status_code == 202
    client.post("/v1/recall", json={"scope": "user:alice", "query": "Karim"})
    # no LLM call; tier 3 embeds each recall's query once, and nothing else (SPEC 6.1)
    assert fake_openai.chat_calls == []
    assert [c.body["input"] for c in fake_openai.embedding_calls] == [["solar project"], ["Karim"]]
    assert client.get("/v1/health").json()["backends"]["chat"]["calls"] == 0
    fake_openai.reset()
    # the model work those writes queued is the background's, not the routes':
    # a run (on demand) is what makes the calls, and only then
    client.put("/v1/background", json={"mode": "on_demand"})
    report = client.post("/v1/background/run").json()
    # the three lifecycle jobs, the embed job the explicit writes queued, and one
    # more after the turn's extraction (coalesced with precompact and session end)
    assert report["processed"]["done"] == 5 and report["calls"] >= 1
    assert len(fake_openai.calls) == report["calls"]
    assert all(app.state.queue.get(j["id"]).status == DONE for j in report["jobs"])


def test_p3_the_recall_engine_holds_no_model_client() -> None:
    """Structural: the engine cannot call what it does not have. At tier 3 it
    holds a query embedder, which can embed a query and nothing else."""
    assert "llm" not in RecallEngine.__module__
    for module in (engine_module, profile_module, dense_module):
        names = {n for n in vars(module) if not n.startswith("__")}
        assert not any("ModelClient" in n or n == "llm" for n in names), module.__name__
    public = {n for n in dir(QueryEmbedder) if not n.startswith("_")}
    assert public == {"embed_query", "model"}


def test_p3_with_dense_recall_off_a_recall_makes_no_call_at_all(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = make_settings(
        tmp_path,
        chat_url=fake_openai.base_url,
        chat_model="fake-chat",
        embeddings_url=fake_openai.base_url,
        embeddings_model="fake-embed",
        recall_dense=False,
        background_mode="paused",
    )
    app = create_app(settings, clock=clock)
    try:
        with TestClient(app, headers=auth("all")) as client:
            remember(client, "Karim leads the solar project", node={"name": "Karim"})
            client.post("/v1/recall", json={"scope": "user:alice", "query": "solar project"})
        assert fake_openai.calls == []
    finally:
        app.state.store.close()
