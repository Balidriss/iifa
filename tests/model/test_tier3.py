"""Tier 3 against the fake embeddings server (SPEC 6, 6.1): vectors computed in
the background with a model stamp, re-embedded after a model change, dense
recall fused with lexical by RRF, and honest degradation (P10)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa import vectors
from iifa.api.app import create_app
from iifa.llm import EMBEDDINGS, Endpoint, ModelClient, QueryEmbedder
from iifa.recall.dense import Dense
from iifa.scopes import Visibility
from iifa.store.repo import SUMMARY, TEXT, Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import Fail, FakeOpenAIServer, Hang
from tests.helpers import auth, remember
from tests.tier2 import SCOPE, run_all, tier2_settings

FRENCH = "Karim dirige la planification du projet solaire."
QUERY = "who is in charge of the photovoltaic installation"
NEAR = [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FAR = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def tier3_settings(tmp_path: Path, fake: FakeOpenAIServer, **overrides: Any) -> Any:
    values: dict[str, Any] = {
        "embeddings_url": fake.base_url,
        "embeddings_model": "fake-embed",
        "query_timeout": 1.5,
        # fake-embed has no profile: the tests calibrate it explicitly (SPEC 6.1)
        "resolve_threshold": 0.8,
        "merge_threshold": 0.92,
        "link_threshold": 0.8,
    }
    values.update(overrides)
    return tier2_settings(tmp_path, fake, **values)


@pytest.fixture
def overrides() -> dict[str, Any]:
    return {}


@pytest.fixture
def app(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock, overrides: dict[str, Any]
) -> Iterator[FastAPI]:
    application = create_app(tier3_settings(tmp_path, fake_openai, **overrides), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def recall(client: TestClient, query: str = QUERY, **extra: Any) -> dict[str, Any]:
    body = {"scope": SCOPE, "query": query, "budget_tokens": 800, **extra}
    response = client.post("/v1/recall", json=body)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def french_fact(client: TestClient, app: FastAPI, fake: FakeOpenAIServer) -> str:
    fake.set_vectors({FRENCH: NEAR, QUERY: NEAR, "Karim": FAR})
    fact = remember(client, FRENCH, node={"name": "Karim", "kind": "person"})
    run_all(client, app)  # the embed job the write queued
    fact_id: str = fact["fact"]["id"]
    return fact_id


# --------------------------------------------------------------------------- vectors


def test_a_write_queues_an_embed_job_that_stamps_vectors(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fact_id = french_fact(client, app, fake_openai)
    calls = fake_openai.embedding_calls
    assert calls and all(c.model == "fake-embed" for c in calls)
    assert FRENCH in calls[0].body["input"]
    stats = app.state.store.vector_stats("fake-embed")
    assert stats["stale"] == 0 and stats["missing"] == 0 and stats["current"] >= 2
    rows = app.state.store.vector_rows(
        Visibility.of_scope(SCOPE), facets=(TEXT,), model="fake-embed"
    )
    assert [r.record_id for r in rows] == [fact_id] and rows[0].dimension == 8
    health = client.get("/v1/health").json()
    assert health["tier"] == 3 and health["stale_vectors"] == 0
    assert health["vectors"]["path"] == "numpy" and health["vectors"]["dense"] == "on"
    assert health["vectors"]["model"] == "fake-embed" and health["vectors"]["current"] >= 2


def test_a_model_change_reembeds_in_the_background(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    first = create_app(tier3_settings(tmp_path, fake_openai), clock=clock)
    with TestClient(first, headers=auth("all")) as client:
        remember(client, FRENCH, node={"name": "Karim", "kind": "person"})
        run_all(client, first)
    first.state.store.close()
    fake_openai.reset()
    second = create_app(
        tier3_settings(tmp_path, fake_openai, embeddings_model="embed-b"), clock=clock
    )
    try:
        with TestClient(second, headers=auth("all")) as client:
            health = client.get("/v1/health").json()
            assert health["stale_vectors"] == 2  # the fact and Karim's name
            assert health["queue"]["pending"] == 1  # the re-embed job, queued at start
            # stale vectors are never compared with fresh ones: dense sees nothing yet
            assert recall(client)["items"] == []
            run_all(client, second)
            assert client.get("/v1/health").json()["stale_vectors"] == 0
            assert {c.model for c in fake_openai.embedding_calls} == {"embed-b"}
    finally:
        second.state.store.close()


# --------------------------------------------------------------------------- dense recall


def test_dense_recall_finds_what_lexical_cannot(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fact_id = french_fact(client, app, fake_openai)
    result = recall(client)
    assert result["tier"] == 3 and result["dense"] == "on" and result["degraded"] is None
    item = next(i for i in result["items"] if i["id"] == fact_id)
    assert item["why"] == "vector neighbour 1.00"


def test_lexical_and_dense_are_fused(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fact_id = french_fact(client, app, fake_openai)
    fake_openai.set_vectors({"projet solaire": NEAR})
    item = next(i for i in recall(client, "projet solaire")["items"] if i["id"] == fact_id)
    assert "query terms 'projet', 'solaire'" in item["why"] and "vector neighbour" in item["why"]


def test_a_summary_never_ranks_a_node_but_stays_shown_and_embedded(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """SPEC 6.1: a summary is model-written text. Recall ranks on names, aliases
    and facts - what was actually said - so a summary sharing the query's words
    and vector never pushes a fact down. It is still shown in the node's handle
    and its vector still serves link proposals (SPEC 5.2)."""
    query, summary = "what does Joseph prefer to drink", "Daniela prefers to drink tea"
    far = {text: FAR for text in ("Иосиф", "Daniela", "Daniela works in Porto.")}
    fake_openai.set_vectors({query: NEAR, summary: NEAR, "Иосиф любит чай.": NEAR, **far})
    fact = remember(client, "Иосиф любит чай.", node={"name": "Иосиф", "kind": "person"})
    daniela = remember(
        client,
        "Daniela works in Porto.",
        node={"name": "Daniela", "kind": "person", "summary": summary},
    )["node"]["id"]
    run_all(client, app)
    store = app.state.store
    rows = store.vector_rows(Visibility.of_scope(SCOPE), facets=(SUMMARY,), model="fake-embed")
    assert [r.record_id for r in rows] == [daniela]  # kept for semantic links
    items = recall(client, query)["items"]
    assert items[0]["id"] == fact["fact"]["id"]  # the only answer comes first
    reached = {i["id"]: i["why"] for i in items}
    # if Daniela is reached at all, it is not by her summary's words or vector
    why = reached.get(daniela, "")
    assert "prefers" not in why and "drink" not in why and "vector neighbour 1.00" not in why
    by_name = next(i for i in recall(client, "Daniela")["items"] if i["id"] == daniela)
    assert by_name["line"] == summary and "query term 'daniela'" in by_name["why"]


# --------------------------------------------------------------------------- degradation (P10)


@pytest.mark.parametrize("overrides", [{"query_timeout": 0.3}])
@pytest.mark.parametrize(("behaviour", "reason"), [(Fail(500), "http_500"), (Hang(3.0), "timeout")])
def test_an_unreachable_embeddings_backend_serves_tier_2_and_says_so(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    behaviour: Any,
    reason: str,
) -> None:
    french_fact(client, app, fake_openai)
    fake_openai.script(embeddings=[behaviour])
    result = recall(client, "Karim")
    assert result["tier"] == 2 and result["dense"] == "unavailable"
    assert result["degraded"] == f"embeddings {reason}"
    assert result["items"]  # lexical + graph still answered


@pytest.mark.parametrize("overrides", [{"recall_dense": False}])
def test_dense_off_makes_no_embedding_call_on_recall(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    french_fact(client, app, fake_openai)
    before = len(fake_openai.embedding_calls)
    assert before > 0  # vectors still serve the background
    result = recall(client, "Karim")
    assert len(fake_openai.embedding_calls) == before
    assert result["tier"] == 2 and result["dense"] == "off" and result["degraded"] is None
    assert client.get("/v1/health").json()["vectors"]["dense"] == "off"


def test_a_supplied_query_vector_costs_no_call_when_its_stamp_matches(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fact_id = french_fact(client, app, fake_openai)
    before = len(fake_openai.embedding_calls)
    result = recall(client, query_vector={"model": "fake-embed", "values": NEAR})
    assert len(fake_openai.embedding_calls) == before
    assert result["query_vector"] == "used" and fact_id in {i["id"] for i in result["items"]}
    for supplied, note in (
        ({"model": "other-model", "values": NEAR}, "ignored: model other-model"),
        ({"model": "fake-embed", "values": [1.0, 0.0]}, "ignored: dimension 2"),
    ):
        result = recall(client, query_vector=supplied)
        assert result["query_vector"] == note
    # ignored, so the service embedded the query itself (once: the cache holds it)
    assert len(fake_openai.embedding_calls) == before + 1


def test_a_repeated_query_is_embedded_once(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    french_fact(client, app, fake_openai)
    before = len(fake_openai.embedding_calls)
    recall(client)
    recall(client)
    assert len(fake_openai.embedding_calls) == before + 1


def test_the_query_embedding_runs_in_the_foreground(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """SPEC 6.1: the query embedding holds new background calls while in flight
    (the gate's foreground), and never waits at the background gate itself."""
    french_fact(client, app, fake_openai)
    gate = app.state.worker.gate
    entered: list[int] = []
    real = gate.foreground

    def spy() -> Any:
        entered.append(gate.foreground_active())
        return real()

    gate.foreground = spy  # type: ignore[method-assign]
    gate.set_limit("embeddings", 1)
    assert gate.acquire("embeddings", timeout=0)  # a background embed holds the only slot
    try:
        result = recall(client, "a query nobody embedded yet")
    finally:
        gate.release("embeddings")
    assert entered == [0] and result["dense"] == "on"
    assert gate.foreground_active() == 0


def test_without_numpy_tier_3_runs_on_the_python_path_and_says_so(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    fact_id = french_fact(client, app, fake_openai)
    monkeypatch.setattr(vectors, "_numpy", None)
    health = client.get("/v1/health").json()["vectors"]
    assert health["path"] == "python" and health["python_path_limit"] == vectors.python_path_limit()
    assert fact_id in {i["id"] for i in recall(client)["items"]}


def test_a_refused_connection_serves_tier_2_and_says_so(tmp_path: Path) -> None:
    """The query embedder's transport failure, whatever the platform does with a
    closed port (a scripted transport, never a real socket), degrades the recall."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = ModelClient(
        None,
        Endpoint(EMBEDDINGS, "http://127.0.0.1:9/v1", "bge-m3"),
        transport=httpx.MockTransport(refuse),
    )
    with Store(tmp_path / "m.db", "k", checkpoint_interval=0) as store:
        wanted = Dense(store, QueryEmbedder(client), timeout=0.5).query_vector("Karim")
    client.close()
    assert wanted.values is None and wanted.dense == "unavailable"
    assert wanted.degraded == "embeddings unreachable"


@pytest.mark.parametrize(
    "overrides",
    [{"resolve_threshold": None, "merge_threshold": None, "link_threshold": None}],
)
def test_an_uncalibrated_model_keeps_dense_recall_and_says_so(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """fake-embed has no profile and no configured thresholds: dense recall works
    (RRF fuses ranks), health says uncalibrated."""
    fact_id = french_fact(client, app, fake_openai)
    assert fact_id in {i["id"] for i in recall(client)["items"]}
    calibration = client.get("/v1/health").json()["vectors"]["calibration"]
    assert calibration == {"status": "uncalibrated", "key": None}


def test_health_reports_the_thresholds_in_use(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    calibration = client.get("/v1/health").json()["vectors"]["calibration"]
    assert calibration == {
        "status": "config",
        "resolve": 0.8,
        "merge": 0.92,
        "link": 0.8,
        "key": None,
        "provisional": False,
    }
