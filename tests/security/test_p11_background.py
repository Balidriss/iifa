"""P11 - the harness owns the model's time, proven through the HTTP API against
the fake model server's call log (SPEC 12): paused means zero calls, on_demand
means none outside a run, a run stops at its budget, yield is honoured, and a
paused service comes back paused with its queue intact."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import Settings
from iifa.ingest.jobs import MODEL_NOOP
from iifa.ingest.queue import DONE, PENDING
from iifa.ingest.worker import EMBED_KIND
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.helpers import auth, make_settings, wait_until

pytestmark = pytest.mark.security

SCOPE = "user:alice"


def tier2_settings(tmp_path: Path, fake: FakeOpenAIServer, **overrides: Any) -> Settings:
    return make_settings(
        tmp_path,
        chat_url=fake.base_url,
        chat_model="fake-chat",
        embeddings_url=fake.base_url,
        embeddings_model="fake-embed",
        worker_poll=0.02,
        **overrides,
    )


@pytest.fixture
def settings(tmp_path: Path, fake_openai: FakeOpenAIServer) -> Settings:
    return tier2_settings(tmp_path, fake_openai)


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


def enqueue_model_job(app: FastAPI, **payload: Any) -> str:
    job = app.state.worker.enqueue(MODEL_NOOP, payload, scopes=[SCOPE])
    return str(job.id)


def job_status(app: FastAPI, job_id: str) -> str:
    return str(app.state.queue.get(job_id).status)


def loop_passes(app: FastAPI) -> threading.Event:
    """Set each time the worker's loop has looked at the mode and chosen to wait
    (the ``_waiting_for_auto`` seam): clear it, then wait on it, and the loop has
    made a pass since - no fixed sleep."""
    passed = threading.Event()
    app.state.worker._waiting_for_auto = passed.set
    return passed


def gate_waits(app: FastAPI, why: str) -> threading.Event:
    """Set once a model call is held at the gate for ``why`` ("yield", "full",
    "foreground"; the gate's ``_waiting`` seam): the call is provably waiting,
    not merely not yet made - no fixed sleep."""
    held = threading.Event()

    def seen(kind: str, reason: str) -> None:
        if reason == why:
            held.set()

    app.state.worker.gate._waiting = seen
    return held


def test_p11_paused_means_zero_model_calls(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    assert client.get("/v1/health").json()["backends"]["chat"]["model"] == "fake-chat"
    paused = client.put("/v1/background", json={"mode": "paused"})
    assert paused.status_code == 200 and paused.json()["mode"] == "paused"
    turn = client.post("/v1/turns", json={"scope": SCOPE, "user": "hi", "assistant": "hello"})
    assert turn.status_code == 202 and turn.json()["queue"] == {"depth": 1, "mode": "paused"}
    passed = loop_passes(app)
    job = enqueue_model_job(app, calls=2)
    passed.clear()  # the enqueue wakes the loop: its next pass sees the job
    assert passed.wait(timeout=30)
    assert fake_openai.calls == []
    health = client.get("/v1/health").json()
    assert health["queue"]["depth"] == 2 and health["queue"]["oldest_pending"] is not None
    assert health["background"]["mode"] == "paused"
    refused = client.post("/v1/background/run", json={})
    assert refused.status_code == 409 and "paused" in refused.json()["detail"]
    assert fake_openai.calls == [] and job_status(app, job) == PENDING


def test_p11_a_paused_service_comes_back_paused_with_its_queue(
    settings: Settings, clock: FakeClock, fake_openai: FakeOpenAIServer
) -> None:
    # the turn's extraction gets a valid, empty answer: one call, no retry
    fake_openai.set_default(chat=chat_json({"nodes": [], "facts": [], "edges": []}))
    first = create_app(settings, clock=clock)
    with TestClient(first, headers=auth("all")) as client:
        client.put("/v1/background", json={"mode": "paused"})
        client.post("/v1/turns", json={"scope": SCOPE, "user": "kept across restarts"})
        job = enqueue_model_job(first)
        assert client.get("/v1/background").json()["queue"]["depth"] == 2
    # the process restarts; the configuration still says "auto"
    assert settings.background_mode == "auto"
    second = create_app(settings, clock=clock)
    # no fixed sleep: the worker's own loop says when it has seen the mode and
    # chosen to wait (the seam, like _before_claim); a pass made, no call made
    seen_paused = loop_passes(second)
    with TestClient(second, headers=auth("all")) as client:
        status = client.get("/v1/background").json()
        assert status["mode"] == "paused" and status["mode_source"] == "persisted"
        assert status["queue"]["depth"] == 2
        assert seen_paused.wait(timeout=30)
        assert fake_openai.calls == []
        second.state.worker._waiting_for_auto = None
        assert client.put("/v1/background", json={"mode": "auto"}).json()["mode"] == "auto"
        # the turn queued an embed job behind the noop (SPEC 6.1; its open state
        # is pinned by the next test): everything drains once resumed
        assert wait_until(lambda: second.state.queue.stats()["depth"] == 0, timeout=30)
        assert job_status(second, job) == DONE
        # the turn's extraction call, then the noop's ping: both after the resume
        assert len(fake_openai.chat_calls) == 2
        assert fake_openai.chat_calls[-1].messages[-1]["content"] == "ping 1/1"


def test_p11_a_turns_embed_job_counts_in_the_queue_until_it_is_done(
    settings: Settings, clock: FakeClock, fake_openai: FakeOpenAIServer
) -> None:
    """What the test above once missed, made deterministic: with an embeddings
    endpoint, a turn queues an embed job behind the jobs already waiting (SPEC
    6.1); when the last of those is done, the embed job may still be open, and
    the queue depth counts it until it finishes. The embed handler is held on an
    event (no timing), so the state the race produced is reached every time.
    The order is fixed too: the embed job is queued when the turn's extraction
    finishes, so the noop goes in while the worker is paused - else, on a slow
    runner, the embed job could come first and hold the worker."""
    fake_openai.set_default(chat=chat_json({"nodes": [], "facts": [], "edges": []}))
    app = create_app(settings, clock=clock)
    release = threading.Event()
    embed = app.state.worker.handlers[EMBED_KIND]

    def held_embed(job: Any, ctx: Any) -> Any:
        release.wait(30)
        return embed(job, ctx)

    app.state.worker.handlers[EMBED_KIND] = held_embed
    with TestClient(app, headers=auth("all")) as client:
        client.put("/v1/background", json={"mode": "paused"})
        client.post("/v1/turns", json={"scope": SCOPE, "user": "a turn"})
        job = enqueue_model_job(app)
        assert not app.state.queue.has_open(EMBED_KIND)  # queued after the extraction
        client.put("/v1/background", json={"mode": "auto"})
        assert wait_until(lambda: job_status(app, job) == DONE, timeout=30)
        assert app.state.queue.has_open(EMBED_KIND)
        assert client.get("/v1/health").json()["queue"]["depth"] == 1
        release.set()
        assert wait_until(
            lambda: client.get("/v1/health").json()["queue"]["depth"] == 0, timeout=30
        )


def test_p11_on_demand_calls_the_model_only_inside_a_run_that_stops_at_its_budget(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    passed = loop_passes(app)
    jobs = [enqueue_model_job(app) for _ in range(4)]
    passed.clear()
    assert passed.wait(timeout=30)
    assert fake_openai.calls == []
    run = client.post("/v1/background/run", json={"budget": {"calls": 2}})
    assert run.status_code == 200
    report = run.json()
    assert report["run_id"].startswith("run_") and report["stopped_by"] == "calls"
    assert report["calls"] == 2 and report["processed"]["done"] == 2
    assert [j["id"] for j in report["jobs"]] == jobs[:2]
    assert report["queue"]["depth"] == 2
    assert len(fake_openai.chat_calls) == 2
    passed.clear()
    assert passed.wait(timeout=30)
    assert len(fake_openai.chat_calls) == 2  # nothing between runs
    by_jobs = client.post("/v1/background/run", json={"max_jobs": 1}).json()
    assert by_jobs["stopped_by"] == "max_jobs" and by_jobs["calls"] == 1
    rest = client.post("/v1/background/run").json()
    assert rest["stopped_by"] == "queue_empty" and rest["calls"] == 1
    assert len(fake_openai.chat_calls) == 4
    assert all(job_status(app, j) == DONE for j in jobs)
    assert client.get("/v1/background").json()["runs"] == 3


def test_p11_yield_is_honoured(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    assert client.get("/v1/background").json()["mode"] == "auto"
    # long enough that it cannot expire while the test looks (it ends on resume)
    yielded = client.post("/v1/background/yield", json={"timeout_seconds": 600})
    assert yielded.json() == {"yielded": True, "timeout_seconds": 600.0}
    assert client.get("/v1/health").json()["background"]["yielded"] is True
    held = gate_waits(app, "yield")
    job = enqueue_model_job(app)
    # the worker has claimed the job and the gate holds its call for the yield
    assert held.wait(timeout=30)
    assert fake_openai.calls == [] and job_status(app, job) != DONE
    app.state.worker.gate._waiting = None
    assert client.post("/v1/background/resume").json() == {"yielded": False}
    assert wait_until(lambda: job_status(app, job) == DONE)
    assert len(fake_openai.chat_calls) == 1
    # without a resume the yield expires: the default timeout comes from the settings
    default = client.post("/v1/background/yield").json()
    assert default["timeout_seconds"] == app.state.settings.yield_timeout == 30.0
    client.post("/v1/background/resume")
    assert client.post("/v1/background/yield", json={"timeout_seconds": 0}).status_code == 422


def test_p11_concurrency_toward_each_endpoint_is_one_by_default_and_adjustable(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    status = client.get("/v1/background").json()
    assert status["concurrency"] == {"chat": 1, "embeddings": 1}
    changed = client.put("/v1/background", json={"concurrency": {"embeddings": 3}}).json()
    assert changed["concurrency"] == {"chat": 1, "embeddings": 3}
    assert client.put("/v1/background", json={"concurrency": {"chat": 0}}).status_code == 422
    assert client.put("/v1/background", json={"mode": "sometimes"}).status_code == 422
    assert client.put("/v1/background", json={"speed": "fast"}).status_code == 422
    # one chat slot: while a call holds it (the test takes it, standing for a
    # call in flight), the worker's next call waits at the gate - shown by the
    # gate's seam, not by a timing gap
    gate = app.state.worker.gate
    full = gate_waits(app, "full")
    assert gate.acquire("chat", timeout=0)
    job = enqueue_model_job(app)
    assert full.wait(timeout=30)
    assert fake_openai.chat_calls == [] and job_status(app, job) != DONE
    # a second slot lets it through while the first is still held
    gate._waiting = None
    assert client.put("/v1/background", json={"concurrency": {"chat": 2}}).status_code == 200
    assert wait_until(lambda: job_status(app, job) == DONE)
    assert len(fake_openai.chat_calls) == 1
    gate.release("chat")
    # a run goes through the same gate: with one slot held it waits too
    client.put("/v1/background", json={"mode": "on_demand", "concurrency": {"chat": 1}})
    full = gate_waits(app, "full")
    assert gate.acquire("chat", timeout=0)
    second = enqueue_model_job(app)
    answers: list[dict[str, Any]] = []
    run = threading.Thread(target=lambda: answers.append(client.post("/v1/background/run").json()))
    run.start()
    assert full.wait(timeout=30)
    assert len(fake_openai.chat_calls) == 1 and job_status(app, second) != DONE
    gate._waiting = None
    gate.release("chat")
    run.join(timeout=30)
    assert not run.is_alive() and answers[0]["calls"] == 1
    assert job_status(app, second) == DONE and len(fake_openai.chat_calls) == 2


def test_p11_health_reports_the_queue_and_the_backends_without_calling_them(
    client: TestClient, fake_openai: FakeOpenAIServer
) -> None:
    for _ in range(3):
        health = client.get("/v1/health").json()
    assert health["backends"]["chat"]["calls"] == 0
    assert health["backends"]["embeddings"]["model"] == "fake-embed"
    assert health["queue"]["depth"] == 0
    assert fake_openai.calls == []


def test_p11_a_pause_between_the_mode_check_and_the_claim_makes_no_call(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """The race the double run of the suite found, made deterministic: the pause
    lands exactly between the worker's mode check and its claim (a test seam, no
    thread, no timing). The claimed job goes back to the queue untouched and the
    fake model's log stays empty (P11: paused means zero calls)."""
    app = create_app(tier2_settings(tmp_path, fake_openai, background_mode="auto"), clock=clock)
    try:
        worker = app.state.worker  # its thread is not started: _step runs here
        job_id = enqueue_model_job(app, calls=1)
        worker._before_claim = lambda: worker.set_mode("paused")
        assert worker._step() == "released"
        assert fake_openai.calls == []
        assert job_status(app, job_id) == PENDING
        assert app.state.queue.get(job_id).attempts == 0  # not an attempt
        # and nothing more happens while paused
        worker._before_claim = None
        assert worker._step(wait=False) == "not_auto"
        assert fake_openai.calls == []
    finally:
        app.state.store.close()


def test_p11_without_a_pause_the_same_step_processes_the_job(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """The control: the seam alone changes nothing."""
    app = create_app(tier2_settings(tmp_path, fake_openai, background_mode="auto"), clock=clock)
    try:
        worker = app.state.worker
        job_id = enqueue_model_job(app, calls=1)
        worker._before_claim = lambda: None
        assert worker._step() == "processed"
        assert len(fake_openai.chat_calls) == 1 and job_status(app, job_id) == DONE
    finally:
        app.state.store.close()
