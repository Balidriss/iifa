"""A background run is visible to every client, not only the one that started it:
``/health`` carries the run in progress (ids and counts), ``GET /background/run/{id}``
its live report, and ``POST /background/run`` has an async form (202 + run id).

No timing: the synchronous run is observed from inside one of its jobs (a test
handler that reads the API while it runs), and the async run's thread is held by
the worker's ``_spawn`` seam, then run inline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.ingest.jobs import NOOP
from iifa.ingest.queue import DONE, Job
from iifa.ingest.worker import JobContext, Worker
from tests.helpers import auth

SCOPE = "user:alice"
PEEK = "peek"
RUN = "run_" + "0" * 26
DREAM = "dream_" + "0" * 26


def enqueue(app: FastAPI, kind: str = NOOP, n: int = 1) -> list[str]:
    worker: Worker = app.state.worker
    return [worker.enqueue(kind, {}, scopes=[SCOPE]).id for _ in range(n)]


def peeking(app: FastAPI, client: TestClient, seen: list[dict[str, Any]]) -> None:
    """A job kind that reads ``/health`` and ``GET /background`` as other tokens
    while the run that took it is in progress."""

    def peek(job: Job, ctx: JobContext) -> None:
        seen.append(
            {
                "health": client.get("/v1/health", headers=auth("read")).json(),
                "background": client.get("/v1/background", headers=auth("admin")).json(),
            }
        )

    app.state.worker.handlers[PEEK] = peek


def hold_spawn(app: FastAPI) -> list[Callable[[], None]]:
    """``start_run`` hands its work here instead of to a thread: the test runs it."""
    held: list[Callable[[], None]] = []
    app.state.worker._spawn = held.append
    return held


def test_health_shows_no_run_when_none_is_in_progress(client: TestClient) -> None:
    assert client.get("/v1/health", headers=auth("read")).json()["background"]["run"] is None
    assert client.get("/v1/background").json()["run"] is None


def test_a_synchronous_run_is_visible_to_another_client_while_it_runs(
    client: TestClient, app: FastAPI
) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    seen: list[dict[str, Any]] = []
    peeking(app, client, seen)
    enqueue(app)
    enqueue(app, PEEK)
    enqueue(app, n=3)
    response = client.post(
        "/v1/background/run",
        json={"budget": {"calls": 40, "seconds": 900}, "max_jobs": 4},
        headers=auth("admin"),
    )
    assert response.status_code == 200
    report = response.json()
    assert len(seen) == 1
    run = seen[0]["health"]["background"]["run"]
    assert run == {
        "run_id": report["run_id"],
        "started_by": "admin",
        "started_at": report["started_at"],
        "budget": {"calls": 40, "tokens": None, "seconds": 900.0, "max_jobs": 4},
        "jobs_done": 1,  # the noop before; the peek itself is still running
        "jobs_left": 2,  # 3 pending, but max_jobs leaves room for 2 more
        "yielded": False,
    }
    background = seen[0]["background"]
    assert background["run"] == run and background["run_in_progress"] is True
    # the synchronous answer keeps its shape, with the run's identity added
    assert report["status"] == "done" and report["stopped_by"] == "max_jobs"
    assert report["started_by"] == "admin" and report["finished_at"] is not None
    assert report["jobs_done"] == 4 and report["processed"]["done"] == 4
    assert client.get("/v1/health").json()["background"]["run"] is None


def test_the_run_summary_says_when_the_background_is_yielded(
    client: TestClient, app: FastAPI
) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    seen: list[dict[str, Any]] = []
    peeking(app, client, seen)
    enqueue(app, PEEK)
    held = hold_spawn(app)
    started = client.post("/v1/background/run", json={"wait": False}).json()
    client.post("/v1/background/yield", json={"timeout_seconds": 600}, headers=auth("write"))
    run = client.get("/v1/health", headers=auth("read")).json()["background"]["run"]
    assert run["run_id"] == started["run_id"] and run["yielded"] is True
    live = client.get(f"/v1/background/run/{started['run_id']}").json()
    assert live["yielded"] is True
    client.post("/v1/background/resume", headers=auth("write"))
    held[0]()
    assert seen[0]["health"]["background"]["run"]["yielded"] is False


def test_an_async_run_answers_202_and_is_followed_by_id(client: TestClient, app: FastAPI) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    jobs = enqueue(app, n=3)
    held = hold_spawn(app)
    response = client.post(
        "/v1/background/run",
        json={"wait": False, "budget": {"tokens": 500}},
        headers=auth("admin"),
    )
    assert response.status_code == 202
    started = response.json()
    assert started["status"] == "running" and started["run_id"].startswith("run_")
    run_id = started["run_id"]
    assert len(held) == 1
    live = client.get(f"/v1/background/run/{run_id}").json()
    assert live["status"] == "running" and live["stopped_by"] is None
    assert live["started_by"] == "admin" and live["finished_at"] is None
    assert live["budget"] == {"calls": None, "tokens": 500, "seconds": None, "max_jobs": None}
    assert live["jobs_done"] == 0 and live["jobs_left"] == 3 and live["yielded"] is False
    health = client.get("/v1/health", headers=auth("read")).json()["background"]["run"]
    assert health["run_id"] == run_id and health["jobs_left"] == 3
    # one run at a time: a second one is refused, and told which run holds it
    busy = client.post("/v1/background/run", json={})
    assert busy.status_code == 409
    assert "busy" in busy.json()["detail"] and run_id in busy.json()["detail"]
    held[0]()  # the run's thread, inline
    done = client.get(f"/v1/background/run/{run_id}").json()
    assert done["status"] == "done" and done["stopped_by"] == "queue_empty"
    assert done["finished_at"] is not None and done["jobs_done"] == 3
    assert [j["id"] for j in done["jobs"]] == jobs
    assert done["jobs_left"] is None and done["yielded"] is None
    assert all(app.state.queue.get(j).status == DONE for j in jobs)
    assert client.get("/v1/health").json()["background"]["run"] is None


def test_an_async_run_is_refused_while_paused(client: TestClient, app: FastAPI) -> None:
    client.put("/v1/background", json={"mode": "paused"})
    held = hold_spawn(app)
    refused = client.post("/v1/background/run", json={"wait": False})
    assert refused.status_code == 409 and "paused" in refused.json()["detail"]
    assert held == []


@pytest.mark.parametrize("run_id", [RUN, DREAM, "nonsense"])
def test_an_unknown_run_is_404(client: TestClient, run_id: str) -> None:
    response = client.get(f"/v1/background/run/{run_id}")
    assert response.status_code == 404 and response.json()["detail"] == "no such run"


def test_wait_must_be_a_boolean(client: TestClient) -> None:
    assert client.post("/v1/background/run", json={"wait": "later"}).status_code == 422
