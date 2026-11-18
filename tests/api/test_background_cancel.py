"""A background run can be stopped early (``POST /background/run/{id}/cancel``,
admin): no new job starts, a job in flight that makes no further model call
finishes (one that does is put back, see tests/model/test_worker.py), the run
ends with ``stopped_by: cancelled`` and the mode is unchanged. ``/health`` and the run's
report say a cancel was asked.

No timing: the async run's thread is held by the worker's ``_spawn`` seam, then
run inline; a cancel in the middle of a run is sent from inside one of its jobs,
and one between a run's last check and its claim through ``_before_run_claim``."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.ingest.jobs import NOOP
from iifa.ingest.queue import DONE, PENDING, Job
from iifa.ingest.worker import JobContext, Worker
from tests.helpers import auth

SCOPE = "user:alice"
CANCEL = "cancel"
RUN = "run_" + "0" * 26
DREAM = "dream_" + "0" * 26


def enqueue(app: FastAPI, kind: str = NOOP, n: int = 1) -> list[str]:
    worker: Worker = app.state.worker
    return [worker.enqueue(kind, {}, scopes=[SCOPE]).id for _ in range(n)]


def hold_spawn(app: FastAPI) -> list[Callable[[], None]]:
    """``start_run`` hands its work here instead of to a thread: the test runs it."""
    held: list[Callable[[], None]] = []
    app.state.worker._spawn = held.append
    return held


def status_of(app: FastAPI, job_id: str) -> str:
    return str(app.state.queue.get(job_id).status)


def start_held(client: TestClient, app: FastAPI) -> tuple[str, list[Callable[[], None]]]:
    held = hold_spawn(app)
    started = client.post("/v1/background/run", json={"wait": False}, headers=auth("admin"))
    assert started.status_code == 202
    return str(started.json()["run_id"]), held


def test_a_cancelled_run_starts_no_job_and_says_cancelled(client: TestClient, app: FastAPI) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    jobs = enqueue(app, n=2)
    run_id, held = start_held(client, app)
    response = client.post(f"/v1/background/run/{run_id}/cancel", headers=auth("admin"))
    assert response.status_code == 200
    assert response.json() == {"run_id": run_id, "status": "cancelling"}
    # every client sees that a cancel was asked, while the run is still in progress
    health = client.get("/v1/health", headers=auth("read")).json()["background"]["run"]
    assert health["run_id"] == run_id and health["cancel_requested"] is True
    assert health["jobs_left"] == 0  # it will start none of the pending jobs
    live = client.get(f"/v1/background/run/{run_id}").json()
    assert live["status"] == "running" and live["cancel_requested"] is True
    assert live["cancelled_by"] == "admin" and live["jobs_left"] == 0
    # asking twice is harmless, and the first token stays the one named
    again = client.post(f"/v1/background/run/{run_id}/cancel", headers=auth("all"))
    assert again.status_code == 200 and again.json()["status"] == "cancelling"
    held[0]()  # the run's thread, inline
    done = client.get(f"/v1/background/run/{run_id}").json()
    assert done["status"] == "done" and done["stopped_by"] == "cancelled"
    assert done["jobs"] == [] and done["cancel_requested"] is True
    assert done["cancelled_by"] == "admin"
    assert all(status_of(app, j) == PENDING for j in jobs)
    # the mode is unchanged, and a later run takes the jobs
    assert client.get("/v1/background").json()["mode"] == "on_demand"
    later = client.post("/v1/background/run", json={}).json()
    assert later["stopped_by"] == "queue_empty" and later["cancel_requested"] is False
    assert all(status_of(app, j) == DONE for j in jobs)


def test_the_job_in_flight_finishes_and_the_next_one_never_starts(
    client: TestClient, app: FastAPI
) -> None:
    """A synchronous run, cancelled by another client from inside its first job."""
    client.put("/v1/background", json={"mode": "on_demand"})
    answers: list[int] = []

    def cancel_from_another_client(job: Job, ctx: JobContext) -> None:
        run = client.get("/v1/health", headers=auth("read")).json()["background"]["run"]
        sent = client.post(f"/v1/background/run/{run['run_id']}/cancel", headers=auth("admin"))
        answers.append(sent.status_code)

    app.state.worker.handlers[CANCEL] = cancel_from_another_client
    [first] = enqueue(app, CANCEL)
    rest = enqueue(app, n=2)
    report = client.post("/v1/background/run", json={}, headers=auth("admin")).json()
    assert answers == [200]
    assert report["stopped_by"] == "cancelled" and report["cancel_requested"] is True
    assert [(j["id"], j["outcome"]) for j in report["jobs"]] == [(first, "done")]
    assert status_of(app, first) == DONE
    assert all(status_of(app, j) == PENDING for j in rest)
    assert client.get("/v1/health").json()["background"]["run"] is None


def test_a_cancel_between_the_check_and_the_claim_starts_no_job(
    client: TestClient, app: FastAPI
) -> None:
    """The cancel lands after the run's last check, before it claims: the
    claimed job goes back untouched, and it never runs."""
    client.put("/v1/background", json={"mode": "on_demand"})
    ran: list[str] = []
    app.state.worker.handlers[CANCEL] = lambda job, ctx: ran.append(job.id)
    [job] = enqueue(app, CANCEL)
    run_id, held = start_held(client, app)
    worker: Worker = app.state.worker

    def cancel_now() -> None:
        worker._before_run_claim = None
        assert worker.cancel_run(run_id) == "cancelling"

    worker._before_run_claim = cancel_now
    held[0]()
    done = client.get(f"/v1/background/run/{run_id}").json()
    assert done["stopped_by"] == "cancelled" and done["jobs"] == []
    assert ran == [] and status_of(app, job) == PENDING
    assert app.state.queue.get(job).attempts == 0


def test_a_finished_run_cannot_be_cancelled(client: TestClient, app: FastAPI) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    report = client.post("/v1/background/run", json={}).json()
    response = client.post(f"/v1/background/run/{report['run_id']}/cancel")
    assert response.status_code == 409 and response.json()["detail"] == "the run is done"
    after = client.get(f"/v1/background/run/{report['run_id']}").json()
    assert after["stopped_by"] == "queue_empty" and after["cancel_requested"] is False
    assert after["cancelled_by"] is None


@pytest.mark.security
@pytest.mark.parametrize("why", ["paused", "shutdown"])
def test_a_pause_or_a_shutdown_between_the_check_and_the_claim_starts_no_job(
    client: TestClient, app: FastAPI, why: str
) -> None:
    """P11 inside a run: a pause (or a shutdown) that lands after the last check
    of the run, before its claim, starts no job: the claimed one goes back."""
    client.put("/v1/background", json={"mode": "on_demand"})
    ran: list[str] = []
    app.state.worker.handlers[CANCEL] = lambda job, ctx: ran.append(job.id)
    [job] = enqueue(app, CANCEL)
    run_id, held = start_held(client, app)
    worker: Worker = app.state.worker

    def interrupt_now() -> None:
        worker._before_run_claim = None
        if why == "paused":
            worker.set_mode("paused")
        else:
            with worker._cond:
                worker._stop = True

    worker._before_run_claim = interrupt_now
    held[0]()
    with worker._cond:
        worker._stop = False
    done = worker.run_report(run_id)
    assert done is not None and done["stopped_by"] == why and done["jobs"] == []
    assert ran == [] and status_of(app, job) == PENDING


@pytest.mark.parametrize("run_id", [RUN, DREAM, "nonsense"])
def test_cancelling_an_unknown_run_is_404(client: TestClient, run_id: str) -> None:
    response = client.post(f"/v1/background/run/{run_id}/cancel")
    assert response.status_code == 404 and response.json()["detail"] == "no such run"


@pytest.mark.security
@pytest.mark.parametrize("token", ["read", "write", "forget", "dream", "review", "alice"])
def test_a_token_without_admin_cannot_cancel_a_run(
    client: TestClient, app: FastAPI, token: str
) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    [job] = enqueue(app)
    run_id, held = start_held(client, app)
    refused = client.post(f"/v1/background/run/{run_id}/cancel", headers=auth(token))
    assert refused.status_code == 403
    live = client.get(f"/v1/background/run/{run_id}").json()
    assert live["cancel_requested"] is False
    held[0]()
    done = client.get(f"/v1/background/run/{run_id}").json()
    assert done["stopped_by"] == "queue_empty" and status_of(app, job) == DONE
