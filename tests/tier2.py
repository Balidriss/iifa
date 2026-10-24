"""Helpers for tests that run the service at tier 2 against the scripted fake
model server (SPEC 13.1, model-dependent layer). The worker is put in
``on_demand`` mode so that every model call happens inside a ``run`` the test
drives: deterministic, never a sleep, never a race."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.config import Settings
from iifa.ingest.queue import DONE
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import make_settings

SCOPE = "user:alice"


def tier2_settings(tmp_path: Path, fake: FakeOpenAIServer, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "chat_url": fake.base_url,
        "chat_model": "fake-chat",
        "background_mode": "on_demand",
        "worker_poll": 0.02,
    }
    values.update(overrides)
    return make_settings(tmp_path, **values)


def empty_answer() -> Reply:
    return chat_json({"nodes": [], "facts": [], "edges": []})


def answer(
    *,
    nodes: list[dict[str, Any]] | None = None,
    facts: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> Reply:
    return chat_json({"nodes": nodes or [], "facts": facts or [], "edges": edges or []})


def run_all(client: TestClient, app: FastAPI, **budget: Any) -> dict[str, Any]:
    """Process the whole queue now and assert every job ended ``done``."""
    body_in: dict[str, Any] = {"budget": {"seconds": 60, **budget}}
    response = client.post("/v1/background/run", json=body_in)
    assert response.status_code == 200, response.text
    report: dict[str, Any] = response.json()
    for job in report["jobs"]:
        assert app.state.queue.get(job["id"]).status == DONE, app.state.queue.get(job["id"])
    return report


def post_turn(
    client: TestClient, user: str, assistant: str = "", *, scope: str = SCOPE, **extra: Any
) -> str:
    body = {"scope": scope, "user": user, "assistant": assistant, **extra}
    response = client.post("/v1/turns", json=body)
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])


def end_session(client: TestClient, session_id: str, *, scope: str = SCOPE) -> str:
    response = client.post("/v1/session/end", json={"scope": scope, "session_id": session_id})
    assert response.status_code == 202, response.text
    return str(response.json()["job_id"])
