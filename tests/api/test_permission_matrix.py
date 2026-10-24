"""Every route x every permission: allowed, or 403. Generated from the route
registry so a new route cannot appear without a declared permission."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.auth import ANY, ROUTE_PERMISSIONS
from iifa.config import PERMISSIONS
from tests.api.conftest import auth

NODE = "node_" + "0" * 26
FACT = "fact_" + "0" * 26
PROP = "prop_" + "0" * 26
PSET = "pset_" + "0" * 26
DREAM = "dream_" + "0" * 26
DOP = "dop_" + "0" * 26

# A minimal, well-formed request for each route (so the only reason for a 4xx is auth).
SAMPLES: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {
    ("GET", "/v1/health"): ("/v1/health", {}),
    ("GET", "/v1/stats"): ("/v1/stats", {}),
    ("POST", "/v1/recall"): ("/v1/recall", {"json": {"scope": "user:alice", "query": "solar"}}),
    ("POST", "/v1/expand"): ("/v1/expand", {"json": {"scope": "user:alice", "ids": [FACT]}}),
    ("GET", "/v1/nodes/{node_id}"): (f"/v1/nodes/{NODE}", {"params": {"scope": "user:alice"}}),
    ("GET", "/v1/lookup"): ("/v1/lookup", {"params": {"scope": "user:alice", "external_id": "x"}}),
    ("POST", "/v1/timeline"): ("/v1/timeline", {"json": {"scope": "user:alice"}}),
    ("POST", "/v1/at"): (
        "/v1/at",
        {"json": {"scope": "user:alice", "node_id": NODE, "date": "2026-01-01"}},
    ),
    ("POST", "/v1/path"): ("/v1/path", {"json": {"scope": "user:alice", "src": NODE, "dst": NODE}}),
    ("GET", "/v1/archive"): ("/v1/archive", {"params": {"scope": "user:alice"}}),
    ("POST", "/v1/remember"): ("/v1/remember", {"json": {"scope": "user:alice", "statement": "x"}}),
    ("POST", "/v1/link"): (
        "/v1/link",
        {"json": {"scope": "user:alice", "src": NODE, "dst": NODE, "type": "knows"}},
    ),
    ("POST", "/v1/archive"): ("/v1/archive", {"json": {"id": FACT}}),
    ("POST", "/v1/restore"): ("/v1/restore", {"json": {"id": FACT}}),
    ("POST", "/v1/forget"): ("/v1/forget", {"json": {"id": FACT}}),
    ("PATCH", "/v1/facts/{fact_id}"): (f"/v1/facts/{FACT}", {"json": {"pinned": True}}),
    ("POST", "/v1/proposals/{proposal_id}"): (
        f"/v1/proposals/{PROP}",
        {"json": {"decision": "reject"}},
    ),
    ("GET", "/v1/proposals"): ("/v1/proposals", {"params": {"scope": "user:alice"}}),
    ("POST", "/v1/session/start"): ("/v1/session/start", {"json": {"scope": "user:alice"}}),
    ("POST", "/v1/turns"): ("/v1/turns", {"json": {"scope": "user:alice", "user": "hello"}}),
    ("POST", "/v1/precompact"): ("/v1/precompact", {"json": {"scope": "user:alice"}}),
    ("POST", "/v1/session/end"): ("/v1/session/end", {"json": {"scope": "user:alice"}}),
    ("GET", "/v1/background"): ("/v1/background", {}),
    ("PUT", "/v1/background"): ("/v1/background", {"json": {"mode": "on_demand"}}),
    ("POST", "/v1/background/run"): ("/v1/background/run", {"json": {"max_jobs": 1}}),
    ("POST", "/v1/background/yield"): ("/v1/background/yield", {"json": {"timeout_seconds": 1}}),
    ("POST", "/v1/background/resume"): ("/v1/background/resume", {}),
    ("POST", "/v1/probes"): (
        "/v1/probes",
        {
            "json": {
                "scope": "user:alice",
                "name": "n",
                "probes": [{"query": "q", "expected": [FACT]}],
            }
        },
    ),
    ("GET", "/v1/probes"): ("/v1/probes", {"params": {"scope": "user:alice"}}),
    ("GET", "/v1/probes/{set_id}"): (f"/v1/probes/{PSET}", {}),
    ("POST", "/v1/probes/{set_id}/run"): (f"/v1/probes/{PSET}/run", {}),
    ("DELETE", "/v1/probes/{set_id}"): (f"/v1/probes/{PSET}", {}),
    ("POST", "/v1/dream"): ("/v1/dream", {"json": {"scope": "user:alice", "dry_run": True}}),
    ("GET", "/v1/dream/{run_id}"): (f"/v1/dream/{DREAM}", {}),
    ("GET", "/v1/dream/{run_id}/plan"): (f"/v1/dream/{DREAM}/plan", {}),
    ("POST", "/v1/dream/{run_id}/undo"): (f"/v1/dream/{DREAM}/undo", {"json": {}}),
    ("POST", "/v1/dream/{run_id}/cancel"): (f"/v1/dream/{DREAM}/cancel", {}),
    ("GET", "/v1/review"): ("/v1/review", {"params": {"scope": "user:alice"}}),
    ("POST", "/v1/review/{item_id}"): (f"/v1/review/{DOP}", {"json": {"decision": "reject"}}),
}


def _routes(app: FastAPI) -> set[tuple[str, str]]:
    """Every (method, path) the app exposes, from the OpenAPI document."""
    return {
        (method.upper(), path)
        for path, methods in app.openapi()["paths"].items()
        for method in methods
        if path.startswith("/v1")
    }


def test_every_route_declares_a_permission_and_has_a_sample(app: FastAPI) -> None:
    routes = _routes(app)
    assert routes, "no routes?"
    assert routes == set(ROUTE_PERMISSIONS), "a route without a declared permission"
    assert routes == set(SAMPLES), "a route without a sample request in the matrix"
    assert set(ROUTE_PERMISSIONS.values()) <= PERMISSIONS | {ANY}


@pytest.mark.security
@pytest.mark.parametrize("route", sorted(SAMPLES))
@pytest.mark.parametrize("held", sorted(PERMISSIONS))
def test_permission_matrix(client: TestClient, route: tuple[str, str], held: str) -> None:
    method, _ = route
    path, kwargs = SAMPLES[route]
    needed = ROUTE_PERMISSIONS[route]
    response = client.request(method, path, headers=auth(held), **kwargs)
    if needed in (ANY, held):
        assert response.status_code not in (401, 403), (route, held, response.text)
    else:
        assert response.status_code == 403, (route, held, response.text)
        assert "permission" in response.json()["detail"]


@pytest.mark.security
@pytest.mark.parametrize("route", sorted(SAMPLES))
def test_no_token_and_bad_token_are_401(client: TestClient, route: tuple[str, str]) -> None:
    method, _ = route
    path, kwargs = SAMPLES[route]
    for headers in (
        {"Authorization": ""},
        {"Authorization": "Bearer nope"},
        {"Authorization": "Basic dXNlcjpwdw=="},
    ):
        response = client.request(method, path, headers=headers, **kwargs)
        assert response.status_code == 401, (route, headers, response.text)
        assert response.headers["WWW-Authenticate"] == "Bearer"
