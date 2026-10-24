"""The lifecycle routes (SPEC 8.1): the profile block at session start, turns
enqueued and nothing more, precompact's block and job, session end's job; and
the operations routes' contracts (SPEC 8.4)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.ingest.queue import DONE, PENDING
from tests.api.conftest import auth, remember
from tests.conftest import FakeClock


def profile(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/v1/session/start", json={"scope": "user:alice", **body})
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


# --------------------------------------------------------------------------- session start


def test_session_start_returns_pinned_facts_then_central_nodes_under_budget(
    client: TestClient,
) -> None:
    empty = profile(client, session_id="s1")
    assert empty == {
        "items": [],
        "used_tokens": 0,
        "budget_tokens": 400,
        "omitted": 0,
        "reason": "empty_profile",
        "tier": 0,
        "degraded": None,
        "session_id": "s1",
    }
    karim = remember(
        client,
        "Karim leads the solar installation planning",
        node={"name": "Karim", "kind": "person", "aliases": ["K"]},
    )
    solar = remember(
        client,
        "The solar project starts in March",
        node={"name": "Solar project", "kind": "project"},
    )
    remember(client, "Karim prefers tea", node_id=karim["node"]["id"])
    preference = remember(client, "Prefers short answers", pinned=True)
    client.post(
        "/v1/link",
        json={
            "scope": "user:alice",
            "src": karim["node"]["id"],
            "dst": solar["node"]["id"],
            "type": "works_on",
        },
    )
    block = profile(client, budget_tokens=2000)
    kinds = [(i["kind"], i["id"]) for i in block["items"]]
    assert kinds == [
        ("fact", preference["fact"]["id"]),
        ("node", karim["node"]["id"]),
        ("node", solar["node"]["id"]),
    ]
    pinned = block["items"][0]
    assert pinned["statement"] == "Prefers short answers" and pinned["node_id"] is None
    assert pinned["flags"] == ["pinned"] and "age_days" not in pinned
    node = block["items"][1]
    assert node == {
        "id": karim["node"]["id"],
        "kind": "node",
        "name": "Karim",
        "node_kind": "person",
        "summary": "",
        "aliases": ["K"],
        "facts": 2,
        "edges": 1,
    }
    assert block["items"][2]["facts"] == 1 and block["items"][2]["edges"] == 1
    assert 0 < block["used_tokens"] <= 2000 and block["reason"] is None


def test_session_start_is_deterministic_and_stable_until_something_changes(
    client: TestClient, clock: FakeClock
) -> None:
    for i in range(5):
        remember(client, f"Nadia fact {i}", node={"name": "Nadia"}, pinned=(i % 2 == 0))
    first = profile(client, budget_tokens=3000)
    clock.advance(days=40)  # time passes: the block must not change
    second = profile(client, budget_tokens=3000)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["items"] and all("age_days" not in i for i in first["items"])
    remember(client, "Nadia moved to Lyon", node={"name": "Nadia"}, pinned=True)
    third = profile(client, budget_tokens=3000)
    assert third != second and len(third["items"]) == len(second["items"]) + 1
    archived = next(i for i in third["items"] if i["kind"] == "fact")
    client.post("/v1/archive", json={"id": archived["id"]})
    fourth = profile(client, budget_tokens=3000)
    assert archived["id"] not in {i["id"] for i in fourth["items"]}
    assert (
        profile(client, budget_tokens=3000, exclude_ids=[fourth["items"][0]["id"]])["items"][0][
            "id"
        ]
        != fourth["items"][0]["id"]
    )


def test_session_start_never_exceeds_its_budget_and_never_cuts_an_item(
    client: TestClient,
) -> None:
    for i in range(12):
        remember(client, f"Pinned statement number {i} " + "x" * 80, pinned=True)
    for budget in (0, 40, 150, 400, 5000):
        block = profile(client, budget_tokens=budget)
        assert block["used_tokens"] <= budget and block["budget_tokens"] == budget
        assert len(block["items"]) + block["omitted"] == 12
        if not block["items"]:
            assert block["reason"] == "budget_too_small"
    assert (
        client.post(
            "/v1/session/start", json={"scope": "user:alice", "budget_tokens": 10**6}
        ).status_code
        == 422
    )
    assert client.post("/v1/session/start", json={}).status_code == 422


# --------------------------------------------------------------------------- turns


def test_turns_are_enqueued_and_nothing_else(client: TestClient, app: FastAPI) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})  # keep the job in the queue
    response = client.post(
        "/v1/turns",
        json={
            "scope": "user:alice",
            "session_id": "s1",
            "user": "who leads the solar project?",
            "assistant": "Karim does, since March.",
            "tool_summaries": ["searched the CRM"],
            "external_id": "turn-7",
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"].startswith("job_") and body["kind"] == "turn"
    assert body["status"] == "pending" and body["queue"] == {"depth": 1, "mode": "on_demand"}
    job = app.state.queue.get(body["job_id"])
    assert job.scopes == ("user:alice",) and job.session_id == "s1"
    assert job.payload["user"].startswith("who leads") and job.payload["external_id"] == "turn-7"
    assert job.payload["tool_summaries"] == ["searched the CRM"]
    assert job.payload["occurred_at"].startswith("2026-09-26T12:00")
    # nothing was written to memory: no source, no fact, no node
    assert client.get("/v1/stats").json()["counts"]["sources"] == 0
    assert (
        client.post("/v1/recall", json={"scope": "user:alice", "query": "solar"}).json()["items"]
        == []
    )
    client.post("/v1/background/run")
    assert app.state.queue.get(body["job_id"]).status == DONE


def test_turns_validate_their_input(client: TestClient) -> None:
    assert client.post("/v1/turns", json={"scope": "user:alice"}).status_code == 422
    assert client.post("/v1/turns", json={"scope": "user:alice", "user": "   "}).status_code == 422
    assert client.post("/v1/turns", json={"user": "no scope"}).status_code == 422
    assert (
        client.post(
            "/v1/turns", json={"scope": "user:alice", "user": "x", "extra": True}
        ).status_code
        == 422
    )
    assert (
        client.post("/v1/turns", json={"scope": "user:alice", "user": "x" * 100_001}).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/turns", json={"scope": "user:bob", "user": "hi"}, headers=auth("alice")
        ).status_code
        == 403
    )
    ok = client.post(
        "/v1/turns", json={"scopes": ["user:alice", "user:alice/agent:coder"], "assistant": "hi"}
    )
    assert ok.status_code == 202


# --------------------------------------------------------------------------- precompact


def test_precompact_enqueues_the_messages_and_returns_what_must_survive(
    client: TestClient, app: FastAPI
) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    pinned = remember(client, "Always answer in French", pinned=True)
    response = client.post(
        "/v1/precompact",
        json={
            "scope": "user:alice",
            "session_id": "s1",
            "trigger": "auto",
            "messages": [
                {"role": "user", "content": "long conversation"},
                {"role": "assistant", "content": "about the solar project"},
            ],
            "budget_tokens": 500,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "precompact" and body["status"] == "pending"
    assert [i["id"] for i in body["items"]] == [pinned["fact"]["id"]]
    assert body["used_tokens"] <= body["budget_tokens"] == 500
    job = app.state.queue.get(body["job_id"])
    assert job.payload == {
        "trigger": "auto",
        "messages": [
            {"role": "user", "content": "long conversation"},
            {"role": "assistant", "content": "about the solar project"},
        ],
    }
    assert client.post("/v1/precompact", json={"scope": "user:alice"}).status_code == 200
    assert (
        client.post(
            "/v1/precompact",
            json={"scope": "user:alice", "messages": [{"role": "", "content": "x"}]},
        ).status_code
        == 422
    )
    too_much = [{"role": "user", "content": "x" * 100_000} for _ in range(5)]
    assert (
        client.post(
            "/v1/precompact", json={"scope": "user:alice", "messages": too_much}
        ).status_code
        == 422
    )


def test_precompact_needs_read_for_the_block_but_still_records(client: TestClient) -> None:
    remember(client, "Always answer in French", pinned=True)
    response = client.post(
        "/v1/precompact", json={"scope": "user:alice", "messages": []}, headers=auth("write")
    )
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == [] and body["reason"] == "read_permission_required"
    assert body["job_id"].startswith("job_")


# --------------------------------------------------------------------------- session end


def test_session_end_enqueues_a_flush(client: TestClient, app: FastAPI) -> None:
    client.put("/v1/background", json={"mode": "on_demand"})
    response = client.post(
        "/v1/session/end", json={"scope": "user:alice", "session_id": "s1", "reason": "clear"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "session_end" and body["status"] == PENDING
    job = app.state.queue.get(body["job_id"])
    assert job.payload == {"reason": "clear"} and job.session_id == "s1"
    assert client.post("/v1/session/end", json={}).status_code == 422


# --------------------------------------------------------------------------- operations


def test_background_status_and_run_report_shapes(client: TestClient) -> None:
    status = client.get("/v1/background").json()
    assert status["mode"] == "auto" and status["worker"] == "running"
    assert status["yielded"] is False and status["run_in_progress"] is False
    assert set(status["queue"]) == {
        "depth",
        "pending",
        "running",
        "failed",
        "done",
        "oldest_pending",
    }
    # on_demand first: the auto worker cannot take a job once the switch returned
    client.put("/v1/background", json={"mode": "on_demand"})
    for _ in range(3):
        client.post("/v1/turns", json={"scope": "user:alice", "user": "x"})
    report = client.post("/v1/background/run", json={"budget": {"seconds": 30}}).json()
    assert report["stopped_by"] == "queue_empty" and report["calls"] == 0
    assert report["processed"]["done"] == 3 and report["queue"]["depth"] == 0
    assert client.post("/v1/background/run", json={"budget": {"calls": -1}}).status_code == 422
    assert client.post("/v1/background/run", json={"max_jobs": 0}).status_code == 422


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/v1/turns", {"user": "hi"}),
        ("/v1/precompact", {}),
        ("/v1/session/end", {}),
    ],
)
def test_lifecycle_writes_need_the_token_s_scopes(
    client: TestClient, route: str, body: dict[str, Any]
) -> None:
    assert (
        client.post(route, json={"scope": "user:carol", **body}, headers=auth("alice")).status_code
        == 403
    )
    assert client.post(
        route, json={"scope": "user:alice/agent:coder", **body}, headers=auth("alice")
    ).status_code in (200, 202)


def test_a_bad_scope_is_a_422_naming_the_segment_and_the_expected_form(
    client: TestClient,
) -> None:
    for path, body in (
        ("/v1/turns", {"user": "hello"}),
        ("/v1/recall", {"query": "hello"}),
        ("/v1/session/start", {}),
    ):
        response = client.post(path, json={"scope": "user:alice/assistant", **body})
        assert response.status_code == 422, response.text
        text = json.dumps(response.json())
        assert "segment 'assistant' needs kind:name, like mode:assistant" in text, text
