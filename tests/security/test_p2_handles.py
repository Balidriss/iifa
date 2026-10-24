"""P2 - progressive disclosure: recall returns handles, content comes on expand."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import remember

pytestmark = pytest.mark.security

LONG = (
    "Karim signed the contract with the supplier on 3 March 2026 after three rounds of negotiation, "
    * 4
)


def test_p2_recall_returns_handles_not_content(client: TestClient) -> None:
    fact = remember(client, LONG.strip(), node={"name": "Karim", "kind": "person"})
    body = client.post(
        "/v1/recall",
        json={"scope": "user:alice", "query": "contract supplier", "budget_tokens": 2000},
    ).json()
    handle = next(i for i in body["items"] if i["id"] == fact["fact"]["id"])
    assert set(handle) == {
        "id",
        "kind",
        "title",
        "line",
        "why",
        "age_days",
        "expand_tokens",
        "flags",
    }
    assert LONG.strip() not in handle["line"] and LONG.strip() not in handle["title"]
    assert len(handle["line"]) <= 140 and handle["line"].endswith("…")
    assert handle["expand_tokens"] > 0 and handle["age_days"] == 0
    assert handle["why"] == "query terms 'contract', 'supplier'"
    assert "statement" not in handle and "excerpt" not in handle


def test_p2_expand_delivers_the_content_the_handle_pointed_to(client: TestClient) -> None:
    fact = remember(client, LONG.strip(), node={"name": "Karim", "kind": "person"})
    body = client.post(
        "/v1/expand",
        json={
            "scope": "user:alice",
            "ids": [fact["fact"]["id"]],
            "level": 2,
            "budget_tokens": 2000,
        },
    ).json()
    assert body["items"][0]["statement"] == LONG.strip()
    assert body["items"][0]["excerpt"] == LONG.strip()
    assert body["items"][0]["node_name"] == "Karim"


def test_p2_node_handles_carry_size_and_count_not_facts(client: TestClient, clock: object) -> None:
    remember(
        client,
        "Karim likes tea",
        node={"name": "Karim", "kind": "person", "summary": "An electrician in Lyon"},
    )
    remember(client, "Karim moved to Lyon", node={"name": "Karim"})
    body = client.post(
        "/v1/recall", json={"scope": "user:alice", "query": "Karim", "budget_tokens": 2000}
    ).json()
    node = next(i for i in body["items"] if i["kind"] == "node")
    assert node["title"] == "Karim" and node["line"] == "An electrician in Lyon"
    assert "facts" not in node and node["expand_tokens"] > 0
