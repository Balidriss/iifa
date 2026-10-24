"""P1 - the budget is a contract. Every content-returning route stays at or
under ``budget_tokens`` for any budget, and never truncates an item."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from iifa.recall import handles
from iifa.tokens import ConservativeTokenizer
from tests.api.conftest import remember

pytestmark = pytest.mark.security

COUNTER = ConservativeTokenizer()


def _seed(client: TestClient) -> list[str]:
    ids = []
    for i in range(6):
        ids.append(
            remember(
                client, f"Fact {i}: the solar project budget line {i} " + "detail " * (i * 15)
            )["fact"]["id"]
        )
    ids.append(
        remember(client, "Karim leads the solar project", node={"name": "Karim", "kind": "person"})[
            "fact"
        ]["id"]
    )
    return ids


def _measured(items: list[dict[str, object]]) -> int:
    """What the agent actually pays for the items, by the same counter the service uses."""
    return sum(COUNTER.count(handles.serialized(i)) + COUNTER.per_item_overhead for i in items)


@pytest.mark.parametrize("budget", [0, 1, 25, 60, 120, 250, 400, 1000, 5000])
def test_p1_recall_never_exceeds_budget(client: TestClient, budget: int) -> None:
    _seed(client)
    body = client.post(
        "/v1/recall",
        json={"scope": "user:alice", "query": "solar project budget", "budget_tokens": budget},
    ).json()
    assert body["budget_tokens"] == budget
    assert body["used_tokens"] <= budget
    assert _measured(body["items"]) == body["used_tokens"]
    if not body["items"]:
        assert body["reason"] == "budget_too_small" and body["omitted"] > 0
    for item in body["items"]:  # every handle is whole
        assert set(item) == {
            "id",
            "kind",
            "title",
            "line",
            "why",
            "age_days",
            "expand_tokens",
            "flags",
        }


@pytest.mark.parametrize("budget", [0, 30, 150, 400, 10000])
def test_p1_expand_never_exceeds_budget(client: TestClient, budget: int) -> None:
    ids = _seed(client)
    body = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": ids, "level": 2, "budget_tokens": budget}
    ).json()
    assert body["used_tokens"] <= budget
    assert _measured(body["items"]) == body["used_tokens"]
    for item in body["items"]:
        assert item["statement"].startswith(("Fact", "Karim"))  # complete, never cut
        assert item["excerpt"] == item["statement"]


def test_p1_huge_items_are_skipped_not_truncated(client: TestClient) -> None:
    huge = remember(client, "solar " * 600)["fact"]["id"]
    small = remember(client, "solar panel supplier chosen")["fact"]["id"]
    body = client.post(
        "/v1/recall", json={"scope": "user:alice", "query": "solar", "budget_tokens": 120}
    ).json()
    assert [i["id"] for i in body["items"]] == [small]
    assert body["omitted"] == 1
    expanded = client.post(
        "/v1/expand", json={"scope": "user:alice", "ids": [huge], "budget_tokens": 200}
    ).json()
    assert expanded == {
        **expanded,
        "items": [],
        "omitted": 1,
        "reason": "budget_too_small",
        "used_tokens": 0,
    }


def test_p1_expand_tokens_on_a_handle_is_the_real_cost_of_expanding_it(client: TestClient) -> None:
    _seed(client)
    recall = client.post(
        "/v1/recall", json={"scope": "user:alice", "query": "solar", "budget_tokens": 4000}
    ).json()
    for handle in recall["items"]:
        one = client.post(
            "/v1/expand", json={"scope": "user:alice", "ids": [handle["id"]], "budget_tokens": 4000}
        ).json()
        assert one["used_tokens"] == handle["expand_tokens"]


def test_p1_default_budget_applies_when_omitted(client: TestClient) -> None:
    _seed(client)
    body = client.post("/v1/recall", json={"scope": "user:alice", "query": "solar"}).json()
    assert body["budget_tokens"] == 400 and body["used_tokens"] <= 400


@given(budget=st.integers(min_value=0, max_value=3000))
@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_p1_property_any_budget_holds_on_the_api(client: TestClient, budget: int) -> None:
    if client.get("/v1/stats").json()["counts"]["facts"] == 0:
        _seed(client)
    body = client.post(
        "/v1/recall",
        json={"scope": "user:alice", "query": "solar project", "budget_tokens": budget},
    ).json()
    assert body["used_tokens"] <= budget
    assert len(json.dumps(body["items"])) <= budget * 4 + 2 if budget < 20 else True
