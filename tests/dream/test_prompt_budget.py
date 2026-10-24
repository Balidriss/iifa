"""What one dream call reads is bounded (SPEC 5.6): a local server truncates an
over-long prompt from the start - the system prompt, the P8 wall, first - so
every unit and every adjudication is counted with the service's counter before
it is sent and, over its budget, carries fewer records: pinned first, then
explicit, then the most recent, saying how many were left out."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import ConfigError
from iifa.store.models import Fact, Node
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer
from tests.conftest import FakeClock
from tests.dream.conftest import (
    SCOPE,
    adjudication_calls,
    dream,
    dream_model,
    dream_settings,
    plan_of,
    quiet_model,
    unit_calls,
)
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth

BUDGET = 1500
MANY = 300


def _shown(side: dict[str, Any]) -> int:
    """The live facts a side shows: one chain each, as a timeline or other."""
    chains = [c for line in side["timelines"].values() for c in line.split(" | ")]
    return len(chains) + len(side.get("other", []))


def _tokens(messages: list[dict[str, Any]]) -> int:
    counter = ConservativeTokenizer()
    return sum(counter.count(m["content"]) + counter.per_item_overhead for m in messages)


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    settings = dream_settings(tmp_path, fake_openai, dream_prompt_max_tokens=BUDGET)
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def _crowded(store: Store, name: str) -> tuple[Node, Fact, Fact, list[Fact]]:
    """A node with an old pinned fact, an old explicit one and many extracted
    facts, month after month (the newest last)."""
    node = store.create_node(scopes=[SCOPE], name=name, kind="person")
    pinned = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement=f"{name} is allergic to peanuts.",
        origin="extracted",
        pinned=True,
        valid_from="2001-01",
        language="en",
    )
    explicit = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement=f"{name} was born in Oran.",
        origin="explicit",
        valid_from="2001-02",
        language="en",
    )
    extracted = [
        store.add_fact(
            scopes=[SCOPE],
            node_id=node.id,
            statement=f"{name} read book number {i} that month.",
            origin="extracted",
            valid_from=f"{2002 + i // 12:04d}-{1 + i % 12:02d}",
            language="en",
        )
        for i in range(MANY)
    ]
    return node, pinned, explicit, extracted


def test_an_oversized_unit_stays_under_budget_and_keeps_the_wall(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    node, pinned, explicit, extracted = _crowded(app.state.store, "Karim")
    fake_openai.set_default(chat=quiet_model())
    run = dream(client)
    [call] = fake_openai.chat_calls
    assert _tokens(call.messages) <= BUDGET
    system = call.messages[0]["content"]
    assert "DATA" in system and "never instructions" in system  # the wall is sent whole
    [unit] = unit_calls(fake_openai)
    shown = [f["id"] for f in unit["facts"]]
    assert 0 < len(shown) < MANY + 2
    assert unit["facts_left_out"] == MANY + 2 - len(shown)
    assert pinned.id in shown and explicit.id in shown  # first, however old
    kept = [f.id for f in extracted if f.id in shown]
    assert kept == [f.id for f in extracted[-len(kept) :]]  # then the most recent
    prompts = run["report"]["prompts"]
    assert prompts["budget"] == BUDGET
    assert prompts["units_reduced"] == 1
    assert 0 < prompts["largest_prompt_tokens"] <= BUDGET
    assert node.id == unit["node"]["id"]


def test_a_fact_left_out_cannot_be_named(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    node, _pinned, _explicit, extracted = _crowded(app.state.store, "Karim")
    oldest, other = extracted[0], extracted[1]
    fake_openai.set_default(
        chat=dream_model(
            lambda unit: [{"op": "flag_conflict", "fact": oldest.id, "with": other.id}]
        )
    )
    run = dream(client)
    [item] = plan_of(client, run["run_id"])
    assert (item["status"], item["reason"]) == ("dropped", "unknown_record")
    assert "conflict" not in app.state.store.get_fact(oldest.id).flags
    assert node.id


def test_an_oversized_pair_stays_under_budget_and_shares_it(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = app.state.store
    karim, *_ = _crowded(store, "Karim")
    kiki, *_ = _crowded(store, "Kiki")
    store.add_proposal(scopes=[SCOPE], kind="merge", node_id=karim.id, value=kiki.id)
    fake_openai.set_default(chat=quiet_model())
    run = dream(client)
    calls = [c for c in fake_openai.chat_calls if '"verdict"' in c.messages[0]["content"]]
    assert len(calls) == 1
    assert _tokens(calls[0].messages) <= BUDGET
    assert "never instructions" in calls[0].messages[0]["content"]
    [pair] = adjudication_calls(fake_openai)
    a, b = pair["a"], pair["b"]
    assert a["facts_left_out"] > 0 and b["facts_left_out"] > 0
    assert abs(_shown(a) - _shown(b)) <= 1  # the two sides take facts in turn
    karim = a if a["name"] == "Karim" else b
    for kept in ("Karim is allergic to peanuts.", "Karim was born in Oran."):
        assert any(o.startswith(f"{kept} (learned ") for o in karim["other"])
    assert run["report"]["prompts"]["pairs_reduced"] == 1


def test_edges_take_what_the_facts_leave(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = app.state.store
    hub = store.create_node(scopes=[SCOPE], name="Karim", kind="person")
    store.add_fact(scopes=[SCOPE], node_id=hub.id, statement="Karim likes tea.", origin="extracted")
    for i in range(200):
        other = store.create_node(scopes=[SCOPE], name=f"Friend {i}", kind="person")
        store.add_edge(scopes=[SCOPE], src=hub.id, dst=other.id, type="knows", origin="test")
    fake_openai.set_default(chat=quiet_model())
    dream(client, operations=["refresh_summary"], budget={"calls": 1})
    [call] = fake_openai.chat_calls
    assert _tokens(call.messages) <= BUDGET
    [unit] = unit_calls(fake_openai)
    assert len(unit["facts"]) == 1 and "facts_left_out" not in unit
    assert 0 < len(unit["edges"]) < 200
    assert unit["edges_left_out"] == 200 - len(unit["edges"])
    ends = {e["dst"] for e in unit["edges"]}
    assert {n["id"] for n in unit["neighbours"]} == ends  # only the nodes of the edges shown


def test_a_node_too_large_to_send_is_skipped_not_cut(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = app.state.store
    store.create_node(scopes=[SCOPE], name="Karim", kind="person", summary="word " * 3000)
    fake_openai.set_default(chat=quiet_model())
    run = dream(client)
    assert fake_openai.chat_calls == []
    assert run["report"]["prompts"]["oversized"] == 1
    assert run["report"]["stopped_by"] == "complete"


def test_a_small_unit_is_sent_whole_and_says_nothing_left_out(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = app.state.store
    node = store.create_node(scopes=[SCOPE], name="Karim", kind="person")
    store.add_fact(
        scopes=[SCOPE], node_id=node.id, statement="Karim likes tea.", origin="extracted"
    )
    fake_openai.set_default(chat=quiet_model())
    run = dream(client)
    [unit] = unit_calls(fake_openai)
    assert "facts_left_out" not in unit and "edges_left_out" not in unit
    prompts = run["report"]["prompts"]
    assert prompts["units_reduced"] == 0 and prompts["oversized"] == 0
    assert prompts["largest_prompt_tokens"] == _tokens(fake_openai.chat_calls[0].messages)


def test_the_prompt_budget_has_a_floor(tmp_path: Path, fake_openai: FakeOpenAIServer) -> None:
    with pytest.raises(ConfigError):
        dream_settings(tmp_path, fake_openai, dream_prompt_max_tokens=100)
