"""Shared pieces of the dream tests (SPEC 5.6): a small memory to dream over,
a scripted dream model behind the fake OpenAI server, and a direct line to the
applier for the operation-level tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import Settings
from iifa.dream.apply import Applier
from iifa.dream.journal import Journal, Run
from iifa.dream.policy import ReviewPolicy
from iifa.store.models import Fact, Node
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, Respond, chat_json
from tests.helpers import auth, make_settings

SCOPE = "user:alice"
MEMORY_START = "<<<MEMORY>>>"
MEMORY_END = "<<<END>>>"


@dataclass
class Memory:
    """Karim with a few facts, Kiki (a second node for Karim), Lyon."""

    karim: Node
    kiki: Node
    lyon: Node
    home: Fact  # extracted, home = Lyon
    tea: Fact  # extracted, routine = morning drink
    green_tea: Fact  # extracted, overlaps tea: the same key and value
    pinned: Fact  # explicit, pinned


def build_memory(store: Store, scope: str = SCOPE) -> Memory:
    karim = store.create_node(scopes=[scope], name="Karim", kind="person", aliases=["K."])
    # Kiki shares the alias "K.": their names fit, so a `same` stays one (SPEC 5.6)
    kiki = store.create_node(scopes=[scope], name="Kiki", kind="person", aliases=["K."])
    lyon = store.create_node(scopes=[scope], name="Lyon", kind="place")
    home = store.add_fact(
        scopes=[scope],
        node_id=karim.id,
        statement="Karim lives in Lyon.",
        origin="extracted",
        attribute="home",
        value="Lyon",
        language="en",
    )
    tea = store.add_fact(
        scopes=[scope],
        node_id=karim.id,
        statement="Karim likes tea in the morning.",
        origin="extracted",
        attribute="routine",
        value="morning drink",
        language="en",
    )
    green = store.add_fact(
        scopes=[scope],
        node_id=karim.id,
        statement="Karim drinks green tea every morning.",
        origin="extracted",
        attribute="routine",
        value="morning drink",
        language="en",
    )
    pinned = store.add_fact(
        scopes=[scope],
        node_id=karim.id,
        statement="Karim is allergic to peanuts.",
        origin="explicit",
        pinned=True,
        language="en",
    )
    return Memory(karim, kiki, lyon, home, tea, green, pinned)


@pytest.fixture
def memory(store: Store) -> Memory:
    return build_memory(store)


@pytest.fixture
def journal(store: Store) -> Journal:
    return Journal(store)


@pytest.fixture
def applier(store: Store, journal: Journal) -> Applier:
    return Applier(store, journal)


def new_run(
    journal: Journal,
    *,
    scope: str = SCOPE,
    dry_run: bool = False,
    policy: ReviewPolicy | None = None,
) -> Run:
    return journal.create_run(
        scope=scope,
        dry_run=dry_run,
        operations=None,
        budget={},
        policy=policy or ReviewPolicy.default(),
    )


def submit(
    store: Store,
    applier: Applier,
    run: Run,
    op: dict[str, Any],
    *,
    ids: list[str] | None = None,
) -> Any:
    """Submit one operation as the runner would, with a snapshot of the records
    it names taken now (``ids``, default every id in the operation)."""
    named = ids if ids is not None else _ids_in(op)
    return applier.submit(run, op, store.revisions(named))


def _ids_in(op: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for value in op.values():
        if (
            isinstance(value, str)
            and "_" in value
            and value.split("_", 1)[0]
            in (
                "node",
                "fact",
                "edge",
            )
        ):
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str) and "_" in v)
    return out


# --------------------------------------------------------------------------- the dream model


def unit_of(body: Any) -> dict[str, Any]:
    """The unit a dream call carries, read back from the request the service sent."""
    content = body["messages"][-1]["content"]
    start = content.index(MEMORY_START) + len(MEMORY_START)
    end = content.rindex(MEMORY_END)
    unit: dict[str, Any] = json.loads(content[start:end])
    return unit


def is_adjudication(body: Any) -> bool:
    """An adjudication call (SPEC 5.6) rather than a unit's call."""
    return '"verdict"' in str(body["messages"][0]["content"])


def pair_of(body: Any) -> dict[str, Any]:
    """The pair an adjudication call carries: ``{"a": {...}, "b": {...}}``."""
    return unit_of(body)


Verdict = Callable[[dict[str, Any]], dict[str, Any] | str]


def dream_model(
    plan: Callable[[dict[str, Any]], list[dict[str, Any]]],
    verdict: Verdict | None = None,
) -> Respond:
    """A scripted dream model: ``plan`` reads the unit and returns operations;
    ``verdict`` reads an adjudicated pair and returns the answer object (or raw
    text); without it every pair is ``unsure``."""

    def answer(body: Any) -> Reply:
        if is_adjudication(body):
            said = (verdict or (lambda pair: {"verdict": "unsure", "reason": "?"}))(pair_of(body))
            return Reply(said) if isinstance(said, str) else chat_json(said)
        return chat_json({"operations": plan(unit_of(body))})

    return Respond(answer)


def adjudication_calls(fake: FakeOpenAIServer) -> list[dict[str, Any]]:
    return [pair_of(c.body) for c in fake.chat_calls if is_adjudication(c.body)]


def unit_calls(fake: FakeOpenAIServer) -> list[dict[str, Any]]:
    return [unit_of(c.body) for c in fake.chat_calls if not is_adjudication(c.body)]


def quiet_model() -> Respond:
    """A dream model that proposes nothing."""
    return dream_model(lambda unit: [])


def dream_settings(tmp_path: Path, fake: FakeOpenAIServer, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "dream_url": fake.base_url,
        "dream_model": "fake-dream",
        "background_mode": "on_demand",
        "worker_poll": 0.02,
    }
    values.update(overrides)
    return make_settings(tmp_path, **values)


# the operator's review declarations, (scope, risk, auto | review); a module
# whose tests need consolidations applied overrides this fixture
MEDIUM_AUTO: tuple[tuple[str, str, str], ...] = (("*", "medium", "auto"),)


@pytest.fixture
def dream_review() -> tuple[tuple[str, str, str], ...]:
    return ()


@pytest.fixture
def dream_app(
    tmp_path: Path,
    fake_openai: FakeOpenAIServer,
    clock: FakeClock,
    dream_review: tuple[tuple[str, str, str], ...],
) -> Iterator[FastAPI]:
    settings = dream_settings(tmp_path, fake_openai, dream_review=dream_review)
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def dream_client(dream_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(dream_app, headers=auth("all")) as test_client:
        yield test_client


def start_dream(client: TestClient, scope: str = SCOPE, **body: Any) -> str:
    response = client.post("/v1/dream", json={"scope": scope, **body})
    assert response.status_code == 202, response.text
    return str(response.json()["run_id"])


def run_queue(client: TestClient, **budget: Any) -> dict[str, Any]:
    response = client.post("/v1/background/run", json={"budget": {"seconds": 60, **budget}})
    assert response.status_code == 200, response.text
    report: dict[str, Any] = response.json()
    return report


def dream_report(client: TestClient, run_id: str) -> dict[str, Any]:
    response = client.get(f"/v1/dream/{run_id}")
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def dream(client: TestClient, scope: str = SCOPE, **body: Any) -> dict[str, Any]:
    """Start a dream, run the queue, return the finished run."""
    run_id = start_dream(client, scope, **body)
    run_queue(client)
    return dream_report(client, run_id)


def plan_of(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    response = client.get(f"/v1/dream/{run_id}/plan", params={"budget_tokens": 16000})
    assert response.status_code == 200, response.text
    items: list[dict[str, Any]] = response.json()["items"]
    return items


def remember_in(client: TestClient, statement: str, scope: str = SCOPE, **extra: Any) -> str:
    response = client.post("/v1/remember", json={"scope": scope, "statement": statement, **extra})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])
