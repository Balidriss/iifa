"""The speaker through the write path (SPEC 4.2, 4.6): a subject naming the
person speaking is never rejected and never becomes a node; it is attached to
the scope's declared owner, or kept free-standing - and then, since several
people may speak in one scope, never supersedes another speaker fact. Chatter
gets the empty answer, and the turns that yield nothing are counted."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.scopes import Visibility
from iifa.store.models import Fact
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, empty_answer, post_turn, run_all, tier2_settings

CODER = "user:alice/agent:coder"
OWNERS = (("user:alice", "Alice", ("Алиса",)),)


def _app(tmp_path: Path, fake: FakeOpenAIServer, clock: FakeClock, **extra: Any) -> FastAPI:
    return create_app(tier2_settings(tmp_path, fake, **extra), clock=clock)


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = _app(tmp_path, fake_openai, clock)
    yield application
    application.state.store.close()


@pytest.fixture
def owned(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = _app(tmp_path, fake_openai, clock, owners=OWNERS)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


@pytest.fixture
def owned_client(owned: FastAPI) -> Iterator[TestClient]:
    with TestClient(owned, headers=auth("all")) as test_client:
        yield test_client


def speaker_says(subject: str, statement: str, value: str, quote: str, **extra: Any) -> Reply:
    fact = {
        "subject": subject,
        "statement": statement,
        "attribute": "home",
        "value": value,
        "quote": quote,
    }
    return answer(nodes=[{"name": subject, "kind": "person"}], facts=[fact], **extra)


def ingest(client: TestClient, app: FastAPI, text: str, *, scope: str = SCOPE) -> dict[str, Any]:
    post_turn(client, text, scope=scope)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def facts(app: FastAPI) -> list[Fact]:
    store = app.state.store
    rows = store.execute("SELECT id FROM facts ORDER BY created_at, id").fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def node_names(app: FastAPI) -> list[str]:
    return [r["name"] for r in app.state.store.execute("SELECT name FROM nodes ORDER BY name")]


# --------------------------------------------------------------------------- no owner declared


def test_a_speaker_fact_without_owner_is_kept_free_standing_and_never_a_node(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "I live in Lyon, and I know Karim."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "User", "kind": "person"}, {"name": "Karim", "kind": "person"}],
                facts=[
                    {
                        "subject": "User",
                        "statement": "The user lives in Lyon.",
                        "attribute": "home",
                        "value": "Lyon",
                        "quote": "I live in Lyon",
                    }
                ],
                edges=[{"src": "User", "dst": "Karim", "type": "knows", "quote": "I know Karim"}],
            )
        ]
    )
    result = ingest(client, app, text)
    assert result["facts"]["written"] == 1 and result["facts"]["speaker_free"] == 1
    assert result["facts"]["incomplete"] == 0
    assert node_names(app) == ["Karim"]  # "User" is no node
    (fact,) = facts(app)
    assert fact.node_id is None and "speaker" in fact.flags
    assert fact.provenance == "owner"
    assert app.state.store.execute("SELECT count(*) FROM edges").fetchone()[0] == 0


def test_free_speaker_facts_never_supersede_one_another(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Without a declared owner several people may speak in one scope: one
    person's "I live in Paris" must not close another's "I live in Lyon"."""
    fake_openai.script(
        chat=[
            speaker_says("User", "The user lives in Lyon.", "Lyon", "I live in Lyon"),
            speaker_says("the user", "The user lives in Paris.", "Paris", "I live in Paris"),
            speaker_says("User", "The user lives in Lyon now.", "Lyon", "I live in Lyon"),
        ]
    )
    ingest(client, app, "I live in Lyon.")
    second = ingest(client, app, "I live in Paris.")
    assert second["facts"]["superseded"] == 0 and second["facts"]["conflict"] == 1
    third = ingest(client, app, "I live in Lyon.")
    assert third["facts"]["written"] == 0 and third["facts"]["duplicate"] == 1
    lyon, paris = facts(app)
    assert lyon.valid_to is None and paris.valid_to is None
    assert "conflict" in paris.flags and paris.conflicts_with == lyon.id


def test_a_speaker_fact_in_another_scope_is_no_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            speaker_says("User", "The user lives in Lyon.", "Lyon", "I live in Lyon"),
            speaker_says("User", "The user lives in Paris.", "Paris", "I live in Paris"),
        ]
    )
    ingest(client, app, "I live in Lyon.", scope="user:alice")
    other = ingest(client, app, "I live in Paris.", scope="user:bob")
    assert other["facts"]["conflict"] == 0  # P9: user:bob does not see user:alice


# --------------------------------------------------------------------------- a declared owner


def test_the_owner_part_is_labelled_with_the_owner_name_and_the_prompt_says_who_speaks(
    owned_client: TestClient, owned: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[speaker_says("Alice", "Alice lives in Lyon.", "Lyon", "I live in Lyon")]
    )
    result = ingest(owned_client, owned, "I live in Lyon.", scope=CODER)
    messages = fake_openai.chat_calls[0].body["messages"]
    assert "Alice: I live in Lyon." in messages[-1]["content"]
    assert "User:" not in messages[-1]["content"]
    assert "Alice" in messages[0]["content"] and "owner" in messages[0]["content"]
    # an owner rarely declares every script of their name (a real model makes "Алиса" a node)
    assert (
        'use "Alice", in exactly this spelling, as the subject and in the statement, '
        "whatever the language of the source" in messages[0]["content"]
    )
    assert result["facts"]["written"] == 1 and result["facts"]["speaker_owner"] == 1
    store = owned.state.store
    owner = store.find_node("Alice", Visibility.of_scope("user:alice"))
    assert owner is not None and owner.kind == "person"
    assert owner.scopes == ("user:alice",)  # the declared scope, not the turn's
    assert "Алиса" in owner.aliases
    (fact,) = facts(owned)
    assert fact.node_id == owner.id and fact.scopes == (CODER,)
    assert "speaker" not in fact.flags and fact.provenance == "owner"


@pytest.mark.parametrize(
    ("subject", "statement", "value", "text"),
    [
        ("User", "L'utilisateur habite à Lyon.", "Lyon", "J'habite à Lyon."),
        ("Пользователь", "Пользователь живёт в Лионе.", "Лион", "Я живу в Лионе."),
        ("用户", "用户住在里昂。", "里昂", "我住在里昂。"),
    ],
)
def test_a_speaker_word_left_by_the_model_is_attached_to_the_owner(
    owned_client: TestClient,
    owned: FastAPI,
    fake_openai: FakeOpenAIServer,
    subject: str,
    statement: str,
    value: str,
    text: str,
) -> None:
    fake_openai.script(chat=[speaker_says(subject, statement, value, text.rstrip(".。"))])
    result = ingest(owned_client, owned, text)
    assert result["facts"]["written"] == 1, result
    assert result["facts"]["incomplete"] == 0 and result["facts"]["cross_script"] == 0
    assert node_names(owned) == ["Alice"]
    (fact,) = facts(owned)
    assert fact.node_id == owned.state.store.find_node("Alice", Visibility.of_scope(SCOPE)).id


def test_the_owner_changes_supersede_and_an_existing_owner_node_is_reused(
    owned_client: TestClient, owned: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = owned.state.store
    existing = store.create_node(scopes=[SCOPE], name="Alice", kind="person")
    fake_openai.script(
        chat=[
            speaker_says("Alice", "Alice lives in Lyon.", "Lyon", "I live in Lyon"),
            speaker_says("User", "The user moved to Paris.", "Paris", "I moved to Paris"),
        ]
    )
    ingest(owned_client, owned, "I live in Lyon.")
    moved = ingest(owned_client, owned, "I moved to Paris.")
    assert moved["facts"]["superseded"] == 1
    lyon, paris = facts(owned)
    assert lyon.node_id == paris.node_id == existing.id
    assert lyon.superseded_by == paris.id and paris.valid_to is None
    assert node_names(owned) == ["Alice"]


def test_a_turn_split_between_two_owners_has_none(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    owners = (("user:alice", "Alice", ()), ("user:bob", "Bob", ()))
    application = _app(tmp_path, fake_openai, clock, owners=owners)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[speaker_says("User", "The user lives in Lyon.", "Lyon", "I live in Lyon")]
            )
            response = client.post(
                "/v1/turns", json={"scopes": ["user:alice", "user:bob"], "user": "I live in Lyon."}
            )
            assert response.status_code == 202
            run_all(client, application)
            assert (
                "User: I live in Lyon." in fake_openai.chat_calls[0].body["messages"][-1]["content"]
            )
            (fact,) = facts(application)
            assert fact.node_id is None and "speaker" in fact.flags
            assert node_names(application) == []
    finally:
        application.state.store.close()


# --------------------------------------------------------------------------- chatter


def test_the_prompt_expects_nothing_from_chatter_and_empty_turns_are_counted(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[empty_answer()])
    result = ingest(client, app, "Il pleut encore ce matin. Merci, c'était utile.")
    assert result["empty_answers"] == 1 and result["facts"]["written"] == 0
    system = fake_openai.chat_calls[0].body["messages"][0]["content"]
    for words in ("weather", "thanks", "the conversation itself", "the assistant"):
        assert words in system
    assert '{"nodes": [], "facts": [], "edges": []}' in system
