"""A memory rooted at the agent (SPEC 4.6): no rule reads the kind of a scope
segment, so ``agent:nova`` with the owner as a node inside it behaves like
``user:alice`` with the agent as a node inside it - the owner's facts and the
self's, the profile, recall, the scopes below and the tokens' bounds."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import TokenConfig
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply
from tests.helpers import TOKENS, auth
from tests.tier2 import answer, post_turn, run_all, tier2_settings

ROOT = "agent:nova"
BELOW = f"{ROOT}/project:solar"
SELF, OWNER = "NOVA", "Alice"
NOVA_TOKEN = "tok-nova"


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    tokens = [
        TokenConfig.from_plaintext(token, name=name, scopes=scopes, permissions=perms)
        for name, (token, scopes, perms) in TOKENS.items()
    ]
    tokens.append(
        TokenConfig.from_plaintext(
            NOVA_TOKEN, name="nova", scopes=(ROOT,), permissions=frozenset({"read", "write"})
        )
    )
    settings = tier2_settings(
        tmp_path,
        fake_openai,
        tokens=tuple(tokens),
        owners=((ROOT, OWNER, ()),),
        selves=((ROOT, SELF, ()),),
    )
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def said(subject: str, statement: str, key: str, value: str, quote: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "statement": statement,
        "attribute": key,
        "value": value,
        "quote": quote,
    }


def both() -> Reply:
    return answer(
        facts=[
            said(OWNER, "Alice lives in Lyon.", "location", "Lyon", "I live in Lyon"),
            said(
                SELF,
                "NOVA confirms the date before a booking.",
                "commitment.bookings",
                "date check",
                "I confirm the date before booking",
            ),
        ]
    )


def turn(client: TestClient, app: FastAPI, user: str, assistant: str, scope: str) -> dict[str, Any]:
    post_turn(client, user, assistant, scope=scope)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return dict(result["facts"])


def test_a_memory_rooted_at_the_agent_holds_the_owner_as_a_node_inside_it(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[both()])
    nova = {"Authorization": f"Bearer {NOVA_TOKEN}"}
    post = client.post(
        "/v1/turns",
        json={
            "scope": ROOT,
            "user": "I live in Lyon.",
            "assistant": "Noted. From now on I confirm the date before booking.",
        },
        headers=nova,
    )
    assert post.status_code == 202, post.text
    jobs = run_all(client, app)["jobs"]
    facts = next(j["result"] for j in jobs if j["kind"] == "turn")["facts"]
    assert (facts["written"], facts["speaker_owner"], facts["self"]) == (2, 1, 1)
    store = app.state.store
    view = Visibility.of_scope(ROOT)
    owner, me = store.find_node(OWNER, view), store.find_node(SELF, view)
    assert owner is not None and owner.kind == "person" and owner.scopes == (ROOT,)
    assert me is not None and me.scopes == (ROOT,)
    assert {f.scopes for f in store.facts([owner.id, me.id], view)} == {(ROOT,)}
    # the profile starts a session with the self's rule and both nodes
    profile = client.post(
        "/v1/session/start", json={"scope": ROOT, "budget_tokens": 2000}, headers=nova
    ).json()
    (rule,) = [i for i in profile["items"] if i["kind"] == "fact"]
    assert rule["attribute"] == "commitment.bookings" and rule["from"] == SELF
    assert {i["name"] for i in profile["items"] if i["kind"] == "node"} == {OWNER, SELF}
    # recall brings the owner's fact back, the owner's own words (no `from`)
    recalled = client.post(
        "/v1/recall", json={"scope": ROOT, "query": "Where does Alice live?"}, headers=nova
    ).json()
    lyon = next(i for i in recalled["items"] if "Lyon" in i.get("line", ""))
    assert "from" not in lyon


def test_the_scopes_below_an_agent_root_inherit_its_owner_and_its_self(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[both()])
    facts = turn(
        client,
        app,
        "I live in Lyon.",
        "Noted. From now on I confirm the date before booking.",
        BELOW,
    )
    assert (facts["written"], facts["speaker_owner"], facts["self"]) == (2, 1, 1)
    store = app.state.store
    # the owner's node and the self's are found or created on the declared root
    ids = []
    for name in (OWNER, SELF):
        node = store.find_node(name, Visibility.of_scope(ROOT))
        assert node is not None and node.scopes == (ROOT,)
        ids.append(node.id)
    # the facts belong to the scope of their source
    held = store.facts(ids, Visibility.of_scope(BELOW))
    assert len(held) == 2 and {f.scopes for f in held} == {(BELOW,)}
    assert store.facts(ids, Visibility.of_scope(ROOT)) == []
    # a session of the root alone does not see what the scope below learnt (P9)
    root_only = client.post("/v1/recall", json={"scope": ROOT, "query": "Lyon"}).json()
    assert not [i for i in root_only["items"] if "Lyon" in i.get("line", "")]


def test_a_token_bound_to_another_root_reaches_nothing_of_an_agent_root(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[both()])
    turn(client, app, "I live in Lyon.", "Noted. I confirm the date before booking.", ROOT)
    for path, body in (
        ("/v1/recall", {"scope": ROOT, "query": "Lyon"}),
        ("/v1/session/start", {"scope": ROOT}),
        ("/v1/turns", {"scope": ROOT, "user": "I live in Paris."}),
    ):
        assert client.post(path, json=body, headers=auth("alice")).status_code == 403, path
    # and the agent's token reaches nothing of a user root
    nova = {"Authorization": f"Bearer {NOVA_TOKEN}"}
    refused = client.post("/v1/recall", json={"scope": "user:alice", "query": "Lyon"}, headers=nova)
    assert refused.status_code == 403
