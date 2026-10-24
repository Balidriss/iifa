"""Near names (SPEC 5.1): a name alike letter by letter is a candidate for the
adjudication, never a match on its own. Ivana's facts never land on Ivan's
node, whichever of the two is named first."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.scopes import Visibility
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.tier2 import SCOPE, answer, post_turn, run_all

VIEW = Visibility.of_scope(SCOPE)
SAID = {"Ivan": "Ivan habite à Lyon.", "Ivana": "Ivana habite à Paris."}


def told(name: str) -> Any:
    statement = SAID[name]
    return answer(
        nodes=[{"name": name, "kind": "person"}],
        facts=[{"subject": name, "statement": statement, "quote": statement.rstrip(".")}],
    )


@pytest.mark.parametrize("first", ["Ivan", "Ivana"])
def test_two_near_names_keep_their_own_facts_in_either_order(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, first: str
) -> None:
    second = "Ivana" if first == "Ivan" else "Ivan"
    fake_openai.script(chat=[told(first)])
    post_turn(client, SAID[first])
    run_all(client, app)
    new = chat_json({"decisions": [{"mention": 0, "choice": "new"}]})
    fake_openai.script(chat=[told(second), new, new])
    post_turn(client, SAID[second])
    report = run_all(client, app)
    turn = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    # the near name reached the adjudication instead of attaching
    assert turn["mentions_adjudicated"] >= 1
    asked = turn["adjudications"][0]
    assert (asked["mention"], asked["chosen"]) == (second, None)
    assert [c["name"] for c in asked["candidates"]] == [first]
    store = app.state.store
    for name in (first, second):
        node = store.find_node(name, VIEW)
        assert node is not None and node.name == name
        assert [f.statement for f in store.facts([node.id], VIEW)] == [SAID[name]]


def test_the_adjudication_may_still_attach_a_near_name(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A typo the model recognises attaches: the model proposes, the service
    checks the choice is one of the candidates."""
    statement = "Youssef habite à Tunis."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Youssef", "kind": "person"}],
                facts=[{"subject": "Youssef", "statement": statement, "quote": "Youssef"}],
            )
        ]
    )
    post_turn(client, statement)
    run_all(client, app)
    store = app.state.store
    youssef = store.find_node("Youssef", VIEW)
    assert youssef is not None
    typo = "Yousef aime le thé."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": "Yousef", "kind": "person"}],
                facts=[{"subject": "Yousef", "statement": typo, "quote": "Yousef"}],
            ),
            chat_json({"decisions": [{"mention": 0, "choice": youssef.id}]}),
        ]
    )
    post_turn(client, typo)
    run_all(client, app)
    assert {f.statement for f in store.facts([youssef.id], VIEW)} == {statement, typo}
    # attached for this source only: a near form is proposed as an alias, never
    # kept, so the next "Yousef" goes to the adjudication again (SPEC 5.1)
    assert store.find_node("Yousef", VIEW) is None
    proposals = store.proposals([youssef.id], VIEW)
    assert [(p.kind, p.value) for p in proposals] == [("alias", "Yousef")]


def test_an_adjudicated_choice_holds_after_a_new_node_is_proposed(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A mention the adjudication attached keeps its node even when an unknown
    mention before it proposes a new node first (the proposal clears the
    resolution cache)."""
    store = app.state.store
    fake_openai.script(chat=[told("Ivan")])
    post_turn(client, SAID["Ivan"])
    run_all(client, app)
    ivan = store.find_node("Ivan", VIEW)
    assert ivan is not None
    said = "Bob boit du thé. Ivann boit du café."
    fake_openai.script(
        chat=[
            answer(
                facts=[
                    {"subject": "Bob", "statement": "Bob boit du thé.", "quote": "Bob boit du thé"},
                    {
                        "subject": "Ivann",
                        "statement": "Ivann boit du café.",
                        "quote": "Ivann boit du café",
                    },
                ]
            ),
            chat_json({"decisions": [{"mention": 0, "choice": ivan.id}]}),
        ]
    )
    post_turn(client, said)
    report = run_all(client, app)
    turn = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert turn["adjudications"][0]["chosen"] == "Ivan"
    assert "Ivann boit du café." in [f.statement for f in store.facts([ivan.id], VIEW)]
    assert store.find_node("Ivann", VIEW) is None or store.find_node("Ivann", VIEW).id == ivan.id
