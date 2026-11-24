"""The agent perspective: with a self declared, the assistant
part is the self speaking; its lessons, views and commitments are written on
its node, a rule of method on the task it serves, and the user perspective
stays exactly as it is."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest.prompts import extraction_messages
from iifa.scopes import Visibility
from iifa.store.models import Fact
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

SELF = "NOVA"


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    settings = tier2_settings(
        tmp_path,
        fake_openai,
        owners=((SCOPE, "Alice", ()),),
        selves=((SCOPE, SELF, ("Nova",)),),
    )
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def fact(
    subject: str, statement: str, key: str, value: str, quote: str, **extra: Any
) -> dict[str, Any]:
    return {
        "subject": subject,
        "statement": statement,
        "attribute": key,
        "value": value,
        "quote": quote,
        **extra,
    }


def ingest(
    client: TestClient, app: FastAPI, user: str, assistant: str = "", **extra: Any
) -> dict[str, Any]:
    post_turn(client, user, assistant, **extra)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def live_facts(app: FastAPI) -> list[Fact]:
    store: Store = app.state.store
    rows = store.execute(
        "SELECT id FROM facts WHERE valid_to IS NULL ORDER BY created_at, id"
    ).fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def test_a_lesson_of_the_agent_is_written_on_its_node_in_the_declared_scope(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = tier2_settings(tmp_path, fake_openai, selves=((SCOPE, SELF, ("Nova",)),))
    application = create_app(settings, clock=clock)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    answer(
                        nodes=[{"name": SELF, "kind": "other"}],
                        facts=[
                            fact(
                                SELF,
                                "NOVA found that the dough rises best after a night in the fridge.",
                                "finding.dough",
                                "after a night in the fridge",
                                "the dough rises best after a night in the fridge",
                            )
                        ],
                    )
                ]
            )
            result = ingest(
                client,
                application,
                "So what was it?",
                "I found it: the dough rises best after a night in the fridge.",
                scope="user:alice/agent:coder",
            )
            assert result["facts"]["written"] == 1 and result["facts"]["self"] == 1
            node = application.state.store.find_node(SELF, Visibility.of_scope(SCOPE))
            assert node is not None and node.scopes == (SCOPE,) and node.aliases == ("Nova",)
            (lesson,) = live_facts(application)
            assert lesson.node_id == node.id
            assert (lesson.provenance, lesson.part_ref) == ("assistant", SELF)
            messages = fake_openai.chat_calls[0].body["messages"]
            assert "\n\nNOVA: I found it" in messages[-1]["content"]
            system = messages[0]["content"]
            assert 'labelled "NOVA:"' in system and "first person" in system
            assert "rule of method" in system
            assert "what the assistant says about itself" not in system
    finally:
        application.state.store.close()


def test_rules_of_method_go_to_the_task_they_serve_as_elements_of_a_set(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    task = "proofing bread dough"
    nodes = [{"name": task, "kind": "task"}]
    fake_openai.script(
        chat=[
            answer(
                nodes=nodes,
                facts=[
                    # the model said one value: the service reads ``method`` as a set
                    fact(
                        task,
                        "Proofing bread dough: leave it a night in the fridge.",
                        "method",
                        "leave it a night in the fridge",
                        "leave it a night in the fridge",
                        cardinality="one",
                    )
                ],
            ),
            answer(
                nodes=nodes,
                facts=[
                    fact(
                        task,
                        "Proofing bread dough: shape the loaf before the second rise.",
                        "method",
                        "shape the loaf before the second rise",
                        "shape the loaf before the second rise",
                        cardinality="one",
                    )
                ],
            ),
        ]
    )
    first = ingest(client, app, "What is worth remembering?", "Leave it a night in the fridge.")
    # a later turn names the task it adds a rule to
    second = ingest(
        client,
        app,
        "Anything else?",
        "For proofing bread dough: shape the loaf before the second rise.",
    )
    assert first["facts"]["method"] == 1 and second["facts"]["method"] == 1
    rules = live_facts(app)
    assert len(rules) == 2 and all("many" in r.flags for r in rules)
    node = app.state.store.find_node(task, Visibility.of_scope(SCOPE))
    assert node is not None and node.kind == "task"
    assert {r.node_id for r in rules} == {node.id}


def test_an_edge_to_the_self_is_confirmed_only_by_the_owners_words(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    nodes = [{"name": SELF, "kind": "other"}, {"name": "Solar", "kind": "project"}]

    def says(quote: str) -> Any:
        return answer(
            nodes=nodes,
            edges=[{"src": SELF, "dst": "Solar", "type": "works_on", "quote": quote}],
        )

    fake_openai.script(
        chat=[
            says("I work on Solar"),
            says("I work on Solar"),
            says("I work on Solar"),
            says("you work on Solar"),
        ]
    )
    for _ in range(3):
        ingest(client, app, "Hello.", "I work on Solar every day.")
    store: Store = app.state.store
    view = Visibility.of_scope(SCOPE)
    me = store.find_node(SELF, view)
    solar = store.find_node("Solar", view)
    assert me is not None and solar is not None
    edges = store.edges_between([me.id, solar.id], view)
    assert [(e.type, e.status) for e in edges] == [("works_on", "proposed")]
    ingest(client, app, "Yes, you work on Solar.", "Indeed.")
    edges = store.edges_between([me.id, solar.id], view)
    # the self takes no co-occurrence: its label stands before every answer
    assert [(e.type, e.status) for e in edges] == [("works_on", "confirmed")]


def test_the_user_perspective_is_unchanged_without_a_self(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """Only an assistant name: what the assistant says of itself next to a page
    is written as before (no self-side rule applies), and the prompt still
    lists it as chatter."""
    settings = tier2_settings(tmp_path, fake_openai, assistants=((SCOPE, SELF),))
    application = create_app(settings, clock=clock)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    answer(
                        nodes=[{"name": "exporting a report", "kind": "topic"}],
                        facts=[
                            fact(
                                "exporting a report",
                                "Exporting a report: add the token to the URL.",
                                "method",
                                "add the token to the URL",
                                "add the token to the URL",
                            )
                        ],
                    )
                ]
            )
            result = ingest(
                client,
                application,
                "How?",
                "To export, add the token to the URL.",
                external=[{"kind": "web", "text": "Add the token to the URL."}],
            )
            assert result["facts"]["written"] == 1 and result["facts"]["method"] == 0
            (written,) = live_facts(application)
            assert "many" not in written.flags
            system = fake_openai.chat_calls[0].body["messages"][0]["content"]
            assert "what the assistant says about itself" in system
            assert "rule of method" not in system
    finally:
        application.state.store.close()


def test_the_user_perspective_prompt_is_byte_for_byte_the_same() -> None:
    """``agent=None`` is the prompt every existing corpus was measured with."""
    kwargs: dict[str, Any] = {"languages": ["en"], "owner": "Alice", "assistant": "NOVA"}
    plain = extraction_messages("User: hi", **kwargs)
    assert extraction_messages("User: hi", agent=None, **kwargs) == plain
    agent = extraction_messages("User: hi", agent="NOVA", **kwargs)
    assert agent != plain
    assert "rule of method" in agent[0]["content"] and "rule of method" not in plain[0]["content"]
    # what a rule is, where it goes, and what a limit is instead
    for words in ("a format or a convention", "never the guide it was read in", "A limit"):
        assert words in agent[0]["content"]
    assert "|task|" in agent[0]["content"] and "|task|" not in plain[0]["content"]


def test_rules_read_in_a_guide_the_owner_vouched_for_are_learnt_and_flagged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The agent learns from what it reads for the owner, at once."""
    task = "writing a recipe card"
    guide = "Recipe cards: list the ingredients in order of use. Give oven temperatures in Celsius."
    fake_openai.script(
        chat=[
            answer(
                nodes=[{"name": task, "kind": "task"}],
                facts=[
                    # restated in the reply
                    fact(
                        task,
                        "Writing a recipe card: list the ingredients in order of use.",
                        "method",
                        "list the ingredients in order of use",
                        "I list the ingredients in order of use",
                    ),
                    # quoted from the guide itself
                    fact(
                        task,
                        "Writing a recipe card: give oven temperatures in Celsius.",
                        "method",
                        "give oven temperatures in Celsius",
                        "Give oven temperatures in Celsius",
                    ),
                ],
            )
        ]
    )
    result = ingest(
        client,
        app,
        "Write the apple pie card, with the guide I attached.",
        "Done. As the guide says, I list the ingredients in order of use.",
        external=[{"kind": "document", "text": guide, "ref": "cards.md", "vouched": True}],
    )
    assert result["facts"]["written"] == 2 and result["facts"]["method"] == 2
    rules = live_facts(app)
    assert all("vouched" in r.flags and "many" in r.flags for r in rules)
    assert {r.provenance for r in rules} == {"assistant", "external"}
    source = app.state.store.get_source(rules[0].source_id)
    assert source.external == "vouched"
