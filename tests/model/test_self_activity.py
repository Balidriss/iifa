"""The activity, in the agent perspective: a turn names what
the owner asks the agent to do, a session keeps it as its current activity,
and the rules of method learnt meanwhile are linked to it beside their
subject - never moved onto it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest import sessions as session_state
from iifa.scopes import Visibility
from iifa.store.models import Fact, Node
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, end_session, post_turn, run_all, tier2_settings

SELF = "NOVA"
VIEW = Visibility.of_scope(SCOPE)

ASK = "Plan the spring vegetable garden, following the guide I attached."
ACTIVITY = "planning the vegetable garden"
GUIDE_NAME = "Verdant planting guide"
GUIDE = (
    "Verdant planting guide.\n"
    "Always water seedlings in the morning.\n"
    "Herbs always go in partial shade."
)
WATER = "Always water seedlings in the morning"
SHADE = "Herbs always go in partial shade"


def make(tmp_path: Path, fake: FakeOpenAIServer, clock: FakeClock, **overrides: Any) -> FastAPI:
    values: dict[str, Any] = {
        "extraction_intent": True,
        "owners": ((SCOPE, "Alice", ()),),
        "selves": ((SCOPE, SELF, ()),),
    }
    values.update(overrides)
    return create_app(tier2_settings(tmp_path, fake, **values), clock=clock)


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = make(tmp_path, fake_openai, clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def rule(subject: str, value: str, quote: str) -> dict[str, Any]:
    return {
        "subject": subject,
        "statement": f"For {subject}: {value}.",
        "attribute": "method",
        "value": value,
        "quote": quote,
        "modality": "fact",
        "duration": "lasting",
    }


def reply(
    *facts: dict[str, Any],
    activity: dict[str, str] | None = None,
    nodes: list[dict[str, Any]] | None = None,
    frame: str = "sincere",
    edges: list[dict[str, Any]] | None = None,
) -> Reply:
    body: dict[str, Any] = {
        "nodes": nodes or [],
        "facts": list(facts),
        "edges": edges or [],
        "frame": frame,
        "activity": activity,
    }
    return chat_json(body)


def asked(name: str = ACTIVITY, quote: str = "Plan the spring vegetable garden") -> dict[str, str]:
    return {"name": name, "quote": quote}


def guide(vouched: bool = True) -> list[dict[str, Any]]:
    return [{"kind": "document", "text": GUIDE, "ref": "verdant.md", "vouched": vouched}]


def ingest(client: TestClient, app: FastAPI, user: str, **extra: Any) -> dict[str, Any]:
    post_turn(client, user, extra.pop("assistant", "Here is the plan."), **extra)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def tasks(app: FastAPI) -> list[Node]:
    store: Store = app.state.store
    rows = store.execute("SELECT id FROM nodes WHERE kind = 'task' ORDER BY created_at, id")
    return [store.get_node(r["id"]) for r in rows.fetchall()]


def rules(app: FastAPI) -> list[Fact]:
    store: Store = app.state.store
    rows = store.execute(
        "SELECT id FROM facts WHERE attribute = 'method' AND valid_to IS NULL"
        " ORDER BY created_at, id"
    ).fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def node_named(app: FastAPI, name: str) -> Node:
    node = app.state.store.find_node(name, VIEW)
    assert node is not None
    return node


def links(app: FastAPI, fact: Fact) -> list[str]:
    """The activities ``fact`` is linked to."""
    return app.state.store.fact_activities([fact.id]).get(fact.id, [])


def current_activity(app: FastAPI, session: str = "s") -> str | None:
    return session_state.load(app.state.store, session, [SCOPE]).activity_id


# --------------------------------------------------------------------------- one turn


def test_a_turn_names_its_activity_and_its_rule_is_linked_beside_its_subject(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    # the model hangs the rule on the guide, as a real model did: the subject
    # stays the guide, the activity is a link beside it
    fake_openai.script(
        chat=[
            reply(
                rule(GUIDE_NAME, "water seedlings in the morning", WATER),
                activity=asked(),
                nodes=[{"name": GUIDE_NAME, "kind": "object"}],
            )
        ]
    )
    result = ingest(client, app, ASK, session_id="s", external=guide())
    assert result["activity"] == {"named": 1, "carried": 0, "refused": 0, "linked": 1}
    (task,) = tasks(app)
    assert (task.name, task.scopes) == (ACTIVITY, (SCOPE,))
    (water,) = rules(app)
    assert water.node_id == node_named(app, GUIDE_NAME).id
    assert links(app, water) == [task.id]
    assert current_activity(app) == task.id
    system = fake_openai.chat_calls[0].body["messages"][0]["content"]
    assert 'Add "activity" to the same JSON object' in system
    assert "Current activity of this session" not in system  # none yet


def test_the_session_carries_its_activity_to_a_rule_read_in_a_later_turn(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            reply(activity=asked()),
            # turn 2 never writes the activity's name: it is still no "known
            # subject the source does not name"
            reply(rule(ACTIVITY, "herbs go in partial shade", SHADE)),
        ]
    )
    ingest(client, app, ASK, session_id="s")
    result = ingest(client, app, "Now the herbs, please.", session_id="s", external=guide())
    assert result["activity"] == {"named": 0, "carried": 1, "refused": 0, "linked": 1}
    assert result["facts"]["written"] == 1 and result["facts"]["unnamed"] == 0
    assert result["adjudication_calls"] == 0
    (task,) = tasks(app)
    (shade,) = rules(app)
    assert shade.node_id == task.id and links(app, shade) == [task.id]
    system = fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert (
        f'Current activity of this session, asked by the person in an earlier turn: "{ACTIVITY}"'
        in system
    )


def test_a_later_activity_replaces_the_current_one(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            reply(activity=asked()),
            reply(activity=asked("baking sourdough bread", "Bake a sourdough bread")),
            reply(rule(GUIDE_NAME, "water seedlings in the morning", WATER)),
        ]
    )
    ingest(client, app, ASK, session_id="s")
    ingest(client, app, "Bake a sourdough bread for Sunday.", session_id="s")
    third = ingest(client, app, "And this?", session_id="s", external=guide())
    assert third["activity"] == {"named": 0, "carried": 1, "refused": 0, "linked": 1}
    _garden, bread = tasks(app)
    assert bread.name == "baking sourdough bread"
    assert current_activity(app) == bread.id
    (water,) = rules(app)
    assert links(app, water) == [bread.id]


def test_a_turn_without_a_session_keeps_its_activity_for_itself(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            reply(activity=asked()),
            reply(rule(GUIDE_NAME, "water seedlings in the morning", WATER)),
        ]
    )
    ingest(client, app, ASK)
    result = ingest(client, app, "And this?", external=guide())
    assert result["activity"] == {"named": 0, "carried": 0, "refused": 0, "linked": 0}
    assert len(tasks(app)) == 1
    (water,) = rules(app)
    assert links(app, water) == []
    assert app.state.store.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


# --------------------------------------------------------------------------- resolution


def test_an_activity_takes_the_task_node_of_its_exact_name_never_a_near_one(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store: Store = app.state.store
    known = store.create_node(scopes=[SCOPE], name="Planning the Vegetable-Garden", kind="task")
    fake_openai.script(
        chat=[
            reply(activity=asked()),
            reply(activity=asked("planning the garden", "Plan the garden")),
        ]
    )
    ingest(client, app, ASK, session_id="s")
    assert [t.id for t in tasks(app)] == [known.id]  # normalised, the same name
    assert current_activity(app) == known.id
    ingest(client, app, "Plan the garden again.", session_id="s")
    first, contained = tasks(app)
    assert first.id == known.id and contained.name == "planning the garden"
    assert current_activity(app) == contained.id


def test_an_activity_named_like_a_node_of_another_kind_is_refused(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    app.state.store.create_node(scopes=[SCOPE], name="vegetable garden", kind="place")
    fake_openai.script(
        chat=[
            reply(
                rule(GUIDE_NAME, "water seedlings in the morning", WATER),
                activity=asked("vegetable garden"),
            )
        ]
    )
    result = ingest(client, app, ASK, session_id="s", external=guide())
    assert result["activity"] == {"named": 0, "carried": 0, "refused": 1, "linked": 0}
    assert tasks(app) == []
    (water,) = rules(app)
    assert links(app, water) == []
    assert current_activity(app) is None


@pytest.mark.parametrize(
    ("name", "quote"),
    [
        ("gardening", "Plan the spring vegetable garden"),  # no word of it in the quote
        ("to do it with her", "do it with her"),  # only short and function words
    ],
)
def test_an_activity_the_owners_words_do_not_hold_is_refused(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, name: str, quote: str
) -> None:
    fake_openai.script(chat=[reply(activity=asked(name, quote))])
    result = ingest(client, app, ASK + " Then do it with her.", session_id="s")
    assert result["activity"]["refused"] == 1
    assert tasks(app) == []


def test_an_activity_of_a_frame_that_is_not_sincere_is_refused(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(chat=[reply(activity=asked(), frame="pretend")])
    result = ingest(client, app, ASK, session_id="s")
    assert result["activity"]["refused"] == 1
    assert tasks(app) == [] and current_activity(app) is None


def test_the_user_perspective_refuses_an_activity_as_an_unknown_key(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = make(tmp_path, fake_openai, clock, selves=())
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(chat=[reply(activity=asked())])
            result = ingest(client, application, ASK, session_id="s")
            assert result["rejected"]["keys"] == 1
            assert result["activity"]["named"] == 0
            assert tasks(application) == []
            system = fake_openai.chat_calls[0].body["messages"][0]["content"]
            assert '"activity"' not in system
    finally:
        application.state.store.close()


# --------------------------------------------------------------------------- what follows the link


def test_a_proposed_rule_keeps_its_activity_and_its_acceptance_writes_it(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[reply(rule(GUIDE_NAME, "water seedlings in the morning", WATER), activity=asked())]
    )
    result = ingest(client, app, ASK, session_id="s", external=guide(vouched=False))
    assert result["facts"]["self_proposed"] == 1 and result["activity"]["linked"] == 1
    (task,) = tasks(app)
    store: Store = app.state.store
    (proposal_id,) = [
        r["id"] for r in store.execute("SELECT id FROM proposals WHERE kind = 'fact'").fetchall()
    ]
    assert store.get_proposal(proposal_id).payload["activities"] == [task.id]
    decided = client.post(f"/v1/proposals/{proposal_id}", json={"decision": "accept"})
    assert decided.status_code == 200, decided.text
    (water,) = rules(app)
    assert links(app, water) == [task.id] and water.proposal_id == proposal_id


def test_p5_forgetting_the_activity_clears_its_links_and_the_session(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            reply(
                rule(GUIDE_NAME, "water seedlings in the morning", WATER),
                activity=asked(),
                nodes=[{"name": GUIDE_NAME, "kind": "object"}],
            )
        ]
    )
    ingest(client, app, ASK, session_id="s", external=guide())
    (task,) = tasks(app)
    assert client.post("/v1/forget", json={"id": task.id}).status_code == 200
    (water,) = rules(app)  # the rule is the guide's: it stays, unlinked
    assert links(app, water) == []
    assert current_activity(app) is None


def test_a_merged_activity_is_followed_to_the_node_it_was_merged_into(
    store: Store,
) -> None:
    old = store.create_node(scopes=[SCOPE], name="planning a garden", kind="task")
    kept = store.create_node(scopes=[SCOPE], name="planning the garden", kind="task")
    state = session_state.load(store, "s", [SCOPE])
    session_state.save(
        store, state, frame=None, frame_episode=False, ended=False, facts=[], activity=old.id
    )
    store.merge_node(old.id, into=kept.id)
    assert session_state.load(store, "s", [SCOPE]).activity_id == kept.id


def test_an_archived_activity_is_no_longer_carried(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            reply(activity=asked()),
            reply(rule(GUIDE_NAME, "water seedlings in the morning", WATER)),
        ]
    )
    ingest(client, app, ASK, session_id="s")
    (task,) = tasks(app)
    assert client.post("/v1/archive", json={"id": task.id}).status_code == 200
    result = ingest(client, app, "And this?", session_id="s", external=guide())
    assert result["activity"]["carried"] == 0
    (water,) = rules(app)
    assert links(app, water) == []
    system = fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert "Current activity of this session" not in system


# --------------------------------------------------------------------------- the links


def test_an_edge_uses_from_the_activity_joins_it_to_its_tool_even_in_a_later_turn(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    uses = {"src": ACTIVITY, "dst": GUIDE_NAME, "type": "uses", "quote": "Verdant planting guide"}
    fake_openai.script(
        chat=[
            reply(activity=asked()),
            reply(nodes=[{"name": GUIDE_NAME, "kind": "object"}], edges=[uses]),
        ]
    )
    ingest(client, app, ASK, session_id="s")
    result = ingest(client, app, "Use this one.", session_id="s", external=guide())
    assert result["adjudication_calls"] == 0
    (task,) = tasks(app)
    store: Store = app.state.store
    edges = store.execute("SELECT src, dst, type FROM edges WHERE type = 'uses'").fetchall()
    assert [tuple(e) for e in edges] == [(task.id, node_named(app, GUIDE_NAME).id, "uses")]
    system = fake_openai.chat_calls[1].body["messages"][0]["content"]
    assert 'of type "uses"' in system


EPISODE = f"NOVA worked on {ACTIVITY} with Alice."


def test_the_close_links_its_episode_to_the_current_activity(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The episode names the activity, which no source of the session wrote
    whole: it is still written, and linked to it."""
    note = {
        "subject": "spring vegetable garden",
        "statement": "The spring vegetable garden needs a planting plan.",
        "quote": "Plan the spring vegetable garden",
        "modality": "fact",
        "duration": "task",
    }
    fake_openai.script(
        chat=[reply(note, activity=asked()), chat_json({"episode": EPISODE, "keep": []})]
    )
    ingest(client, app, ASK, session_id="s")
    end_session(client, "s")
    jobs = run_all(client, app)["jobs"]
    closed = next(j["result"] for j in jobs if j["kind"] == "session_end")["session"]
    assert closed["written"] == 1 and closed["proposed"] == 0
    (task,) = tasks(app)
    store: Store = app.state.store
    (episode_id,) = [
        r["id"]
        for r in store.execute("SELECT id FROM facts WHERE flags LIKE '%episode%'").fetchall()
    ]
    episode = store.get_fact(episode_id)
    assert episode.statement == EPISODE and links(app, episode) == [task.id]
    assert episode.node_id == node_named(app, SELF).id  # on the self, beside the activity


def test_expand_shows_the_activity_through_a_merge_and_never_an_invisible_one(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[reply(rule(GUIDE_NAME, "water seedlings in the morning", WATER), activity=asked())]
    )
    ingest(client, app, ASK, session_id="s", external=guide())
    (task,) = tasks(app)
    (water,) = rules(app)

    def shown() -> Any:
        body = {"scope": SCOPE, "ids": [water.id]}
        (item,) = client.post("/v1/expand", json=body).json()["items"]
        return item.get("activities")

    assert shown() == [{"id": task.id, "name": ACTIVITY}]
    store: Store = app.state.store
    kept = store.create_node(scopes=[SCOPE], name="vegetable garden planning", kind="task")
    store.merge_node(task.id, into=kept.id)
    assert shown() == [{"id": kept.id, "name": kept.name}]
    hidden = store.create_node(scopes=["user:bob"], name="a task bob holds", kind="task")
    store.execute(
        "UPDATE fact_activities SET activity_id = ? WHERE fact_id = ?", (hidden.id, water.id)
    )
    assert shown() is None  # P9: never named to a reader who cannot see it


# --------------------------------------------------------------------------- several activities


def test_a_session_of_three_activities_links_each_rule_to_its_own_and_the_episode_to_all(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A garden plan, then a map of the plan, then labels for the map: three
    activities, each serving the one before. Each rule goes to the activity
    of its turn; the close's one episode is linked to the three."""
    note = {
        "subject": "spring vegetable garden",
        "statement": "The spring vegetable garden needs a planting plan.",
        "quote": "Plan the spring vegetable garden",
        "modality": "fact",
        "duration": "task",
    }
    fake_openai.script(
        chat=[
            reply(note, rule(ACTIVITY, "herbs go in partial shade", SHADE), activity=asked()),
            reply(
                rule("drawing a map", "water seedlings in the morning", WATER),
                activity=asked("drawing a map", "Draw a map of the plan"),
            ),
            reply(activity=asked("writing labels", "write labels for the map")),
            chat_json({"episode": "NOVA planned the garden with Alice.", "keep": []}),
        ]
    )
    ingest(client, app, ASK, session_id="s", external=guide())
    ingest(client, app, "Draw a map of the plan.", session_id="s", external=guide())
    ingest(client, app, "Now write labels for the map.", session_id="s")
    plan, drawing, labels = tasks(app)
    assert current_activity(app) == labels.id
    shade, water = rules(app)
    assert links(app, shade) == [plan.id] and links(app, water) == [drawing.id]
    system = fake_openai.chat_calls[2].body["messages"][0]["content"]
    assert "A new thing to make is a new activity" in system
    assert "never knitting a scarf for Sam" in system
    end_session(client, "s")
    run_all(client, app)
    store: Store = app.state.store
    (episode_id,) = [
        r["id"]
        for r in store.execute("SELECT id FROM facts WHERE flags LIKE '%episode%'").fetchall()
    ]
    assert set(links(app, store.get_fact(episode_id))) == {plan.id, drawing.id, labels.id}


def test_an_activity_named_again_is_current_again(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            reply(activity=asked()),
            reply(activity=asked("drawing a map", "Draw a map of the plan")),
            reply(activity=asked()),
        ]
    )
    ingest(client, app, ASK, session_id="s")
    ingest(client, app, "Draw a map of the plan.", session_id="s")
    ingest(client, app, ASK, session_id="s")
    plan, drawing = tasks(app)
    state = session_state.load(app.state.store, "s", [SCOPE])
    assert state.activities == (drawing.id, plan.id) and state.activity_id == plan.id


def test_p5_forgetting_a_linked_fact_deletes_its_links(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[reply(rule(GUIDE_NAME, "water seedlings in the morning", WATER), activity=asked())]
    )
    ingest(client, app, ASK, session_id="s", external=guide())
    (water,) = rules(app)
    assert client.post("/v1/forget", json={"id": water.id}).status_code == 200
    store: Store = app.state.store
    assert store.execute("SELECT count(*) FROM fact_activities").fetchone()[0] == 0


# --------------------------------------------------------------------------- what a close does not know


def test_a_rule_kept_at_the_close_is_linked_to_no_activity(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Which activity was current at a note's turn is not kept: linked to the
    session's last one, a rule from an earlier activity would be on the wrong
    one, silently. Unlinked, it is only missing from a block."""
    # the owner's own words: a rule kept at the close is written, not proposed
    said = "Herbs always go in partial shade."
    unsure = {**rule(ACTIVITY, "herbs go in partial shade", SHADE), "duration": "unsure"}
    fake_openai.script(
        chat=[
            reply(unsure, activity=asked()),
            reply(activity=asked("drawing a map", "Draw a map of the plan")),
            chat_json({"episode": None, "keep": [0]}),
        ]
    )
    ingest(client, app, f"{ASK} {said}", session_id="s")
    ingest(client, app, "Draw a map of the plan.", session_id="s")
    end_session(client, "s")
    closed = next(j["result"] for j in run_all(client, app)["jobs"])["session"]
    assert closed["written"] == 1 and closed["proposed"] == 0
    (shade,) = rules(app)
    assert shade.node_id == tasks(app)[0].id  # its subject, as the model gave it
    assert links(app, shade) == []


def test_a_proposed_episode_keeps_only_live_activities(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    note = {
        "subject": "spring vegetable garden",
        "statement": "The spring vegetable garden needs a planting plan.",
        "quote": "Plan the spring vegetable garden",
        "modality": "fact",
        "duration": "task",
    }
    fake_openai.script(
        chat=[
            reply(note, activity=asked()),
            reply(activity=asked("drawing a map", "Draw a map of the plan")),
            # it names a known subject no source of the session names: proposed
            chat_json({"episode": "NOVA planned the garden with Bob.", "keep": []}),
        ]
    )
    app.state.store.create_node(scopes=[SCOPE], name="Bob", kind="person")
    ingest(client, app, ASK, session_id="s")
    ingest(client, app, "Draw a map of the plan.", session_id="s")
    plan, drawing = tasks(app)
    assert client.post("/v1/archive", json={"id": drawing.id}).status_code == 200
    end_session(client, "s")
    run_all(client, app)
    store: Store = app.state.store
    (proposal_id,) = [
        r["id"]
        for r in store.execute(
            "SELECT id FROM proposals WHERE kind = 'fact' AND value LIKE 'NOVA planned%'"
        ).fetchall()
    ]
    assert store.get_proposal(proposal_id).payload["activities"] == [plan.id]


def test_a_malformed_activity_in_the_user_perspective_is_counted_once(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = make(tmp_path, fake_openai, clock, selves=())
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[chat_json({"nodes": [], "facts": [], "edges": [], "activity": "draw"})]
            )
            result = ingest(client, application, ASK, session_id="s")
            assert result["rejected"]["keys"] == 1
    finally:
        application.state.store.close()
