"""Intent before content: the mechanics only. A faked model
cannot prove what a real one reads - the ``--intent`` corpus measures that -
but it proves what the service does with what it is told: a non-sincere frame
or a modality other than ``fact`` writes no fact from the content, a missing or
unsure frame or modality makes a proposal (never a fact), a frame carries over
its session until it is ended, and ``record: false`` writes nothing at all."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest import sessions
from iifa.ingest.extract import Extractor
from iifa.ingest.worker import Budget, JobContext, ModelSession
from iifa.store.models import Fact
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, post_turn, run_all, tier2_settings

pytestmark = pytest.mark.security


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    settings = tier2_settings(tmp_path, fake_openai, extraction_intent=True)
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def fact(
    subject: str,
    statement: str,
    quote: str,
    *,
    key: str | None = None,
    value: str | None = None,
    **intent: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {"subject": subject, "statement": statement, "quote": quote}
    if key is not None:
        out.update(attribute=key, value=value)
    out.update(intent)
    return out


def sincere(subject: str, statement: str, quote: str, **extra: Any) -> dict[str, Any]:
    return fact(subject, statement, quote, modality="fact", duration="lasting", **extra)


def reply(
    facts: list[dict[str, Any]],
    *,
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    **frame: Any,
) -> Reply:
    names = {f["subject"] for f in facts if f["subject"] != "User"}
    body: dict[str, Any] = {
        "nodes": nodes if nodes is not None else [{"name": n, "kind": "person"} for n in names],
        "facts": facts,
        "edges": edges or [],
    }
    body.update(frame)
    return chat_json(body)


def ingest(client: TestClient, app: FastAPI, user: str, **extra: Any) -> dict[str, Any]:
    post_turn(client, user, **extra)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def all_facts(app: FastAPI) -> list[Fact]:
    store: Store = app.state.store
    rows = store.execute("SELECT id FROM facts ORDER BY created_at, id").fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def fact_proposals(app: FastAPI) -> list[Any]:
    store: Store = app.state.store
    rows = store.execute(
        "SELECT id FROM proposals WHERE kind = 'fact' ORDER BY created_at, id"
    ).fetchall()
    return [store.get_proposal(r["id"]) for r in rows]


def node_names(app: FastAPI) -> set[str]:
    return {r["name"] for r in app.state.store.execute("SELECT name FROM nodes").fetchall()}


TEST_TURN = "For this exercise, I am a pilot named Daniela and I live in Porto."


# --------------------------------------------------------------------------- the frame


def test_a_non_sincere_frame_writes_nothing_of_its_content_and_one_episode(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    episode = sincere("User", "User played a pilot in an exercise.", TEST_TURN, episode=True)
    fake_openai.script(
        chat=[
            reply(
                [
                    sincere(
                        "Daniela", "Daniela lives in Porto.", TEST_TURN, key="home", value="Porto"
                    ),
                    episode,
                    {**episode, "statement": "User tried a second test."},  # one per frame
                ],
                edges=[{"src": "Daniela", "dst": "Porto", "type": "lives_in", "quote": TEST_TURN}],
                frame="pretend",
            )
        ]
    )
    result = ingest(client, app, TEST_TURN)
    facts = all_facts(app)
    assert [f.statement for f in facts] == ["User played a pilot in an exercise."]
    (written,) = facts
    assert written.attribute == "activity" and written.value is None
    assert "episode" in written.flags and "speaker" in written.flags
    assert node_names(app) == set()  # neither the role nor the city became a node
    assert app.state.store.execute("SELECT count(*) FROM edges").fetchone()[0] == 0
    assert result["dropped_by_intent"] == {"frame": {"pretend": 1}, "modality": {}}
    assert result["intent"]["episodes"] == {"kept": 1, "proposed": 0, "dropped": 1}
    assert fact_proposals(app) == []


@pytest.mark.parametrize("frame", ["pretend", "example", "irony", "voice", "staged"])
def test_every_non_sincere_frame_writes_no_fact_from_its_content(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, frame: str
) -> None:
    quote = "Say Karim lives in Porto."
    fake_openai.script(
        chat=[
            reply(
                [sincere("Karim", "Karim lives in Porto.", quote, key="home", value="Porto")],
                frame=frame,
            )
        ]
    )
    result = ingest(client, app, quote)
    assert all_facts(app) == [] and node_names(app) == set()
    assert result["dropped_by_intent"]["frame"] == {frame: 1}
    assert fact_proposals(app) == []


def test_an_episode_that_carries_a_value_of_the_content_is_only_proposed(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    leaking = sincere("User", "User played a pilot living in Porto.", TEST_TURN, episode=True)
    fake_openai.script(
        chat=[
            reply(
                [
                    sincere(
                        "Daniela", "Daniela lives in Porto.", TEST_TURN, key="home", value="Porto"
                    ),
                    leaking,
                ],
                frame="pretend",
            )
        ]
    )
    result = ingest(client, app, TEST_TURN)
    assert all_facts(app) == []
    (proposal,) = fact_proposals(app)
    assert proposal.value == "User played a pilot living in Porto."
    assert proposal.payload["why"] == "intent:episode"
    assert result["intent"]["episodes"]["proposed"] == 1


def test_an_episode_about_someone_else_is_dropped(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    fake_openai.script(
        chat=[
            reply(
                [sincere("Daniela", "Daniela took part in a test.", TEST_TURN, episode=True)],
                frame="pretend",
            )
        ]
    )
    result = ingest(client, app, TEST_TURN)
    assert all_facts(app) == [] and fact_proposals(app) == []
    assert result["intent"]["episodes"]["dropped"] == 1


# --------------------------------------------------------------------------- the modality


@pytest.mark.parametrize("modality", ["question", "hypothesis", "plan", "guess", "proposal"])
def test_a_modality_other_than_fact_writes_no_fact(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, modality: str
) -> None:
    quote = "If Karim moved to Porto, the commute would be long."
    fake_openai.script(
        chat=[
            reply(
                [
                    fact(
                        "Karim",
                        "Karim lives in Porto.",
                        quote,
                        key="home",
                        value="Porto",
                        modality=modality,
                        duration="lasting",
                    )
                ],
                nodes=[{"name": "Karim", "kind": "person"}, {"name": "Porto", "kind": "place"}],
                edges=[{"src": "Karim", "dst": "Porto", "type": "lives_in", "quote": quote}],
                frame="sincere",
            )
        ]
    )
    result = ingest(client, app, quote)
    assert all_facts(app) == [] and fact_proposals(app) == []
    assert node_names(app) == set()  # nodes and edges named only by what was dropped
    assert result["dropped_by_intent"] == {"frame": {}, "modality": {modality: 1}}


def test_the_sincere_facts_beside_them_are_written_as_before(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "I test cars for a living. Karim might move to Porto one day."
    fake_openai.script(
        chat=[
            reply(
                [
                    sincere(
                        "User",
                        "User tests cars for a living.",
                        "I test cars for a living",
                        key="role",
                        value="car tester",
                    ),
                    fact(
                        "Karim",
                        "Karim lives in Porto.",
                        "Karim might move to Porto one day",
                        key="home",
                        value="Porto",
                        modality="plan",
                        duration="lasting",
                    ),
                ],
                nodes=[],
                frame="sincere",
            )
        ]
    )
    result = ingest(client, app, text)
    assert [f.statement for f in all_facts(app)] == ["User tests cars for a living."]
    assert result["facts"]["written"] == 1
    assert result["dropped_by_intent"]["modality"] == {"plan": 1}


# --------------------------------------------------------------------------- doubt makes a proposal


@pytest.mark.parametrize(
    ("frame", "modality", "why"),
    [
        (None, "fact", "intent:frame:unsure"),  # no frame given
        ("unsure", "fact", "intent:frame:unsure"),
        ("sincere", None, "intent:modality:unsure"),  # no modality given
        ("sincere", "unsure", "intent:modality:unsure"),
        ("theatre", "fact", "intent:frame:unsure"),  # outside the closed list: not given
    ],
)
def test_a_missing_or_unsure_frame_or_modality_makes_a_proposal_never_a_fact(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    frame: str | None,
    modality: str | None,
    why: str,
) -> None:
    quote = "Karim lives in Porto."
    intent = {"duration": "lasting"} | ({"modality": modality} if modality else {})
    frames = {"frame": frame} if frame else {}
    fake_openai.script(
        chat=[
            reply(
                [
                    fact(
                        "Karim", "Karim lives in Porto.", quote, key="home", value="Porto", **intent
                    )
                ],
                **frames,
            )
        ]
    )
    result = ingest(client, app, quote)
    assert all_facts(app) == []
    assert node_names(app) == set()  # a proposal creates no node
    (proposal,) = fact_proposals(app)
    assert proposal.status == "pending" and proposal.node_id is None
    assert proposal.payload["why"] == why
    assert result["facts"]["proposed"] == 1 and result["facts"]["written"] == 0
    listed = client.get("/v1/proposals", params={"scope": SCOPE}, headers=auth("all")).json()
    assert [item["why"] for item in listed["items"]] == [why]


def test_a_proposal_lands_on_the_node_its_subject_already_names(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    karim = app.state.store.create_node(scopes=[SCOPE], name="Karim", kind="person")
    quote = "Karim lives in Porto."
    fake_openai.script(
        chat=[
            reply(
                [fact("Karim", "Karim lives in Porto.", quote, modality="unsure")], frame="sincere"
            )
        ]
    )
    ingest(client, app, quote)
    (proposal,) = fact_proposals(app)
    assert proposal.node_id == karim.id


# --------------------------------------------------------------------------- frames over a session


def test_a_frame_holds_over_its_session_until_a_turn_ends_it(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    inside = "My e-mail is daniela@example.org."
    after = "I moved to Porto last week."
    fake_openai.script(
        chat=[
            reply(
                [sincere("User", "User played a role in an exercise.", TEST_TURN, episode=True)],
                frame="pretend",
                frame_holds=True,
            ),
            # read alone as sincere, while the session's frame is open: a doubt
            reply(
                [
                    sincere(
                        "User",
                        "User's e-mail is daniela@example.org.",
                        inside,
                        key="email",
                        value="daniela@example.org",
                    )
                ],
                frame="sincere",
            ),
            reply(
                [sincere("User", "User moved to Porto.", after, key="home", value="Porto")],
                frame="sincere",
                frame_ends=True,
            ),
        ]
    )
    first = ingest(client, app, TEST_TURN, session_id="s1")
    assert first["intent"]["frame_opened"] == 1
    second = ingest(client, app, inside, session_id="s1")
    assert second["intent"]["frames"] == {"unsure": 1} and second["intent"]["frame_carried"] == 1
    # the second call saw the open frame, as data between its own markers
    shown = fake_openai.chat_calls[1].body["messages"]
    assert "<<<SESSION>>>" in shown[-1]["content"] and "still open: pretend" in shown[-1]["content"]
    assert "<<<SESSION>>>" in shown[0]["content"]  # the wall names the block
    third = ingest(client, app, after, session_id="s1")
    assert third["intent"]["frame_ended"] == 1
    statements = [f.statement for f in all_facts(app)]
    assert statements == ["User played a role in an exercise.", "User moved to Porto."]
    (proposal,) = fact_proposals(app)
    assert proposal.value == "User's e-mail is daniela@example.org."
    # no frame is left open, and the trace holds the sincere turn only:
    # nothing said inside the frame reaches the session's close
    state = sessions.load(app.state.store, "s1", [SCOPE])
    assert state.frame is None and state.facts == []
    assert [t.excerpt for t in state.turns] == [after]


def test_p9_a_frame_is_never_carried_into_another_scope_set(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    other = "Karim lives in Porto."
    fake_openai.script(
        chat=[
            reply([], frame="pretend", frame_holds=True),
            reply([sincere("Karim", other, other, key="home", value="Porto")], frame="sincere"),
        ]
    )
    ingest(client, app, TEST_TURN, session_id="shared-id")
    result = ingest(client, app, other, session_id="shared-id", scope="user:bob")
    assert result["intent"]["frame_carried"] == 0
    assert [f.statement for f in all_facts(app)] == [other]
    assert "<<<SESSION>>>" not in fake_openai.chat_calls[1].body["messages"][-1]["content"]


def test_a_frame_opened_meanwhile_by_another_turn_stays_open(
    app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Two turns of one session processed side by side: the one that read no
    frame must not close the frame the other opened while it was calling the
    model. The seam runs between the model calls and the write."""
    store: Store = app.state.store
    fake_openai.script(
        chat=[
            reply(
                [sincere("Karim", "Karim lives in Porto.", "Karim lives in Porto.")],
                frame="sincere",
            )
        ]
    )
    session = ModelSession(app.state.models, Budget(), lambda: 0.0)
    ctx = JobContext(store, app.state.queue, session, store.clock, app.state.worker.pipeline)
    extractor = Extractor(ctx, [SCOPE])

    def another_turn_opens_a_frame() -> None:
        state = sessions.load(store, "s1", [SCOPE])
        sessions.save(store, state, frame="pretend", frame_episode=True, ended=False, facts=[])

    extractor._before_write = another_turn_opens_a_frame
    extractor.run("Karim lives in Porto.", kind="turn", session_id="s1")
    assert sessions.load(store, "s1", [SCOPE]).frame == "pretend"
    assert [f.statement for f in all_facts(app)] == ["Karim lives in Porto."]


# --------------------------------------------------------------------------- record: false


def test_a_turn_sent_with_record_false_writes_nothing_at_all(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store: Store = app.state.store
    response = client.post(
        "/v1/turns",
        json={"scope": SCOPE, "user": "Test: I live in Porto.", "record": False},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"] is None and body["status"] == "not_recorded"
    assert store.execute("SELECT count(*) FROM queue").fetchone()[0] == 0
    run_all(client, app)
    assert fake_openai.chat_calls == []
    assert store.execute("SELECT count(*) FROM sources").fetchone()[0] == 0
    assert all_facts(app) == []


def test_the_intent_can_be_turned_off_and_the_prompt_then_does_not_ask_for_it(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    settings = tier2_settings(tmp_path, fake_openai, extraction_intent=False)
    application = create_app(settings, clock=clock)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[reply([fact("Karim", "Karim lives in Porto.", "Karim lives in Porto.")])]
            )
            result = ingest(client, application, "Karim lives in Porto.")
            assert result["intent"] is None and result["facts"]["written"] == 1
            system = fake_openai.chat_calls[0].body["messages"][0]["content"]
            assert '"frame"' not in system and '"modality"' not in system
    finally:
        application.state.store.close()


def test_a_non_sincere_frame_links_no_known_names_either(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Tier-0 co-occurrence reads the raw text: in a pretend frame it must not."""
    store: Store = app.state.store
    store.create_node(scopes=[SCOPE], name="Karim", kind="person")
    store.create_node(scopes=[SCOPE], name="Nadia", kind="person")
    fake_openai.script(chat=[reply([], nodes=[], frame="pretend")])
    ingest(client, app, "For this test I am Karim writing to Nadia.")
    assert store.execute("SELECT count(*) FROM edges").fetchone()[0] == 0
