"""The agent perspective beyond one turn (SPEC 4.2, 5.8, 8.1, 8.3): the
session's close writes its episode about the self, the profile carries the
self's commitments, and the agent may remember about itself."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest.session_prompts import CONSOLIDATION_START
from iifa.scopes import Visibility
from iifa.store.models import Fact
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, end_session, post_turn, run_all, tier2_settings

SELF = "NOVA"
VIEW = Visibility.of_scope(SCOPE)


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


def noted(subject: str, statement: str, quote: str, duration: str, **extra: Any) -> dict[str, Any]:
    return {
        "subject": subject,
        "statement": statement,
        "quote": quote,
        "modality": "fact",
        "duration": duration,
        **extra,
    }


def sincere(*facts: dict[str, Any]) -> Reply:
    return chat_json({"nodes": [], "facts": list(facts), "edges": [], "frame": "sincere"})


def closed(client: TestClient, app: FastAPI, session: str = "s") -> dict[str, Any]:
    end_session(client, session)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "session_end")
    return dict(result["session"])


def live_facts(app: FastAPI) -> list[Fact]:
    store: Store = app.state.store
    rows = store.execute(
        "SELECT id FROM facts WHERE valid_to IS NULL ORDER BY created_at, id"
    ).fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def remember_self(client: TestClient, statement: str, **extra: Any) -> Any:
    return client.post(
        "/v1/remember",
        json={"scope": SCOPE, "statement": statement, "about_self": True, **extra},
    )


# --------------------------------------------------------------------------- the episode


DEBUG = "Let's plan the spring garden."
EPISODE = "NOVA planned the spring garden with Alice; the tomatoes go by the south wall."


def test_the_close_writes_one_episode_about_the_self_naming_the_owner(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    # the self's node exists already, and no source of the session names it:
    # the self is never "a known subject the session did not name"
    assert remember_self(client, "NOVA answers in short paragraphs.").status_code == 201
    fake_openai.script(
        chat=[
            sincere(
                noted("spring garden", "The spring garden needs a planting plan.", DEBUG, "task")
            ),
            chat_json({"episode": EPISODE, "keep": []}),
        ]
    )
    post_turn(client, DEBUG, session_id="s")
    run_all(client, app)
    result = closed(client, app)
    assert result["written"] == 1 and result["proposed"] == 0
    me = app.state.store.find_node(SELF, VIEW)
    assert me is not None
    # one episode, on the self's node, none on the owner's
    (episode,) = [f for f in live_facts(app) if "episode" in f.flags]
    assert episode.statement == EPISODE and episode.node_id == me.id
    assert episode.attribute == "activity" and "episode" in episode.flags
    assert episode.provenance == "owner"  # it leans on the owner's own words
    (call,) = [
        c.body["messages"]
        for c in fake_openai.chat_calls
        if c.body["messages"][0]["content"].startswith(CONSOLIDATION_START)
    ]
    system = call[0]["content"]
    assert f'about {SELF} (use "{SELF}" as its subject)' in system
    assert "Alice" in system


def test_an_episode_of_the_agents_own_work_is_written_from_its_words(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    said = "I found it: the south wall gets the most sun."
    fake_openai.script(
        chat=[
            sincere(
                noted(
                    "spring garden",
                    "The spring garden's south wall gets the most sun.",
                    said,
                    "task",
                )
            ),
            chat_json({"episode": EPISODE, "keep": []}),
        ]
    )
    post_turn(client, "Go on.", said, session_id="s")
    run_all(client, app)
    result = closed(client, app)
    assert result["written"] == 1
    episode = next(f for f in live_facts(app) if f.statement == EPISODE)
    assert (episode.provenance, episode.part_ref) == ("assistant", SELF)


def test_an_episode_from_a_turn_whose_pages_were_all_vouched_is_written_and_flagged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    said = "Following your seed catalogue, the tomatoes go by the south wall."
    fake_openai.script(
        chat=[
            sincere(
                noted(
                    "spring garden",
                    "The spring garden's tomatoes go by the south wall.",
                    said,
                    "task",
                )
            ),
            chat_json({"episode": EPISODE, "keep": []}),
        ]
    )
    post_turn(
        client,
        "Go on.",
        said,
        session_id="s",
        external=[{"kind": "document", "text": "Seed catalogue, spring.", "vouched": True}],
    )
    run_all(client, app)
    result = closed(client, app)
    assert result["written"] == 1 and result["proposed"] == 0
    episode = next(f for f in live_facts(app) if f.statement == EPISODE)
    assert "vouched" in episode.flags and "episode" in episode.flags


def test_without_a_self_the_episode_stays_the_owners(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = make(tmp_path, fake_openai, clock, selves=())
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    sincere(
                        noted("spring garden", "The spring garden needs a plan.", DEBUG, "task")
                    ),
                    chat_json({"episode": "Alice planned the spring garden.", "keep": []}),
                ]
            )
            post_turn(client, DEBUG, session_id="s")
            run_all(client, application)
            closed(client, application)
            (episode,) = live_facts(application)
            owner = application.state.store.find_node("Alice", VIEW)
            assert owner is not None and episode.node_id == owner.id
            system = next(
                c.body["messages"][0]["content"]
                for c in fake_openai.chat_calls
                if c.body["messages"][0]["content"].startswith(CONSOLIDATION_START)
            )
            assert 'about Alice (use "Alice" as its subject)' in system
    finally:
        application.state.store.close()


# --------------------------------------------------------------------------- the profile


def commitment(value: str, quote: str) -> Reply:
    return answer(
        facts=[
            {
                "subject": SELF,
                "statement": f"NOVA's commitment before a booking: {value}.",
                "attribute": "commitment.bookings",
                "value": value,
                "quote": quote,
            }
        ]
    )


def test_the_profile_carries_the_selfs_live_commitments_after_the_pinned_facts(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = make(tmp_path, fake_openai, clock, extraction_intent=False)
    try:
        with TestClient(application, headers=auth("all")) as client:
            # a self with no node yet: nothing of it in the profile
            empty = client.post("/v1/session/start", json={"scope": SCOPE}).json()
            assert empty["items"] == [] and empty["reason"] == "empty_profile"
            pinned = client.post(
                "/v1/remember",
                json={"scope": SCOPE, "statement": "Alice is allergic to nuts.", "pinned": True},
            ).json()["fact"]["id"]
            fake_openai.script(
                chat=[
                    commitment("date check", "I will confirm the date first"),
                    # the same topic changed, by the same speaker: it replaces
                    commitment("Alice approval", "from now on I ask Alice"),
                ]
            )
            post_turn(client, "Book the table.", "I will confirm the date first.")
            run_all(client, application)
            post_turn(client, "Ask me before.", "Understood: from now on I ask Alice.")
            run_all(client, application)
            (current,) = [
                f for f in live_facts(application) if f.attribute == "commitment.bookings"
            ]
            assert current.value == "Alice approval"
            body = {"scope": SCOPE, "budget_tokens": 2000}
            first = client.post("/v1/session/start", json=body).json()
            again = client.post("/v1/session/start", json=body).json()
            assert first == again  # deterministic: a cacheable prefix
            ids = [i["id"] for i in first["items"]]
            assert ids[:2] == [pinned, current.id]
            item = first["items"][1]
            assert item["kind"] == "fact" and "pinned" not in item["flags"]
            assert item["from"] == SELF
            assert all(i["kind"] == "node" for i in first["items"][2:])
    finally:
        application.state.store.close()


def test_the_profile_carries_the_selfs_preferences_too_and_not_the_owners(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = make(tmp_path, fake_openai, clock, extraction_intent=False)

    def preference(subject: str, statement: str, key: str, value: str, quote: str) -> Reply:
        return answer(
            facts=[
                {
                    "subject": subject,
                    "statement": statement,
                    "attribute": key,
                    "value": value,
                    "quote": quote,
                }
            ]
        )

    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    preference(
                        SELF,
                        "NOVA prefers to keep its answers short.",
                        "preference.answers",
                        "short answers",
                        "I will keep my answers short",
                    ),
                    preference(
                        "Alice", "Alice prefers tea.", "preference.drink", "tea", "I prefer tea"
                    ),
                ]
            )
            post_turn(client, "Hi.", "I will keep my answers short.")
            run_all(client, application)
            post_turn(client, "I prefer tea.", "Noted.")
            run_all(client, application)
            held = {f.attribute: f.id for f in live_facts(application)}
            assert set(held) == {"preference.answers", "preference.drink"}
            items = client.post(
                "/v1/session/start", json={"scope": SCOPE, "budget_tokens": 2000}
            ).json()["items"]
            facts = [i for i in items if i["kind"] == "fact"]
            # a rule of the self, not a pin; the owner's preference is not one
            assert [i["id"] for i in facts] == [held["preference.answers"]]
            assert "pinned" not in facts[0]["flags"] and facts[0]["from"] == SELF
    finally:
        application.state.store.close()


def test_a_profile_without_a_self_has_no_commitments(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = make(tmp_path, fake_openai, clock, selves=(), extraction_intent=False)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(chat=[commitment("date check", "I will confirm the date first")])
            post_turn(client, "Book the table.", "I will confirm the date first.")
            run_all(client, application)
            items = client.post("/v1/session/start", json={"scope": SCOPE}).json()["items"]
            assert all(i["kind"] == "node" for i in items)
    finally:
        application.state.store.close()


# --------------------------------------------------------------------------- remember


def test_remember_about_the_self_writes_an_extracted_fact_on_its_node(
    client: TestClient, app: FastAPI
) -> None:
    response = remember_self(
        client,
        "NOVA confirms the date before a booking.",
        attribute="commitment.bookings",
        value="date check",
    )
    assert response.status_code == 201, response.text
    body = response.json()
    me = app.state.store.find_node(SELF, VIEW)
    assert me is not None and me.scopes == (SCOPE,) and body["node"]["id"] == me.id
    assert body["echo"] is False
    fact = app.state.store.get_fact(body["fact"]["id"])
    assert (fact.origin, fact.confidence, fact.provenance, fact.part_ref) == (
        "extracted",
        1.0,
        "assistant",
        SELF,
    )
    recalled = client.post("/v1/recall", json={"scope": SCOPE, "query": "date booking"}).json()
    assert next(i for i in recalled["items"] if i["id"] == fact.id)["from"] == SELF


def test_remember_about_the_self_said_again_is_an_echo_and_another_value_a_conflict(
    client: TestClient, app: FastAPI
) -> None:
    first = remember_self(
        client, "NOVA answers in short paragraphs.", attribute="preference.answers", value="short"
    ).json()["fact"]["id"]
    # the same statement, then the same value in other words
    for statement in ("NOVA answers in short paragraphs.", "NOVA keeps its answers short."):
        again = remember_self(client, statement, attribute="preference.answers", value="short")
        assert again.status_code == 201
        assert again.json()["echo"] is True and again.json()["fact"]["id"] == first
    other = remember_self(
        client, "NOVA answers at length.", attribute="preference.answers", value="long"
    ).json()
    assert other["echo"] is False
    store: Store = app.state.store
    new = store.get_fact(other["fact"]["id"])
    assert "conflict" in new.flags and new.conflicts_with == first
    assert store.get_fact(first).valid_to is None  # never closed
    assert len(live_facts(app)) == 2


def test_remember_about_the_self_is_refused_without_a_self_or_with_a_node(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock, client: TestClient
) -> None:
    both = remember_self(client, "NOVA answers briefly.", node={"name": "X", "kind": "other"})
    assert both.status_code == 422
    (tmp_path / "plain").mkdir()
    plain = make(tmp_path / "plain", fake_openai, clock, selves=())
    try:
        with TestClient(plain, headers=auth("all")) as other:
            assert remember_self(other, "NOVA answers briefly.").status_code == 422
    finally:
        plain.state.store.close()


# --------------------------------------------------------------------------- one topic, two families


def rule(key: str, value: str, quote: str, statement: str) -> Reply:
    return answer(
        facts=[
            {
                "subject": SELF,
                "statement": statement,
                "attribute": key,
                "value": value,
                "quote": quote,
            }
        ]
    )


def turn(client: TestClient, app: FastAPI, user: str, assistant: str) -> dict[str, Any]:
    post_turn(client, user, assistant)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return dict(result["facts"])


def test_a_rule_filed_under_the_other_family_with_the_same_value_is_an_echo(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """The model files one rule as a preference in one turn and as a commitment
    in the next: without any review setting, that is still one rule."""
    application = make(tmp_path, fake_openai, clock, extraction_intent=False)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    rule(
                        "preference.answers",
                        "short answers",
                        "I will keep my answers short",
                        "NOVA prefers to keep its answers short.",
                    ),
                    rule(
                        "commitment.answers",
                        "short answers",
                        "my answers stay short",
                        "NOVA commits to short answers.",
                    ),
                    # said by the owner this time: still the rule held
                    rule(
                        "commitment.answers",
                        "short answers",
                        "keep your answers short",
                        "NOVA must keep its answers short.",
                    ),
                ]
            )
            turn(client, application, "Hi.", "I will keep my answers short.")
            again = turn(client, application, "Hi.", "As agreed, my answers stay short.")
            assert (again["written"], again["self_echo"], again["conflict"]) == (0, 1, 0)
            owner = turn(client, application, "Please keep your answers short.", "Noted.")
            assert (owner["written"], owner["conflict"]) == (0, 0)
            # the owner's words are not the self's echo: a plain duplicate
            assert (owner["duplicate"], owner["self_echo"]) == (1, 0)
            (held,) = live_facts(application)
            assert held.attribute == "preference.answers" and "conflict" not in held.flags
    finally:
        application.state.store.close()


def test_another_value_under_the_other_family_is_a_visible_conflict_never_a_close(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """A preference and a commitment of one topic with two values: nothing tells
    whether one refines or contradicts the other, so neither is closed."""
    application = make(tmp_path, fake_openai, clock, extraction_intent=False)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    rule(
                        "preference.answers",
                        "short answers",
                        "I will keep my answers short",
                        "NOVA prefers to keep its answers short.",
                    ),
                    rule(
                        "commitment.answers",
                        "full detail",
                        "I will answer in full detail",
                        "NOVA commits to answering in full detail.",
                    ),
                    # another topic: no conflict with either
                    rule(
                        "commitment.reports",
                        "full detail",
                        "reports get full detail",
                        "NOVA commits to full detail in reports.",
                    ),
                    # the reverse order: a preference beside a commitment held
                    rule(
                        "preference.reports",
                        "one page",
                        "I prefer one-page reports",
                        "NOVA prefers one-page reports.",
                    ),
                ]
            )
            turn(client, application, "Hi.", "I will keep my answers short.")
            other = turn(client, application, "More detail.", "I will answer in full detail.")
            assert (other["written"], other["conflict"], other["superseded"]) == (1, 1, 0)
            apart = turn(client, application, "And reports?", "Yes, reports get full detail.")
            assert (apart["written"], apart["conflict"]) == (1, 0)
            back = turn(client, application, "Hm.", "Though I prefer one-page reports.")
            assert (back["written"], back["conflict"], back["superseded"]) == (1, 1, 0)
            held = {f.attribute: f for f in live_facts(application)}
            assert set(held) == {
                "preference.answers",
                "commitment.answers",
                "commitment.reports",
                "preference.reports",
            }
            newer = held["commitment.answers"]
            assert "conflict" in newer.flags
            assert newer.conflicts_with == held["preference.answers"].id
            assert held["preference.reports"].conflicts_with == held["commitment.reports"].id
            # both start the next session, the conflict shown
            items = client.post(
                "/v1/session/start", json={"scope": SCOPE, "budget_tokens": 2000}
            ).json()["items"]
            shown = {i["attribute"]: i["flags"] for i in items if i["kind"] == "fact"}
            assert set(shown) == set(held)
            # on both sides of each pair, though only the newer fact stores it
            for key in ("commitment.answers", "preference.answers", "preference.reports"):
                assert "conflict" in shown[key], key
            assert "conflict" in shown["commitment.reports"]
    finally:
        application.state.store.close()


def test_a_commitment_still_replaces_the_commitment_of_its_topic_beside_a_twin(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """The value the preference holds, said as a commitment while another
    commitment is live: not an echo of the preference - the commitment of
    the topic changes, as it does with no preference beside it."""
    application = make(tmp_path, fake_openai, clock, extraction_intent=False)
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    rule(
                        "preference.answers",
                        "short answers",
                        "I will keep my answers short",
                        "NOVA prefers to keep its answers short.",
                    ),
                    rule(
                        "commitment.answers",
                        "full detail",
                        "I will answer in full detail",
                        "NOVA commits to answering in full detail.",
                    ),
                    rule(
                        "commitment.answers",
                        "short answers",
                        "from now on I commit to short answers",
                        "NOVA commits to short answers.",
                    ),
                ]
            )
            turn(client, application, "Hi.", "I will keep my answers short.")
            turn(client, application, "More detail.", "I will answer in full detail.")
            back = turn(client, application, "Too long.", "From now on I commit to short answers.")
            assert (back["written"], back["superseded"], back["conflict"]) == (1, 1, 0)
            held = {f.attribute: f for f in live_facts(application)}
            assert set(held) == {"preference.answers", "commitment.answers"}
            assert held["commitment.answers"].value == "short answers"
            assert "conflict" not in held["commitment.answers"].flags
    finally:
        application.state.store.close()


def test_remember_about_the_self_reads_the_two_rule_families_as_one_topic(
    client: TestClient, app: FastAPI
) -> None:
    first = remember_self(
        client, "NOVA answers in short paragraphs.", attribute="preference.answers", value="short"
    ).json()["fact"]["id"]
    again = remember_self(
        client, "NOVA commits to short answers.", attribute="commitment.answers", value="short"
    ).json()
    assert again["echo"] is True and again["fact"]["id"] == first
    other = remember_self(
        client, "NOVA commits to long answers.", attribute="commitment.answers", value="long"
    ).json()
    assert other["echo"] is False
    new = app.state.store.get_fact(other["fact"]["id"])
    assert "conflict" in new.flags and new.conflicts_with == first
    assert app.state.store.get_fact(first).valid_to is None
    # the same value on another topic is another rule
    apart = remember_self(
        client, "NOVA writes short reports.", attribute="commitment.reports", value="short"
    ).json()
    assert apart["echo"] is False
    assert "conflict" not in app.state.store.get_fact(apart["fact"]["id"]).flags
    # the preference's value said as a commitment while another commitment is
    # held: not an echo of the preference, a conflict with that commitment
    back = remember_self(
        client, "NOVA commits to brevity again.", attribute="commitment.answers", value="short"
    ).json()
    assert back["echo"] is False
    again = app.state.store.get_fact(back["fact"]["id"])
    assert "conflict" in again.flags and again.conflicts_with == other["fact"]["id"]


# --------------------------------------------------------------------------- the profile's bound


def test_the_selfs_rules_never_push_the_central_nodes_out_of_a_small_profile(
    client: TestClient, app: FastAPI
) -> None:
    """Commitments first, then preferences, the newest first in each; ahead of
    the nodes they take at most a third of the budget, the others come after
    the nodes and take what is left."""
    topics = [f"topic{i:02d}" for i in range(12)]
    for topic in topics:
        made = remember_self(
            client,
            f"NOVA prefers to handle {topic} with care, slowly and in the order given.",
            attribute=f"preference.{topic}",
            value=f"care {topic}",
        )
        assert made.status_code == 201, made.text
    for topic in ("bookings", "dates"):
        remember_self(
            client,
            f"NOVA confirms {topic} with the owner before acting on them.",
            attribute=f"commitment.{topic}",
            value=f"confirm {topic}",
        )
    for name in ("Solar", "Lumen", "Harbor"):
        client.post(
            "/v1/remember",
            json={
                "scope": SCOPE,
                "statement": f"{name} is a project of Alice.",
                "node": {"name": name, "kind": "project"},
            },
        )
    engine = app.state.engine
    whole = client.post("/v1/session/start", json={"scope": SCOPE, "budget_tokens": 8000}).json()
    assert whole["omitted"] == 0
    rules = [i for i in whole["items"] if i["kind"] == "fact"]
    assert [i["attribute"] for i in rules] == [
        "commitment.dates",
        "commitment.bookings",
        *(f"preference.{t}" for t in reversed(topics)),
    ]
    nodes = [i["id"] for i in whole["items"] if i["kind"] == "node"]
    assert len(nodes) == 4  # the three projects and the self
    budget = sum(engine.cost(i) for i in whole["items"]) // 2
    body = {"scope": SCOPE, "budget_tokens": budget}
    small = client.post("/v1/session/start", json=body).json()
    assert small == client.post("/v1/session/start", json=body).json()  # still deterministic
    kinds = [i["kind"] for i in small["items"]]
    first_node = kinds.index("node")
    ahead = small["items"][:first_node]
    assert ahead and all(i["kind"] == "fact" for i in ahead)
    assert sum(engine.cost(i) for i in ahead) <= budget // 3
    assert [i["attribute"] for i in ahead[:2]] == ["commitment.dates", "commitment.bookings"]
    # every node is there: no rule took its place
    assert [i["id"] for i in small["items"] if i["kind"] == "node"] == nodes
    # the rules that did not fit ahead come after the nodes, in the same order
    behind = small["items"][first_node + len(nodes) :]
    assert behind and all(i["kind"] == "fact" for i in behind)
    order = [i["id"] for i in rules]
    shown = [i["id"] for i in small["items"] if i["kind"] == "fact"]
    assert shown == [i for i in order if i in shown]
    assert small["used_tokens"] <= budget and small["omitted"] > 0
