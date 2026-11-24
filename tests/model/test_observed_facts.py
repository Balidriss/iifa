"""What was observed is true at a date: the state of
something that changes without anyone telling the memory - a shelf, a
service, a queue - seen at one moment. The model reads it (the ``observed``
duration) and the service reads it from provenance (a lasting fact in a
``tool`` part). An observed fact is memory, flagged ``observed``, dated by
``observed_at``; a later observation of the same value moves that date; a
newer observation of the same key closes the older one, past P6's guards;
recall ranks it lower as it ages; its handles say ``observed`` and count
``age_days`` from the observation."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.store.models import Fact
from iifa.store.repo import Store
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer, Reply, chat_json
from tests.helpers import auth
from tests.tier2 import SCOPE, post_turn, run_all, tier2_settings

SHELF = "Pantry"
EMPTY = "The Pantry shelf is empty."
THREE = "The Pantry shelf holds 3 jars."
SPICES = "The Pantry shelf holds the spices."
SPARE = "The Pantry shelf is where spare jars go."
MONDAY, THURSDAY = "2026-09-07", "2026-09-10"


def _app(tmp_path: Path, fake: FakeOpenAIServer, clock: FakeClock, **switches: Any) -> FastAPI:
    values: dict[str, Any] = {"extraction_intent": True, **switches}
    return create_app(tier2_settings(tmp_path, fake, **values), clock=clock)


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = _app(tmp_path, fake_openai, clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def said(
    statement: str, value: str, *, duration: str = "observed", key: str = "content"
) -> dict[str, Any]:
    return {
        "subject": SHELF,
        "statement": statement,
        "attribute": key,
        "value": value,
        "cardinality": "one",
        "quote": statement.rstrip("."),
        "modality": "fact",
        "duration": duration,
    }


def reply(*facts: dict[str, Any]) -> Reply:
    return chat_json(
        {
            "nodes": [{"name": SHELF, "kind": "object"}],
            "facts": list(facts),
            "edges": [],
            "frame": "sincere",
            "frame_holds": False,
            "frame_ends": False,
        }
    )


def turn(
    client: TestClient,
    app: FastAPI,
    fake: FakeOpenAIServer,
    *replies: Reply,
    day: str = MONDAY,
    user: str | None = None,
    tool: str | None = None,
    web: str | None = None,
) -> dict[str, Any]:
    """One turn whose extraction answers ``replies[0]`` (the rest: later calls);
    the turn's words are the facts' quotes, in the owner's part unless a
    ``tool`` or ``web`` part carries them."""
    fake.script(chat=list(replies))
    external = [{"kind": "tool", "text": tool, "ref": "ls"}] if tool else []
    external += [{"kind": "web", "text": web, "ref": "https://example.com/a"}] if web else []
    post_turn(
        client,
        user if user is not None else "Look.",
        occurred_at=f"{day}T10:00:00Z",
        external=external,
    )
    jobs = run_all(client, app)["jobs"]
    return dict(next(j["result"] for j in jobs if j["kind"] == "turn"))


def facts_of(app: FastAPI) -> list[Fact]:
    store: Store = app.state.store
    rows = store.execute("SELECT id FROM facts ORDER BY created_at, id").fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def live(facts: list[Fact]) -> list[str]:
    return [f.statement for f in facts if f.valid_to is None and f.superseded_by is None]


# ------------------------------------------------------------------ who says it was observed


def test_an_observed_fact_is_memory_flagged_and_dated_by_its_source(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    result = turn(client, app, fake_openai, reply(said(EMPTY, "empty")), user=EMPTY)
    (fact,) = facts_of(app)
    assert fact.statement == EMPTY and "observed" in fact.flags
    assert fact.observed_at == MONDAY
    assert result["facts"]["observed"] == 1 and result["facts"]["observed_by_tool"] == 0
    assert result["intent"]["session"] == {"task": 0, "unsure": 0}  # not session state


def test_the_intent_prompt_offers_the_observed_duration(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    turn(client, app, fake_openai, reply(said(EMPTY, "empty")), user=EMPTY)
    system = fake_openai.chat_calls[0].body["messages"][0]["content"]
    assert '"observed"' in system


def test_a_lasting_fact_from_a_tool_part_is_observed_and_from_a_page_is_not(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A tool's output is a snapshot; a page follows the model's reading."""
    result = turn(
        client,
        app,
        fake_openai,
        reply(
            said(THREE, "3 jars", duration="lasting"),
            said(SPARE, "spare jars", duration="lasting", key="purpose"),
        ),
        tool=THREE,
        web=SPARE,
    )
    by_statement = {f.statement: f for f in facts_of(app)}
    assert "observed" in by_statement[THREE].flags
    assert by_statement[THREE].observed_at == MONDAY
    assert "observed" not in by_statement[SPARE].flags
    assert by_statement[SPARE].observed_at is None
    assert result["facts"]["observed"] == 1 and result["facts"]["observed_by_tool"] == 1


def test_a_task_fact_from_a_tool_part_stays_session_state(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    result = turn(
        client, app, fake_openai, reply(said(THREE, "3 jars", duration="task")), tool=THREE
    )
    assert facts_of(app) == []
    assert result["intent"]["session"]["task"] == 1


def test_with_the_intent_reading_off_the_tool_rule_alone_applies(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = _app(tmp_path, fake_openai, clock, extraction_intent=False)
    try:
        with TestClient(application, headers=auth("all")) as client:
            turn(
                client,
                application,
                fake_openai,
                reply(said(THREE, "3 jars"), said(SPARE, "spare", key="purpose")),
                user=SPARE,
                tool=THREE,
            )
            by_statement = {f.statement: f for f in facts_of(application)}
            assert "observed" in by_statement[THREE].flags
            assert "observed" not in by_statement[SPARE].flags  # its duration is not read
    finally:
        application.state.store.close()


# ------------------------------------------------------------------ seen again, replaced


def test_a_newer_observation_of_the_same_key_closes_the_older_without_a_question(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """ "Empty" a week ago, "the spices" today, under ``content``."""
    turn(client, app, fake_openai, reply(said(EMPTY, "empty")), user=EMPTY)
    result = turn(
        client, app, fake_openai, reply(said(SPICES, "the spices")), day=THURSDAY, user=SPICES
    )
    assert len(fake_openai.chat_calls) == 2  # two extractions, no replacement question
    assert result["facts"]["superseded"] == 1 and result["facts"]["conflict"] == 0
    old, new = facts_of(app)
    assert old.superseded_by == new.id and live([old, new]) == [SPICES]


def test_two_values_observed_in_one_source_both_stay(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    result = turn(
        client,
        app,
        fake_openai,
        reply(said(THREE, "3 jars"), said(SPICES, "the spices")),
        user=f"{THREE} {SPICES}",
    )
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 0
    assert sorted(live(facts_of(app))) == sorted([THREE, SPICES])


def test_an_older_observation_arriving_late_is_kept_in_conflict(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    turn(client, app, fake_openai, reply(said(SPICES, "the spices")), day=THURSDAY, user=SPICES)
    result = turn(client, app, fake_openai, reply(said(EMPTY, "empty")), day=MONDAY, user=EMPTY)
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    newer, older = facts_of(app)
    assert newer.valid_to is None and "conflict" in older.flags


def test_an_observation_never_closes_a_lasting_fact_on_its_own(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Against a lasting fact the ordinary rules apply: a free-form key goes to
    the replacement question, and without an answer both stay in conflict."""
    turn(
        client,
        app,
        fake_openai,
        reply(said(SPARE, "spare", duration="lasting")),
        user=SPARE,
    )
    result = turn(
        client, app, fake_openai, reply(said(SPICES, "the spices")), day=THURSDAY, user=SPICES
    )
    assert len(fake_openai.chat_calls) == 3  # the replacement question was asked
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert sorted(live(facts_of(app))) == sorted([SPARE, SPICES])


def test_a_lasting_fact_never_closes_observations_on_their_own(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    turn(client, app, fake_openai, reply(said(SPICES, "the spices")), user=SPICES)
    result = turn(
        client,
        app,
        fake_openai,
        reply(said(SPARE, "spare", duration="lasting")),
        day=THURSDAY,
        user=SPARE,
    )
    assert len(fake_openai.chat_calls) == 3
    assert result["facts"]["superseded"] == 0
    assert sorted(live(facts_of(app))) == sorted([SPARE, SPICES])


def test_a_tool_observation_never_closes_what_the_owner_observed(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """P6 before the observation rule: external content never closes a fact
    that did not come from external content."""
    turn(client, app, fake_openai, reply(said(EMPTY, "empty")), user=EMPTY)
    result = turn(
        client, app, fake_openai, reply(said(SPICES, "the spices")), day=THURSDAY, tool=SPICES
    )
    assert result["facts"]["superseded"] == 0 and result["facts"]["conflict"] == 1
    assert sorted(live(facts_of(app))) == sorted([EMPTY, SPICES])


def test_a_tool_observation_closes_an_older_tool_observation(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    turn(client, app, fake_openai, reply(said(EMPTY, "empty", duration="lasting")), tool=EMPTY)
    result = turn(
        client,
        app,
        fake_openai,
        reply(said(SPICES, "the spices", duration="lasting")),
        day=THURSDAY,
        tool=SPICES,
    )
    assert result["facts"]["superseded"] == 1
    assert live(facts_of(app)) == [SPICES]


def test_the_same_value_observed_again_moves_its_date_and_writes_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    turn(client, app, fake_openai, reply(said(SPICES, "the spices")), user=SPICES)
    (held,) = facts_of(app)
    result = turn(
        client, app, fake_openai, reply(said(SPICES, "the spices")), day=THURSDAY, user=SPICES
    )
    (again,) = facts_of(app)
    assert again.id == held.id and again.observed_at == THURSDAY and again.rev > held.rev
    assert result["facts"]["written"] == 0 and result["facts"]["observed_again"] == 1


def test_the_same_value_in_other_words_observed_again_moves_its_date(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    turn(client, app, fake_openai, reply(said(SPICES, "the spices")), user=SPICES)
    other = "The spices sit on the Pantry shelf."
    result = turn(
        client, app, fake_openai, reply(said(other, "the spices")), day=THURSDAY, user=other
    )
    (held,) = facts_of(app)
    assert held.statement == SPICES and held.observed_at == THURSDAY
    assert result["facts"]["observed_again"] == 1


def test_the_owner_saying_it_again_leaves_the_date(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    turn(client, app, fake_openai, reply(said(SPICES, "the spices")), user=SPICES)
    result = turn(
        client,
        app,
        fake_openai,
        reply(said(SPICES, "the spices", duration="lasting")),
        day=THURSDAY,
        user=SPICES,
    )
    (held,) = facts_of(app)
    assert held.observed_at == MONDAY and result["facts"]["observed_again"] == 0


# ------------------------------------------------------------------ proposals keep it


def test_a_proposed_observation_keeps_its_flag_and_day_once_accepted(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    """A tool's output about the owner is only proposed (P6); accepted, it is
    still an observation of that day."""
    application = _app(tmp_path, fake_openai, clock, owners=((SCOPE, "Alice", ()),))
    disk = "Alice's disk is 90% full."
    fact = {
        "subject": "Alice",
        "statement": disk,
        "attribute": "disk",
        "value": "90% full",
        "quote": disk.rstrip("."),
        "modality": "fact",
        "duration": "lasting",
    }
    try:
        with TestClient(application, headers=auth("all")) as client:
            fake_openai.script(
                chat=[
                    chat_json(
                        {
                            "nodes": [],
                            "facts": [fact],
                            "edges": [],
                            "frame": "sincere",
                            "frame_holds": False,
                            "frame_ends": False,
                        }
                    )
                ]
            )
            post_turn(
                client,
                "Check my disk.",
                occurred_at=f"{MONDAY}T10:00:00Z",
                external=[{"kind": "tool", "text": disk, "ref": "df"}],
            )
            run_all(client, application)
            store: Store = application.state.store
            (row,) = store.execute("SELECT id FROM proposals WHERE kind = 'fact'").fetchall()
            proposal = store.get_proposal(row["id"])
            assert "observed" in proposal.payload["flags"]
            assert proposal.payload["observed_at"] == MONDAY
            decided = client.post(f"/v1/proposals/{row['id']}", json={"decision": "accept"})
            assert decided.status_code == 200, decided.text
            (accepted,) = facts_of(application)
            assert "observed" in accepted.flags and accepted.observed_at == MONDAY
    finally:
        application.state.store.close()


# ------------------------------------------------------------------ handles and ranking


def _observed(store: Store, node_id: str, statement: str, day: str | None) -> Fact:
    return store.add_fact(
        scopes=[SCOPE],
        node_id=node_id,
        statement=statement,
        origin="extracted",
        attribute=None,
        value=None,
        flags=["observed"] if day else [],
        observed_at=day,
    )


def test_a_handle_says_observed_and_counts_its_age_from_the_observation(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    store: Store = app.state.store
    node = store.create_node(scopes=[SCOPE], name=SHELF, kind="object")
    fact = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement=EMPTY,
        origin="extracted",
        valid_from="2026-01-01",
        flags=["observed"],
        observed_at=(clock() - timedelta(days=7)).date().isoformat(),
    )
    items = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "Pantry empty", "budget_tokens": 2000}
    ).json()["items"]
    handle = next(i for i in items if i["id"] == fact.id)
    assert "observed" in handle["flags"] and handle["age_days"] == 7
    expanded = client.post(
        "/v1/expand", json={"scope": SCOPE, "ids": [fact.id], "level": 1, "budget_tokens": 2000}
    ).json()["items"][0]
    assert expanded["observed_at"] == fact.observed_at
    viewed = client.get(f"/v1/nodes/{node.id}", params={"scope": SCOPE, "budget_tokens": 2000})
    assert viewed.json()["facts"][0]["observed_at"] == fact.observed_at


def _ranking(app: FastAPI, client: TestClient, clock: FakeClock) -> tuple[list[str], Fact, Fact]:
    """Two facts the query matches alike: an old observation written first (it
    wins the tie on its id) and a lasting one."""
    store: Store = app.state.store
    node = store.create_node(scopes=[SCOPE], name=SHELF, kind="object")
    old = (clock() - timedelta(days=14)).date().isoformat()
    seen = _observed(store, node.id, "The Pantry shelf status is green.", old)
    kept = _observed(store, node.id, "The Pantry shelf status is amber.", None)
    items = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "Pantry status", "budget_tokens": 2000}
    ).json()["items"]
    return [i["id"] for i in items if i["kind"] == "fact"], seen, kept


def test_an_aged_observation_ranks_below_a_lasting_fact_it_tied_with(
    client: TestClient, app: FastAPI, clock: FakeClock
) -> None:
    order, seen, kept = _ranking(app, client, clock)
    assert order.index(kept.id) < order.index(seen.id)


def test_a_half_life_of_zero_turns_the_decay_off(
    tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock
) -> None:
    application = _app(tmp_path, fake_openai, clock, observed_half_life_days=0.0)
    try:
        with TestClient(application, headers=auth("all")) as client:
            order, seen, kept = _ranking(application, client, clock)
            assert order.index(seen.id) < order.index(kept.id)  # the tie, as without it
    finally:
        application.state.store.close()


# ------------------------------------------------------------- a newer observation in review


def _in_review(
    store: Store,
    *,
    value: str = "up",
    flags: tuple[str, ...] = ("observed",),
    proposal_flags: tuple[str, ...] = ("observed",),
    proposal_day: str = THURSDAY,
    scopes: tuple[str, ...] = (SCOPE,),
) -> tuple[Fact, str]:
    """An observation "the relay is down" made on Monday, and an observation of
    the same attribute made on ``proposal_day`` that is only a fact proposal."""
    node = store.create_node(scopes=[SCOPE], name="Relay", kind="object")
    fact = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement="The relay is down.",
        origin="extracted",
        attribute="state",
        value="down",
        flags=list(flags),
        observed_at=MONDAY if "observed" in flags else None,
    )
    proposal = store.add_proposal(
        scopes=scopes,
        kind="fact",
        node_id=node.id,
        value=f"The relay is {value}.",
        payload={
            "attribute": "state",
            "value": value,
            "flags": list(proposal_flags),
            "observed_at": proposal_day if "observed" in proposal_flags else None,
        },
    )
    return fact, proposal.id


def _flags(client: TestClient, fact_id: str) -> tuple[list[str], list[str], list[str]]:
    """The fact's flags in a recall handle, in its expansion, and in its node's
    expansion (an absent key: no flag)."""
    items = client.post(
        "/v1/recall", json={"scope": SCOPE, "query": "relay down", "budget_tokens": 2000}
    ).json()["items"]
    handle = next(i for i in items if i["id"] == fact_id)

    def expand(record_id: str) -> dict[str, Any]:
        body = {"scope": SCOPE, "ids": [record_id], "level": 1, "budget_tokens": 4000}
        item: dict[str, Any] = client.post("/v1/expand", json=body).json()["items"][0]
        return item

    fact = expand(fact_id)
    node = expand(fact["node_id"])
    listed = next(f.get("flags", []) for f in node["facts"] if f["id"] == fact_id)
    return handle["flags"], fact["flags"], listed


def test_an_observation_with_a_newer_one_in_review_is_flagged_until_it_is_decided(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    fact, proposal = _in_review(store)
    handle, expanded, listed = _flags(client, fact.id)
    assert "newer_pending" in handle and "newer_pending" in expanded
    assert "newer_pending" in listed  # the node's expansion lists it flagged
    assert "newer_pending" not in store.get_fact(fact.id).flags  # computed, never stored
    store.decide_proposal(proposal, accept=False)
    handle, expanded, listed = _flags(client, fact.id)
    assert "newer_pending" not in [*handle, *expanded, *listed]


@pytest.mark.parametrize(
    "case",
    [
        "lasting",
        "same value",
        "older observation proposed",
        "same day",
        "proposal not an observation",
        "invisible proposal",
        "seen again later",
    ],
)
def test_no_newer_observation_in_review_no_flag(
    client: TestClient, app: FastAPI, case: str
) -> None:
    """A lasting fact; the same value proposed; an observation proposed from an
    earlier day or the same day; a proposal that is no observation (a role, a
    hypothesis); a proposal the reader cannot see (P9); a fact seen again after
    the proposed observation: no flag."""
    store: Store = app.state.store
    fact, _ = _in_review(
        store,
        value="down" if case == "same value" else "up",
        flags=() if case == "lasting" else ("observed",),
        proposal_flags=() if case == "proposal not an observation" else ("observed",),
        proposal_day={"older observation proposed": "2026-09-01", "same day": MONDAY}.get(
            case, THURSDAY
        ),
        scopes=("user:bob",) if case == "invisible proposal" else (SCOPE,),
    )
    if case == "seen again later":
        store.observe_again(fact.id, "2026-09-12")
    handle, expanded, listed = _flags(client, fact.id)
    assert "newer_pending" not in [*handle, *expanded, *listed]


# ------------------------------------------------------- a newer observation under another key


def _two_keys(
    store: Store,
    *,
    day: str = THURSDAY,
    flags: tuple[str, ...] = ("observed",),
    scopes: tuple[str, ...] = (SCOPE,),
    closed: bool = False,
) -> tuple[Fact, Fact]:
    """The relay observed down on Monday (``state``), then observed holding
    four jobs on ``day`` under another key (``queue``)."""
    node = store.create_node(scopes=[SCOPE], name="Relay", kind="object")
    older = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement="The relay is down.",
        origin="extracted",
        attribute="state",
        value="down",
        flags=["observed"],
        observed_at=MONDAY,
    )
    newer = store.add_fact(
        scopes=list(scopes),
        node_id=node.id,
        statement="The relay holds four jobs.",
        origin="extracted",
        attribute="queue",
        value="four jobs",
        flags=list(flags),
        observed_at=day if "observed" in flags else None,
    )
    if closed:
        store.close_fact(newer.id)
    return older, newer


def test_an_observation_with_a_newer_one_under_another_key_is_flagged(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    older, newer = _two_keys(store)
    handle, expanded, listed = _flags(client, older.id)
    assert "newer_observed" in handle and "newer_observed" in expanded
    assert "newer_observed" in listed
    assert "newer_observed" not in store.get_fact(older.id).flags  # computed, never stored
    body = {"scope": SCOPE, "ids": [newer.id], "level": 1, "budget_tokens": 2000}
    latest = client.post("/v1/expand", json=body).json()["items"][0]
    assert "newer_observed" not in latest["flags"]


@pytest.mark.parametrize("case", ["same day", "invisible", "lasting", "closed"])
def test_no_newer_observation_under_another_key_no_flag(
    client: TestClient, app: FastAPI, case: str
) -> None:
    """Observed the same day; a newer observation the reader cannot see (P9);
    a newer fact that is no observation; a newer observation since closed."""
    store: Store = app.state.store
    older, _ = _two_keys(
        store,
        day=MONDAY if case == "same day" else THURSDAY,
        flags=() if case == "lasting" else ("observed",),
        scopes=("user:bob",) if case == "invisible" else (SCOPE,),
        closed=case == "closed",
    )
    handle, expanded, listed = _flags(client, older.id)
    assert "newer_observed" not in [*handle, *expanded, *listed]


# ------------------------------------------------------------ a conflict seen from both sides


def _contradicted(
    store: Store,
    *,
    scopes: tuple[str, ...] = (SCOPE,),
    older_origin: str = "extracted",
    newer_origin: str = "extracted",
    newer_provenance: str = "owner",
) -> tuple[Fact, Fact]:
    """The relay observed down, then a later lasting fact under the same key
    written in conflict with it."""
    node = store.create_node(scopes=[SCOPE], name="Relay", kind="object")
    older = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement="The relay is down.",
        origin=older_origin,
        attribute="state",
        value="down",
        flags=["observed"],
        observed_at=MONDAY,
        provenance="owner",
    )
    newer = store.add_fact(
        scopes=list(scopes),
        node_id=node.id,
        statement="The relay is retired.",
        origin=newer_origin,
        provenance=newer_provenance,
        attribute="state",
        value="retired",
        flags=["conflict"],
        conflicts_with=older.id,
    )
    return older, newer


def test_a_fact_another_live_fact_contradicts_shows_the_conflict_too(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    older, _ = _contradicted(store)
    handle, expanded, listed = _flags(client, older.id)
    assert "conflict" in handle and "conflict" in expanded and "conflict" in listed
    assert "conflict" not in store.get_fact(older.id).flags  # computed, never stored


def test_an_explicit_fact_contradicted_by_an_explicit_one_shows_the_conflict(
    client: TestClient, app: FastAPI
) -> None:
    store: Store = app.state.store
    older, _ = _contradicted(store, older_origin="explicit", newer_origin="explicit")
    assert "conflict" in _flags(client, older.id)[0]


@pytest.mark.parametrize(
    "case",
    [
        "closed",
        "archived",
        "invisible",
        "resolved",
        "extracted against explicit",
        "external against the owner",
    ],
)
def test_no_open_conflict_no_conflict_flag_on_the_older_fact(
    client: TestClient, app: FastAPI, case: str
) -> None:
    """The newer fact closed, archived, out of the reader's view (P9), no
    longer flagged ``conflict``, or of less authority (P6: an extraction
    against an explicit fact, external content against the owner's words):
    nothing shows on the older one, so a page cannot cast doubt on it."""
    store: Store = app.state.store
    older, newer = _contradicted(
        store,
        scopes=("user:bob",) if case == "invisible" else (SCOPE,),
        older_origin="explicit" if case == "extracted against explicit" else "extracted",
        newer_provenance="external" if case == "external against the owner" else "owner",
    )
    if case == "closed":
        store.close_fact(newer.id)
    elif case == "archived":
        store.archive(newer.id)
    elif case == "resolved":
        store.set_fact_flags(newer.id, [], conflicts_with=older.id)
    handle, expanded, listed = _flags(client, older.id)
    assert "conflict" not in [*handle, *expanded, *listed]
