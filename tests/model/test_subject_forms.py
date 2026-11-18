"""One subject in two forms (SPEC 5.1) and a supersession that reaches a known
duplicate (SPEC 5.4), against the scripted fake model.

A file named by its slug and by its title are candidates of each other: the
adjudication attaches the second form, or a new node is paired with the first
by a merge proposal. A correction that names one form then flags the stale
value on the other: it never closes it, and nothing is merged at write time."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.config import Settings
from iifa.scopes import Visibility
from iifa.store.models import Fact
from iifa.store.repo import Store
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)


@pytest.fixture
def settings(tmp_path: Path, fake_openai: FakeOpenAIServer) -> Settings:
    """The corrections here change a free-form ``content`` within one language:
    with the replacement question off, such a change supersedes without a
    question (SPEC 5.4), so each turn costs the calls these tests count."""
    return tier2_settings(tmp_path, fake_openai, replacement_question=False)


NEW = chat_json({"decisions": [{"mention": 0, "choice": "new"}]})


def said(
    name: str, value: str, *, kind: str = "object", key: str = "content", cardinality: str = "one"
) -> tuple[str, Any]:
    """A turn saying ``name`` shows ``value``, and the extraction answer for it."""
    statement = f"{name} montre {value}."
    return statement, answer(
        nodes=[{"name": name, "kind": kind}],
        facts=[
            {
                "subject": name,
                "statement": statement,
                "attribute": key,
                "value": value,
                "cardinality": cardinality,  # a free-form key: one value unless told (SPEC 5.4)
                "quote": statement.rstrip("."),
            }
        ],
    )


def tell(
    client: TestClient,
    app: FastAPI,
    fake: FakeOpenAIServer,
    turn_said: tuple[str, Any],
    *more: Any,
    scope: str = SCOPE,
) -> dict[str, Any]:
    """One turn and its extraction answer, then ``more`` (the adjudication's
    answers), each consumed by one call; the turn's extraction report."""
    statement, reply = turn_said
    before = len(fake.chat_calls)
    fake.script(chat=[reply, *more])
    post_turn(client, statement, scope=scope)
    report = run_all(client, app)
    assert len(fake.chat_calls) - before == 1 + len(more)
    turn: dict[str, Any] = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    return turn


def live(store: Store, node_id: str) -> list[Fact]:
    return store.facts([node_id], VIEW)


def node_id(store: Store, name: str) -> str:
    node = store.find_node(name, VIEW)
    assert node is not None, name
    return node.id


PAIRS = [
    ("le jardin du nord", "Résumé jardin du nord"),  # lower case against a title
    ("le_jardin_du_nord.pdf", "Résumé : Le jardin du nord"),  # alike by no letter score
]


@pytest.mark.parametrize(("first", "second"), PAIRS)
@pytest.mark.parametrize("corrected", ["first", "second"])
def test_a_correction_on_one_form_flags_the_stale_value_on_the_other(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    first: str,
    second: str,
    corrected: str,
) -> None:
    store = app.state.store
    tell(client, app, fake_openai, said(first, "un bassin"))
    turn = tell(client, app, fake_openai, said(second, "un bassin"), NEW)
    # the second form reached the adjudication with the first as its candidate
    asked = turn["adjudications"][0]
    assert (asked["mention"], asked["chosen"]) == (second, None)
    assert [c["name"] for c in asked["candidates"]] == [first]
    # the model said "new": two nodes, paired by a merge proposal, nothing merged
    a, b = node_id(store, first), node_id(store, second)
    assert a != b
    assert store.merge_candidates(b, VIEW) == [a]
    assert turn["duplicates"] == {"merge_proposed": 1, "flagged": 0}
    queue = client.get("/v1/review", params={"scope": SCOPE}).json()["items"]
    assert [(i["kind"], i["node_id"], i["value"]) for i in queue] == [("merge", b, a)]

    # a correction that names only one of the two forms
    target, corrected_id, other = (first, a, b) if corrected == "first" else (second, b, a)
    (stale,) = live(store, other)
    turn = tell(client, app, fake_openai, said(target, "une fontaine"))
    assert turn["facts"]["superseded"] == 1
    assert turn["duplicates"] == {"merge_proposed": 0, "flagged": 1}
    (current,) = live(store, corrected_id)
    assert current.value == "une fontaine"
    # the known duplicate keeps the old value live (never closed), in conflict now
    flagged = store.get_fact(stale.id)
    assert flagged.valid_to is None and flagged.superseded_by is None
    assert "conflict" in flagged.flags and flagged.conflicts_with == current.id
    # what a reader sees: the stale value is no longer silently current
    (shown,) = client.post("/v1/expand", json={"scope": SCOPE, "ids": [stale.id]}).json()["items"]
    assert "conflict" in shown["flags"] and shown["conflicts_with"] == current.id
    # nothing was merged: both nodes are live, the proposal still pending
    assert store.merge_candidates(a, VIEW) == [b]


def test_the_adjudication_may_attach_the_second_form(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = app.state.store
    tell(client, app, fake_openai, said("le jardin du nord", "un bassin"))
    north = node_id(store, "le jardin du nord")
    attach = chat_json({"decisions": [{"mention": 0, "choice": north}]})
    turn = tell(
        client, app, fake_openai, said("Résumé jardin du nord", "un dessin", key="format"), attach
    )
    assert turn["adjudications"][0]["chosen"] == "le jardin du nord"
    assert turn["nodes"]["created"] == 0
    assert {f.value for f in live(store, north)} == {"un bassin", "un dessin"}
    assert store.merge_candidates(north, VIEW) == []


def test_two_titles_in_two_forms_are_adjudicated_but_never_paired(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Two titles, "Photo jardin du nord" and "Résumé jardin du nord", may be two subjects:
    candidates for the adjudication, but neither is written as a slug, so a
    "new" leaves no merge proposal and a correction on one leaves the other."""
    store = app.state.store
    tell(client, app, fake_openai, said("Photo jardin du nord", "un bassin"))
    turn = tell(client, app, fake_openai, said("Résumé jardin du nord", "un bassin"), NEW)
    assert turn["mentions_adjudicated"] >= 1
    photo, resume = node_id(store, "Photo jardin du nord"), node_id(store, "Résumé jardin du nord")
    assert store.merge_candidates(resume, VIEW) == []
    turn = tell(client, app, fake_openai, said("Résumé jardin du nord", "une fontaine"))
    assert turn["duplicates"] == {"merge_proposed": 0, "flagged": 0}
    (kept,) = live(store, photo)
    assert kept.value == "un bassin" and "conflict" not in kept.flags


def test_words_in_common_neither_attach_nor_pair_nor_flag(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = app.state.store
    tell(client, app, fake_openai, said("projet-nord-bleu", "un bassin"))
    turn = tell(client, app, fake_openai, said("Projet nord vert", "un bassin"))
    assert turn["mentions_adjudicated"] == 0 and turn["nodes"]["created"] == 1
    blue, green = node_id(store, "projet-nord-bleu"), node_id(store, "Projet nord vert")
    assert store.merge_candidates(green, VIEW) == []
    tell(client, app, fake_openai, said("Projet nord vert", "une forêt"))
    (kept,) = live(store, blue)
    assert kept.value == "un bassin" and "conflict" not in kept.flags


def test_near_names_are_never_paired_nor_flagged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Karim and Karima: a candidate for the adjudication (SPEC 5.1), never a
    merge proposal at write time, and a correction on one leaves the other."""
    store = app.state.store
    tell(client, app, fake_openai, said("Karim", "Lyon", kind="person", key="home"))
    turn = tell(client, app, fake_openai, said("Karima", "Lyon", kind="person", key="home"), NEW)
    assert turn["adjudications"][0]["chosen"] is None
    karim, karima = node_id(store, "Karim"), node_id(store, "Karima")
    assert store.merge_candidates(karima, VIEW) == []
    tell(client, app, fake_openai, said("Karim", "Paris", kind="person", key="home"))
    (kept,) = live(store, karima)
    assert kept.value == "Lyon" and "conflict" not in kept.flags
    assert {f.value for f in live(store, karim)} == {"Paris"}


def test_a_rejected_merge_is_no_known_duplicate(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A reviewer who rejected the pair said they are two subjects: a
    correction on one leaves the other as it is."""
    store = app.state.store
    tell(client, app, fake_openai, said("le jardin du nord", "un bassin"))
    tell(client, app, fake_openai, said("Résumé jardin du nord", "un bassin"), NEW)
    a, b = node_id(store, "le jardin du nord"), node_id(store, "Résumé jardin du nord")
    (proposal,) = store.pending_proposals(VIEW, kind="merge")
    decided = client.post(f"/v1/review/{proposal.id}", json={"decision": "reject"})
    assert decided.status_code == 200, decided.text
    turn = tell(client, app, fake_openai, said("Résumé jardin du nord", "une fontaine"))
    assert turn["duplicates"]["flagged"] == 0
    (kept,) = live(store, a)
    assert kept.value == "un bassin" and "conflict" not in kept.flags
    assert {f.value for f in live(store, b)} == {"une fontaine"}


def test_a_merge_proposal_never_widens_visibility(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Dream rule 12: a write in a child scope sees the parent's node and may
    adjudicate against it, but a new node there is never paired with it."""
    store = app.state.store
    tell(client, app, fake_openai, said("le jardin du nord", "un bassin"))
    child = f"{SCOPE}/project:garden"
    turn = tell(
        client, app, fake_openai, said("Résumé jardin du nord", "un bassin"), NEW, scope=child
    )
    assert turn["adjudications"][0]["candidates"][0]["name"] == "le jardin du nord"
    assert turn["nodes"]["created"] == 1
    assert turn["duplicates"]["merge_proposed"] == 0
    assert store.pending_proposals(Visibility.of_scope(child), kind="merge") == []


def facts_of(name: str, *facts: tuple[str, str, str]) -> tuple[str, Any]:
    """A turn stating several ``(key, value, statement)`` of ``name``."""
    text = " ".join(statement for _, _, statement in facts)
    return text, answer(
        nodes=[{"name": name, "kind": "object"}],
        facts=[
            {
                "subject": name,
                "statement": statement,
                "attribute": key,
                "value": value,
                "cardinality": "one",  # a free-form key: one current value (SPEC 5.4)
                "quote": statement.rstrip("."),
            }
            for key, value, statement in facts
        ],
    )


def test_only_the_closed_value_of_the_same_key_is_flagged(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The duplicate's other keys stay as they are; a value in another form
    ("March 2026" for "mars 2026") is the closed value too; a fact already in
    conflict is left as it is."""
    store = app.state.store
    slug, title = "le-jardin-du-nord.pdf", "Résumé jardin du nord"
    tell(
        client,
        app,
        fake_openai,
        facts_of(
            slug,
            ("content", "un bassin", f"{slug} montre un bassin."),
            ("date.release", "mars 2026", f"{slug} sort en mars 2026."),
        ),
    )
    turn = tell(
        client,
        app,
        fake_openai,
        facts_of(
            title,
            ("content", "un bassin", f"{title} montre un bassin."),
            ("date.release", "March 2026", f"{title} is released in March 2026."),
            ("format", "un dessin", f"{title} est un dessin."),
        ),
        NEW,
    )
    assert turn["duplicates"]["merge_proposed"] == 1
    a, b = node_id(store, slug), node_id(store, title)
    turn = tell(
        client,
        app,
        fake_openai,
        facts_of(
            slug,
            ("content", "une fontaine", f"{slug} montre une fontaine."),
            ("date.release", "avril 2026", f"{slug} sort en avril 2026."),
        ),
    )
    assert turn["duplicates"]["flagged"] == 2
    flagged = {f.attribute: f for f in live(store, b)}
    assert set(flagged) == {"content", "date.release", "format"}
    assert "conflict" in flagged["content"].flags and "conflict" in flagged["date.release"].flags
    assert "conflict" not in flagged["format"].flags
    first_flag = flagged["content"].conflicts_with
    # a second correction finds the stale fact already in conflict: left as it is
    turn = tell(
        client,
        app,
        fake_openai,
        facts_of(slug, ("content", "un motif", f"{slug} montre un motif.")),
    )
    assert turn["facts"]["superseded"] == 1 and turn["duplicates"]["flagged"] == 0
    assert store.get_fact(flagged["content"].id).conflicts_with == first_flag
    assert {f.value for f in live(store, a)} == {"un motif", "avril 2026"}


def test_one_source_naming_both_forms_pairs_them(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Neither form known yet: the answer proposes both nodes at once, so no
    adjudication sees them together; the two nodes it creates are paired."""
    store = app.state.store
    slug, title = "le-jardin-du-nord.pdf", "Résumé jardin du nord"
    text = f"{slug} montre un bassin. {title} fait trois pages."
    reply = answer(
        nodes=[{"name": slug, "kind": "object"}, {"name": title, "kind": "object"}],
        facts=[
            {"subject": slug, "statement": f"{slug} montre un bassin.", "quote": f"{slug} montre"},
            {"subject": title, "statement": f"{title} fait trois pages.", "quote": title},
        ],
    )
    turn = tell(client, app, fake_openai, (text, reply))
    assert turn["nodes"]["created"] == 2 and turn["duplicates"]["merge_proposed"] == 1
    a, b = node_id(store, slug), node_id(store, title)
    assert store.merge_candidates(a, VIEW) == [b]


def test_a_new_value_on_the_form_without_it_flags_the_other(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The correction lands on the node that never held the key: nothing closes
    there, yet the duplicate's other value is no longer silently current."""
    store = app.state.store
    tell(client, app, fake_openai, said("le jardin du nord", "un bassin"))
    turn = tell(
        client, app, fake_openai, said("Résumé jardin du nord", "un dessin", key="format"), NEW
    )
    assert turn["duplicates"]["merge_proposed"] == 1
    a = node_id(store, "le jardin du nord")
    turn = tell(client, app, fake_openai, said("Résumé jardin du nord", "une fontaine"))
    assert turn["facts"]["superseded"] == 0 and turn["duplicates"]["flagged"] == 1
    (stale,) = live(store, a)
    assert stale.value == "un bassin" and "conflict" in stale.flags and stale.valid_to is None
    # the same value on the duplicate is no contradiction
    turn = tell(client, app, fake_openai, said("le jardin du nord", "une fontaine"))
    assert turn["duplicates"]["flagged"] == 0


def test_an_element_of_a_set_on_one_form_flags_nothing_on_the_other(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """SPEC 5.4 (sets): an element adds to a set and closes nothing, so it
    contradicts nothing on a known duplicate either."""
    store = app.state.store
    tell(
        client,
        app,
        fake_openai,
        said("le jardin du nord", "une liste d'oiseaux", key="questions", cardinality="many"),
    )
    turn = tell(
        client, app, fake_openai, said("Résumé jardin du nord", "un dessin", key="format"), NEW
    )
    assert turn["duplicates"]["merge_proposed"] == 1
    a = node_id(store, "le jardin du nord")
    turn = tell(
        client,
        app,
        fake_openai,
        said("Résumé jardin du nord", "une carte du quartier", key="questions", cardinality="many"),
    )
    assert turn["duplicates"]["flagged"] == 0
    (held,) = live(store, a)
    assert held.value == "une liste d'oiseaux" and "conflict" not in held.flags
