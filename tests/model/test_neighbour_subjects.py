"""Neighbouring subjects: a turn describing two subjects side by side.
A fact whose quote names the other subject and not its own, or whose value
names both, is not attached on its own: it becomes a fact proposal, counted
``facts.neighbour``. Two subjects that each get their own facts lose nothing."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.ingest.extract import _distinctive, quote_names_other, value_merges
from iifa.scopes import Visibility
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)
ZIP, PDF = "backup.zip", "report.pdf"
FILES = [{"name": ZIP, "kind": "object"}, {"name": PDF, "kind": "object"}]
MIXED = "a PDF of twelve pages and a zip with three folders"
SIDE_BY_SIDE = f"I received {ZIP} and {PDF}. Both are here: {MIXED}."
APART = f"{ZIP} is a zip with three folders. {PDF} is a PDF of twelve pages."


def fact(subject: str, statement: str, value: str | None, quote: str, **extra: Any) -> Any:
    return {
        "subject": subject,
        "statement": statement,
        "attribute": "content" if value is not None else None,
        "value": value,
        "quote": quote,
        **extra,
    }


ZIP_FACT = fact(
    ZIP,
    f"{ZIP} is a zip with three folders.",
    "a zip with three folders",
    f"{ZIP} is a zip with three folders",
)
PDF_FACT = fact(
    PDF,
    f"{PDF} is a PDF of twelve pages.",
    "a PDF of twelve pages",
    f"{PDF} is a PDF of twelve pages",
)


def ingest(
    client: TestClient, app: FastAPI, fake: FakeOpenAIServer, text: str, reply: Any
) -> dict[str, Any]:
    fake.script(chat=[reply])
    post_turn(client, text)
    jobs = run_all(client, app)["jobs"]
    result: dict[str, Any] = next(j["result"] for j in jobs if j["kind"] == "turn")
    return result


def statements(app: FastAPI, name: str) -> list[str]:
    node = app.state.store.find_node(name, VIEW)
    if node is None:
        return []
    return sorted(f.statement for f in app.state.store.facts([node.id], VIEW))


def proposals(app: FastAPI) -> list[tuple[str, str]]:
    rows = app.state.store.execute(
        "SELECT n.name, p.value FROM proposals p JOIN nodes n ON n.id = p.node_id"
        " WHERE p.kind = 'fact' AND p.status = 'pending' ORDER BY p.value"
    ).fetchall()
    return [(r["name"], r["value"]) for r in rows]


# ------------------------------------------------------------------ the shape


def test_a_value_describing_both_files_is_not_attached_to_the_zip(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    wrong = fact(ZIP, f"{ZIP} contains {MIXED}.", MIXED, f"Both are here: {MIXED}")
    result = ingest(client, app, fake_openai, SIDE_BY_SIDE, answer(nodes=FILES, facts=[wrong]))
    assert statements(app, ZIP) == []
    assert proposals(app) == [(ZIP, f"{ZIP} contains {MIXED}.")]
    assert result["facts"]["neighbour"] == 1 and result["facts"]["proposed"] == 1
    assert result["facts"]["written"] == 0


def test_the_pdf_sentence_attached_to_the_zip_is_caught_and_the_pdf_keeps_its_fact(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    wrong = fact(
        ZIP,
        f"{ZIP} is a PDF of twelve pages.",
        "a PDF of twelve pages",
        f"{PDF} is a PDF of twelve pages",
    )
    result = ingest(client, app, fake_openai, APART, answer(nodes=FILES, facts=[wrong, PDF_FACT]))
    assert statements(app, ZIP) == []
    assert statements(app, PDF) == [PDF_FACT["statement"]]
    assert proposals(app) == [(ZIP, wrong["statement"])]
    assert result["facts"]["neighbour"] == 1 and result["facts"]["written"] == 1


def test_the_other_name_given_as_subject_as_written_does_not_name_the_subject(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The model handing the other file's name as the source spelling of its
    subject still leaves the fact a proposal."""
    wrong = fact(
        ZIP,
        f"{ZIP} is a PDF of twelve pages.",
        "a PDF of twelve pages",
        f"{PDF} is a PDF of twelve pages",
        subject_as_written=PDF,
    )
    result = ingest(client, app, fake_openai, APART, answer(nodes=FILES, facts=[wrong]))
    assert statements(app, ZIP) == [] and result["facts"]["neighbour"] == 1


@pytest.mark.parametrize(
    "trick",
    [
        {"subject_as_written": "PDF"},  # a word of the other file's name
        {"subject_as_written": "clean"},  # a word of the quote, no name at all
    ],
)
def test_the_model_cannot_name_the_subject_into_the_quote(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, trick: dict[str, str]
) -> None:
    wrong = fact(
        ZIP,
        f"{ZIP} is a PDF of twelve pages.",
        "a PDF of twelve pages",
        f"{PDF} is a PDF of twelve pages",
        **trick,
    )
    result = ingest(client, app, fake_openai, APART, answer(nodes=FILES, facts=[wrong]))
    assert statements(app, ZIP) == [] and result["facts"]["neighbour"] == 1


def test_an_alias_made_of_the_other_files_words_does_not_name_the_subject(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    nodes = [{"name": ZIP, "kind": "object", "aliases": ["PDF"]}, FILES[1]]
    wrong = fact(
        ZIP,
        f"{ZIP} is a PDF of twelve pages.",
        "a PDF of twelve pages",
        f"{PDF} is a PDF of twelve pages",
    )
    result = ingest(client, app, fake_openai, APART, answer(nodes=nodes, facts=[wrong]))
    assert statements(app, ZIP) == [] and result["facts"]["neighbour"] == 1


def test_a_subject_an_edge_links_is_the_facts_object(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A verb-first quote of a relation ("leads the solar project"), with the
    relation stated as an edge: the project is Karim's object, not a neighbour."""
    text = "Karim leads the solar project since March."
    nodes = [{"name": "Karim", "kind": "person"}, {"name": "Solar project", "kind": "project"}]
    role = fact("Karim", "Karim leads the solar project.", "lead", "leads the solar project")
    edge = {"src": "Karim", "dst": "Solar project", "type": "manages", "quote": "leads"}
    reply = answer(nodes=nodes, facts=[role], edges=[edge])
    result = ingest(client, app, fake_openai, text, reply)
    assert statements(app, "Karim") == ["Karim leads the solar project."]
    assert result["facts"]["neighbour"] == 0


def test_the_neighbour_known_from_memory_counts_though_the_answer_omits_it(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """A known subject the text names (tier-0 detection) is a neighbour too."""
    ingest(client, app, fake_openai, f"{PDF} arrived.", answer(nodes=[FILES[1]]))
    wrong = fact(
        ZIP,
        f"{ZIP} is a PDF of twelve pages.",
        "a PDF of twelve pages",
        f"{PDF} is a PDF of twelve pages",
    )
    result = ingest(client, app, fake_openai, APART, answer(nodes=[FILES[0]], facts=[wrong]))
    assert statements(app, ZIP) == [] and result["facts"]["neighbour"] == 1


def test_a_proposal_is_decided_by_review_and_then_attached(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    wrong = fact(ZIP, f"{ZIP} contains {MIXED}.", MIXED, f"Both are here: {MIXED}")
    ingest(client, app, fake_openai, SIDE_BY_SIDE, answer(nodes=FILES, facts=[wrong]))
    [row] = app.state.store.execute("SELECT id FROM proposals WHERE kind = 'fact'").fetchall()
    accept = {"decision": "accept"}
    # an agent token (read, write) holds no review: it can neither apply nor hide it
    refused = client.post(f"/v1/proposals/{row['id']}", json=accept, headers=auth("alice"))
    assert refused.status_code == 403
    assert statements(app, ZIP) == []
    accepted = client.post(f"/v1/proposals/{row['id']}", json=accept)
    assert accepted.status_code == 200, accepted.text
    assert statements(app, ZIP) == [wrong["statement"]]


# ------------------------------------------------------------------ controls: nothing lost


def test_two_files_each_with_their_own_facts_are_both_written(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    result = ingest(
        client, app, fake_openai, APART, answer(nodes=FILES, facts=[ZIP_FACT, PDF_FACT])
    )
    assert statements(app, ZIP) == [ZIP_FACT["statement"]]
    assert statements(app, PDF) == [PDF_FACT["statement"]]
    assert result["facts"]["neighbour"] == 0 and proposals(app) == []


def test_a_zip_holding_the_pdf_is_the_zips_fact(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = f"{ZIP} is a zip archive holding {PDF} and two images."
    held = fact(ZIP, text, f"a zip archive holding {PDF} and two images", text.rstrip("."))
    result = ingest(client, app, fake_openai, text, answer(nodes=FILES, facts=[held]))
    assert statements(app, ZIP) == [text] and result["facts"]["neighbour"] == 0


def test_two_people_in_one_turn_each_keep_their_facts(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Karim moved to Lyon. Nadia works as a nurse. He loves it there."
    people = [{"name": "Karim", "kind": "person"}, {"name": "Nadia", "kind": "person"}]
    facts = [
        fact("Karim", "Karim moved to Lyon.", "Lyon", "Karim moved to Lyon"),
        fact("Nadia", "Nadia works as a nurse.", "nurse", "Nadia works as a nurse"),
        fact("Karim", "Karim loves Lyon.", None, "He loves it there"),
    ]
    result = ingest(client, app, fake_openai, text, answer(nodes=people, facts=facts))
    assert statements(app, "Karim") == ["Karim loves Lyon.", "Karim moved to Lyon."]
    assert statements(app, "Nadia") == ["Nadia works as a nurse."]
    assert result["facts"]["neighbour"] == 0


def test_a_neighbour_named_by_the_value_is_the_facts_object(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Karim called. He moved to Lyon."
    nodes = [{"name": "Karim", "kind": "person"}, {"name": "Lyon", "kind": "place"}]
    moved = fact("Karim", "Karim moved to Lyon.", "Lyon", "He moved to Lyon")
    result = ingest(client, app, fake_openai, text, answer(nodes=nodes, facts=[moved]))
    assert statements(app, "Karim") == ["Karim moved to Lyon."]
    assert result["facts"]["neighbour"] == 0


def test_the_sentence_of_the_other_person_is_not_attached_on_its_own(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    text = "Karim moved to Lyon. Nadia works as a nurse."
    people = [{"name": "Karim", "kind": "person"}, {"name": "Nadia", "kind": "person"}]
    wrong = fact("Karim", "Karim works as a nurse.", "nurse", "Nadia works as a nurse")
    result = ingest(client, app, fake_openai, text, answer(nodes=people, facts=[wrong]))
    assert statements(app, "Karim") == [] and result["facts"]["neighbour"] == 1


# ------------------------------------------------------------------ the owner


@pytest.fixture
def owned(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    owners = (("user:alice", "Alice", ()),)
    application = create_app(tier2_settings(tmp_path, fake_openai, owners=owners), clock=clock)
    yield application
    application.state.store.close()


def test_the_owner_speaking_beside_another_person_keeps_her_fact(
    owned: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The person speaking is named by her part's label, not in her sentence."""
    text = "Dinner with Karim tonight, I live in Lyon now."
    nodes = [{"name": "Alice", "kind": "person"}, {"name": "Karim", "kind": "person"}]
    home = fact(
        "Alice", "Alice lives in Lyon.", "Lyon", "Dinner with Karim tonight, I live in Lyon now"
    )
    with TestClient(owned, headers=auth("all")) as client:
        result = ingest(client, owned, fake_openai, text, answer(nodes=nodes, facts=[home]))
        assert statements(owned, "Alice") == ["Alice lives in Lyon."]
    assert result["facts"]["neighbour"] == 0 and result["facts"]["proposed"] == 0


# ------------------------------------------------------------------ the rules, as functions


@pytest.mark.parametrize(
    ("quote", "own", "others", "value", "expected"),
    [
        (f"{PDF} is a PDF of twelve pages", [ZIP], {"b": [PDF]}, "", True),
        (f"{ZIP} is a zip archive", [ZIP], {"b": [PDF]}, "", False),
        ("Karim Benali is moving", ["Karim"], {"b": ["Karim Benali"]}, "", True),
        ("Karim Benali is moving", ["Karim Benali"], {"b": ["Karim"]}, "", False),
        ("Karim and Nadia came", ["Karim"], {"n": ["Nadia"]}, "", False),
        # a near or inflected form is never read as the subject: proposed
        ("Анна видела Ивана", ["Иван"], {"a": ["Анна"]}, "", True),
        ("Nadia met Karima", ["Karim"], {"n": ["Nadia"]}, "", True),
        ("Nadia leads the project", ["Solar project"], {"n": ["Nadia"]}, "", True),
        ("Иван и Анна пьют чай", ["Ivan"], {"a": ["Анна"]}, "", False),  # transliterated
        ("Alice says he left", ["Bob"], {"a": ["Alice"]}, "", True),
        ("he moved to Paris", ["Karim"], {"p": ["Paris"]}, "Paris", False),  # the object
        (
            "leads the solar project",
            ["Karim"],
            {"s": ["Solar project"]},
            "Karim leads the solar project",  # the statement names its object
            False,
        ),
        ("he left", ["Karim"], {"n": ["Nadia"]}, "", False),  # no name at all
        ("Karima moved", ["Karim"], {"k": ["Karima"]}, "", True),  # a word, not a substring
        ("王伟住在北京", ["李娜"], {"w": ["王伟"]}, "", False),  # no spaces: not split
    ],
)
def test_quote_names_other(
    quote: str, own: list[str], others: dict[str, list[str]], value: str, expected: bool
) -> None:
    assert quote_names_other(quote, own, others, objects=value) is expected


@pytest.mark.parametrize(
    ("value", "forms", "expected"),
    [
        (MIXED, {"z": [ZIP], "p": [PDF]}, True),
        ("a zip with three folders", {"z": [ZIP], "p": [PDF]}, False),
        (f"{PDF} and two images", {"z": [ZIP], "p": [PDF]}, False),
        (f"a zip archive holding {PDF}", {"z": [ZIP], "p": [PDF]}, False),  # names it whole
        ("a zip with three photos", {"z": [ZIP], "p": ["photos.zip"]}, False),
        ("Karim", {"z": ["Alice"], "p": ["Karim"]}, False),  # a relation
        ("version 2 of the 2026 plan", {"z": ["plan 2026"], "p": ["memo 2026"]}, False),
        (None, {"z": [ZIP], "p": [PDF]}, False),
    ],
)
def test_value_merges(value: str | None, forms: dict[str, list[str]], expected: bool) -> None:
    assert value_merges(value, "z", _distinctive(forms), forms) is expected
