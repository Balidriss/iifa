"""Near-form aliases (SPEC 5.1): a grounded alias one letter or an ending away
from the node's name or from one of its aliases ("Ивана" for Иван, "Карим" for
Карима) is a proposal, never kept on its own - kept, it would attach every
later mention of the near name by exact match, before the adjudication could
see it. Found with huihui_ai/Qwen3.8-abliterated:27b on the traps corpus: 5
wrong attachments, 3 false supersessions. A far nickname (Kiki) and the same
name in another script (Ivan / Иван) are still kept."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.scopes import Visibility
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.tier2 import SCOPE, answer, post_turn, run_all

VIEW = Visibility.of_scope(SCOPE)
KAZAN = "Иван живёт в Казани."  # Ivan lives in Kazan
JOB = "У Ивана новая работа."  # Ivan has a new job (his name in the genitive)
TEA = "Ивана предпочитает чай."  # Ivana prefers tea
COFFEE = "Ивана любит кофе."  # Ivana likes coffee
NEW = chat_json({"decisions": [{"mention": 0, "choice": "new"}]})


def told(name: str, statement: str, aliases: tuple[str, ...] = (), **fact: Any) -> dict[str, Any]:
    reply: dict[str, Any] = answer(
        nodes=[{"name": name, "kind": "person", "aliases": list(aliases)}],
        facts=[{"subject": name, "statement": statement, "quote": statement.rstrip(".")} | fact],
    )
    return reply


def turn(
    client: TestClient, app: FastAPI, fake: FakeOpenAIServer, text: str, *replies: Any
) -> dict[str, Any]:
    fake.script(chat=list(replies))
    post_turn(client, text)
    report = run_all(client, app)
    result: dict[str, Any] = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    return result


def node(app: FastAPI, name: str) -> Any:
    found = app.state.store.find_node(name, VIEW)
    assert found is not None, name
    return found


def statements(app: FastAPI, name: str) -> list[str]:
    return sorted(f.statement for f in app.state.store.facts([node(app, name).id], VIEW))


def alias_proposals(app: FastAPI, name: str) -> list[str]:
    found = app.state.store.proposals([node(app, name).id], VIEW)
    return [p.value for p in found if p.kind == "alias"]


@pytest.mark.parametrize("order", ["alias_on_a_new_node", "alias_on_a_known_node", "ivana_first"])
def test_an_inflected_alias_never_attaches_ivanas_facts_to_ivan(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer, order: str
) -> None:
    both = f"{KAZAN} {JOB}"  # the source holds "Ивана": the alias is grounded
    ivan_turn = told("Иван", KAZAN, ("Ивана",))
    if order == "alias_on_a_new_node":
        result = turn(client, app, fake_openai, both, ivan_turn)
    elif order == "alias_on_a_known_node":
        turn(client, app, fake_openai, KAZAN, told("Иван", KAZAN))
        result = turn(client, app, fake_openai, both, ivan_turn)
    else:  # Ivana known first: "Ивана" is another node's name, already a proposal
        turn(client, app, fake_openai, TEA, told("Ивана", TEA))
        result = turn(client, app, fake_openai, both, ivan_turn, NEW)
        assert result["adjudication_calls"] == 1  # Иван is a near name of Ivana's
    assert node(app, "Иван").aliases == ()
    assert result["aliases"]["kept"] == 0 and result["aliases"]["proposed"] == 1
    assert alias_proposals(app, "Иван") == ["Ивана"]
    if order != "ivana_first":
        assert result["aliases"]["near"] == 1
        # Ivana's first mention reaches the adjudication instead of attaching
        later = turn(client, app, fake_openai, TEA, told("Ивана", TEA), NEW)
        assert later["adjudication_calls"] == 1
        [asked] = later["adjudications"]
        assert (asked["mention"], asked["chosen"]) == ("Ивана", None)
        assert [c["name"] for c in asked["candidates"]] == ["Иван"]
    turn(client, app, fake_openai, COFFEE, told("Ивана", COFFEE))
    assert statements(app, "Иван") == [KAZAN]
    assert statements(app, "Ивана") == sorted([TEA, COFFEE])


def test_a_near_form_of_another_alias_is_proposed_and_the_spelling_kept(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The 27B's Карима: "Karima" is her name in Latin letters (kept), "Карим"
    an ending away (proposed); "Karim" is near "Karima", the alias just kept."""
    said = "Карима живёт в Лионе."
    text = f"{said} В паспорте Karima, но друзья зовут её Карим или Karim."
    result = turn(
        client, app, fake_openai, text, told("Карима", said, ("Karima", "Карим", "Karim"))
    )
    assert node(app, "Карима").aliases == ("Karima",)
    assert sorted(alias_proposals(app, "Карима")) == ["Karim", "Карим"]
    assert result["aliases"]["near"] == 2 and result["aliases"]["kept"] == 1


@pytest.mark.parametrize(
    ("name", "said", "text", "alias"),
    [
        ("Karim", "Karim aime le thé.", "Karim aime le thé. On l'appelle Kiki.", "Kiki"),
        ("Иван", KAZAN, f"{KAZAN} Sur son passeport : Ivan.", "Ivan"),
        ("Ivan", "Ivan habite à Lyon.", "Ivan habite à Lyon. En russe, Иван.", "Иван"),
    ],
)
def test_a_far_nickname_and_the_same_name_in_another_script_are_kept(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    name: str,
    said: str,
    text: str,
    alias: str,
) -> None:
    result = turn(client, app, fake_openai, text, told(name, said, (alias,)))
    assert node(app, name).aliases == (alias,)
    assert result["aliases"]["near"] == 0 and result["aliases"]["proposed"] == 0


def test_an_inflected_subject_as_written_names_the_subject_but_is_no_alias(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Ivan has a new job, his name in the genitive: the adjudication attaches
    the fact to him, and "Ивана" is not stored as his alias for every source."""
    turn(client, app, fake_openai, KAZAN, told("Иван", KAZAN))
    ivan = node(app, "Иван").id
    attach = chat_json({"decisions": [{"mention": 0, "choice": ivan}]})
    reply = told("Иван", JOB, subject_as_written="Ивана")
    result = turn(client, app, fake_openai, JOB, reply, attach)
    assert statements(app, "Иван") == sorted([KAZAN, JOB])
    assert node(app, "Иван").aliases == ()
    assert result["aliases"]["near"] == 1
    later = turn(client, app, fake_openai, TEA, told("Ивана", TEA), NEW)
    assert later["adjudication_calls"] == 1
    assert statements(app, "Иван") == sorted([KAZAN, JOB])
