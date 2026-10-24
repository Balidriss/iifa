"""The source form of a subject (SPEC 5.1): when the model writes a subject the
source does not hold ("Иван") but a near form of it is in the quote ("Ивана"),
the two must lead to exactly the same node, or the fact goes to the
adjudication under the source's form, never attached alone - whichever of
Ivan and Ivana is known first."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.ingest.extract import source_form
from iifa.ingest.resolve import DEFAULT_FUZZY_THRESHOLD
from iifa.scopes import Visibility
from tests.fakes.openai_server import FakeOpenAIServer, chat_json
from tests.tier2 import SCOPE, answer, post_turn, run_all

VIEW = Visibility.of_scope(SCOPE)
TEA = "Ивана предпочитает чай."  # Ivana prefers tea
SLIP = "Иван предпочитает чай."  # what the model wrote: Ivan prefers tea


def told(name: str, statement: str, quote: str) -> Any:
    return answer(
        nodes=[{"name": name, "kind": "person"}],
        facts=[{"subject": name, "statement": statement, "quote": quote}],
    )


def know(client: TestClient, app: FastAPI, fake: FakeOpenAIServer, name: str, said: str) -> str:
    fake.script(chat=[told(name, said, said.rstrip("."))])
    post_turn(client, said)
    run_all(client, app)
    node = app.state.store.find_node(name, VIEW)
    assert node is not None
    node_id: str = node.id
    return node_id


def slip(client: TestClient, app: FastAPI, fake: FakeOpenAIServer, choice: str) -> dict[str, Any]:
    """The Ivana turn, the model writing Ivan as its subject; ``choice`` is the
    adjudication's answer."""
    decision = chat_json({"decisions": [{"mention": 0, "choice": choice}]})
    fake.script(chat=[told("Иван", SLIP, TEA.rstrip(".")), decision, decision])
    post_turn(client, TEA)
    report = run_all(client, app)
    result: dict[str, Any] = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    return result


def statements(app: FastAPI, node_id: str) -> list[str]:
    return [f.statement for f in app.state.store.facts([node_id], VIEW)]


def test_ivan_known_ivana_never_seen_the_tea_does_not_land_on_ivan(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ivan = know(client, app, fake_openai, "Иван", "Иван живёт в Казани.")
    result = slip(client, app, fake_openai, "new")
    [asked] = result["adjudications"]
    assert asked["mention"] == "Ивана" and asked["chosen"] is None
    assert [c["name"] for c in asked["candidates"]] == ["Иван"]
    assert result["source_forms"] == 1
    assert statements(app, ivan) == ["Иван живёт в Казани."]
    ivana = app.state.store.find_node("Ивана", VIEW)  # named as the source writes it
    assert ivana is not None and statements(app, ivana.id) == [SLIP]
    # the context the adjudication read is the source sentence
    prompt = fake_openai.chat_calls[-1].messages[1]["content"]
    assert "Mention 0: 'Ивана'" in prompt and "Ивана предпочитает чай" in prompt


def test_both_known_the_adjudication_chooses_between_them(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ivana = know(client, app, fake_openai, "Ивана", "Ивана живёт в Париже.")
    ivan = know(client, app, fake_openai, "Иван", "Иван живёт в Казани.")
    result = slip(client, app, fake_openai, ivana)
    [asked] = result["adjudications"]
    assert asked["mention"] == "Ивана" and asked["chosen"] == "Ивана"
    assert {c["name"] for c in asked["candidates"]} == {"Иван", "Ивана"}
    assert statements(app, ivan) == ["Иван живёт в Казани."]
    assert sorted(statements(app, ivana)) == sorted(["Ивана живёт в Париже.", SLIP])


def test_ivana_known_first_ivan_never_seen(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    ivana = know(client, app, fake_openai, "Ивана", "Ивана живёт в Париже.")
    result = slip(client, app, fake_openai, ivana)
    # "Иван" is no node, only a near name of Ivana's: the adjudication decides
    assert [a["chosen"] for a in result["adjudications"]] == ["Ивана"]
    assert sorted(statements(app, ivana)) == sorted(["Ивана живёт в Париже.", SLIP])


def test_a_real_inflection_costs_one_adjudication_and_attaches(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Ivan has a new job, told with his name in the genitive ("Ивана"): the
    source form costs one adjudication, which attaches it to Ivan."""
    ivan = know(client, app, fake_openai, "Иван", "Иван живёт в Казани.")
    job = "У Ивана новая работа."
    decision = chat_json({"decisions": [{"mention": 0, "choice": ivan}]})
    fake_openai.script(chat=[told("Иван", "Иван получил новую работу.", job.rstrip(".")), decision])
    post_turn(client, job)
    report = run_all(client, app)
    result = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert result["source_forms"] == 1 and result["mentions_adjudicated"] == 1
    assert "Иван получил новую работу." in statements(app, ivan)


def test_the_subject_itself_in_the_quote_changes_nothing(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Ivan and Ivana in one sentence: the subject is written as is, no rule."""
    ivan = know(client, app, fake_openai, "Иван", "Иван живёт в Казани.")
    both = "Иван и Ивана пьют чай."
    fake_openai.script(chat=[told("Иван", "Иван пьёт чай.", both.rstrip("."))])
    post_turn(client, both)
    report = run_all(client, app)
    result = next(j for j in report["jobs"] if j["kind"] == "turn")["result"]
    assert result["source_forms"] == 0 and result["mentions_adjudicated"] == 0
    assert "Иван пьёт чай." in statements(app, ivan)


@pytest.mark.parametrize(
    ("subject", "quote", "form"),
    [
        ("Иван", "Ивана предпочитает чай", "Ивана"),
        ("Иван", "У Ивана новая работа", "Ивана"),
        ("Иван", "Иван и Ивана пьют чай", None),  # the subject itself is there
        ("Karim", "Karima habite à Lyon", "Karima"),
        ("Karim Benali", "Karima Benali habite à Lyon", "Karima Benali"),
        # across scripts, once transliterated (a scope forcing English, say)
        ("Ivan", "Ивана предпочитает чай", "Ивана"),
        ("Ivan", "Иван пьёт чай", None),  # the same name in another script
        ("Ivan", "Он пьёт чай", None),  # a pronoun: no source form
        ("Ivan", "Nadia drinks tea", None),  # nothing near
        ("Sofia", "София переехала в Париж", None),  # transliterated "sofiya": one name
        ("Daniel", "Daniela lives in Lyon", "Daniela"),  # one letter apart: two names
    ],
)
def test_the_source_form_of_a_subject(subject: str, quote: str, form: str | None) -> None:
    assert source_form(subject, quote, DEFAULT_FUZZY_THRESHOLD) == form


def test_the_node_the_model_named_is_not_created_empty(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """Ivana known, Ivan never seen: the model's node "Иван" is a new name, but
    its one fact goes to Ivana under the source form; no empty "Иван" node."""
    ivana = know(client, app, fake_openai, "Ивана", "Ивана живёт в Париже.")
    before = len(app.state.store.execute("SELECT id FROM nodes").fetchall())
    new = chat_json({"decisions": [{"mention": 0, "choice": "new"}]})
    attach = chat_json({"decisions": [{"mention": 0, "choice": ivana}]})
    fake_openai.script(chat=[told("Иван", SLIP, TEA.rstrip(".")), new, attach])
    post_turn(client, TEA)
    run_all(client, app)
    assert SLIP in statements(app, ivana)
    assert app.state.store.find_node("Иван", VIEW) is None
    assert len(app.state.store.execute("SELECT id FROM nodes").fetchall()) == before
