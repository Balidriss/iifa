"""`subject_as_written` (SPEC 4.2): a known subject keeps its known name in
`subject` (for resolution) and its spelling in the source in
`subject_as_written`, which names it in the statement's script and becomes a
source-anchored alias - only when the source and the statement hold that
spelling and it is no other node's name or alias."""

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
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth, remember
from tests.tier2 import SCOPE, answer, post_turn, run_all, tier2_settings

VIEW = Visibility.of_scope(SCOPE)
FRENCH = "Karim dirige la planification de l'installation du projet solaire depuis mars 2026."
RUSSIAN = "Карим руководит планированием установки солнечного проекта с марта 2026 года."
ARABIC = "كريم يقود تخطيط تركيب المشروع الشمسي منذ مارس 2026."


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(tier2_settings(tmp_path, fake_openai), clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def known_karim_fact(
    statement: str, quote: str, as_written: str | None, subject: str = "Karim"
) -> Any:
    fact: dict[str, Any] = {"subject": subject, "statement": statement, "quote": quote}
    if as_written is not None:
        fact["subject_as_written"] = as_written
    return answer(nodes=[{"name": subject, "kind": "person"}], facts=[fact])


def turn_result(client: TestClient, app: FastAPI, text: str) -> dict[str, Any]:
    post_turn(client, text)
    result: dict[str, Any] = run_all(client, app)["jobs"][0]["result"]
    return result


def test_the_known_subject_keeps_its_facts_in_every_script(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The manual run's order (French, then Russian, then Arabic): the model
    reuses the known name "Karim" and gives the source's spelling; the Russian
    and Arabic facts are kept, no retry, and the spellings become aliases."""
    fake_openai.script(
        chat=[
            known_karim_fact(
                "Karim dirige la planification du projet solaire.", "dirige la planification", None
            ),
            known_karim_fact(
                "Карим руководит планированием установки солнечного проекта.",
                "руководит планированием",
                "карим",  # the model's copy; the alias keeps the source's spelling
            ),
            known_karim_fact("كريم يقود تخطيط تركيب المشروع الشمسي.", "يقود تخطيط", "كريم"),
        ]
    )
    results = [turn_result(client, app, text) for text in (FRENCH, RUSSIAN, ARABIC)]
    for result in results:
        assert result["facts"]["written"] == 1 and result["facts"]["cross_script"] == 0
        assert result["retries"] == 0
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None and set(karim.aliases) == {"Карим", "كريم"}
    assert len(store.facts([karim.id], VIEW)) == 3
    assert store.counts()["nodes"] == 1


def test_a_spelling_absent_from_the_source_is_ignored(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    remember(client, FRENCH, node={"name": "Karim", "kind": "person"})
    wrong = known_karim_fact(
        "Каримов руководит планированием установки солнечного проекта.",
        "руководит планированием",
        "Каримов",
    )
    fake_openai.script(chat=[wrong, wrong])
    result = turn_result(client, app, RUSSIAN)
    assert result["facts"]["written"] == 0 and result["facts"]["cross_script"] == 1
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None and karim.aliases == ()


def test_a_spelling_absent_from_the_statement_is_ignored(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    remember(client, FRENCH, node={"name": "Karim", "kind": "person"})
    fragment = known_karim_fact(
        "Руководит планированием установки солнечного проекта.",
        "руководит планированием",
        "Карим",
    )
    fake_openai.script(chat=[fragment, fragment])
    result = turn_result(client, app, RUSSIAN)
    assert result["facts"]["written"] == 0 and result["facts"]["incomplete"] == 1
    karim = app.state.store.find_node("Karim", VIEW)
    assert karim is not None and karim.aliases == ()


def test_a_spelling_that_is_another_nodes_name_is_ignored(
    client: TestClient, app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    """The guard of SPEC 5.1: "Иван" is another known node's name; writing it as Karim's
    spelling in the source must not name Karim, nor become his alias."""
    remember(client, FRENCH, node={"name": "Karim", "kind": "person"})
    ivan = remember(client, "Иван любит чай.", node={"name": "Иван", "kind": "person"})
    swap = known_karim_fact("Иван руководит проектом.", "руководит проектом", "Иван")
    fake_openai.script(chat=[swap, swap])
    result = turn_result(client, app, "Иван руководит проектом.")
    assert result["facts"]["written"] == 0 and result["facts"]["cross_script"] == 1
    store = app.state.store
    karim = store.find_node("Karim", VIEW)
    assert karim is not None and karim.aliases == ()
    assert store.facts([ivan["node"]["id"]], VIEW)[0].statement == "Иван любит чай."


def test_the_prompt_asks_for_the_source_spelling_of_a_known_subject() -> None:
    system = extraction_messages("x", languages=["ru"], known_names=["Karim"])[0]["content"]
    assert (
        '"subject_as_written": "<the subject exactly as the source spells it, or null>"' in system
    )
    assert (
        "Subjects already known to memory (reuse these exact names as subject when the "
        "source means them; when the source spells one differently, also give that "
        "spelling as subject_as_written): Karim"
    ) in system
