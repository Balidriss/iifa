"""The checks on what a real model writes (SPEC 4.2, 4.4b, 5.1, 5.2, 7.4b):
complete statements that name their subject, aliases grounded in the source,
sentence-level co-occurrence, answer caps sized from the source, and the JSON
object taken out of a plain answer when JSON mode is off."""

from __future__ import annotations

import json

import pytest

from iifa.ingest.checks import (
    cooccurring,
    grounded,
    is_complete,
    names_subject,
    source_spelling,
    words,
)
from iifa.ingest.extract import PipelineConfig
from iifa.ingest.schema import extract_object, parse_adjudication, parse_answer

# --------------------------------------------------------------------------- complete statements


@pytest.mark.parametrize(
    "statement",
    [
        "Karim leads the installation planning.",
        "Karim dirige la planification.",
        "Émilie déteste les réunions tôt le matin.",
        "Карим руководит планированием установки.",
        "كريم يقود تخطيط التركيب.",
        "卡里姆负责太阳能项目。",
        "カリムは設置計画を率いている。",
        "Reply in French.",
    ],
)
def test_a_sentence_is_complete(statement: str) -> None:
    assert is_complete(statement)


@pytest.mark.parametrize(
    "statement",
    ["dirige", "déteste", "Karim", "руководит", "l'installation d'Émilie", "负责", "率いる", " "],
)
def test_a_fragment_is_not(statement: str) -> None:
    # "l'installation d'Émilie" is two words once the elided articles are set aside
    assert not is_complete(statement)


def test_elided_articles_are_not_words() -> None:
    assert words("l'installation d'Émilie qu'il aime") == ["installation", "emilie", "il", "aime"]
    assert words("L’avis de l’équipe") == ["avis", "de", "equipe"]


@pytest.mark.parametrize(
    ("statement", "names"),
    [
        ("Karim dirige la planification.", ["Karim"]),
        ("La planification est dirigée par karim.", ["Karim"]),
        ("Le rôle d'Émilie est la planification.", ["Émilie"]),  # elision
        ("L'équipe d'Emilie préfère le thé.", ["Émilie"]),  # diacritics folded
        ("Задачи Карима включают планирование.", ["Карим"]),  # inflection
        ("Иваном руководит Карим с марта.", ["Иван"]),
        ("卡里姆负责太阳能项目的安装规划。", ["卡里姆"]),  # no spaces: substring
        ("Mimi préfère les réponses courtes.", ["Émilie Durand", "Mimi"]),  # an alias
        ("Karim leads the planning.", ["Karim Benali"]),  # one long word of the name
        ("The solar project starts in March.", ["Solar project"]),
    ],
)
def test_a_statement_names_its_subject(statement: str, names: list[str]) -> None:
    assert names_subject(statement, names)


@pytest.mark.parametrize(
    ("statement", "names"),
    [
        ("Il dirige la planification du projet.", ["Karim"]),
        ("Karimova dirige la planification.", ["Karim"]),  # a different, longer word
        ("Он руководит планированием.", ["Карим"]),
        ("负责太阳能项目的安装规划。", ["卡里姆"]),
        ("The team meets on Monday.", ["The AI Lab"]),  # a name of short words needs them all
    ],
)
def test_a_statement_that_does_not_name_its_subject(statement: str, names: list[str]) -> None:
    assert not names_subject(statement, names)


@pytest.mark.parametrize(
    ("statement", "names"),
    [
        ("Karim leads the planning.", ["Карим"]),  # a translation: the caller exempts it
        ("Карим переехал в Лион.", ["Karim"]),  # the answer must also spell it "Карим"
        ("カリムは設置計画を率いている。", ["Karim"]),
    ],
)
def test_a_name_in_another_script_never_names_the_subject(statement: str, names: list[str]) -> None:
    # SPEC 4.2: matching across scripts without the model's help is for embeddings
    assert not names_subject(statement, names)


def test_a_name_in_the_statements_script_is_still_required() -> None:
    # one of the names shares the script: that one must be named
    assert not names_subject("Он руководит планированием.", ["Karim", "Карим"])
    assert names_subject("Карим руководит планированием.", ["Karim", "Карим"])
    assert names_subject("カリムは設置計画を率いている。", ["カリム"])  # kana name, mixed sentence


def test_arabic_attached_prefixes_are_a_known_limitation() -> None:
    # SPEC 4.2: "وكريم" (and Karim) is not split, so it does not count as naming كريم.
    # Documented, not guessed: the fact is asked for again, then dropped.
    assert names_subject("كريم يقود التخطيط.", ["كريم"])
    assert not names_subject("وكريم يقود التخطيط.", ["كريم"])


# --------------------------------------------------------------------------- grounded aliases


@pytest.mark.parametrize(
    ("alias", "text"),
    [
        ("Mimi", "User: Mimi (c'est Émilie) préfère le thé."),
        ("mimi", "User: MIMI préfère le thé."),
        ("Emilie", "Émilie préfère le thé."),
        ("Karim Benali", "Hier, Karim  Benali a appelé."),
        ("Карим", "Карим руководит проектом."),
        ("卡里姆", "卡里姆负责太阳能项目。"),
    ],
)
def test_an_alias_found_in_the_source_is_grounded(alias: str, text: str) -> None:
    assert grounded(alias, text)


@pytest.mark.parametrize(
    ("alias", "text"),
    [
        ("Карим", "Иван руководит проектом."),  # an alias a real model hallucinated
        ("Kari", "Karim leads the planning."),  # a piece of a word is not the word
        ("K.", "Karim leads the planning."),
        ("", "anything"),
    ],
)
def test_an_alias_absent_from_the_source_is_not(alias: str, text: str) -> None:
    assert not grounded(alias, text)


# --------------------------------------------------------------------------- co-occurrence


def test_co_occurrence_pairs_only_nodes_of_the_same_sentence() -> None:
    text = "User: Karim dirige le projet solaire.\nAssistant: Émilie préfère le thé."
    forms = {"k": ["Karim"], "s": ["projet solaire"], "e": ["Émilie"]}
    assert cooccurring(text, forms, {}, order=["k", "s", "e"], limit=12) == [("k", "s")]


def test_a_quote_span_places_a_node_in_its_sentence() -> None:
    # Карима (inflected) is not the form "Карим": the fact quote places the node
    text = "Проект ведёт команда Карима. Иван любит чай."
    forms = {"k": ["Карим"], "p": ["Проект"], "i": ["Иван"]}
    spans = {"k": [(0, 27)]}
    pairs = cooccurring(text, forms, spans, order=["k", "p", "i"], limit=12)
    assert pairs == [("k", "p")]


def test_co_occurrence_is_bounded_per_sentence() -> None:
    names = [f"Person{i}" for i in range(6)]
    text = " and ".join(names) + " met."
    forms = {n: [n] for n in names}
    pairs = cooccurring(text, forms, {}, order=names, limit=3)
    assert pairs == [("Person0", "Person1"), ("Person0", "Person2"), ("Person1", "Person2")]


# --------------------------------------------------------------------------- answer caps


def test_the_answer_cap_grows_with_the_source_and_stops_at_the_bound() -> None:
    config = PipelineConfig(answer_base_tokens=256, answer_tokens_per_source_token=3.0)
    short, long = config.answer_cap("x" * 70), config.answer_cap("x" * 7000)
    assert 256 < short < long
    assert short == 256 + 3 * 20  # 70 characters are 20 tokens by the conservative estimate
    assert config.answer_cap("x" * 1_000_000) == config.answer_max_tokens


def test_the_cap_configuration_is_checked() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(answer_base_tokens=0)
    with pytest.raises(ValueError):
        PipelineConfig(answer_base_tokens=5000, answer_max_tokens=4096)


# --------------------------------------------------------------------------- JSON out of a plain answer

ANSWER = {"nodes": [{"name": "Karim"}], "facts": [], "edges": []}


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(ANSWER),
        "```json\n" + json.dumps(ANSWER) + "\n```",
        '<think>The user wants {"nodes": []} maybe.</think>\n' + json.dumps(ANSWER),
        "Here is the result:\n```\n" + json.dumps(ANSWER) + "\n```\nHope this helps {:)}",
        'Example: {"a": 1}. Answer: ' + json.dumps(ANSWER),
        "<think>\nunterminated reasoning " + json.dumps(ANSWER),
    ],
)
def test_the_answer_is_found_in_a_plain_answer(content: str) -> None:
    assert extract_object(content, ("nodes", "facts", "edges")) == ANSWER
    parsed = parse_answer(content)
    assert parsed.usable and [n.name for n in parsed.nodes] == ["Karim"]
    assert parsed.rejected.total == 0


def test_a_plain_answer_without_the_expected_keys_falls_back_to_its_first_object() -> None:
    assert extract_object('noise {"other": 1} {"x": 2}', ("nodes",)) == {"other": 1}
    assert extract_object("no object at all", ("nodes",)) is None
    assert extract_object('<think>{"nodes": []}</think> nothing after', ("nodes",)) is None


def test_adjudication_answers_are_found_the_same_way() -> None:
    content = '<think>candidate 0?</think> {"decisions": [{"mention": 0, "choice": "new"}]}'
    assert parse_adjudication(content, 1) == ({0: "new"}, 0)


@pytest.mark.parametrize(
    ("alias", "text", "spelling"),
    [
        ("карим", "Карим руководит проектом.", "Карим"),
        ("emilie", "Hier, Émilie a appelé.", "Émilie"),
        ("KARIM benali", "Hier, Karim  Benali a appelé.", "Karim  Benali"),
        ("卡里姆", "卡里姆负责太阳能项目。", "卡里姆"),
        ("Kari", "Karim leads.", None),
    ],
)
def test_the_source_spelling_is_the_exact_span_of_the_source(
    alias: str, text: str, spelling: str | None
) -> None:
    """An alias is stored as the source writes it, not as the model copied it."""
    assert source_spelling(alias, text) == spelling
    assert grounded(alias, text) is (spelling is not None)
