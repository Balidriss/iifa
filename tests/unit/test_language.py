"""P12's detector: scripts, function words, sentences of a mixed source, and an
honest ``und`` when the text holds no evidence (SPEC 13.1: multilingual fixtures)."""

from __future__ import annotations

import pytest

from iifa.language import (
    UNDETERMINED,
    describe,
    detect,
    is_code,
    language_name,
    language_of_span,
    languages_present,
    mismatch,
    script_counts,
    sentence_languages,
    sentences,
)
from tests.fixtures.corpus import SAMPLES, statements


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("en", "en"),
        ("fr", "fr"),
        ("fr_accents", "fr"),
        ("ru", "ru"),
        ("ar", "ar"),
        ("zh", "zh"),
        ("ja", "ja"),
    ],
)
def test_fixture_samples_detect_their_language(sample: str, expected: str) -> None:
    assert detect(SAMPLES[sample]) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Karim leads the installation planning of the solar project.", "en"),
        ("Karim dirige la planification de l'installation du projet solaire.", "fr"),
        ("Karim préfère le thé", "fr"),
        ("Karim a déménagé à Lyon", "fr"),
        ("Karim ist seit März für das Solarprojekt verantwortlich.", "de"),
        ("Karim trabaja en el proyecto solar desde marzo.", "es"),
        ("Karim lavora sul progetto solare da marzo.", "it"),
        ("Karim trabalha no projeto solar e mora em Lisboa desde março.", "pt"),
        ("Karim werkt sinds maart aan het zonneproject.", "nl"),
        ("Карим руководит планированием.", "ru"),
        ("Карім керує плануванням установки.", "uk"),
        ("كريم يقود تخطيط تركيب المشروع الشمسي.", "ar"),
        ("کریم پروژه خورشیدی را رهبری می‌کند.", "fa"),
        ("卡里姆负责太阳能项目的安装规划。", "zh"),
        ("カリムは太陽光プロジェクトの設置計画を率いている。", "ja"),
        ("카림은 태양광 프로젝트를 이끌고 있다.", "ko"),
        ("Ο Καρίμ ηγείται του ηλιακού έργου.", "el"),
        ("כרים מוביל את פרויקט השמש.", "he"),
        ("करीम सौर परियोजना का नेतृत्व करते हैं।", "hi"),
        ("คาริมเป็นผู้นำโครงการพลังงานแสงอาทิตย์", "th"),
    ],
)
def test_scripts_and_function_words(text: str, expected: str) -> None:
    assert detect(text) == expected


@pytest.mark.parametrize(
    "text",
    ["Karim prefers tea", "Émilie", "Karim", "2026-03-01", "", "   ", "Lyon Paris Berlin"],
)
def test_no_evidence_means_undetermined_never_a_guess(text: str) -> None:
    assert detect(text) == UNDETERMINED


def test_shared_words_prove_nothing() -> None:
    # "a" is an English article and a French verb; "plus" exists in both: no verdict.
    assert detect("a plus") == UNDETERMINED
    # one unique French word is enough, and a French accent breaks ties
    assert detect("le plus") == "fr"
    assert detect("des réponses") == "fr"


def test_mixed_source_is_judged_sentence_by_sentence() -> None:
    text = (
        "Karim leads the solar project since March. "
        "Émilie préfère les réponses courtes. "
        "Карим переехал в Лион."
    )
    spans = sentences(text)
    assert [text[s:e] for s, e in spans] == [
        "Karim leads the solar project since March",
        "Émilie préfère les réponses courtes",
        "Карим переехал в Лион",
    ]
    assert [lang for _, _, lang in sentence_languages(text)] == ["en", "fr", "ru"]
    # most frequent sentence language first; ties go to the whole text's verdict
    assert languages_present(text + " Karim is an electrician.") == ["en", "fr", "ru"]
    assert languages_present(text)[0] == detect(text)
    assert languages_present("Karim. Lyon.") == [] and languages_present("le thé") == ["fr"]
    french = text.index("Émilie")
    assert language_of_span(text, french, french + len("Émilie")) == "fr"
    russian = text.index("Карим переехал")
    assert language_of_span(text, russian, russian + 5) == "ru"
    # a span with no evidence of its own falls back to its sentence, then the text
    assert language_of_span(text, 0, 5) == "en"
    assert language_of_span("Karim. Lyon.", 0, 5) == UNDETERMINED
    assert language_of_span(text, -10, 10**6) in ("en", "fr", "ru")


def test_sentences_handle_cjk_punctuation_and_blank_pieces() -> None:
    text = "卡里姆负责规划。\n\n  カリムは率いている！ ..."
    assert [text[s:e] for s, e in sentences(text)] == ["卡里姆负责规划", "カリムは率いている"]
    assert sentences("") == [] and sentences("...") == []


def test_synthetic_corpus_is_mostly_determined_and_never_wrong() -> None:
    """Every determined verdict on the synthetic corpus is the language the
    statement was generated in; undetermined ones are the short English forms."""
    expected = {
        "works on": "en",
        "moved to": "en",
        "prefers": "en",
        "manages": "en",
        "dislikes": "en",
        "studies": "en",
        "travaille": "fr",
        "déménagé": "fr",
        "préfère": "fr",
        "gère": "fr",
        "déteste": "fr",
        "étudie": "fr",
        "работает": "ru",
        "переехал": "ru",
        "предпочитает": "ru",
        "управляет": "ru",
        "не любит": "ru",
        "изучает": "ru",
    }
    determined = 0
    for statement in statements(300):
        truth = next(lang for marker, lang in expected.items() if marker in statement)
        found = detect(statement)
        if found != UNDETERMINED:
            determined += 1
            assert found == truth, statement
    assert determined >= 200


def test_helpers() -> None:
    assert mismatch("en", "fr") and not mismatch("fr", "fr")
    assert not mismatch(UNDETERMINED, "fr") and not mismatch("fr", UNDETERMINED)
    assert language_name("fr") == "French" and language_name("xx") == "xx"
    assert describe([]) == "the language of the source"
    assert describe(["fr"]) == "French" and describe(["fr", "ar"]) == "French and Arabic"
    assert describe(["en", "fr", "ru"]) == "English, French and Russian"
    assert is_code("fr") and is_code("ast") and not is_code("FR") and not is_code("french")
    counts = script_counts("Karim كريم 卡里姆 123")
    assert counts["latin"] == 5 and counts["arabic"] == 4 and counts["han"] == 3
