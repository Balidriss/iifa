"""A word that mixes scripts is malformed (SPEC 4.2): the check that would have
caught qwen3.5:9b's "انتقل يوسف إلى كazan office" before it closed a true fact.
What cannot be told apart is never rejected."""

from __future__ import annotations

import pytest

from iifa.ingest.checks import mixed_script_words


@pytest.mark.parametrize(
    ("statement", "words"),
    [
        ("انتقل يوسف إلى كazan office.", ["كazan"]),  # Arabic + Latin in one word
        ("Kазань is where Ivan lives now.", ["Kазань"]),  # Latin K + Cyrillic
        ("Иван живёт в Lyonе с семьёй.", ["Lyonе"]),
        ("Maria visits Αthens every summer.", ["Αthens"]),  # Greek Alpha + Latin
        ("Karim and كريم and Karimكريم.", ["Karimكريم"]),
    ],
)
def test_a_word_mixing_two_spaced_scripts_is_found(statement: str, words: list[str]) -> None:
    assert mixed_script_words(statement) == words


@pytest.mark.parametrize(
    "statement",
    [
        "Karim lives in Lyon since March 2026.",
        "Карим живёт в Лионе с 2026 года.",
        "كريم يقود تخطيط المشروع الشمسي منذ مارس 2026.",
        "يعمل يوسف في مكتب Google في كازان.",  # a whole Latin word next to Arabic ones
        "مَكْتَب يوسف في كازان.",  # Arabic with its marks
        "健二はIBMで働いている。",  # Latin inside Japanese: no spaces there
        "美玲在Google的办公室工作。",
        "健二は大阪のオフィスで働いている。",  # Chinese characters and kana
        "Émilie préfère le thé vert, l'e-mail et les réunions courtes.",
        "Karim bought an iPhone 15 and a Москва-2026 ticket.",  # hyphen separates words
        "Karim's Москва2026 ticket arrived.",  # digits are not letters
        "",
    ],
)
def test_what_cannot_be_told_apart_is_not_rejected(statement: str) -> None:
    assert mixed_script_words(statement) == []
