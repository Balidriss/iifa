"""P12 for undetermined texts: a short statement may hold no word of one
language only (13 of 26 English statements of the bench corpus); it is still
judged by its script, and only by its script."""

from __future__ import annotations

import pytest

from iifa.language import UNDETERMINED, detect, writing_mismatch


@pytest.mark.parametrize(
    "statement", ["Kenji works as a journalist.", "Nadia prefers tea.", "Karim lives in Paris."]
)
def test_these_short_english_statements_are_undetermined(statement: str) -> None:
    assert detect(statement) == UNDETERMINED


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("Kenji works as a journalist.", "ja"),
        ("Meiling lives in Paris.", "zh"),
        ("Karim lives in Kazan.", "ru"),
        ("Youssef lives in Tunis.", "ar"),
    ],
)
def test_an_undetermined_statement_in_another_script_is_a_mismatch(
    statement: str, expected: str
) -> None:
    assert writing_mismatch(statement, expected)


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("Nadia prefers tea.", "fr"),  # the known hole: same script, nothing judged
        ("Nadia prefers tea.", "en"),
        ("Nadia prefers tea.", UNDETERMINED),
        ("Карим", "ru"),
        ("2026", "ja"),  # no letters: nothing to judge
        ("Nadia prefers tea.", "sw"),  # a language whose script is not known here
    ],
)
def test_nothing_else_is_judged(statement: str, expected: str) -> None:
    assert not writing_mismatch(statement, expected)


def test_a_determined_statement_is_judged_as_before() -> None:
    assert writing_mismatch("Karim habite à Lyon depuis le printemps.", "en")
    assert not writing_mismatch("Karim habite à Lyon depuis le printemps.", "fr")
    assert writing_mismatch("Карим живёт в Лионе.", "fr")
