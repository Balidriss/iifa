"""Values in other forms and other languages (SPEC 5.4): a deterministic
comparison that says same, different or unknown - never a guess. A name is
never proved different (exonyms), a Hijri month is never converted."""

from __future__ import annotations

import pytest

from iifa.ingest.values import (
    DIFFERENT,
    SAME,
    UNKNOWN,
    compare_values,
    near_form,
    parse_date,
    precision,
    same_name,
    transliterate,
)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("mars 2026", "March 2026"),
        ("марта 2026 года", "2026-03"),
        ("март 2026", "mars 2026"),
        ("2026年3月", "März 2026"),
        ("2026년 3월", "marzo 2026"),
        ("مارس 2026", "آذار 2026"),
        ("03/2026", "2026-03"),
        ("15/03/2026", "2026-03-15"),
        ("15 mars 2026", "March 15, 2026"),
        ("2026-03", "2026-03-15"),  # one refines the other
        ("mars 2026", "15 марта 2026"),
    ],
)
def test_the_same_date_in_other_forms_and_languages(a: str, b: str) -> None:
    assert compare_values(a, b) == SAME
    assert compare_values(b, a) == SAME


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("mars 2026", "avril 2026"),
        ("March 2026", "апреля 2026"),
        ("15 mars 2026", "2026-03-16"),
        ("2026-03", "2027-03"),
        ("مارس 2026", "نيسان 2026"),
    ],
)
def test_different_dates_are_proved_different(a: str, b: str) -> None:
    assert compare_values(a, b) == DIFFERENT


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("رمضان 1447", "2026-03"),  # another calendar: never converted
        ("03/04/2026", "2026-04-03"),  # day/month order cannot be told
        ("spring 2026", "2026-03"),
    ],
)
def test_what_cannot_be_read_as_a_date_is_unknown(a: str, b: str) -> None:
    assert compare_values(a, b) == UNKNOWN


@pytest.mark.parametrize(
    ("a", "b", "verdict"),
    [
        ("42", "٤٢", SAME),
        ("42 kg", "٤٢ kg", SAME),
        ("3,5", "3.5", SAME),
        ("４２", "42", SAME),
        ("42", "43", DIFFERENT),
        ("42 kg", "43 kg", DIFFERENT),
        ("42 kg", "42 lb", UNKNOWN),
        ("42 kg", "42", UNKNOWN),
    ],
)
def test_numbers_in_any_digits(a: str, b: str, verdict: str) -> None:
    assert compare_values(a, b) == verdict


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Lyon", "Lyon"),
        ("Lyon", "Лион"),
        ("Berlin", "Берлин"),
        ("Kazan", "Казань"),
        ("Montréal", "Монреаль"),  # one letter apart on a long name
        ("Casablanca", "Касабланка"),
        ("Tunis", "Тунис"),
        ("Berlin", "برلين"),  # consonant skeletons
    ],
)
def test_the_same_name_across_scripts(a: str, b: str) -> None:
    assert compare_values(a, b) == SAME
    assert compare_values(b, a) == SAME


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Lyon", "Париж"),  # truly different, but a name is never proved different
        ("London", "Londres"),  # an exonym: the same city
        ("Pékin", "北京"),
        ("Osaka", "大阪"),  # no table for Chinese characters
        ("Casablanca", "カサブランカ"),  # kana: l/r and added vowels
        ("Lyon", "ليون"),  # a two-letter skeleton is too short to tell
        ("Lyon", "2026"),
        ("", "Lyon"),
    ],
)
def test_names_are_never_proved_different(a: str, b: str) -> None:
    assert compare_values(a, b) == UNKNOWN


def test_precision_prefers_the_fuller_date() -> None:
    assert precision("2026-03-15") > precision("mars 2026") > precision("Lyon")
    assert precision("15 марта 2026") == precision("2026-03-15")
    assert parse_date("mars 2026") == (2026, 3, None)
    assert parse_date("Lyon") is None


def test_transliteration_is_for_comparison_only() -> None:
    assert transliterate("Казань") == "kazan"
    assert (transliterate("Лион"), transliterate("Lyon")) == ("lion", "lyon")  # y/i: compared
    assert transliterate("カサブランカ") == "kasaburanka"
    assert transliterate("Émilie") == "emilie"


def test_same_name_confirms_a_transliteration_never_an_adapted_form() -> None:
    assert same_name("Samir", "Самир") and same_name("Alice", "alice")
    assert same_name("Alisa", "Алиса")  # a third party's name: the subject rule keeps it out
    assert not same_name("Alice", "Алиса")  # the Russian form, not a transliteration
    assert not same_name("Alice", "") and not same_name(" ", "Alice")


@pytest.mark.parametrize(
    ("alias", "name"),
    [
        ("Ивана", "Иван"),  # an ending added
        ("Иваном", "Иван"),  # an ending of two letters
        ("Карим", "Карима"),  # an ending dropped
        ("Karim", "Карима"),  # across scripts, once transliterated
        ("Ivana", "Иван"),
        ("Karym", "Karim"),  # one letter replaced inside
        ("Ivana Petrova", "Ivan Petrov"),  # word by word
        ("Ivana Petrov", "Ivan Petrov"),
        ("كريمة", "كريم"),  # an ending in an abjad, read as written
    ],
)
def test_a_near_form_is_one_letter_or_an_ending_apart(alias: str, name: str) -> None:
    assert near_form(alias, name) and near_form(name, alias)


@pytest.mark.parametrize(
    ("alias", "name"),
    [
        ("Ivan", "Иван"),  # one spelling in two scripts
        ("Karima", "Карима"),
        ("Иван", "иван"),  # case only
        ("Kiki", "Karim"),  # a far nickname
        ("Mimi", "Émilie"),
        ("Париж", "Paris"),  # an exonym: across scripts, an ending is not near
        ("Иваном", "Ivan"),
        ("Karimova", "Karim"),  # a longer, different word
        ("Karim Benali", "Karim"),  # a full name: another word count
        ("Jo", "Joe"),  # too short to tell
        ("王小红", "王小明"),  # no table, no spaces: not judged
        ("", "Ivan"),
    ],
)
def test_a_spelling_a_nickname_or_another_name_is_not_a_near_form(alias: str, name: str) -> None:
    assert not near_form(alias, name) and not near_form(name, alias)
