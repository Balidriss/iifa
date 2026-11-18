"""What a name or an alias is (SPEC 5.1): Markdown stripped, unbalanced brackets
and quotes refused, never repaired by a guess; names that legitimately hold
brackets, apostrophes, hyphens or another script are kept as they are."""

from __future__ import annotations

import pytest

from iifa.names import balanced, balanced_span, plain_name, strip_markdown


@pytest.mark.parametrize(
    ("raw", "plain"),
    [
        (
            "r\N{LATIN SMALL LETTER E WITH ACUTE}union du **mardi 3 mars",
            "r\N{LATIN SMALL LETTER E WITH ACUTE}union du mardi 3 mars",
        ),
        ("**Alice**", "Alice"),
        ("__Alice__", "Alice"),
        ("*Karim*", "Karim"),
        ("_Karim_", "Karim"),
        ("~~Bob~~", "Bob"),
        ("`Bob`", "Bob"),
        ("``Bob``", "Bob"),
        ("[Le Orme](https://example.org/orme)", "Le Orme"),
        ("![Le Orme](orme.png)", "Le Orme"),
        ("## Le Orme", "Le Orme"),
        ("> Le Orme", "Le Orme"),
        ("- Le Orme", "Le Orme"),
        ("* **Le Orme**", "Le Orme"),
        ("  Le   Orme  ", "Le Orme"),
        ("**- Alice**", "Alice"),
        ("**> Carol**", "Carol"),
        ("_Le Orme_ ouvre", "Le Orme ouvre"),
    ],
)
def test_markdown_is_stripped(raw: str, plain: str) -> None:
    assert plain_name(raw) == plain


@pytest.mark.parametrize(
    "raw",
    [
        "Le Orme (Orme Vallon",
        "Le Orme Orme Vallon)",
        "(a [b) c]",
        "Le [Orme",
        'Le "Orme',
        "\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}Le Orme",
        "\N{LEFT DOUBLE QUOTATION MARK}Le Orme",
        "\N{LEFT CORNER BRACKET}\N{CJK UNIFIED IDEOGRAPH-4E1C}\N{CJK UNIFIED IDEOGRAPH-4EAC}",
        "**",
        "``",
        "- ",
        "",
    ],
)
def test_unbalanced_or_empty_is_no_name(raw: str) -> None:
    assert plain_name(raw) is None


@pytest.mark.parametrize(
    "name",
    [
        "Lyon (office)",
        "Le Orme (Orme Vallon)",
        "Agence [Nord] {B}",
        "O'Brien",
        "d'Artagnan",
        "O\N{RIGHT SINGLE QUOTATION MARK}Brien",
        "Jean-Pierre",
        "Marie-Anne de La Tour",
        "john_doe",
        "E*Trade",
        "1. FC K\N{LATIN SMALL LETTER O WITH DIAERESIS}ln",
        '"Le Orme"',
        "\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}Le Orme\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}",
        "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}Le Orme\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}",
        "\N{DOUBLE LOW-9 QUOTATION MARK}Le Orme\N{LEFT DOUBLE QUOTATION MARK}",
        "\N{CYRILLIC CAPITAL LETTER I}\N{CYRILLIC SMALL LETTER VE}\N{CYRILLIC SMALL LETTER A}"
        "\N{CYRILLIC SMALL LETTER EN}",
        "\N{CJK UNIFIED IDEOGRAPH-674E}\N{CJK UNIFIED IDEOGRAPH-5C0F}\N{CJK UNIFIED IDEOGRAPH-9F99}",
        "\N{ARABIC LETTER MEEM}\N{ARABIC LETTER HAH}\N{ARABIC LETTER MEEM}\N{ARABIC LETTER DAL}",
        "\N{LEFT CORNER BRACKET}\N{CJK UNIFIED IDEOGRAPH-4E1C}\N{RIGHT CORNER BRACKET}",
        "R2-D2",
        "*NSYNC",
        "_why",
        "john_",
        "x_y_z",
    ],
)
def test_real_names_are_kept_as_written(name: str) -> None:
    assert plain_name(name) == name


def test_an_apostrophe_is_not_a_quote() -> None:
    assert balanced("l'agence d'Ivan") and balanced("Rock 'n' Roll")


def test_strip_markdown_flattens_the_lines() -> None:
    assert strip_markdown("**Le\nOrme**") == "Le Orme"


def test_a_span_takes_the_bracket_its_source_holds_beside_it() -> None:
    text = "On passe au Lyon (bureau), mardi."
    start = text.index("Lyon")
    end = text.index("bureau") + len("bureau")
    assert text[slice(*balanced_span(text, start, end))] == "Lyon (bureau)"
    quoted = 'Le "Orme" ouvre.'
    end = quoted.index("Orme") + len("Orme")
    assert quoted[slice(*balanced_span(quoted, 0, end))] == 'Le "Orme"'


def test_a_span_no_neighbour_balances_is_left_as_it_is() -> None:
    text = "Le Orme (Orme Vallon, mardi."
    end = text.index("Vallon") + len("Vallon")
    assert balanced_span(text, 0, end) == (0, end)
    assert plain_name(text[:end]) is None


@pytest.mark.parametrize("raw", ["**- Alice**", "#**# Bob", "*_Karim_*", "[**Le Orme**](u)"])
def test_stripping_twice_changes_nothing(raw: str) -> None:
    once = strip_markdown(raw)
    assert strip_markdown(once) == once


def test_markdown_between_a_word_and_its_bracket_does_not_stop_the_completion() -> None:
    text = "a Lyon (*bureau*) demain"
    start, end = text.index("Lyon"), text.index("bureau") + len("bureau")
    span = text[slice(*balanced_span(text, start, end))]
    assert plain_name(span) == "Lyon (bureau)"
