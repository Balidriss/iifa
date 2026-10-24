"""The default token counter must be conservative in every script (P1, SPEC 7.3)."""

from __future__ import annotations

import pytest

from iifa.recall.handles import serialized
from iifa.tokens import ConservativeTokenizer, get_tokenizer
from tests.fixtures.corpus import SAMPLES


def test_ascii_prose_is_chars_over_3_5_rounded_up() -> None:
    counter = ConservativeTokenizer()
    assert counter.count("") == 0
    assert counter.count("abcdefg") == 2  # 7 / 3.5
    assert counter.count("abcdefgh") == 3  # 8 / 3.5 -> 2.29 -> 3
    assert counter.count("hello world") == 4


def test_identifiers_digits_and_punctuation_cost_more_than_prose() -> None:
    counter = ConservativeTokenizer()
    assert counter.count("01J8Z3M4Q5R6S7T8V9W0X1Y2Z3") == 26  # one token per character
    assert counter.count("abcdefghijklmnopqrstuvwxyz") == 8
    assert counter.count("2026-03-01") == 9  # 8 digits + 2 dashes / 2
    assert counter.count('{"a":[]}') == 4  # 7 punctuation chars / 2 + one letter / 3.5


def test_non_ascii_scripts_cost_utf8_length_minus_one_per_character() -> None:
    counter = ConservativeTokenizer()
    assert counter.count("Карим") == 5  # two-byte script: 1 per character
    assert counter.count("كريم") == 4
    assert counter.count("太阳能") == 6  # three-byte script: 2 per character
    assert counter.count("🙂") == 3


def test_registry_and_unknown_name() -> None:
    assert get_tokenizer("conservative").name == "conservative"
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_tokenizer("magic")


@pytest.mark.exact_tokenizer
@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_p1_conservative_estimate_is_at_or_above_the_exact_count(language: str) -> None:
    """When an exact counter is installed (CI on Linux warms its cache), the
    estimate must never be lower than the truth, in any script (SPEC 7.3)."""
    exact = _exact_or_skip()
    text = SAMPLES[language]
    # The JSON form is what the budget counts: the service's own serializer,
    # which keeps non-ASCII characters as they are (never \uXXXX escapes).
    as_handle = serialized({"id": "fact_01J8Z3M4Q5R6S7T8V9W0X1Y2Z3", "line": text, "flags": []})
    for candidate in (text, as_handle):
        assert ConservativeTokenizer().count(candidate) >= exact.count(candidate), candidate


def _exact_or_skip():  # type: ignore[no-untyped-def]
    pytest.importorskip("tiktoken")
    try:
        return get_tokenizer("tiktoken:cl100k_base")
    except Exception as exc:  # the cache is cold and the socket guard blocks the download
        pytest.skip(f"exact tokenizer not available offline: {type(exc).__name__}")
