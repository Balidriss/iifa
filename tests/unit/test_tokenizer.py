"""Query tokenization: case, accents and scripts."""

from __future__ import annotations

from iifa.recall.tokenizer import normalize, tokenize


def test_tokenize_folds_case_and_diacritics() -> None:
    assert tokenize("Émilie préfère les RÉPONSES courtes") == [
        "emilie",
        "prefere",
        "les",
        "reponses",
        "courtes",
    ]


def test_tokenize_splits_on_punctuation_and_keeps_digits() -> None:
    assert tokenize("Karim (كريم) said: 3月 — mars 2026!") == [
        "karim",
        "كريم",
        "said",
        "3月",
        "mars",
        "2026",
    ]


def test_tokenize_handles_cyrillic_and_empty() -> None:
    assert tokenize("Карим — руководит.") == ["карим", "руководит"]
    assert tokenize("   ") == []


def test_normalize_is_stable_for_name_matching() -> None:
    assert normalize("  Solar   Project ") == normalize("solar-project") == "solar project"
    assert normalize("Émilie") == "emilie"
