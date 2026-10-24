"""Query-side tokenization, aligned with FTS5's ``unicode61 remove_diacritics 2``.

Documents are tokenized by SQLite; queries by this module. The two must agree
on ordinary text in every script the store may hold, and a test checks that
every token this function produces for the multilingual fixtures is a term
FTS5 indexed for the same text.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize", "tokenize"]


_LATIN_LIMIT = 0x250  # Basic Latin, Latin-1 and Latin Extended-A/B


def _fold(text: str) -> str:
    """Lower-case, and strip diacritics from Latin letters only - exactly what
    ``unicode61 remove_diacritics 2`` does. Marks on other scripts (Japanese
    dakuten, Arabic harakat) are part of the letter and stay."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", text):
        if ord(ch) < 128:
            out.append(ch.lower())
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        if ord(decomposed[0]) < _LATIN_LIMIT:
            out.append("".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower())
        else:
            out.append(ch.lower())
    return "".join(out)


def tokenize(text: str) -> list[str]:
    """Lower-cased, diacritics-stripped tokens split on anything not alphanumeric."""
    tokens: list[str] = []
    current: list[str] = []
    for ch in _fold(text):
        if ch.isalnum():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def normalize(text: str) -> str:
    """The canonical form of a name or alias for exact-but-forgiving matching."""
    return " ".join(tokenize(text))
