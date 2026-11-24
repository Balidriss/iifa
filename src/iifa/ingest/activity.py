"""The activity of a source, in the agent perspective.

What the owner asks the agent to do is what a later request repeats in other
words, and a rule of method never resembles that request. So the extraction
names the activity a source asks for, and the service keeps it only when the
owner's own words hold it: the quote lies in an owner part (checked by the
writer, which knows the parts) and a word of the name is written whole in
that quote. The check reads words, never meaning: it cannot tell a good name
from a bad one, only that the owner wrote part of it.
"""

from __future__ import annotations

from collections.abc import Sequence

from iifa.ingest.checks import unspaced_chars
from iifa.language import function_words
from iifa.recall.tokenizer import tokenize

MIN_WORD = 3  # the shortest word of a name that grounds it in the quote
UNSPACED_RUN = 2  # letters in a row, in a script written without spaces
UNSPACED_NAME = 4  # letters of a generic activity name written without spaces

_FUNCTION_WORDS = frozenset(t for word in function_words() for t in tokenize(word))


def matched_words(name: str, text: str) -> int:
    """How many words of ``name`` are written whole in ``text``: words of three
    letters or more that are not function words, or, in a script written
    without spaces, words two of whose letters in a row are there."""
    said = tokenize(text)
    words, joined = set(said), " ".join(said)
    count = 0
    for word in dict.fromkeys(tokenize(name)):
        if unspaced_chars(word):
            count += any(word[i : i + UNSPACED_RUN] in joined for i in range(len(word) - 1))
        elif len(word) >= MIN_WORD and word not in _FUNCTION_WORDS:
            count += word in words
    return count


def content_words(name: str) -> list[str]:
    """The words of ``name`` that count (``matched_words``): three letters or
    more and not a function word, or any word of a script written without
    spaces."""
    return [
        w
        for w in dict.fromkeys(tokenize(name))
        if unspaced_chars(w) or (len(w) >= MIN_WORD and w not in _FUNCTION_WORDS)
    ]


def two_words(name: str) -> bool:
    """``group_activities``: does ``name`` hold two words that count,
    or, in a script written without spaces, one of at least ``UNSPACED_NAME``
    letters? One word is the verb a model kept ("writing"), not an activity."""
    words = content_words(name)
    return len(words) >= 2 or any(unspaced_chars(w) and len(w) >= UNSPACED_NAME for w in words)


def shared_by(name: str, others: Sequence[str]) -> bool:
    """``group_activities``: does ``name`` hold at least one word that
    counts, and is every such word written whole in each of ``others``?"""
    words = content_words(name)
    return bool(words) and all(
        all(matched_words(word, other) > 0 for word in words) for other in others
    )


def grounded_activity(name: str, quote: str) -> bool:
    """Is a word of ``name`` written whole in ``quote`` (``matched_words``)?"""
    return matched_words(name, quote) > 0


__all__ = ["content_words", "grounded_activity", "matched_words", "shared_by", "two_words"]
