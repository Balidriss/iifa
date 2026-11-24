"""What a node name or an alias is, whatever path writes it.

A model often answers in Markdown, and the source it reads may be a reply
written in Markdown: ``**Name A`` or ``Name A (Name B``
are debris of formatting, not names. Every name and alias the store keeps goes
through ``plain_name`` first: Markdown is stripped, and what is left with an
unbalanced bracket or quote is refused, never repaired - the missing half
could belong in more than one place ("Name A (Name B)" or "Name A").
Nothing here depends on a language: only characters are read.
"""

from __future__ import annotations

import re

from iifa.recall.tokenizer import normalize

__all__ = [
    "InvalidName",
    "balanced",
    "balanced_span",
    "plain_name",
    "require_name",
    "strip_markdown",
]


class InvalidName(ValueError):
    """A name or alias that is not one once its formatting is removed."""


_WORDISH = r"[^\W_]"  # a letter or a digit
_LINK = re.compile(r"!?\[([^\[\]]*)\]\([^()\s]*\)")  # [text](url) and ![alt](url) -> text
_PREFIX = re.compile(r"^(?:#{1,6}\s+|>\s*|[-+*\N{BULLET}]\s+)+")  # heading, quote, bullet
_RUNS = re.compile(r"\*{2,}|_{2,}|~{2,}|`+")  # bold, strike-through, code ticks
# a single * or _ opening a word and the same one closing a later word (emphasis);
# never one inside a word (john_doe, E*Trade) nor one alone (*NSYNC, _why, john_)
_EMPHASIS = re.compile(rf"(?<!{_WORDISH})([*_])(?={_WORDISH})(.*?)(?<={_WORDISH})\1(?!{_WORDISH})")
_PASSES = 4  # a marker uncovered by stripping another is stripped too ("**- Alice**")

# brackets that nest: each closer must meet its own opener
_CLOSERS = {
    ")": "(",
    "]": "[",
    "}": "{",
    "\N{RIGHT CORNER BRACKET}": "\N{LEFT CORNER BRACKET}",
    "\N{RIGHT WHITE CORNER BRACKET}": "\N{LEFT WHITE CORNER BRACKET}",
    "\N{FULLWIDTH RIGHT PARENTHESIS}": "\N{FULLWIDTH LEFT PARENTHESIS}",
    "\N{FULLWIDTH RIGHT SQUARE BRACKET}": "\N{FULLWIDTH LEFT SQUARE BRACKET}",
    "\N{RIGHT BLACK LENTICULAR BRACKET}": "\N{LEFT BLACK LENTICULAR BRACKET}",
    "\N{RIGHT ANGLE BRACKET}": "\N{LEFT ANGLE BRACKET}",
    "\N{RIGHT DOUBLE ANGLE BRACKET}": "\N{LEFT DOUBLE ANGLE BRACKET}",
}
_OPENERS = frozenset(_CLOSERS.values())
# quotes whose direction differs between languages ("x", «x», »x«, „x“, “x”, ”x”):
# only their number is read - as many of one side as of the other, or an even count
_GUILLEMETS = (
    (
        "\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}",
        "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}",
    ),
    (
        "\N{SINGLE LEFT-POINTING ANGLE QUOTATION MARK}",
        "\N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK}",
    ),
)
_DOUBLE_QUOTES = (
    '"',
    "\N{LEFT DOUBLE QUOTATION MARK}",
    "\N{RIGHT DOUBLE QUOTATION MARK}",
    "\N{DOUBLE LOW-9 QUOTATION MARK}",
    "\N{DOUBLE HIGH-REVERSED-9 QUOTATION MARK}",
)


def strip_markdown(text: str) -> str:
    """``text`` on one line without Markdown: links keep their text, headings,
    quotes and bullets lose their marker, bold, italics, strike-through and code
    ticks disappear. A numbered item is left alone ("1. FC Köln" is a name), so
    is a ``*`` or ``_`` inside a word or without a partner. Stripping again
    changes nothing."""
    flat = " ".join(text.split())
    for _ in range(_PASSES):
        before = flat
        flat = _LINK.sub(r"\1", flat)
        flat = _RUNS.sub("", flat)
        flat = _EMPHASIS.sub(r"\2", flat)
        flat = " ".join(_PREFIX.sub("", flat).split())
        if flat == before:
            break
    return flat


def balanced(text: str) -> bool:
    """Does every bracket of ``text`` meet its partner, in order, and does every
    quote? An apostrophe is not a quote (O'Brien, d'Artagnan)."""
    stack: list[str] = []
    for ch in text:
        if ch in _OPENERS:
            stack.append(ch)
        elif ch in _CLOSERS and (not stack or stack.pop() != _CLOSERS[ch]):
            return False
    if stack:
        return False
    if any(text.count(left) != text.count(right) for left, right in _GUILLEMETS):
        return False
    return sum(text.count(q) for q in _DOUBLE_QUOTES) % 2 == 0


_EDGE_MARKS = frozenset(_CLOSERS) | _OPENERS | frozenset(_DOUBLE_QUOTES)
_EDGE_MARKS |= frozenset(mark for pair in _GUILLEMETS for mark in pair)
_EDGE_MARKS |= frozenset("*_~`")  # Markdown between a word and its bracket: "(*bureau*)"
_WIDEN = 5  # brackets, quotes and Markdown marks next to a span, at most, on each side


def balanced_span(text: str, start: int, end: int) -> tuple[int, int]:
    """``text[start:end]`` widened over the brackets and quotes right next to it,
    when that balances it: a span of words ends on a word, so "Lyon (bureau)"
    in the source reads "Lyon (bureau" until its own bracket closes it. Nothing
    is ever added that the source does not hold right there; an unbalanced span
    that no neighbouring mark balances is returned as it is."""
    if balanced(text[start:end]):
        return start, end
    for before in range(_WIDEN + 1):
        if before and (start - before < 0 or text[start - before] not in _EDGE_MARKS):
            break
        for after in range(_WIDEN + 1):
            if after and (end + after > len(text) or text[end + after - 1] not in _EDGE_MARKS):
                break
            if balanced(text[start - before : end + after]):
                return start - before, end + after
    return start, end


def require_name(text: str) -> str:
    """``plain_name`` or ``InvalidName`` (a ``ValueError``: a pydantic validator
    turns it into a refused item, a route into a 422)."""
    plain = plain_name(text)
    if plain is None:
        raise InvalidName("not a name once Markdown is stripped, or unbalanced brackets")
    return plain


def plain_name(text: str) -> str | None:
    """The name ``text`` holds once Markdown is stripped, or None when nothing
    of a name is left or a bracket or a quote is unbalanced."""
    flat = strip_markdown(text)
    if not normalize(flat) or not balanced(flat):
        return None
    return flat
