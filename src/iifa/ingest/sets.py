"""Sets: aspects that hold several elements at once - the members
of a club, the ingredients of a recipe - told one element at a time. The model
proposes a cardinality per fact (``one`` / ``many`` / ``unsure``) and, for a
removed or corrected element, the value it ``removes``; the service decides
with what the store already knows about the key. Free-form keys only: the
vocabulary keys keep their own rules. When we cannot tell, we do not
supersede: an unsure cardinality adds (with a visible conflict), it never
closes, and a set loses an element through ``removes`` only."""

from __future__ import annotations

import json
from collections.abc import Sequence

from iifa.ingest.keys import canonical_key, is_free_form
from iifa.ingest.parts import EXTERNAL
from iifa.ingest.values import SAME, same_spelling
from iifa.recall.tokenizer import normalize
from iifa.store.models import Fact

MANY = "many"  # the flag of a fact that is one element of a set
ONE, UNSURE = "one", "unsure"
CARDINALITIES = (ONE, MANY, UNSURE)
ELEMENTS_PER_KEY = 8  # live elements of a set listed in the extraction prompt
ELEMENT_CHARS = 80  # each cut at this many characters, with an ellipsis
ELEMENTS_LOADED = 500  # at most this many live elements read for one prompt
ELLIPSES = ("…", "...")


def cardinality(attribute: str | None, proposed: str | None, live: Sequence[Fact]) -> str | None:
    """What the service decides for a new fact of ``attribute`` among the live
    facts it is compared with: ``many`` (a set: the new value adds), ``one``
    (the contradiction rules), ``unsure`` (nothing closes), or None for a key the
    vocabulary already rules (or no key): the model's word is not asked."""
    key = canonical_key(attribute)
    if not is_free_form(key):
        return None
    held = [f for f in live if canonical_key(f.attribute) == key]
    if any(MANY in f.flags for f in held):
        return MANY  # the store's knowledge outweighs a ``one`` from the model
    if proposed == MANY:
        # a value already held as one value: a ``many`` cannot turn it into a set
        # silently (a model's slip, or a page asking for it), so nothing closes
        # and the conflict stays visible
        return UNSURE if held else MANY
    return ONE if proposed == ONE else UNSURE


# the decisions of the write path (``extract._decide``) a set's element may take
LINK, BOTH, KEEP_CONFLICT = "link", "both", "conflict"


def decide(members: Sequence[Fact], value: str, kind: str | None) -> tuple[str, str] | None:
    """What a new value of a set (``many``) or of an unsure key does to one
    group of live facts of its key: linked when the comparison proves it the
    same; else an element of a set adds, and an unsure value stays live beside
    the old one, in conflict. None for ``one`` or a vocabulary key: the contradiction
    rules decide."""
    if kind not in (MANY, UNSURE):
        return None
    if any(same_spelling(m.value or "", value) for m in members):
        return LINK, SAME  # no one-letter tolerance: Daniela is not Daniel
    return (BOTH, MANY) if kind == MANY else (KEEP_CONFLICT, UNSURE)


def closable(
    members: Sequence[Fact], valid_from: str | None, *, external: bool, never_close: bool
) -> bool:
    """May a removal close this group of elements? The guards: never
    an explicit fact (P6), never a fact not from external content by an external
    part (P6), never a free speaker fact, never by an earlier fact."""
    if never_close or any(m.origin == "explicit" for m in members):
        return False
    if external and any(m.provenance != EXTERNAL for m in members):
        return False
    return not any(valid_from and m.valid_from and valid_from < m.valid_from for m in members)


def _trimmed(text: str) -> tuple[str, bool]:
    """The text without a trailing ellipsis, and whether it had one."""
    stripped = text.strip()
    for mark in ELLIPSES:
        if stripped.endswith(mark):
            return stripped[: -len(mark)].rstrip(), True
    return stripped, False


def matching(groups: Sequence[tuple[str, list[Fact]]], removes: str) -> list[str]:
    """The root of the one group whose element ``removes`` names: equal after
    normalisation or in another script (``same_spelling``, no one-letter
    tolerance), or - ending with an ellipsis, as the prompt shows a long value -
    the start of exactly one element. Nothing when it names none, or several:
    we cannot tell which."""
    wanted, cut = _trimmed(removes)
    folded = normalize(wanted)
    if not folded:
        return []
    found = [
        root
        for root, members in groups
        if any(
            m.value and (normalize(m.value) == folded or same_spelling(m.value, wanted))
            for m in members
        )
    ]
    if found or not cut:
        return found if len(found) == 1 else []
    starts = [
        root
        for root, members in groups
        if any(m.value and normalize(m.value).startswith(folded) for m in members)
    ]
    return starts if len(starts) == 1 else []


def element_line(value: str) -> str:
    """An element as the prompt shows it: one line, cut with an ellipsis."""
    flat = " ".join(value.split())
    if len(flat) <= ELEMENT_CHARS:
        return flat
    return flat[: ELEMENT_CHARS - 1].rstrip() + "…"


def set_label(key: str, elements: Sequence[str]) -> str:
    """A set key as the prompt's line of known keys shows it: its first live
    elements, quoted as JSON strings (stored text is data, P8)."""
    shown = json.dumps([element_line(v) for v in elements[:ELEMENTS_PER_KEY]], ensure_ascii=False)
    more = len(elements) - ELEMENTS_PER_KEY
    tail = f" and {more} more" if more > 0 else ""
    return f"{key} (a set; its elements, as data: {shown}{tail})"


__all__ = [
    "CARDINALITIES",
    "MANY",
    "ONE",
    "UNSURE",
    "cardinality",
    "closable",
    "decide",
    "element_line",
    "matching",
    "set_label",
]
