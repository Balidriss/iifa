"""Attribute keys (SPEC 5.4): a contradiction is only seen when two facts carry
the same key, so keys are steered (a base vocabulary in the prompt) and aligned
(a canonical form and a table of unambiguous variants here, near free-form keys
by embeddings at tier 3 in the extractor). The rule throughout: when we cannot
tell, we do not supersede - closing a fact that is still true is worse than
missing a contradiction. Keys are identifiers, not memory text: they are
language-neutral, whatever the language of the statement (P12 is about
statements and values)."""

from __future__ import annotations

import re
import unicodedata

# One name for each aspect that holds one current value. There is no
# ``location``: told "home only", a model still keyed offices ``location``.
BASE_KEYS: tuple[str, ...] = ("home", "workplace", "role", "employer", "team")
# families take a topic: "prefers tea" must not replace "prefers short answers",
# "married" must not replace "on sick leave"; a bare family key never supersedes
FAMILIES: tuple[str, ...] = ("preference", "status", "date")

VOCABULARY_TEXT = (
    "home (where the subject lives), workplace (where they work: an office, a site, a "
    "city), role (job, occupation), employer, team, preference.<topic> (e.g. "
    "preference.drink, preference.answers: one key per topic), status.<topic> (e.g. "
    "status.marital, status.work), date.<event> (e.g. date.birth, date.deadline)"
)

# Unambiguous variants only. Left out on purpose, as free-form keys that never
# close a vocabulary one: location, city (a home or a work city), work, title,
# position (anything), works_at (a place or an employer), office, based_in.
_SYNONYMS: dict[str, str] = {
    **dict.fromkeys(
        (
            "lives",
            "lives_in",
            "living_in",
            "home_city",
            "residence",
            "place_of_residence",
            "address",
            "home_address",
            "moved_to",
        ),
        "home",
    ),
    **dict.fromkeys(
        ("work_location", "place_of_work", "work_place", "office_location"), "workplace"
    ),
    **dict.fromkeys(
        ("job", "job_title", "occupation", "profession", "works_as", "job_role"), "role"
    ),
    **dict.fromkeys(("works_for", "company", "organisation", "organization"), "employer"),
    **dict.fromkeys(("prefers", "preferences", "likes"), "preference"),
}
_TOPIC_BEFORE = re.compile(r"^(?:preference|preferred|favorite|favourite)_(?P<topic>[a-z0-9_]+)$")
_TOPIC_AFTER = re.compile(r"^(?P<topic>[a-z0-9_]+)_preference$")
_STATUS_AFTER = re.compile(r"^(?P<topic>[a-z0-9_]+)_status$")
_DATE_BEFORE = re.compile(r"^date_of_(?P<event>[a-z0-9_]+)$")
_DATE_AFTER = re.compile(r"^(?P<event>[a-z0-9_]+)_date$")
_SEPARATORS = re.compile(r"[^\w.]+")  # letters of every script kept


def _fold(raw: str) -> str:
    """Lower case, diacritics removed (``rôle`` -> ``role``); spaces, hyphens and
    punctuation to ``_``; dots kept (families); letters of every script kept."""
    decomposed = unicodedata.normalize("NFKD", raw.strip().lower())
    bare = "".join(c for c in decomposed if not unicodedata.combining(c))
    folded = _SEPARATORS.sub("_", bare)
    folded = re.sub(r"_*\._*", ".", folded)
    return folded.strip("_.")


def canonical_key(raw: str | None) -> str | None:
    """The canonical form of a key, or None when there is none."""
    if raw is None:
        return None
    key = _fold(raw)
    if not key or not re.search(r"[^\W_]", key):
        return None
    if "." in key:
        family, _, topic = key.partition(".")
        family = _SYNONYMS.get(family, family)
        return f"{family}.{topic.replace('.', '_')}" if topic else family
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    for pattern, family in (
        (_TOPIC_BEFORE, "preference"),
        (_TOPIC_AFTER, "preference"),
        (_STATUS_AFTER, "status"),
        (_DATE_BEFORE, "date"),
        (_DATE_AFTER, "date"),
    ):
        match = pattern.match(key)
        if match:
            return f"{family}.{match.group(1)}"
    return key


def family_of(key: str | None) -> str | None:
    """``preference`` for ``preference.drink`` and for a bare ``preference``; None
    for a key outside the families."""
    if key is None:
        return None
    head = key.partition(".")[0]
    return head if head in FAMILIES else None


def supersedable(key: str | None) -> bool:
    """May a new value of this (canonical) key close an old one? Not without a key,
    and not for a bare family key: every preference would share it."""
    return key is not None and key not in FAMILIES


def is_free_form(key: str | None) -> bool:
    """A key outside the vocabulary: neither a base key nor a family key. Only
    these are aligned by embeddings (SPEC 5.4 step 4)."""
    return key is not None and key not in BASE_KEYS and family_of(key) is None


# keys whose values replace each other (SPEC 5.4): the only ones a changed value
# in another language may be adjudicated for; never role or team, which may
# hold several values at once
SINGLE_VALUED: tuple[str, ...] = ("home", "workplace", "employer")
SINGLE_VALUED_FAMILIES: tuple[str, ...] = ("status", "date")


# SPEC 5.4: aspects that may hold several values at once; across languages
# their adjudication asks "replaces, or both true?"
MULTI_VALUED: tuple[str, ...] = ("role", "team")
MULTI_VALUED_FAMILIES: tuple[str, ...] = ("preference",)


def multi_valued(key: str | None) -> bool:
    return key in MULTI_VALUED or (supersedable(key) and family_of(key) in MULTI_VALUED_FAMILIES)


def single_valued(key: str | None) -> bool:
    return key in SINGLE_VALUED or (supersedable(key) and family_of(key) in SINGLE_VALUED_FAMILIES)


def same_key(a: str | None, b: str | None) -> bool:
    """Do two keys name the same aspect (compared canonically on both sides)?"""
    left, right = canonical_key(a), canonical_key(b)
    return left is not None and left == right


def key_text(key: str) -> str:
    """What is embedded for a key: its words (``commute_mode`` -> ``commute mode``);
    for a family key, its topic only (``preference.drink`` -> ``drink``)."""
    if family_of(key) is not None:
        key = key.partition(".")[2]
    return key.replace(".", " ").replace("_", " ")


__all__ = [
    "BASE_KEYS",
    "FAMILIES",
    "VOCABULARY_TEXT",
    "canonical_key",
    "family_of",
    "is_free_form",
    "key_text",
    "same_key",
    "single_valued",
    "supersedable",
]
