"""The strict schema an extraction answer must match (SPEC 6, tier 2; P8).

The model proposes, the service applies: an answer is a JSON object with three
lists, ``nodes``, ``facts`` and ``edges``, whose items are validated one by one
against closed pydantic models (no extra keys, bounded strings, enumerated node
kinds, edge types normalised to a small vocabulary). Anything else, an unknown
top-level key, an item with a field the schema does not know, a value out of
range, is **rejected and counted, never written**. This is the same wall dream
rule 1 builds: stored text may carry hostile instructions, and the worst they
can do is make the model propose something the schema refuses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

NODE_KINDS = ("person", "project", "place", "object", "organisation", "topic", "event", "other")
NodeKind = Literal[
    "person", "project", "place", "object", "organisation", "topic", "event", "other"
]

MAX_NODES = 100
MAX_FACTS = 200
MAX_EDGES = 200
NAME_CHARS = 200
STATEMENT_CHARS = 500
QUOTE_CHARS = 1000
SUMMARY_CHARS = 300

COOCCURRENCE = "co_occurs"

# Relation phrases a model tends to write, mapped to the vocabulary (SPEC 4.3).
EDGE_VOCABULARY: dict[str, str] = {
    "works on": "works_on",
    "work on": "works_on",
    "working on": "works_on",
    "works at": "works_at",
    "works for": "works_at",
    "employed by": "works_at",
    "works with": "works_with",
    "collaborates with": "works_with",
    "colleague of": "works_with",
    "lives in": "lives_in",
    "located in": "located_in",
    "based in": "located_in",
    "member of": "member_of",
    "part of": "part_of",
    "belongs to": "part_of",
    "manages": "manages",
    "leads": "manages",
    "reports to": "reports_to",
    "knows": "knows",
    "married to": "married_to",
    "sibling of": "sibling_of",
    "sister of": "sibling_of",
    "brother of": "sibling_of",
    "parent of": "parent_of",
    "child of": "child_of",
    "friend of": "friend_of",
    "owns": "owns",
    "uses": "uses",
    "prefers": "prefers",
    "about": "about",
    "related to": "related_to",
    "co occurs": COOCCURRENCE,
}

_TYPE = re.compile(r"^[a-z][a-z0-9_]{0,59}$")
_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2}([T ].*)?)?$")


def normalize_edge_type(raw: str) -> str | None:
    """``"Works On"`` -> ``works_on``; a phrase from the vocabulary maps to its
    canonical type; anything that does not normalise to a short identifier is
    refused (``None``)."""
    phrase = " ".join(re.sub(r"[^0-9a-z]+", " ", raw.strip().lower()).split())
    if not phrase:
        return None
    canonical = EDGE_VOCABULARY.get(phrase)
    if canonical is None:
        canonical = phrase.replace(" ", "_")
    return canonical if _TYPE.match(canonical) else None


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _clean_text(value: str) -> str:
    return " ".join(value.split())


class ProposedNode(_Item):
    name: str = Field(min_length=1, max_length=NAME_CHARS)
    kind: NodeKind = "other"
    aliases: list[str] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=SUMMARY_CHARS)

    @field_validator("name", "summary")
    @classmethod
    def _flat(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for alias in value:
            if not isinstance(alias, str):
                raise ValueError("an alias must be a string")
            flat = _clean_text(alias)
            if len(flat) > NAME_CHARS:
                raise ValueError("an alias must be at most 200 characters")
            if flat and flat not in out:
                out.append(flat)
        return out


class ProposedFact(_Item):
    subject: str | None = Field(default=None, max_length=NAME_CHARS)
    # the subject exactly as the source spells it, when that differs from the
    # known name in ``subject`` (SPEC 4.2)
    subject_as_written: str | None = Field(default=None, max_length=NAME_CHARS)
    statement: str = Field(min_length=1, max_length=STATEMENT_CHARS)
    attribute: str | None = Field(default=None, max_length=100)
    value: str | None = Field(default=None, max_length=STATEMENT_CHARS)
    valid_from: str | None = Field(default=None, max_length=40)
    valid_to: str | None = Field(default=None, max_length=40)
    quote: str | None = Field(default=None, max_length=QUOTE_CHARS)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("statement", "subject", "subject_as_written", "attribute", "value")
    @classmethod
    def _flat(cls, value: str | None) -> str | None:
        if value is None:
            return None
        flat = _clean_text(value)
        return flat or None

    @field_validator("statement")
    @classmethod
    def _non_empty(cls, value: str | None) -> str:
        if not value:
            raise ValueError("a statement is a sentence")
        return value

    @field_validator("valid_from", "valid_to")
    @classmethod
    def _date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        if not _DATE.match(text):
            raise ValueError("a date is ISO 8601 (YYYY-MM, YYYY-MM-DD or a timestamp)")
        return text


class ProposedEdge(_Item):
    src: str = Field(min_length=1, max_length=NAME_CHARS)
    dst: str = Field(min_length=1, max_length=NAME_CHARS)
    type: str = Field(min_length=1, max_length=80)
    quote: str | None = Field(default=None, max_length=QUOTE_CHARS)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("src", "dst")
    @classmethod
    def _flat(cls, value: str) -> str:
        flat = _clean_text(value)
        if not flat:
            raise ValueError("an edge end is a name")
        return flat

    @field_validator("type")
    @classmethod
    def _vocabulary(cls, value: str) -> str:
        canonical = normalize_edge_type(value)
        if canonical is None:
            raise ValueError("an edge type is a short identifier or a known relation phrase")
        return canonical


@dataclass
class Rejections:
    """What the wall refused, by category. Counts only: never content."""

    answer: int = 0  # answers that were not a JSON object at all
    keys: int = 0  # top-level keys the schema does not know
    nodes: int = 0
    facts: int = 0
    edges: int = 0

    @property
    def total(self) -> int:
        return self.answer + self.keys + self.nodes + self.facts + self.edges

    def as_dict(self) -> dict[str, int]:
        return {
            "answer": self.answer,
            "keys": self.keys,
            "nodes": self.nodes,
            "facts": self.facts,
            "edges": self.edges,
        }

    def add(self, other: Rejections) -> None:
        self.answer += other.answer
        self.keys += other.keys
        self.nodes += other.nodes
        self.facts += other.facts
        self.edges += other.edges


@dataclass
class Answer:
    nodes: list[ProposedNode] = field(default_factory=list)
    facts: list[ProposedFact] = field(default_factory=list)
    edges: list[ProposedEdge] = field(default_factory=list)
    rejected: Rejections = field(default_factory=Rejections)
    usable: bool = True

    @property
    def empty(self) -> bool:
        return not (self.nodes or self.facts or self.edges)


ANSWER_KEYS = ("nodes", "facts", "edges")
ADJUDICATION_KEYS = ("decisions",)
VALUE_VERDICT_KEYS = ("verdicts",)
VALUE_VERDICTS = ("same", "different", "unsure")
REPLACEMENT_KEYS = ("replacements",)
REPLACEMENT_VERDICTS = ("same", "replaces", "both", "unsure")

_THINKING = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"</?(think|thinking|reasoning)>", re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z]*[ \t]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


def _objects(text: str) -> list[dict[str, Any]]:
    """Every JSON object that starts at a ``{`` of ``text`` and parses, in order,
    skipping over the inside of each one found."""
    found: list[dict[str, Any]] = []
    position = text.find("{")
    while position >= 0:
        try:
            value, end = _DECODER.raw_decode(text, position)
        except ValueError:
            position = text.find("{", position + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        position = text.find("{", end)
    return found


def extract_object(content: str, keys: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """The JSON object of a model's answer, with or without JSON mode (SPEC 7.4b).

    Reasoning blocks (``<think>…</think>``) are removed first, so an object a
    model wrote while thinking is never taken for its answer; an unterminated
    one loses only its tag. Then, in order: the whole text, the inside of each
    code fence, every object found in the text. The first object that carries
    one of ``keys`` wins, else the first object at all (the schema then counts
    what it does not know)."""
    text = _THINKING.sub(" ", content)
    text = _THINK_TAG.sub(" ", text).strip()
    candidates: list[dict[str, Any]] = []
    try:
        whole = json.loads(text)
    except ValueError:
        whole = None
    if isinstance(whole, dict):
        candidates.append(whole)
    for fenced in _FENCE.findall(text):
        candidates.extend(_objects(fenced))
    candidates.extend(_objects(text))
    if not candidates:
        return None
    for candidate in candidates:
        if any(key in candidate for key in keys):
            return candidate
    return candidates[0]


def parse_answer(content: str) -> Answer:
    """Validate a model's extraction answer item by item. ``usable`` is False when
    the answer was not a JSON object at all (worth one retry); otherwise every
    invalid part is counted in ``rejected`` and the valid parts are kept."""
    data = extract_object(content, ANSWER_KEYS)
    if data is None:
        return Answer(rejected=Rejections(answer=1), usable=False)
    answer = Answer()
    for key in list(data):
        if key not in ("nodes", "facts", "edges"):
            answer.rejected.keys += 1
    answer.nodes = _items(data.get("nodes"), ProposedNode, MAX_NODES, answer.rejected, "nodes")
    answer.facts = _items(data.get("facts"), ProposedFact, MAX_FACTS, answer.rejected, "facts")
    answer.edges = _items(data.get("edges"), ProposedEdge, MAX_EDGES, answer.rejected, "edges")
    return answer


def _items(raw: Any, model: type[Any], limit: int, rejected: Rejections, name: str) -> list[Any]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        setattr(rejected, name, getattr(rejected, name) + 1)
        return []
    out: list[Any] = []
    for index, item in enumerate(raw):
        if index >= limit or not isinstance(item, dict):
            setattr(rejected, name, getattr(rejected, name) + 1)
            continue
        try:
            out.append(model.model_validate(item))
        except ValidationError:
            setattr(rejected, name, getattr(rejected, name) + 1)
    return out


class AdjudicationDecision(_Item):
    mention: int = Field(ge=0)
    choice: str = Field(min_length=1, max_length=64)


def parse_adjudication(content: str, mentions: int) -> tuple[dict[int, str], int]:
    """``{"decisions": [{"mention": i, "choice": "<node id>" | "new"}]}`` ->
    ``{i: choice}`` plus the number of rejected parts. A mention the answer does
    not decide, or decides with something that is neither a candidate id nor
    ``new``, stays undecided (the caller creates a new node: the safe default)."""
    data = extract_object(content, ADJUDICATION_KEYS)
    if data is None:
        return {}, 1
    rejected = sum(1 for key in data if key != "decisions")
    raw = data.get("decisions")
    if not isinstance(raw, list):
        return {}, rejected + 1
    decisions: dict[int, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            rejected += 1
            continue
        try:
            decision = AdjudicationDecision.model_validate(item)
        except ValidationError:
            rejected += 1
            continue
        if decision.mention >= mentions or decision.mention in decisions:
            rejected += 1
            continue
        decisions[decision.mention] = decision.choice.strip()
    return decisions, rejected


def parse_value_verdicts(content: str, pairs: int) -> tuple[dict[int, str], int]:
    """``{"verdicts": [{"pair": i, "verdict": "same" | "different" | "unsure"}]}``
    -> ``{i: verdict}`` plus the number of rejected parts (SPEC 5.4). A pair the
    answer does not settle, or settles with anything else, is ``unsure``: the
    conflict stays, the safe default."""
    return _parse_verdicts(content, pairs, "verdicts", VALUE_VERDICTS)


def parse_replacement_verdicts(content: str, pairs: int) -> tuple[dict[int, str], int]:
    """``{"replacements": [{"pair": i, "verdict": "same" | "replaces" | "both" |
    "unsure"}]}`` -> ``{i: verdict}`` plus the rejected parts (SPEC 5.4: role,
    team, preferences). Anything unsettled or invalid is ``unsure``."""
    return _parse_verdicts(content, pairs, "replacements", REPLACEMENT_VERDICTS)


def _parse_verdicts(
    content: str, pairs: int, key: str, words: tuple[str, ...]
) -> tuple[dict[int, str], int]:
    data = extract_object(content, (key,))
    out = dict.fromkeys(range(pairs), "unsure")
    if data is None:
        return out, 1
    rejected = sum(1 for name in data if name != key)
    raw = data.get(key)
    if not isinstance(raw, list):
        return out, rejected + 1
    settled: set[int] = set()
    for item in raw:
        pair, word = (
            (item.get("pair"), item.get("verdict")) if isinstance(item, dict) else (None, None)
        )
        if (
            not isinstance(pair, int)
            or isinstance(pair, bool)
            or not 0 <= pair < pairs
            or pair in settled
            or word not in words
        ):
            rejected += 1
            continue
        settled.add(pair)
        out[pair] = str(word)
    return out, rejected


__all__ = [
    "ADJUDICATION_KEYS",
    "ANSWER_KEYS",
    "COOCCURRENCE",
    "EDGE_VOCABULARY",
    "MAX_EDGES",
    "MAX_FACTS",
    "MAX_NODES",
    "NODE_KINDS",
    "Answer",
    "ProposedEdge",
    "ProposedFact",
    "ProposedNode",
    "Rejections",
    "extract_object",
    "normalize_edge_type",
    "parse_adjudication",
    "parse_answer",
    "parse_value_verdicts",
]
