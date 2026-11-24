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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from iifa.names import plain_name, require_name

NODE_KINDS = (
    "person",
    "project",
    "place",
    "object",
    "organisation",
    "topic",
    "event",
    "task",
    "other",
)
NodeKind = Literal[
    "person", "project", "place", "object", "organisation", "topic", "event", "task", "other"
]

MAX_NODES = 100
MAX_FACTS = 200
MAX_EDGES = 200
MAX_ALIASES = 20
NAME_CHARS = 200
STATEMENT_CHARS = 500
QUOTE_CHARS = 1000
NORM_CHARS = 200  # a value's comparison form (SPEC 5.4)
SUMMARY_CHARS = 300

COOCCURRENCE = "co_occurs"

# SPEC 4.2, intent before content: how a source's words are meant (one frame per
# answer), whether a fact is stated as true (its modality), and how long it holds
FRAMES = ("sincere", "pretend", "example", "irony", "voice", "staged", "unsure")
MODALITIES = ("fact", "question", "hypothesis", "plan", "guess", "proposal", "unsure")
DURATIONS = ("lasting", "task", "observed", "unsure")
Frame = Literal["sincere", "pretend", "example", "irony", "voice", "staged", "unsure"]
Modality = Literal["fact", "question", "hypothesis", "plan", "guess", "proposal", "unsure"]
Duration = Literal["lasting", "task", "observed", "unsure"]

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
    # null: a name the source says exists but does not give (SPEC 5.1)
    aliases: list[str] = Field(default_factory=list, max_length=MAX_ALIASES)
    summary: str = Field(default="", max_length=SUMMARY_CHARS)
    # SPEC 5.1: aliases left out - null (not given), or no name once plain
    _unknown_aliases: int = PrivateAttr(default=0)
    _malformed_aliases: int = PrivateAttr(default=0)

    @model_validator(mode="wrap")
    @classmethod
    def _count_aliases(cls, data: Any, handler: ModelWrapValidatorHandler[ProposedNode]) -> Any:
        if not isinstance(data, dict):
            return handler(data)  # an instance keeps the counts it was built with
        unknown = malformed = 0
        raw = data.get("aliases")
        if isinstance(raw, list):
            unknown = sum(1 for a in raw if a is None)
            malformed = sum(
                1 for a in raw if isinstance(a, str) and a.strip() and plain_name(a) is None
            )
        node = handler(data)
        node._unknown_aliases, node._malformed_aliases = unknown, malformed
        return node

    @property
    def unknown_aliases(self) -> int:
        return self._unknown_aliases

    @property
    def malformed_aliases(self) -> int:
        return self._malformed_aliases

    @field_validator("name")
    @classmethod
    def _plain(cls, value: str) -> str:
        return require_name(value)  # SPEC 5.1

    @field_validator("summary")
    @classmethod
    def _flat(cls, value: str) -> str:
        return _clean_text(value)

    @field_validator("aliases", mode="before")
    @classmethod
    def _aliases(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("aliases are a list")
        if len(value) > MAX_ALIASES:  # the bound holds on what the model wrote
            raise ValueError("at most 20 aliases")
        out: list[str] = []
        for alias in value:
            if alias is None:
                continue  # SPEC 5.1: a name not given is no alias, nor a proposal
            if not isinstance(alias, str):
                raise ValueError("an alias must be a string")
            if len(_clean_text(alias)) > NAME_CHARS:
                raise ValueError("an alias must be at most 200 characters")
            plain = plain_name(alias)  # None: debris, left out and counted
            if plain is not None and plain not in out:
                out.append(plain)
        return out


class ProposedFact(_Item):
    subject: str | None = Field(default=None, max_length=NAME_CHARS)
    # the subject exactly as the source spells it, when that differs from the
    # known name in ``subject`` (SPEC 4.2)
    subject_as_written: str | None = Field(default=None, max_length=NAME_CHARS)
    statement: str = Field(min_length=1, max_length=STATEMENT_CHARS)
    attribute: str | None = Field(default=None, max_length=100)
    value: str | None = Field(default=None, max_length=STATEMENT_CHARS)
    # the value's comparison form (SPEC 5.4): compared, never shown nor searched
    value_norm: str | None = None
    valid_from: str | None = Field(default=None, max_length=40)
    valid_to: str | None = Field(default=None, max_length=40)
    quote: str | None = Field(default=None, max_length=QUOTE_CHARS)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # SPEC 5.4 (sets): one current value, one element of a set, or unsure; and
    # the value of an element this fact removes or corrects
    cardinality: Literal["one", "many", "unsure"] | None = None
    removes: str | None = Field(default=None, max_length=STATEMENT_CHARS)
    # SPEC 4.2: read only when the intent reading is on; None means not given
    modality: Modality | None = None
    duration: Duration | None = None
    episode: bool = False  # the one sincere fact a non-sincere frame may leave

    @field_validator("modality", "duration", mode="before")
    @classmethod
    def _word(cls, value: Any, info: ValidationInfo) -> Any:
        """Folded; a word outside the closed list is read as not given (SPEC
        4.2: a doubt, so a proposal or session state), never as the fact's death."""
        if not isinstance(value, str):
            return value
        word = value.strip().lower()
        allowed = MODALITIES if info.field_name == "modality" else DURATIONS
        return word if word in allowed else None

    @field_validator("statement", "attribute", "value", "removes", "value_norm")
    @classmethod
    def _flat(cls, value: str | None) -> str | None:
        if value is None:
            return None
        flat = _clean_text(value)
        return flat or None

    @field_validator("subject")
    @classmethod
    def _subject(cls, value: str | None) -> str | None:
        if value is None or not _clean_text(value):
            return None
        return require_name(value)  # SPEC 5.1: a subject names a node

    @field_validator("subject_as_written")
    @classmethod
    def _written(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return plain_name(value)  # an optional spelling: debris only drops it

    @field_validator("statement")
    @classmethod
    def _non_empty(cls, value: str | None) -> str:
        if not value:
            raise ValueError("a statement is a sentence")
        return value

    @field_validator("cardinality", mode="before")
    @classmethod
    def _cardinality(cls, value: Any) -> Any:
        """Any other word is no word (unsure, SPEC 5.4): the fact is kept."""
        word = value.strip().lower() if isinstance(value, str) else None
        return word if word in ("one", "many", "unsure") else None

    @field_validator("removes", mode="before")
    @classmethod
    def _removes(cls, value: Any) -> Any:
        """Not a short text: no removal (nothing closes), the fact is kept."""
        if not isinstance(value, str) or len(value) > STATEMENT_CHARS:
            return None
        return value

    @field_validator("value_norm")
    @classmethod
    def _bounded_form(cls, value: str | None) -> str | None:
        # an over-long form is dropped, never the fact it came with
        return value if value is None or len(value) <= NORM_CHARS else None

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


class ProposedActivity(_Item):
    """SPEC 4.2, agent perspective: what the owner asks the agent to do in the
    source, named in the owner's words, and the owner's words that ask it."""

    name: str = Field(min_length=1, max_length=NAME_CHARS)
    quote: str = Field(min_length=1, max_length=QUOTE_CHARS)

    @field_validator("name")
    @classmethod
    def _plain(cls, value: str) -> str:
        return require_name(value)


class ProposedEdge(_Item):
    src: str = Field(min_length=1, max_length=NAME_CHARS)
    dst: str = Field(min_length=1, max_length=NAME_CHARS)
    type: str = Field(min_length=1, max_length=80)
    quote: str | None = Field(default=None, max_length=QUOTE_CHARS)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("src", "dst")
    @classmethod
    def _flat(cls, value: str) -> str:
        if not _clean_text(value):
            raise ValueError("an edge end is a name")
        return require_name(value)

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
    # SPEC 4.2: the frame of the source's words (None: not given), whether it is
    # set to govern the turns that follow, whether this source ends an open one
    frame: str | None = None
    frame_holds: bool = False
    frame_ends: bool = False
    # SPEC 4.2, agent perspective: the activity the owner asks for (None: none
    # given or refused); ``activity_given``: the key was there, valid or null
    activity: ProposedActivity | None = None
    activity_given: bool = False
    # SPEC 4.2: the secrets the model saw in the source, exactly as written
    # (kept by ``iifa.ingest.credentials`` only when the source writes them)
    secrets: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.nodes or self.facts or self.edges)


ANSWER_KEYS = ("nodes", "facts", "edges")
INTENT_KEYS = ("frame", "frame_holds", "frame_ends")
ACTIVITY_KEY = "activity"
SECRETS_KEY = "secrets"
_NO_SECRET = frozenset({"", "none", "null", "n/a", "na", "nil", "-", "aucun", "keine", "ninguno"})
MAX_SECRETS = 5000  # a list only removes from memory: a pasted list of codes fits
SECRET_CHARS = 8192  # a private key block
ADJUDICATION_KEYS = ("decisions",)
VALUE_VERDICT_KEYS = ("verdicts",)
VALUE_VERDICTS = ("same", "different", "unsure")
REPLACEMENT_KEYS = ("replacements",)
REPLACEMENT_VERDICTS = ("same", "replaces", "both", "unsure")
RESTATEMENT_KEYS = ("restatements",)
OPTIONAL_KEYS = (ACTIVITY_KEY, SECRETS_KEY)

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
        if key not in ANSWER_KEYS and key not in INTENT_KEYS and key not in OPTIONAL_KEYS:
            answer.rejected.keys += 1
    _intent(data, answer)
    _secrets(data, answer)
    _activity(data, answer)
    answer.nodes = _items(data.get("nodes"), ProposedNode, MAX_NODES, answer.rejected, "nodes")
    answer.facts = _items(data.get("facts"), ProposedFact, MAX_FACTS, answer.rejected, "facts")
    answer.edges = _items(data.get("edges"), ProposedEdge, MAX_EDGES, answer.rejected, "edges")
    return answer


def _intent(data: dict[str, Any], answer: Answer) -> None:
    """The answer's frame and its two flags (SPEC 4.2). A frame outside the closed
    list, or a flag that is not a boolean, is rejected and counted: the frame is
    then not given (read as ``unsure``), the flag false."""
    frame = data.get("frame")
    if isinstance(frame, str) and frame.strip().lower() in FRAMES:
        answer.frame = frame.strip().lower()
    elif frame is not None:
        answer.rejected.keys += 1
    for name in ("frame_holds", "frame_ends"):
        flag = data.get(name)
        if isinstance(flag, bool):
            setattr(answer, name, flag)
        elif flag is not None:
            answer.rejected.keys += 1


def _secrets(data: dict[str, Any], answer: Answer) -> None:
    """The secrets the answer lists (SPEC 4.2): strings, at most ``MAX_SECRETS``
    of at most ``SECRET_CHARS`` characters. Anything else is rejected and
    counted; a valid list only removes words from memory, never adds any."""
    raw = data.get(SECRETS_KEY)
    if raw is None:
        return
    if isinstance(raw, str):
        if raw.strip().casefold() in _NO_SECRET:
            return  # "none": the model's way of writing []
        raw = [raw]  # one secret written without its list
    if not isinstance(raw, list):
        answer.rejected.keys += 1
        return
    for item in raw[:MAX_SECRETS]:
        if isinstance(item, int) and not isinstance(item, bool):
            item = str(item)  # a code written as a number  # noqa: PLW2901
        if isinstance(item, str) and item.strip().casefold() in _NO_SECRET:
            continue  # ["none"]: the model's way of writing []
        if isinstance(item, str) and 0 < len(item.strip()) <= SECRET_CHARS:
            answer.secrets.append(item.strip())
        else:
            answer.rejected.keys += 1
    answer.rejected.keys += max(0, len(raw) - MAX_SECRETS)


def _activity(data: dict[str, Any], answer: Answer) -> None:
    """The activity the answer names (SPEC 4.2). Null is no activity; anything
    that is not a valid ``{"name", "quote"}`` object is rejected and counted.
    Whether the perspective asks for it, and whether the source grounds it, is
    the writer's to check."""
    if ACTIVITY_KEY not in data:
        return
    raw = data[ACTIVITY_KEY]
    try:
        answer.activity = None if raw is None else ProposedActivity.model_validate(raw)
    except ValidationError:
        answer.rejected.keys += 1
        return
    answer.activity_given = True  # valid or null: the writer decides on it


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


def parse_restatement_verdicts(content: str, pairs: int) -> tuple[dict[int, str], int]:
    """``{"restatements": [{"pair": i, "verdict": "same" | "different" |
    "unsure"}]}`` -> ``{i: verdict}`` plus the rejected parts (SPEC 5.4: a value
    told again within one language). Anything unsettled or invalid is
    ``unsure``, which leaves the supersession as it would be without a model."""
    return _parse_verdicts(content, pairs, "restatements", VALUE_VERDICTS)


def parse_rule_verdicts(content: str, pairs: int) -> tuple[dict[int, str], int]:
    """``{"rules": [{"pair": i, "verdict": "same" | "different" | "unsure"}]}``
    -> ``{i: verdict}`` plus the rejected parts (SPEC 4.2: a rule of the self
    said again in other words). Anything unsettled or invalid is ``unsure``:
    the rule is asked about, the safe default."""
    return _parse_verdicts(content, pairs, "rules", VALUE_VERDICTS)


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
    "ACTIVITY_KEY",
    "ADJUDICATION_KEYS",
    "ANSWER_KEYS",
    "COOCCURRENCE",
    "EDGE_VOCABULARY",
    "MAX_EDGES",
    "MAX_FACTS",
    "MAX_NODES",
    "NODE_KINDS",
    "Answer",
    "ProposedActivity",
    "ProposedEdge",
    "ProposedFact",
    "ProposedNode",
    "Rejections",
    "extract_object",
    "normalize_edge_type",
    "parse_adjudication",
    "parse_answer",
    "parse_replacement_verdicts",
    "parse_restatement_verdicts",
    "parse_rule_verdicts",
    "parse_value_verdicts",
]
