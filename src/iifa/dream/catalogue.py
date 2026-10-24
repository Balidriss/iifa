"""The closed catalogue of dream operations and the strict schema of the dream
model's answer (SPEC 5.6 rule 1, P8).

The model proposes, the service applies. An answer is ONE JSON object,
``{"operations": [...]}``; each item is validated alone against a closed
pydantic model of its operation (no extra key, bounded strings, ids of the
right kind). Anything else is **rejected and counted, never written**, and the
valid items beside it are still considered. This is the prompt-injection wall:
a memory that says "delete everything" can at worst make the model propose
something this schema, then the service's checks, refuse.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from iifa.ingest.schema import normalize_edge_type

# SPEC 5.6, in the order of its table
OPERATIONS: tuple[str, ...] = (
    "refresh_summary",
    "decay_edges",
    "reembed",
    "propose_edge",
    "create_theme",
    "consolidate_facts",
    "merge_nodes",
    "supersede_fact",
    "flag_conflict",
)
SERVICE_OPERATIONS: tuple[str, ...] = ("decay_edges", "reembed")  # run without a model
MODEL_OPERATIONS: tuple[str, ...] = tuple(o for o in OPERATIONS if o not in SERVICE_OPERATIONS)
# a merge is proposed only by the adjudication of a pair, never in a unit's answer
ADJUDICATED_OPERATIONS: tuple[str, ...] = ("merge_nodes",)
UNIT_OPERATIONS: tuple[str, ...] = tuple(
    o for o in MODEL_OPERATIONS if o not in ADJUDICATED_OPERATIONS
)
RISK: dict[str, str] = {
    "refresh_summary": "low",
    "decay_edges": "low",
    "reembed": "low",
    "propose_edge": "low",
    "create_theme": "medium",
    "consolidate_facts": "medium",
    "merge_nodes": "high",
    "supersede_fact": "high",
    "flag_conflict": "low",
}

MAX_OPERATIONS = 20  # per answer
MAX_MEMBERS = 20  # of a theme, of a consolidation
SUMMARY_CHARS = 300
NAME_CHARS = 200
STATEMENT_CHARS = 500
REASON_CHARS = 300

# rejection causes, counted in the dream report
OUTSIDE_CATALOGUE = "outside_catalogue"
NOT_MODEL = "not_model"
NOT_REQUESTED = "not_requested"
SCHEMA = "schema"
TOO_MANY = "too_many"
BAD_ANSWER = "bad_answer"
ADJUDICATION_ONLY = "adjudication_only"

_ULID = "[0-9A-HJKMNP-TV-Z]{26}"
NodeId = Annotated[str, StringConstraints(pattern=f"^node_{_ULID}$")]
FactId = Annotated[str, StringConstraints(pattern=f"^fact_{_ULID}$")]


def _flat(value: str) -> str:
    return " ".join(value.split())


class _Operation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    reason: str = Field(default="", max_length=REASON_CHARS)

    @field_validator("reason")
    @classmethod
    def _flat_reason(cls, value: str) -> str:
        return _flat(value)


class RefreshSummary(_Operation):
    op: Literal["refresh_summary"]
    node: NodeId
    summary: str = Field(min_length=1, max_length=SUMMARY_CHARS)

    @field_validator("summary")
    @classmethod
    def _one_line(cls, value: str) -> str:
        flat = _flat(value)
        if not flat:
            raise ValueError("an empty summary")
        return flat


class ProposeEdge(_Operation):
    op: Literal["propose_edge"]
    src: NodeId
    dst: NodeId
    type: str = Field(min_length=1, max_length=60)

    @field_validator("type")
    @classmethod
    def _normalised(cls, value: str) -> str:
        normalised = normalize_edge_type(value)
        if normalised is None:
            raise ValueError("an edge type that does not normalise")
        return normalised


class CreateTheme(_Operation):
    op: Literal["create_theme"]
    name: str = Field(min_length=1, max_length=NAME_CHARS)
    members: list[NodeId] = Field(min_length=2, max_length=MAX_MEMBERS)
    summary: str | None = Field(default=None, max_length=SUMMARY_CHARS)

    @field_validator("name")
    @classmethod
    def _flat_name(cls, value: str) -> str:
        flat = _flat(value)
        if not flat:
            raise ValueError("an empty name")
        return flat

    @field_validator("summary")
    @classmethod
    def _flat_summary(cls, value: str | None) -> str | None:
        return None if value is None else (_flat(value) or None)


class ConsolidateFacts(_Operation):
    op: Literal["consolidate_facts"]
    node: NodeId
    facts: list[FactId] = Field(min_length=2, max_length=MAX_MEMBERS)
    statement: str = Field(min_length=1, max_length=STATEMENT_CHARS)

    @field_validator("statement")
    @classmethod
    def _flat_statement(cls, value: str) -> str:
        flat = _flat(value)
        if not flat:
            raise ValueError("an empty statement")
        return flat


class MergeNodes(_Operation):
    op: Literal["merge_nodes"]
    keep: NodeId
    absorb: NodeId


class SupersedeFact(_Operation):
    op: Literal["supersede_fact"]
    fact: FactId
    by: FactId


class FlagConflict(_Operation):
    op: Literal["flag_conflict"]
    fact: FactId
    with_: FactId = Field(alias="with")


SCHEMAS: dict[str, type[_Operation]] = {
    "refresh_summary": RefreshSummary,
    "propose_edge": ProposeEdge,
    "create_theme": CreateTheme,
    "consolidate_facts": ConsolidateFacts,
    "merge_nodes": MergeNodes,
    "supersede_fact": SupersedeFact,
    "flag_conflict": FlagConflict,
}


class Rejected(ValueError):
    """An item the schema refuses; ``cause`` is what the report counts."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


def parse_operation(item: Any, *, allowed: Collection[str] = MODEL_OPERATIONS) -> dict[str, Any]:
    """One item of an answer, validated: a plain dict with ``op`` and its fields
    (``with`` keeps its JSON name). Raises ``Rejected``."""
    if not isinstance(item, dict) or not isinstance(item.get("op"), str):
        raise Rejected(SCHEMA)
    name = item["op"]
    if name not in OPERATIONS:
        raise Rejected(OUTSIDE_CATALOGUE)
    if name in SERVICE_OPERATIONS:
        raise Rejected(NOT_MODEL)
    if name not in allowed:
        raise Rejected(NOT_REQUESTED)
    try:
        parsed = SCHEMAS[name].model_validate(item)
    except ValidationError as exc:
        raise Rejected(SCHEMA) from exc
    return parsed.model_dump(by_alias=True)


@dataclass
class Parsed:
    operations: list[dict[str, Any]] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)
    error: str | None = None


_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(?P<body>.*?)\n?```\s*$", re.S)


def parse_answer(content: str, *, allowed: Collection[str]) -> Parsed:
    """The dream model's answer, item by item (SPEC 5.6). An answer that is not
    the expected object is a ``bad_answer``: nothing of it is used."""
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    try:
        data = json.loads(text)
    except ValueError:
        return Parsed(error=BAD_ANSWER)
    if not isinstance(data, dict) or set(data) != {"operations"}:
        return Parsed(error=BAD_ANSWER)
    items = data["operations"]
    if not isinstance(items, list):
        return Parsed(error=BAD_ANSWER)
    out = Parsed()
    for item in items:
        if len(out.operations) >= MAX_OPERATIONS:
            out.rejected[TOO_MANY] += 1
            continue
        if isinstance(item, dict) and item.get("op") in ADJUDICATED_OPERATIONS:
            out.rejected[ADJUDICATION_ONLY] += 1
            continue
        try:
            out.operations.append(parse_operation(item, allowed=allowed))
        except Rejected as exc:
            out.rejected[exc.cause] += 1
    return out


# --------------------------------------------------------------------------- the adjudication

SAME, DIFFERENT, UNSURE = "same", "different", "unsure"
VERDICTS: tuple[str, ...] = (SAME, DIFFERENT, UNSURE)


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["same", "different", "unsure"]
    reason: str = Field(default="", max_length=REASON_CHARS)

    @field_validator("reason")
    @classmethod
    def _flat_reason(cls, value: str) -> str:
        return _flat(value)


@dataclass(frozen=True)
class Verdict:
    verdict: str | None
    reason: str = ""
    error: str | None = None


def parse_verdict(content: str) -> Verdict:
    """The adjudication's answer (SPEC 5.6): ``{"verdict": "same" | "different" |
    "unsure", "reason": "..."}`` and nothing else, or a ``bad_answer``."""
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    try:
        parsed = _Verdict.model_validate(json.loads(text))
    except (ValueError, ValidationError):
        return Verdict(None, error=BAD_ANSWER)
    return Verdict(parsed.verdict, parsed.reason)


def named_ids(op: dict[str, Any]) -> list[str]:
    """The record ids an operation names, in field order, once each."""
    out: list[str] = []
    for key in ("node", "src", "dst", "keep", "absorb", "fact", "by", "with"):
        value = op.get(key)
        if isinstance(value, str) and value not in out:
            out.append(value)
    for key in ("members", "facts"):
        for value in op.get(key) or ():
            if value not in out:
                out.append(value)
    return out


__all__ = [
    "ADJUDICATED_OPERATIONS",
    "ADJUDICATION_ONLY",
    "BAD_ANSWER",
    "DIFFERENT",
    "MAX_OPERATIONS",
    "MODEL_OPERATIONS",
    "OPERATIONS",
    "RISK",
    "SAME",
    "SERVICE_OPERATIONS",
    "UNIT_OPERATIONS",
    "UNSURE",
    "VERDICTS",
    "Parsed",
    "Rejected",
    "Verdict",
    "named_ids",
    "parse_answer",
    "parse_operation",
    "parse_verdict",
]
