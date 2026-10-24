"""Plain records as they come out of the store. Every record carries the set of
scopes it belongs to and an optional integrator-supplied ``external_id``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    name: str
    aliases: tuple[str, ...]
    summary: str
    scopes: tuple[str, ...]
    external_id: str | None
    created_at: str
    updated_at: str
    archived_at: str | None
    language: str | None = None
    flags: tuple[str, ...] = ()
    merged_into: str | None = None  # the node a dream merged it into (SPEC 5.6)
    rev: int = 0  # grows with every change to the record (SPEC 4.4)


@dataclass(frozen=True)
class Fact:
    id: str
    node_id: str | None
    statement: str
    attribute: str | None
    value: str | None
    valid_from: str | None
    valid_to: str | None
    origin: str
    confidence: float
    source_id: str | None
    excerpt_start: int | None
    excerpt_end: int | None
    superseded_by: str | None
    pinned: bool
    flags: tuple[str, ...]
    scopes: tuple[str, ...]
    external_id: str | None
    created_at: str
    updated_at: str
    archived_at: str | None
    language: str | None = None
    conflicts_with: str | None = None
    same_as: str | None = None  # the first fact of its equivalence group (SPEC 5.4)
    provenance: str | None = None  # owner | assistant | external: its part of the source (P6)
    proposal_id: str | None = None  # the fact proposal it was accepted from (SPEC 4.4b)
    consolidated_into: str | None = None  # the fact a dream consolidated it into (SPEC 5.6)
    rev: int = 0  # grows with every change to the record (SPEC 4.4)


@dataclass(frozen=True)
class Edge:
    id: str
    src: str
    dst: str
    type: str
    weight: float
    status: str
    origin: str
    source_id: str | None
    last_used_at: str | None
    use_count: int
    valid_from: str | None
    valid_to: str | None
    scopes: tuple[str, ...]
    external_id: str | None
    created_at: str
    updated_at: str
    archived_at: str | None
    evidence: tuple[str, ...] = ()
    rev: int = 0  # grows with every change to the record (SPEC 4.4)


@dataclass(frozen=True)
class Source:
    id: str
    kind: str
    text: str | None
    text_sha256: str
    reference: str | None
    scopes: tuple[str, ...]
    external_id: str | None
    created_at: str
    session_id: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class Proposal:
    """Something the write path would not decide alone (SPEC 4.4b): an alias the
    model attached to a node without it appearing in the source, a merge, or a
    fact about the owner that was not the owner's own words (``payload`` holds
    the rest of that fact; ``node_id`` may then be None)."""

    id: str
    kind: str
    node_id: str | None
    value: str
    source_id: str | None
    excerpt_start: int | None
    excerpt_end: int | None
    confidence: float | None
    status: str
    scopes: tuple[str, ...]
    created_at: str
    decided_at: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    applied_id: str | None = None  # the fact an accepted fact proposal wrote


@dataclass(frozen=True)
class Probe:
    """One question of a probe set and the ids of the records that answer it."""

    id: str
    query: str
    expected: tuple[str, ...]


@dataclass(frozen=True)
class ProbeSet:
    """Stored questions that measure recall in one scope (SPEC 4.7, 11.2)."""

    id: str
    scope: str
    name: str
    budget_tokens: int
    created_at: str
    probes: tuple[Probe, ...] = ()
