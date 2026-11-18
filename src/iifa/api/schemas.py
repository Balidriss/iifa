"""Request and response bodies. Every content-returning call accepts a view
(``scope`` | ``scopes`` | ``subtree``), ``budget_tokens`` and ``exclude_ids``."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from iifa.api.paging import MAX_CURSOR_CHARS
from iifa.names import require_name
from iifa.scopes import ScopeError, validate_scope
from iifa.store.corrections import KIND_MAX_CHARS
from iifa.store.repo import NAME_MAX_CHARS

Scope = Field(min_length=1, max_length=512)


def _check_scope(value: str) -> str:
    try:
        return validate_scope(value)
    except ScopeError as exc:
        raise ValueError(str(exc)) from exc


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ViewMixin(Strict):
    scope: str | None = Field(
        default=None, description="This scope and its ancestors (default mode)."
    )
    scopes: list[str] | None = Field(
        default=None, min_length=1, description="Exactly these scopes."
    )
    subtree: str | None = Field(
        default=None, description="A scope and everything below it (admin)."
    )

    @field_validator("scope", "subtree")
    @classmethod
    def _valid_scope(cls, value: str | None) -> str | None:
        return None if value is None else _check_scope(value)

    @field_validator("scopes")
    @classmethod
    def _valid_scopes(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else [_check_scope(v) for v in value]


class BudgetMixin(Strict):
    budget_tokens: int | None = Field(default=None, ge=0)


class WriteScopes(Strict):
    scope: str | None = None
    scopes: list[str] | None = Field(default=None, min_length=1)

    @field_validator("scope")
    @classmethod
    def _valid_scope(cls, value: str | None) -> str | None:
        return None if value is None else _check_scope(value)

    @field_validator("scopes")
    @classmethod
    def _valid_scopes(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else [_check_scope(v) for v in value]

    def target_scopes(self) -> list[str]:
        if self.scopes and self.scope:
            raise ValueError("give scope or scopes, not both")
        if self.scopes:
            return list(dict.fromkeys(self.scopes))
        if self.scope:
            return [self.scope]
        raise ValueError("a write needs scope or scopes")


class QueryVectorIn(Strict):
    """A query vector the harness already computed (SPEC 6.1): used only when it
    carries the configured embedding model and the stored dimension."""

    model: str = Field(min_length=1, max_length=200)
    values: list[float] = Field(min_length=1, max_length=8192)


class RecallRequest(ViewMixin, BudgetMixin):
    query: str = Field(min_length=1, max_length=8000)
    exclude_ids: list[str] = Field(default_factory=list, max_length=2000)
    query_vector: QueryVectorIn | None = None


class ExpandRequest(ViewMixin, BudgetMixin):
    ids: list[str] = Field(min_length=1, max_length=500)
    level: Literal[1, 2] = 1
    cursor: str | None = None


NameText = Annotated[str, StringConstraints(min_length=1, max_length=NAME_MAX_CHARS)]


class NodeRef(Strict):
    name: NameText
    kind: str = Field(default="other", min_length=1, max_length=KIND_MAX_CHARS)
    aliases: list[NameText] = Field(default_factory=list, max_length=50)
    summary: str = Field(default="", max_length=1000)
    external_id: str | None = Field(default=None, max_length=200)

    @field_validator("name")
    @classmethod
    def _plain_name(cls, value: str) -> str:
        return require_name(value)  # SPEC 5.1: Markdown stripped, debris a 422

    @field_validator("aliases")
    @classmethod
    def _plain_aliases(cls, value: list[str]) -> list[str]:
        return [require_name(alias) for alias in value if alias.strip()]


class RememberRequest(WriteScopes):
    statement: str = Field(min_length=1, max_length=4000)
    node: NodeRef | None = None
    node_id: str | None = None
    attribute: str | None = Field(default=None, max_length=200)
    value: str | None = Field(default=None, max_length=2000)
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    pinned: bool = False
    external_id: str | None = Field(default=None, max_length=200)
    source_reference: str | None = Field(default=None, max_length=500)
    about_self: bool = Field(
        default=False,
        description="A fact about the self of the target scopes (SPEC 8.3, agent perspective).",
    )


class LinkRequest(WriteScopes):
    src: str
    dst: str
    type: str = Field(min_length=1, max_length=60)
    weight: float = Field(default=1.0, gt=0.0, le=10.0)
    external_id: str | None = Field(default=None, max_length=200)


class RecordRequest(Strict):
    id: str


class ForgetRequest(Strict):
    id: str


class FactPatch(Strict):
    node_id: str | None = None
    detach: bool = False
    pinned: bool | None = None
    scopes: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description="Replace the fact's scope set (SPEC 4.6): the token must own every"
        " scope added and every scope removed.",
    )

    @field_validator("scopes")
    @classmethod
    def _valid_scopes(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else list(dict.fromkeys(_check_scope(v) for v in value))


class ProposalDecision(Strict):
    decision: Literal["accept", "reject"]


class EdgeDecision(Strict):
    """A person's decision on a proposed edge (SPEC 5.7); ``rev`` is the edge's
    revision the author read."""

    decision: Literal["confirm", "reject"]
    rev: int | None = Field(default=None, ge=0)


class NodePatch(Strict):
    """A person's correction of a node (SPEC 5.7): rename it, change its kind,
    add and remove aliases; ``rev`` is the revision the author read."""

    name: NameText | None = None
    keep_old_name: bool = Field(
        default=True, description="A rename keeps the old name as an alias unless false."
    )
    kind: str | None = Field(default=None, min_length=1, max_length=KIND_MAX_CHARS)
    add_aliases: list[NameText] = Field(default_factory=list, max_length=50)
    remove_aliases: list[NameText] = Field(default_factory=list, max_length=50)
    rev: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _changes_something(self) -> NodePatch:
        if (
            self.name is None
            and self.kind is None
            and not (self.add_aliases or self.remove_aliases)
        ):
            raise ValueError("give name, kind, add_aliases or remove_aliases")
        return self


class NodeMerge(Strict):
    """Two nodes a person knows are one (SPEC 5.7): ``absorb`` is merged into
    ``keep``; ``revs`` (node id -> rev) are the revisions the author read."""

    keep: str = Field(min_length=1, max_length=64)
    absorb: str = Field(min_length=1, max_length=64)
    revs: dict[str, int] | None = Field(default=None, max_length=2)


class ProbeIn(Strict):
    query: str = Field(min_length=1, max_length=2000)
    expected: list[Annotated[str, StringConstraints(min_length=1, max_length=64)]] = Field(
        min_length=1, max_length=50, description="Ids of the facts or nodes that answer it."
    )


class ProbeSetCreate(Strict):
    """A probe set (SPEC 11.2): questions and the records that answer them, in one scope."""

    scope: str
    name: str = Field(min_length=1, max_length=200)
    budget_tokens: int | None = Field(default=None, ge=1)
    probes: list[ProbeIn] = Field(min_length=1, max_length=500)

    @field_validator("scope")
    @classmethod
    def _valid_scope(cls, value: str) -> str:
        return _check_scope(value)


class RecordResponse(BaseModel):
    id: str
    kind: str
    scopes: list[str]
    external_id: str | None = None


class RememberResponse(BaseModel):
    fact: dict[str, Any]
    node: dict[str, Any] | None
    source_id: str | None
    created_node: bool
    # SPEC 8.3: about the self, the same thing said again: nothing was written
    echo: bool = False


# --------------------------------------------------------------------------- lifecycle (SPEC 8.1)

TURN_TEXT_CHARS = 100_000
PRECOMPACT_TOTAL_CHARS = 400_000
ToolSummary = Annotated[str, StringConstraints(max_length=4000)]


class SessionStartRequest(ViewMixin, BudgetMixin):
    """The profile block for a scope: pinned facts and the most central nodes."""

    session_id: str | None = Field(default=None, max_length=200)
    exclude_ids: list[str] = Field(default_factory=list, max_length=2000)


class ExternalPart(Strict):
    """Content from outside the conversation, quoted in a turn (P6): a tool's
    result, a web page, a document. Never a fact about the owner."""

    kind: Literal["tool", "web", "document", "other"] = "other"
    text: str = Field(max_length=TURN_TEXT_CHARS)
    ref: str | None = Field(default=None, max_length=2000)
    vouched: bool = Field(
        default=False,
        description="The host's statement that the owner gave this content or asked for it:"
        " the agent perspective may learn from it (SPEC 4.2). Never covers the owner's facts.",
    )


class TurnRequest(WriteScopes):
    """One turn to record. Enqueued only; nothing is extracted on this call.
    ``user`` is the owner's words; ``tool_summaries`` and ``external`` are
    content from outside the conversation (P6)."""

    session_id: str | None = Field(default=None, max_length=200)
    user: str = Field(default="", max_length=TURN_TEXT_CHARS)
    assistant: str = Field(default="", max_length=TURN_TEXT_CHARS)
    tool_summaries: list[ToolSummary] = Field(default_factory=list, max_length=50)
    external: list[ExternalPart] = Field(default_factory=list, max_length=20)
    occurred_at: str | None = Field(default=None, max_length=40)
    external_id: str | None = Field(default=None, max_length=200)
    # SPEC 8.1: the host's statement that this turn is not to be remembered (a
    # test started inside a real session): nothing of it is written, not even queued
    record: bool = True

    @model_validator(mode="after")
    def _something_to_record(self) -> TurnRequest:
        if not (
            self.user.strip()
            or self.assistant.strip()
            or any(self.tool_summaries)
            or any(part.text.strip() for part in self.external)
        ):
            raise ValueError(
                "a turn needs user text, assistant text, a tool summary or an external part"
            )
        return self


class Message(Strict):
    role: str = Field(min_length=1, max_length=20)
    content: str = Field(max_length=TURN_TEXT_CHARS)


class PrecompactRequest(WriteScopes, BudgetMixin):
    """The messages a harness is about to drop, and a budget for what must survive."""

    session_id: str | None = Field(default=None, max_length=200)
    trigger: str | None = Field(default=None, max_length=40)
    messages: list[Message] = Field(default_factory=list, max_length=500)
    exclude_ids: list[str] = Field(default_factory=list, max_length=2000)

    @model_validator(mode="after")
    def _bounded(self) -> PrecompactRequest:
        if sum(len(m.content) for m in self.messages) > PRECOMPACT_TOTAL_CHARS:
            raise ValueError(f"messages exceed {PRECOMPACT_TOTAL_CHARS} characters in total")
        return self


class SessionEndRequest(WriteScopes):
    session_id: str | None = Field(default=None, max_length=200)
    reason: str | None = Field(default=None, max_length=40)


# --------------------------------------------------------------------------- operations (SPEC 8.4)

BackgroundMode = Literal["auto", "paused", "on_demand"]


class Concurrency(Strict):
    chat: int | None = Field(default=None, ge=1, le=64)
    embeddings: int | None = Field(default=None, ge=1, le=64)


class BackgroundUpdate(Strict):
    mode: BackgroundMode | None = None
    concurrency: Concurrency | None = None


class RunBudget(Strict):
    calls: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    seconds: float | None = Field(default=None, ge=0)


class RunRequest(Strict):
    budget: RunBudget | None = None
    max_jobs: int | None = Field(default=None, ge=1)
    # false: answer 202 with the run id at once; follow with GET /background/run/{id}
    wait: bool = True


class YieldRequest(Strict):
    timeout_seconds: float | None = Field(default=None, gt=0, le=3600)


# --------------------------------------------------------------------------- reading (SPEC 8.2)

DateText = Annotated[str, StringConstraints(min_length=4, max_length=40, strip_whitespace=True)]


class TimelineRequest(ViewMixin, BudgetMixin):
    """Fact changes of one node, or of the whole view, inside a date range."""

    node_id: str | None = Field(default=None, max_length=64)
    since: DateText | None = None
    until: DateText | None = None
    limit: int = Field(default=200, ge=1, le=1000)
    cursor: str | None = Field(default=None, max_length=MAX_CURSOR_CHARS)


class AtRequest(ViewMixin, BudgetMixin):
    """What was known about a node at a date (P7)."""

    node_id: str = Field(min_length=1, max_length=64)
    date: DateText


class PathRequest(ViewMixin, BudgetMixin):
    """How two nodes are related: a bounded search over visible, live edges."""

    src: str = Field(min_length=1, max_length=64)
    dst: str = Field(min_length=1, max_length=64)
    max_hops: int = Field(default=3, ge=1, le=6)


# --------------------------------------------------------------------------- dream (SPEC 8.6)

DreamOperation = Literal[
    "refresh_summary",
    "decay_edges",
    "reembed",
    "propose_edge",
    "create_theme",
    "consolidate_facts",
    "merge_nodes",
    "supersede_fact",
    "flag_conflict",
]
ReviewAction = Literal["auto", "review"]


class DreamBudget(Strict):
    calls: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    seconds: float | None = Field(default=None, ge=0)
    operations: int | None = Field(default=None, ge=0)


class ReviewPolicyIn(Strict):
    low: ReviewAction | None = None
    medium: ReviewAction | None = None
    high: ReviewAction | None = None


class DreamStart(Strict):
    """Start a dream over one scope (SPEC 5.6): which operations, what budget,
    a dry run or not, a review policy that may only be stricter than the
    configured one."""

    scope: str
    operations: list[DreamOperation] | None = Field(default=None, min_length=1)
    budget: DreamBudget | None = None
    dry_run: bool = False
    review_policy: ReviewPolicyIn | None = None

    @field_validator("scope")
    @classmethod
    def _valid_scope(cls, value: str) -> str:
        return _check_scope(value)


class DreamUndo(Strict):
    op_id: str | None = Field(default=None, max_length=64)


class ReviewDecision(Strict):
    decision: Literal["accept", "reject"]
    keep: str | None = Field(
        default=None,
        max_length=64,
        description="Accepting a merge proposal: the node to keep (default: the one with"
        " more live facts, else the older; required when the two kinds differ).",
    )
