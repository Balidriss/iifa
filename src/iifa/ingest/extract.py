"""Extraction: from a source's text to nodes, facts and edges, written with their
provenance (P6), in the language of the source (P12), superseding what they
contradict (P7), and never a byte the schema did not admit (P8).

The order of work matters for P11 and for restarts: every model call happens
**first** (extraction per chunk, one batched adjudication), then everything is
written in **one transaction**. A run whose budget runs out mid-job releases
the job untouched, and nothing half-written is left behind; the job runs again
later from the start.

Tier 0 (no chat endpoint) runs the deterministic part only: known names and
aliases mentioned in the text become co-occurrence edges (SPEC 6).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict

from iifa import log, vectors
from iifa.calibration import Thresholds, key_threshold_for, thresholds_for
from iifa.ingest import intent as intent_reading
from iifa.ingest import schema, sets
from iifa.ingest import sessions as session_state
from iifa.ingest.activity import grounded_activity
from iifa.ingest.checks import (
    cooccurring,
    grounded,
    is_complete,
    mixed_script_words,
    names_subject,
    script_of,
    scripts_in,
    shares_script,
    source_spelling,
    unspaced_chars,
)
from iifa.ingest.keys import (
    RULE_FAMILIES,
    canonical_key,
    family_of,
    is_free_form,
    key_text,
    replacement_asked,
    rule_topic,
    same_key,
    single_valued,
    supersedable,
)
from iifa.ingest.parts import ASSISTANT, EXTERNAL, OWNER, Part, SourceText
from iifa.ingest.prompts import (
    adjudication_messages,
    extraction_messages,
    replacement_adjudication_messages,
    restatement_adjudication_messages,
    rule_adjudication_messages,
    value_adjudication_messages,
)
from iifa.ingest.resolve import (
    ADJUDICATED,
    AMBIGUOUS,
    EXACT,
    FUZZY,
    NEW,
    Candidate,
    Resolution,
    Resolver,
    form_duplicate,
    similarity,
)
from iifa.ingest.schema import Answer, ProposedEdge, ProposedFact, Rejections, parse_answer
from iifa.ingest.speaker import is_speaker, speaker_words
from iifa.ingest.values import (
    DIFFERENT,
    SAME,
    UNKNOWN,
    compare_values,
    near_form,
    precision,
    same_name,
    same_spelling,
    shares_words,
    transliterate,
)
from iifa.language import (
    UNDETERMINED,
    detect,
    fold,
    language_of_span,
    languages_present,
    main_script,
    writing_mismatch,
)
from iifa.llm import TRUNCATED, ModelError
from iifa.names import plain_name
from iifa.recall.tokenizer import normalize, tokenize
from iifa.scopes import ROOT_SCOPE, Visibility, ancestors
from iifa.store.models import Edge, Fact, Node
from iifa.store.repo import (
    CONFIRMED,
    MERGE,
    NAME,
    PENDING,
    PROPOSED,
    REJECTED,
    SELF_RULE,
    Store,
)
from iifa.tokens import ConservativeTokenizer

if TYPE_CHECKING:
    from iifa.ingest.worker import JobContext

EXTRACTED = "extracted"
CONFLICT, LANGUAGE_MISMATCH = "conflict", "language_mismatch"
SPEAKER = "speaker"  # a free-standing fact about the person speaking (SPEC 4.2)
OBSERVED = intent_reading.OBSERVED  # SPEC 4.2: the duration word is also the flag
TOOL = "tool"  # SPEC 4.2: an external part whose facts are observations
SEMANTIC_CANDIDATES = 5  # near nodes offered to the adjudication per mention
RULE_CANDIDATES = 5  # SPEC 4.2: known rules a rule of the self is compared with
RULE_GROUP = "rule:"  # prefix of a rule comparison's group in ``_Plan.adjudicated``
KEYS_PER_SUBJECT = 8  # attribute keys listed per known subject in the prompt
PAIR_QUOTE_CHARS = 300  # SPEC 5.4: the source's words given with each fact of a pair
INCOMPLETE, UNNAMED, CROSS_SCRIPT = "incomplete", "unnamed", "cross_script"  # fragment causes
MIXED_SCRIPT = "mixed_script"  # a word mixing two spaced scripts (SPEC 4.2)
METHOD = "method"  # SPEC 4.2, agent perspective: the key of a rule of method
TASK_KIND = "task"  # SPEC 4.2: the kind of an activity's node
SELF_OUTSIDE = "self:outside"  # SPEC 4.2: why a self-side fact is only proposed
# SPEC 4.4: the external content a source held; the flag of a self-side fact
# learnt from a source whose external parts were all vouched (SPEC 4.2)
NONE, VOUCHED, UNVOUCHED = "none", "vouched", "unvouched"
# words as ``tokenize`` splits them (letters and digits), spelt as written
_WORD = re.compile(r"[^\W_]+")


_SPEAKER_WORDS = tuple(sorted(speaker_words()))  # sorted once, not per fact


def words_of(text: str) -> list[str]:
    return _WORD.findall(text)


def word_windows(text: str, size: int) -> list[str]:
    """Every run of ``size`` consecutive words of ``text``, joined by a space."""
    words = words_of(text)
    if size <= 0:
        return []
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


_ESTIMATOR = ConservativeTokenizer()


@dataclass(frozen=True)
class Owner:
    """Who the memory of a scope belongs to (SPEC 4.6): the declared scope, where
    the owner's node lives, its name and its aliases."""

    scope: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    """What the write path is allowed to do and how far (SPEC 5.2, 5.3, P12)."""

    chunk_chars: int = 12000
    max_chunks: int = 40
    answer_max_tokens: int = 4096
    answer_base_tokens: int = 512
    answer_tokens_per_source_token: float = 3.0
    adjudication_tokens_per_mention: int = 64
    json_mode: bool = True
    # configured overrides; None: from the embedding model's profile (SPEC 6.1)
    resolve_threshold: float | None = None
    merge_threshold: float | None = None
    link_threshold: float | None = None
    key_threshold: float | None = None
    embed_batch: int = 32
    known_names_in_prompt: int = 40
    confirm_after_sources: int = 2
    cooccurrence_confirm_after_sources: int = 3
    cooccurrence_weight: float = 0.5
    max_cooccurrence_nodes: int = 12
    edge_half_life_days: float = 30.0
    edge_prune_below: float = 0.1
    summary_nodes_per_session: int = 20
    summary_facts: int = 20
    fuzzy_threshold: float = 0.86
    language_overrides: tuple[tuple[str, str], ...] = ()
    owners: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    assistants: tuple[tuple[str, str], ...] = ()  # SPEC 4.6: (scope, name)
    # SPEC 4.6: (scope, name, aliases) - the agent perspective's self
    selves: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    review_self_rules: bool = False  # SPEC 4.2: a self's rules wait in review
    replacement_question: bool = True  # SPEC 5.4: on by default (measured)
    restatement_question: bool = True  # SPEC 5.4: a value told again in other words
    value_norm: bool = False  # SPEC 5.4: ask and compare each value's comparison form
    # SPEC 4.2: intent before content (frame, modality, duration) and the session
    # state it needs (SPEC 5.8); a session nobody ends closes after this idle time
    intent: bool = True
    session_idle_minutes: float = 120.0
    session_facts_in_prompt: int = 20

    def __post_init__(self) -> None:
        if not (0 < self.answer_base_tokens <= self.answer_max_tokens):
            raise ValueError("answer_base_tokens must be within 1..answer_max_tokens")
        if self.answer_tokens_per_source_token < 0:
            raise ValueError("answer_tokens_per_source_token must be 0 or more")
        if self.session_idle_minutes < 0:
            raise ValueError("session_idle_minutes must be 0 (never) or more")

    def answer_cap(self, source: str) -> int:
        """``max_tokens`` for an extraction call over ``source`` (SPEC 7.4b): a base
        plus a multiple of the source's estimated tokens (an answer quotes its
        source), never above ``answer_max_tokens``."""
        tokens = _ESTIMATOR.count(source)
        wanted = self.answer_base_tokens + math.ceil(self.answer_tokens_per_source_token * tokens)
        return min(self.answer_max_tokens, wanted)

    def adjudication_cap(self, mentions: int) -> int:
        """``max_tokens`` for one batched adjudication call over ``mentions``."""
        wanted = self.answer_base_tokens + self.adjudication_tokens_per_mention * mentions
        return min(self.answer_max_tokens, wanted)

    def thresholds(self, model: str) -> Thresholds | None:
        """The similarity thresholds for ``model`` (profile, configuration, or None
        when uncalibrated: no semantic resolution, merge or link)."""
        return thresholds_for(
            model,
            resolve=self.resolve_threshold,
            merge=self.merge_threshold,
            link=self.link_threshold,
        )

    def key_threshold_of(self, model: str) -> float | None:
        """SPEC 5.4 step 4: the cosine that aligns a new attribute key with an
        existing one for ``model``, or None (off)."""
        return key_threshold_for(model, configured=self.key_threshold)

    def forced_language(self, scopes: Iterable[str]) -> str | None:
        """The language a scope (or one of its ancestors, ``*`` last) forces, if
        any (P12)."""
        table = dict(self.language_overrides)
        for scope in scopes:
            for candidate in (*ancestors(scope), ROOT_SCOPE):
                if candidate in table:
                    return table[candidate]
        return None

    def owner(self, scopes: Iterable[str]) -> Owner | None:
        """The owner a source written into ``scopes`` speaks as (SPEC 4.6): the
        nearest declaration of each scope, and only when they all agree - a
        source split between two owners, or with a scope that has none, has no
        owner (who speaks cannot be told)."""
        return _declared(self.owners, scopes)

    def agent(self, scopes: Iterable[str]) -> Owner | None:
        """The self whose own memory a source written into ``scopes`` also is
        (SPEC 4.2, 4.6), by the owner's rule; None: the user perspective."""
        return _declared(self.selves, scopes)

    def assistant(self, scopes: Iterable[str]) -> str | None:
        """The assistant's declared name for a source written into ``scopes``
        (SPEC 4.6), by the same rule as the owner, a declaration on ``*`` holding
        everywhere; a self is the assistant of its scope, the nearer of the two
        declarations winning; None: ``Assistant``."""
        table = dict(self.assistants)
        table.update({scope: name for scope, name, _aliases in self.selves})
        found = {
            next((table[c] for c in (*ancestors(s), ROOT_SCOPE) if c in table), None)
            for s in scopes
        }
        return found.pop() if len(found) == 1 else None


def _declared(
    declarations: Iterable[tuple[str, str, tuple[str, ...]]], scopes: Iterable[str]
) -> Owner | None:
    """The nearest declaration of each scope, when they all agree (SPEC 4.6)."""
    table = {scope: Owner(scope, name, tuple(aliases)) for scope, name, aliases in declarations}
    found: set[Owner | None] = set()
    for scope in scopes:
        found.add(next((table[c] for c in ancestors(scope) if c in table), None))
    if len(found) != 1:
        return None
    return found.pop()


def write_view(scopes: Sequence[str]) -> Visibility:
    """What a write into ``scopes`` may see while resolving and checking: the
    scope and its ancestors for a single scope, exactly those scopes otherwise
    (the same rule the ``remember`` route applies)."""
    return Visibility.of_scope(scopes[0]) if len(scopes) == 1 else Visibility.of_scopes(scopes)


# --------------------------------------------------------------------------- the report


@dataclass
class ExtractionReport:
    source_id: str | None = None
    tier: int = 0
    chunks: int = 0
    calls: int = 0
    retries: int = 0
    truncated: int = 0
    languages: list[str] = field(default_factory=list)
    nodes_created: int = 0
    nodes_matched: int = 0
    nodes_mixed_script: int = 0  # SPEC 4.2: names mixing two scripts, never a node
    mentions_adjudicated: int = 0
    adjudication_calls: int = 0
    source_forms: int = 0  # SPEC 5.1: a subject contested by its form in the source
    # SPEC 11.1: each mention sent to the adjudication, its candidates, the answer
    adjudications: list[dict[str, Any]] = field(default_factory=list)
    facts_written: int = 0
    facts_duplicate: int = 0
    facts_superseded: int = 0
    values_same: int = 0  # SPEC 5.4: proved the same value in another form
    values_different: int = 0  # SPEC 5.4: proved different across languages, superseded
    values_norm: int = 0  # SPEC 5.4: linked across languages by their comparison forms
    values_adjudicated: dict[str, int] = field(
        default_factory=lambda: {"same": 0, "different": 0, "unsure": 0}
    )
    # SPEC 5.4: role, team, preferences - "replaces, or both true?"
    values_replacement: dict[str, int] = field(
        default_factory=lambda: {"same": 0, "replaces": 0, "both": 0, "unsure": 0}
    )
    # SPEC 5.4: within one language, "the same value told again?"
    values_restatement: dict[str, int] = field(
        default_factory=lambda: {"same": 0, "different": 0, "unsure": 0}
    )
    facts_kept_same: int = 0  # SPEC 5.4: the same value told again, the fact held kept
    facts_conflict: int = 0
    facts_language_mismatch: int = 0
    facts_incomplete: int = 0
    facts_cross_script: int = 0
    facts_mixed_script: int = 0
    facts_speaker_owner: int = 0  # SPEC 4.2: about the speaker, attached to the owner
    facts_speaker_free: int = 0  # SPEC 4.2: about the speaker, no owner declared
    facts_proposed: int = 0  # P6 and SPEC 5.1: fact proposals, whatever the cause
    facts_neighbour: int = 0  # SPEC 5.1: proposed, the source shows another subject
    facts_unnamed: int = 0  # SPEC 5.1: proposed, a known subject neither source nor session names
    facts_observed: int = 0  # SPEC 4.2: observed facts written, true at a date
    facts_observed_by_tool: int = 0  # ...of which the provenance rule made so (a tool part)
    facts_observed_again: int = 0  # SPEC 4.2: held observations whose day moved
    # SPEC 4.2, agent perspective: written about the self, rules of method
    # written, self-side facts proposed, and the self's own facts said again
    facts_self: int = 0
    facts_method: int = 0
    facts_self_proposed: int = 0
    facts_self_echo: int = 0
    # SPEC 4.2: a rule of the self against the known ones - "the same rule?"
    rules_compared: dict[str, int] = field(
        default_factory=lambda: {"same": 0, "different": 0, "unsure": 0}
    )
    # SPEC 4.2, agent perspective: an activity checked and set by the source, the
    # session's current one used, activities refused, rules linked to one
    activity_named: int = 0
    activity_carried: int = 0
    activity_refused: int = 0
    activity_linked: int = 0
    empty_answers: int = 0  # SPEC 4.2: chunks the model found nothing durable in
    aliases_kept: int = 0
    aliases_proposed: int = 0
    aliases_invented: int = 0
    aliases_descriptive: int = 0  # SPEC 5.1: a description: proposed if grounded, else dropped
    aliases_near: int = 0  # SPEC 5.1: a near form of the node's name, proposed rather than kept
    aliases_unknown: int = 0  # SPEC 5.1: null, a name the source does not give
    aliases_malformed: int = 0  # SPEC 5.1: no name once Markdown is stripped, or unbalanced
    keys_renamed: int = 0  # SPEC 5.4 step 3: a synonym or a variant made canonical
    keys_matched: int = 0  # SPEC 5.4 step 4: a near existing key, by embeddings
    # SPEC 5.4 (sets): elements added, removed by ``removes``, removals naming
    # no element, and values left in conflict by an unsure cardinality
    elements: dict[str, int] = field(
        default_factory=lambda: {"added": 0, "removed": 0, "unmatched": 0, "unsure": 0}
    )
    edges_proposed: int = 0
    edges_evidenced: int = 0
    edges_confirmed: int = 0
    # SPEC 5.1: a new node paired with a known one in another form, for review
    duplicates_merge_proposed: int = 0
    # SPEC 5.4: a known duplicate's stale value flagged by a supersession
    duplicates_flagged: int = 0
    rejected: Rejections = field(default_factory=Rejections)
    # SPEC 4.2: what the intent reading did (None: not read - off, or tier 0)
    intent: intent_reading.IntentReport | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "tier": self.tier,
            "chunks": self.chunks,
            "calls": self.calls,
            "retries": self.retries,
            "truncated": self.truncated,
            "languages": list(self.languages),
            "nodes": {
                "created": self.nodes_created,
                "matched": self.nodes_matched,
                "mixed_script": self.nodes_mixed_script,
            },
            "mentions_adjudicated": self.mentions_adjudicated,
            "adjudication_calls": self.adjudication_calls,
            "source_forms": self.source_forms,
            "adjudications": [dict(a) for a in self.adjudications],
            "empty_answers": self.empty_answers,
            "facts": {
                "written": self.facts_written,
                "duplicate": self.facts_duplicate,
                "superseded": self.facts_superseded,
                "kept_same": self.facts_kept_same,
                "conflict": self.facts_conflict,
                "language_mismatch": self.facts_language_mismatch,
                "incomplete": self.facts_incomplete,
                "cross_script": self.facts_cross_script,
                "mixed_script": self.facts_mixed_script,
                "speaker_owner": self.facts_speaker_owner,
                "speaker_free": self.facts_speaker_free,
                "proposed": self.facts_proposed,
                "neighbour": self.facts_neighbour,
                "unnamed": self.facts_unnamed,
                "observed": self.facts_observed,
                "observed_by_tool": self.facts_observed_by_tool,
                "observed_again": self.facts_observed_again,
                "self": self.facts_self,
                "method": self.facts_method,
                "self_proposed": self.facts_self_proposed,
                "self_echo": self.facts_self_echo,
            },
            # SPEC 4.2: a rule of the self against the known ones
            "self_rules": {"compared": dict(self.rules_compared)},
            "activity": {
                "named": self.activity_named,
                "carried": self.activity_carried,
                "refused": self.activity_refused,
                "linked": self.activity_linked,
            },
            "aliases": {
                "kept": self.aliases_kept,
                "proposed": self.aliases_proposed,
                "invented": self.aliases_invented,
                "descriptive": self.aliases_descriptive,
                "near": self.aliases_near,
                "unknown": self.aliases_unknown,
                "malformed": self.aliases_malformed,
            },
            "keys": {"renamed": self.keys_renamed, "matched": self.keys_matched},
            "elements": dict(self.elements),
            "values": {
                "same": self.values_same,
                "different": self.values_different,
                "norm": self.values_norm,
                "adjudicated": dict(self.values_adjudicated),
                "replacement": dict(self.values_replacement),
                "restatement": dict(self.values_restatement),
            },
            "edges": {
                "proposed": self.edges_proposed,
                "evidenced": self.edges_evidenced,
                "confirmed": self.edges_confirmed,
            },
            "duplicates": {
                "merge_proposed": self.duplicates_merge_proposed,
                "flagged": self.duplicates_flagged,
            },
            "rejected": self.rejected.as_dict(),
            "intent": self.intent.as_dict() if self.intent is not None else None,
            "dropped_by_intent": self.intent.dropped()
            if self.intent is not None
            else {"frame": {}, "modality": {}},
        }


# --------------------------------------------------------------------------- text helpers

_BREAK = re.compile("[.!?\u3002\uff01\uff1f]\\s+|\n+")  # sentence ends (Latin, CJK) or lines


def chunks(text: str, size: int, *, limit: int) -> list[tuple[int, str]]:
    """``(offset, piece)`` pieces of at most ``size`` characters, cut at sentence
    or line ends when one exists; at most ``limit`` pieces (the rest is dropped
    and the caller reports it)."""
    if size < 1:
        raise ValueError("chunk size must be positive")
    out: list[tuple[int, str]] = []
    start = 0
    while start < len(text) and len(out) < limit:
        end = min(len(text), start + size)
        if end < len(text):
            cut = None
            for match in _BREAK.finditer(text, start, end):
                cut = match.end()
            if cut is not None and cut > start + size // 4:
                end = cut
        piece = text[start:end]
        if piece.strip():
            out.append((start, piece))
        start = end
    return out


def locate(text: str, quote: str | None, *, hint: int = 0) -> tuple[int, int] | None:
    """The span of ``quote`` in ``text`` (P6: excerpt offsets), exact first, then
    ignoring case, then ignoring how whitespace was written; ``None`` when the
    model's quote is not in the source."""
    if not quote:
        return None
    needle = quote.strip()
    if not needle:
        return None
    position = text.find(needle, hint)
    if position < 0:
        position = text.find(needle)
    if position >= 0:
        return position, position + len(needle)
    lowered, folded = text.lower(), needle.lower()
    position = lowered.find(folded)
    if position >= 0:
        return position, position + len(needle)
    pattern = r"\s+".join(re.escape(part) for part in needle.split())
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


def mentioned_nodes(
    text: str, index: Sequence[tuple[str, str, str, str, tuple[str, ...]]]
) -> list[str]:
    """Tier-0 mention detection (SPEC 6): ids of the known nodes whose name or
    alias appears in ``text`` as whole tokens, in order of first appearance."""
    haystack = " " + " ".join(tokenize(text)) + " "
    found: list[tuple[int, str]] = []
    for node_id, _kind, _name, name_norm, aliases in index:
        best: int | None = None
        for candidate in (name_norm, *aliases):
            if len(candidate) < 2:
                continue
            position = haystack.find(f" {candidate} ")
            if position >= 0 and (best is None or position < best):
                best = position
        if best is not None:
            found.append((best, node_id))
    return [node_id for _, node_id in sorted(found)]


# --------------------------------------------------------------------------- the pipeline


class Extractor:
    def __init__(self, ctx: JobContext, scopes: Sequence[str]) -> None:
        self.ctx = ctx
        self.store: Store = ctx.store
        self.config: PipelineConfig = ctx.config
        self.scopes = list(dict.fromkeys(scopes))
        self.view = write_view(self.scopes)
        self.resolver = Resolver(self.store, self.view, fuzzy_threshold=self.config.fuzzy_threshold)
        self.report = ExtractionReport()
        self.source = SourceText("")
        self._part_tokens: dict[str, list[list[str]]] = {}  # provenance -> tokens per part
        self._read_at: str | None = None  # the day the source was read (P6)
        # P6, SPEC 4.4: the external content the source held, when its parts
        # are not all known (a session's close); None: read from the parts
        self._external: str | None = None
        # SPEC 4.6: who speaks, and the owner's node when it exists already
        self.owner = self.config.owner(self.scopes)
        self.owner_node: Node | None = None
        if self.owner is not None:
            self.owner_node = self.store.find_node(
                self.owner.name, Visibility.of_scope(self.owner.scope)
            )
        # SPEC 4.2, 4.6: the agent perspective's self, and its node when it exists
        self.agent = self.config.agent(self.scopes)
        self.agent_node: Node | None = None
        if self.agent is not None:
            self.agent_node = self.store.find_node(
                self.agent.name, Visibility.of_scope(self.agent.scope)
            )
        # SPEC 5.8: what the session's state shows the extraction calls
        self._session_block: str | None = None
        # SPEC 4.2, agent perspective: the activity this source names (its name,
        # and its node when one exists), the session's current one, and the
        # node the source's activity was written as
        self._activity: tuple[str, Node | None] | None = None
        self._carried: Node | None = None
        self._activity_written: str | None = None
        # SPEC 5.8: at a close, every activity of the session (the episode's links)
        self._episode_activities: list[str] = []
        # SPEC 4.2: an answer was read in a frame other than sincere
        self._unmeant = False
        # SPEC 5.1, a known subject the source does not name: the session's
        # earlier sources, read once, and the verdict per known node
        self._session_id: str | None = None
        self._session_cache: list[str] | None = None
        self._named: dict[str, bool] = {}
        # a test seam between the model calls and the write; never set in service
        self._before_write: Callable[[], None] | None = None

    # ------------------------------------------------------------------ entry point

    def run(
        self,
        text: str,
        *,
        kind: str,
        parts: Sequence[Part] | None = None,
        session_id: str | None = None,
        reference: str | None = None,
        external_id: str | None = None,
    ) -> ExtractionReport:
        """``parts``: who said what (P6); without them the whole text is taken as
        the owner's words (a source that is not a turn: a text handed in whole)."""
        report = self.report
        if not text.strip():
            return report
        self.source = SourceText(
            text, tuple(parts) if parts is not None else (Part(0, len(text), OWNER),)
        )
        self._part_tokens = {}
        self._read_at = day_read(reference, self.store.now())
        self._session_id = session_id
        forced = self.config.forced_language(self.scopes)
        languages = languages_present(text)
        report.languages = [forced] if forced else languages
        source_language = forced or (languages[0] if languages else None)
        report.tier = 2 if self.ctx.models.has_chat else 0

        answers: list[tuple[int, Answer]] = []
        reading: _IntentOutcome | None = None
        state: session_state.SessionState | None = None
        if report.tier == 2 and self.config.intent:
            # SPEC 4.2 and 5.8: the session's open frame and work in progress
            state = session_state.load(self.store, session_id, self.scopes)
            self._session_block = session_state.render(state, self.config.session_facts_in_prompt)
            self._carried = self._current_activity(state)
        if report.tier == 2:
            answers = self._extract_all(text, languages, forced)
            given = answers  # the intent reading keeps content only
            if state is not None:
                reading = self._read_intent(answers, state, reference)
                answers = reading.kept
            self._activity = self._activity_of(given, text)
        else:
            report.chunks = len(chunks(text, self.config.chunk_chars, limit=self.config.max_chunks))

        # every model call is done (extraction, then adjudication): now write once
        plan = self._plan(text, answers, forced)
        if self._before_write is not None:
            self._before_write()
        with self.store.lock, self.store.transaction():
            source = self.store.add_source(
                scopes=self.scopes,
                kind=kind,
                text=text,
                reference=reference,
                external_id=external_id,
                session_id=session_id,
                language=source_language,
                external=self._source_external(),
            )
            report.source_id = source.id
            self._write(source.id, text, plan)
            if reading is not None and state is not None:
                self._write_intent(source.id, text, reading, state, forced, session_id)
        log.event(
            "extracted",
            source=report.source_id,
            tier=report.tier,
            chunks=report.chunks,
            calls=report.calls,
            nodes=report.nodes_created,
            facts=report.facts_written,
            edges=report.edges_proposed,
            rejected=report.rejected.total,
            mismatches=report.facts_language_mismatch,
        )
        return report

    # ------------------------------------------------------------------ model calls

    def _extract_all(
        self, text: str, languages: list[str], forced: str | None
    ) -> list[tuple[int, Answer]]:
        pieces = chunks(text, self.config.chunk_chars, limit=self.config.max_chunks)
        self.report.chunks = len(pieces)
        known = self.resolver.known_names(self.config.known_names_in_prompt)
        known_keys = self._known_keys()
        answers: list[tuple[int, Answer]] = []
        for offset, piece in pieces:
            answer = self._extract_chunk(piece, offset, text, languages, forced, known, known_keys)
            if answer is not None:
                answers.append((offset, answer))
        return answers

    def _call(self, messages: list[dict[str, str]], cap: int) -> str:
        """One call capped at ``cap`` tokens. A truncated answer is never parsed
        (SPEC 7.4b): it is asked again once with twice the cap, up to the bound;
        still truncated, or already at the bound, ``ModelError("length")`` goes up."""
        models, json_mode = self.ctx.models, self.config.json_mode
        try:
            result = models.chat(messages, json_object=json_mode, max_tokens=cap)
        except ModelError as exc:
            if exc.reason != TRUNCATED:
                raise
            self.report.calls += 1
            self.report.truncated += 1
            bigger = min(2 * cap, self.config.answer_max_tokens)
            if bigger <= cap:
                raise
            result = models.chat(messages, json_object=json_mode, max_tokens=bigger)
        self.report.calls += 1
        return result.content

    def _extract_chunk(
        self,
        piece: str,
        offset: int,
        text: str,
        languages: list[str],
        forced: str | None,
        known: list[str],
        known_keys: list[tuple[str, list[str]]],
    ) -> Answer | None:
        cap = self.config.answer_cap(piece)

        def ask(note: str | None) -> Answer:
            content = self._call(
                extraction_messages(
                    piece,
                    languages=languages,
                    forced_language=forced,
                    known_names=known,
                    known_keys=known_keys,
                    retry_note=note,
                    owner=self.owner.name if self.owner is not None else None,
                    assistant=self.config.assistant(self.scopes),
                    external=any(p.provenance == EXTERNAL for p in self.source.parts),
                    value_norm=self.config.value_norm,
                    intent=self.config.intent,
                    session=self._session_block,
                    agent=self.agent.name if self.agent is not None else None,
                    activity=self._carried.name if self._carried is not None else None,
                ),
                cap,
            )
            return parse_answer(content)

        answer = ask(None)
        if not answer.usable:
            # not JSON at all: one more chance, then the chunk yields nothing
            self.report.retries += 1
            self.report.rejected.add(answer.rejected)
            answer = ask("Answer with the JSON object only, no prose, no code fence.")
            if not answer.usable:
                self.report.rejected.add(answer.rejected)
                return None
        wrong = self._mismatched(answer, piece, offset, text, forced)
        fragments = self._fragments(answer, piece, offset, text)
        if wrong or fragments:
            # P12 and SPEC 4.2: rejected and asked again once; what is still in the
            # wrong language is stored flagged by ``_write``, what is still a
            # fragment is dropped below
            self.report.retries += 1
            self.report.rejected.add(answer.rejected)
            notes: list[str] = []
            broken = [i for i, cause in fragments.items() if cause != MIXED_SCRIPT]
            if broken:
                notes.append(
                    f"Your previous answer wrote {len(broken)} statement(s) that were "
                    "fragments. Every statement must be a complete sentence that names its "
                    "subject (the subject's name, then what is true about it), in the "
                    "language of its source sentence."
                )
            unnamed = sum(1 for cause in fragments.values() if cause == CROSS_SCRIPT)
            if unnamed:
                notes.append(
                    f"{unnamed} statement(s) are written in a script in which their subject's "
                    "name does not appear. Write the subject as the source writes it; when the "
                    "source spells a known subject in another script, list that spelling in "
                    'the subject\'s aliases (e.g. {"name": "Karim", "aliases": ["Карим"]}).'
                )
            mixed = [
                word
                for i, cause in fragments.items()
                if cause == MIXED_SCRIPT
                for word in mixed_script_words(
                    f"{answer.facts[i].subject or ''} {answer.facts[i].statement}"
                )
            ]
            if mixed:
                notes.append(
                    f"{len(mixed)} word(s) of your previous answer mix two scripts in one word "
                    f"({', '.join(dict.fromkeys(mixed))}). Write every word in one script, as the "
                    "source writes it."
                )
            if wrong:
                expected = sorted({lang for _, lang in wrong})
                notes.append(
                    f"Your previous answer wrote {len(wrong)} item(s) in the wrong language. "
                    "Every statement, value and summary must be in the language of its source "
                    f"sentence ({', '.join(expected)}). Do not translate."
                )
            retried = ask(" ".join(notes))
            if retried.usable:
                answer = retried
        self.report.rejected.add(answer.rejected)
        still = self._fragments(answer, piece, offset, text)
        if still:
            answer.facts = [f for i, f in enumerate(answer.facts) if i not in still]
            self.report.facts_incomplete += len(still)
            self.report.facts_cross_script += sum(1 for c in still.values() if c == CROSS_SCRIPT)
            self.report.facts_mixed_script += sum(1 for c in still.values() if c == MIXED_SCRIPT)
            self.report.rejected.facts += len(still)
        if not (answer.nodes or answer.facts or answer.edges):
            self.report.empty_answers += 1  # SPEC 4.2: chatter, as the prompt expects
        return answer

    def _fragments(self, answer: Answer, piece: str, offset: int, text: str) -> dict[int, str]:
        """The facts that are fragments (SPEC 4.2), by index, with the cause:
        ``incomplete`` (not a sentence), ``unnamed`` (a sentence that does not
        name its subject), ``cross_script`` (no name of the subject is written in
        the statement's script at all), ``mixed_script`` (a word mixing two
        scripts written with spaces, "كazan"). A free-standing fact, and a true
        translation (P12's to judge), are checked for completeness only."""
        out: dict[int, str] = {}
        groups = _NameGroups(self, answer, piece)
        for index, fact in enumerate(answer.facts):
            if mixed_script_words(fact.statement) or mixed_script_words(fact.subject or ""):
                out[index] = MIXED_SCRIPT
                continue
            if not is_complete(fact.statement):
                out[index] = INCOMPLETE
                continue
            if not fact.subject or self._translated(fact, piece, offset, text):
                continue
            if is_speaker(fact.subject) and self.owner is None:
                continue  # SPEC 4.2: a free speaker fact, checked for completeness only
            if self._owner_form_named(fact, offset):
                continue
            names = groups.names(fact.subject)
            if self._means_owner(fact.subject):
                # SPEC 4.2: the owner is named by a name, an alias or a speaker word
                names = [*names, *self._owner_names()]
            written = self._as_written(fact, answer, piece)
            if written is not None:
                names = [*names, written]
            if not shares_script(fact.statement, names):
                out[index] = CROSS_SCRIPT
            elif not names_subject(fact.statement, names):
                out[index] = UNNAMED
        return out

    def _owner_form_named(self, fact: ProposedFact, offset: int) -> bool:
        """SPEC 4.2, for the naming check only: the statement of a fact whose
        subject is exactly the owner's declared name, quoted from the owner's
        part, names the owner by any form of that name that transliterates to it
        ("Самир" for Samir; "Alice" inside a Russian statement). It confirms a
        subject already fixed; it never fixes one."""
        owner = self.owner
        if owner is None or fact.subject != owner.name:
            return False
        span = locate(self.source.text, fact.quote, hint=offset)
        if self.source.provenance_of(span) != OWNER:
            return False
        return names_owner(fact.statement, owner.name)

    def _owner_names(self) -> list[str]:
        if self.owner is None:
            return []
        stored = self.owner_node.aliases if self.owner_node is not None else ()
        return [self.owner.name, *self.owner.aliases, *stored, *_SPEAKER_WORDS]

    def _means_owner(self, mention: str | None) -> bool:
        """Does ``mention`` name the declared owner: a speaker word, the owner's
        name or one of its aliases, or a mention resolving to the owner's node?"""
        if self.owner is None or not mention:
            return False
        if is_speaker(mention):
            return True
        wanted = normalize(mention)
        if wanted in {normalize(n) for n in (self.owner.name, *self.owner.aliases)}:
            return True
        if self.owner_node is None:
            return False
        resolution = self.resolver.resolve(mention)
        return resolution.method not in (AMBIGUOUS, NEW) and resolution.key == self.owner_node.id

    def stored_names(self, key: str | None) -> list[str]:
        """The name and aliases of the stored node ``key`` (none for a new one)."""
        if key is None or key.startswith("new:"):
            return []
        node = self.store.get_node(key)
        return [node.name, *node.aliases]

    def _as_written(self, fact: ProposedFact, answer: Answer, text: str) -> str | None:
        """``subject_as_written`` when it may name the subject (SPEC 4.2): the source
        and the statement hold that spelling, and it is no other node's name or
        alias (of the answer, or known in the view). Otherwise None: ignored."""
        written, subject = fact.subject_as_written, fact.subject
        if not written or not subject:
            return None
        spelling = source_spelling(written, text)
        if spelling is None or not names_subject(fact.statement, [written]):
            return None
        spelling = plain_name(spelling)  # SPEC 5.1: the source's span, without debris
        if spelling is None:
            return None
        wanted, own_name = normalize(written), normalize(subject)
        for node in answer.nodes:
            if normalize(node.name) == own_name:
                continue
            forms = [node.name, *(a for a in node.aliases if grounded(a, text))]
            if wanted in {normalize(f) for f in forms}:
                return None
        resolution = self.resolver.resolve(subject)
        own = resolution.key if resolution.method not in (AMBIGUOUS, NEW) else None
        found = self.store.find_node(written, self.view)
        if found is not None and found.id != own:
            return None
        return spelling  # stored as the source writes it

    def _translated(self, fact: ProposedFact, piece: str, offset: int, text: str) -> bool:
        """Is the statement in another language than its source sentence (whatever
        a scope forces)? Then naming is not judged here: P12 is."""
        return writing_mismatch(
            fact.statement, self._expected_language(fact, piece, offset, text, None)
        )

    def _mismatched(
        self, answer: Answer, piece: str, offset: int, text: str, forced: str | None
    ) -> list[tuple[int, str]]:
        """Indices of the facts whose language differs from their source sentence's."""
        wrong: list[tuple[int, str]] = []
        for index, fact in enumerate(answer.facts):
            expected = self._expected_language(fact, piece, offset, text, forced)
            if writing_mismatch(fact.statement, expected):
                wrong.append((index, expected))
        return wrong

    def _expected_language(
        self, fact: ProposedFact, piece: str, offset: int, text: str, forced: str | None
    ) -> str:
        if forced:
            return forced
        span = locate(text, fact.quote, hint=offset)
        if span is not None:
            return language_of_span(text, *span)
        return detect(piece) if detect(piece) != UNDETERMINED else detect(text)

    # ------------------------------------------------------------------ resolution

    def _plan(self, text: str, answers: list[tuple[int, Answer]], forced: str | None) -> _Plan:
        plan = _Plan()
        self._register_owner(plan)
        self._register_agent(plan)
        self._register_activity(plan)
        self._plan_nodes(plan, answers, text)
        replaced = self._source_forms(answers, text)
        mentions: list[str] = []
        for _, answer in answers:
            mentions.extend(f.subject for f in answer.facts if f.subject)
            mentions.extend(m for e in answer.edges for m in (e.src, e.dst))
        quotes = {f.subject: f.quote for _, a in answers for f in a.facts if f.subject and f.quote}
        self._resolve_mentions(plan, list(dict.fromkeys(mentions)), text, quotes)
        _drop_unused(plan, replaced)
        # SPEC 4.2: a valid subject_as_written becomes a source-anchored alias
        for _, answer in answers:
            for fact in answer.facts:
                spelling = self._as_written(fact, answer, text)
                subject_key = plan.mention_keys.get(fact.subject or "")
                if spelling is None or subject_key is None:
                    continue
                if (fact.subject or "") in plan.unnamed:
                    # SPEC 5.1: a subject the source does not name and the
                    # adjudication did not attach takes no alias; a reviewer decides
                    plan.doubtful.append((subject_key, fact.subject or "", spelling))
                    continue
                names = [fact.subject or "", *self.stored_names(subject_key)]
                if subject_key in plan.proposals:
                    names.append(plan.proposals[subject_key].name)
                if any(near_form(spelling, known) for known in names):
                    self.report.aliases_near += 1  # names the subject here, never stored
                    continue
                plan.aliases.setdefault(subject_key, []).append(spelling)
        for offset, answer in answers:
            for proposed in answer.facts:
                fact = self._as_planned(proposed)
                plan.facts.append((offset, fact))
                plan.fates.append(self._fate(plan, offset, fact))
            for edge in answer.edges:
                plan.edges.append(edge)
        # tier-0 mention detection: known names in the text join the co-occurrence set
        if not self._unmeant:  # SPEC 4.2: nothing of a non-sincere frame's content
            for node_id in mentioned_nodes(text, self.store.node_index(self.view)):
                plan.order(node_id)
        self._neighbours(plan, text)
        self._unnamed(plan)
        self._align_keys(plan)
        self._adjudicate_values(plan)
        self._adjudicate_rules(plan)
        plan.forced = forced
        self.report.nodes_mixed_script += len(plan.mixed_names)
        return plan

    def _as_planned(self, fact: ProposedFact) -> ProposedFact:
        """SPEC 4.2, agent perspective: a rule of method is an element of a set,
        whatever the answer says."""
        if self.agent is not None and canonical_key(fact.attribute) == METHOD:
            return fact.model_copy(update={"cardinality": sets.MANY})
        return fact

    def _plan_nodes(self, plan: _Plan, answers: list[tuple[int, Answer]], text: str) -> None:
        """The nodes the answers propose, resolvable by the facts and edges that
        follow: a known node, or a new one; a speaker word is neither (SPEC 5.1)."""
        resolver = self.resolver
        nodes = [n for _, a in answers for n in a.nodes]
        self._prepare_semantic(answers)
        decided = self._adjudicate_names(nodes, text)
        forms = _AnswerForms(self, [a for _, a in answers], text)
        for index, proposal in enumerate(nodes):
            self.report.aliases_unknown += proposal.unknown_aliases
            self.report.aliases_malformed += proposal.malformed_aliases
            if is_speaker(proposal.name):
                continue
            if mixed_script_words(proposal.name):
                plan.mixed_names.add(proposal.name)  # SPEC 4.2: never a node
                continue
            existing = resolver.resolve(proposal.name, kind=proposal.kind)
            own = (
                existing.key if existing.key is not None and existing.method != AMBIGUOUS else None
            )
            if own is None and existing.method == AMBIGUOUS:
                own = decided.get(proposal.name)  # attached by the adjudication, or None
            candidates = list(proposal.aliases)
            if own is not None:
                # the answer's own name for a known node resolves the facts and
                # edges that follow ("Mimi" for Émilie)
                candidates.insert(0, proposal.name)
            kept, doubtful = forms.split(index, proposal.name, own, candidates)
            if (
                own is not None
                and existing.method in (EXACT, FUZZY)
                and self._unnamed_match(plan, own, text)
            ):
                # SPEC 5.1: a known subject the source does not name is not this
                # answer's node; its facts go to the adjudication, its aliases
                # are proposals
                known = {normalize(n) for n in (proposal.name, *self.stored_names(own))}
                offered = [a for a in (*doubtful, *kept) if normalize(a) not in known]
                plan.note_aliases(own, proposal.name, [], offered)
                continue
            if own is not None:
                plan.note_aliases(own, proposal.name, kept, doubtful)
                resolver.propose(own, proposal.name, proposal.kind, kept)
                plan.order(own)
                continue
            key = f"new:{len(plan.proposals)}"
            plan.proposals[key] = proposal.model_copy(update={"aliases": kept})
            plan.note_aliases(key, proposal.name, [], doubtful)
            resolver.propose(key, proposal.name, proposal.kind, kept)
            plan.order(key)

    def _source_forms(self, answers: list[tuple[int, Answer]], text: str) -> set[str]:
        """SPEC 5.1: a subject the source holds only in a near form ("Иван" for
        "Ивана ...") is resolved under both; unless both lead to exactly the same
        node, the fact's mention becomes the source form, contested: the
        adjudication decides, with the source sentence, and "new" names the node
        as the source writes it. Returns the subjects replaced (folded)."""
        resolver = self.resolver
        replaced: set[str] = set()
        for offset, answer in answers:
            for index, fact in enumerate(answer.facts):
                subject = fact.subject
                if not subject or is_speaker(subject) or mixed_script_words(subject):
                    continue
                span = locate(text, fact.quote, hint=offset)
                if span is None:
                    continue
                form = source_form(subject, text[span[0] : span[1]], self.config.fuzzy_threshold)
                if form is None:
                    continue
                said, read = resolver.resolve(subject), resolver.resolve(form)
                if _settled(said) and _settled(read) and said.key == read.key:
                    continue
                keys = [k for r in (said, read) for k in _keys_of(r)]
                resolver.contest(form, keys)
                answer.facts[index] = fact.model_copy(update={"subject": form})
                replaced.add(normalize(subject))
                self.report.source_forms += 1
        return replaced

    # ------------------------------------------------------------------ the speaker (SPEC 4.2, P6)

    def _register_owner(self, plan: _Plan) -> None:
        """The declared owner's node, resolvable from now on: the existing one, or
        one to create in the declared scope if a fact or a proposal needs it."""
        owner = self.owner
        if owner is None:
            return
        if self.owner_node is not None:
            key = self.owner_node.id
        else:
            key = f"new:{len(plan.proposals)}"
            plan.proposals[key] = schema.ProposedNode(
                name=owner.name, kind="person", aliases=list(owner.aliases)
            )
            plan.node_scopes[key] = [owner.scope]
        self.resolver.propose(key, owner.name, "person", owner.aliases)
        plan.owner_key = key

    def _register_agent(self, plan: _Plan) -> None:
        """SPEC 4.6: the self's node, resolvable like the owner's: the existing
        one, or one to create in the declared scope if a fact or a proposal needs it."""
        agent = self.agent
        if agent is None:
            return
        if self.agent_node is not None:
            key = self.agent_node.id
        else:
            key = f"new:{len(plan.proposals)}"
            plan.proposals[key] = schema.ProposedNode(
                name=agent.name, kind="other", aliases=list(agent.aliases)
            )
            plan.node_scopes[key] = [agent.scope]
        self.resolver.propose(key, agent.name, "other", agent.aliases)
        plan.agent_key = key

    # ------------------------------------------------------------ the activity (SPEC 4.2)

    def _current_activity(self, state: session_state.SessionState) -> Node | None:
        """The session's current activity, while its node is a live, visible task
        (agent perspective, a session the host named)."""
        if not state.key:
            return None
        return self._activity_node(state.activity_id)

    def _activity_node(self, node_id: str | None) -> Node | None:
        """A current activity's node, while it is a live, visible task (agent
        perspective)."""
        if self.agent is None or node_id is None:
            return None
        node = self.store.nodes_by_id([node_id], self.view).get(node_id)
        if node is None or node.kind != TASK_KIND or node.archived_at is not None:
            return None
        return node

    def _activity_of(
        self, answers: list[tuple[int, Answer]], text: str
    ) -> tuple[str, Node | None] | None:
        """SPEC 4.2: the activity the source asks for - the first one the answers
        give that passes the checks - with its node when one holds its name. In
        the user perspective the key is unknown: refused and counted."""
        report = self.report
        if self.agent is None:
            # a malformed one was counted by the schema already
            report.rejected.keys += sum(1 for _, a in answers if a.activity_given)
            return None
        for offset, answer in answers:
            if answer.activity is None:
                continue
            activity = answer.activity
            found = self._checked_activity(offset, activity.name, activity.quote, text)
            if found is not None:
                report.activity_named += 1
                return found
            report.activity_refused += 1
        return None

    def _checked_activity(
        self, offset: int, name: str, quote: str, text: str
    ) -> tuple[str, Node | None] | None:
        """The checks of SPEC 4.2 on one activity: a sincere frame, the quote in
        an owner part, a word of the name written whole in it, and a name that
        is a new task or exactly one known task node (never a node of another
        kind, nor a near or containing name)."""
        if self._unmeant:
            return None
        span = locate(text, quote, hint=offset)
        part = self.source.part_of(span)
        if span is None or part is None or part.provenance != OWNER:
            return None
        if (
            not grounded_activity(name, text[span[0] : span[1]])
            or is_speaker(name)  # SPEC 4.2, 5.1: never a node under such a name
            or mixed_script_words(name)
        ):
            return None
        held = [r["id"] for r in self.store.lookup_name(name, self.view)]
        if not held:
            return name, None
        nodes = self.store.nodes_by_id(held, self.view)
        if len(held) != 1 or held[0] not in nodes or nodes[held[0]].kind != TASK_KIND:
            return None  # when we cannot tell, nothing attaches
        return name, nodes[held[0]]

    def _register_activity(self, plan: _Plan) -> None:
        """SPEC 4.2: the source's activity (else the session's current one) is
        resolvable like the owner's node; a new one is created with the source."""
        if self._activity is not None:
            name, node = self._activity
            if node is not None:
                key = node.id
                self.resolver.propose(key, node.name, TASK_KIND, list(node.aliases))
            else:
                key = f"new:{len(plan.proposals)}"
                plan.proposals[key] = schema.ProposedNode(name=name, kind="task")
                self.resolver.propose(key, name, TASK_KIND, [])
            plan.order(key)
            plan.activity_key, plan.activity_named = key, True
        elif self._carried is not None:
            node = self._carried
            self.resolver.propose(node.id, node.name, TASK_KIND, list(node.aliases))
            plan.activity_key = node.id

    def _fate(self, plan: _Plan, offset: int, fact: ProposedFact) -> _Fate:
        """Whose words the fact's quote is (P6), whether the fact is about the
        owner (SPEC 4.2), and so whether it may be written or only proposed: a
        fact about the owner is written from the owner's own words only, and not
        when its value comes from outside the conversation alone. In the agent
        perspective, a self-side fact (SPEC 4.2) follows its own rule."""
        span = locate(self.source.text, fact.quote, hint=offset)
        part = self.source.part_of(span)
        provenance = part.provenance if part is not None else None
        subject = fact.subject or ""
        speaker = is_speaker(subject)
        about_owner = (speaker and self.owner is None) or (
            plan.owner_key is not None and plan.mention_keys.get(subject) == plan.owner_key
        )
        propose = about_owner and (provenance != OWNER or self._value_from_outside(fact.value))
        speaker_free = speaker and self.owner is None
        about_agent = (
            plan.agent_key is not None and plan.mention_keys.get(subject) == plan.agent_key
        )
        method = self.agent is not None and canonical_key(fact.attribute) == METHOD
        self_side = (about_agent or method) and not about_owner
        outside = self_side and not self._self_written(fact, provenance)
        # SPEC 4.2: a rule of the self waits for a reviewer when the operator
        # asks for it; what came from outside keeps its own, stronger reason
        rule = (
            self.config.review_self_rules
            and about_agent
            and not about_owner
            and not outside
            and family_of(canonical_key(fact.attribute)) in RULE_FAMILIES
        )
        return _Fate(
            span,
            provenance,
            about_owner,
            speaker_free,
            propose or outside or rule,
            part=part,
            about_agent=about_agent and not about_owner,
            method=method,
            self_side=self_side,
            self_outside=outside,
            self_rule=rule,
        )

    def _rule_known(
        self,
        node_id: str | None,
        fact: ProposedFact,
        known: list[tuple[str | None, str | None, str]] | None = None,
    ) -> bool:
        """SPEC 4.2: is this rule of the self already held, waiting or refused?
        A live fact of the node, or a pending or rejected proposal about it,
        that says the same statement, or gives the same value to the same key
        or to the same topic of the other rule family (the model files one
        rule as a commitment in one turn and as a preference in the next).
        The agent states its rules again in most replies: without this the
        review queue would fill with what the reviewer already decided."""
        if node_id is None:
            return False
        key, value, said = (
            canonical_key(fact.attribute),
            normalize(fact.value or ""),
            fold(fact.statement),
        )
        topic = rule_topic(key)

        def same(attribute: str | None, held: str | None, statement: str) -> bool:
            if fold(statement) == said:
                return True
            if not value or normalize(held or "") != value:
                return False
            other = canonical_key(attribute)
            return other == key or (topic is not None and rule_topic(other) == topic)

        return any(same(*held) for held in (known or self._rules_known(node_id)))

    def _rules_known(self, node_id: str) -> list[tuple[str | None, str | None, str]]:
        """SPEC 4.2: what a rule of the self is compared with, as (key, value,
        statement): the live facts of the node, then the pending and the
        rejected fact proposals about it, each read through the write's view (P9)."""
        held: list[tuple[str | None, str | None, str]] = [
            (f.attribute, f.value, f.statement) for f in self.store.facts([node_id], self.view)
        ]
        held.extend(
            (p.payload.get("attribute"), p.payload.get("value"), p.value)
            for status in (PENDING, REJECTED)
            for p in self.store.proposals([node_id], self.view, status=status)
            if p.kind == "fact"
        )
        return held

    def _adjudicate_rules(self, plan: _Plan) -> None:
        """SPEC 4.2: a rule of the self that restates no known rule word for word
        may still be one said again in other words - the model that extracts
        words a rule differently from one turn to the next. Each such rule goes
        to the chat model with the known rules nearest to it, in one batched
        call before the write transaction ("the same rule?"). Only ``same``
        changes anything: the rule is then an echo. ``different``, ``unsure``,
        an invalid answer or a truncated one leave it asked about."""
        if not self.ctx.models.has_chat:
            return
        batch: _Pairs = ([], [])
        known: dict[str, list[tuple[str | None, str | None, str]]] = {}
        for index, (_offset, fact) in enumerate(plan.facts):
            fate = plan.fates[index]
            node_id = plan.mention_keys.get(fact.subject or "")
            if not fate.self_rule or node_id is None or node_id.startswith("new:"):
                continue
            if node_id not in known:
                known[node_id] = self._rules_known(node_id)
            if self._rule_known(node_id, fact, known[node_id]):
                continue  # word for word: no question needed
            rules = list(
                dict.fromkeys(
                    statement
                    for key, _value, statement in known[node_id]
                    if family_of(canonical_key(key)) in RULE_FAMILIES
                )
            )
            for rank, held in enumerate(self._nearest_rules(fact.statement, rules)):
                sides: tuple[dict[str, Any], dict[str, Any]] = (
                    {"statement": held},
                    {"statement": fact.statement},
                )
                if (said := self._new_quote(fate)) is not None:
                    sides[1]["quote"] = said
                self._pair(batch, (index, f"{RULE_GROUP}{rank}"), node_id, None, sides)
        self._ask(
            plan,
            batch,
            rule_adjudication_messages,
            schema.parse_rule_verdicts,
            self.report.rules_compared,
        )

    def _nearest_rules(self, statement: str, rules: list[str]) -> list[str]:
        """At most RULE_CANDIDATES of ``rules`` to compare ``statement`` with.
        Embeddings only RANK them (no threshold: the chat model decides).
        With no embeddings endpoint, or one that fails (a ranking is not worth
        a failed turn), the first ones in the order ``_rules_known`` gives:
        the rules held, then those waiting, then those refused - a held rule
        is the one a restatement must not miss."""
        if len(rules) <= RULE_CANDIDATES:
            return rules
        models = self.ctx.models
        if not models.has_embeddings:
            return rules[:RULE_CANDIDATES]
        try:
            result = models.embed([statement, *rules])
            self.report.calls += 1
            mine, *others = (vectors.normalise(v) for v in result.vectors)
        except (ModelError, ValueError):
            return rules[:RULE_CANDIDATES]
        scored = sorted(
            (
                (sum(a * b for a, b in zip(mine, other, strict=True)), -i)
                for i, other in enumerate(others)
            ),
            reverse=True,
        )
        return [rules[-i] for _score, i in scored[:RULE_CANDIDATES]]

    def _source_external(self) -> str:
        """P6, SPEC 4.4: the content from outside the conversation the source
        held: none, only parts the host vouched for, or one it did not."""
        if self._external is not None:
            return self._external
        outside = [p for p in self.source.parts if p.provenance == EXTERNAL]
        if not outside:
            return NONE
        return VOUCHED if all(p.vouched for p in outside) else UNVOUCHED

    def _self_written(self, fact: ProposedFact, provenance: str | None) -> bool:
        """SPEC 4.2 and P6, agent perspective: may a self-side fact be written? From
        the owner's words when its value does not come from outside alone; from
        the assistant part, or a vouched part, of a source whose external parts
        were all vouched (or none). The parts and the host decide, never the
        words, so a paraphrased page does not pass."""
        if provenance == OWNER:
            return not self._value_from_outside(fact.value)
        # every external part vouched (or none): the reply and those parts
        return provenance in (ASSISTANT, EXTERNAL) and self._source_external() != UNVOUCHED

    def _value_from_outside(self, value: str | None) -> bool:
        """Does ``value`` appear (whole words, folded) in an external part and in
        no owner part? A model quoting "I live" may take the city from the page."""
        wanted = tokenize(value or "")
        if not wanted:
            return False

        def holds(provenance: str) -> bool:
            if provenance not in self._part_tokens:  # once per source, not per fact
                self._part_tokens[provenance] = [tokenize(t) for t in self.source.texts(provenance)]
            return any(_contains(tokens, wanted) for tokens in self._part_tokens[provenance])

        return holds(EXTERNAL) and not holds(OWNER)

    def _outside_part(self, value: str | None) -> Part | None:
        """The first external part that holds ``value`` (whole words, folded)."""
        wanted = tokenize(value or "")
        for part in self.source.parts:
            if part.provenance == EXTERNAL and _contains(
                tokenize(self.source.text[part.start : part.end]), wanted
            ):
                return part
        return None

    def _part_fields(self, provenance: str | None, part: Part | None) -> PartFields:
        """P6: an external fact keeps its part's kind, the short form of its ref
        and the day the source was read; an assistant's fact its declared name
        (SPEC 4.6); the owner's facts keep none of them."""
        if provenance == ASSISTANT:  # SPEC 4.6: the assistant's declared name
            named = part.ref if part is not None and part.provenance == ASSISTANT else None
            return PartFields(part_kind=None, part_ref=named, read_at=None)
        if provenance != EXTERNAL:
            return PartFields(part_kind=None, part_ref=None, read_at=None)
        external = part if part is not None and part.provenance == EXTERNAL else None
        return PartFields(
            part_kind=external.kind if external else None,
            part_ref=external.ref if external else None,
            read_at=self._read_at,
        )

    # ------------------------------------------------------------ neighbouring subjects (SPEC 5.1)

    def _neighbours(self, plan: _Plan, text: str) -> None:
        """SPEC 5.1, neighbouring subjects: when the source has two subjects or
        more, a fact whose quote names another of them and not its own, or whose
        value names its own subject and another one, is not attached on its own:
        it becomes a fact proposal (counted ``facts.neighbour``)."""
        forms = self._subject_forms(plan)
        if len(forms) < 2:
            return
        words = {key: {w for name in names for w in tokenize(name)} for key, names in forms.items()}
        distinctive = _distinctive(forms)
        linked = {
            frozenset((a, b))
            for edge in plan.edges
            if (a := plan.mention_keys.get(edge.src)) and (b := plan.mention_keys.get(edge.dst))
        }
        for index, (_offset, fact) in enumerate(plan.facts):
            fate = plan.fates[index]
            own = plan.mention_keys.get(fact.subject or "")
            if own is None or own not in forms or fate.propose:
                continue
            foreign = set().union(*(w for key, w in words.items() if key != own))
            # a form made of other subjects' words only ("PDF" beside report.pdf)
            # names none of them in particular: it never names the subject here
            own_forms = [f for f in forms[own] if not set(tokenize(f)) <= foreign]
            # a subject an edge of the answer links to this one is its object
            others = {
                key: names
                for key, names in forms.items()
                if key != own and frozenset((own, key)) not in linked
            }
            quoted = (
                fate.span is not None
                and not (fate.about_owner or fate.speaker_free or fate.about_agent)
                and quote_names_other(
                    text[fate.span[0] : fate.span[1]],
                    own_forms,
                    others,
                    objects=f"{fact.statement} {fact.value or ''}",
                )
            )
            if quoted or value_merges(fact.value, own, distinctive, forms):
                plan.fates[index] = dataclasses.replace(fate, propose=True, neighbour=True)

    # ------------------------------------------------- a known subject not named (SPEC 5.1)

    def _unnamed(self, plan: _Plan) -> None:
        """SPEC 5.1, a known subject the source does not name: a fact whose subject
        the adjudication did not attach (``plan.unnamed``) is a fact proposal on
        the known node, counted ``facts.unnamed``."""
        for index, (_offset, fact) in enumerate(plan.facts):
            fate = plan.fates[index]
            if fate.propose or fate.about_owner or fate.speaker_free or fate.about_agent:
                continue
            if (fact.subject or "") in plan.unnamed:
                plan.fates[index] = dataclasses.replace(fate, propose=True, unnamed=True)

    def _unnamed_match(self, plan: _Plan, key: str, text: str) -> bool:
        """SPEC 5.1: is the stored node ``key``, matched on its own (exact or
        contained), a subject neither the source nor its session names?
        True when none of the node's names or aliases is written there (whole
        words, the same spelling in another script, or as letters in a script
        written without spaces): a word of the name ("Karim" of "Karim Benali",
        "report" of "Nadia's annual report") is not enough. The owner is named by the label
        of their part, never here."""
        if key.startswith("new:") or key in (plan.owner_key, plan.agent_key, plan.activity_key):
            return False
        if key in self._episode_activities:  # SPEC 5.8: set from the owner's words
            return False
        named = self._named.get(key)
        if named is None:
            named = self._names(text, self.stored_names(key))
            self._named[key] = named
        return not named

    def _names(self, text: str, names: Sequence[str]) -> bool:
        """Does ``text`` or an earlier source of the session write one of ``names``?"""
        return any(_written_in(t, names) for t in (text, *self._session_texts()))

    def _session_texts(self) -> list[str]:
        """The session's earlier sources this writer sees (none outside a
        session), read once."""
        if self._session_cache is None:
            self._session_cache = (
                self.store.session_texts(self._session_id, self.view) if self._session_id else []
            )
        return self._session_cache

    def _subject_forms(self, plan: _Plan) -> dict[str, list[str]]:
        """The subjects of the source, by key: the nodes the answers propose, the
        nodes their facts and edges resolve to, the known nodes the text names;
        each with its names, its aliases and the mentions that resolved to it.
        A speaker word is no name (SPEC 4.2)."""
        forms: dict[str, list[str]] = {}
        for key in plan.ordered:
            proposal = plan.proposals.get(key)
            if key.startswith("new:"):
                names = [proposal.name, *proposal.aliases] if proposal is not None else []
            else:
                names = self.stored_names(key)
            if key == plan.owner_key and self.owner is not None:
                names = [*names, self.owner.name, *self.owner.aliases]
            forms[key] = [*names, *plan.aliases.get(key, [])]
        for mention, key in plan.mention_keys.items():
            if key in forms and not is_speaker(mention):
                forms[key].append(mention)
        return {key: [n for n in names if n] for key, names in forms.items()}

    # ------------------------------------------------------------------ attribute keys (SPEC 5.4)

    def _adjudicate_values(self, plan: _Plan) -> None:
        """SPEC 5.4: the pairs the deterministic comparison leaves ``unknown`` across
        languages - the new fact later, no explicit fact - go to the chat model
        before the write transaction: single-valued keys in one batched call ("the
        same thing?"), role, team and preferences in another ("replaces, or both
        true?"). Within one language, a change whose values share words may be the
        same value told again: asked in a third call ("the same value?"), where only
        ``same`` changes anything. ``unsure``, an invalid answer or a truncated one
        changes nothing."""
        if not self.ctx.models.has_chat:
            return
        values, replacements, restatements = self._unknown_pairs(plan)
        self._ask(
            plan,
            values,
            value_adjudication_messages,
            schema.parse_value_verdicts,
            self.report.values_adjudicated,
        )
        self._ask(
            plan,
            replacements,
            replacement_adjudication_messages,
            schema.parse_replacement_verdicts,
            self.report.values_replacement,
        )
        self._ask(
            plan,
            restatements,
            restatement_adjudication_messages,
            schema.parse_restatement_verdicts,
            self.report.values_restatement,
        )

    def _unknown_pairs(self, plan: _Plan) -> tuple[_Pairs, _Pairs, _Pairs]:
        """The pairs left ``unknown``, by question: single-valued keys across
        languages; role, team, preferences and free-form keys of one value, in one
        language too; and the other changes within one language that may restate
        the value held (SPEC 5.4)."""
        values: _Pairs = ([], [])
        replacements: _Pairs = ([], [])
        restatements: _Pairs = ([], [])
        lives: dict[str, list[Fact]] = {}  # node id -> its live facts, read once
        for index, (_offset, fact) in enumerate(plan.facts):
            node_id = plan.mention_keys.get(fact.subject or "")
            key = canonical_key(fact.attribute)
            fate = plan.fates[index]
            if node_id is None or node_id.startswith("new:") or not fact.value or fate.propose:
                continue
            if not supersedable(key):
                continue
            asked = self.config.replacement_question and replacement_asked(key)
            batch = values if single_valued(key) else replacements if asked else None
            if batch is None and not self.config.restatement_question:
                continue
            language = detect(fact.statement)
            if node_id not in lives:
                lives[node_id] = self.store.facts([node_id], self.view)
            live = lives[node_id]
            if any(fold(old.statement) == fold(fact.statement) for old in live) or (
                # an element of a set, or an unsure key: it closes nothing
                sets.cardinality(fact.attribute, fact.cardinality, live) in (sets.MANY, sets.UNSURE)
            ):
                continue
            norms = self._norms(live) if fact.value_norm else {}
            # SPEC 5.4: what a new observation decides needs no question; its own
            # source is not stored yet, so no held fact comes from it
            observed = (self._read_at, "") if self._observation(fact, fate)[0] else None
            for root, members in _groups(live, key):
                if any(m.value and normalize(m.value) == normalize(fact.value) for m in members):
                    continue
                verdict = _decide(
                    members,
                    fact,
                    language,
                    None,
                    external=fate.provenance == EXTERNAL,
                    several=self.config.replacement_question,
                    norms=norms,
                    observed=observed,
                    bound=fate.provenance if fate.self_side else None,
                )
                target = (index, root)
                if verdict == (SUPERSEDE, "same_language"):
                    held = _restated(members, fact, language)
                    if self.config.restatement_question and held is not None:
                        self._pair(restatements, target, node_id, key, _sides(held, fact))
                elif verdict[1] == UNKNOWN and batch is not None:
                    sides = _sides(members[0], fact, language)
                    if batch is replacements:
                        self._quote(sides, members[0], fate)
                    self._pair(batch, target, node_id, key, sides)
        return values, replacements, restatements

    def _quote(
        self, sides: tuple[dict[str, Any], dict[str, Any]], first: Fact, fate: _Fate
    ) -> None:
        """SPEC 5.4: the statements may have lost "now" or "also"; the quotes keep them."""
        if (said := self._stored_quote(first)) is not None:
            sides[0]["quote"] = said
        if (said := self._new_quote(fate)) is not None:
            sides[1]["quote"] = said

    def _pair(
        self,
        batch: _Pairs,
        target: tuple[int, str],
        node_id: str,
        key: str | None,
        sides: tuple[dict[str, Any], dict[str, Any]],
    ) -> None:
        """One pair of a batched question, and the (fact, group) its verdict is for."""
        batch[0].append(
            {
                "pair": len(batch[0]),
                "subject": self.store.get_node(node_id).name,
                "key": key,
                "a": sides[0],
                "b": sides[1],
            }
        )
        batch[1].append(target)

    def _norms(self, live: Sequence[Fact]) -> dict[str, str]:
        """The comparison forms of the live facts a new one is compared with
        (SPEC 5.4); none when the switch is off."""
        if not self.config.value_norm:
            return {}
        return self.store.value_norms([f.id for f in live])

    def _stored_quote(self, fact: Fact) -> str | None:
        """The exact words of a stored fact's source, read through the write's view
        (P9); None when the source is not retained, not visible or not located."""
        if fact.source_id is None or fact.excerpt_start is None or fact.excerpt_end is None:
            return None
        source = self.store.source_if_visible(fact.source_id, self.view)
        if source is None or source.text is None:
            return None
        said = source.text[fact.excerpt_start : fact.excerpt_end].strip()
        return said[:PAIR_QUOTE_CHARS] or None

    def _new_quote(self, fate: _Fate) -> str | None:
        """The new fact's words in the source being extracted, without the label
        the service wrote in front of its part."""
        if fate.span is None:
            return None
        start, end = fate.span
        for part in self.source.parts:
            if part.start < end <= part.end:
                start = max(start, part.start)
                break
        said = self.source.text[start:end].strip()
        return said[:PAIR_QUOTE_CHARS] or None

    def _ask(
        self,
        plan: _Plan,
        batch: _Pairs,
        messages: Callable[[Sequence[dict[str, Any]]], list[dict[str, str]]],
        parse: Callable[[str, int], tuple[dict[int, str], int]],
        tally: dict[str, int],
    ) -> None:
        """One batched call for one question; its verdicts go to the plan."""
        pairs, targets = batch
        if not pairs:
            return
        try:
            content = self._call(messages(pairs), self.config.adjudication_cap(len(pairs)))
        except ModelError as exc:
            if exc.reason != TRUNCATED:
                raise
            content = ""  # truncated twice: every pair stays unsure
        verdicts, rejected = parse(content, len(pairs))
        self.report.rejected.keys += rejected
        for pair, target in enumerate(targets):
            plan.adjudicated[target] = verdicts[pair]
            tally[verdicts[pair]] += 1

    def _known_keys(self) -> list[tuple[str, list[str]]]:
        """The canonical keys of the known subjects' live facts, for the prompt; a
        bare family key is never offered for reuse (it never supersedes). A set
        shows its live elements (SPEC 5.4), so that a removal can name one."""
        nodes = self.resolver.known_nodes(self.config.known_names_in_prompt)
        ids = [node_id for node_id, _ in nodes]
        stored = self.store.attribute_keys(ids, self.view)
        elements: dict[tuple[str, str], list[str]] = {}
        shown = sets.ELEMENTS_LOADED
        for element in self.store.flagged_facts(ids, self.view, flag=sets.MANY, limit=shown):
            key = canonical_key(element.attribute)
            if key and element.value and element.node_id:
                values = elements.setdefault((element.node_id, key), [])
                if element.value not in values:
                    values.append(element.value)
        out: list[tuple[str, list[str]]] = []
        for node_id, name in nodes:
            keys = list(
                dict.fromkeys(
                    k for k in map(canonical_key, stored.get(node_id, ())) if supersedable(k)
                )
            )
            if keys:
                out.append(
                    (
                        name,
                        [
                            sets.set_label(k, elements[node_id, k])
                            if (node_id, k) in elements
                            else k
                            for k in keys[:KEYS_PER_SUBJECT]
                            if k
                        ],
                    )
                )
        return out

    def _align_keys(self, plan: _Plan) -> None:
        """Step 3 for every fact (canonical key, unambiguous variants), then step 4
        (tier 3) for the free-form or topical keys a known subject does not have
        yet. Before the write transaction: no model call happens inside it."""
        pending: list[tuple[int, str, str]] = []
        for index, (offset, proposed) in enumerate(plan.facts):
            key = canonical_key(proposed.attribute)
            if key != proposed.attribute:
                if key is not None:
                    self.report.keys_renamed += 1
                plan.facts[index] = (offset, proposed.model_copy(update={"attribute": key}))
            node_id = plan.mention_keys.get(proposed.subject or "")
            if supersedable(key) and node_id is not None and not node_id.startswith("new:"):
                assert key is not None
                pending.append((index, node_id, key))
        self._match_keys(plan, pending)

    @staticmethod
    def _key_candidates(key: str, existing: list[str]) -> list[str]:
        """What step 4 may match ``key`` with: for a free-form key, the subject's
        other free-form keys; for a topical family key, the same family's other
        topics. A vocabulary key is never renamed nor a target (SPEC 5.4)."""
        if key in existing:
            return []
        if is_free_form(key):
            return [k for k in existing if is_free_form(k)]
        family = family_of(key)
        if family is not None and supersedable(key):
            return [k for k in existing if family_of(k) == family and supersedable(k)]
        return []

    def _match_keys(self, plan: _Plan, pending: list[tuple[int, str, str]]) -> None:
        models = self.ctx.models
        model = models.embeddings_model if models.has_embeddings else None
        threshold = self.config.key_threshold_of(model) if model is not None else None
        if threshold is None or not pending:
            return  # no endpoint, or no threshold for this model: step 4 is off
        stored = self.store.attribute_keys(list({n for _, n, _ in pending}), self.view)
        existing = {
            node_id: list(dict.fromkeys(k for k in map(canonical_key, keys) if k))
            for node_id, keys in stored.items()
        }
        unknown = [
            (i, key, candidates)
            for i, node_id, key in pending
            if (candidates := self._key_candidates(key, existing.get(node_id, [])))
        ]
        if not unknown:
            return
        texts = list(dict.fromkeys(key_text(k) for _, key, found in unknown for k in (key, *found)))
        result = models.embed(texts)
        self.report.calls += 1
        unit = {t: vectors.normalise(v) for t, v in zip(texts, result.vectors, strict=True)}
        for index, key, candidates in unknown:
            mine = unit[key_text(key)]
            scored = [
                (sum(a * b for a, b in zip(mine, unit[key_text(k)], strict=True)), k)
                for k in candidates
            ]
            score, best = max(scored)
            if score >= threshold:
                offset, fact = plan.facts[index]
                plan.facts[index] = (offset, fact.model_copy(update={"attribute": best}))
                self.report.keys_matched += 1

    def _prepare_semantic(self, answers: list[tuple[int, Answer]]) -> None:
        """SPEC 5.1 step 3 (tier 3): embed every mention of the answers that steps
        1-2 leave new, in one call, and hand the nodes close by name vector to the
        resolver as candidates. Only the nodes this source's scopes see are
        compared (P9); a candidate is for the adjudication, never a match."""
        models = self.ctx.models
        model = models.embeddings_model if models.has_embeddings else None
        thresholds = self.config.thresholds(model) if model is not None else None
        if model is None or thresholds is None:
            return  # no endpoint, or uncalibrated (SPEC 6.1)
        resolver = self.resolver
        mentions: list[str] = []
        for _, answer in answers:
            mentions.extend(n.name for n in answer.nodes)
            mentions.extend(f.subject for f in answer.facts if f.subject)
            mentions.extend(m for e in answer.edges for m in (e.src, e.dst))
        fresh = [m for m in dict.fromkeys(mentions) if resolver.resolve(m).method == NEW]
        rows = self.store.vector_rows(self.view, facets=(NAME,), model=model) if fresh else []
        if not rows:
            return
        result = models.embed(fresh)
        self.report.calls += 1
        for mention, values in zip(fresh, result.vectors, strict=True):
            found: list[Candidate] = []
            for index, score in vectors.rank(values, [r.blob for r in rows]):
                if score < thresholds.resolve or len(found) >= SEMANTIC_CANDIDATES:
                    break
                node = self.store.get_node(rows[index].record_id)
                found.append(Candidate(node.id, node.name, node.kind, score, node.summary))
            if found:
                resolver.set_semantic(mention, found)

    def _adjudicate_names(
        self, nodes: Sequence[schema.ProposedNode], text: str
    ) -> dict[str, str | None]:
        """The answers' own node names that stay ambiguous (fuzzy or semantic
        candidates) go to the batched adjudication before the plan is built; a
        name the adjudication does not attach becomes a new node."""
        kinds = {n.name: n.kind for n in nodes}
        resolved = [self.resolver.resolve(name, kind=kind) for name, kind in kinds.items()]
        ambiguous = [r for r in resolved if r.method == AMBIGUOUS]
        if not ambiguous or not self.ctx.models.has_chat:
            return {}
        self._adjudicate(ambiguous, text)
        return {r.mention: r.key for r in ambiguous}

    def _resolve_mentions(
        self,
        plan: _Plan,
        mentions: list[str],
        text: str,
        quotes: Mapping[str, str] | None = None,
    ) -> None:
        """Steps 1-2 for every mention, one batched step 4 for what stays ambiguous
        (tier 2), a new node for what is still unresolved."""
        resolver = self.resolver
        speakers = [m for m in mentions if is_speaker(m)]
        plan.mixed_names.update(m for m in mentions if mixed_script_words(m))
        mentions = [m for m in mentions if not is_speaker(m) and not mixed_script_words(m)]
        for mention in speakers:
            # SPEC 5.1: the declared owner, or no node at all (a free-standing fact)
            if plan.owner_key is not None:
                plan.mention_keys[mention] = plan.owner_key
                plan.order(plan.owner_key)
        for mention in mentions:
            # SPEC 5.1, a known subject the source does not name: a candidate only
            found = resolver.resolve(mention)
            if (
                found.method in (EXACT, FUZZY)
                and found.key is not None
                and self._unnamed_match(plan, found.key, text)
            ):
                plan.unnamed[mention] = found.key
                resolver.contest(mention, [found.key])
        ambiguous = [r for r in map(resolver.resolve, mentions) if r.method == AMBIGUOUS]
        if ambiguous and self.ctx.models.has_chat:
            self._adjudicate(ambiguous, text, quotes)
        for mention in mentions:
            resolution = resolver.resolve(mention)
            if resolution.method == AMBIGUOUS:
                resolver.decide(resolution, None)
            known = plan.unnamed.get(mention)
            if known is not None:
                if resolution.key is None:
                    # not attached, and no new node under a name memory already
                    # holds: its facts are proposals on the known node
                    plan.mention_keys[mention] = known
                    continue
                del plan.unnamed[mention]  # the adjudication attached it
            if resolution.key is None:
                key = f"new:{len(plan.proposals)}"
                plan.proposals[key] = schema.ProposedNode(name=mention)
                resolver.propose(key, mention, "other")
                resolution.key = key
                resolution.method = NEW
            plan.mention_keys[mention] = resolution.key
            plan.order(resolution.key)

    def _adjudicate(
        self, ambiguous: list[Any], text: str, quotes: Mapping[str, str] | None = None
    ) -> None:
        """Step 4, one batched call. A mention the source does not write (SPEC
        5.1, a known subject the source does not name) is shown with the
        sentence its fact quotes, or the start of the source."""
        batch = []
        for resolution in ambiguous:
            snippet = _context(text, resolution.mention)
            if not snippet:
                quote = (quotes or {}).get(resolution.mention, "")
                located = _context(text, quote, width=40) if quote else ""
                snippet = located or " ".join(text[:CONTEXT_HEAD].split())
            batch.append(
                (
                    resolution.mention,
                    snippet,
                    [(c.key, c.name, c.kind, c.summary) for c in resolution.candidates],
                )
            )
        try:
            content = self._call(
                adjudication_messages(batch), self.config.adjudication_cap(len(batch))
            )
        except ModelError as exc:
            if exc.reason != TRUNCATED:
                raise
            # truncated twice: every mention stays undecided (a new node: the safe default)
            content = ""
        decisions, rejected = schema.parse_adjudication(content, len(batch))
        self.report.rejected.keys += rejected
        self.report.mentions_adjudicated += len(batch)
        self.report.adjudication_calls += 1
        for index, resolution in enumerate(ambiguous):
            semantic = {c.key for c in self.resolver.semantic_candidates(resolution.mention)}
            names = {c.key: c.name for c in resolution.candidates}
            self.resolver.decide(resolution, decisions.get(index))
            self.report.adjudications.append(
                {
                    "mention": resolution.mention,
                    "candidates": [
                        {"name": c.name, "score": round(c.score, 4), "semantic": c.key in semantic}
                        for c in resolution.candidates
                    ],
                    "chosen": names.get(resolution.key or "")
                    if resolution.method == ADJUDICATED
                    else None,
                }
            )

    # ------------------------------------------------------------------ writing

    def _write(self, source_id: str, text: str, plan: _Plan) -> None:
        store, report = self.store, self.report
        ids: dict[str, str] = {}  # key -> node id
        created: list[Node] = []  # SPEC 5.1: a source may name both forms itself
        for key in plan.ordered:
            if key.startswith("new:"):
                proposal = plan.proposals[key]
                # SPEC 4.6: the owner's node lives in the scope that declares it
                scopes = plan.node_scopes.get(key, self.scopes)
                view = self.view if key not in plan.node_scopes else write_view(scopes)
                # the exact check again, inside the lock: a remember may have raced us
                found = store.find_node(proposal.name, view)
                if found is not None:
                    ids[key] = found.id
                    report.nodes_matched += 1
                    self._add_aliases(found, [*proposal.aliases, *plan.aliases.get(key, [])])
                    continue
                node = store.create_node(
                    scopes=scopes,
                    name=proposal.name,
                    kind=proposal.kind,
                    aliases=proposal.aliases,
                )
                ids[key] = node.id
                report.nodes_created += 1
                report.aliases_kept += len(node.aliases)
                # a later answer of the same job may have named it once more
                self._add_aliases(node, plan.aliases.get(key, []))
                self._propose_form_merges(node, source_id, created)
                created.append(node)
            else:
                ids[key] = key
                report.nodes_matched += 1
                self._add_aliases(store.get_node(key), plan.aliases.get(key, []))
        self._propose_aliases(source_id, text, plan, ids)
        activity = self._activity_id(plan, ids)

        touched: list[str] = []
        seen_statements: set[tuple[str | None, str]] = set()
        for index, (offset, fact) in enumerate(plan.facts):
            node_id = ids.get(plan.mention_keys.get(fact.subject or "", ""))
            if node_id is None and plan.fates[index].unnamed:
                node_id = plan.unnamed.get(fact.subject or "")  # SPEC 5.1: a known node
            signature = (node_id, fold(fact.statement))
            if signature in seen_statements:
                report.facts_duplicate += 1
                continue
            seen_statements.add(signature)
            fate = plan.fates[index]
            # SPEC 4.2, 5.8: a rule, or the close's episode, beside its subject
            linked = self._links(fact, fate, activity)
            if fate.propose:
                if fate.self_rule and (_rule_same(plan, index) or self._rule_known(node_id, fact)):
                    report.facts_self_echo += 1  # SPEC 4.2: said again, asked once
                    continue
                why = SELF_OUTSIDE if fate.self_outside else SELF_RULE if fate.self_rule else None
                self._propose_fact(
                    source_id, text, fact, node_id, fate, plan.forced, why, activities=linked
                )
                continue
            verdicts = {g: v for (i, g), v in plan.adjudicated.items() if i == index}
            written = self._write_fact(
                source_id,
                text,
                offset,
                fact,
                node_id,
                plan.forced,
                verdicts,
                fate,
                activities=linked,
            )
            if written is not None and node_id is not None and node_id not in touched:
                touched.append(node_id)

        self._write_edges(source_id, text, plan, ids, touched)

    def _links(self, fact: ProposedFact, fate: _Fate, activity: str | None) -> list[str]:
        """SPEC 4.2, 5.8: a rule is linked to the activity it was learnt during, a
        close's episode to every activity of its session."""
        if fact.episode and self._episode_activities:
            return list(self._episode_activities)
        return [activity] if activity is not None and fate.method else []

    def _activity_id(self, plan: _Plan, ids: Mapping[str, str]) -> str | None:
        """SPEC 4.2: the node of the activity the source's rules are linked to,
        kept for the session when the source named it."""
        key = plan.activity_key
        # a known node no fact names is not among the nodes written: its id is its key
        activity = None if key is None else ids.get(key, None if key.startswith("new:") else key)
        if plan.activity_named:
            self._activity_written = activity
        elif activity is not None:
            self.report.activity_carried += 1
        return activity

    def _propose_form_merges(self, node: Node, source_id: str, created: Sequence[Node]) -> None:
        """SPEC 5.1, one subject in two forms: a new node a known one - or one
        this same write created - may name in another form (a slug or a file
        name and a title) is paired with it by a merge proposal, for the review
        queue and the dream; never merged here. Only nodes of the same scope set
        (dream rule 12: never widen visibility)."""
        store = self.store
        siblings = [n.id for n in created if form_duplicate(node.name, node.kind, n.name, n.kind)]
        for other_id in [*self.resolver.form_nodes(node.name, node.kind), *siblings]:
            if set(store.scopes_of(other_id)) != set(node.scopes):
                continue
            store.add_proposal(
                scopes=node.scopes,
                kind=MERGE,
                node_id=node.id,
                value=other_id,
                source_id=source_id,
            )
            self.report.duplicates_merge_proposed += 1

    def _add_aliases(self, node: Node, aliases: Iterable[str]) -> None:
        """Add the grounded aliases a node does not carry yet, and count them."""
        known = {normalize(node.name), *(normalize(a) for a in node.aliases)}
        new = []
        for alias in aliases:
            norm = normalize(alias)
            if norm and norm not in known:
                known.add(norm)
                new.append(alias)
        if new:
            self.store.update_node(node.id, add_aliases=new)
            self.report.aliases_kept += len(new)

    def _propose_aliases(
        self, source_id: str, text: str, plan: _Plan, ids: Mapping[str, str]
    ) -> None:
        """Record the aliases the source does not hold as proposals (SPEC 4.4b),
        with the scope set of the source and the span of the node's name in it."""
        for key, name, alias in plan.doubtful:
            node_id = ids.get(key)
            if node_id is None:
                continue
            node = self.store.get_node(node_id)
            if normalize(alias) in {normalize(node.name), *(normalize(a) for a in node.aliases)}:
                continue
            self.store.add_proposal(
                scopes=self.scopes,
                kind="alias",
                node_id=node_id,
                value=alias,
                source_id=source_id,
                excerpt=locate(text, name),
            )
            self.report.aliases_proposed += 1

    def _write_fact(
        self,
        source_id: str,
        text: str,
        offset: int,
        fact: ProposedFact,
        node_id: str | None,
        forced: str | None,
        verdicts: dict[str, str] | None = None,
        fate: _Fate | None = None,
        *,
        activities: Sequence[str] = (),
    ) -> Fact | None:
        store, report = self.store, self.report
        fate = fate or _Fate(locate(text, fact.quote, hint=offset), None, False, False, False)
        span = fate.span
        language = detect(fact.statement)
        flags = self._flags(fact, text, span, forced, fate)
        observed, by_tool = self._observation(fact, fate)
        if observed:
            flags.append(OBSERVED)
        live = self._live(node_id, fate)
        echo = fate.self_side and fate.provenance == ASSISTANT  # SPEC 4.2: not evidence
        for old in live:
            if fold(old.statement) == fold(fact.statement):
                report.facts_duplicate += 1
                report.facts_self_echo += echo
                if observed and not echo:
                    self._seen_again(live, old.same_as or old.id)
                return None
        # SPEC 5.4: a bare family key ("preference" without a topic) never drives a
        # supersession or a conflict; when we cannot tell, we do not supersede
        speaking = fate.speaker_free
        # SPEC 5.4 (sets): None for a vocabulary key, else one / many / unsure
        kind = (
            sets.cardinality(fact.attribute, fact.cardinality, live)
            if node_id or speaking
            else None
        )
        removed_by: list[Fact] = []
        fact, removed, refused = self._elements(live, fact, kind, fate, flags, removed_by)
        judged = _Judgement()
        if (node_id or speaking) and fact.value and supersedable(canonical_key(fact.attribute)):
            judged = self._judge(
                live,
                fact,
                language,
                verdicts or {},
                fate,
                kind,
                removed,
                observed=(self._read_at, source_id) if observed else None,
            )
        judged.superseded[:0] = removed_by
        if echo and self._echoed(judged):
            return None
        if judged.duplicate_of is not None:
            # nothing new is written, but what the new evidence closes is closed,
            # by the fact that already states the same value
            report.facts_duplicate += 1
            if observed:
                self._seen_again(live, judged.duplicate_of)
            self._close(judged.superseded, fact.valid_from, judged.duplicate_of)
            self._flag_refused(judged.duplicate_of, refused)
            self._reach_duplicates(node_id, fact, language, fate, judged.duplicate_of)
            return None
        if judged.kept is not None:
            # SPEC 5.4: the fact held stays as it is; groups the new value proves
            # outdated on their own are closed by it, as for a duplicate
            report.facts_kept_same += 1
            if observed:
                self._seen_again(live, judged.kept)
            self._close(judged.superseded, fact.valid_from, judged.kept)
            self._flag_refused(judged.kept, refused)
            self._reach_duplicates(node_id, fact, language, fate, judged.kept)
            return None
        conflicts_with = refused or judged.conflicts_with
        if conflicts_with is not None and CONFLICT not in flags:
            flags.append(CONFLICT)
        written = store.add_fact(
            scopes=self.scopes,
            statement=fact.statement,
            origin=EXTRACTED,
            node_id=node_id,
            source_id=source_id,
            excerpt=span,
            attribute=canonical_key(fact.attribute),
            value=fact.value,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            confidence=fact.confidence,
            flags=flags,
            language=None if language == UNDETERMINED else language,
            conflicts_with=conflicts_with,
            same_as=judged.same_as,
            provenance=fate.provenance,
            **self._part_fields(fate.provenance, fate.part),
            value_norm=fact.value_norm if self.config.value_norm and fact.value else None,
            observed_at=self._read_at if observed else None,
            activities=activities,
        )
        self._count_written(fate, conflicts_with)
        report.activity_linked += bool(activities)
        report.facts_observed += observed
        report.facts_observed_by_tool += by_tool
        self.report.elements["added"] += sets.MANY in flags
        self._close(judged.superseded, fact.valid_from, written.id)
        self._reach_duplicates(node_id, fact, language, fate, written.id)
        return written

    def _echoed(self, judged: _Judgement) -> bool:
        """SPEC 4.2: the self's own fact said again (the same value of the same
        key, or the value held told in other words) writes, closes and moves
        nothing; counted."""
        if judged.duplicate_of is None and judged.kept is None:
            return False
        report = self.report
        report.facts_duplicate += judged.duplicate_of is not None
        report.facts_kept_same += judged.duplicate_of is None
        report.facts_self_echo += 1
        return True

    def _judge(
        self,
        live: Sequence[Fact],
        fact: ProposedFact,
        language: str,
        verdicts: Mapping[str, str],
        fate: _Fate,
        kind: str | None = None,
        removed: set[str] | frozenset[str] = frozenset(),
        observed: tuple[str | None, str] | None = None,
    ) -> _Judgement:
        """What a new value does to each equivalence group of its key (SPEC 5.4);
        a group ``removes`` named is left to the removal, and an element of a set
        is decided by the set's rule (SPEC 5.4, sets). ``observed``: the day and
        the source of a new observation (SPEC 5.4, observations)."""
        judged, value = _Judgement(), normalize(fact.value or "")
        norms = self._norms(live) if fact.value_norm else {}
        for root, members in _groups(live, fact.attribute):
            if root in removed:
                continue  # the element this fact removes or corrects
            if any(m.value and normalize(m.value) == value for m in members):
                # a duplicate of this group; the other groups are still decided
                judged.duplicate_of = judged.duplicate_of or root
                continue
            element = sets.decide(members, fact.value or "", kind)  # None: not a set's question
            self.report.elements[sets.UNSURE] += element == (KEEP_CONFLICT, sets.UNSURE)
            decision, why = element or _decide(
                members,
                fact,
                language,
                verdicts.get(root),
                external=fate.provenance == EXTERNAL,
                never_close=fate.speaker_free,
                several=self.config.replacement_question,
                norms=norms,
                observed=observed,
                bound=fate.provenance if fate.self_side else None,
            )
            self.report.values_same += why == SAME
            if decision == KEEP:
                judged.kept = judged.kept or root
            elif decision == LINK:
                judged.same_as = judged.same_as or root
                self.report.values_norm += why == NORM
            elif decision == SUPERSEDE:
                judged.superseded.extend(members)  # the whole group states the outdated value
                self.report.values_different += why == DIFFERENT
            elif decision == KEEP_CONFLICT:
                judged.conflicts_with = judged.conflicts_with or root
            # BOTH: both values hold at once, neither closed nor in conflict
        return judged

    def _elements(
        self,
        live: Sequence[Fact],
        fact: ProposedFact,
        kind: str | None,
        fate: _Fate,
        flags: list[str],
        superseded: list[Fact],
    ) -> tuple[ProposedFact, set[str], str | None]:
        """SPEC 5.4 (sets): an element of a set is flagged ``many``; ``removes``
        closes only the elements of a set it names, where the guards allow a
        close (added to ``superseded``), else flags the conflict. Returns the
        fact to write (a removal without a value loses its key: a change, not an
        element), the groups it named, and the element it conflicts with."""
        fact, removed, conflicts_with = self._remove_elements(live, fact, kind, fate, superseded)
        if conflicts_with is not None:
            flags.append(CONFLICT)  # a removal that names no element, or may not close it
        if kind == sets.MANY and fact.attribute and fact.value:
            flags.append(sets.MANY)
        return fact, removed, conflicts_with

    def _flag_refused(self, held: str, conflicts_with: str | None) -> None:
        """A correction onto an element already held writes nothing; a removal it
        could not make stays visible on that element (SPEC 5.4)."""
        if conflicts_with is not None:
            self.store.flag_fact(held, CONFLICT, conflicts_with=conflicts_with)

    def _remove_elements(
        self,
        live: Sequence[Fact],
        fact: ProposedFact,
        kind: str | None,
        fate: _Fate,
        superseded: list[Fact],
    ) -> tuple[ProposedFact, set[str], str | None]:
        """What ``removes`` closes, and the element it conflicts with: the first
        one it may not close, or the set's first when it names none."""
        if kind != sets.MANY or not fact.removes:
            return fact, set(), None
        groups = _groups(live, fact.attribute)
        found = sets.matching(groups, fact.removes)
        if not fact.value:
            fact = fact.model_copy(update={"attribute": None})
        if not found:
            self.report.elements["unmatched"] += 1
            return fact, set(), groups[0][0] if groups else None
        conflicts_with: str | None = None
        members_of = dict(groups)
        for root in found:
            members = members_of[root]
            if sets.closable(
                members,
                fact.valid_from,
                external=fate.provenance == EXTERNAL,
                never_close=fate.speaker_free
                or (fate.self_side and any(m.provenance != fate.provenance for m in members)),
            ):
                superseded.extend(members)  # the whole group states the removed element
                self.report.elements["removed"] += 1
            else:
                conflicts_with = conflicts_with or root
        return fact, set(found), conflicts_with

    def _observation(self, fact: ProposedFact, fate: _Fate) -> tuple[bool, bool]:
        """SPEC 4.2: is the fact observed, and is it so by the provenance rule
        alone? The model's ``observed`` (read only with the intent reading on),
        or a fact whose quote lies in an external part of kind ``tool``."""
        said = self.config.intent and fact.duration == intent_reading.OBSERVED
        tool = (
            fate.provenance == EXTERNAL
            and fate.part is not None
            and fate.part.provenance == EXTERNAL
            and fate.part.kind == TOOL
        )
        return said or tool, tool and not said

    def _seen_again(self, live: Sequence[Fact], root: str) -> None:
        """SPEC 4.2: a held value observed again - every observed fact of its
        group takes the source's day."""
        if self._read_at is None:
            return
        for held in live:
            if (held.same_as or held.id) == root and held.observed_at is not None:
                self.report.facts_observed_again += self.store.observe_again(held.id, self._read_at)

    def _close(self, superseded: Sequence[Fact], valid_to: str | None, by: str) -> None:
        for old in superseded:
            self.store.close_fact(old.id, valid_to=valid_to, superseded_by=by)
            self.report.facts_superseded += 1

    def _reach_duplicates(
        self, node_id: str | None, fact: ProposedFact, language: str, fate: _Fate, by: str
    ) -> None:
        """SPEC 5.4: a new value reaches a known duplicate (a pending merge
        proposal pairs the two nodes). A live fact there that the new one would
        close or contradict on one node - same key, another value - is flagged
        ``conflict`` against ``by``, never closed: until the merge is decided we
        cannot tell the two nodes are one subject."""
        if node_id is None or not fact.value or not supersedable(canonical_key(fact.attribute)):
            return
        store = self.store
        for other_id in store.merge_candidates(node_id, self.view):
            held = store.facts([other_id], self.view)
            kind = sets.cardinality(fact.attribute, fact.cardinality, held)
            for _root, members in _groups(held, fact.attribute):
                if any(m.value and normalize(m.value) == normalize(fact.value) for m in members):
                    continue  # the duplicate already holds the new value
                if sets.decide(members, fact.value, kind) is not None:
                    continue  # an element of a set, or an unsure key: it closes nothing
                decision, _why = _decide(
                    members,
                    fact,
                    language,
                    None,
                    external=fate.provenance == EXTERNAL,
                    several=self.config.replacement_question,
                )
                if decision not in (SUPERSEDE, KEEP_CONFLICT):
                    continue  # the same value in another form, or both true at once
                for member in members:
                    if CONFLICT not in member.flags:
                        store.flag_fact(member.id, CONFLICT, conflicts_with=by)
                        self.report.duplicates_flagged += 1

    def _live(self, node_id: str | None, fate: _Fate) -> list[Fact]:
        """The live facts a new one is compared with: its node's, or, for a free
        speaker fact, the other free speaker facts in view (compared, never
        closed: SPEC 5.4)."""
        if node_id:
            return self.store.facts([node_id], self.view)
        if fate.speaker_free:
            return self.store.free_facts(self.view, flag=SPEAKER)
        return []

    def _count_written(self, fate: _Fate, conflicts_with: str | None) -> None:
        report = self.report
        report.facts_written += 1
        if fate.speaker_free:
            report.facts_speaker_free += 1
        elif fate.about_owner:
            report.facts_speaker_owner += 1
        report.facts_self += fate.about_agent
        report.facts_method += fate.method
        if conflicts_with:
            report.facts_conflict += 1

    def _flags(
        self,
        fact: ProposedFact,
        text: str,
        span: tuple[int, int] | None,
        forced: str | None,
        fate: _Fate,
    ) -> list[str]:
        """P12's flag when the statement is not in its source's language, and
        ``speaker`` on a free fact about the person speaking (SPEC 4.2)."""
        expected = forced or (language_of_span(text, *span) if span else detect(text))
        flags: list[str] = []
        if writing_mismatch(fact.statement, expected):
            flags.append(LANGUAGE_MISMATCH)
            self.report.facts_language_mismatch += 1
        if fate.speaker_free:
            flags.append(SPEAKER)
        if fate.self_side and not fate.propose and self._source_external() == VOUCHED:
            flags.append(VOUCHED)  # SPEC 4.2: learnt from what the host vouched for
        if fact.episode and self.config.intent:
            flags.append(intent_reading.EPISODE)  # SPEC 4.2: what a frame or a session left
        return flags

    def _propose_fact(
        self,
        source_id: str,
        text: str,
        fact: ProposedFact,
        node_id: str | None,
        fate: _Fate,
        forced: str | None,
        why: str | None = None,
        *,
        activities: Sequence[str] = (),
    ) -> None:
        """A fact the write path does not write on its own - about the owner, not
        in the owner's words (P6), shown by its source to be about another
        subject (SPEC 5.1), or whose intent is in doubt (SPEC 4.2), whoever it is
        about (``why``) - becomes a fact proposal, decided only by a ``review``
        token (SPEC 4.4b)."""
        language = detect(fact.statement)
        flags = self._flags(fact, text, fate.span, forced, fate)
        observed, _by_tool = self._observation(fact, fate)
        if observed:
            flags.append(OBSERVED)
        kind = sets.cardinality(fact.attribute, fact.cardinality, self._live(node_id, fate))
        if kind == sets.MANY and fact.value:
            flags.append(sets.MANY)  # SPEC 5.4: accepted, it is an element of the set
        self.store.add_proposal(
            scopes=self.scopes,
            kind="fact",
            node_id=node_id,
            value=fact.statement,
            source_id=source_id,
            excerpt=fate.span,
            confidence=fact.confidence,
            payload={
                "attribute": canonical_key(fact.attribute),
                "value": fact.value,
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "language": None if language == UNDETERMINED else language,
                "flags": flags,
                "provenance": fate.provenance,
                **self._part_fields(fate.provenance, fate.part),
                "observed_at": self._read_at if observed else None,
                "confidence": fact.confidence,
                **({"why": why, "subject": fact.subject} if why else {}),
                **({"activities": list(activities)} if activities else {}),
            },
        )
        self.report.facts_proposed += 1
        self.report.activity_linked += bool(activities)
        self.report.facts_neighbour += fate.neighbour
        self.report.facts_unnamed += fate.unnamed
        self.report.facts_self_proposed += fate.self_side

    # ------------------------------------------------------------------ intent (SPEC 4.2, 5.8)

    def _about_speaker(self, subject: str | None) -> bool:
        return bool(subject) and (is_speaker(subject or "") or self._means_owner(subject))

    def _read_intent(
        self,
        answers: list[tuple[int, Answer]],
        state: session_state.SessionState,
        reference: str | None,
    ) -> _IntentOutcome:
        """Each answer read in turn, the frame carried from one to the next and
        from the session's state: what goes on to the plan, what is proposed,
        what is kept as session state."""
        report = self.report.intent = intent_reading.IntentReport()
        carry = intent_reading.Carry(state.frame, state.frame_episode)
        outcome = _IntentOutcome()
        for offset, answer in answers:
            reading, carry = intent_reading.read(
                answer,
                carry,
                about_speaker=self._about_speaker,
                report=report,
                date=reference,
            )
            outcome.kept.append((offset, reading.kept))
            outcome.proposed.extend((offset, f, reading.frame) for f in reading.proposed)
            outcome.session.extend((offset, f) for f in reading.session)
            if self._session_id:
                # SPEC 4.2, 5.8: an observation stays memory, and its note tells
                # the session's close what was done; the close never keeps it
                outcome.session.extend(
                    (offset, f.model_copy(update={"duration": intent_reading.OBSERVED}))
                    for f in reading.kept.facts
                    if self._noted_observation(f, offset)
                )
            outcome.ended = outcome.ended or reading.ended
            self._unmeant = self._unmeant or reading.frame != intent_reading.SINCERE
        outcome.carry = carry
        return outcome

    def _noted_observation(self, fact: ProposedFact, offset: int) -> bool:
        """SPEC 4.2: read as observed - by the model, or by the provenance rule
        for a lasting fact quoted from a ``tool`` part."""
        if fact.duration == intent_reading.OBSERVED:
            return True
        if fact.duration != intent_reading.LASTING:
            return False
        part = self.source.part_of(locate(self.source.text, fact.quote, hint=offset))
        return part is not None and part.provenance == EXTERNAL and part.kind == TOOL

    def _write_intent(
        self,
        source_id: str,
        text: str,
        outcome: _IntentOutcome,
        state: session_state.SessionState,
        forced: str | None,
        session_id: str | None = None,
    ) -> None:
        """Inside the write transaction: the facts whose intent is in doubt
        become fact proposals on the node their subject already names (none is
        created for them), the session's state takes the facts of the work in
        progress, the frame now open and, in a session the host named, the
        turn's trace when every answer was read as sincere (SPEC 5.8)."""
        for offset, fact, frame in outcome.proposed:
            span = locate(text, fact.quote, hint=offset)
            part = self.source.part_of(span)
            provenance = part.provenance if part is not None else None
            speaker = self._about_speaker(fact.subject)
            node: Node | None = None
            if speaker:
                node = self.owner_node
            elif fact.subject:
                node = self.store.find_node(fact.subject, self.view)
            free = speaker and self.owner is None
            method = self.agent is not None and canonical_key(fact.attribute) == METHOD
            fate = _Fate(span, provenance, speaker, free, True, part=part, method=method)
            why = (
                "intent:episode"
                if fact.episode
                else f"intent:frame:{frame}"
                if frame != intent_reading.SINCERE
                else f"intent:modality:{fact.modality or intent_reading.UNSURE}"
            )
            self._propose_fact(
                source_id,
                text,
                fact,
                node.id if node else None,
                fate,
                forced,
                why,
                activities=self._linked_activity() if method else [],
            )
        new: list[session_state.NewSessionFact] = []
        for offset, fact in outcome.session:
            span = locate(text, fact.quote, hint=offset)
            # P6 at the close as at the write: a value only an external part
            # holds is not the speaker's words, whoever quoted it
            outside = self._value_from_outside(fact.value)
            part = self._outside_part(fact.value) if outside else self.source.part_of(span)
            provenance = EXTERNAL if outside else (part.provenance if part else None)
            new.append(
                session_state.NewSessionFact(
                    source_id=source_id,
                    duration=fact.duration or intent_reading.UNSURE,
                    fact=fact,
                    provenance=provenance,
                    span=span,
                    **self._part_fields(provenance, part),
                )
            )
        trace: tuple[str, str] | None = None
        if session_id and outcome.kept and not self._unmeant:
            excerpt = session_state.trace_of(text, self.source.parts)
            trace = (source_id, excerpt) if excerpt else None
        session_state.save(
            self.store,
            state,
            frame=outcome.carry.frame,
            frame_episode=outcome.carry.episode,
            ended=outcome.ended,
            facts=new,
            trace=trace,
            activity=self._activity_written if session_id else None,
        )

    def _linked_activity(self) -> list[str]:
        """SPEC 4.2: the activity a rule of this source is linked to - the one the
        source named and wrote, else the session's current one."""
        if self._activity_written is not None:
            return [self._activity_written]
        return [self._carried.id] if self._carried is not None else []

    # ------------------------------------------------------------------ answers already read

    def prepare(
        self,
        text: str,
        parts: Sequence[Part],
        answers: list[tuple[int, Answer]],
        *,
        session_id: str | None = None,
        read_at: str | None = None,
        external: str = UNVOUCHED,
        activities: Sequence[str] = (),
    ) -> _Plan:
        """Plan answers that were read already (a session's consolidation, SPEC
        5.8): no extraction call, the same resolution, contradiction and
        provenance rules (P6) - only the adjudications the plan needs may call
        the model. ``external``: what the source held from outside (SPEC 4.4);
        ``parts`` are only the notes' own; ``activities``: the session's, linked
        to the episode (SPEC 5.8) - a kept rule is linked to none, since which
        activity was current at its turn is not kept. Write the plan with
        ``commit``."""
        self.source = SourceText(text, tuple(parts))
        self._external = external
        # SPEC 4.2: a close names no activity; it keeps the session's current one
        self._activity, self._carried = None, None
        nodes = (self._activity_node(a) for a in activities)
        self._episode_activities = [n.id for n in nodes if n is not None]
        self._part_tokens = {}
        self._read_at = read_at
        self._session_id = session_id
        self.report.tier = 2 if self.ctx.models.has_chat else 0
        return self._plan(text, answers, self.config.forced_language(self.scopes))

    def commit(self, source_id: str, text: str, plan: _Plan) -> None:
        """Write a prepared plan into an existing source; the caller holds the
        store's lock and transaction."""
        self.report.source_id = source_id
        self._write(source_id, text, plan)

    def _write_edges(
        self,
        source_id: str,
        text: str,
        plan: _Plan,
        ids: Mapping[str, str],
        touched: list[str],
    ) -> None:
        store, report, config = self.store, self.report, self.config
        involved = list(dict.fromkeys(ids[k] for k in plan.ordered if k in ids))
        existing = {(e.src, e.dst, e.type): e for e in store.edges_between(involved, self.view)}
        me = ids.get(plan.agent_key) if plan.agent_key is not None else None

        def observe(
            src: str,
            dst: str,
            type_: str,
            weight: float,
            confirm_after: int,
            evidence: bool = True,
        ) -> None:
            edge = existing.get((src, dst, type_))
            if edge is None and type_ == schema.COOCCURRENCE:
                edge = existing.get((dst, src, type_))
            if edge is None:
                created = store.add_edge(
                    scopes=self.scopes,
                    src=src,
                    dst=dst,
                    type=type_,
                    origin=EXTRACTED,
                    status=PROPOSED,
                    weight=weight,
                    source_id=source_id,
                )
                existing[(src, dst, type_)] = created
                report.edges_proposed += 1
                return
            if not evidence:
                return  # SPEC 4.2: the self's own words never confirm its edges
            count = store.add_evidence(edge.id, source_id)
            report.edges_evidenced += 1
            if edge.status == PROPOSED and count >= confirm_after:
                existing[(edge.src, edge.dst, edge.type)] = store.confirm_edge(edge.id)
                report.edges_confirmed += 1

        for proposed in plan.edges:
            if proposed.src in plan.unnamed or proposed.dst in plan.unnamed:
                continue  # SPEC 5.1: a known subject the source does not name
            src = ids.get(plan.mention_keys.get(proposed.src, ""))
            dst = ids.get(plan.mention_keys.get(proposed.dst, ""))
            if src is None or dst is None or src == dst:
                continue
            evidence = (
                me not in (src, dst)
                or self.source.provenance_of(locate(text, proposed.quote)) == OWNER
            )
            observe(src, dst, proposed.type, 1.0, config.confirm_after_sources, evidence)
            for node_id in (src, dst):
                if node_id not in touched:
                    touched.append(node_id)

        for a, b in self._cooccurrences(text, plan, ids, involved):
            if me in (a, b):
                continue  # SPEC 4.2: the self's label stands before every answer
            src, dst = (a, b) if a < b else (b, a)
            observe(
                src,
                dst,
                schema.COOCCURRENCE,
                config.cooccurrence_weight,
                config.cooccurrence_confirm_after_sources,
            )

    def _cooccurrences(
        self, text: str, plan: _Plan, ids: Mapping[str, str], involved: list[str]
    ) -> list[tuple[str, str]]:
        """SPEC 5.2: the pairs of nodes that share a sentence, never merely the same
        turn. A node is in a sentence by its name, an alias, a mention that resolved
        to it, or the quote of one of its facts or edges."""
        ids_of: dict[str, str | None] = {
            mention: ids.get(key) for mention, key in plan.mention_keys.items()
        }
        forms: dict[str, list[str]] = {}
        for node_id in involved:
            node = self.store.get_node(node_id)
            forms[node_id] = [node.name, *node.aliases]
        for mention, mentioned in ids_of.items():
            if mentioned is not None:
                forms.setdefault(mentioned, []).append(mention)
        spans: dict[str, list[tuple[int, int]]] = {}
        quoted = [
            (ids_of.get(f.subject or ""), locate(text, f.quote, hint=o)) for o, f in plan.facts
        ]
        for edge in plan.edges:
            edge_span = locate(text, edge.quote)
            quoted.extend((ids_of.get(mention), edge_span) for mention in (edge.src, edge.dst))
        for quoted_id, span in quoted:
            if quoted_id is not None and span is not None:
                spans.setdefault(quoted_id, []).append(span)
        return cooccurring(
            text, forms, spans, order=involved, limit=self.config.max_cooccurrence_nodes
        )


LINK, SUPERSEDE, KEEP_CONFLICT, BOTH = "link", "supersede", "conflict", "both"
KEEP = "keep"  # SPEC 5.4: the same value told again in one language
NORM = "norm"  # SPEC 5.4: linked by the comparison forms of the two values
_Pairs = tuple[list[dict[str, Any]], list[tuple[int, str]]]


def _rule_same(plan: _Plan, index: int) -> bool:
    """SPEC 4.2: did the model say the rule at ``index`` restates a known one?"""
    return any(
        verdict == SAME
        for (fact, group), verdict in plan.adjudicated.items()
        if fact == index and group.startswith(RULE_GROUP)
    )


def _language_of(fact: Fact) -> str:
    return fact.language or detect(fact.statement)


def _one_language(language: str, statement: str, member: Fact) -> bool:
    """Are a new statement and a stored fact within one language (SPEC 5.4)?
    Both determined: when equal. Either undetermined: when written in one
    script (another script is surely another language; within one script, a
    closure replaces a value with an equivalent one, measured harmless)."""
    other = _language_of(member)
    if UNDETERMINED not in (language, other):
        return language == other
    script = main_script(statement)
    # no letters on either side ("2024-05-01"): we cannot tell, so not one language
    return script is not None and script == main_script(member.statement)


def _groups(live: Sequence[Fact], key: str | None) -> list[tuple[str, list[Fact]]]:
    """The live facts carrying ``key``, by equivalence group, oldest first."""
    out: dict[str, list[Fact]] = {}
    for fact in sorted(live, key=lambda f: (f.created_at, f.id)):
        if same_key(fact.attribute, key):
            out.setdefault(fact.same_as or fact.id, []).append(fact)
    return list(out.items())


def _sides(
    first: Fact, fact: ProposedFact, language: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The two facts of a pair: values and statements, and their languages when
    the question is across languages."""
    old: dict[str, Any] = {"value": first.value, "statement": first.statement}
    new: dict[str, Any] = {"value": fact.value, "statement": fact.statement}
    if language is not None:
        old["language"], new["language"] = _language_of(first), language
    return old, new


def _restated(members: Sequence[Fact], fact: ProposedFact, language: str) -> Fact | None:
    """The member whose value the new one may tell again (SPEC 5.4), the one the
    restatement question is asked about: in the new statement's language, sharing
    words with it; none when the comparison proves a member different (another
    number in the same unit) or the new fact says when its value ended."""
    value = fact.value or ""
    if fact.valid_to or any(compare_values(m.value or "", value) == DIFFERENT for m in members):
        return None
    return next(
        (
            m
            for m in members
            if m.value
            and shares_words(m.value, value)
            and _one_language(language, fact.statement, m)
        ),
        None,
    )


def _kept_or_linked(
    members: Sequence[Fact],
    fact: ProposedFact,
    one_language: bool,
    *,
    external: bool,
    never_close: bool,
) -> str:
    """A value that is the group's own (SPEC 5.4): within one language the fact
    held is kept. It is linked instead - written beside it - across languages
    (P12); when the new value is more precise (a day refining a month) so that
    recall can prefer it, or contradicts a member (a month and its linked day, a
    new day); when it ends (``valid_to``); when it would give a group of
    external facts its first non-external one (P6); and between free speaker
    facts (several people may speak in one scope)."""
    value = fact.value or ""
    if (
        not one_language
        or never_close
        or fact.valid_to
        or any(compare_values(m.value or "", value) == DIFFERENT for m in members)
        or precision(value) > max(precision(m.value or "") for m in members)
        or (not external and all(m.provenance == EXTERNAL for m in members))
    ):
        return LINK
    return KEEP


def _same_form(
    members: Sequence[Fact], fact: ProposedFact, norms: Mapping[str, str] | None
) -> bool:
    """SPEC 5.4: is a member's comparison form the new fact's, exactly (case and
    spaces aside)? Both must carry one; forms that differ prove nothing. No
    tolerance: forms are already canonical, and one letter tells "painter"
    from "printer"."""
    if not norms or not fact.value_norm:
        return False
    new = _form(fact.value_norm)
    return any(m.id in norms and _form(norms[m.id]) == new for m in members)


def _form(norm: str) -> str:
    return " ".join(fold(norm).split())


def _later(fact: ProposedFact, members: Sequence[Fact]) -> bool:
    """Is the new fact later than every member (its valid_from not earlier;
    without dates, the later source is the later one)?"""
    new = fact.valid_from
    return not any(new and m.valid_from and new < m.valid_from for m in members)


def _decide(  # noqa: PLR0911, PLR0912 - one return per rule of SPEC 5.4
    members: Sequence[Fact],
    fact: ProposedFact,
    language: str,
    adjudicated: str | None,
    *,
    external: bool = False,
    never_close: bool = False,
    several: bool = False,
    norms: Mapping[str, str] | None = None,
    observed: tuple[str | None, str] | None = None,
    bound: str | None = None,
) -> tuple[str, str]:
    """What a new value does to one equivalence group of the same key (SPEC 5.4),
    and why: linked when proved the same; never closing an explicit fact (P6),
    a fact that did not come from external content when the new one did (P6),
    nor a free speaker fact (``never_close``: several people may speak in one
    scope); superseding within one language as before, except for role, team,
    preferences and free-form keys when the replacement question is on
    (``several``): they may hold several values, or two unrelated statements
    under one generic key, and are asked "replaces, or both true?" in one
    language too - a free-form value proved different (another number in the
    same form) still supersedes; across languages, superseding only a later
    value proved or adjudicated different, and linking without a question a
    value whose comparison form (``norms``, by fact id) compares the same as the
    new fact's when nothing was proved (``norm``); otherwise the conflict stays
    (``unknown``: what may be adjudicated). A new observation (``observed``: its
    day and source) against a group of observations closes it from a later
    source, without a question; one source's two observations both hold, and an
    older one arriving late stays in conflict. ``bound``: the provenance of a
    self-side fact (SPEC 4.2), which closes only facts of that provenance."""
    value = fact.value or ""
    key = canonical_key(fact.attribute)
    free_form = several and is_free_form(key)
    several = several and replacement_asked(key)
    verdicts = [compare_values(m.value or "", value) for m in members]
    one_language = any(_one_language(language, fact.statement, m) for m in members)
    guards = {"external": external, "never_close": never_close}
    if SAME in verdicts:
        return _kept_or_linked(members, fact, one_language, **guards), SAME
    if any(m.origin == "explicit" for m in members):
        return KEEP_CONFLICT, "explicit"
    if never_close:
        return KEEP_CONFLICT, "speaker"
    if bound is not None and any(m.provenance != bound for m in members):
        return KEEP_CONFLICT, "self"  # the owner corrects the agent, or the reverse
    if external and any(m.provenance != EXTERNAL for m in members):
        return KEEP_CONFLICT, "external"
    if observed is not None and (decided := _observations(members, observed)) is not None:
        return decided
    if not several and one_language:
        if adjudicated == "same":  # the restatement question (SPEC 5.4)
            return _kept_or_linked(members, fact, one_language, **guards), "adjudicated"
        return SUPERSEDE, "same_language"
    if (
        not one_language
        and DIFFERENT not in verdicts
        # P6 the other way round: an owner's fact never joins external content
        and (external or all(m.provenance != EXTERNAL for m in members))
        and _same_form(members, fact, norms)
    ):
        return LINK, NORM  # a form only ever links: never a supersession (SPEC 5.4)
    if not _later(fact, members):
        return KEEP_CONFLICT, "earlier"
    # another number may be an addition to a role; a free-form value that differs
    # only by its number or date is the same aspect
    if DIFFERENT in verdicts and (free_form or not several):
        return SUPERSEDE, DIFFERENT
    if adjudicated in ("different", "replaces"):
        return SUPERSEDE, "adjudicated"
    if adjudicated == "same":
        return _kept_or_linked(members, fact, one_language, **guards), "adjudicated"
    if adjudicated == "both":
        return BOTH, "adjudicated"  # role, team, preferences: both are true
    return KEEP_CONFLICT, UNKNOWN


def _observations(
    members: Sequence[Fact], observed: tuple[str | None, str]
) -> tuple[str, str] | None:
    """SPEC 5.4: a new observation (its day and source) against a group of
    observations - closed from a later source, both kept from one source, an
    older one arriving late in conflict; None when a member is not observed."""
    if not all(m.observed_at for m in members):
        return None
    day, source = observed
    if any(m.source_id == source for m in members):
        return BOTH, OBSERVED  # one observation holding both values
    if day and any((m.observed_at or "") > day for m in members):
        return KEEP_CONFLICT, "earlier"
    return SUPERSEDE, OBSERVED


@dataclass
class _Judgement:
    """What a new value does to the live groups of its key (SPEC 5.4)."""

    duplicate_of: str | None = None  # a group already states this very value
    kept: str | None = None  # a group states it in other words: kept as held
    same_as: str | None = None  # linked to this group's first fact
    superseded: list[Fact] = field(default_factory=list)
    conflicts_with: str | None = None


@dataclass(frozen=True)
class _Fate:
    """What P6 and SPEC 4.2 say about one planned fact."""

    span: tuple[int, int] | None
    provenance: str | None  # owner | assistant | external, None: nobody's words
    about_owner: bool
    speaker_free: bool  # about the speaker, no owner declared: no node
    propose: bool  # not written on its own: a fact proposal (P6, SPEC 5.1)
    neighbour: bool = False  # SPEC 5.1: its source shows another subject rather than its own
    unnamed: bool = False  # SPEC 5.1: a known subject neither its source nor its session names
    part: Part | None = None  # the part its quote lies in (P6)
    # SPEC 4.2, agent perspective: about the self; a rule of method; either of
    # them (self-side); and proposed by the self-side rule
    about_agent: bool = False
    method: bool = False
    self_side: bool = False
    self_outside: bool = False
    self_rule: bool = False  # a rule of the self the operator wants reviewed


@dataclass
class _IntentOutcome:
    """What the intent reading made of a job's answers (SPEC 4.2)."""

    kept: list[tuple[int, Answer]] = field(default_factory=list)
    proposed: list[tuple[int, ProposedFact, str]] = field(default_factory=list)  # with frame
    session: list[tuple[int, ProposedFact]] = field(default_factory=list)
    carry: intent_reading.Carry = field(default_factory=intent_reading.Carry)
    ended: bool = False


DESCRIPTION_WORDS = 4  # more words than this: a description, not a name (SPEC 5.1)
DESCRIPTION_RATIO = 3  # three words or more, and this many times the name's length


def descriptive(alias: str, name: str) -> bool:
    """Is ``alias`` a description rather than a name (SPEC 5.1)? More than four
    words, or at least three and more than three times as long as ``name``
    (letters counted, spaces left out)."""
    words = alias.split()
    if len(words) > DESCRIPTION_WORDS:
        return True
    letters, own = len("".join(words)), len("".join(name.split()))
    return len(words) >= 3 and letters > DESCRIPTION_RATIO * own


def _contains(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    size = len(needle)
    return any(
        list(haystack[i : i + size]) == list(needle) for i in range(len(haystack) - size + 1)
    )


@dataclass
class _Plan:
    proposals: dict[str, schema.ProposedNode] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    # (key, the node's name in the answer, alias the source does not hold)
    doubtful: list[tuple[str, str, str]] = field(default_factory=list)
    mention_keys: dict[str, str] = field(default_factory=dict)
    ordered: list[str] = field(default_factory=list)
    facts: list[tuple[int, ProposedFact]] = field(default_factory=list)
    # SPEC 5.4: (index in facts, equivalence group) -> same | different | unsure
    adjudicated: dict[tuple[int, str], str] = field(default_factory=dict)
    edges: list[ProposedEdge] = field(default_factory=list)
    forced: str | None = None
    # SPEC 4.6: the owner's node key, and the scopes of nodes created elsewhere
    owner_key: str | None = None
    agent_key: str | None = None  # SPEC 4.6: the self's node key
    # SPEC 4.2: the activity's node key, and whether this source named it
    activity_key: str | None = None
    activity_named: bool = False
    node_scopes: dict[str, list[str]] = field(default_factory=dict)
    fates: list[_Fate] = field(default_factory=list)  # one per fact, same order
    mixed_names: set[str] = field(default_factory=set)  # SPEC 4.2: refused as nodes
    # SPEC 5.1: mention -> the known node it matched, neither named nor attached
    unnamed: dict[str, str] = field(default_factory=dict)

    def order(self, key: str) -> None:
        if key not in self.ordered:
            self.ordered.append(key)

    def note_aliases(
        self, key: str, name: str, kept: Sequence[str], doubtful: Sequence[str]
    ) -> None:
        """Aliases to add to the node ``key`` (grounded in the source) and aliases
        to propose for it (not grounded, SPEC 5.1)."""
        if kept:
            self.aliases.setdefault(key, []).extend(a for a in kept if a)
        self.doubtful.extend((key, name, a) for a in doubtful if a)


class _AnswerForms:
    """Sorts the aliases the answers give a node (SPEC 5.1): kept when the source
    holds them and they are no other node's name or alias (of these answers, or
    known in the view) nor a near form of the node's own names; a proposal when
    the source does not hold them, when they belong to another node (a merge
    candidate), or when they are a near form ("Ивана" for Иван); dropped and
    counted as invented when their script appears nowhere in the source."""

    def __init__(self, extractor: Extractor, answers: Sequence[Answer], text: str) -> None:
        self._extractor = extractor
        self._text = text
        self._scripts = scripts_in(text)
        resolver = extractor.resolver
        # (node index, form, the node's name, the known node its name resolves to)
        self._forms: list[tuple[int, str, str, str | None]] = []
        for index, node in enumerate(n for a in answers for n in a.nodes):
            resolution = resolver.resolve(node.name)
            known = (
                resolution.key
                if resolution.key is not None and resolution.method not in (AMBIGUOUS, NEW)
                else None
            )
            name = normalize(node.name)
            for form in (node.name, *(a for a in node.aliases if grounded(a, text))):
                self._forms.append((index, normalize(form), name, known))

    def split(
        self, index: int, name: str, own: str | None, aliases: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        """``(kept, proposed)`` for the aliases of the ``index``-th node of the
        answers, whose name is ``name`` and which resolved to ``own`` (a known or
        pending node key, or None). Invented ones are counted and left out."""
        kept: list[str] = []
        proposed: list[str] = []
        report = self._extractor.report
        names = [name, *self._extractor.stored_names(own)]  # what a near form is near to
        for alias in aliases:
            script = script_of(alias)
            if (script is not None and script not in self._scripts) or mixed_script_words(alias):
                report.aliases_invented += 1  # a script the source lacks, or two in one word
            elif normalize(alias) != normalize(name) and descriptive(alias, name):
                report.aliases_descriptive += 1  # SPEC 5.1: a description, never kept
                if grounded(alias, self._text):
                    proposed.append(alias)  # else the model's own words: no name to propose
            elif (spelling := source_spelling(alias, self._text)) is None or (
                self._elsewhere(alias, index, name, own)
            ):
                proposed.append(alias)
            elif (plain := plain_name(spelling)) is None:
                report.aliases_malformed += 1  # SPEC 5.1: the source's span is debris
            elif any(near_form(alias, known) for known in names):
                report.aliases_near += 1  # SPEC 5.1: it would attach a near name on its own
                proposed.append(alias)
            else:
                kept.append(plain)  # the source's spelling, not the model's copy
                names.append(plain)
        return kept, proposed

    def _elsewhere(self, alias: str, index: int, name: str, own: str | None) -> bool:
        """Is ``alias`` another node's name or alias? The same node listed twice in
        the answers (same name, or resolving to the same known node) is not another."""
        wanted, own_name = normalize(alias), normalize(name)
        for other, form, other_name, known in self._forms:
            if other == index or form != wanted or other_name == own_name:
                continue
            if own is not None and known == own:
                continue
            return True
        found = self._extractor.store.find_node(alias, self._extractor.view)
        return found is not None and found.id != own


class _NameGroups:
    """What names a subject in one answer (SPEC 4.2): its name, its aliases
    (stored, or given by the answer and found in the source), and every mention
    of the answer (fact subject, edge end, node name or grounded alias) that
    resolves to the same node. Mentions are grouped when they belong to one of the
    answer's nodes or resolve, exactly or fuzzily, to the same known node;
    adjudication comes later and is not used here."""

    def __init__(self, extractor: Extractor, answer: Answer, piece: str) -> None:
        self._parent: dict[str, str] = {}
        self._written: dict[str, list[str]] = {}
        self._keys: dict[str, str] = {}  # group member -> known node id
        resolver = extractor.resolver
        mentions: list[str] = []
        for node in answer.nodes:
            forms = [node.name, *(a for a in node.aliases if grounded(a, piece))]
            mentions.extend(forms)
            for form in forms[1:]:
                self._union(forms[0], form)
        mentions.extend(f.subject for f in answer.facts if f.subject)
        mentions.extend(m for e in answer.edges for m in (e.src, e.dst))
        for mention in mentions:
            self._find(mention)
            resolution = resolver.resolve(mention)
            if resolution.key is not None and resolution.method not in (AMBIGUOUS, NEW):
                marker = f"\x00{resolution.key}"
                self._union(mention, marker)
                if not resolution.key.startswith("new:"):
                    self._keys[marker] = resolution.key
        self._store = extractor.store

    def _norm(self, text: str) -> str:
        return text if text.startswith("\x00") else normalize(text)

    def _find(self, text: str) -> str:
        key = self._norm(text)
        if key not in self._parent:
            self._parent[key] = key
        if not text.startswith("\x00"):
            self._written.setdefault(key, [])
            if text not in self._written[key]:
                self._written[key].append(text)
        return self._root(key)

    def _root(self, key: str) -> str:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        return root

    def _union(self, a: str, b: str) -> None:
        root_a, root_b = self._find(a), self._find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def names(self, subject: str) -> list[str]:
        root = self._find(subject)
        names: list[str] = []
        for member in self._parent:
            if self._root(member) != root:
                continue
            names.extend(self._written.get(member, []))
            node_id = self._keys.get(member)
            if node_id is not None:
                node = self._store.get_node(node_id)
                names.extend([node.name, *node.aliases])
        return list(dict.fromkeys(names))


def source_form(subject: str, quote: str, threshold: float) -> str | None:
    """SPEC 5.1: the words of ``quote`` (as many as ``subject`` has) near the
    subject without being the same name once folded or transliterated - the
    nearest, as the source spells it. None when the quote holds the subject
    itself, or nothing near it (a pronoun, a translation)."""
    wanted = normalize(subject)
    latin = normalize(transliterate(subject))
    if not wanted:
        return None
    forms = word_windows(quote, len(words_of(subject)))
    if any(same_spelling(form, subject) for form in forms):
        return None  # the subject itself is in the quote, in one script or another
    pairs = [(f, normalize(f), normalize(transliterate(f))) for f in forms]
    best, best_score = None, 0.0
    for form, folded, roman in pairs:
        score = max(similarity(folded, wanted), similarity(roman, latin) if latin else 0.0)
        if score >= threshold and score > best_score:
            best, best_score = form, score
    return best


NAME_WORD = 3  # the shortest word of a name that names its subject inside a value


def _spans(tokens: Sequence[str], needle: Sequence[str]) -> list[tuple[int, int]]:
    """Where ``needle`` occurs in ``tokens``, as whole tokens (``_contains``, with
    the positions)."""
    size = len(needle)
    if not size:
        return []
    return [
        (i, i + size)
        for i in range(len(tokens) - size + 1)
        if _contains(tokens[i : i + size], needle)
    ]


def _occurrences(tokens: Sequence[str], forms: Iterable[str]) -> list[tuple[int, int]]:
    return [span for form in forms for span in _spans(tokens, tokenize(form))]


def _inside(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
    """Is ``inner`` a strictly shorter part of ``outer``?"""
    return outer[0] <= inner[0] and inner[1] <= outer[1] and inner != outer


def _names_whole(tokens: Sequence[str], forms: Iterable[str]) -> bool:
    """Is one of ``forms`` written in ``tokens`` in full, or in another script
    with the same spelling (``same_spelling``: Иван for Ivan)?"""
    text = " ".join(tokens)
    for form in forms:
        wanted = tokenize(form)
        if not wanted:
            continue
        if _spans(tokens, wanted):
            return True
        if any(same_spelling(window, form) for window in word_windows(text, len(wanted))):
            return True
    return False


def _written_in(text: str, names: Iterable[str]) -> bool:
    """SPEC 5.1: is one of ``names`` written in ``text`` - whole words, the same
    spelling in another script, or, for a name in a script written without
    spaces, its letters (folded)?"""
    names = [n for n in names if n]
    if _names_whole(tokenize(text), names):
        return True
    unspaced = [n for n in names if unspaced_chars(n)]
    if not unspaced:
        return False
    folded = normalize(text).replace(" ", "")
    return any(normalize(n).replace(" ", "") in folded for n in unspaced)


def quote_names_other(
    quote: str, own: Sequence[str], others: Mapping[str, Sequence[str]], *, objects: str = ""
) -> bool:
    """SPEC 5.1: does ``quote`` name another subject - one of its names or
    aliases, whole words - and not its own, by one of ``own`` written in full
    (or in another script, same spelling) once the other subjects' names are
    set aside? Never a near or inflected form: "Karima" is not Karim. A part
    of a longer name is that name's ("Karim" inside "Karim Benali"); another
    subject that ``objects`` (the fact's statement and value) names whole is
    its object ("Karim leads the solar project", quoted "leads the solar
    project")."""
    tokens = tokenize(quote)
    said = tokenize(objects)
    own_spans = _occurrences(tokens, own)
    other_spans = [
        span
        for names in others.values()
        if not any(tokenize(n) and _contains(said, tokenize(n)) for n in names)
        for span in _occurrences(tokens, names)
    ]
    other_spans = [o for o in other_spans if not any(_inside(o, s) for s in own_spans)]
    if not other_spans:
        return False
    if any(not any(_inside(s, o) for o in other_spans) for s in own_spans):
        return False
    covered = {i for start, end in other_spans for i in range(start, end)}
    rest = [token for i, token in enumerate(tokens) if i not in covered]
    return not _names_whole(rest, own)


def _distinctive(forms: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    """The words of each subject's names that no other subject's names hold, of
    at least three letters and not digits only ("zip" names neither of
    backup.zip and photos.zip)."""
    words_by_key = {
        key: {
            w for name in names for w in tokenize(name) if len(w) >= NAME_WORD and not w.isdigit()
        }
        for key, names in forms.items()
    }
    out: dict[str, set[str]] = {}
    for key, own in words_by_key.items():
        shared = set().union(*(w for k, w in words_by_key.items() if k != key))
        out[key] = own - shared
    return out


def value_merges(
    value: str | None,
    own: str,
    distinctive: Mapping[str, set[str]],
    forms: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    """SPEC 5.1: does ``value`` hold a word of its own subject's names and a word
    of another subject's - a value describing both? A value that names the
    other subject in full refers to it ("a zip holding report.pdf"): no merge."""
    if not value:
        return False
    tokens = tokenize(value)
    held = set(tokens)
    if not held & distinctive.get(own, set()):
        return False
    for key, words in distinctive.items():
        if key == own or not held & words:
            continue
        names = (forms or {}).get(key, ())
        if not any(tokenize(n) and _contains(tokens, tokenize(n)) for n in names):
            return True
    return False


def names_owner(statement: str, name: str) -> bool:
    """SPEC 4.2: does ``statement`` hold the owner's ``name`` in any form that
    transliterates to it ("Самир" for Samir), counted in as many words as the
    name has ("Иван-Пётр" for Ivan-Petr)?"""
    return any(same_name(form, name) for form in word_windows(statement, len(words_of(name))))


def _settled(resolution: Resolution) -> bool:
    return resolution.key is not None and resolution.method not in (AMBIGUOUS, NEW)


def _drop_unused(plan: _Plan, replaced: set[str]) -> None:
    """SPEC 5.1: a new node the answer named only for subjects the source form
    replaced holds no fact and no edge: it is not created."""
    used = set(plan.mention_keys.values()) | {plan.owner_key, plan.activity_key}
    unused = [
        key
        for key, proposal in plan.proposals.items()
        if key.startswith("new:") and key not in used and normalize(proposal.name) in replaced
    ]
    for key in unused:
        del plan.proposals[key]
        plan.aliases.pop(key, None)
        plan.ordered[:] = [k for k in plan.ordered if k != key]
        plan.doubtful[:] = [d for d in plan.doubtful if d[0] != key]


def _keys_of(resolution: Resolution) -> list[str]:
    if _settled(resolution):
        return [str(resolution.key)]
    return [c.key for c in resolution.candidates]


CONTEXT_HEAD = 160  # characters of a source shown when no mention locates it


def _context(text: str, mention: str, width: int = 80) -> str:
    position = text.lower().find(mention.lower())
    if position < 0:
        return ""
    start = max(0, position - width)
    end = min(len(text), position + len(mention) + width)
    return " ".join(text[start:end].split())


class PartFields(TypedDict):
    """Where an external fact was read (P6): its part's kind, short ref, day."""

    part_kind: str | None
    part_ref: str | None
    read_at: str | None


def day_read(reference: str | None, now: str) -> str:
    """P6: the day a source was read - ``reference`` when it is a date or a
    timestamp (a turn's ``occurred_at``), else the day of ``now``."""
    if reference:
        try:
            return datetime.fromisoformat(reference.strip()).date().isoformat()
        except ValueError:
            pass
    return now[:10]


def extract_text(
    ctx: JobContext,
    text: str,
    *,
    scopes: Sequence[str],
    kind: str,
    parts: Sequence[Part] | None = None,
    session_id: str | None = None,
    reference: str | None = None,
    external_id: str | None = None,
) -> dict[str, Any]:
    """Extract one source and return the report as a plain dict (a job result)."""
    extractor = Extractor(ctx, scopes)
    return extractor.run(
        text,
        kind=kind,
        parts=parts,
        session_id=session_id,
        reference=reference,
        external_id=external_id,
    ).as_dict()


__all__ = [
    "CONFIRMED",
    "CONFLICT",
    "EXTRACTED",
    "LANGUAGE_MISMATCH",
    "Edge",
    "ExtractionReport",
    "Extractor",
    "Node",
    "PipelineConfig",
    "chunks",
    "extract_text",
    "locate",
    "mentioned_nodes",
    "write_view",
]
