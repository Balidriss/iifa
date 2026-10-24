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

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from iifa import log, vectors
from iifa.calibration import Thresholds, key_threshold_for, thresholds_for
from iifa.ingest import schema
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
)
from iifa.ingest.keys import (
    canonical_key,
    family_of,
    is_free_form,
    key_text,
    multi_valued,
    same_key,
    single_valued,
    supersedable,
)
from iifa.ingest.parts import EXTERNAL, OWNER, Part, SourceText
from iifa.ingest.prompts import (
    adjudication_messages,
    extraction_messages,
    replacement_adjudication_messages,
    value_adjudication_messages,
)
from iifa.ingest.resolve import (
    ADJUDICATED,
    AMBIGUOUS,
    NEW,
    Candidate,
    Resolution,
    Resolver,
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
    same_name,
    same_spelling,
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
from iifa.recall.tokenizer import normalize, tokenize
from iifa.scopes import Visibility, ancestors
from iifa.store.models import Edge, Fact, Node
from iifa.store.repo import CONFIRMED, NAME, PROPOSED, Store
from iifa.tokens import ConservativeTokenizer

if TYPE_CHECKING:
    from iifa.ingest.worker import JobContext

EXTRACTED = "extracted"
CONFLICT, LANGUAGE_MISMATCH = "conflict", "language_mismatch"
SPEAKER = "speaker"  # a free-standing fact about the person speaking (SPEC 4.2)
SEMANTIC_CANDIDATES = 5  # near nodes offered to the adjudication per mention
KEYS_PER_SUBJECT = 8  # attribute keys listed per known subject in the prompt
PAIR_QUOTE_CHARS = 300  # SPEC 5.4: the source's words given with each fact of a pair
INCOMPLETE, UNNAMED, CROSS_SCRIPT = "incomplete", "unnamed", "cross_script"  # fragment causes
MIXED_SCRIPT = "mixed_script"  # a word mixing two spaced scripts (SPEC 4.2)
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
    replacement_question: bool = True  # SPEC 5.4: on by default (measured)

    def __post_init__(self) -> None:
        if not (0 < self.answer_base_tokens <= self.answer_max_tokens):
            raise ValueError("answer_base_tokens must be within 1..answer_max_tokens")
        if self.answer_tokens_per_source_token < 0:
            raise ValueError("answer_tokens_per_source_token must be 0 or more")

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
        """The language a scope (or one of its ancestors) forces, if any (P12)."""
        table = dict(self.language_overrides)
        for scope in scopes:
            for candidate in ancestors(scope):
                if candidate in table:
                    return table[candidate]
        return None

    def owner(self, scopes: Iterable[str]) -> Owner | None:
        """The owner a source written into ``scopes`` speaks as (SPEC 4.6): the
        nearest declaration of each scope, and only when they all agree - a
        source split between two owners, or with a scope that has none, has no
        owner (who speaks cannot be told)."""
        table = {scope: Owner(scope, name, tuple(aliases)) for scope, name, aliases in self.owners}
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
    values_same: int = 0  # SPEC 5.4: proved the same value in another form, linked
    values_different: int = 0  # SPEC 5.4: proved different across languages, superseded
    values_adjudicated: dict[str, int] = field(
        default_factory=lambda: {"same": 0, "different": 0, "unsure": 0}
    )
    # SPEC 5.4: role, team, preferences - "replaces, or both true?"
    values_replacement: dict[str, int] = field(
        default_factory=lambda: {"same": 0, "replaces": 0, "both": 0, "unsure": 0}
    )
    facts_conflict: int = 0
    facts_language_mismatch: int = 0
    facts_incomplete: int = 0
    facts_cross_script: int = 0
    facts_mixed_script: int = 0
    facts_speaker_owner: int = 0  # SPEC 4.2: about the speaker, attached to the owner
    facts_speaker_free: int = 0  # SPEC 4.2: about the speaker, no owner declared
    facts_proposed: int = 0  # P6: about the owner, not in the owner's words
    empty_answers: int = 0  # SPEC 4.2: chunks the model found nothing durable in
    aliases_kept: int = 0
    aliases_proposed: int = 0
    aliases_invented: int = 0
    aliases_descriptive: int = 0  # SPEC 5.1: a description, proposed rather than kept
    aliases_near: int = 0  # SPEC 5.1: a near form of the node's name, proposed rather than kept
    keys_renamed: int = 0  # SPEC 5.4 step 3: a synonym or a variant made canonical
    keys_matched: int = 0  # SPEC 5.4 step 4: a near existing key, by embeddings
    edges_proposed: int = 0
    edges_evidenced: int = 0
    edges_confirmed: int = 0
    rejected: Rejections = field(default_factory=Rejections)

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
                "conflict": self.facts_conflict,
                "language_mismatch": self.facts_language_mismatch,
                "incomplete": self.facts_incomplete,
                "cross_script": self.facts_cross_script,
                "mixed_script": self.facts_mixed_script,
                "speaker_owner": self.facts_speaker_owner,
                "speaker_free": self.facts_speaker_free,
                "proposed": self.facts_proposed,
            },
            "aliases": {
                "kept": self.aliases_kept,
                "proposed": self.aliases_proposed,
                "invented": self.aliases_invented,
                "descriptive": self.aliases_descriptive,
                "near": self.aliases_near,
            },
            "keys": {"renamed": self.keys_renamed, "matched": self.keys_matched},
            "values": {
                "same": self.values_same,
                "different": self.values_different,
                "adjudicated": dict(self.values_adjudicated),
                "replacement": dict(self.values_replacement),
            },
            "edges": {
                "proposed": self.edges_proposed,
                "evidenced": self.edges_evidenced,
                "confirmed": self.edges_confirmed,
            },
            "rejected": self.rejected.as_dict(),
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
        # SPEC 4.6: who speaks, and the owner's node when it exists already
        self.owner = self.config.owner(self.scopes)
        self.owner_node: Node | None = None
        if self.owner is not None:
            self.owner_node = self.store.find_node(
                self.owner.name, Visibility.of_scope(self.owner.scope)
            )

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
        forced = self.config.forced_language(self.scopes)
        languages = languages_present(text)
        report.languages = [forced] if forced else languages
        source_language = forced or (languages[0] if languages else None)
        report.tier = 2 if self.ctx.models.has_chat else 0

        answers: list[tuple[int, Answer]] = []
        if report.tier == 2:
            answers = self._extract_all(text, languages, forced)
        else:
            report.chunks = len(chunks(text, self.config.chunk_chars, limit=self.config.max_chunks))

        # every model call is done (extraction, then adjudication): now write once
        plan = self._plan(text, answers, forced)
        with self.store.lock, self.store.transaction():
            source = self.store.add_source(
                scopes=self.scopes,
                kind=kind,
                text=text,
                reference=reference,
                external_id=external_id,
                session_id=session_id,
                language=source_language,
            )
            report.source_id = source.id
            self._write(source.id, text, plan)
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
                    external=any(p.provenance == EXTERNAL for p in self.source.parts),
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
        self._plan_nodes(plan, answers, text)
        replaced = self._source_forms(answers, text)
        mentions: list[str] = []
        for _, answer in answers:
            mentions.extend(f.subject for f in answer.facts if f.subject)
            mentions.extend(m for e in answer.edges for m in (e.src, e.dst))
        self._resolve_mentions(plan, list(dict.fromkeys(mentions)), text)
        _drop_unused(plan, replaced)
        # SPEC 4.2: a valid subject_as_written becomes a source-anchored alias
        for _, answer in answers:
            for fact in answer.facts:
                spelling = self._as_written(fact, answer, text)
                subject_key = plan.mention_keys.get(fact.subject or "")
                if spelling is None or subject_key is None:
                    continue
                names = [fact.subject or "", *self.stored_names(subject_key)]
                if subject_key in plan.proposals:
                    names.append(plan.proposals[subject_key].name)
                if any(near_form(spelling, known) for known in names):
                    self.report.aliases_near += 1  # names the subject here, never stored
                    continue
                plan.aliases.setdefault(subject_key, []).append(spelling)
        for offset, answer in answers:
            for fact in answer.facts:
                plan.facts.append((offset, fact))
                plan.fates.append(self._fate(plan, offset, fact))
            for edge in answer.edges:
                plan.edges.append(edge)
        # tier-0 mention detection: known names in the text join the co-occurrence set
        for node_id in mentioned_nodes(text, self.store.node_index(self.view)):
            plan.order(node_id)
        self._align_keys(plan)
        self._adjudicate_values(plan)
        plan.forced = forced
        self.report.nodes_mixed_script += len(plan.mixed_names)
        return plan

    def _plan_nodes(self, plan: _Plan, answers: list[tuple[int, Answer]], text: str) -> None:
        """The nodes the answers propose, resolvable by the facts and edges that
        follow: a known node, or a new one; a speaker word is neither (SPEC 5.1)."""
        resolver = self.resolver
        nodes = [n for _, a in answers for n in a.nodes]
        self._prepare_semantic(answers)
        decided = self._adjudicate_names(nodes, text)
        forms = _AnswerForms(self, [a for _, a in answers], text)
        for index, proposal in enumerate(nodes):
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

    def _fate(self, plan: _Plan, offset: int, fact: ProposedFact) -> _Fate:
        """Whose words the fact's quote is (P6), whether the fact is about the
        owner (SPEC 4.2), and so whether it may be written or only proposed: a
        fact about the owner is written from the owner's own words only, and not
        when its value comes from outside the conversation alone."""
        span = locate(self.source.text, fact.quote, hint=offset)
        provenance = self.source.provenance_of(span)
        subject = fact.subject or ""
        speaker = is_speaker(subject)
        about_owner = (speaker and self.owner is None) or (
            plan.owner_key is not None and plan.mention_keys.get(subject) == plan.owner_key
        )
        propose = about_owner and (provenance != OWNER or self._value_from_outside(fact.value))
        return _Fate(span, provenance, about_owner, speaker and self.owner is None, propose)

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

    # ------------------------------------------------------------------ attribute keys (SPEC 5.4)

    def _adjudicate_values(self, plan: _Plan) -> None:
        """SPEC 5.4: the pairs the deterministic comparison leaves ``unknown`` across
        languages - the new fact later, no explicit fact - go to the chat model
        before the write transaction: single-valued keys in one batched call ("the
        same thing?"), role, team and preferences in another ("replaces, or both
        true?"). ``unsure``, an invalid answer or a truncated one changes nothing."""
        if not self.ctx.models.has_chat:
            return
        values, replacements = self._unknown_pairs(plan)
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

    def _unknown_pairs(self, plan: _Plan) -> tuple[_Pairs, _Pairs]:
        """The pairs left ``unknown`` across languages, by question: single-valued
        keys, and role, team and preferences."""
        values: _Pairs = ([], [])
        replacements: _Pairs = ([], [])
        for index, (_offset, fact) in enumerate(plan.facts):
            node_id = plan.mention_keys.get(fact.subject or "")
            key = canonical_key(fact.attribute)
            fate = plan.fates[index]
            if node_id is None or node_id.startswith("new:") or not fact.value or fate.propose:
                continue
            asked = self.config.replacement_question and multi_valued(key)
            batch = values if single_valued(key) else replacements if asked else None
            if batch is None:
                continue
            language = detect(fact.statement)
            live = self.store.facts([node_id], self.view)
            if any(fold(old.statement) == fold(fact.statement) for old in live):
                continue
            for root, members in _groups(live, key):
                if any(m.value and normalize(m.value) == normalize(fact.value) for m in members):
                    continue
                external = fate.provenance == EXTERNAL
                verdict = _decide(
                    members,
                    fact,
                    language,
                    None,
                    external=external,
                    several=self.config.replacement_question,
                )
                if verdict[1] != UNKNOWN:
                    continue
                first = members[0]
                old: dict[str, Any] = {
                    "value": first.value,
                    "statement": first.statement,
                    "language": _language_of(first),
                }
                new: dict[str, Any] = {
                    "value": fact.value,
                    "statement": fact.statement,
                    "language": language,
                }
                if batch is replacements:
                    # SPEC 5.4: the statements may have lost "now" or "also"
                    if (said := self._stored_quote(first)) is not None:
                        old["quote"] = said
                    if (said := self._new_quote(fate)) is not None:
                        new["quote"] = said
                batch[0].append(
                    {
                        "pair": len(batch[0]),
                        "subject": self.store.get_node(node_id).name,
                        "key": key,
                        "a": old,
                        "b": new,
                    }
                )
                batch[1].append((index, root))
        return values, replacements

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
        bare family key is never offered for reuse (it never supersedes)."""
        nodes = self.resolver.known_nodes(self.config.known_names_in_prompt)
        stored = self.store.attribute_keys([node_id for node_id, _ in nodes], self.view)
        out: list[tuple[str, list[str]]] = []
        for node_id, name in nodes:
            keys = list(
                dict.fromkeys(
                    k for k in map(canonical_key, stored.get(node_id, ())) if supersedable(k)
                )
            )
            if keys:
                out.append((name, [k for k in keys if k][:KEYS_PER_SUBJECT]))
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

    def _resolve_mentions(self, plan: _Plan, mentions: list[str], text: str) -> None:
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
        ambiguous = [r for r in map(resolver.resolve, mentions) if r.method == AMBIGUOUS]
        if ambiguous and self.ctx.models.has_chat:
            self._adjudicate(ambiguous, text)
        for mention in mentions:
            resolution = resolver.resolve(mention)
            if resolution.method == AMBIGUOUS:
                resolver.decide(resolution, None)
            if resolution.key is None:
                key = f"new:{len(plan.proposals)}"
                plan.proposals[key] = schema.ProposedNode(name=mention)
                resolver.propose(key, mention, "other")
                resolution.key = key
                resolution.method = NEW
            plan.mention_keys[mention] = resolution.key
            plan.order(resolution.key)

    def _adjudicate(self, ambiguous: list[Any], text: str) -> None:
        batch = []
        for resolution in ambiguous:
            snippet = _context(text, resolution.mention)
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
            else:
                ids[key] = key
                report.nodes_matched += 1
                self._add_aliases(store.get_node(key), plan.aliases.get(key, []))
        self._propose_aliases(source_id, text, plan, ids)

        touched: list[str] = []
        seen_statements: set[tuple[str | None, str]] = set()
        for index, (offset, fact) in enumerate(plan.facts):
            node_id = ids.get(plan.mention_keys.get(fact.subject or "", ""))
            signature = (node_id, fold(fact.statement))
            if signature in seen_statements:
                report.facts_duplicate += 1
                continue
            seen_statements.add(signature)
            fate = plan.fates[index]
            if fate.propose:
                self._propose_fact(source_id, text, fact, node_id, fate, plan.forced)
                continue
            verdicts = {g: v for (i, g), v in plan.adjudicated.items() if i == index}
            written = self._write_fact(
                source_id, text, offset, fact, node_id, plan.forced, verdicts, fate
            )
            if written is not None and node_id is not None and node_id not in touched:
                touched.append(node_id)

        self._write_edges(source_id, text, plan, ids, touched)

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
    ) -> Fact | None:
        store, report = self.store, self.report
        fate = fate or _Fate(locate(text, fact.quote, hint=offset), None, False, False, False)
        span = fate.span
        language = detect(fact.statement)
        flags = self._flags(fact, text, span, forced, fate)
        conflicts_with: str | None = None
        same_as: str | None = None
        superseded: list[Fact] = []
        live = self._live(node_id, fate)
        for old in live:
            if fold(old.statement) == fold(fact.statement):
                report.facts_duplicate += 1
                return None
        # SPEC 5.4: a bare family key ("preference" without a topic) never drives a
        # supersession or a conflict; when we cannot tell, we do not supersede
        speaking = fate.speaker_free
        duplicate_of: str | None = None
        if (node_id or speaking) and fact.value and supersedable(canonical_key(fact.attribute)):
            for root, members in _groups(live, fact.attribute):
                if any(m.value and normalize(m.value) == normalize(fact.value) for m in members):
                    # a duplicate of this group; the other groups are still decided
                    duplicate_of = duplicate_of or root
                    continue
                decision, why = _decide(
                    members,
                    fact,
                    language,
                    (verdicts or {}).get(root),
                    external=fate.provenance == EXTERNAL,
                    never_close=speaking,
                    several=self.config.replacement_question,
                )
                if decision == LINK:
                    same_as = same_as or root
                    report.values_same += why == SAME
                elif decision == SUPERSEDE:
                    superseded.extend(members)  # the whole group states the outdated value
                    report.values_different += why == DIFFERENT
                elif decision == BOTH:
                    pass  # both values hold at once: neither closed nor in conflict
                else:
                    if CONFLICT not in flags:
                        flags.append(CONFLICT)
                    conflicts_with = conflicts_with or root
        if duplicate_of is not None:
            # nothing new is written, but what the new evidence closes is closed,
            # by the fact that already states the same value
            report.facts_duplicate += 1
            self._close(superseded, fact.valid_from, duplicate_of)
            return None
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
            same_as=same_as,
            provenance=fate.provenance,
        )
        self._count_written(fate, conflicts_with)
        self._close(superseded, fact.valid_from, written.id)
        return written

    def _close(self, superseded: Sequence[Fact], valid_to: str | None, by: str) -> None:
        for old in superseded:
            self.store.close_fact(old.id, valid_to=valid_to, superseded_by=by)
            self.report.facts_superseded += 1

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
        return flags

    def _propose_fact(
        self,
        source_id: str,
        text: str,
        fact: ProposedFact,
        node_id: str | None,
        fate: _Fate,
        forced: str | None,
    ) -> None:
        """P6: a fact about the owner that is not the owner's own words becomes a
        fact proposal, decided only by a ``review`` token (SPEC 4.4b)."""
        language = detect(fact.statement)
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
                "flags": self._flags(fact, text, fate.span, forced, fate),
                "provenance": fate.provenance,
                "confidence": fact.confidence,
            },
        )
        self.report.facts_proposed += 1

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

        def observe(src: str, dst: str, type_: str, weight: float, confirm_after: int) -> None:
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
            count = store.add_evidence(edge.id, source_id)
            report.edges_evidenced += 1
            if edge.status == PROPOSED and count >= confirm_after:
                existing[(edge.src, edge.dst, edge.type)] = store.confirm_edge(edge.id)
                report.edges_confirmed += 1

        for proposed in plan.edges:
            src = ids.get(plan.mention_keys.get(proposed.src, ""))
            dst = ids.get(plan.mention_keys.get(proposed.dst, ""))
            if src is None or dst is None or src == dst:
                continue
            observe(src, dst, proposed.type, 1.0, config.confirm_after_sources)
            for node_id in (src, dst):
                if node_id not in touched:
                    touched.append(node_id)

        for a, b in self._cooccurrences(text, plan, ids, involved):
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
_Pairs = tuple[list[dict[str, Any]], list[tuple[int, str]]]


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


def _later(fact: ProposedFact, members: Sequence[Fact]) -> bool:
    """Is the new fact later than every member (its valid_from not earlier;
    without dates, the later source is the later one)?"""
    new = fact.valid_from
    return not any(new and m.valid_from and new < m.valid_from for m in members)


def _decide(  # noqa: PLR0911 - one return per rule of SPEC 5.4
    members: Sequence[Fact],
    fact: ProposedFact,
    language: str,
    adjudicated: str | None,
    *,
    external: bool = False,
    never_close: bool = False,
    several: bool = False,
) -> tuple[str, str]:
    """What a new value does to one equivalence group of the same key (SPEC 5.4),
    and why: linked when proved the same; never closing an explicit fact (P6),
    a fact that did not come from external content when the new one did (P6),
    nor a free speaker fact (``never_close``: several people may speak in one
    scope); superseding within one language as before, except for role, team
    and preferences when the replacement question is on (``several``): they may
    hold several values and are asked "replaces, or both true?" in one language
    too; across languages, superseding only a later
    value proved or adjudicated different; otherwise the conflict stays
    (``unknown``: what may be adjudicated)."""
    value = fact.value or ""
    several = several and multi_valued(canonical_key(fact.attribute))
    verdicts = [compare_values(m.value or "", value) for m in members]
    if SAME in verdicts:
        return LINK, SAME
    if any(m.origin == "explicit" for m in members):
        return KEEP_CONFLICT, "explicit"
    if never_close:
        return KEEP_CONFLICT, "speaker"
    if external and any(m.provenance != EXTERNAL for m in members):
        return KEEP_CONFLICT, "external"
    if not several and any(_one_language(language, fact.statement, m) for m in members):
        return SUPERSEDE, "same_language"
    if not _later(fact, members):
        return KEEP_CONFLICT, "earlier"
    if DIFFERENT in verdicts and not several:  # another number may be an addition
        return SUPERSEDE, DIFFERENT
    if adjudicated in ("different", "replaces"):
        return SUPERSEDE, "adjudicated"
    if adjudicated == "same":
        return LINK, "adjudicated"
    if adjudicated == "both":
        return BOTH, "adjudicated"  # role, team, preferences: both are true
    return KEEP_CONFLICT, UNKNOWN


@dataclass(frozen=True)
class _Fate:
    """What P6 and SPEC 4.2 say about one planned fact."""

    span: tuple[int, int] | None
    provenance: str | None  # owner | assistant | external, None: nobody's words
    about_owner: bool
    speaker_free: bool  # about the speaker, no owner declared: no node
    propose: bool  # about the owner, not in the owner's words: a fact proposal


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
    node_scopes: dict[str, list[str]] = field(default_factory=dict)
    fates: list[_Fate] = field(default_factory=list)  # one per fact, same order
    mixed_names: set[str] = field(default_factory=set)  # SPEC 4.2: refused as nodes

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
                proposed.append(alias)
            elif (spelling := source_spelling(alias, self._text)) is None or (
                self._elsewhere(alias, index, name, own)
            ):
                proposed.append(alias)
            elif any(near_form(alias, known) for known in names):
                report.aliases_near += 1  # SPEC 5.1: it would attach a near name on its own
                proposed.append(alias)
            else:
                kept.append(spelling)  # the source's spelling, not the model's copy
                names.append(spelling)
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
    used = set(plan.mention_keys.values()) | {plan.owner_key}
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


def _context(text: str, mention: str, width: int = 80) -> str:
    position = text.lower().find(mention.lower())
    if position < 0:
        return ""
    start = max(0, position - width)
    end = min(len(text), position + len(mention) + width)
    return " ".join(text[start:end].split())


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
