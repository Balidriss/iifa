"""Closing a session (SPEC 5.8): its work in progress, consolidated or expired.

When a session ends - the host calls session end, or it went idle - the facts
it kept as session state (``task``, ``unsure``, and in a named session a note of
each ``observed`` fact, never kept) and the newest turns of its
trace (the person's own words, sincere turns only) are read by the chat model
in **one call**, made only when the session holds a session fact: it may
write one **episode** ("Alice rehearsed a job interview with
the assistant as the recruiter"), and it may name, among the
``unsure`` ones only, those that are lasting truths. Like the dream, the
model proposes through a closed schema and the service decides (P8): the
answer is a JSON object with two keys, anything else is rejected and counted;
the episode and every kept fact then go through the ordinary write path of
their source - resolution, contradictions (P7), provenance (P6: the episode is
a fact about the person speaking, written only when it can lean on their own
words, otherwise proposed), the language checks (P12). Everything else
expires: the session's facts are deleted. A ``task`` fact never becomes
memory on its own.

The model call happens first, then everything is written in one transaction
(a budget that runs out puts the job back untouched, SPEC 7.4b).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from iifa import log
from iifa.ingest import sessions
from iifa.ingest.checks import is_complete, mixed_script_words
from iifa.ingest.extract import Extractor, Owner, _written_in, write_view
from iifa.ingest.intent import ACTIVITY, FACT, LASTING
from iifa.ingest.parts import OWNER, Part
from iifa.ingest.schema import STATEMENT_CHARS, Answer, ProposedFact, extract_object
from iifa.ingest.session_prompts import consolidation_messages
from iifa.language import UNDETERMINED, detect, writing_mismatch
from iifa.llm import TRUNCATED, ModelError
from iifa.recall.tokenizer import normalize
from iifa.store.repo import NotFound

if TYPE_CHECKING:
    from iifa.ingest.worker import JobContext

CONSOLIDATION_KEYS = ("episode", "keep")
MAX_NOTES = 40  # the newest session facts one consolidation reads
MAX_NOTE_CHARS = 8000  # and at most this many characters of them
MAX_TURNS = 40  # the newest turns of the trace one consolidation reads
MAX_TURN_CHARS = 6000  # and at most this many characters of them
SPEAKER_SUBJECT = "User"  # how the person speaking is named without a declared owner


@dataclass
class Consolidation:
    """What the model answered, once checked against the closed schema."""

    episode: str | None = None
    keep: list[int] = field(default_factory=list)
    rejected: int = 0
    usable: bool = True


def parse_consolidation(content: str, unsure: Sequence[int]) -> Consolidation:
    """``{"episode": "<one sentence>" | null, "keep": [<index>, ...]}``. An
    unknown key, an episode that is not a short string, an index that is not
    one of the ``unsure`` notes: rejected and counted, never applied."""
    data = extract_object(content, CONSOLIDATION_KEYS)
    if data is None:
        return Consolidation(rejected=1, usable=False)
    out = Consolidation(rejected=sum(1 for key in data if key not in CONSOLIDATION_KEYS))
    episode = data.get("episode")
    if isinstance(episode, str) and 0 < len(" ".join(episode.split())) <= STATEMENT_CHARS:
        out.episode = " ".join(episode.split())
    elif episode is not None:
        out.rejected += 1
    keep = data.get("keep")
    if isinstance(keep, list):
        for index in keep:
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and index in unsure
                and index not in out.keep
            ):
                out.keep.append(index)
            else:
                out.rejected += 1
    elif keep is not None:
        out.rejected += 1
    return out


def _window(facts: Sequence[sessions.SessionFact]) -> list[sessions.SessionFact]:
    """The newest facts, in order, within the note bounds: the work notes
    first, the observation notes in the room they leave (SPEC 5.8)."""
    chosen: set[int] = set()
    used = 0
    for observed in (False, True):
        for index in range(len(facts) - 1, -1, -1):
            fact = facts[index]
            if (fact.duration == sessions.OBSERVED) != observed:
                continue
            size = len(fact.statement) + 16
            if len(chosen) >= MAX_NOTES or (chosen and used + size > MAX_NOTE_CHARS):
                break
            chosen.add(index)
            used += size
    return [facts[i] for i in sorted(chosen)]


def _turn_window(turns: Sequence[sessions.SessionTurn]) -> list[sessions.SessionTurn]:
    """The newest turns of the trace, in order, within the turn bounds."""
    chosen: list[sessions.SessionTurn] = []
    used = 0
    for turn in reversed(turns):
        size = len(turn.excerpt) + 4
        if len(chosen) >= MAX_TURNS or (chosen and used + size > MAX_TURN_CHARS):
            break
        chosen.append(turn)
        used += size
    return list(reversed(chosen))


def _language(texts: Sequence[str], forced: str | None) -> str | None:
    """The episode's target language (P12): the scope's forced one, else the
    majority language of the notes and turns (undetermined ones left out)."""
    if forced:
        return forced
    counts = Counter(lang for lang in (detect(text) for text in texts) if lang != UNDETERMINED)
    return counts.most_common(1)[0][0] if counts else None


def close_session(
    ctx: JobContext, session_id: str | None, scopes: Sequence[str], *, reason: str
) -> dict[str, Any]:
    """Consolidate and close one session's state; a report of counts only."""
    store = ctx.store
    state = sessions.load(store, session_id, scopes)
    report: dict[str, Any] = {
        "reason": reason,
        "session": "none" if state.id is None else "closed",
        "facts": len(state.facts),
        "read": 0,
        "unread": 0,
        "turns": 0,
        "calls": 0,
        "rejected": 0,
        "episode": None,
        "kept": 0,
        "written": 0,
        "proposed": 0,
        "expired": 0,
    }
    if state.id is None:
        return report
    # an observation note is memory already: it goes with the session, unexpired
    notes = sum(1 for f in state.facts if f.duration == sessions.OBSERVED)
    if not state.facts or not ctx.models.has_chat:
        report["expired"] = max(0, sessions.close(store, state) - notes)
        log.event("session_closed", facts=report["facts"], expired=report["expired"])
        return report
    window = _window(state.facts)
    turns = [sessions.safe_line(t.excerpt) for t in _turn_window(state.turns)]
    report["read"] = len(window)
    report["turns"] = len(turns)
    report["unread"] = len(state.facts) - len(window)  # older than the window: they expire
    config = ctx.config
    forced = config.forced_language(state.scopes)
    owner = config.owner(state.scopes)
    agent = config.agent(state.scopes)
    subject = owner.name if owner is not None else SPEAKER_SUBJECT
    partner: str | None = None
    if agent is not None:
        # SPEC 5.8, agent perspective: one episode, about the self
        partner = subject if owner is not None else "the user"
        subject = agent.name
    language = _language([*(f.statement for f in window), *turns], forced)
    answer, calls = _ask(ctx, window, turns, subject, language, partner)
    report["calls"], report["rejected"] = calls, answer.rejected
    # SPEC 5.8: an episode naming a known subject the session never named is
    # only proposed - the model may have filled a generic word with it
    # SPEC 4.2, 5.8: the episode's activities - none in the user perspective,
    # else the session's that are still live, visible task nodes (one list
    # for the episode written and the episode proposed)
    state = replace(state, activities=_live_activities(ctx, state) if agent else ())
    doubtful = _names_unnamed_subject(ctx, state, answer.episode, owner, agent)
    plans = _plans(ctx, state, window, answer, subject, write_episode=not doubtful)
    with store.lock, store.transaction():
        for extractor, source_id, text, plan in plans:
            extractor.commit(source_id, text, plan)
            report["written"] += extractor.report.facts_written
            report["proposed"] += extractor.report.facts_proposed
        if doubtful and answer.episode:
            report["proposed"] += _propose_episode(ctx, state, window, answer.episode, subject)
        report["expired"] = max(0, sessions.close(store, state) - len(answer.keep) - notes)
    report["episode"] = "asked" if answer.episode else None
    report["kept"] = len(answer.keep)
    log.event(
        "session_closed",
        facts=report["facts"],
        written=report["written"],
        proposed=report["proposed"],
        expired=report["expired"],
    )
    return report


def _ask(
    ctx: JobContext,
    window: Sequence[sessions.SessionFact],
    turns: Sequence[str],
    subject: str,
    language: str | None,
    partner: str | None = None,
) -> tuple[Consolidation, int]:
    """The one consolidation call, asked again once when the answer is not a
    JSON object or the episode is not in the target language (P12)."""
    items = [(i, f.duration, sessions.safe_line(f.statement)) for i, f in enumerate(window)]
    unsure = [i for i, f in enumerate(window) if f.duration == sessions.UNSURE]
    config, calls = ctx.config, 0
    note: str | None = None
    answer = Consolidation(usable=False)
    for _attempt in range(2):
        try:
            result = ctx.models.chat(
                consolidation_messages(
                    items,
                    subject=subject,
                    language=language,
                    turns=turns,
                    retry_note=note,
                    partner=partner,
                ),
                json_object=config.json_mode,
                max_tokens=config.answer_cap(" ".join(f.statement for f in window)),
            )
            content = result.content
        except ModelError as exc:
            if exc.reason != TRUNCATED:
                raise
            content = ""  # a truncated answer is never parsed: nothing is kept
        calls += 1
        answer = parse_consolidation(content, unsure)
        if not answer.usable:
            note = "Answer with the JSON object only, no prose, no code fence."
            continue
        if answer.episode and language and writing_mismatch(answer.episode, language):
            note = (
                "Your previous answer wrote the episode in the wrong language. Write it in the "
                "language of the notes. Do not translate."
            )
            continue
        break
    return answer, calls


def _plans(
    ctx: JobContext,
    state: sessions.SessionState,
    window: Sequence[sessions.SessionFact],
    answer: Consolidation,
    subject: str,
    *,
    write_episode: bool = True,
) -> list[tuple[Extractor, str, str, Any]]:
    """Each source's facts to write - the kept ones, and the episode under the
    source of the person's own words when there are any (P6) - planned through
    the ordinary write path, before the write transaction."""
    by_source: dict[str, list[tuple[ProposedFact, sessions.SessionFact]]] = {}
    for index in answer.keep:
        note = window[index]
        if note.source_id is not None:
            fact = note.fact.model_copy(update={"modality": FACT, "duration": LASTING})
            by_source.setdefault(note.source_id, []).append((fact, note))
    episode = answer.episode if write_episode else None
    if episode and is_complete(episode) and not mixed_script_words(episode):
        anchor = _anchor(window)
        if anchor is not None and anchor.source_id is not None:
            fact = ProposedFact(
                subject=subject,
                statement=episode,
                attribute=ACTIVITY,
                valid_from=(state.started_at or "")[:10] or None,
                quote=anchor.fact.quote,
                confidence=0.7,
                modality=FACT,
                duration=LASTING,
                episode=True,
            )
            by_source.setdefault(anchor.source_id, []).append((fact, anchor))
    plans: list[tuple[Extractor, str, str, Any]] = []
    for source_id, items in by_source.items():
        try:
            source = ctx.store.get_source(source_id)
        except NotFound:  # forgotten meanwhile: nothing of it is written
            continue
        text = source.text or ""
        parts = [
            Part(
                note.span[0],
                note.span[1],
                note.provenance,
                kind=note.part_kind,
                ref=note.part_ref,
            )
            for _, note in items
            if note.span is not None and note.provenance is not None
        ]
        # P6: one source is one turn, read on one day
        read_at = next((note.read_at for _, note in items if note.read_at), None)
        extractor = Extractor(ctx, list(state.scopes))
        plan = extractor.prepare(
            text,
            parts,
            [(0, Answer(facts=[f for f, _ in items]))],
            session_id=state.key or None,
            read_at=read_at,
            external=source.external,  # P6: the parts' offsets are not kept
            activities=state.activities,  # SPEC 5.8: the episode's links
        )
        plans.append((extractor, source_id, text, plan))
    return plans


def _live_activities(ctx: JobContext, state: sessions.SessionState) -> tuple[str, ...]:
    view = write_view(list(state.scopes))
    nodes = ctx.store.nodes_by_id(list(state.activities), view)
    return tuple(a for a in state.activities if a in nodes and nodes[a].kind == "task")


def _anchor(window: Sequence[sessions.SessionFact]) -> sessions.SessionFact | None:
    """The note whose source the episode is written under: the person's own
    words when there are any (P6), else any note with a source."""
    anchor = next((f for f in window if f.provenance == OWNER and f.source_id), None)
    return anchor or next((f for f in window if f.source_id), None)


def _names_unnamed_subject(
    ctx: JobContext,
    state: sessions.SessionState,
    episode: str | None,
    owner: Owner | None,
    agent: Owner | None = None,
) -> bool:
    """SPEC 5.8: does ``episode`` name a known subject - a visible node's name
    or alias, whole words (SPEC 5.1), not the person speaking nor the self nor
    the session's current activity (set from the owner's words) - that no
    source of the session names?"""
    if not episode:
        return False
    view = write_view(list(state.scopes))
    told = ctx.store.session_texts(state.key, view) if state.key else []
    speaker = {
        normalize(n)
        for declared in (owner, agent)
        if declared is not None
        for n in (declared.name, *declared.aliases)
    }
    for node_id, _kind, name, name_norm, alias_norms in ctx.store.node_index(view):
        if name_norm in speaker or node_id in state.activities:
            continue
        names = [name, *alias_norms]
        if _written_in(episode, names) and not any(_written_in(t, names) for t in told):
            return True
    return False


def _propose_episode(
    ctx: JobContext,
    state: sessions.SessionState,
    window: Sequence[sessions.SessionFact],
    episode: str,
    subject: str,
) -> int:
    """The episode as a fact proposal on the person speaking (SPEC 4.4b), decided
    only by a ``review`` token; 1 when it was recorded."""
    anchor = _anchor(window)
    if anchor is None or not is_complete(episode) or mixed_script_words(episode):
        return 0
    node = ctx.store.find_node(subject, write_view(list(state.scopes)))
    ctx.store.add_proposal(
        scopes=state.scopes,
        kind="fact",
        node_id=node.id if node else None,
        value=episode,
        source_id=anchor.source_id,
        excerpt=anchor.span,
        confidence=0.7,
        payload={
            "attribute": ACTIVITY,
            "value": None,
            "valid_from": (state.started_at or "")[:10] or None,
            "flags": ["episode"],
            "provenance": anchor.provenance,
            "confidence": 0.7,
            "why": "session:unnamed",
            "subject": subject,
            **({"activities": list(state.activities)} if state.activities else {}),
        },
    )
    return 1


__all__ = ["CONSOLIDATION_KEYS", "Consolidation", "close_session", "parse_consolidation"]
