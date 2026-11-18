"""Intent before content (SPEC 4.2): was it meant, as a fact, for good?

The extraction answer says, for the source as a whole, how its words are meant
(the **frame**: ``sincere``, ``pretend``, ``example``, ``irony``, ``voice``,
``staged``, or ``unsure``), and for each fact whether it is stated as true (its
**modality**: ``fact``, ``question``, ``hypothesis``, ``plan``, ``guess``,
``proposal``, or ``unsure``) and how long it holds (its **duration**:
``lasting``, ``task``, ``observed``, or ``unsure``). The model reads; the service decides,
here, from those closed words only - never from a word list, which would be
language-bound and would drop "I test cars for a living":

* only a ``sincere`` frame and a ``fact`` modality write a fact;
* a non-sincere frame writes nothing from its content, nodes and edges
  included, but may leave **one episode**: a sincere fact about what the person
  speaking did ("rehearsed a job interview with the assistant as the recruiter"), stored
  under the key ``activity`` and flagged ``episode``, never carrying a value of
  the content it reframes (one that does is only proposed);
* a frame or a modality that is ``unsure`` - or not given at all - makes a
  **fact proposal** (SPEC 4.4b): a wrong fact costs more than a missing one;
* an ``observed`` duration is memory like ``lasting``, true at a date: the
  write path flags and dates it;
* a ``task`` or ``unsure`` duration keeps the fact as **session state**
  (``iifa.ingest.sessions``), never as memory: the session's consolidation
  decides what is worth keeping;
* a frame set to govern the turns that follow is **carried** by the session
  until a turn ends it or the session closes. A turn the model reads as
  sincere while a carried frame is open, without saying the frame ended, is a
  disagreement, so a doubt: its facts are proposed.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from iifa.ingest.schema import Answer, ProposedEdge, ProposedFact
from iifa.recall.tokenizer import normalize, tokenize

SINCERE, UNSURE = "sincere", "unsure"
FACT: Final = "fact"
LASTING: Final = "lasting"
TASK: Final = "task"
OBSERVED: Final = "observed"  # true at a date: memory, dated (SPEC 4.2)
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ACTIVITY = "activity"  # the key every episode is stored under
EPISODE = "episode"  # the flag every episode carries


@dataclass(frozen=True)
class Carry:
    """The frame a session carries from turn to turn, and whether its one
    episode is written already."""

    frame: str | None = None
    episode: bool = False


@dataclass
class IntentReport:
    """What the intent reading did to one job: counts and closed words only."""

    frames: Counter[str] = field(default_factory=Counter)  # the frame each answer was read in
    dropped_frame: Counter[str] = field(default_factory=Counter)
    dropped_modality: Counter[str] = field(default_factory=Counter)
    proposed: int = 0
    session_task: int = 0
    session_unsure: int = 0
    episodes: Counter[str] = field(default_factory=Counter)  # kept | proposed | dropped
    opened: int = 0
    ended: int = 0
    carried: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": dict(sorted(self.frames.items())),
            "proposed": self.proposed,
            "session": {"task": self.session_task, "unsure": self.session_unsure},
            "episodes": {k: self.episodes.get(k, 0) for k in ("kept", "proposed", "dropped")},
            "frame_opened": self.opened,
            "frame_ended": self.ended,
            "frame_carried": self.carried,
        }

    def dropped(self) -> dict[str, dict[str, int]]:
        return {
            "frame": dict(sorted(self.dropped_frame.items())),
            "modality": dict(sorted(self.dropped_modality.items())),
        }


@dataclass
class Reading:
    """One answer, split by intent: what goes on to the write path, what is
    only proposed, what is kept as session state."""

    kept: Answer
    proposed: list[ProposedFact] = field(default_factory=list)
    session: list[ProposedFact] = field(default_factory=list)
    frame: str = UNSURE  # the frame the answer was read in, carried frame included
    ended: bool = False  # this answer ended the frame the session carried


def read(
    answer: Answer,
    carry: Carry,
    *,
    about_speaker: Callable[[str | None], bool],
    report: IntentReport,
    date: str | None = None,
) -> tuple[Reading, Carry]:
    """Split ``answer`` by intent, given the frame the session carries. Returns
    the reading and the frame carried to the next answer. ``about_speaker``
    tells whether a subject is the person speaking (an episode is about them);
    ``date`` is the source's date, the episode's ``valid_from`` when the model
    gave none."""
    said = answer.frame or UNSURE
    ended = answer.frame_ends and carry.frame is not None
    if answer.frame_ends:
        report.ended += int(ended)
        carry = Carry()
    if carry.frame is not None:
        report.carried += 1
        frame = UNSURE if said in (SINCERE, UNSURE) else said
    else:
        frame = said
    report.frames[frame] += 1
    kept = Answer(rejected=answer.rejected, usable=answer.usable)
    reading = Reading(kept=kept, frame=frame, ended=ended)

    if frame == UNSURE:
        reading.proposed = list(answer.facts)
        report.proposed += len(answer.facts)
        return reading, carry

    if frame != SINCERE:
        return reading, _reframe(answer, carry, reading, about_speaker, report, date)
    _sincere(answer, reading, report)
    return reading, carry


def _reframe(
    answer: Answer,
    carry: Carry,
    reading: Reading,
    about_speaker: Callable[[str | None], bool],
    report: IntentReport,
    date: str | None,
) -> Carry:
    """A non-sincere frame: nothing of its content goes on, its one episode
    does (or is proposed when it carries a value of the content); the frame is
    carried when it holds, or when the session carried it in. Returns the
    frame carried out."""
    frame = reading.frame
    episode_done = carry.episode if carry.frame is not None else False
    episode = _episode(answer, episode_done, about_speaker, report, date)
    if episode is not None:
        if _leaks(episode, [f for f in answer.facts if not f.episode]):
            reading.proposed.append(episode)
            report.proposed += 1
            report.episodes["proposed"] += 1
        else:
            reading.kept.facts.append(episode)
            report.episodes["kept"] += 1
        episode_done = True
    report.dropped_frame[frame] += sum(1 for f in answer.facts if not f.episode)
    if answer.frame_holds or carry.frame is not None:
        report.opened += int(carry.frame is None)
        return Carry(frame, episode_done)
    return carry


def _sincere(answer: Answer, reading: Reading, report: IntentReport) -> None:
    """A sincere frame: each fact by its modality, then by its duration."""
    removed: list[ProposedFact] = []
    for said in answer.facts:
        # in a sincere frame there is no content to reframe: no fact is an episode
        fact = said.model_copy(update={"episode": False}) if said.episode else said
        modality = fact.modality or UNSURE
        if modality == UNSURE:
            reading.proposed.append(fact)
            report.proposed += 1
            removed.append(fact)
        elif modality != FACT:
            report.dropped_modality[modality] += 1
            removed.append(fact)
        elif (fact.duration or UNSURE) in (LASTING, OBSERVED):
            reading.kept.facts.append(fact)
        else:
            reading.session.append(fact)
            if fact.duration == TASK:
                report.session_task += 1
            else:
                report.session_unsure += 1
            removed.append(fact)
    reading.kept.edges, reading.kept.nodes = _prune(answer, reading.kept.facts, removed)


def _episode(
    answer: Answer,
    done: bool,
    about_speaker: Callable[[str | None], bool],
    report: IntentReport,
    date: str | None,
) -> ProposedFact | None:
    """The one episode a non-sincere frame may leave: the first fact marked
    ``episode``, about the person speaking, when the frame has none yet. Stored
    under ``activity`` with no value (an episode never replaces another) and
    flagged by the write path. Every other one is dropped, and counted."""
    candidates = [f for f in answer.facts if f.episode]
    chosen: ProposedFact | None = None
    for fact in candidates:
        if chosen is None and not done and about_speaker(fact.subject):
            chosen = fact
        else:
            report.episodes["dropped"] += 1
    if chosen is None:
        return None
    return chosen.model_copy(
        update={
            "attribute": ACTIVITY,
            "value": None,
            "valid_to": None,
            "valid_from": chosen.valid_from or _day(date),
            "modality": FACT,
            "duration": LASTING,
        }
    )


def _leaks(episode: ProposedFact, content: Sequence[ProposedFact]) -> bool:
    """Does the episode carry a value of the content it reframes (whole words,
    folded)? A role's city in "played a pilot named X from Y"."""
    words = tokenize(episode.statement)
    for fact in content:
        wanted = tokenize(fact.value or "")
        if wanted and any(
            words[i : i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1)
        ):
            return True
    return False


def _prune(
    answer: Answer, kept: Sequence[ProposedFact], removed: Sequence[ProposedFact]
) -> tuple[list[ProposedEdge], list[Any]]:
    """The edges and nodes that stay once some facts left the answer: an edge
    quoting the words of a fact that left goes with it ("if I moved to Porto"
    and its ``lives_in``), and a node named only by facts and edges that left
    is not created."""
    if not removed:
        return list(answer.edges), list(answer.nodes)
    gone_quotes = [q for q in (_norm(f.quote) for f in removed) if q]
    edges = [e for e in answer.edges if not any(_overlaps(_norm(e.quote), q) for q in gone_quotes)]
    dropped_edges = [e for e in answer.edges if e not in edges]
    used = {normalize(f.subject or "") for f in kept}
    used |= {normalize(name) for e in edges for name in (e.src, e.dst)}
    gone = {normalize(f.subject or "") for f in removed}
    gone |= {normalize(name) for e in dropped_edges for name in (e.src, e.dst)}
    nodes = [n for n in answer.nodes if normalize(n.name) in used or normalize(n.name) not in gone]
    return edges, nodes


def _norm(text: str | None) -> str:
    return " ".join(tokenize(text or ""))


def _overlaps(a: str, b: str) -> bool:
    """Does one run of words hold the other (whole words: "art" is not in "start")?"""
    return bool(a) and bool(b) and (f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def _day(date: str | None) -> str | None:
    """The source's day, when its date is one (a precompact's trigger is not)."""
    day = (date or "")[:10]
    return day if _DAY.match(day) else None


__all__ = [
    "ACTIVITY",
    "EPISODE",
    "FACT",
    "LASTING",
    "OBSERVED",
    "SINCERE",
    "TASK",
    "UNSURE",
    "Carry",
    "IntentReport",
    "Reading",
    "read",
]
