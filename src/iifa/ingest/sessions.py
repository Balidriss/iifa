"""Session state: what the service keeps for one session while it runs.

The extraction reads one source at a time, yet how a turn is meant often depends
on the turns before it ("for this test, I'm a customer named Alex" governs what
follows) and a work session changes the state of one thing turn after turn. So
the service keeps, per session, the **frame** an earlier turn opened
and the **session facts**: facts that hold only for the work in progress
(``task``) or whose duration the model could not tell (``unsure``), and, in a
session the host named, a note of each fact read as ``observed`` (it is memory
already; the note only tells the close what was done). They are
shown to the next extraction calls of the session, never recalled, never
exported, and they live until the session is closed (``iifa.ingest.consolidate``).
In a session the host named, it also keeps the **turn trace**: the person's own
words of every turn read as sincere, bounded, read by the close only, so that
it can say what the session was about when the extraction kept few notes.

A session is keyed by the host's session id **and** the exact scope set of its
turns (P9): two tokens that happen to send the same session id into different
scopes never see each other's state. A turn sent without a session id belongs
to the scope set's default session (key ``""``), which the idle timeout closes.

Everything here is data written by the model from a source: it is encrypted
like the rest of the file, it goes with its source when the source is
forgotten (P5), and a forgotten node takes the session facts about it, and the
traces that name it, along.

In the agent perspective, a session the host named also keeps the
**activities** its turns named in the owner's words, the last
named being the **current** one. The current one is shown to the next
extraction calls, so that a rule read later is linked to what the owner
asked; the close links its episode to all of them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from iifa.ids import new_id
from iifa.ingest.credentials import mask
from iifa.ingest.parts import OWNER, Part
from iifa.ingest.schema import ProposedFact
from iifa.recall.tokenizer import normalize
from iifa.store.repo import Store

TASK, UNSURE = "task", "unsure"
OBSERVED = "observed"  # a note of an observation, memory already
SESSION_CLOSE = "session_close"  # the job that closes a session nobody ended
TURN_TRACE_CHARS = 400  # the person's words one turn leaves in the trace, at most


def scopes_key(scopes: Iterable[str]) -> str:
    """The scope set as the state stores it: sorted, without duplicates."""
    return json.dumps(sorted(set(scopes)), ensure_ascii=False)


def session_key(session_id: str | None) -> str:
    return session_id or ""


@dataclass(frozen=True)
class SessionFact:
    """A fact kept as session state: never memory until a consolidation keeps it."""

    id: str
    source_id: str | None
    duration: str
    statement: str
    provenance: str | None
    span: tuple[int, int] | None
    fact: ProposedFact
    created_at: str
    part_kind: str | None = None  # P6: an external note's part, as a fact keeps it
    part_ref: str | None = None
    read_at: str | None = None


@dataclass(frozen=True)
class SessionTurn:
    """One turn of the trace: the person's own words, read by the close only."""

    id: str
    source_id: str | None
    excerpt: str
    created_at: str


@dataclass(frozen=True)
class NewSessionFact:
    source_id: str
    duration: str
    fact: ProposedFact
    provenance: str | None
    span: tuple[int, int] | None
    part_kind: str | None = None
    part_ref: str | None = None
    read_at: str | None = None


@dataclass
class SessionState:
    key: str
    scopes: tuple[str, ...]
    id: str | None = None  # None: nothing is stored for this session yet
    frame: str | None = None
    frame_episode: bool = False
    rev: int = 0
    # the activities the session named, the current one last
    activities: tuple[str, ...] = ()
    started_at: str | None = None
    last_seen_at: str | None = None
    facts: list[SessionFact] = field(default_factory=list)
    turns: list[SessionTurn] = field(default_factory=list)

    @property
    def activity_id(self) -> str | None:
        """The current activity, the last one named."""
        return self.activities[-1] if self.activities else None


def load(store: Store, session_id: str | None, scopes: Sequence[str]) -> SessionState:
    """The state of one session (an empty one when nothing is stored)."""
    key, scope_set = session_key(session_id), scopes_key(scopes)
    with store.lock:
        row = store.execute(
            "SELECT * FROM sessions WHERE session_key = ? AND scopes = ?", (key, scope_set)
        ).fetchone()
        if row is None:
            return SessionState(key, tuple(json.loads(scope_set)))
        return _state(store, row)


def get(store: Store, state_id: str) -> SessionState | None:
    with store.lock:
        row = store.execute("SELECT * FROM sessions WHERE id = ?", (state_id,)).fetchone()
        return None if row is None else _state(store, row)


def _state(store: Store, row: Any) -> SessionState:
    facts = [
        SessionFact(
            id=r["id"],
            source_id=r["source_id"],
            duration=r["duration"],
            statement=r["statement"],
            provenance=r["provenance"],
            span=None
            if r["excerpt_start"] is None
            else (int(r["excerpt_start"]), int(r["excerpt_end"])),
            fact=ProposedFact.model_validate(json.loads(r["fact"])),
            created_at=r["created_at"],
            part_kind=r["part_kind"],
            part_ref=r["part_ref"],
            read_at=r["read_at"],
        )
        for r in store.execute(
            "SELECT * FROM session_facts WHERE session_id = ? ORDER BY created_at, id",
            (row["id"],),
        ).fetchall()
    ]
    turns = [
        SessionTurn(
            id=r["id"], source_id=r["source_id"], excerpt=r["excerpt"], created_at=r["created_at"]
        )
        for r in store.execute(
            "SELECT * FROM session_turns WHERE session_id = ? ORDER BY created_at, id",
            (row["id"],),
        ).fetchall()
    ]
    return SessionState(
        key=row["session_key"],
        scopes=tuple(json.loads(row["scopes"])),
        id=row["id"],
        frame=row["frame"],
        frame_episode=bool(row["frame_episode"]),
        rev=int(row["rev"]),
        activities=_activities(store, row["id"]),
        started_at=row["started_at"],
        last_seen_at=row["last_seen_at"],
        facts=facts,
        turns=turns,
    )


def _activities(store: Store, state_id: str) -> tuple[str, ...]:
    """The session's activities, the current one last, each merged
    one read as the node it was merged into (once, at its latest place)."""
    rows = store.execute(
        "SELECT activity_id FROM session_activities WHERE session_id = ? ORDER BY seq",
        (state_id,),
    ).fetchall()
    followed = [store.merge_target(r["activity_id"]) for r in rows]
    latest = {node: i for i, node in enumerate(followed) if node is not None}
    return tuple(sorted(latest, key=latest.__getitem__))


def trace_of(text: str, parts: Sequence[Part], *, secrets: Sequence[str] = ()) -> str | None:
    """The words a turn leaves in the trace: its owner parts only (P6: what the
    assistant answered or a page said is not the person's), each of ``secrets``
    masked (P5), whitespace collapsed, cut on a word within
    ``TURN_TRACE_CHARS``; None when empty."""
    words = " ".join(
        " ".join(mask(text[part.start : part.end], secrets).split())
        for part in parts
        if part.provenance == OWNER
    ).strip()
    if len(words) > TURN_TRACE_CHARS:
        cut = words[: TURN_TRACE_CHARS + 1]
        words = (cut.rsplit(" ", 1)[0] if " " in cut else cut[:TURN_TRACE_CHARS]).rstrip()
    return words or None


def save(
    store: Store,
    state: SessionState,
    *,
    frame: str | None,
    frame_episode: bool,
    ended: bool,
    facts: Sequence[NewSessionFact],
    trace: tuple[str, str] | None = None,
    activity: str | None = None,
) -> str | None:
    """Write what one source changed in its session, inside the caller's
    transaction: the frame now open (or none), whether its episode is written,
    the new session facts, the turn's trace ``(source_id, excerpt)`` and the
    activity this source named (None: the current one stays).
    ``state`` is what the extraction read before its model calls; when the row
    changed since (another turn of the session was
    processed meanwhile), a frame that turn opened stays open unless this source
    ended it: an open frame is the cautious reading. A session left with no
    frame, no fact and no trace is deleted. Returns the row's id, or None when deleted."""
    now = store.now()
    with store.lock, store.transaction():
        row = store.execute(
            "SELECT id, frame, frame_episode, rev FROM sessions"
            " WHERE session_key = ? AND scopes = ?",
            (state.key, scopes_key(state.scopes)),
        ).fetchone()
        changed = row is not None and (state.id is None or int(row["rev"]) != state.rev)
        if changed and not ended and frame is None:
            frame, frame_episode = row["frame"], bool(row["frame_episode"])
        if row is None:
            if frame is None and activity is None and not facts and trace is None:
                return None
            state_id = new_id("sess")
            store.execute(
                "INSERT INTO sessions (id, session_key, scopes, frame, frame_episode, rev,"
                " started_at, last_seen_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    state_id,
                    state.key,
                    scopes_key(state.scopes),
                    frame,
                    int(frame_episode and frame is not None),
                    now,
                    now,
                ),
            )
        else:
            state_id = row["id"]
            store.execute(
                "UPDATE sessions SET frame = ?, frame_episode = ?, rev = rev + 1,"
                " last_seen_at = ? WHERE id = ?",
                (frame, int(frame_episode and frame is not None), now, state_id),
            )
        if activity is not None:
            # named (again), it is the current one; another turn's stay
            store.execute(
                "INSERT INTO session_activities (session_id, activity_id, seq) VALUES (?, ?,"
                " (SELECT coalesce(max(seq), 0) + 1 FROM session_activities WHERE session_id = ?))"
                " ON CONFLICT (session_id, activity_id) DO UPDATE SET seq = excluded.seq",
                (state_id, activity, state_id),
            )
        for item in facts:
            store.execute(
                "INSERT INTO session_facts (id, session_id, source_id, duration, statement,"
                " subject_norm, provenance, excerpt_start, excerpt_end, fact, created_at,"
                " part_kind, part_ref, read_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id("sfact"),
                    state_id,
                    item.source_id,
                    item.duration,
                    item.fact.statement,
                    normalize(item.fact.subject or "") or None,
                    item.provenance,
                    item.span[0] if item.span else None,
                    item.span[1] if item.span else None,
                    item.fact.model_dump_json(),
                    now,
                    item.part_kind,
                    item.part_ref,
                    item.read_at,
                ),
            )
        if trace is not None:
            store.execute(
                "INSERT INTO session_turns (id, session_id, source_id, excerpt, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (new_id("sturn"), state_id, trace[0], trace[1], now),
            )
        if frame is None and activity is None:
            left = store.execute(
                "SELECT 1 FROM session_facts WHERE session_id = ?"
                " UNION ALL SELECT 1 FROM session_turns WHERE session_id = ?"
                " UNION ALL SELECT 1 FROM session_activities WHERE session_id = ? LIMIT 1",
                (state_id, state_id, state_id),
            ).fetchone()
            if left is None:
                drop(store, state_id)
                return None
        return str(state_id)


def drop(store: Store, state_id: str) -> int:
    """Delete a session's state, facts and trace included (they expire)."""
    with store.lock, store.transaction():
        facts = store.execute(
            "DELETE FROM session_facts WHERE session_id = ?", (state_id,)
        ).rowcount
        store.execute("DELETE FROM session_turns WHERE session_id = ?", (state_id,))
        store.execute("DELETE FROM sessions WHERE id = ?", (state_id,))
        return int(facts)


def idle(store: Store, before: str) -> list[SessionState]:
    """The sessions last seen before ``before`` that no open job closes yet
    (a close job that failed for good no longer counts), oldest first."""
    with store.lock:
        rows = store.execute(
            "SELECT * FROM sessions WHERE last_seen_at < ? AND (closing_job IS NULL"
            " OR closing_job NOT IN (SELECT id FROM queue WHERE status IN ('pending', 'running')))"
            " ORDER BY last_seen_at, id",
            (before,),
        ).fetchall()
        return [_state(store, row) for row in rows]


def release_closing(store: Store, state_id: str) -> None:
    """The session is active again: no job closes it."""
    with store.lock, store.transaction():
        store.execute("UPDATE sessions SET closing_job = NULL WHERE id = ?", (state_id,))


def mark_closing(store: Store, state_id: str, job_id: str) -> None:
    with store.lock, store.transaction():
        store.execute("UPDATE sessions SET closing_job = ? WHERE id = ?", (job_id, state_id))


def open_count(store: Store) -> int:
    with store.lock:
        return int(store.execute("SELECT count(*) FROM sessions").fetchone()[0])


def close(store: Store, state: SessionState) -> int:
    """After a consolidation read ``state``: its facts and trace are deleted (they
    expire), and the session with them unless a turn changed it
    meanwhile - then what that turn added stays, for the next close. Returns the facts
    deleted. The caller holds the store's transaction."""
    if state.id is None:
        return 0
    with store.lock, store.transaction():
        deleted = 0
        for fact in state.facts:
            deleted += int(
                store.execute("DELETE FROM session_facts WHERE id = ?", (fact.id,)).rowcount
            )
        for turn in state.turns:
            store.execute("DELETE FROM session_turns WHERE id = ?", (turn.id,))
        row = store.execute("SELECT rev FROM sessions WHERE id = ?", (state.id,)).fetchone()
        if row is not None and int(row["rev"]) == state.rev:
            store.execute("DELETE FROM session_facts WHERE session_id = ?", (state.id,))
            store.execute("DELETE FROM session_turns WHERE session_id = ?", (state.id,))
            store.execute("DELETE FROM sessions WHERE id = ?", (state.id,))
        elif row is not None:
            store.execute("UPDATE sessions SET closing_job = NULL WHERE id = ?", (state.id,))
        return deleted


def safe_line(text: str) -> str:
    """Stored text never closes the block it is shown in (P8)."""
    return " ".join(text.replace("<<<", "< < <").replace(">>>", "> > >").split())


def render(state: SessionState, limit: int) -> str | None:
    """What the next extraction call of the session is shown, or
    None when there is nothing: the open frame, then the newest session facts."""
    lines: list[str] = []
    if state.frame is not None:
        lines.append(f"Frame opened by an earlier turn of this session, still open: {state.frame}.")
    # an observation note is memory already: not work in progress to show
    work = [f for f in state.facts if f.duration != OBSERVED]
    shown = work[-limit:] if limit > 0 else []
    if shown:
        lines.append("Work in progress noted earlier in this session (not memory):")
        lines.extend(f"- {safe_line(f.statement)}" for f in shown)
    return "\n".join(lines) if lines else None


__all__ = [
    "OBSERVED",
    "SESSION_CLOSE",
    "TASK",
    "TURN_TRACE_CHARS",
    "UNSURE",
    "NewSessionFact",
    "SessionFact",
    "SessionState",
    "SessionTurn",
    "close",
    "drop",
    "get",
    "idle",
    "load",
    "mark_closing",
    "open_count",
    "release_closing",
    "render",
    "safe_line",
    "save",
    "scopes_key",
    "session_key",
    "trace_of",
]
