"""What the ``--intent`` and ``--worksession`` corpora measure (SPEC 11.1).

``--intent``: the turns not meant as facts (hypotheses, questions, guesses,
possibilities, examples, test personas) must write nothing from their content
- counted like chatter - and leave exactly the episodes an ideal model leaves;
their sincere **controls**, the turns that state the same values, must lose
nothing (over-filtering is the opposite failure, and it counts as one).

``--worksession``: after the session's close, no fact of the work in progress
is live in memory, its state is gone, one episode is left, and the lasting
events told in between were all written.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from bench.synthetic import WORK_SESSION, Corpus, Turn

from iifa.store.repo import Store

EPISODE = "episode"


class IntentTally:
    """Read from each turn's extraction report: what intent turns still
    produced, what their controls kept, and what the intent reading dropped."""

    def __init__(self) -> None:
        self.unmeant = {
            "turns": 0,
            "turns_with_facts": 0,
            "facts": 0,
            "nodes": 0,
            "proposed": 0,
            "episodes": 0,
            "expected_episodes": 0,
        }
        self.by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"turns": 0, "facts": 0})
        self.controls = {"turns": 0, "written": 0, "duplicate": 0, "proposed": 0, "lost": 0}
        self.dropped: dict[str, dict[str, int]] = {"frame": {}, "modality": {}}
        self.session = {"task": 0, "unsure": 0}

    def add(self, turn: Turn, result: Mapping[str, Any]) -> None:
        facts = result.get("facts") or {}
        intent = result.get("intent") or {}
        episodes = int((intent.get("episodes") or {}).get("kept") or 0)
        for axis, counts in (result.get("dropped_by_intent") or {}).items():
            for word, n in counts.items():
                self.dropped[axis][word] = self.dropped[axis].get(word, 0) + int(n)
        for name in ("task", "unsure"):
            self.session[name] += int((intent.get("session") or {}).get(name) or 0)
        written = int(facts.get("written") or 0)
        if turn.intent is not None and turn.intent != "task":
            content = max(0, written - episodes)
            created = int((result.get("nodes") or {}).get("created") or 0)
            u = self.unmeant
            u["turns"] += 1
            u["facts"] += content
            u["nodes"] += created
            u["proposed"] += int(facts.get("proposed") or 0)
            u["episodes"] += episodes
            u["expected_episodes"] += int(turn.episode)
            u["turns_with_facts"] += int(bool(content or created))
            self.by_kind[turn.intent]["turns"] += 1
            self.by_kind[turn.intent]["facts"] += content
        if turn.control:
            c = self.controls
            c["turns"] += 1
            c["written"] += int(written > 0)
            c["duplicate"] += int(written == 0 and int(facts.get("duplicate") or 0) > 0)
            c["proposed"] += int(written == 0 and int(facts.get("proposed") or 0) > 0)
            c["lost"] += int(written == 0 and not facts.get("duplicate"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "unmeant": dict(self.unmeant),
            "by_kind": {k: dict(v) for k, v in sorted(self.by_kind.items())},
            "controls": dict(self.controls),
            "dropped_by_intent": {k: dict(sorted(v.items())) for k, v in self.dropped.items()},
            "session_state": dict(self.session),
        }


def worksession_audit(store: Store, corpus: Corpus, turns: Mapping[str, str]) -> dict[str, Any]:
    """After the work session's close: its task facts live in memory (must be
    0), its state left (must be 0), the episodes written from it (one), and
    the lasting events told inside it that were written."""
    work = {t.key: t for t in corpus.turns if t.session == WORK_SESSION}
    rows = store.execute("SELECT id, source_id, valid_to, archived_at, flags FROM facts").fetchall()
    task_live = episodes = 0
    written_events: set[str] = set()
    for row in rows:
        turn = work.get(turns.get(row["source_id"] or "") or "")
        if turn is None:
            continue
        live = row["valid_to"] is None and row["archived_at"] is None
        if EPISODE in (row["flags"] or ""):
            episodes += 1
        elif turn.intent == "task" and live:
            task_live += 1
        elif turn.event is not None:
            written_events.add(turn.event)
    left = store.execute("SELECT count(*) FROM session_facts").fetchone()[0]
    events = [t.event for t in work.values() if t.event is not None]
    return {
        "task_turns": sum(1 for t in work.values() if t.intent == "task"),
        "task_facts_live": task_live,
        "session_facts_left": int(left),
        "episodes": episodes,
        "lasting_events": len(events),
        "lasting_written": len(written_events),
    }


def oracle_consolidation(corpus: Corpus) -> dict[str, Any]:
    """What an ideal model answers when the work session closes: its episode."""
    return dict(corpus.gold.get(WORK_SESSION) or {"episode": None, "keep": []})


__all__ = ["IntentTally", "oracle_consolidation", "worksession_audit"]
