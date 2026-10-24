"""Merge candidates and their adjudication (SPEC 5.6).

A wrong merge silently corrupts memory, so a merge is never decided on a name:
not by the write path, not by a cosine, not by the unit's model reading one
node. Before a unit's call, each candidate of its node - the nodes a pending
merge proposal pairs it with, then the nodes whose name vector reaches
``resolve`` - is adjudicated: one call reads both nodes whole and answers a
closed verdict. ``same`` submits a ``merge_nodes`` (always reviewed), ``different``
changes nothing, ``unsure`` leaves a visible proposal. A ``same`` whose names do
not fit is downgraded to ``unsure`` by the service. A real run records each
verdict for the reviewer; a dry run only counts them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from iifa.dream.apply import Applier, changeable, dream_view
from iifa.dream.bounds import OVERSIZED, by_priority, largest, prompt_tokens, shown_in_order
from iifa.dream.catalogue import BAD_ANSWER, SAME, UNSURE, VERDICTS, parse_verdict
from iifa.dream.journal import APPLIED, PLANNED, QUEUED, REJECTED, Journal, Run, pair_key
from iifa.dream.prompts import adjudication_messages
from iifa.dream.units import Fitted
from iifa.ingest.embed import neighbours
from iifa.ingest.values import same_spelling
from iifa.llm import TRUNCATED, ChatResult, ModelError
from iifa.recall.tokenizer import normalize
from iifa.store.models import Fact, Node
from iifa.store.repo import MERGE, NAME, NotFound, Store
from iifa.tokens import Tokenizer

TRUNCATED_ANSWER = "truncated"  # as the unit's answers are counted
MAX_CHAIN = 6  # values shown per timeline: the live one and those it replaced, latest kept
# why a candidate costs no call, counted in the report
SKIPS: tuple[str, ...] = ("kind", "scope", "unchanged", "rejected", "queued")

NameNeighbours = Callable[[str], Sequence[tuple[str, float]]]
NamesClose = Callable[[str, str], bool]  # name cosine at least the calibrated merge


class Chat(Protocol):
    def chat(self, messages: Sequence[Mapping[str, Any]], **params: Any) -> ChatResult: ...


def name_neighbours(store: Store, model: str, resolve: float) -> NameNeighbours:
    """The candidates name vectors give (SPEC 5.6): the nodes the node's scopes
    see whose name vector of ``model`` is at least ``resolve`` from its own."""

    def find(node_id: str) -> list[tuple[str, float]]:
        return neighbours(store, node_id, NAME, model, resolve)

    return find


def names_close(store: Store, model: str, merge: float) -> NamesClose:
    """Do two nodes' name vectors of ``model`` reach the calibrated ``merge``
    threshold (SPEC 6.1)? No threshold of its own."""

    def close(a_id: str, b_id: str) -> bool:
        return any(other == b_id for other, _ in neighbours(store, a_id, NAME, model, merge))

    return close


def names_fit(a: Node, b: Node, close: NamesClose | None) -> bool:
    """SPEC 5.6: ``same`` only when the names fit - one spelling in two scripts
    (the strict comparison, never Daniela for Daniel), a shared name or alias,
    or a name cosine at the calibrated merge threshold."""
    forms_a, forms_b = (a.name, *a.aliases), (b.name, *b.aliases)
    shared = {normalize(f) for f in forms_a} & {normalize(f) for f in forms_b}
    if shared - {""}:
        return True
    if any(same_spelling(x, y) for x in forms_a for y in forms_b):
        return True
    return close is not None and close(a.id, b.id)


def empty_report() -> dict[str, Any]:
    return {
        "calls": 0,
        "verdicts": dict.fromkeys(VERDICTS, 0),
        "answers": {BAD_ANSWER: 0, TRUNCATED_ANSWER: 0},
        "skipped": dict.fromkeys(SKIPS, 0),
        "proposals_created": 0,
        "names_unfit": 0,  # `same` downgraded to `unsure`: the names do not fit
        "pairs": [],
    }


@dataclass(frozen=True)
class Pair:
    """Two nodes and their live facts, read in one go with their revisions."""

    a: Node
    b: Node
    facts_a: tuple[Fact, ...]
    facts_b: tuple[Fact, ...]
    snapshot: dict[str, int]
    history: tuple[Fact, ...] = ()  # closed facts a later fact of either node replaced

    @property
    def fingerprint(self) -> str:
        """What the model read: it is not asked again until one of these moves."""
        text = ";".join(f"{k}:{v}" for k, v in sorted(self.snapshot.items()))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def keep_absorb(self) -> tuple[str, str]:
        """The node with more live facts is kept, else the older one."""
        if len(self.facts_b) > len(self.facts_a):
            return self.b.id, self.a.id
        return self.a.id, self.b.id

    def as_prompt(
        self, facts_a: Sequence[Fact] | None = None, facts_b: Sequence[Fact] | None = None
    ) -> dict[str, Any]:
        """What the model reads: names, summaries and, per key, a timeline of
        values - no id, no scope, no source; with ``facts_a`` / ``facts_b``, only
        those live facts (and what they replaced), and how many were left out.

        A change over time is not a contradiction, and a model told so still read
        "architect vs accountant" as one: so every value says when memory learned
        it, in one order shared by A and B, with its date when known."""
        learned = {
            f.id: i
            for i, f in enumerate(
                sorted((*self.facts_a, *self.facts_b, *self.history), key=_learned_key), 1
            )
        }
        replaced: dict[str, list[Fact]] = {}
        for fact in self.history:
            replaced.setdefault(str(fact.superseded_by), []).append(fact)
        shown_a = self.facts_a if facts_a is None else facts_a
        shown_b = self.facts_b if facts_b is None else facts_b
        return {
            "a": _side(self.a, shown_a, len(self.facts_a), learned, replaced),
            "b": _side(self.b, shown_b, len(self.facts_b), learned, replaced),
        }

    def fitted(self, tokenizer: Tokenizer, budget: int) -> Fitted | None:
        """The adjudication within ``budget`` tokens: the two sides take facts in
        turn, each by priority (SPEC 5.6); None when the two nodes alone are over."""
        ranked_a, ranked_b = by_priority(self.facts_a), by_priority(self.facts_b)

        def split(k: int) -> tuple[int, int]:
            a = min(len(ranked_a), (k + 1) // 2)
            b = min(len(ranked_b), k - a)
            return min(len(ranked_a), k - b), b

        def render(k: int) -> list[dict[str, str]]:
            a, b = split(k)
            return adjudication_messages(
                self.as_prompt(
                    shown_in_order(ranked_a[:a], self.facts_a),
                    shown_in_order(ranked_b[:b], self.facts_b),
                )
            )

        total = len(ranked_a) + len(ranked_b)
        k = largest(total, lambda n: prompt_tokens(render(n), tokenizer) <= budget)
        if k < 0:
            return None
        messages = render(k)
        return Fitted(
            messages=messages,
            ids=set(self.snapshot),
            tokens=prompt_tokens(messages, tokenizer),
            reduced=k < total,
        )


def _learned_key(fact: Fact) -> tuple[str, str]:
    return (fact.created_at, fact.id)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _chain(fact: Fact, learned: Mapping[str, int], replaced: Mapping[str, list[Fact]]) -> str:
    """The live fact and the values it replaced, oldest first: "Paris (learned
    4th) -> Lyon (learned 6th, from 2026-01)"."""
    chain, seen = [fact], {fact.id}
    while len(chain) < MAX_CHAIN:
        older = [f for f in replaced.get(chain[0].id, ()) if f.id not in seen]
        if not older:
            break
        previous = max(older, key=lambda f: learned[f.id])
        chain.insert(0, previous)
        seen.add(previous.id)

    def label(f: Fact) -> str:
        text = f.value if f.attribute and f.value else f.statement
        when = f", from {f.valid_from}" if f.valid_from else ""
        return f"{text} (learned {_ordinal(learned[f.id])}{when})"

    return " -> ".join(label(f) for f in chain)


def _side(
    node: Node,
    facts: Sequence[Fact],
    live: int,
    learned: Mapping[str, int],
    replaced: Mapping[str, list[Fact]],
) -> dict[str, Any]:
    keyed: dict[str, list[str]] = {}
    other: list[str] = []
    for fact in sorted(facts, key=lambda f: learned[f.id]):
        chain = _chain(fact, learned, replaced)
        if fact.attribute:
            keyed.setdefault(fact.attribute, []).append(chain)
        else:
            other.append(chain)
    out: dict[str, Any] = {
        "name": node.name,
        "kind": node.kind,
        "aliases": list(node.aliases),
        "summary": node.summary,
        "timelines": {key: " | ".join(keyed[key]) for key in sorted(keyed)},
    }
    if other:
        out["other"] = other
    if len(facts) < live:
        out["facts_left_out"] = live - len(facts)
    return out


def read_pair(store: Store, scope: str, a_id: str, b_id: str) -> Pair:
    first, second = pair_key(a_id, b_id)
    view = dream_view(scope)
    with store.lock:
        a, b = store.get_node(first), store.get_node(second)
        every = store.facts([a.id, b.id], view, live_only=False)
        facts = [f for f in every if f.valid_to is None]
        history = tuple(
            f
            for f in every
            if f.valid_to is not None and f.superseded_by is not None and not f.consolidated_into
        )
        facts_a = tuple(f for f in facts if f.node_id == a.id)
        facts_b = tuple(f for f in facts if f.node_id == b.id)
        snapshot = store.revisions([a.id, b.id, *(f.id for f in (*facts, *history))])
    return Pair(a, b, facts_a, facts_b, snapshot, history)


class Adjudicator:
    def __init__(
        self,
        store: Store,
        journal: Journal,
        applier: Applier,
        *,
        per_unit: int,
        answer_max_tokens: int,
        json_mode: bool,
        tokenizer: Tokenizer,
        prompt_max_tokens: int,
        names_close: NamesClose | None = None,
    ) -> None:
        self.store = store
        self.journal = journal
        self.applier = applier
        self.per_unit = per_unit
        self.answer_max_tokens = answer_max_tokens
        self.json_mode = json_mode
        self.tokenizer = tokenizer
        self.prompt_max_tokens = prompt_max_tokens
        # tier 3 with a calibrated model: the name cosine reaches ``merge``; else None
        self.names_close = names_close

    def candidates(self, scope: str, node: Node, neighbours: NameNeighbours | None) -> list[str]:
        """Proposals first, then name vectors best first; once each."""
        out = list(self.store.merge_candidates(node.id, dream_view(scope)))
        if neighbours is not None:
            ranked = sorted(neighbours(node.id), key=lambda pair: -pair[1])
            out.extend(other for other, _score in ranked)
        return [o for o in dict.fromkeys(out) if o != node.id]

    def _skip(self, scope: str, node: Node, other_id: str) -> str | None:  # noqa: PLR0911
        try:
            other = self.store.get_node(other_id)
        except NotFound:
            return "scope"
        if other.archived_at is not None or set(other.scopes) != set(node.scopes):
            return "scope"
        if not changeable(other.scopes, scope):
            return "scope"
        if other.kind != node.kind:
            return "kind"
        if self.store.merge_rejected(node.id, other_id) or self.journal.merge_ops(
            node.id, other_id, status=REJECTED
        ):
            return "rejected"
        if self.journal.merge_ops(node.id, other_id, status=QUEUED):
            return "queued"
        return None

    def unit(
        self,
        run: Run,
        node: Node,
        chat: Chat,
        state: dict[str, Any],
        neighbours: NameNeighbours | None,
    ) -> int:
        """Adjudicate the node's candidates; returns how many operations it
        submitted that count against the run's budget. ``state`` keeps what this
        run already judged, so a run stopped among them resumes without asking
        twice. Budget and model errors propagate: the caller stops the run there."""
        report: dict[str, Any] = state["adjudications"]
        done = {tuple(p) for p in state["adjudicated"]}
        asked = counted = 0
        for other_id in self.candidates(run.scope, node, neighbours):
            key = pair_key(node.id, other_id)
            if key in done:
                continue
            if asked >= self.per_unit:
                break
            skip = self._skip(run.scope, node, other_id)
            if skip is None:
                pair = read_pair(self.store, run.scope, node.id, other_id)
                latest = None if run.dry_run else self.journal.latest_adjudication(*key)
                if latest is not None and latest.fingerprint == pair.fingerprint:
                    skip = "unchanged"
            if skip is not None:
                report["skipped"][skip] += 1
                done.add(key)
                state["adjudicated"] = sorted(done)
                continue
            fitted = pair.fitted(self.tokenizer, self.prompt_max_tokens)
            prompts = state["prompts"]
            if fitted is None:
                prompts[OVERSIZED] += 1  # never cut from the start by the server
            else:
                asked += 1
                prompts["pairs_reduced"] += int(fitted.reduced)
                prompts["largest_prompt_tokens"] = max(
                    prompts["largest_prompt_tokens"], fitted.tokens
                )
                counted += self._ask(run, node, pair, fitted, chat, report)
            done.add(key)
            state["adjudicated"] = sorted(done)
        return counted

    def _ask(
        self,
        run: Run,
        node: Node,
        pair: Pair,
        fitted: Fitted,
        chat: Chat,
        report: dict[str, Any],
    ) -> int:
        try:
            result = chat.chat(
                fitted.messages,
                max_tokens=self.answer_max_tokens,
                json_object=self.json_mode,
            )
        except ModelError as exc:
            if exc.reason != TRUNCATED:
                raise
            report["calls"] += 1
            report["answers"][TRUNCATED_ANSWER] += 1
            return 0
        report["calls"] += 1
        verdict = parse_verdict(result.content)
        if verdict.verdict is None:
            report["answers"][verdict.error or BAD_ANSWER] += 1
            return 0
        decided = verdict.verdict
        if decided == SAME and not names_fit(pair.a, pair.b, self.names_close):
            decided = UNSURE  # a downgrade of label, never a merge decision
            report["names_unfit"] += 1
        report["verdicts"][decided] += 1
        said = {"a": pair.a.id, "b": pair.b.id, "verdict": decided}
        if decided != verdict.verdict:
            said["model_verdict"] = verdict.verdict
        report["pairs"].append(said)
        op_id: str | None = None
        proposal_id: str | None = None
        counted = 0
        if decided == SAME:
            keep, absorb = pair.keep_absorb
            entry = self.applier.submit(
                run,
                {"op": "merge_nodes", "keep": keep, "absorb": absorb, "reason": verdict.reason},
                dict(pair.snapshot),
            )
            op_id = entry.id
            counted = int(entry.status in (APPLIED, QUEUED, PLANNED))
        elif (
            decided == UNSURE
            and not run.dry_run
            and not self.store.merge_proposed(pair.a.id, pair.b.id)
        ):
            other = pair.b.id if node.id == pair.a.id else pair.a.id
            proposal_id = self.store.add_proposal(
                scopes=node.scopes, kind=MERGE, node_id=node.id, value=other
            ).id
            report["proposals_created"] += 1
        if not run.dry_run:
            self.journal.add_adjudication(
                run.id,
                a=pair.a.id,
                b=pair.b.id,
                verdict=decided,
                reason=verdict.reason,
                fingerprint=pair.fingerprint,
                records=pair.snapshot,
                proposal_id=proposal_id,
                op_id=op_id,
            )
        return counted


__all__ = [
    "SKIPS",
    "Adjudicator",
    "NameNeighbours",
    "NamesClose",
    "Pair",
    "empty_report",
    "name_neighbours",
    "names_close",
    "names_fit",
    "read_pair",
]
