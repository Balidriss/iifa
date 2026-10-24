"""Checking, applying, journalling and undoing dream operations (SPEC 5.6).

Every operation goes through ``Applier.submit`` in one transaction under the
store lock: the revisions of the records it reads are compared with the
snapshot the model saw (rule 8: a record that moved or is gone drops it as
``stale``), then the rules of the catalogue table are checked against memory as
it is, then the operation is applied, queued for review or - in a dry run -
planned, and journalled with what it takes to revert it (rule 4).

Undo reverts exactly what an operation did, provided every record it changed
still carries the revision it left there; otherwise a later write depends on
it and the undo is refused, naming the records. The dream's own later writes
are not such a dependency: when one of its operations moves a record that an
earlier entry recorded, the earlier entry's expected revision follows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any

from iifa.dream.catalogue import RISK, named_ids, parse_operation
from iifa.dream.journal import (
    APPLIED,
    DROPPED,
    PLANNED,
    QUEUED,
    REJECTED,
    UNDONE,
    Entry,
    Journal,
    Run,
)
from iifa.dream.policy import REVIEW
from iifa.ingest.checks import is_complete, mixed_script_words, names_subject
from iifa.ingest.keys import canonical_key, multi_valued, same_key, supersedable
from iifa.ingest.summaries import LANGUAGE_MISMATCH, facts_language
from iifa.ingest.values import DIFFERENT, SAME, compare_values, same_spelling
from iifa.language import UNDETERMINED, detect, main_script, writing_mismatch
from iifa.recall.tokenizer import normalize
from iifa.scopes import Visibility, ancestors, within
from iifa.store.models import Edge, Fact, Node
from iifa.store.repo import Store, decayed_weight

DREAM_ORIGIN = "dream"
DREAM_EDGE_WEIGHT = 0.5  # a proposed edge is used at reduced weight (SPEC 5.2)
CONFLICT = "conflict"
EXTERNAL = "external"


class Drop(Exception):
    """The operation is not applied; ``reason`` is what the journal and report say."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class UndoBlocked(Exception):
    """An undo that cannot happen: what later writes depend on, a wiped entry, or
    an entry that was never applied."""

    def __init__(
        self, reason: str, *, operations: Sequence[str] = (), records: Iterable[str] = ()
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.operations = list(operations)
        self.records = sorted(set(records))


class DecisionError(Exception):
    """A review decision on an entry that is not queued."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


def dream_view(scope: str) -> Visibility:
    """What a dream on ``scope`` reads: the scope, its ancestors (context only)
    and everything below it."""
    return Visibility(exact=tuple(ancestors(scope)), subtrees=(scope,))


def changeable(scopes: Iterable[str], scope: str) -> bool:
    """Rule 12: a dream changes only records whose every scope is at or below its own."""
    values = list(scopes)
    return bool(values) and all(within(s, scope) for s in values)


def says_one_thing(facts: Sequence[Fact]) -> bool:
    """SPEC 5.6 rule 14: every fact has the same key and an equivalent value -
    the same spelling, or one ``same_as`` group (rule 13 keeps groups out of a
    consolidation today; this stays as defense in depth). A fact without a key
    never says the same thing as another."""

    def group(fact: Fact) -> str:
        return fact.same_as or fact.id

    return all(
        same_key(a.attribute, b.attribute)
        and (same_spelling(a.value or "", b.value or "") or group(a) == group(b))
        for a, b in combinations(facts, 2)
    )


Applied = tuple[dict[str, Any], list[str]]  # (changes, the records whose state undo restores)


@dataclass
class _Checked:
    op: dict[str, Any]
    reads: list[str]
    watch: list[str]
    apply: Callable[[], Applied]
    reason: str | None = None  # a downgrade says why
    records: list[str] = field(default_factory=list)


class Applier:
    def __init__(
        self,
        store: Store,
        journal: Journal,
        *,
        forced_language: Callable[[Sequence[str]], str | None] | None = None,
    ) -> None:
        self.store = store
        self.journal = journal
        self.forced_language = forced_language or (lambda scopes: None)

    # ------------------------------------------------------------------ submit

    def submit(
        self,
        run: Run,
        op: Mapping[str, Any],
        snapshot: dict[str, int],
        *,
        unit_ids: set[str] | None = None,
    ) -> Entry:
        """Check one operation the model proposed and apply, queue or plan it.
        ``snapshot`` holds the revisions the model saw; the dream's own writes
        update it, so its next operation is not made stale by them."""
        parsed = parse_operation(dict(op))
        named = named_ids(parsed)
        store = self.store
        with store.lock, store.transaction():
            name, reason = parsed["op"], None
            route: str | None = None
            changes: dict[str, Any] | None = None
            after: dict[str, int] = {}
            records = list(named)
            reads: list[str] = list(named)
            try:
                if unit_ids is not None and not set(named) <= unit_ids:
                    raise Drop("unknown_record")
                if any(i not in snapshot for i in named):
                    raise Drop("unknown_record")
                self._verify(named, snapshot)
                self._in_scope(named, run.scope)
                checked = self._check(run.scope, parsed)
                name, reason = checked.op["op"], checked.reason
                reads = list(dict.fromkeys([*named, *checked.reads]))
                self._verify([r for r in checked.reads if r in snapshot], snapshot)
                route = run.policy.route(name)
                records = list(dict.fromkeys([*named, *checked.records]))
                if run.dry_run:
                    status = PLANNED
                elif route == REVIEW:
                    status = QUEUED
                else:
                    changes, changed = self._tracked(checked.watch, checked.apply)
                    after = store.revisions(changed)
                    records = list(dict.fromkeys([*records, *changed]))
                    status = APPLIED
                    current = store.revisions(reads)
                    snapshot.update({r: current[r] for r in reads if r in current})
                payload = checked.op
            except Drop as drop:
                status, reason, payload = DROPPED, drop.reason, parsed
            return self.journal.add_op(
                run.id,
                op=name,
                risk=RISK[name],
                route=route,
                status=status,
                reason=reason,
                payload=payload,
                reason_given=str(parsed.get("reason") or "") or None,
                snapshot={r: snapshot[r] for r in reads if r in snapshot},
                changes=changes,
                after_revs=after,
                records=records,
            )

    def record_service(
        self,
        run: Run,
        name: str,
        *,
        apply: Callable[[], Applied] | None,
        watch: Sequence[str] = (),
        summary: Mapping[str, Any] | None = None,
    ) -> Entry:
        """Journal an operation the service runs itself (``decay_edges``,
        ``reembed``): planned in a dry run, else applied now."""
        store = self.store
        with store.lock, store.transaction():
            changes: dict[str, Any] | None = None
            changed: list[str] = []
            if run.dry_run or apply is None:
                status = PLANNED if run.dry_run else APPLIED
            else:
                changes, changed = self._tracked(list(watch), apply)
                status = APPLIED
            return self.journal.add_op(
                run.id,
                op=name,
                risk=RISK[name],
                route=run.policy.route(name),
                status=status,
                reason=None,
                payload=dict(summary or {}),
                reason_given=None,
                snapshot={},
                changes=changes,
                after_revs=store.revisions(changed),
                records=changed,
            )

    # ------------------------------------------------------------------ review

    def decide(self, op_id: str, *, accept: bool) -> Entry:
        """A reviewer's decision on a queued operation. Accepted, it is checked
        again against memory as it is now - its snapshot first (rule 8)."""
        store = self.store
        with store.lock, store.transaction():
            entry = self.journal.get_op(op_id)
            if entry.status != QUEUED or entry.payload is None:
                raise DecisionError(entry.status)
            run = self.journal.get_run(entry.run_id)
            now = store.now()
            if not accept:
                self._follow_proposal(entry.payload, REJECTED)
                return self.journal.update_op(op_id, status=REJECTED, decided_at=now)
            try:
                self._verify(list(entry.snapshot), entry.snapshot)
                checked = self._check(run.scope, entry.payload)
                changes, changed = self._tracked(checked.watch, checked.apply)
            except Drop as drop:
                return self.journal.update_op(
                    op_id, status=DROPPED, reason=drop.reason, decided_at=now
                )
            self.journal.add_records(op_id, changed)
            return self.journal.update_op(
                op_id,
                op=checked.op["op"],
                status=APPLIED,
                reason=checked.reason,
                decided_at=now,
                changes=changes,
                after_revs=store.revisions(changed),
            )

    def _follow_proposal(self, op: Mapping[str, Any], status: str) -> str | None:
        """A merge proposal between the two nodes follows its operation (SPEC 4.4b)."""
        if op.get("op") != "merge_nodes":
            return None
        proposal = self.store.merge_proposal_between(str(op["keep"]), str(op["absorb"]))
        if proposal is None:
            return None
        self.store.set_proposal_status(proposal.id, status)
        return proposal.id

    # ------------------------------------------------------------------ undo

    def undo_op(self, op_id: str) -> Entry:
        store = self.store
        with store.lock, store.transaction():
            entry = self.journal.get_op(op_id)
            self._undoable(entry)
            blocked = self._blocked(entry)
            if blocked:
                raise UndoBlocked("depends", operations=[entry.id], records=blocked)
            self._revert(entry)
            return self.journal.update_op(op_id, status=UNDONE, undone_at=store.now())

    def undo_run(self, run_id: str) -> int:
        """All or nothing over the run's applied operations, latest first."""
        store = self.store
        with store.lock, store.transaction():
            entries = [e for e in reversed(self.journal.ops_of_run(run_id)) if e.status == APPLIED]
            forgotten = [e.id for e in entries if e.forgotten]
            if forgotten:
                raise UndoBlocked("forgotten", operations=forgotten)
            blocked_ops: list[str] = []
            blocked_records: set[str] = set()
            for entry in entries:
                records = self._blocked(entry)
                if records:
                    blocked_ops.append(entry.id)
                    blocked_records.update(records)
            if blocked_ops:
                raise UndoBlocked("depends", operations=blocked_ops, records=blocked_records)
            for entry in entries:
                self._revert(entry)
                self.journal.update_op(entry.id, status=UNDONE, undone_at=store.now())
            self._remove_created_proposals(run_id)
            return len(entries)

    def rollback(self, run_id: str) -> list[str]:
        """Rule 9: undo every applied operation that can be undone, latest first;
        drop the queued ones. Returns the operations a later write keeps."""
        store = self.store
        incomplete: list[str] = []
        with store.lock, store.transaction():
            for entry in reversed(self.journal.ops_of_run(run_id)):
                if entry.status == QUEUED:
                    self.journal.update_op(entry.id, status=DROPPED, reason="rolled_back")
                elif entry.status == APPLIED:
                    if entry.forgotten or self._blocked(entry):
                        incomplete.append(entry.id)
                        continue
                    self._revert(entry)
                    self.journal.update_op(entry.id, status=UNDONE, undone_at=store.now())
            self._remove_created_proposals(run_id)
        return sorted(incomplete, key=lambda i: self.journal.get_op(i).seq)

    def _remove_created_proposals(self, run_id: str) -> None:
        """SPEC 5.6: the merge proposals the run's ``unsure`` verdicts left go with
        it while nobody decided them; its verdicts stay as a record."""
        for proposal_id in self.journal.proposals_created(run_id):
            self.store.remove_pending_proposal(proposal_id)

    def _undoable(self, entry: Entry) -> None:
        if entry.forgotten:
            raise UndoBlocked("forgotten", operations=[entry.id])
        if entry.status != APPLIED:
            raise UndoBlocked("not_applied", operations=[entry.id])

    def _blocked(self, entry: Entry) -> list[str]:
        current = self.store.revisions(list(entry.after_revs))
        return [r for r, rev in entry.after_revs.items() if current.get(r) != rev]

    # ------------------------------------------------------------------ plumbing

    def _verify(self, ids: Iterable[str], snapshot: Mapping[str, int]) -> None:
        wanted = [i for i in ids if i in snapshot]
        current = self.store.revisions(wanted)
        if any(current.get(i) != snapshot[i] for i in wanted):
            raise Drop("stale")

    def _in_scope(self, ids: Iterable[str], scope: str) -> None:
        for record_id in ids:
            if not changeable(self.store.scopes_of(record_id), scope):
                raise Drop("outside_scope")

    def _tracked(self, watch: Sequence[str], fn: Callable[[], Applied]) -> Applied:
        """Run ``fn`` and let earlier applied entries follow the revisions it moves
        on the records in ``watch`` (the dream's own writes are no dependency)."""
        ids = list(dict.fromkeys(watch))
        before = self.store.revisions(ids)
        result = fn()
        after = self.store.revisions(ids)
        for record_id, rev in before.items():
            moved = after.get(record_id)
            if moved is None or moved == rev:
                continue
            for entry in self.journal.applied_touching(record_id):
                if entry.after_revs.get(record_id) == rev:
                    revs = {**entry.after_revs, record_id: moved}
                    self.journal.update_op(entry.id, after_revs=revs)
        return result

    def _node(self, node_id: str) -> Node:
        node = self.store.get_node(node_id)
        if node.archived_at is not None:
            raise Drop("not_live")
        return node

    def _fact(self, fact_id: str) -> Fact:
        fact = self.store.get_fact(fact_id)
        if fact.archived_at is not None or fact.valid_to is not None:
            raise Drop("not_live")
        if fact.node_id is not None and self.store.get_node(fact.node_id).archived_at is not None:
            raise Drop("not_live")
        return fact

    # ------------------------------------------------------------------ checks

    def _check(self, scope: str, op: Mapping[str, Any]) -> _Checked:
        name = op["op"]
        checks: dict[str, Callable[[str, Mapping[str, Any]], _Checked]] = {
            "refresh_summary": self._refresh_summary,
            "propose_edge": self._propose_edge,
            "create_theme": self._create_theme,
            "consolidate_facts": self._consolidate,
            "merge_nodes": self._merge,
            "supersede_fact": self._supersede,
            "flag_conflict": self._flag,
        }
        return checks[name](scope, op)

    def _refresh_summary(self, scope: str, op: Mapping[str, Any]) -> _Checked:
        store = self.store
        node = self._node(op["node"])
        facts = store.facts([node.id], dream_view(scope))
        if not facts:
            raise Drop("no_facts")
        language = facts_language(facts, self.forced_language(node.scopes))
        summary = str(op["summary"])
        if language != UNDETERMINED and writing_mismatch(summary, language):
            raise Drop("language_mismatch")

        def apply() -> Applied:
            before = {"summary": node.summary, "language": node.language, "flags": list(node.flags)}
            written = detect(summary)
            store.update_node(
                node.id,
                summary=summary,
                language=None if written == UNDETERMINED else written,
                flags=set(node.flags) - {LANGUAGE_MISMATCH},
            )
            return {"node": node.id, "before": before}, [node.id]

        return _Checked(dict(op), [f.id for f in facts], [node.id], apply)

    def _propose_edge(self, scope: str, op: Mapping[str, Any]) -> _Checked:
        store = self.store
        if op["src"] == op["dst"]:
            raise Drop("same_record")
        src, dst = self._node(op["src"]), self._node(op["dst"])
        if set(src.scopes) != set(dst.scopes):
            raise Drop("scope_mismatch")
        kind = str(op["type"])
        if any(e.type == kind for e in store.edges_between([src.id, dst.id], dream_view(scope))):
            raise Drop("duplicate")

        def apply() -> Applied:
            edge = store.add_edge(
                scopes=src.scopes,
                src=src.id,
                dst=dst.id,
                type=kind,
                origin=DREAM_ORIGIN,
                status="proposed",
                weight=DREAM_EDGE_WEIGHT,
            )
            return {"created": [edge.id]}, [edge.id]

        return _Checked(dict(op), [], [src.id, dst.id], apply)

    def _create_theme(self, scope: str, op: Mapping[str, Any]) -> _Checked:
        store = self.store
        members_ids = list(op["members"])
        if len(set(members_ids)) != len(members_ids):
            raise Drop("same_record")
        members = [self._node(m) for m in members_ids]
        scopes = {frozenset(m.scopes) for m in members}
        if len(scopes) != 1:
            raise Drop("scope_mismatch")
        shared = members[0].scopes
        name, summary = str(op["name"]), op.get("summary")
        if mixed_script_words(name) or (summary and mixed_script_words(str(summary))):
            raise Drop("mixed_script")
        if store.topic_named(name, Visibility.of_scopes(shared)) is not None:
            raise Drop("duplicate")

        def apply() -> Applied:
            theme = store.create_node(
                scopes=shared, name=name, kind="topic", summary=str(summary or "")
            )
            created = [theme.id]
            for member in members:
                edge = store.add_edge(
                    scopes=shared,
                    src=member.id,
                    dst=theme.id,
                    type="about",
                    origin=DREAM_ORIGIN,
                    status="confirmed",
                )
                created.append(edge.id)
            return {"created": created}, created

        return _Checked(dict(op), [], members_ids, apply)

    def _consolidate(self, scope: str, op: Mapping[str, Any]) -> _Checked:  # noqa: PLR0912
        store = self.store
        fact_ids = list(op["facts"])
        if len(set(fact_ids)) != len(fact_ids):
            raise Drop("same_record")
        node = self._node(op["node"])
        facts = [store.get_fact(f) for f in fact_ids]
        for fact in facts:
            if fact.node_id != node.id:
                raise Drop("wrong_node")
        for fact in facts:
            self._fact(fact.id)
        for fact in facts:
            if fact.origin == "explicit":
                raise Drop("explicit")
            if fact.origin != "extracted":
                raise Drop("origin")
            if fact.pinned:
                raise Drop("pinned")
            if CONFLICT in fact.flags:
                raise Drop("conflict")
        if any(
            f.same_as is not None or store.equivalents([f.id], dream_view(scope))[f.id][1:]
            for f in facts
        ):
            raise Drop("same_as")
        if len({frozenset(f.scopes) for f in facts}) != 1:
            raise Drop("scope_mismatch")
        languages = {f.language or detect(f.statement) for f in facts} - {UNDETERMINED}
        if len(languages) > 1:
            raise Drop("language")
        if not says_one_thing(facts):
            raise Drop("other_value")
        target = self.forced_language(facts[0].scopes) or (
            next(iter(languages)) if languages else UNDETERMINED
        )
        statement = str(op["statement"])
        if mixed_script_words(statement):
            raise Drop("mixed_script")
        if not is_complete(statement):
            raise Drop("incomplete")
        if target != UNDETERMINED and writing_mismatch(statement, target):
            raise Drop("language_mismatch")
        if not names_subject(statement, [node.name, *node.aliases]):
            raise Drop("unnamed")

        def apply() -> Applied:
            provenances = {f.provenance for f in facts}
            dates = sorted(f.valid_from for f in facts if f.valid_from)
            written = detect(statement)
            if written == UNDETERMINED:
                written = target  # checked by script above (P12)
            new = store.add_fact(
                scopes=facts[0].scopes,
                statement=statement,
                origin="extracted",
                node_id=node.id,
                attribute=facts[0].attribute,
                value=facts[0].value,
                valid_from=dates[0] if dates else None,
                confidence=min(f.confidence for f in facts),
                language=None if written == UNDETERMINED else written,
                provenance=provenances.pop() if len(provenances) == 1 else None,
            )
            closed = [
                {
                    "id": f.id,
                    "valid_to": f.valid_to,
                    "superseded_by": f.superseded_by,
                    "consolidated_into": f.consolidated_into,
                }
                for f in facts
            ]
            store.consolidate_facts(fact_ids, into=new.id)
            return {"node": node.id, "created": [new.id], "closed": closed}, [new.id, *fact_ids]

        return _Checked(dict(op), [], [node.id, *fact_ids], apply)

    def _merge(self, scope: str, op: Mapping[str, Any]) -> _Checked:
        """Always reviewed (rule 5): a wrong merge silently corrupts memory. A
        merge that moves an explicit fact moves it unchanged (rule 3)."""
        store = self.store
        if op["keep"] == op["absorb"]:
            raise Drop("same_record")
        keep, absorb = self._node(op["keep"]), self._node(op["absorb"])
        if keep.kind != absorb.kind:
            raise Drop("kind_mismatch")
        if set(keep.scopes) != set(absorb.scopes):
            raise Drop("scope_mismatch")
        fact_ids = store.node_fact_ids(absorb.id)
        edge_ids = store.node_edge_ids(absorb.id)

        def apply() -> Applied:
            known = {normalize(keep.name), *(normalize(a) for a in keep.aliases)}
            added: list[str] = []
            for name in (absorb.name, *absorb.aliases):
                if normalize(name) not in known:
                    known.add(normalize(name))
                    added.append(name)
            for fact_id in fact_ids:
                store.attach_fact(fact_id, keep.id)
            moved: list[dict[str, str]] = []
            closed: list[dict[str, str | None]] = []
            for edge_id in edge_ids:
                edge = store.get_edge(edge_id)
                src = keep.id if edge.src == absorb.id else edge.src
                dst = keep.id if edge.dst == absorb.id else edge.dst
                if src == dst:  # it would join the kept node to itself
                    if edge.valid_to is None:
                        store.close_edge(edge_id)
                        closed.append({"id": edge_id, "valid_to": None})
                    continue
                store.move_edge(edge_id, src=src, dst=dst)
                moved.append({"id": edge_id, "src": edge.src, "dst": edge.dst})
            if added:
                store.update_node(keep.id, add_aliases=added)
            store.merge_node(absorb.id, into=keep.id)
            proposal = self._follow_proposal(op, "accepted")
            changes = {
                "keep": keep.id,
                "absorb": absorb.id,
                "absorb_archived_at": absorb.archived_at,
                "aliases_added": added,
                "moved_facts": fact_ids,
                "moved_edges": moved,
                "closed_edges": closed,
                "proposal": proposal,
            }
            changed = [
                keep.id,
                absorb.id,
                *fact_ids,
                *(m["id"] for m in moved),
                *(str(c["id"]) for c in closed),
            ]
            return changes, changed

        return _Checked(
            dict(op),
            [],
            [keep.id, absorb.id, *fact_ids, *edge_ids],
            apply,
            records=[*fact_ids, *edge_ids],
        )

    def _supersede(self, scope: str, op: Mapping[str, Any]) -> _Checked:
        store = self.store
        if op["fact"] == op["by"]:
            raise Drop("same_record")
        old, by = self._fact(op["fact"]), self._fact(op["by"])
        cause = _supersede_cause(old, by)
        if cause is not None:  # the write path would not close it: a conflict to settle
            downgraded = {
                "op": "flag_conflict",
                "fact": by.id,
                "with": old.id,
                "reason": op.get("reason", ""),
            }
            checked = self._flag(scope, downgraded)
            checked.reason = f"downgraded:{cause}"
            return checked

        def apply() -> Applied:
            closed = [
                {
                    "id": old.id,
                    "valid_to": old.valid_to,
                    "superseded_by": old.superseded_by,
                    "consolidated_into": old.consolidated_into,
                }
            ]
            store.close_fact(old.id, valid_to=by.valid_from or None, superseded_by=by.id)
            return {"closed": closed}, [old.id]

        watch = [old.id, by.id, *([old.node_id] if old.node_id else [])]
        return _Checked(dict(op), [], watch, apply)

    def _flag(self, scope: str, op: Mapping[str, Any]) -> _Checked:
        store = self.store
        if op["fact"] == op["with"]:
            raise Drop("same_record")
        fact, other = self._fact(op["fact"]), self._fact(op["with"])
        if CONFLICT in fact.flags and fact.conflicts_with == other.id:
            raise Drop("duplicate")

        def apply() -> Applied:
            before = {
                "fact": fact.id,
                "flags": list(fact.flags),
                "conflicts_with": fact.conflicts_with,
            }
            store.set_fact_flags(fact.id, {*fact.flags, CONFLICT}, conflicts_with=other.id)
            return before, [fact.id]

        return _Checked(dict(op), [], [fact.id, other.id], apply)

    # ------------------------------------------------------------------ service operations

    def decay_plan(
        self, scope: str, *, half_life_days: float, prune_below: float, now: datetime
    ) -> list[tuple[Edge, float, bool]]:
        """SPEC 5.3 over the edges the dream may change: ``(edge, new weight,
        pruned)`` for each one decay would change."""
        out: list[tuple[Edge, float, bool]] = []
        for edge in self.store.live_proposed_edges(Visibility.of_subtree(scope)):
            if not changeable(edge.scopes, scope):
                continue
            decayed_at = self.store.edge_decayed_at(edge.id)
            anchor = max(s for s in (edge.last_used_at, decayed_at, edge.created_at) if s)
            weight = decayed_weight(edge.weight, anchor, now, half_life_days)
            if weight < prune_below:
                out.append((edge, edge.weight, True))
            elif weight != edge.weight:
                out.append((edge, weight, False))
        return out

    def apply_decay(self, plan: Sequence[tuple[Edge, float, bool]]) -> Applied:
        store = self.store
        now = store.now()
        before: list[dict[str, Any]] = []
        for edge, weight, pruned in plan:
            before.append(
                {
                    "id": edge.id,
                    "weight": edge.weight,
                    "decayed_at": store.edge_decayed_at(edge.id),
                    "valid_to": edge.valid_to,
                }
            )
            store.set_edge_life(
                edge.id, weight=weight, decayed_at=now, valid_to=now if pruned else None
            )
        return {"edges": before}, [e.id for e, _, _ in plan]

    # ------------------------------------------------------------------ revert

    def _revert(self, entry: Entry) -> None:
        changes = entry.changes or {}
        watch = [*entry.after_revs, *entry.records]

        def revert() -> Applied:
            self._revert_changes(entry.op, changes)
            return {}, []

        self._tracked(watch, revert)

    def _revert_changes(self, name: str, changes: Mapping[str, Any]) -> None:  # noqa: PLR0912
        store = self.store
        if name == "refresh_summary":
            before = changes["before"]
            store.set_summary(
                changes["node"],
                summary=before["summary"],
                language=before["language"],
                flags=before["flags"],
            )
        elif name in ("propose_edge", "create_theme"):
            for record_id in reversed(changes.get("created", [])):
                if store.revisions([record_id]):
                    store.remove_created(record_id)
        elif name == "consolidate_facts":
            for closed in changes["closed"]:
                store.reopen_fact(
                    closed["id"],
                    valid_to=closed["valid_to"],
                    superseded_by=closed["superseded_by"],
                    consolidated_into=closed["consolidated_into"],
                )
            for record_id in changes.get("created", []):
                if store.revisions([record_id]):
                    store.remove_created(record_id)
        elif name == "merge_nodes":
            for closed in changes["closed_edges"]:
                edge = store.get_edge(closed["id"])
                store.set_edge_life(
                    edge.id,
                    weight=edge.weight,
                    decayed_at=store.edge_decayed_at(edge.id),
                    valid_to=closed["valid_to"],
                )
            for moved in changes["moved_edges"]:
                store.move_edge(moved["id"], src=moved["src"], dst=moved["dst"])
            for fact_id in changes["moved_facts"]:
                store.attach_fact(fact_id, changes["absorb"])
            if changes["aliases_added"]:
                store.remove_aliases(changes["keep"], changes["aliases_added"])
            store.unmerge_node(changes["absorb"], archived_at=changes["absorb_archived_at"])
            if changes.get("proposal"):
                store.set_proposal_status(changes["proposal"], "pending")
        elif name == "supersede_fact":
            for closed in changes["closed"]:
                store.reopen_fact(
                    closed["id"],
                    valid_to=closed["valid_to"],
                    superseded_by=closed["superseded_by"],
                    consolidated_into=closed["consolidated_into"],
                )
        elif name == "flag_conflict":
            store.set_fact_flags(
                changes["fact"], changes["flags"], conflicts_with=changes["conflicts_with"]
            )
        elif name == "decay_edges":
            for edge in changes["edges"]:
                store.set_edge_life(
                    edge["id"],
                    weight=edge["weight"],
                    decayed_at=edge["decayed_at"],
                    valid_to=edge["valid_to"],
                )
        # reembed only queued derived work: nothing to revert


def _supersede_cause(old: Fact, by: Fact) -> str | None:  # noqa: PLR0911, PLR0912
    """Why the write path (SPEC 5.4) would not let ``by`` close ``old``, or None:
    one return per rule, in the order the write path applies them."""
    if old.node_id is None or old.node_id != by.node_id:
        return "node"
    if old.origin == "explicit":
        return "explicit"
    if old.origin != "extracted":
        return "origin"
    if old.pinned:
        return "pinned"
    if set(old.scopes) != set(by.scopes):
        return "scope"
    key = canonical_key(old.attribute)
    if key is None or not same_key(old.attribute, by.attribute):
        return "key"
    if multi_valued(key):
        return "several"
    if not supersedable(key):
        return "key"
    if old.valid_from and by.valid_from:
        if by.valid_from < old.valid_from:
            return "earlier"
    elif by.created_at < old.created_at:
        return "earlier"
    if by.provenance == EXTERNAL and old.provenance != EXTERNAL:
        return "external"
    verdict = compare_values(old.value or "", by.value or "")
    if verdict == SAME:
        return "same_value"
    if verdict != DIFFERENT and not _one_language(old, by):
        return "unproven"  # across languages, only a value proven different closes
    return None


def _one_language(a: Fact, b: Fact) -> bool:
    """SPEC 5.4: both determined and equal, or, either undetermined, one script."""
    left, right = a.language or detect(a.statement), b.language or detect(b.statement)
    if UNDETERMINED not in (left, right):
        return left == right
    script = main_script(a.statement)
    return script is not None and script == main_script(b.statement)


__all__ = [
    "Applier",
    "DecisionError",
    "Drop",
    "UndoBlocked",
    "changeable",
    "dream_view",
]
