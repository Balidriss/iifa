"""A person's corrections of a node and decisions on a proposed edge (SPEC 5.7),
and the mechanics of a merge that the dream's ``merge_nodes`` shares (SPEC 5.6).

A correction - a rename, aliases added or removed, a kind changed, two nodes
merged, a proposed edge confirmed or rejected - is applied in one transaction
under the store lock and journalled with the name of the token that made it,
what was asked, how to revert it and the revisions right after. Undo reverts
exactly, provided every record it changed still carries the revision it left
there; otherwise a later write depends on it and the undo is refused, naming
the records (the rule of a dream's undo, SPEC 5.6 rule 4). Entries hold content, so ``Store.forget``
wipes the ones that mention a forgotten record (P5).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from iifa.ids import new_id
from iifa.names import InvalidName
from iifa.recall.tokenizer import normalize
from iifa.scopes import Visibility
from iifa.store.models import Edge, Node, Proposal
from iifa.store.repo import (
    ACCEPTED,
    MERGE,
    PENDING,
    PROPOSED,
    NotFound,
    Store,
    checked_name,
)

UPDATE_NODE, MERGE_NODES = "update_node", "merge_nodes"
CONFIRM_EDGE, REJECT_EDGE = "confirm_edge", "reject_edge"
EDGE_DECISIONS = {"confirm": CONFIRM_EDGE, "reject": REJECT_EDGE}
APPLIED, UNDONE = "applied", "undone"
KIND_MAX_CHARS = 40


def _kept_name(text: str) -> str | None:
    """``text`` as the store keeps it, or None when the name rule refuses it."""
    try:
        return checked_name(text)
    except InvalidName:
        return None


class CorrectionRefused(Exception):
    """A correction or an undo that does not happen. ``reason`` says why
    (``stale``, ``not_live``, ``same_record``, ``kind_mismatch``,
    ``scope_mismatch``, ``no_alias``, ``no_kind``, ``not_proposed``,
    ``depends``, ``forgotten``, ``not_applied``); ``records`` names what is in the way, ``revs`` the
    current revisions when the author's were stale."""

    def __init__(
        self,
        reason: str,
        *,
        records: Iterable[str] = (),
        revs: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.records = sorted(set(records))
        self.revs = dict(revs or {})


@dataclass(frozen=True)
class Correction:
    id: str
    op: str
    status: str
    token: str
    payload: dict[str, Any] | None
    changes: dict[str, Any] | None
    before_revs: dict[str, int]
    after_revs: dict[str, int]
    records: tuple[str, ...]
    forgotten: bool
    created_at: str
    undone_at: str | None
    undone_by: str | None


# --------------------------------------------------------------------------- merge mechanics


def merge_refusal(keep: Node, absorb: Node, *, by_person: bool = False) -> str | None:
    """The one merge guard (SPEC 5.6, 5.7), for the dream and for a person: two
    distinct live nodes with identical scope sets, whose names and aliases all
    pass the name rule (they join the kept node's), and of the same kind for
    the dream. A person's merge (``by_person``) may join two kinds: the kept
    node keeps its own. Returns the reason a merge is refused, or None."""
    if keep.id == absorb.id:
        return "same_record"
    if keep.archived_at is not None or absorb.archived_at is not None:
        return "not_live"
    if keep.kind != absorb.kind and not by_person:
        return "kind_mismatch"
    if set(keep.scopes) != set(absorb.scopes):
        return "scope_mismatch"
    try:
        for name in (absorb.name, *absorb.aliases):
            checked_name(name)
    except InvalidName:
        return "invalid_name"  # a name stored before the rule: correct it first
    return None


def merge_nodes(
    store: Store, keep: Node, absorb: Node, fact_ids: Sequence[str], edge_ids: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """Merge ``absorb`` into ``keep`` (SPEC 5.6, 5.7): its name and aliases join
    ``keep``'s, its facts (unchanged) and edges move to ``keep``, an edge left
    joining ``keep`` to itself is closed, ``absorb`` is archived with
    ``merged_into``. Returns how to revert it and the records it changed. The
    caller holds the store lock and a transaction, and follows the proposal."""
    known = {normalize(keep.name), *(normalize(a) for a in keep.aliases)}
    added: list[str] = []
    for name in (absorb.name, *absorb.aliases):
        plain = checked_name(name)  # what the store keeps (SPEC 5.1), so undo finds it
        if normalize(plain) not in known:
            known.add(normalize(plain))
            added.append(plain)
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
    changes = {
        "keep": keep.id,
        "absorb": absorb.id,
        "absorb_archived_at": absorb.archived_at,
        "aliases_added": added,
        "moved_facts": list(fact_ids),
        "moved_edges": moved,
        "closed_edges": closed,
    }
    changed = [
        keep.id,
        absorb.id,
        *fact_ids,
        *(m["id"] for m in moved),
        *(str(c["id"]) for c in closed),
    ]
    return changes, changed


def unmerge_nodes(store: Store, changes: Mapping[str, Any]) -> None:
    """Revert ``merge_nodes`` exactly from what it returned (the proposal aside)."""
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


def default_keep(store: Store, a: Node, b: Node) -> tuple[Node, Node]:
    """``(keep, absorb)`` when the person did not say: the node with more live
    facts is kept, else the older (SPEC 5.7, as the dream decides, SPEC 5.6)."""
    counts = store.fact_counts([a.id, b.id], Visibility.of_scopes(a.scopes))
    first, second = sorted((a, b), key=lambda n: (n.created_at, n.id))
    if counts.get(second.id, 0) > counts.get(first.id, 0):
        return second, first
    return first, second


# --------------------------------------------------------------------------- the journal


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class Corrections:
    """Corrections through the store's connection and lock (one writer)."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def get(self, correction_id: str) -> Correction:
        store = self.store
        with store.lock:
            row = store.execute(
                "SELECT * FROM corrections WHERE id = ?", (correction_id,)
            ).fetchone()
            if row is None:
                raise NotFound(correction_id)
            records = tuple(
                r["record_id"]
                for r in store.execute(
                    "SELECT record_id FROM correction_records WHERE correction_id = ?"
                    " ORDER BY record_id",
                    (correction_id,),
                )
            )
        return Correction(
            id=row["id"],
            op=row["op"],
            status=row["status"],
            token=row["token"],
            payload=_loads(row["payload"]),
            changes=_loads(row["changes"]),
            before_revs=_loads(row["before_revs"]) or {},
            after_revs=_loads(row["after_revs"]) or {},
            records=records,
            forgotten=bool(row["forgotten"]),
            created_at=row["created_at"],
            undone_at=row["undone_at"],
            undone_by=row["undone_by"],
        )

    def _add(
        self,
        *,
        op: str,
        token: str,
        payload: Mapping[str, Any],
        changes: Mapping[str, Any],
        before: Mapping[str, int],
        changed: Sequence[str],
        records: Iterable[str],
    ) -> Correction:
        store = self.store
        correction_id = new_id("corr")
        store.execute(
            "INSERT INTO corrections (id, op, status, token, payload, changes, before_revs,"
            " after_revs, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                correction_id,
                op,
                APPLIED,
                token,
                _dumps(dict(payload)),
                _dumps(dict(changes)),
                _dumps({r: before[r] for r in dict.fromkeys(changed) if r in before}),
                _dumps(store.revisions(list(changed))),
                store.now(),
            ),
        )
        for record_id in dict.fromkeys(records):
            store.execute(
                "INSERT OR IGNORE INTO correction_records (correction_id, record_id) VALUES (?, ?)",
                (correction_id, record_id),
            )
        return self.get(correction_id)

    # ------------------------------------------------------------------ a node's names and kind

    def patch_node(
        self,
        node_id: str,
        *,
        token: str,
        name: str | None = None,
        keep_old_name: bool = True,
        kind: str | None = None,
        add_aliases: Sequence[str] = (),
        remove_aliases: Sequence[str] = (),
        rev: int | None = None,
    ) -> tuple[Node, Correction | None]:
        """Rename a node, change its kind, add and remove aliases (SPEC 5.7). A
        rename keeps the old name as an alias unless ``keep_old_name`` is false
        or the name rule refuses it; an alias spelled like the name is dropped.
        Nothing changed: no entry."""
        store = self.store
        with store.lock, store.transaction():
            node = store.get_node(node_id)
            if rev is not None and node.rev != rev:
                raise CorrectionRefused("stale", records=[node.id], revs={node.id: node.rev})
            if node.archived_at is not None:  # restore it first; a merged one, undo the merge
                raise CorrectionRefused("not_live", records=[node.id])
            new_name = None if name is None else checked_name(name)
            if new_name == node.name:
                new_name = None
            new_kind = None if kind is None else " ".join(kind.split())
            if new_kind is not None and not 0 < len(new_kind) <= KIND_MAX_CHARS:
                raise CorrectionRefused("no_kind")
            if new_kind == node.kind:
                new_kind = None
            current = {normalize(a): a for a in node.aliases}
            missing = [a for a in remove_aliases if normalize(a) not in current]
            if missing:
                raise CorrectionRefused("no_alias", records=missing)
            wanted = dict(current)
            for alias in remove_aliases:
                wanted.pop(normalize(alias), None)
            old = _kept_name(node.name) if new_name is not None and keep_old_name else None
            if old is not None:
                wanted.setdefault(normalize(old), old)
            for alias in add_aliases:
                clean = checked_name(alias)
                wanted.setdefault(normalize(clean), clean)
            name_norm = normalize(new_name or node.name)
            if new_name is not None or name_norm not in current:
                wanted.pop(name_norm, None)
            removed = [text for norm, text in current.items() if norm not in wanted]
            added = [text for norm, text in wanted.items() if norm not in current]
            if new_name is None and new_kind is None and not removed and not added:
                return node, None
            before = store.revisions([node.id])
            if removed:
                store.remove_aliases(node.id, removed)
            store.update_node(node.id, name=new_name, kind=new_kind, add_aliases=added)
            changes = {
                "node": node.id,
                "before": {"name": node.name, "kind": node.kind, "aliases": list(node.aliases)},
                "aliases_added": added,
                "aliases_removed": removed,
            }
            payload = {
                "name": name,
                "keep_old_name": keep_old_name,
                "kind": kind,
                "add_aliases": list(add_aliases),
                "remove_aliases": list(remove_aliases),
            }
            entry = self._add(
                op=UPDATE_NODE,
                token=token,
                payload=payload,
                changes=changes,
                before=before,
                changed=[node.id],
                records=[node.id],
            )
            return store.get_node(node.id), entry

    # ------------------------------------------------------------------ a person's merge

    def merge(
        self,
        keep_id: str,
        absorb_id: str,
        *,
        token: str,
        proposal: Proposal | None = None,
        revs: Mapping[str, int] | None = None,
    ) -> Correction:
        """Merge two nodes a person knows are one (SPEC 5.7) through their merge
        proposal - ``proposal``, else the pending one between them, else one
        recorded now - which is accepted. The checks are the dream's - two
        distinct live nodes with identical scope sets - except the kind: a
        person may merge two kinds, and the kept node keeps its own."""
        store = self.store
        if keep_id == absorb_id:
            raise CorrectionRefused("same_record", records=[keep_id])
        with store.lock, store.transaction():
            keep, absorb = store.get_node(keep_id), store.get_node(absorb_id)
            if revs:
                current = store.revisions([keep.id, absorb.id])
                if any(current.get(i) != r for i, r in revs.items()):
                    raise CorrectionRefused(
                        "stale",
                        records=[i for i in revs if current.get(i) != revs[i]],
                        revs=current,
                    )
            refusal = merge_refusal(keep, absorb, by_person=True)
            if refusal is not None:
                raise CorrectionRefused(refusal, records=[keep.id, absorb.id])
            created = False
            if proposal is None:
                proposal = store.merge_proposal_between(keep.id, absorb.id)
            if proposal is None:
                proposal = store.add_proposal(
                    scopes=keep.scopes, kind=MERGE, node_id=keep.id, value=absorb.id
                )
                created = True
            fact_ids = store.node_fact_ids(absorb.id)
            edge_ids = store.node_edge_ids(absorb.id)
            before = store.revisions([keep.id, absorb.id, *fact_ids, *edge_ids])
            changes, changed = merge_nodes(store, keep, absorb, fact_ids, edge_ids)
            store.set_proposal_status(proposal.id, ACCEPTED)
            changes = {**changes, "proposal": proposal.id, "proposal_created": created}
            return self._add(
                op=MERGE_NODES,
                token=token,
                payload={"keep": keep.id, "absorb": absorb.id, "proposal": proposal.id},
                changes=changes,
                before=before,
                changed=changed,
                records=[keep.id, absorb.id, *fact_ids, *edge_ids, proposal.id],
            )

    # ------------------------------------------------------------------ a proposed edge

    def decide_edge(
        self, edge_id: str, decision: str, *, token: str, rev: int | None = None
    ) -> tuple[Edge, Correction]:
        """A person's decision on a live proposed edge (SPEC 5.7): ``confirm``
        makes it confirmed, ``reject`` archives it (reversible, P5)."""
        op = EDGE_DECISIONS.get(decision)
        if op is None:
            raise ValueError(f"unknown edge decision {decision!r}")
        store = self.store
        with store.lock, store.transaction():
            edge = store.get_edge(edge_id)
            if rev is not None and edge.decision_rev != rev:  # the API's rev of an edge
                raise CorrectionRefused(
                    "stale", records=[edge.id], revs={edge.id: edge.decision_rev}
                )
            if edge.archived_at is not None or edge.valid_to is not None:
                raise CorrectionRefused("not_live", records=[edge.id])
            if edge.status != PROPOSED:
                raise CorrectionRefused("not_proposed", records=[edge.id])
            before = store.revisions([edge.id])
            if op == CONFIRM_EDGE:
                store.confirm_edge(edge.id)
            else:
                store.archive(edge.id)
            # an undo compares the decision revision: a reinforcement or a decay
            # since does not block it (SPEC 5.7); after_revs keeps the full rev
            decided = store.get_edge(edge.id).decision_rev
            entry = self._add(
                op=op,
                token=token,
                payload={"edge": edge.id, "decision": decision},
                changes={"edge": edge.id, "status": edge.status, "decision_rev": decided},
                before=before,
                changed=[edge.id],
                records=[edge.id],
            )
            return store.get_edge(edge.id), entry

    # ------------------------------------------------------------------ undo

    def undo(self, correction_id: str, *, token: str) -> Correction:
        """Revert one correction exactly, or refuse naming what depends on it."""
        store = self.store
        with store.lock, store.transaction():
            entry = self.get(correction_id)
            if entry.forgotten:
                raise CorrectionRefused("forgotten")
            if entry.status != APPLIED or entry.changes is None:
                raise CorrectionRefused("not_applied")
            blocked = self._blocking(entry)
            if blocked:
                raise CorrectionRefused("depends", records=blocked)
            changes = entry.changes
            if entry.op == UPDATE_NODE:
                before = changes["before"]
                store.restore_node_names(
                    changes["node"],
                    name=before["name"],
                    kind=before["kind"],
                    aliases=before["aliases"],
                )
            elif entry.op == CONFIRM_EDGE:
                store.set_edge_status(changes["edge"], changes["status"])
            elif entry.op == REJECT_EDGE:
                store.restore(changes["edge"])
            else:
                unmerge_nodes(store, changes)
                self._unaccept(changes)
            self._follow(entry)
            store.execute(
                "UPDATE corrections SET status = ?, undone_at = ?, undone_by = ? WHERE id = ?",
                (UNDONE, store.now(), token, entry.id),
            )
            return self.get(entry.id)

    def _blocking(self, entry: Correction) -> list[str]:
        """The records that moved since ``entry``: their full revision, or for a
        decision on an edge, the edge's decision revision (SPEC 4.4, 5.7)."""
        store = self.store
        changes = entry.changes or {}
        if entry.op in (CONFIRM_EDGE, REJECT_EDGE):
            edge_id = changes["edge"]
            try:
                current = store.get_edge(edge_id).decision_rev
            except NotFound:
                return [edge_id]
            return [] if current == changes["decision_rev"] else [edge_id]
        now = store.revisions(list(entry.after_revs))
        return [r for r, rev in entry.after_revs.items() if now.get(r) != rev]

    def _follow(self, entry: Correction) -> None:
        """An undo puts back exactly the state an earlier correction left, under a
        new revision: the earlier entry that left it follows, so it can be undone
        in turn (later corrections are undone first, latest first)."""
        store = self.store
        now = store.revisions(list(entry.before_revs))
        for record_id, rev in entry.before_revs.items():
            if record_id not in now:
                continue
            rows = store.execute(
                "SELECT c.id FROM corrections c JOIN correction_records r"
                " ON r.correction_id = c.id WHERE r.record_id = ? AND c.status = ? AND c.id != ?",
                (record_id, APPLIED, entry.id),
            ).fetchall()
            for row in rows:
                earlier = self.get(row["id"])
                if earlier.after_revs.get(record_id) == rev:
                    revs = {**earlier.after_revs, record_id: now[record_id]}
                    store.execute(
                        "UPDATE corrections SET after_revs = ? WHERE id = ?",
                        (_dumps(revs), earlier.id),
                    )

    def _unaccept(self, changes: Mapping[str, Any]) -> None:
        """The proposal a merge accepted is pending again; the one it recorded for a
        direct merge goes with it."""
        store = self.store
        proposal_id = changes.get("proposal")
        if not proposal_id:
            return
        try:
            store.set_proposal_status(proposal_id, PENDING)
        except NotFound:
            return  # forgotten with its source meanwhile
        if changes.get("proposal_created"):
            store.remove_pending_proposal(proposal_id)


__all__ = [
    "APPLIED",
    "CONFIRM_EDGE",
    "EDGE_DECISIONS",
    "MERGE_NODES",
    "REJECT_EDGE",
    "UNDONE",
    "UPDATE_NODE",
    "Correction",
    "CorrectionRefused",
    "Corrections",
    "default_keep",
    "merge_nodes",
    "merge_refusal",
    "unmerge_nodes",
]
