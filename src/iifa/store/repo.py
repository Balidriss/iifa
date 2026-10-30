"""The repository: every read and write of nodes, facts, edges, sources, scope
sets, probe sets and the lexical index goes through ``Store``. One connection, one lock,
one writer.

Visibility (a ``Visibility`` view, SPEC section 4.6) is applied inside every
read query, before any statistic or score is computed (P9)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import exp, log
from typing import Any

from iifa import vectors
from iifa.ids import kind_of, new_id
from iifa.recall.tokenizer import normalize, tokenize
from iifa.scopes import Visibility, validate_scope, within
from iifa.store.checkpoint import DEFAULT_INTERVAL, Checkpointer
from iifa.store.connection import BACKSTOP_AUTOCHECKPOINT_PAGES, open_database
from iifa.store.migrate import migrate
from iifa.store.models import Edge, Fact, Node, Probe, ProbeSet, Proposal, Source

Clock = Callable[[], datetime]

ARCHIVABLE = ("node", "fact", "edge")
FORGETTABLE = ("node", "fact", "edge", "src")
_TABLES = {"node": "nodes", "fact": "facts", "edge": "edges", "src": "sources"}

PROPOSED, CONFIRMED = "proposed", "confirmed"
PENDING, ACCEPTED, REJECTED = "pending", "accepted", "rejected"
ALIAS, MERGE, FACT = "alias", "merge", "fact"
PROPOSAL_KINDS = (ALIAS, MERGE, FACT)
FACT_PAYLOAD_KEYS = (
    "attribute",
    "value",
    "valid_from",
    "valid_to",
    "language",
    "flags",
    "provenance",
    "confidence",
)
# the embedded facets of a record (SPEC 6.1)
TEXT, NAME, SUMMARY = "text", "name", "summary"
REINDEX_NODES = "reindex_nodes"  # service_state mark set by migration 11


@dataclass(frozen=True)
class PendingVector:
    """A facet whose vector is missing or stale, with the text to embed."""

    record_id: str
    facet: str
    text: str


@dataclass(frozen=True)
class VectorRow:
    record_id: str
    facet: str
    dimension: int
    blob: bytes


def node_name_text(node: Node) -> str:
    """What a node's name vector embeds: its name, then its aliases."""
    return "; ".join([node.name, *node.aliases])


MAX_EDGE_WEIGHT = 10.0
REINFORCE_STEP = 0.25
DEFAULT_NODE_INDEX_LIMIT = 5000


class NotFound(LookupError):
    """No such record, or none the caller may see."""


class ProposalDecided(ValueError):
    """A proposal that was already accepted or rejected is not decided again."""


class ProposalNotApplicable(ValueError):
    """A proposal that can be rejected but not accepted directly: a merge, which only a
    reviewed ``merge_nodes`` of a dream applies (SPEC 4.4b, 5.6)."""


class ScopeMismatch(ValueError):
    """Records whose scope sets differ may not be merged or consolidated (§5.6)."""


@dataclass(frozen=True)
class Posting:
    """One (term, document) pair inside the visible corpus, with its term frequency."""

    term: str
    rowid: int
    doc_id: str
    kind: str
    tf: int
    ntok: int


@dataclass(frozen=True)
class CorpusStats:
    documents: int
    avg_length: float


def utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def _ph(values: Sequence[object]) -> str:
    return ",".join("?" for _ in values)


def _visible(view: Visibility, record_column: str) -> tuple[str, list[str]]:
    """Predicate: the record in ``record_column`` has at least one scope inside ``view``.

    A correlated scalar count, on purpose. ``EXISTS`` is rewritten to a join by
    SQLite 3.51 and then returns one row per matching scope of a multi-scope
    record; ``IN (SELECT ...)`` materialises every visible id on each query.
    The count probes the ``(record_id, scope)`` primary key once per row.
    """
    pred, params = view.sql("rs.scope")
    sql = (
        f"(SELECT count(*) FROM record_scopes rs WHERE rs.record_id = {record_column}"  # noqa: S608
        f" AND {pred}) > 0"
    )
    return sql, params


_LEX_LIVE = (
    " ((d.kind = 'fact' AND f.archived_at IS NULL AND f.valid_to IS NULL"
    " AND (f.node_id IS NULL OR fn.archived_at IS NULL))"
    " OR (d.kind = 'node' AND n.archived_at IS NULL))"
)


class Store:
    """Repository over one SQLCipher file. Thread-safe through a single lock."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        key: str,
        *,
        clock: Clock = utc_now,
        retain_sources: bool = True,
        plaintext_for_tests: bool = False,
        checkpoint_interval: float = DEFAULT_INTERVAL,
    ) -> None:
        """``checkpoint_interval`` is how often (seconds) a background thread moves
        WAL frames into the file so that no write pays for it; ``0`` disables the
        thread and leaves SQLite's own checkpointing on the writing connection."""
        self.path = os.fspath(path)
        self.clock = clock
        self.retain_sources = retain_sources
        self._lock = threading.RLock()
        self._checkpoint_lock = threading.Lock()
        background = checkpoint_interval > 0
        self._conn = open_database(
            path,
            key,
            plaintext_for_tests=plaintext_for_tests,
            autocheckpoint_pages=BACKSTOP_AUTOCHECKPOINT_PAGES if background else None,
        )
        self._checkpointer: Checkpointer | None = None
        try:
            with self._lock:
                migrate(self._conn)
                self._reindex_marked_nodes()
            if background:
                self._checkpointer = Checkpointer(
                    path,
                    key,
                    interval=checkpoint_interval,
                    lock=self._checkpoint_lock,
                    plaintext_for_tests=plaintext_for_tests,
                ).start()
        except Exception:
            self._conn.close()
            raise

    # ------------------------------------------------------------------ plumbing

    def close(self) -> None:
        if self._checkpointer is not None:
            self._checkpointer.stop()
            self._checkpointer = None
        with self._lock:
            self._conn.close()

    @property
    def checkpointer(self) -> Checkpointer | None:
        """The background checkpoint thread, or None when disabled."""
        return self._checkpointer

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def lock(self) -> threading.RLock:
        """Hold it across several store calls that must be seen as one unit."""
        return self._lock

    def now(self) -> str:
        return _iso(self.clock())

    def now_plus(self, delta: timedelta) -> str:
        """The clock's time plus ``delta``, as the store writes timestamps."""
        return _iso(self.clock() + delta)

    def checkpoint(self, mode: str = "PASSIVE") -> dict[str, int]:
        """Move WAL frames into the database file now. ``PASSIVE`` never blocks
        readers or the writer; ``TRUNCATE`` also resets the WAL file (what
        ``forget`` needs). Returns the frames in the WAL and the frames moved."""
        if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            raise ValueError(f"unknown checkpoint mode {mode!r}")
        with self._lock, self._checkpoint_lock:
            row = self._execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
            return {"busy": int(row[0]), "log_frames": int(row[1]), "checkpointed": int(row[2])}

    def _execute(self, sql: str, params: Sequence[object] = ()) -> Any:
        return self._conn.execute(sql, tuple(params))

    def _transaction(self) -> _Transaction:
        return _Transaction(self)

    def execute(self, sql: str, params: Sequence[object] = ()) -> Any:
        """Run one statement on the store's connection (hold ``lock`` while you do)."""
        return self._execute(sql, params)

    def transaction(self) -> _Transaction:
        """A savepoint: nested store calls inside it commit or roll back as one."""
        return _Transaction(self)

    # ------------------------------------------------------------------ service state

    def get_state(self, key: str) -> str | None:
        """Operational state the service wrote for itself (the background mode)."""
        with self._lock:
            row = self._execute("SELECT value FROM service_state WHERE key = ?", (key,)).fetchone()
            return None if row is None else str(row["value"])

    def set_state(self, key: str, value: str) -> None:
        with self._lock, self._transaction():
            self._execute(
                "INSERT INTO service_state (key, value) VALUES (?, ?)"
                " ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ------------------------------------------------------------------ scopes

    def scopes_of(self, record_id: str) -> tuple[str, ...]:
        with self._lock:
            rows = self._execute(
                "SELECT scope FROM record_scopes WHERE record_id = ? ORDER BY scope", (record_id,)
            ).fetchall()
            if not rows:
                raise NotFound(record_id)
            return tuple(r["scope"] for r in rows)

    def _scopes_map(self, record_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Scope sets of many records in one query."""
        out: dict[str, list[str]] = {}
        if not record_ids:
            return {}
        rows = self._execute(
            f"SELECT record_id, scope FROM record_scopes WHERE record_id IN ({_ph(record_ids)})"  # noqa: S608
            " ORDER BY record_id, scope",
            record_ids,
        ).fetchall()
        for row in rows:
            out.setdefault(row["record_id"], []).append(row["scope"])
        return {k: tuple(v) for k, v in out.items()}

    def _set_scopes(self, record_id: str, scopes: Iterable[str]) -> tuple[str, ...]:
        unique = tuple(dict.fromkeys(validate_scope(s) for s in scopes))
        if not unique:
            raise ValueError("a record needs at least one scope")
        for scope in unique:
            self._execute(
                "INSERT OR IGNORE INTO record_scopes (record_id, scope) VALUES (?, ?)",
                (record_id, scope),
            )
        return unique

    def _replace_scopes(self, record_id: str, scopes: Iterable[str]) -> tuple[str, ...]:
        """The record's scope set becomes exactly ``scopes``: the rows it leaves are
        deleted, the new ones inserted (each change bumps ``rev`` by trigger)."""
        unique = tuple(dict.fromkeys(validate_scope(s) for s in scopes))
        if not unique:
            raise ValueError("a record needs at least one scope")
        self._execute(
            f"DELETE FROM record_scopes WHERE record_id = ? AND scope NOT IN ({_ph(unique)})",  # noqa: S608
            (record_id, *unique),
        )
        return self._set_scopes(record_id, unique)

    def require_identical_scopes(self, record_ids: Sequence[str]) -> tuple[str, ...]:
        """The guard a dream must pass before merging or consolidating records (§5.6):
        every record has exactly the same scope set, or nothing happens."""
        if not record_ids:
            raise ValueError("no records")
        sets = {self.scopes_of(r) for r in record_ids}
        if len(sets) != 1:
            raise ScopeMismatch("records have different scope sets; merging would widen visibility")
        return sets.pop()

    def lookup_external(self, external_id: str, view: Visibility) -> list[dict[str, str]]:
        """Records carrying ``external_id`` that the view may see."""
        out: list[dict[str, str]] = []
        with self._lock:
            for kind, table in _TABLES.items():
                vis, params = _visible(view, "t.id")
                rows = self._execute(
                    f"SELECT t.id FROM {table} t WHERE t.external_id = ? AND {vis} ORDER BY t.id",  # noqa: S608
                    (external_id, *params),
                ).fetchall()
                out.extend({"id": r["id"], "kind": kind} for r in rows)
        return out

    def lookup_name(self, name: str, view: Visibility) -> list[dict[str, str]]:
        """Visible, live nodes whose name or an alias normalises to ``name``
        (what ``memory_about`` uses), closest scope first."""
        norm = normalize(name)
        if not norm:
            return []
        with self._lock:
            vis, params = _visible(view, "n.id")
            rows = self._execute(
                f"SELECT n.* FROM nodes n WHERE n.archived_at IS NULL AND {vis}"  # noqa: S608
                " AND (n.name_norm = ? OR n.id IN"
                " (SELECT node_id FROM node_aliases WHERE alias_norm = ?))"
                " ORDER BY n.created_at, n.id",
                (*params, norm, norm),
            ).fetchall()
            nodes = sorted(
                self._nodes_from_rows(rows),
                key=lambda n: (view.rank(n.scopes), n.created_at, n.id),
            )
            return [{"id": n.id, "kind": "node", "name": n.name} for n in nodes]

    # ------------------------------------------------------------------ nodes

    def create_node(
        self,
        *,
        scopes: Iterable[str],
        name: str,
        kind: str = "other",
        aliases: Iterable[str] = (),
        summary: str = "",
        external_id: str | None = None,
        language: str | None = None,
    ) -> Node:
        node_id = new_id("node")
        now = self.now()
        with self._lock, self._transaction():
            self._execute(
                "INSERT INTO nodes (id, kind, name, name_norm, summary, external_id, created_at,"
                " updated_at, language) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (node_id, kind, name, normalize(name), summary, external_id, now, now, language),
            )
            self._set_scopes(node_id, scopes)
            for alias in dict.fromkeys(a.strip() for a in aliases if a.strip()):
                self._execute(
                    "INSERT OR IGNORE INTO node_aliases (node_id, alias, alias_norm)"
                    " VALUES (?, ?, ?)",
                    (node_id, alias, normalize(alias)),
                )
            self._reindex_node(node_id)
            return self.get_node(node_id)

    def get_node(self, node_id: str) -> Node:
        """Any node by id, archived or not, whatever its scopes (internal use)."""
        with self._lock:
            row = self._execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row is None:
                raise NotFound(node_id)
            return self._node_from_row(row)

    def _nodes_from_rows(self, rows: Sequence[Any]) -> list[Node]:
        ids = [r["id"] for r in rows]
        scopes = self._scopes_map(ids)
        aliases: dict[str, list[str]] = {}
        if ids:
            for r in self._execute(
                f"SELECT node_id, alias FROM node_aliases WHERE node_id IN ({_ph(ids)})"  # noqa: S608
                " ORDER BY alias",
                ids,
            ):
                aliases.setdefault(r["node_id"], []).append(r["alias"])
        return [
            self._node_from_row(r, tuple(aliases.get(r["id"], ())), scopes.get(r["id"], ()))
            for r in rows
        ]

    def _node_from_row(
        self,
        row: Any,
        aliases: tuple[str, ...] | None = None,
        scopes: tuple[str, ...] | None = None,
    ) -> Node:
        if aliases is None:
            aliases = tuple(
                r["alias"]
                for r in self._execute(
                    "SELECT alias FROM node_aliases WHERE node_id = ? ORDER BY alias", (row["id"],)
                )
            )
        return Node(
            id=row["id"],
            kind=row["kind"],
            name=row["name"],
            aliases=aliases,
            summary=row["summary"],
            scopes=self.scopes_of(row["id"]) if scopes is None else scopes,
            external_id=row["external_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived_at=row["archived_at"],
            language=row["language"],
            flags=tuple(f for f in row["flags"].split(",") if f),
            merged_into=row["merged_into"],
            rev=int(row["rev"]),
        )

    def find_node(self, name: str, view: Visibility) -> Node | None:
        """Exact or normalised match on a name or alias among visible, live nodes,
        preferring the node closest to the requester's scope."""
        norm = normalize(name)
        if not norm:
            return None
        with self._lock:
            vis, params = _visible(view, "n.id")
            rows = self._execute(
                f"SELECT n.* FROM nodes n WHERE n.archived_at IS NULL AND {vis}"  # noqa: S608
                " AND (n.name_norm = ? OR n.id IN"
                " (SELECT node_id FROM node_aliases WHERE alias_norm = ?))"
                " ORDER BY n.created_at, n.id",
                (*params, norm, norm),
            ).fetchall()
            if not rows:
                return None
            nodes = self._nodes_from_rows(rows)
            return min(nodes, key=lambda n: (view.rank(n.scopes), n.created_at, n.id))

    def update_node(
        self,
        node_id: str,
        *,
        summary: str | None = None,
        add_aliases: Iterable[str] = (),
        language: str | None = None,
        flags: Iterable[str] | None = None,
    ) -> Node:
        """Change the summary (with the language it is written in, P12), add aliases,
        replace the flags. ``updated_at`` moves only when something changed."""
        with self._lock, self._transaction():
            before = self.get_node(node_id)
            if summary is not None:
                self._execute(
                    "UPDATE nodes SET summary = ?, language = coalesce(?, language),"
                    " updated_at = ? WHERE id = ?",
                    (summary, language, self.now(), node_id),
                )
                if summary != before.summary:  # SPEC 6.1: embedded text changed
                    self._drop_vector(node_id, SUMMARY)
            elif language is not None:
                self._execute("UPDATE nodes SET language = ? WHERE id = ?", (language, node_id))
            if flags is not None:
                self._execute(
                    "UPDATE nodes SET flags = ? WHERE id = ?",
                    (",".join(sorted(set(flags))), node_id),
                )
            added = 0
            for alias in add_aliases:
                if alias.strip():
                    added += self._execute(
                        "INSERT OR IGNORE INTO node_aliases (node_id, alias, alias_norm)"
                        " VALUES (?, ?, ?)",
                        (node_id, alias.strip(), normalize(alias)),
                    ).rowcount
            if added:
                self._drop_vector(node_id, NAME)
            self._reindex_node(node_id)
            return self.get_node(node_id)

    def _facts_of_node(self, node_id: str) -> list[str]:
        return [
            r["id"] for r in self._execute("SELECT id FROM facts WHERE node_id = ?", (node_id,))
        ]

    def nodes_by_id(self, node_ids: Sequence[str], view: Visibility) -> dict[str, Node]:
        """Visible, unarchived nodes among ``node_ids``."""
        if not node_ids:
            return {}
        with self._lock:
            vis, params = _visible(view, "n.id")
            rows = self._execute(
                f"SELECT n.* FROM nodes n WHERE n.id IN ({_ph(node_ids)})"  # noqa: S608
                f" AND n.archived_at IS NULL AND {vis} ORDER BY n.id",
                (*node_ids, *params),
            ).fetchall()
            return {n.id: n for n in self._nodes_from_rows(rows)}

    def node_index(
        self, view: Visibility, *, limit: int = DEFAULT_NODE_INDEX_LIMIT
    ) -> list[tuple[str, str, str, str, tuple[str, ...]]]:
        """``(id, kind, name, name_norm, alias_norms)`` of the visible, live nodes, most
        recently updated first, bounded: what fuzzy resolution (SPEC 5.1 step 2)
        compares a mention against. Names only, never a fact (P9 applies:
        only visible nodes are candidates)."""
        if view.is_empty or limit <= 0:
            return []
        with self._lock:
            vis, params = _visible(view, "n.id")
            rows = self._execute(
                f"SELECT n.id, n.kind, n.name, n.name_norm FROM nodes n WHERE n.archived_at IS NULL"  # noqa: S608
                f" AND {vis} ORDER BY n.updated_at DESC, n.id DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
            ids = [r["id"] for r in rows]
            aliases: dict[str, list[str]] = {}
            if ids:
                for r in self._execute(
                    f"SELECT node_id, alias_norm FROM node_aliases WHERE node_id IN ({_ph(ids)})"  # noqa: S608
                    " ORDER BY alias_norm",
                    ids,
                ):
                    aliases.setdefault(r["node_id"], []).append(r["alias_norm"])
            return [
                (r["id"], r["kind"], r["name"], r["name_norm"], tuple(aliases.get(r["id"], ())))
                for r in rows
            ]

    def nodes_touched_by_session(self, session_id: str, view: Visibility) -> list[str]:
        """Ids of the visible, live nodes that gained a fact or an edge from a source
        recorded under ``session_id``, in creation order (what a session end
        refreshes the summaries of)."""
        if not session_id or view.is_empty:
            return []
        with self._lock:
            vis, params = _visible(view, "n.id")
            rows = self._execute(
                "SELECT DISTINCT n.id, n.created_at FROM nodes n WHERE n.archived_at IS NULL"  # noqa: S608
                f" AND {vis} AND n.id IN ("
                " SELECT f.node_id FROM facts f JOIN sources s ON s.id = f.source_id"
                "  WHERE s.session_id = ? AND f.node_id IS NOT NULL"
                " UNION SELECT e.src FROM edges e JOIN edge_evidence ev ON ev.edge_id = e.id"
                "  JOIN sources s ON s.id = ev.source_id WHERE s.session_id = ?"
                " UNION SELECT e.dst FROM edges e JOIN edge_evidence ev ON ev.edge_id = e.id"
                "  JOIN sources s ON s.id = ev.source_id WHERE s.session_id = ?)"
                " ORDER BY n.created_at, n.id",
                (*params, session_id, session_id, session_id),
            ).fetchall()
            return [r["id"] for r in rows]

    # ------------------------------------------------------------------ sources

    def add_source(
        self,
        *,
        scopes: Iterable[str],
        kind: str,
        text: str,
        reference: str | None = None,
        external_id: str | None = None,
        session_id: str | None = None,
        language: str | None = None,
    ) -> Source:
        source_id = new_id("src")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock, self._transaction():
            self._execute(
                "INSERT INTO sources (id, kind, text, text_sha256, reference, external_id,"
                " created_at, session_id, language) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    kind,
                    text if self.retain_sources else None,
                    digest,
                    reference,
                    external_id,
                    self.now(),
                    session_id,
                    language,
                ),
            )
            self._set_scopes(source_id, scopes)
            return self.get_source(source_id)

    def get_source(self, source_id: str) -> Source:
        with self._lock:
            row = self._execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
            if row is None:
                raise NotFound(source_id)
            return Source(
                id=row["id"],
                kind=row["kind"],
                text=row["text"],
                text_sha256=row["text_sha256"],
                reference=row["reference"],
                scopes=self.scopes_of(row["id"]),
                external_id=row["external_id"],
                created_at=row["created_at"],
                session_id=row["session_id"],
                language=row["language"],
            )

    def source_if_visible(self, source_id: str, view: Visibility) -> Source | None:
        with self._lock:
            vis, params = _visible(view, "s.id")
            row = self._execute(
                f"SELECT s.id FROM sources s WHERE s.id = ? AND {vis}",  # noqa: S608
                (source_id, *params),
            ).fetchone()
            return None if row is None else self.get_source(source_id)

    # ------------------------------------------------------------------ facts

    def add_fact(
        self,
        *,
        scopes: Iterable[str],
        statement: str,
        origin: str,
        node_id: str | None = None,
        source_id: str | None = None,
        excerpt: tuple[int, int] | None = None,
        attribute: str | None = None,
        value: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 1.0,
        pinned: bool = False,
        flags: Iterable[str] = (),
        external_id: str | None = None,
        language: str | None = None,
        conflicts_with: str | None = None,
        same_as: str | None = None,
        provenance: str | None = None,
        proposal_id: str | None = None,
    ) -> Fact:
        fact_id = new_id("fact")
        now = self.now()
        with self._lock, self._transaction():
            if node_id is not None:
                self.get_node(node_id)
            self._execute(
                "INSERT INTO facts (id, node_id, statement, attribute, value, valid_from, valid_to,"
                " origin, confidence, source_id, excerpt_start, excerpt_end, pinned, flags,"
                " external_id, created_at, updated_at, language, conflicts_with, same_as,"
                " provenance, proposal_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact_id,
                    node_id,
                    statement,
                    attribute,
                    value,
                    valid_from,
                    valid_to,
                    origin,
                    confidence,
                    source_id,
                    excerpt[0] if excerpt else None,
                    excerpt[1] if excerpt else None,
                    int(pinned),
                    ",".join(sorted(set(flags))),
                    external_id,
                    now,
                    now,
                    language,
                    conflicts_with,
                    same_as,
                    provenance,
                    proposal_id,
                ),
            )
            unique = self._set_scopes(fact_id, scopes)
            if node_id is not None:
                self._execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (now, node_id))
            self._index(fact_id, "fact", self._fact_text(statement, attribute, value), unique)
            self._refresh_hidden(fact_id, "fact")
            return self.get_fact(fact_id)

    @staticmethod
    def _fact_text(statement: str, attribute: str | None, value: str | None) -> str:
        return " ".join(part for part in (statement, attribute, value) if part)

    def get_fact(self, fact_id: str) -> Fact:
        """Any fact by id, archived or not, whatever its scopes (internal use)."""
        with self._lock:
            row = self._execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                raise NotFound(fact_id)
            return self._fact_from_row(row)

    def attach_fact(
        self,
        fact_id: str,
        node_id: str | None,
        *,
        pinned: bool | None = None,
        scopes: Sequence[str] | None = None,
    ) -> Fact:
        """Attach a free-standing fact to a node (or detach it), pin it, and/or
        replace its scope set (SPEC 4.6) - all in one transaction."""
        with self._lock, self._transaction():
            fact = self.get_fact(fact_id)
            if node_id is not None:
                self.get_node(node_id)
            now = self.now()
            self._execute(
                "UPDATE facts SET node_id = ?, updated_at = ? WHERE id = ?", (node_id, now, fact_id)
            )
            if pinned is not None:
                self._execute("UPDATE facts SET pinned = ? WHERE id = ?", (int(pinned), fact_id))
            if scopes is not None and tuple(sorted(set(scopes))) != fact.scopes:
                self._replace_scopes(fact_id, scopes)
                self._unindex(fact_id)
                text = self._fact_text(fact.statement, fact.attribute, fact.value)
                self._index(fact_id, "fact", text, self.scopes_of(fact_id))
            if node_id is not None:
                self._execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (now, node_id))
            self._refresh_hidden(fact_id, "fact")
            return self.get_fact(fact_id)

    def close_fact(
        self, fact_id: str, *, valid_to: str | None = None, superseded_by: str | None = None
    ) -> Fact:
        """Supersede (P7): the fact stops being live at ``valid_to`` (now by default)
        and leaves recall; it stays readable, with the fact that replaced it."""
        with self._lock, self._transaction():
            fact = self.get_fact(fact_id)
            if superseded_by is not None:
                self.get_fact(superseded_by)
            now = self.now()
            self._execute(
                "UPDATE facts SET valid_to = coalesce(valid_to, ?), superseded_by = coalesce(?,"
                " superseded_by), updated_at = ? WHERE id = ?",
                (valid_to or now, superseded_by, now, fact_id),
            )
            if fact.node_id is not None:
                self._execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (now, fact.node_id))
            self._refresh_hidden(fact_id, "fact")
            return self.get_fact(fact_id)

    def flag_fact(self, fact_id: str, *flags: str, conflicts_with: str | None = None) -> Fact:
        """Add flags (``conflict``, ``language_mismatch``) and, for a conflict, the
        fact it stands against; nothing else about the fact changes."""
        with self._lock, self._transaction():
            fact = self.get_fact(fact_id)
            merged = ",".join(sorted({*fact.flags, *flags}))
            self._execute(
                "UPDATE facts SET flags = ?, conflicts_with = coalesce(?, conflicts_with),"
                " updated_at = ? WHERE id = ?",
                (merged, conflicts_with, self.now(), fact_id),
            )
            return self.get_fact(fact_id)

    def _fact_visible_sql(self, view: Visibility) -> tuple[str, list[str]]:
        vis, params = _visible(view, "f.id")
        return (
            f"{vis} AND f.archived_at IS NULL AND (f.node_id IS NULL OR n.archived_at IS NULL)",
            params,
        )

    def free_facts(self, view: Visibility, *, flag: str) -> list[Fact]:
        """The live, visible free-standing facts carrying ``flag`` (SPEC 4.2: the
        speaker facts a new one is compared with), most recent first."""
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT f.* FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE f.node_id IS NULL AND (',' || f.flags || ',') LIKE ? AND {vis}"
                " AND f.valid_to IS NULL ORDER BY f.created_at DESC, f.id DESC",
                (f"%,{flag},%", *params),
            ).fetchall()
            return self._facts_from_rows(rows)

    def facts(
        self,
        node_ids: Sequence[str],
        view: Visibility,
        *,
        live_only: bool = True,
        limit: int | None = None,
    ) -> list[Fact]:
        """Visible facts of ``node_ids`` (pinned first, then most recent)."""
        if not node_ids:
            return []
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            live = " AND f.valid_to IS NULL" if live_only else ""
            limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
            rows = self._execute(
                "SELECT f.* FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE f.node_id IN ({_ph(node_ids)}) AND {vis}{live}"
                f" ORDER BY f.pinned DESC, f.created_at DESC, f.id DESC{limit_sql}",
                (*node_ids, *params),
            ).fetchall()
            return self._facts_from_rows(rows)

    def fact_roots(self, fact_ids: Sequence[str]) -> dict[str, str]:
        """The first fact of each fact's equivalence group (SPEC 5.4): its
        ``same_as``, or itself."""
        if not fact_ids:
            return {}
        with self._lock:
            rows = self._execute(
                f"SELECT id, same_as FROM facts WHERE id IN ({_ph(fact_ids)})",  # noqa: S608
                tuple(fact_ids),
            ).fetchall()
        return {r["id"]: r["same_as"] or r["id"] for r in rows}

    def equivalents(self, roots: Sequence[str], view: Visibility) -> dict[str, list[Fact]]:
        """The visible, live members of each group, the first fact included,
        oldest first (P9: filtered by the view before anything else)."""
        if not roots:
            return {}
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT f.* FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE (f.id IN ({_ph(roots)}) OR f.same_as IN ({_ph(roots)}))"
                f" AND {vis} AND f.valid_to IS NULL ORDER BY f.created_at, f.id",
                (*roots, *roots, *params),
            ).fetchall()
        out: dict[str, list[Fact]] = {root: [] for root in roots}
        for member in self._facts_from_rows(rows):
            out.setdefault(member.same_as or member.id, []).append(member)
        return out

    def attribute_keys(self, node_ids: Sequence[str], view: Visibility) -> dict[str, list[str]]:
        """The attribute keys the visible live facts of ``node_ids`` carry, as
        stored, most recent first (SPEC 5.4: what the prompt asks to reuse)."""
        if not node_ids:
            return {}
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT f.node_id, f.attribute FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE f.node_id IN ({_ph(node_ids)}) AND {vis} AND f.valid_to IS NULL"
                " AND f.attribute IS NOT NULL ORDER BY f.created_at DESC, f.id DESC",
                (*node_ids, *params),
            ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            keys = out.setdefault(row["node_id"], [])
            if row["attribute"] not in keys:
                keys.append(row["attribute"])
        return out

    def fact_counts(self, node_ids: Sequence[str], view: Visibility) -> dict[str, int]:
        """Number of live, visible facts per node."""
        if not node_ids:
            return {}
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT f.node_id AS node_id, count(*) AS n"  # noqa: S608
                " FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"
                f" WHERE f.node_id IN ({_ph(node_ids)}) AND {vis} AND f.valid_to IS NULL"
                " GROUP BY f.node_id",
                (*node_ids, *params),
            ).fetchall()
            return {r["node_id"]: int(r["n"]) for r in rows}

    def facts_top(self, node_ids: Sequence[str], view: Visibility, per_node: int) -> list[Fact]:
        """The ``per_node`` best live, visible facts of each node, in one query."""
        if not node_ids:
            return []
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT * FROM (SELECT f.*, row_number() OVER (PARTITION BY f.node_id"  # noqa: S608
                " ORDER BY f.pinned DESC, f.created_at DESC, f.id DESC) AS rn"
                " FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"
                f" WHERE f.node_id IN ({_ph(node_ids)}) AND {vis} AND f.valid_to IS NULL)"
                " WHERE rn <= ? ORDER BY node_id, rn",
                (*node_ids, *params, per_node),
            ).fetchall()
            return self._facts_from_rows(rows)

    def pinned_facts(self, view: Visibility) -> list[Fact]:
        """Live, visible, pinned facts in creation order: the stable part of the
        profile block (SPEC 8.1). Nothing here depends on the current time."""
        if view.is_empty:
            return []
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT f.* FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE f.pinned = 1 AND f.valid_to IS NULL AND {vis}"
                " ORDER BY f.created_at, f.id",
                params,
            ).fetchall()
            return self._facts_from_rows(rows)

    def central_nodes(self, view: Visibility, limit: int) -> list[tuple[Node, int, int]]:
        """The most connected visible nodes: ``(node, live facts, live edges)``,
        most central first, ties broken by creation order so the result is
        deterministic. Facts and edges are counted only when visible, and an
        edge only when its far node is visible too (P9)."""
        if view.is_empty or limit <= 0:
            return []
        with self._lock:
            vis_n, p_n = _visible(view, "n.id")
            vis_f, p_f = _visible(view, "f.id")
            vis_e, p_e = _visible(view, "e.id")
            vis_o, p_o = _visible(view, "o.id")
            rows = self._execute(
                "SELECT * FROM (SELECT n.*,"  # noqa: S608
                " (SELECT count(*) FROM facts f WHERE f.node_id = n.id"
                f"  AND f.archived_at IS NULL AND f.valid_to IS NULL AND {vis_f}) AS nfacts,"
                " (SELECT count(*) FROM edges e JOIN nodes o"
                "  ON o.id = CASE WHEN e.src = n.id THEN e.dst ELSE e.src END"
                "  WHERE (e.src = n.id OR e.dst = n.id) AND e.archived_at IS NULL"
                f"  AND e.valid_to IS NULL AND o.archived_at IS NULL AND {vis_e} AND {vis_o})"
                " AS nedges"
                f" FROM nodes n WHERE n.archived_at IS NULL AND {vis_n})"
                " ORDER BY (nedges + nfacts) DESC, created_at, id LIMIT ?",
                (*p_f, *p_e, *p_o, *p_n, int(limit)),
            ).fetchall()
            nodes = self._nodes_from_rows(rows)
            return [
                (node, int(row["nfacts"]), int(row["nedges"]))
                for node, row in zip(nodes, rows, strict=True)
            ]

    def fact_subjects(self, fact_ids: Sequence[str]) -> dict[str, str | None]:
        """``fact id -> node id`` for facts already known to be visible (from postings)."""
        if not fact_ids:
            return {}
        with self._lock:
            rows = self._execute(
                f"SELECT id, node_id FROM facts WHERE id IN ({_ph(fact_ids)})",  # noqa: S608
                fact_ids,
            ).fetchall()
            return {r["id"]: r["node_id"] for r in rows}

    def facts_by_id(self, fact_ids: Sequence[str], view: Visibility) -> dict[str, Fact]:
        if not fact_ids:
            return {}
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT f.* FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE f.id IN ({_ph(fact_ids)}) AND {vis} ORDER BY f.id",
                (*fact_ids, *params),
            ).fetchall()
            return {f.id: f for f in self._facts_from_rows(rows)}

    def facts_known_at(self, node_id: str, view: Visibility, at: str) -> list[Fact]:
        """What was known about ``node_id`` at ``at`` (P7): the visible facts recorded
        by then and not closed by then, live or since superseded, oldest first."""
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            rows = self._execute(
                "SELECT f.* FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE f.node_id = ? AND {vis} AND f.created_at <= ?"
                " AND (f.valid_to IS NULL OR f.valid_to > ?)"
                " ORDER BY f.created_at, f.id",
                (node_id, *params, at, at),
            ).fetchall()
            return self._facts_from_rows(rows)

    def fact_history(
        self,
        view: Visibility,
        *,
        node_id: str | None,
        since: str | None,
        until: str | None,
        limit: int,
    ) -> list[Fact]:
        """Visible facts, live or closed, of one node or of the whole view, whose
        life touched ``[since, until]``: recorded before ``until`` and not closed
        before ``since``. Newest first, bounded (the timeline's raw material)."""
        if view.is_empty or limit <= 0:
            return []
        with self._lock:
            vis, params = self._fact_visible_sql(view)
            clauses = [vis]
            args: list[object] = list(params)
            if node_id is not None:
                clauses.append("f.node_id = ?")
                args.append(node_id)
            if until is not None:
                clauses.append("f.created_at <= ?")
                args.append(until)
            if since is not None:
                clauses.append("(f.valid_to IS NULL OR f.valid_to >= ?)")
                args.append(since)
            rows = self._execute(
                "SELECT f.* FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"  # noqa: S608
                f" WHERE {' AND '.join(clauses)} ORDER BY f.created_at DESC, f.id DESC LIMIT ?",
                (*args, int(limit)),
            ).fetchall()
            return self._facts_from_rows(rows)

    # ------------------------------------------------------------------ edges

    def add_edge(
        self,
        *,
        scopes: Iterable[str],
        src: str,
        dst: str,
        type: str,
        origin: str,
        status: str = "confirmed",
        weight: float = 1.0,
        source_id: str | None = None,
        external_id: str | None = None,
    ) -> Edge:
        if status not in (PROPOSED, CONFIRMED):
            raise ValueError(f"unknown edge status {status!r}")
        edge_id = new_id("edge")
        now = self.now()
        with self._lock, self._transaction():
            self.get_node(src)
            self.get_node(dst)
            self._execute(
                "INSERT INTO edges (id, src, dst, type, weight, status, origin, source_id,"
                " external_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (edge_id, src, dst, type, weight, status, origin, source_id, external_id, now, now),
            )
            self._set_scopes(edge_id, scopes)
            if source_id is not None:
                self._execute(
                    "INSERT OR IGNORE INTO edge_evidence (edge_id, source_id) VALUES (?, ?)",
                    (edge_id, source_id),
                )
            return self.get_edge(edge_id)

    def add_evidence(self, edge_id: str, source_id: str) -> int:
        """Record that ``source_id`` observed the edge; returns the number of distinct
        sources that now evidence it (what SPEC 5.2's confirmation rule counts)."""
        with self._lock, self._transaction():
            self.get_edge(edge_id)
            self.get_source(source_id)
            self._execute(
                "INSERT OR IGNORE INTO edge_evidence (edge_id, source_id) VALUES (?, ?)",
                (edge_id, source_id),
            )
            self._execute("UPDATE edges SET updated_at = ? WHERE id = ?", (self.now(), edge_id))
            row = self._execute(
                "SELECT count(*) AS n FROM edge_evidence WHERE edge_id = ?", (edge_id,)
            ).fetchone()
            return int(row["n"])

    def confirm_edge(self, edge_id: str) -> Edge:
        """A confirmed edge never decays away (SPEC 5.3)."""
        with self._lock, self._transaction():
            self.get_edge(edge_id)
            self._execute(
                "UPDATE edges SET status = ?, updated_at = ? WHERE id = ?",
                (CONFIRMED, self.now(), edge_id),
            )
            return self.get_edge(edge_id)

    def reinforce_edge(self, edge_id: str, *, step: float = REINFORCE_STEP) -> Edge | None:
        """The expand signal (SPEC 5.3): the edge was used by a recall the agent then
        expanded. Its use count grows, its last use is now, its weight grows by
        ``step`` up to ``MAX_EDGE_WEIGHT``. A closed or archived edge is left alone."""
        with self._lock, self._transaction():
            row = self._execute(
                "SELECT 1 FROM edges WHERE id = ? AND archived_at IS NULL AND valid_to IS NULL",
                (edge_id,),
            ).fetchone()
            if row is None:
                return None
            now = self.now()
            self._execute(
                "UPDATE edges SET use_count = use_count + 1, last_used_at = ?,"
                " weight = min(?, weight + ?), updated_at = ? WHERE id = ?",
                (now, MAX_EDGE_WEIGHT, float(step), now, edge_id),
            )
            return self.get_edge(edge_id)

    def close_edge(self, edge_id: str, *, valid_to: str | None = None) -> Edge:
        with self._lock, self._transaction():
            self.get_edge(edge_id)
            now = self.now()
            self._execute(
                "UPDATE edges SET valid_to = coalesce(valid_to, ?), updated_at = ? WHERE id = ?",
                (valid_to or now, now, edge_id),
            )
            return self.get_edge(edge_id)

    def decay_edges(
        self, *, half_life_days: float, prune_below: float, view: Visibility | None = None
    ) -> dict[str, int]:
        """SPEC 5.3 with the store's clock: every live ``proposed`` edge loses weight
        by a half-life counted from the later of its last use and its last decay
        (or its creation), and one that falls under ``prune_below`` is closed,
        never deleted (P5). Confirmed edges are never touched. Limited to the
        edges ``view`` sees when given. Idempotent on the same clock reading: the
        decayed weight is a function of time, not of how many times decay ran."""
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        now_dt = self.clock()
        now = _iso(now_dt)
        counts = {"decayed": 0, "pruned": 0}
        columns = "id, weight, last_used_at, decayed_at, created_at"
        with self._lock, self._transaction():
            if view is None:
                rows = self._execute(
                    f"SELECT {columns} FROM edges WHERE status = ?"  # noqa: S608
                    " AND archived_at IS NULL AND valid_to IS NULL",
                    (PROPOSED,),
                ).fetchall()
            else:
                vis, params = _visible(view, "e.id")
                rows = self._execute(
                    f"SELECT {columns} FROM edges e"  # noqa: S608
                    f" WHERE e.status = ? AND e.archived_at IS NULL AND e.valid_to IS NULL"
                    f" AND {vis}",
                    (PROPOSED, *params),
                ).fetchall()
            for row in rows:
                anchor = max(
                    stamp
                    for stamp in (row["last_used_at"], row["decayed_at"], row["created_at"])
                    if stamp
                )
                weight = decayed_weight(float(row["weight"]), anchor, now_dt, half_life_days)
                if weight < prune_below:
                    self._execute(
                        "UPDATE edges SET valid_to = ?, decayed_at = ?, updated_at = ?"
                        " WHERE id = ?",
                        (now, now, now, row["id"]),
                    )
                    counts["pruned"] += 1
                elif weight != float(row["weight"]):
                    self._execute(
                        "UPDATE edges SET weight = ?, decayed_at = ?, updated_at = ? WHERE id = ?",
                        (weight, now, now, row["id"]),
                    )
                    counts["decayed"] += 1
        return counts

    def get_edge(self, edge_id: str) -> Edge:
        with self._lock:
            row = self._execute("SELECT * FROM edges WHERE id = ?", (edge_id,)).fetchone()
            if row is None:
                raise NotFound(edge_id)
            return self._edges_from_rows([row])[0]

    def edges_between(self, node_ids: Sequence[str], view: Visibility) -> list[Edge]:
        """Live, visible edges whose *both* ends are among ``node_ids`` (what the
        write path checks before proposing a link between two mentioned nodes)."""
        if len(node_ids) < 2:
            return []
        return [e for e in self.edges_of(node_ids, view) if e.src in node_ids and e.dst in node_ids]

    def edges_of(self, node_ids: Sequence[str], view: Visibility) -> list[Edge]:
        """Live edges touching ``node_ids`` whose edge *and both ends* are visible
        and unarchived (P9: expansion never traverses an invisible record)."""
        if not node_ids:
            return []
        with self._lock:
            ph = _ph(node_ids)
            vis_e, p_e = _visible(view, "e.id")
            vis_s, p_s = _visible(view, "s.id")
            vis_d, p_d = _visible(view, "d.id")
            rows = self._execute(
                "SELECT e.* FROM edges e JOIN nodes s ON s.id = e.src JOIN nodes d ON d.id = e.dst"  # noqa: S608
                f" WHERE (e.src IN ({ph}) OR e.dst IN ({ph}))"
                " AND e.archived_at IS NULL AND e.valid_to IS NULL"
                " AND s.archived_at IS NULL AND d.archived_at IS NULL"
                f" AND {vis_e} AND {vis_s} AND {vis_d}"
                " ORDER BY e.weight DESC, e.created_at, e.id",
                (*node_ids, *node_ids, *p_e, *p_s, *p_d),
            ).fetchall()
            return self._edges_from_rows(rows)

    # ------------------------------------------------------------------ vectors (SPEC 6.1)

    def put_vector(
        self, record_id: str, facet: str, *, model: str, text: str, values: Sequence[float]
    ) -> bool:
        """Store the unit-normalised vector of one facet, stamped with the model,
        the dimension and the hash of the embedded text (replacing any older one).

        The embed job reads a text, calls the model, then writes: whatever changed
        in between is checked here, in the write's own transaction. When the record
        is gone or no longer live (forgotten, archived, closed), or its current
        text is not the embedded one, nothing is written and False is returned (P5:
        no vector derived from forgotten content; no stale vector kept for good)."""
        blob = vectors.pack(values)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock, self._transaction():
            current = self._current_facet_text(record_id, facet)
            if current is None or hashlib.sha256(current.encode("utf-8")).hexdigest() != digest:
                return False
            self._execute(
                "INSERT INTO vectors (record_id, facet, model, dimension, text_sha256, vector,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (record_id, facet) DO UPDATE SET model = excluded.model,"
                " dimension = excluded.dimension, text_sha256 = excluded.text_sha256,"
                " vector = excluded.vector, created_at = excluded.created_at",
                (record_id, facet, model, len(values), digest, blob, self.now()),
            )
        return True

    def _current_facet_text(self, record_id: str, facet: str) -> str | None:
        """What the facet of a LIVE record embeds now, or None."""
        if facet == TEXT:
            row = self._execute(
                "SELECT f.statement FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"
                " WHERE f.id = ? AND f.archived_at IS NULL AND f.valid_to IS NULL"
                " AND (f.node_id IS NULL OR n.archived_at IS NULL)",
                (record_id,),
            ).fetchone()
            return None if row is None else str(row[0])
        row = self._execute(
            "SELECT * FROM nodes WHERE id = ? AND archived_at IS NULL", (record_id,)
        ).fetchone()
        if row is None:
            return None
        node = self._nodes_from_rows([row])[0]
        if facet == NAME:
            return node_name_text(node)
        if facet == SUMMARY:
            return node.summary or None
        return None

    def _drop_vector(self, record_id: str, facet: str | None = None) -> int:
        if facet is None:
            cursor = self._execute("DELETE FROM vectors WHERE record_id = ?", (record_id,))
        else:
            cursor = self._execute(
                "DELETE FROM vectors WHERE record_id = ? AND facet = ?", (record_id, facet)
            )
        return int(cursor.rowcount)

    def pending_vectors(self, model: str, *, limit: int) -> list[PendingVector]:
        """Live facets with no vector of ``model`` (missing, or stale after a model
        change), oldest records first: what the background ``embed`` job computes."""
        if limit <= 0:
            return []
        out: list[PendingVector] = []
        with self._lock:
            facts = self._execute(
                "SELECT f.id, f.statement FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"
                " LEFT JOIN vectors v ON v.record_id = f.id AND v.facet = ? AND v.model = ?"
                " WHERE f.archived_at IS NULL AND f.valid_to IS NULL"
                " AND (f.node_id IS NULL OR n.archived_at IS NULL) AND v.record_id IS NULL"
                " ORDER BY f.id LIMIT ?",
                (TEXT, model, limit),
            ).fetchall()
            out.extend(PendingVector(r["id"], TEXT, r["statement"]) for r in facts)
            rows = self._execute(
                "SELECT n.* FROM nodes n WHERE n.archived_at IS NULL AND ("
                " NOT EXISTS (SELECT 1 FROM vectors v WHERE v.record_id = n.id AND v.facet = ?"
                " AND v.model = ?) OR (n.summary != '' AND NOT EXISTS (SELECT 1 FROM vectors v"
                " WHERE v.record_id = n.id AND v.facet = ? AND v.model = ?)))"
                " ORDER BY n.id LIMIT ?",
                (NAME, model, SUMMARY, model, limit),
            ).fetchall()
            current = (
                {
                    (r["record_id"], r["facet"])
                    for r in self._execute(
                        f"SELECT record_id, facet FROM vectors WHERE model = ?"  # noqa: S608
                        f" AND record_id IN ({_ph([r['id'] for r in rows])})",
                        (model, *[r["id"] for r in rows]),
                    )
                }
                if rows
                else set()
            )
            for node in self._nodes_from_rows(rows):
                if (node.id, NAME) not in current:
                    out.append(PendingVector(node.id, NAME, node_name_text(node)))
                if node.summary and (node.id, SUMMARY) not in current:
                    out.append(PendingVector(node.id, SUMMARY, node.summary))
        return out[:limit]

    def vector_stats(self, model: str) -> dict[str, int]:
        """``total`` rows, ``current`` (of ``model``), ``stale`` (another model, never
        compared) and ``missing`` (live facets with no vector at all)."""
        with self._lock:
            total = int(self._execute("SELECT count(*) FROM vectors").fetchone()[0])
            current = int(
                self._execute("SELECT count(*) FROM vectors WHERE model = ?", (model,)).fetchone()[
                    0
                ]
            )
            missing = int(
                self._execute(
                    "SELECT (SELECT count(*) FROM facts f LEFT JOIN nodes n ON n.id = f.node_id"
                    " WHERE f.archived_at IS NULL AND f.valid_to IS NULL"
                    " AND (f.node_id IS NULL OR n.archived_at IS NULL) AND NOT EXISTS"
                    " (SELECT 1 FROM vectors v WHERE v.record_id = f.id AND v.facet = ?))"
                    " + (SELECT count(*) FROM nodes n WHERE n.archived_at IS NULL AND NOT EXISTS"
                    " (SELECT 1 FROM vectors v WHERE v.record_id = n.id AND v.facet = ?))"
                    " + (SELECT count(*) FROM nodes n WHERE n.archived_at IS NULL"
                    " AND n.summary != '' AND NOT EXISTS"
                    " (SELECT 1 FROM vectors v WHERE v.record_id = n.id AND v.facet = ?))",
                    (TEXT, NAME, SUMMARY),
                ).fetchone()[0]
            )
        return {"total": total, "current": current, "stale": total - current, "missing": missing}

    def vector_dimension(self, model: str) -> int | None:
        """The dimension stored for ``model``, or None before its first vector."""
        with self._lock:
            row = self._execute(
                "SELECT dimension FROM vectors WHERE model = ? LIMIT 1", (model,)
            ).fetchone()
        return None if row is None else int(row[0])

    def vector_rows(
        self, view: Visibility, *, facets: Sequence[str], model: str
    ) -> list[VectorRow]:
        """The vectors of ``model`` for ``facets`` of the live records ``view`` sees,
        filtered in SQL before anything is compared (P9). Ordered by record id."""
        if view.is_empty or not facets:
            return []
        out: list[VectorRow] = []
        with self._lock:
            if TEXT in facets:
                vis, params = _visible(view, "f.id")
                rows = self._execute(
                    "SELECT v.record_id, v.facet, v.dimension, v.vector FROM vectors v"  # noqa: S608
                    " JOIN facts f ON f.id = v.record_id LEFT JOIN nodes n ON n.id = f.node_id"
                    " WHERE v.facet = ? AND v.model = ? AND f.archived_at IS NULL"
                    " AND f.valid_to IS NULL AND (f.node_id IS NULL OR n.archived_at IS NULL)"
                    f" AND {vis} ORDER BY v.record_id",
                    (TEXT, model, *params),
                ).fetchall()
                out.extend(VectorRow(r[0], r[1], int(r[2]), bytes(r[3])) for r in rows)
            node_facets = [f for f in facets if f in (NAME, SUMMARY)]
            if node_facets:
                vis, params = _visible(view, "n.id")
                rows = self._execute(
                    "SELECT v.record_id, v.facet, v.dimension, v.vector FROM vectors v"  # noqa: S608
                    " JOIN nodes n ON n.id = v.record_id"
                    f" WHERE v.facet IN ({_ph(node_facets)}) AND v.model = ?"
                    f" AND n.archived_at IS NULL AND {vis} ORDER BY v.record_id, v.facet",
                    (*node_facets, model, *params),
                ).fetchall()
                out.extend(VectorRow(r[0], r[1], int(r[2]), bytes(r[3])) for r in rows)
        return out

    # ------------------------------------------------------------------ proposals (SPEC 4.4b)

    def add_proposal(
        self,
        *,
        scopes: Iterable[str],
        kind: str,
        node_id: str | None,
        value: str,
        source_id: str | None = None,
        excerpt: tuple[int, int] | None = None,
        confidence: float | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Proposal:
        """Record a proposal with the scope set of its source. The same pending
        proposal (kind, node, normalised value) is recorded once: the existing
        one is returned. Only a fact proposal may have no node (a free speaker
        fact, SPEC 4.4b); its ``payload`` holds the rest of the fact."""
        if kind not in PROPOSAL_KINDS:
            raise ValueError(f"unknown proposal kind {kind!r}")
        if node_id is None and kind != FACT:
            raise ValueError(f"a {kind} proposal needs a node")
        text = " ".join(value.split())
        norm = normalize(text)
        if not norm:
            raise ValueError("a proposal needs a value")
        extra = {k: v for k, v in (payload or {}).items() if k in FACT_PAYLOAD_KEYS}
        with self._lock, self._transaction():
            if node_id is not None:
                self.get_node(node_id)
            row = self._execute(
                "SELECT id FROM proposals WHERE node_id IS ? AND kind = ? AND value_norm = ?"
                " AND status = ?",
                (node_id, kind, norm, PENDING),
            ).fetchone()
            if row is not None:
                return self.get_proposal(row["id"])
            proposal_id = new_id("prop")
            start, end = excerpt if excerpt is not None else (None, None)
            self._execute(
                "INSERT INTO proposals (id, kind, node_id, value, value_norm, source_id,"
                " excerpt_start, excerpt_end, confidence, status, created_at, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proposal_id,
                    kind,
                    node_id,
                    text,
                    norm,
                    source_id,
                    start,
                    end,
                    confidence,
                    PENDING,
                    self.now(),
                    json.dumps(extra, ensure_ascii=False) if extra else None,
                ),
            )
            self._set_scopes(proposal_id, scopes)
            return self.get_proposal(proposal_id)

    def get_proposal(self, proposal_id: str) -> Proposal:
        """Any proposal by id, whatever its status or scopes (internal use)."""
        with self._lock:
            row = self._execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
            if row is None:
                raise NotFound(proposal_id)
            return self._proposal_from_row(row, self.scopes_of(proposal_id))

    def proposals(
        self, node_ids: Sequence[str], view: Visibility, *, status: str = PENDING
    ) -> list[Proposal]:
        """The proposals about ``node_ids`` visible in ``view`` (P9), oldest first."""
        if not node_ids or view.is_empty:
            return []
        with self._lock:
            vis, params = _visible(view, "p.id")
            rows = self._execute(
                f"SELECT p.* FROM proposals p WHERE p.node_id IN ({_ph(node_ids)})"  # noqa: S608
                f" AND p.status = ? AND {vis} ORDER BY p.created_at, p.id",
                (*node_ids, status, *params),
            ).fetchall()
            scopes = self._scopes_map([r["id"] for r in rows])
            return [self._proposal_from_row(r, scopes.get(r["id"], ())) for r in rows]

    def pending_proposals(self, view: Visibility, *, kind: str | None = None) -> list[Proposal]:
        """Every pending proposal visible in ``view`` (P9), of ``kind`` when given,
        oldest first: the only place a fact proposal without a node shows."""
        if view.is_empty:
            return []
        with self._lock:
            vis, params = _visible(view, "p.id")
            kind_sql, kind_params = (" AND p.kind = ?", (kind,)) if kind else ("", ())
            rows = self._execute(
                f"SELECT p.* FROM proposals p WHERE p.status = ?{kind_sql} AND {vis}"  # noqa: S608
                " ORDER BY p.created_at, p.id",
                (PENDING, *kind_params, *params),
            ).fetchall()
            scopes = self._scopes_map([r["id"] for r in rows])
            return [self._proposal_from_row(r, scopes.get(r["id"], ())) for r in rows]

    def decide_proposal(self, proposal_id: str, *, accept: bool) -> Proposal:
        """Accept (apply it: an alias proposal adds the alias to its node; a fact
        proposal writes an explicit fact that keeps its provenance, SPEC 4.4b) or
        reject (only the status changes). A decided proposal raises
        ``ProposalDecided``."""
        with self._lock, self._transaction():
            proposal = self.get_proposal(proposal_id)
            if proposal.status != PENDING:
                raise ProposalDecided(proposal_id)
            if accept and proposal.kind == MERGE:
                # SPEC 6.1: a merge belongs to the dream and its review queue, never applied here
                raise ProposalNotApplicable(proposal_id)
            applied: str | None = None
            if accept and proposal.kind == FACT:
                applied = self._apply_fact_proposal(proposal)
            elif accept:
                assert proposal.node_id is not None  # only a fact proposal goes without
                self.update_node(proposal.node_id, add_aliases=[proposal.value])
            self._execute(
                "UPDATE proposals SET status = ?, decided_at = ?, applied_id = ? WHERE id = ?",
                (ACCEPTED if accept else REJECTED, self.now(), applied, proposal_id),
            )
            return self.get_proposal(proposal_id)

    def _apply_fact_proposal(self, proposal: Proposal) -> str:
        """An explicit fact (a human vouched for it, P6) that keeps where it came
        from: the source and span, the part's provenance, the proposal. Like
        ``remember``, it closes nothing."""
        data = proposal.payload
        excerpt = (
            (proposal.excerpt_start, proposal.excerpt_end)
            if proposal.excerpt_start is not None and proposal.excerpt_end is not None
            else None
        )
        confidence = data.get("confidence")
        fact = self.add_fact(
            scopes=proposal.scopes,
            statement=proposal.value,
            origin="explicit",
            node_id=proposal.node_id,
            source_id=proposal.source_id,
            excerpt=excerpt,
            attribute=data.get("attribute"),
            value=data.get("value"),
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            confidence=float(confidence) if confidence is not None else 1.0,
            flags=[str(f) for f in data.get("flags") or ()],
            language=data.get("language"),
            provenance=data.get("provenance"),
            proposal_id=proposal.id,
        )
        return fact.id

    def merge_proposed(self, a: str, b: str) -> bool:
        """Is a merge of these two nodes already proposed, in either direction?"""
        with self._lock:
            row = self._execute(
                "SELECT 1 FROM proposals WHERE kind = ? AND status = ?"
                " AND ((node_id = ? AND value = ?) OR (node_id = ? AND value = ?)) LIMIT 1",
                (MERGE, PENDING, a, b, b, a),
            ).fetchone()
        return row is not None

    def _forget_proposals(self, column: str, record_id: str, *, kind: str | None = None) -> int:
        """Hard-delete the proposals about a node or grounded in a source (P5); with
        ``kind``, only those (a merge proposal names its other node in ``value``)."""
        extra, params = (" AND kind = ?", (record_id, kind)) if kind else ("", (record_id,))
        ids = [
            r["id"]
            for r in self._execute(
                f"SELECT id FROM proposals WHERE {column} = ?{extra}",  # noqa: S608
                params,
            )
        ]
        for proposal_id in ids:
            self._drop_scopes(proposal_id)
            self._execute("DELETE FROM proposals WHERE id = ?", (proposal_id,))
        return len(ids)

    @staticmethod
    def _proposal_from_row(row: Any, scopes: tuple[str, ...]) -> Proposal:
        return Proposal(
            id=row["id"],
            kind=row["kind"],
            node_id=row["node_id"],
            value=row["value"],
            source_id=row["source_id"],
            excerpt_start=row["excerpt_start"],
            excerpt_end=row["excerpt_end"],
            confidence=row["confidence"],
            status=row["status"],
            scopes=scopes,
            created_at=row["created_at"],
            decided_at=row["decided_at"],
            payload=json.loads(row["payload"]) if row["payload"] else {},
            applied_id=row["applied_id"],
        )

    # ------------------------------------------------------------------ probe sets (SPEC 11.2)

    def create_probe_set(
        self,
        *,
        scope: str,
        name: str,
        budget_tokens: int,
        probes: Sequence[tuple[str, Sequence[str]]],
    ) -> ProbeSet:
        """Store a probe set. Every expected id must be a fact or node visible from
        ``scope`` (scope and ancestors); anything else, unknown ids included, raises
        the same ``ScopeMismatch`` (no existence leak, P9)."""
        scope = validate_scope(scope)
        if any(not ids for _query, ids in probes):
            raise ValueError("every probe expects at least one record")
        view = Visibility.of_scope(scope)
        set_id = new_id("pset")
        with self._lock, self._transaction():
            wanted = list(dict.fromkeys(i for _query, ids in probes for i in ids))
            visible = set(self.facts_by_id([i for i in wanted if kind_of(i) == "fact"], view))
            visible |= set(self.nodes_by_id([i for i in wanted if kind_of(i) == "node"], view))
            for record_id in wanted:
                if record_id not in visible:
                    raise ScopeMismatch(
                        f"expected record not visible from the set's scope: {record_id}"
                    )
            self._execute(
                "INSERT INTO probe_sets (id, scope, name, budget_tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (set_id, scope, name, budget_tokens, self.now()),
            )
            for position, (query, ids) in enumerate(probes):
                probe_id = new_id("probe")
                self._execute(
                    "INSERT INTO probes (id, set_id, position, query) VALUES (?, ?, ?, ?)",
                    (probe_id, set_id, position, query),
                )
                for rank, record_id in enumerate(dict.fromkeys(ids)):
                    self._execute(
                        "INSERT INTO probe_expected (probe_id, record_id, position)"
                        " VALUES (?, ?, ?)",
                        (probe_id, record_id, rank),
                    )
            return self.get_probe_set(set_id)

    def get_probe_set(self, set_id: str) -> ProbeSet:
        """A probe set with its probes, whatever the caller may see (the route checks
        the token's scopes)."""
        with self._lock:
            row = self._execute("SELECT * FROM probe_sets WHERE id = ?", (set_id,)).fetchone()
            if row is None:
                raise NotFound(set_id)
            probes = self._execute(
                "SELECT id, query FROM probes WHERE set_id = ? ORDER BY position", (set_id,)
            ).fetchall()
            expected: dict[str, list[str]] = {p["id"]: [] for p in probes}
            for e in self._execute(
                "SELECT e.probe_id, e.record_id FROM probe_expected e JOIN probes p"
                " ON p.id = e.probe_id WHERE p.set_id = ? ORDER BY e.probe_id, e.position",
                (set_id,),
            ):
                expected[e["probe_id"]].append(e["record_id"])
            return ProbeSet(
                id=row["id"],
                scope=row["scope"],
                name=row["name"],
                budget_tokens=int(row["budget_tokens"]),
                created_at=row["created_at"],
                probes=tuple(Probe(p["id"], p["query"], tuple(expected[p["id"]])) for p in probes),
            )

    def probe_sets(self, scope: str) -> list[tuple[ProbeSet, int]]:
        """The sets of exactly ``scope`` (without their probes) and their sizes, oldest first."""
        with self._lock:
            rows = self._execute(
                "SELECT s.*, (SELECT count(*) FROM probes p WHERE p.set_id = s.id) AS size"
                " FROM probe_sets s WHERE s.scope = ? ORDER BY s.id",
                (scope,),
            ).fetchall()
            return [
                (
                    ProbeSet(
                        r["id"], r["scope"], r["name"], int(r["budget_tokens"]), r["created_at"]
                    ),
                    int(r["size"]),
                )
                for r in rows
            ]

    def delete_probe_set(self, set_id: str) -> int:
        with self._lock, self._transaction():
            self._execute(
                "DELETE FROM probe_expected WHERE probe_id IN"
                " (SELECT id FROM probes WHERE set_id = ?)",
                (set_id,),
            )
            self._execute("DELETE FROM probes WHERE set_id = ?", (set_id,))
            return int(self._execute("DELETE FROM probe_sets WHERE id = ?", (set_id,)).rowcount)

    def supersession_chain(self, fact_id: str, *, limit: int = 16) -> list[str]:
        """The facts that replaced ``fact_id``, in order (P7: ``superseded_by``),
        bounded so that a cycle cannot loop."""
        chain: list[str] = []
        current = fact_id
        with self._lock:
            for _ in range(limit):
                row = self._execute(
                    "SELECT superseded_by FROM facts WHERE id = ?", (current,)
                ).fetchone()
                if row is None or row["superseded_by"] is None or row["superseded_by"] in chain:
                    break
                current = row["superseded_by"]
                chain.append(current)
        return chain

    def successors(self, record_id: str, *, limit: int = 16) -> list[str]:
        """What replaced a record, in order: for a fact, the fact that superseded it
        or a dream consolidated it into, and so on; for a node, the node a dream
        merged it into, and so on (SPEC 11.2). Bounded, so a cycle cannot loop."""
        kind = kind_of(record_id)
        if kind == "fact":
            sql = (
                "SELECT coalesce(superseded_by, consolidated_into) AS next FROM facts WHERE id = ?"
            )
        elif kind == "node":
            sql = "SELECT merged_into AS next FROM nodes WHERE id = ?"
        else:
            return []
        chain: list[str] = []
        current = record_id
        with self._lock:
            for _ in range(limit):
                row = self._execute(sql, (current,)).fetchone()
                if row is None or row["next"] is None or row["next"] in chain:
                    break
                current = row["next"]
                chain.append(current)
        return chain

    def _reroot(self, fact_id: str) -> None:
        """P5 (SPEC 5.4): the group a forgotten first fact led is led by its next
        member; no ``same_as`` names a forgotten fact."""
        members = [
            r["id"]
            for r in self._execute(
                "SELECT id FROM facts WHERE same_as = ? ORDER BY created_at, id", (fact_id,)
            )
        ]
        if not members:
            return
        head, rest = members[0], members[1:]
        self._execute("UPDATE facts SET same_as = NULL WHERE id = ?", (head,))
        if rest:
            self._execute(
                f"UPDATE facts SET same_as = ? WHERE id IN ({_ph(rest)})",  # noqa: S608
                (head, *rest),
            )

    def _forget_expected(self, record_id: str) -> int:
        """Delete every probe that expected a forgotten record, question included
        (P5): a question may carry the forgotten content, and a probe that lost an
        answer no longer measures the same thing (SPEC 11.2). Returns how many."""
        touched = [
            r["probe_id"]
            for r in self._execute(
                "SELECT probe_id FROM probe_expected WHERE record_id = ?", (record_id,)
            )
        ]
        for probe_id in touched:
            self._execute("DELETE FROM probe_expected WHERE probe_id = ?", (probe_id,))
            self._execute("DELETE FROM probes WHERE id = ?", (probe_id,))
        return len(touched)

    # ------------------------------------------------------------------ dream (SPEC 4.4, 5.6)

    def revisions(self, record_ids: Sequence[str]) -> dict[str, int]:
        """The current ``rev`` of each node, fact or edge among ``record_ids`` that
        exists (SPEC 4.4); a forgotten or unknown id is simply absent."""
        by_kind: dict[str, list[str]] = {}
        for record_id in dict.fromkeys(record_ids):
            kind = kind_of(record_id)
            if kind in ARCHIVABLE:
                by_kind.setdefault(kind, []).append(record_id)
        out: dict[str, int] = {}
        with self._lock:
            for kind, ids in by_kind.items():
                rows = self._execute(
                    f"SELECT id, rev FROM {_TABLES[kind]} WHERE id IN ({_ph(ids)})",  # noqa: S608
                    ids,
                ).fetchall()
                out.update({r["id"]: int(r["rev"]) for r in rows})
        return out

    def node_ids_after(self, view: Visibility, after: str, limit: int) -> list[str]:
        """Live nodes the view sees with an id above ``after``, in id order (ids
        sort by creation, SPEC 7): what a dream walks, unit by unit."""
        if view.is_empty or limit <= 0:
            return []
        with self._lock:
            vis, params = _visible(view, "n.id")
            rows = self._execute(
                f"SELECT n.id FROM nodes n WHERE n.archived_at IS NULL AND n.id > ? AND {vis}"  # noqa: S608
                " ORDER BY n.id LIMIT ?",
                (after, *params, int(limit)),
            ).fetchall()
            return [r["id"] for r in rows]

    def live_proposed_edges(self, view: Visibility) -> list[Edge]:
        """The live ``proposed`` edges the view sees (what decay works on, SPEC 5.3)."""
        if view.is_empty:
            return []
        with self._lock:
            vis, params = _visible(view, "e.id")
            rows = self._execute(
                f"SELECT e.* FROM edges e WHERE e.status = ? AND e.archived_at IS NULL"  # noqa: S608
                f" AND e.valid_to IS NULL AND {vis} ORDER BY e.id",
                (PROPOSED, *params),
            ).fetchall()
            return self._edges_from_rows(rows)

    def topic_named(self, name: str, view: Visibility) -> Node | None:
        """A live ``topic`` node of that (normalised) name the view sees."""
        norm = normalize(name)
        if not norm:
            return None
        with self._lock:
            vis, params = _visible(view, "n.id")
            row = self._execute(
                f"SELECT n.* FROM nodes n WHERE n.kind = 'topic' AND n.archived_at IS NULL"  # noqa: S608
                f" AND n.name_norm = ? AND {vis} ORDER BY n.id LIMIT 1",
                (norm, *params),
            ).fetchone()
            return None if row is None else self._node_from_row(row)

    def remove_aliases(self, node_id: str, aliases: Iterable[str]) -> Node:
        """Remove aliases from a node (undoing a merge, SPEC 5.6)."""
        with self._lock, self._transaction():
            self.get_node(node_id)
            removed = 0
            for alias in aliases:
                removed += self._execute(
                    "DELETE FROM node_aliases WHERE node_id = ? AND alias_norm = ?",
                    (node_id, normalize(alias)),
                ).rowcount
            if removed:
                self._drop_vector(node_id, NAME)
                self._execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (self.now(), node_id))
            self._reindex_node(node_id)
            return self.get_node(node_id)

    def set_summary(
        self, node_id: str, *, summary: str, language: str | None, flags: Iterable[str]
    ) -> Node:
        """Put back exactly a summary, its language and the node's flags (undo)."""
        with self._lock, self._transaction():
            before = self.get_node(node_id)
            self._execute(
                "UPDATE nodes SET summary = ?, language = ?, flags = ?, updated_at = ?"
                " WHERE id = ?",
                (summary, language, ",".join(sorted(set(flags))), self.now(), node_id),
            )
            if summary != before.summary:
                self._drop_vector(node_id, SUMMARY)
            self._reindex_node(node_id)
            return self.get_node(node_id)

    def consolidate_facts(
        self, fact_ids: Sequence[str], *, into: str, valid_to: str | None = None
    ) -> None:
        """Close facts as consolidated into another one (SPEC 5.6): not a change
        of value, so ``superseded_by`` is left alone."""
        with self._lock, self._transaction():
            self.get_fact(into)
            now = self.now()
            for fact_id in fact_ids:
                fact = self.get_fact(fact_id)
                self._execute(
                    "UPDATE facts SET valid_to = coalesce(valid_to, ?), consolidated_into = ?,"
                    " updated_at = ? WHERE id = ?",
                    (valid_to or now, into, now, fact_id),
                )
                if fact.node_id is not None:
                    self._execute(
                        "UPDATE nodes SET updated_at = ? WHERE id = ?", (now, fact.node_id)
                    )
                self._refresh_hidden(fact_id, "fact")

    def reopen_fact(
        self,
        fact_id: str,
        *,
        valid_to: str | None,
        superseded_by: str | None,
        consolidated_into: str | None,
    ) -> Fact:
        """Put back exactly how a fact was closed, or that it was not (undo)."""
        with self._lock, self._transaction():
            fact = self.get_fact(fact_id)
            now = self.now()
            self._execute(
                "UPDATE facts SET valid_to = ?, superseded_by = ?, consolidated_into = ?,"
                " updated_at = ? WHERE id = ?",
                (valid_to, superseded_by, consolidated_into, now, fact_id),
            )
            if fact.node_id is not None:
                self._execute("UPDATE nodes SET updated_at = ? WHERE id = ?", (now, fact.node_id))
            self._refresh_hidden(fact_id, "fact")
            return self.get_fact(fact_id)

    def set_fact_flags(
        self, fact_id: str, flags: Iterable[str], *, conflicts_with: str | None
    ) -> Fact:
        """Put back exactly a fact's flags and the fact it conflicts with (undo)."""
        with self._lock, self._transaction():
            self.get_fact(fact_id)
            self._execute(
                "UPDATE facts SET flags = ?, conflicts_with = ?, updated_at = ? WHERE id = ?",
                (",".join(sorted(set(flags))), conflicts_with, self.now(), fact_id),
            )
            return self.get_fact(fact_id)

    def merge_node(self, node_id: str, *, into: str) -> Node:
        """Archive a node as merged into another (SPEC 5.6): it stays, with
        ``merged_into``, and leaves recall like any archived node."""
        with self._lock, self._transaction():
            self.get_node(into)
            self.get_node(node_id)
            now = self.now()
            self._execute(
                "UPDATE nodes SET merged_into = ?, archived_at = coalesce(archived_at, ?),"
                " updated_at = ? WHERE id = ?",
                (into, now, now, node_id),
            )
            self._refresh_hidden(node_id, "node")
            return self.get_node(node_id)

    def unmerge_node(self, node_id: str, *, archived_at: str | None) -> Node:
        """Undo ``merge_node``: no ``merged_into``, the archive state put back."""
        with self._lock, self._transaction():
            self.get_node(node_id)
            self._execute(
                "UPDATE nodes SET merged_into = NULL, archived_at = ?, updated_at = ? WHERE id = ?",
                (archived_at, self.now(), node_id),
            )
            self._refresh_hidden(node_id, "node")
            return self.get_node(node_id)

    def move_edge(self, edge_id: str, *, src: str, dst: str) -> Edge:
        """Give an edge other ends (a merge moves the absorbed node's edges)."""
        with self._lock, self._transaction():
            self.get_edge(edge_id)
            self.get_node(src)
            self.get_node(dst)
            self._execute(
                "UPDATE edges SET src = ?, dst = ?, updated_at = ? WHERE id = ?",
                (src, dst, self.now(), edge_id),
            )
            return self.get_edge(edge_id)

    def set_edge_life(
        self,
        edge_id: str,
        *,
        weight: float,
        decayed_at: str | None,
        valid_to: str | None,
    ) -> Edge:
        """Set exactly an edge's weight, last decay and end (decay, and its undo)."""
        with self._lock, self._transaction():
            self.get_edge(edge_id)
            self._execute(
                "UPDATE edges SET weight = ?, decayed_at = ?, valid_to = ?, updated_at = ?"
                " WHERE id = ?",
                (weight, decayed_at, valid_to, self.now(), edge_id),
            )
            return self.get_edge(edge_id)

    def edge_decayed_at(self, edge_id: str) -> str | None:
        with self._lock:
            row = self._execute("SELECT decayed_at FROM edges WHERE id = ?", (edge_id,)).fetchone()
            if row is None:
                raise NotFound(edge_id)
            return None if row["decayed_at"] is None else str(row["decayed_at"])

    def node_fact_ids(self, node_id: str) -> list[str]:
        """Every fact of a node, live or not, archived or not (a merge moves them all)."""
        with self._lock:
            return self._facts_of_node(node_id)

    def node_edge_ids(self, node_id: str) -> list[str]:
        """Every edge touching a node, live or not."""
        with self._lock:
            return [
                r["id"]
                for r in self._execute(
                    "SELECT id FROM edges WHERE src = ? OR dst = ? ORDER BY id", (node_id, node_id)
                )
            ]

    def merge_proposal_between(self, a: str, b: str) -> Proposal | None:
        """The pending merge proposal pairing two nodes, either way round."""
        with self._lock:
            row = self._execute(
                "SELECT id FROM proposals WHERE kind = ? AND status = ?"
                " AND ((node_id = ? AND value = ?) OR (node_id = ? AND value = ?))"
                " ORDER BY created_at, id LIMIT 1",
                (MERGE, PENDING, a, b, b, a),
            ).fetchone()
            return None if row is None else self.get_proposal(row["id"])

    def merge_rejected(self, a: str, b: str) -> bool:
        """Did a reviewer reject a merge proposal pairing these two nodes?"""
        with self._lock:
            row = self._execute(
                "SELECT 1 FROM proposals WHERE kind = ? AND status = ?"
                " AND ((node_id = ? AND value = ?) OR (node_id = ? AND value = ?)) LIMIT 1",
                (MERGE, REJECTED, a, b, b, a),
            ).fetchone()
        return row is not None

    def remove_pending_proposal(self, proposal_id: str) -> bool:
        """Delete a proposal a dream created, while nobody decided it (SPEC 5.6:
        undoing a run removes the merge proposals its ``unsure`` verdicts left)."""
        with self._lock, self._transaction():
            row = self._execute(
                "SELECT status FROM proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
            if row is None or row["status"] != PENDING:
                return False
            self._forget_proposals("id", proposal_id)
            return True

    def set_proposal_status(self, proposal_id: str, status: str) -> Proposal:
        """Set a proposal's status (a merge proposal follows its dream operation)."""
        if status not in (PENDING, ACCEPTED, REJECTED):
            raise ValueError(f"unknown proposal status {status!r}")
        with self._lock, self._transaction():
            self.get_proposal(proposal_id)
            self._execute(
                "UPDATE proposals SET status = ?, decided_at = ? WHERE id = ?",
                (status, None if status == PENDING else self.now(), proposal_id),
            )
            return self.get_proposal(proposal_id)

    def merge_candidates(self, node_id: str, view: Visibility) -> list[str]:
        """The nodes a pending merge proposal visible in the view pairs with ``node_id``."""
        with self._lock:
            vis, params = _visible(view, "p.id")
            rows = self._execute(
                "SELECT p.node_id, p.value FROM proposals p WHERE p.kind = ? AND p.status = ?"  # noqa: S608
                f" AND (p.node_id = ? OR p.value = ?) AND {vis} ORDER BY p.created_at, p.id",
                (MERGE, PENDING, node_id, node_id, *params),
            ).fetchall()
        out: list[str] = []
        for row in rows:
            other = row["value"] if row["node_id"] == node_id else row["node_id"]
            if other and other not in out:
                out.append(other)
        return out

    def remove_created(self, record_id: str) -> None:
        """Delete a node, fact or edge a dream operation created, when undoing it
        (SPEC 5.6: undo deletes what that operation created - nothing else). The
        journal is left alone: it is the journal doing the undo."""
        kind = self._kind(record_id, ARCHIVABLE)
        with self._lock, self._transaction():
            removed: list[str] = []
            if kind == "node":
                self._forget_node(record_id, removed)
            elif kind == "fact":
                self._forget_facts([record_id], removed)
            else:
                self._forget_edges([record_id], removed)

    def probe_sets_within(self, scope: str) -> list[ProbeSet]:
        """Every probe set of ``scope`` or of a scope below it, with its probes
        (what a dream on ``scope`` is measured by, SPEC 5.6 rule 9)."""
        with self._lock:
            rows = self._execute("SELECT id, scope FROM probe_sets ORDER BY id").fetchall()
            return [self.get_probe_set(r["id"]) for r in rows if within(r["scope"], scope)]

    # ------------------------------------------------------------------ archive / restore / forget

    def archive(self, record_id: str) -> str:
        """Reversible: the record leaves recall, expansion and profiles, but stays."""
        kind = self._kind(record_id, ARCHIVABLE)
        with self._lock, self._transaction():
            table = _TABLES[kind]
            if not self._exists(table, record_id):
                raise NotFound(record_id)
            now = self.now()
            self._execute(
                f"UPDATE {table} SET archived_at = coalesce(archived_at, ?), updated_at = ?"  # noqa: S608
                " WHERE id = ?",
                (now, now, record_id),
            )
            self._refresh_hidden(record_id, kind)
        return kind

    def restore(self, record_id: str) -> str:
        kind = self._kind(record_id, ARCHIVABLE)
        with self._lock, self._transaction():
            table = _TABLES[kind]
            if not self._exists(table, record_id):
                raise NotFound(record_id)
            self._execute(
                f"UPDATE {table} SET archived_at = NULL, updated_at = ? WHERE id = ?",  # noqa: S608
                (self.now(), record_id),
            )
            self._refresh_hidden(record_id, kind)
        return kind

    def archived(self, view: Visibility) -> list[dict[str, str]]:
        with self._lock:
            out: list[dict[str, str]] = []
            for kind in ARCHIVABLE:
                vis, params = _visible(view, "t.id")
                rows = self._execute(
                    f"SELECT t.id, t.archived_at FROM {_TABLES[kind]} t"  # noqa: S608
                    f" WHERE t.archived_at IS NOT NULL AND {vis}"
                    " ORDER BY t.archived_at DESC, t.id",
                    params,
                ).fetchall()
                out.extend(
                    {"id": r["id"], "kind": kind, "archived_at": r["archived_at"]} for r in rows
                )
            return out

    def forget(self, record_id: str) -> dict[str, int]:
        """Irreversible hard delete of the record and everything hanging off it (P5).

        Returns counts per table. Followed by the housekeeping that makes the
        bytes unrecoverable: FTS optimise and WAL checkpoint (``secure_delete``
        is always on).
        """
        kind = self._kind(record_id, FORGETTABLE)
        with self._lock:
            with self._transaction():
                removed: list[str] = []
                if kind == "node":
                    counts = self._forget_node(record_id, removed)
                elif kind == "fact":
                    counts = self._forget_facts([record_id], removed)
                elif kind == "edge":
                    counts = self._forget_edges([record_id], removed)
                else:
                    counts = self._forget_source(record_id, removed)
                counts["dream_ops_wiped"] = self._wipe_journal(removed)
                counts["adjudications_deleted"] = self._forget_adjudications(removed)
            self._scrub()
        return counts

    def _wipe_journal(self, record_ids: Sequence[str]) -> int:
        """P5 (SPEC 5.6): the dream journal entries that mention a forgotten record
        lose their content (payload, the model's reason, before/after states) and
        the record's id; they can no longer be undone, and a queued one is
        dropped. Returns how many entries were wiped."""
        wiped: set[str] = set()
        for record_id in record_ids:
            ops = [
                r["op_id"]
                for r in self._execute(
                    "SELECT op_id FROM dream_op_records WHERE record_id = ?", (record_id,)
                )
            ]
            for op_id in ops:
                self._execute(
                    "UPDATE dream_ops SET payload = NULL, reason_given = NULL, snapshot = NULL,"
                    " changes = NULL, after_revs = NULL, forgotten = 1,"
                    " reason = CASE WHEN status = 'queued' THEN 'forgotten' ELSE reason END,"
                    " status = CASE WHEN status = 'queued' THEN 'dropped' ELSE status END"
                    " WHERE id = ?",
                    (op_id,),
                )
                wiped.add(op_id)
            self._execute("DELETE FROM dream_op_records WHERE record_id = ?", (record_id,))
        return len(wiped)

    def _forget_adjudications(self, record_ids: Sequence[str]) -> int:
        """P5 (SPEC 5.6): an adjudication's reason may quote what it read, so every
        adjudication that read a forgotten record is deleted."""
        deleted: set[str] = set()
        for record_id in record_ids:
            for row in self._execute(
                "SELECT adjudication_id FROM dream_adjudication_records WHERE record_id = ?",
                (record_id,),
            ).fetchall():
                deleted.add(row["adjudication_id"])
        for adjudication_id in deleted:
            self._execute("DELETE FROM dream_adjudications WHERE id = ?", (adjudication_id,))
        return len(deleted)

    def _forget_node(self, node_id: str, removed: list[str]) -> dict[str, int]:
        if not self._exists("nodes", node_id):
            raise NotFound(node_id)
        fact_ids = [
            r["id"] for r in self._execute("SELECT id FROM facts WHERE node_id = ?", (node_id,))
        ]
        edge_ids = [
            r["id"]
            for r in self._execute(
                "SELECT id FROM edges WHERE src = ? OR dst = ?", (node_id, node_id)
            )
        ]
        counts = _merge(
            self._forget_facts(fact_ids, removed), self._forget_edges(edge_ids, removed)
        )
        counts["proposals"] = self._forget_proposals("node_id", node_id)
        counts["proposals"] += self._forget_proposals("value", node_id, kind=MERGE)
        counts["vectors"] = counts.get("vectors", 0) + self._drop_vector(node_id)
        counts["probes_deleted"] = counts.get("probes_deleted", 0) + self._forget_expected(node_id)
        self._execute("DELETE FROM node_aliases WHERE node_id = ?", (node_id,))
        self._unindex(node_id)
        self._drop_scopes(node_id)
        # forget never brings back what it did not delete: merged nodes stay archived
        self._execute("UPDATE nodes SET merged_into = NULL WHERE merged_into = ?", (node_id,))
        self._execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        removed.append(node_id)
        counts["nodes"] = 1
        return counts

    def _forget_facts(self, fact_ids: Sequence[str], removed: list[str]) -> dict[str, int]:
        counts = {"facts": 0, "sources": 0, "vectors": 0, "probes_deleted": 0}
        for fact_id in fact_ids:
            row = self._execute("SELECT source_id FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if row is None:
                raise NotFound(fact_id)
            self._unindex(fact_id)
            self._drop_scopes(fact_id)
            counts["vectors"] += self._drop_vector(fact_id)
            counts["probes_deleted"] += self._forget_expected(fact_id)
            self._reroot(fact_id)
            # the originals of a forgotten consolidation stay closed (SPEC 5.6)
            self._execute(
                "UPDATE facts SET consolidated_into = NULL WHERE consolidated_into = ?", (fact_id,)
            )
            self._execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            removed.append(fact_id)
            counts["facts"] += 1
            if row["source_id"] is not None:
                counts["sources"] += self._delete_source_if_orphan(row["source_id"])
        return counts

    def _forget_edges(self, edge_ids: Sequence[str], removed: list[str]) -> dict[str, int]:
        counts = {"edges": 0}
        for edge_id in edge_ids:
            self._drop_scopes(edge_id)
            cursor = self._execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            if cursor.rowcount == 0:
                raise NotFound(edge_id)
            removed.append(edge_id)
            counts["edges"] += 1
        return counts

    def _forget_source(self, source_id: str, removed: list[str]) -> dict[str, int]:
        if not self._exists("sources", source_id):
            raise NotFound(source_id)
        fact_ids = [
            r["id"] for r in self._execute("SELECT id FROM facts WHERE source_id = ?", (source_id,))
        ]
        edge_ids = [
            r["id"] for r in self._execute("SELECT id FROM edges WHERE source_id = ?", (source_id,))
        ]
        counts = _merge(
            self._forget_facts(fact_ids, removed), self._forget_edges(edge_ids, removed)
        )
        counts["proposals"] = self._forget_proposals("source_id", source_id)
        self._drop_scopes(source_id)
        cursor = self._execute("DELETE FROM sources WHERE id = ?", (source_id,))
        counts["sources"] = counts.get("sources", 0) + int(cursor.rowcount)
        return counts

    def _delete_source_if_orphan(self, source_id: str) -> int:
        used = self._execute(
            "SELECT 1 FROM facts WHERE source_id = ?"
            " UNION ALL SELECT 1 FROM edges WHERE source_id = ?"
            " UNION ALL SELECT 1 FROM proposals WHERE source_id = ? LIMIT 1",
            (source_id, source_id, source_id),
        ).fetchone()
        if used is not None:
            return 0
        self._drop_scopes(source_id)
        return int(self._execute("DELETE FROM sources WHERE id = ?", (source_id,)).rowcount)

    def _drop_scopes(self, record_id: str) -> None:
        self._execute("DELETE FROM record_scopes WHERE record_id = ?", (record_id,))

    def _scrub(self) -> None:
        """Push deleted tokens out of the FTS segments and deleted pages out of the WAL."""
        self._execute("INSERT INTO lex (lex) VALUES ('optimize')")
        self.checkpoint("TRUNCATE")

    @staticmethod
    def _kind(record_id: str, allowed: Sequence[str]) -> str:
        kind = kind_of(record_id)
        if kind is None or kind not in allowed:
            raise NotFound(record_id)
        return kind

    def _exists(self, table: str, record_id: str) -> bool:
        row = self._execute(f"SELECT 1 FROM {table} WHERE id = ?", (record_id,)).fetchone()  # noqa: S608
        return row is not None

    # ------------------------------------------------------------------ lexical index

    def _reindex_marked_nodes(self) -> None:
        """Migration 11: rebuild every node's lexical document once (names and
        aliases only), then clear the mark - all in one transaction."""
        with self._transaction():
            if self._execute(
                "SELECT 1 FROM service_state WHERE key = ?", (REINDEX_NODES,)
            ).fetchone():
                for row in self._execute("SELECT id FROM nodes ORDER BY id").fetchall():
                    self._reindex_node(row["id"])
                self._execute("DELETE FROM service_state WHERE key = ?", (REINDEX_NODES,))

    def _reindex_node(self, node_id: str) -> None:
        # SPEC 6.1: names and aliases only - a summary is model-written text,
        # shown in handles, never ranked by recall
        row = self._execute("SELECT name FROM nodes WHERE id = ?", (node_id,)).fetchone()
        aliases = [
            r["alias"]
            for r in self._execute("SELECT alias FROM node_aliases WHERE node_id = ?", (node_id,))
        ]
        text = " ".join(part for part in (row["name"], *aliases) if part)
        self._unindex(node_id)
        self._index(node_id, "node", text, self.scopes_of(node_id))
        self._refresh_hidden(node_id, "node")

    def _index(self, doc_id: str, kind: str, text: str, scopes: Sequence[str]) -> None:
        ntok = len(tokenize(text))
        cursor = self._execute(
            "INSERT INTO lex_docs (doc_id, kind, ntok) VALUES (?, ?, ?)", (doc_id, kind, ntok)
        )
        rowid = cursor.lastrowid
        self._execute("INSERT INTO lex (rowid, text) VALUES (?, ?)", (rowid, text))
        for scope in scopes:
            self._execute("INSERT INTO lex_scopes (doc, scope) VALUES (?, ?)", (rowid, scope))
            self._execute(
                "INSERT INTO lex_scope_stats (scope, docs, tokens) VALUES (?, 1, ?)"
                " ON CONFLICT (scope) DO UPDATE SET docs = docs + 1,"
                " tokens = tokens + excluded.tokens",
                (scope, ntok),
            )

    def _unindex(self, doc_id: str) -> None:
        row = self._execute(
            "SELECT rowid, ntok FROM lex_docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return
        for scope_row in self._execute(
            "SELECT scope FROM lex_scopes WHERE doc = ?", (row["rowid"],)
        ):
            self._execute(
                "UPDATE lex_scope_stats SET docs = docs - 1, tokens = tokens - ? WHERE scope = ?",
                (row["ntok"], scope_row["scope"]),
            )
        self._execute("DELETE FROM lex_scopes WHERE doc = ?", (row["rowid"],))
        self._execute("DELETE FROM lex WHERE rowid = ?", (row["rowid"],))
        self._execute("DELETE FROM lex_docs WHERE rowid = ?", (row["rowid"],))

    def _refresh_hidden(self, record_id: str, kind: str) -> None:
        """Recompute the ``hidden`` flag of the documents ``record_id`` affects:
        the record itself and, for a node, every fact attached to it."""
        if kind == "edge":
            return
        doc_ids = [record_id]
        if kind == "node":
            doc_ids.extend(self._facts_of_node(record_id))
        for doc_id in doc_ids:
            self._execute(
                "UPDATE lex_docs SET hidden = ("
                " SELECT CASE WHEN f.id IS NOT NULL THEN"
                "  (f.archived_at IS NOT NULL OR f.valid_to IS NOT NULL"
                "   OR (f.node_id IS NOT NULL AND fn.archived_at IS NOT NULL))"
                " ELSE (n.archived_at IS NOT NULL) END"
                " FROM lex_docs d LEFT JOIN facts f ON f.id = d.doc_id"
                " LEFT JOIN nodes fn ON fn.id = f.node_id"
                " LEFT JOIN nodes n ON n.id = d.doc_id WHERE d.doc_id = ?)"
                " WHERE doc_id = ?",
                (doc_id, doc_id),
            )

    def postings(self, terms: Sequence[str], view: Visibility) -> list[Posting]:
        """Documents matching each term over the visible, live corpus only.

        The view is applied here, before any score exists (P9): only visible
        documents contribute to what the caller sees, including the statistics
        derived from these postings. Hidden documents (archived records, facts
        of archived nodes, closed facts) are excluded the same way. Term
        frequency is 1: a fact is one sentence, and repeating a word in it
        does not make it more relevant.
        """
        if not terms or view.is_empty:
            return []
        with self._lock:
            pred, params = view.sql("ls.scope")
            out: list[Posting] = []
            for term in dict.fromkeys(terms):
                match = '"' + term.replace('"', '""') + '"'
                # FTS5 drives (CROSS JOIN), then one index probe per row into
                # lex_docs and lex_scopes. A document in several visible scopes
                # joins once per scope; the dict below keeps it once.
                rows = self._execute(
                    "SELECT d.rowid AS rowid, d.doc_id AS doc_id, d.kind AS kind, d.ntok AS ntok"  # noqa: S608
                    " FROM lex CROSS JOIN lex_docs d ON d.rowid = lex.rowid"
                    f" CROSS JOIN lex_scopes ls ON ls.doc = d.rowid AND {pred}"
                    " WHERE lex MATCH ? AND d.hidden = 0",
                    (*params, match),
                ).fetchall()
                seen: dict[int, Posting] = {}
                for r in rows:
                    if r["rowid"] not in seen:
                        seen[r["rowid"]] = Posting(
                            term, r["rowid"], r["doc_id"], r["kind"], 1, r["ntok"]
                        )
                out.extend(seen[k] for k in sorted(seen))
            return out

    def corpus_stats(self, view: Visibility) -> CorpusStats:
        """Document count and average length over the visible corpus (P9).

        Read from per-scope aggregates, never by scanning: a document counts
        once per visible scope it belongs to, hidden documents included. Only
        visible scopes contribute.
        """
        if view.is_empty:
            return CorpusStats(0, 0.0)
        with self._lock:
            pred, params = view.sql("scope")
            row = self._execute(
                f"SELECT coalesce(sum(docs), 0) AS n, coalesce(sum(tokens), 0) AS t"  # noqa: S608
                f" FROM lex_scope_stats WHERE {pred}",
                params,
            ).fetchone()
            documents = int(row["n"])
            return CorpusStats(documents, float(row["t"]) / documents if documents else 0.0)

    def indexed_terms(self, doc_id: str) -> set[str]:
        """The FTS5 tokens of one document (used to test tokenizer parity)."""
        with self._lock:
            row = self._execute("SELECT rowid FROM lex_docs WHERE doc_id = ?", (doc_id,)).fetchone()
            if row is None:
                raise NotFound(doc_id)
            return {
                r["term"]
                for r in self._execute(
                    "SELECT DISTINCT term FROM lex_vocab WHERE doc = ?", (row["rowid"],)
                )
            }

    # ------------------------------------------------------------------ stats

    def counts(self) -> dict[str, int]:
        with self._lock:
            out: dict[str, int] = {}
            for kind in FORGETTABLE:
                table = _TABLES[kind]
                out[table] = int(self._execute(f"SELECT count(*) FROM {table}").fetchone()[0])  # noqa: S608
                if kind in ARCHIVABLE:
                    out[f"{table}_archived"] = int(
                        self._execute(
                            f"SELECT count(*) FROM {table} WHERE archived_at IS NOT NULL"  # noqa: S608
                        ).fetchone()[0]
                    )
            out["lexical_documents"] = int(
                self._execute("SELECT count(*) FROM lex_docs").fetchone()[0]
            )
            out["queue"] = int(self._execute("SELECT count(*) FROM queue").fetchone()[0])
            return out

    # ------------------------------------------------------------------ row mapping

    def _facts_from_rows(self, rows: Sequence[Any]) -> list[Fact]:
        scopes = self._scopes_map([r["id"] for r in rows])
        return [self._fact_from_row(r, scopes.get(r["id"], ())) for r in rows]

    def _fact_from_row(self, row: Any, scopes: tuple[str, ...] | None = None) -> Fact:
        return Fact(
            id=row["id"],
            node_id=row["node_id"],
            statement=row["statement"],
            attribute=row["attribute"],
            value=row["value"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            origin=row["origin"],
            confidence=float(row["confidence"]),
            source_id=row["source_id"],
            excerpt_start=row["excerpt_start"],
            excerpt_end=row["excerpt_end"],
            superseded_by=row["superseded_by"],
            pinned=bool(row["pinned"]),
            flags=tuple(f for f in row["flags"].split(",") if f),
            scopes=self.scopes_of(row["id"]) if scopes is None else scopes,
            external_id=row["external_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived_at=row["archived_at"],
            language=row["language"],
            conflicts_with=row["conflicts_with"],
            same_as=row["same_as"],
            provenance=row["provenance"],
            proposal_id=row["proposal_id"],
            consolidated_into=row["consolidated_into"],
            rev=int(row["rev"]),
        )

    def _edges_from_rows(self, rows: Sequence[Any]) -> list[Edge]:
        ids = [r["id"] for r in rows]
        scopes = self._scopes_map(ids)
        evidence: dict[str, list[str]] = {}
        if ids:
            for r in self._execute(
                f"SELECT edge_id, source_id FROM edge_evidence WHERE edge_id IN ({_ph(ids)})"  # noqa: S608
                " ORDER BY source_id",
                ids,
            ):
                evidence.setdefault(r["edge_id"], []).append(r["source_id"])
        return [
            self._edge_from_row(r, scopes.get(r["id"], ()), tuple(evidence.get(r["id"], ())))
            for r in rows
        ]

    def _edge_from_row(
        self,
        row: Any,
        scopes: tuple[str, ...] | None = None,
        evidence: tuple[str, ...] = (),
    ) -> Edge:
        return Edge(
            id=row["id"],
            src=row["src"],
            dst=row["dst"],
            type=row["type"],
            weight=float(row["weight"]),
            status=row["status"],
            origin=row["origin"],
            source_id=row["source_id"],
            last_used_at=row["last_used_at"],
            use_count=int(row["use_count"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            scopes=self.scopes_of(row["id"]) if scopes is None else scopes,
            external_id=row["external_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            archived_at=row["archived_at"],
            evidence=evidence,
            rev=int(row["rev"]),
        )


def decayed_weight(weight: float, since: str, now: datetime, half_life_days: float) -> float:
    """``weight`` after the time from ``since`` (an ISO timestamp) to ``now`` at the
    given half-life (SPEC 5.3): a pure function of time, rounded to six decimals
    so that running it twice on the same clock reading changes nothing."""
    try:
        then = datetime.fromisoformat(since)
    except ValueError:
        return weight
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    days = max(0.0, (now - then).total_seconds() / 86400.0)
    if days == 0.0:
        return weight
    return round(weight * exp(-log(2.0) * days / half_life_days), 6)


class _Transaction:
    """A savepoint-based transaction so nested store calls compose."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._name = f"sp{id(self)}"

    def __enter__(self) -> None:
        self._store._execute(f"SAVEPOINT {self._name}")

    def __exit__(self, exc_type: object, *exc: object) -> None:
        if exc_type is None:
            self._store._execute(f"RELEASE {self._name}")
        else:
            self._store._execute(f"ROLLBACK TO {self._name}")
            self._store._execute(f"RELEASE {self._name}")


def _merge(*counts: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in counts:
        for k, v in c.items():
            out[k] = out.get(k, 0) + v
    return out
