"""Tier-0 recall and expansion: FTS candidates, one graph hop, rank, pack (P1, P2, P9)."""

from __future__ import annotations

import base64
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from iifa.ingest.checks import names_subject
from iifa.ingest.values import precision
from iifa.language import detect
from iifa.recall import graph, handles
from iifa.recall.dense import OFF, Dense, DenseHit, SuppliedVector
from iifa.recall.handles import NODE_EXPANSION_FACTS
from iifa.recall.lexical import LexicalHit, bm25
from iifa.recall.packer import pack
from iifa.recall.tokenizer import tokenize
from iifa.scopes import Visibility
from iifa.store.models import Fact, Node
from iifa.store.repo import Store
from iifa.tokens import Tokenizer
from iifa.vectors import rrf

TIER = 0  # with no model at all; the app sets ``RecallEngine.tier`` to 2 with a chat endpoint
REINFORCEMENT_MEMORY = 4096


@dataclass(frozen=True)
class Candidate:
    """A ranked candidate whose handle is built only if it can still fit."""

    score: float
    id: str
    build: Callable[[dict[str, Fact]], dict[str, Any] | None]
    fact_id: str | None = None
    node: Node | None = None
    why: str = ""


def preferred(members: Sequence[Fact], language: str) -> Fact:
    """The one fact of an equivalence group recall injects (SPEC 5.4): the one in
    the query's language, else the most precise (a full date before a month),
    else the group's first (members come oldest first)."""
    pool = [m for m in members if m.language == language] or list(members)
    return max(enumerate(pool), key=lambda im: (precision(im[1].value or ""), -im[0]))[1]


BATCH = 32


@dataclass(frozen=True)
class _Hit:
    """A ranked record before its handle exists: lexical, dense, or both fused."""

    doc_id: str
    kind: str
    score: float
    why: str


def _fuse(lexical: list[_Hit], dense: list[DenseHit]) -> list[_Hit]:
    """Reciprocal rank fusion of the lexical and dense rankings (SPEC 6.1); the
    ``why`` keeps what each side saw."""
    if not dense:
        return lexical
    whys: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    for hit in lexical:
        whys.setdefault(hit.doc_id, []).append(hit.why)
        kinds[hit.doc_id] = hit.kind
    for found in dense:
        whys.setdefault(found.record_id, []).append(f"vector neighbour {found.score:.2f}")
        kinds[found.record_id] = found.kind
    fused = rrf([[h.doc_id for h in lexical], [d.record_id for d in dense]])
    return [_Hit(doc_id, kinds[doc_id], score, "; ".join(whys[doc_id])) for doc_id, score in fused]


class RecallEngine:
    def __init__(
        self,
        store: Store,
        tokenizer: Tokenizer,
        *,
        max_candidates: int = 200,
        neighbour_facts: int = 3,
        max_neighbours: int = 10,
        tier: int = TIER,
        dense: Dense | None = None,
        subject_anchoring: bool = True,
    ) -> None:
        self.store = store
        self.tokenizer = tokenizer
        self.max_candidates = max_candidates
        self.neighbour_facts = neighbour_facts
        self.max_neighbours = max_neighbours
        self.tier = tier
        self.dense = dense  # tier 3 (SPEC 6.1); None without an embeddings endpoint
        self.subject_anchoring = subject_anchoring  # SPEC 5.5
        # SPEC 5.3: which edge surfaced which handle, so that an expand of that
        # handle reinforces the edge. Bounded, per process, best effort.
        self._via: OrderedDict[str, str] = OrderedDict()
        self._via_lock = threading.Lock()
        # No handle can cost less than an empty one: past this, nothing else fits.
        self.min_handle_cost = self.cost(
            {
                "id": "fact_" + "0" * 26,
                "kind": "fact",
                "title": "",
                "line": "",
                "why": "",
                "age_days": None,
                "expand_tokens": 0,
                "flags": [],
            }
        )

    # ------------------------------------------------------------------ helpers

    def cost(self, payload: Any) -> int:
        """What ``payload`` costs the agent: its serialized text plus the item overhead."""
        return self.tokenizer.count(handles.serialized(payload)) + self.tokenizer.per_item_overhead

    def _now(self) -> datetime:
        return self.store.clock()

    # ------------------------------------------------------------------ recall

    def recall(
        self,
        query: str,
        *,
        view: Visibility,
        budget_tokens: int,
        exclude_ids: Sequence[str] = (),
        query_vector: SuppliedVector | None = None,
    ) -> dict[str, Any]:
        answer, _ = self.measured_recall(
            query,
            view=view,
            budget_tokens=budget_tokens,
            exclude_ids=exclude_ids,
            query_vector=query_vector,
        )
        return answer

    def measured_recall(
        self,
        query: str,
        *,
        view: Visibility,
        budget_tokens: int,
        exclude_ids: Sequence[str] = (),
        query_vector: SuppliedVector | None = None,
    ) -> tuple[dict[str, Any], int]:
        """``recall``, and the summary tokens of what it returned (SPEC 11.2): for
        each node handle whose line is a summary, what that line costs above the
        line of a node without one - never negative, at most the bound. For the
        probes only: the answer an agent receives is unchanged."""
        terms = tokenize(query)
        hits = self._lexical(terms, view)
        served = min(self.tier, 2)
        dense = OFF
        degraded: str | None = None
        note: str | None = None
        if self.dense is not None:
            wanted = self.dense.query_vector(query, query_vector)
            dense, degraded, note = wanted.dense, wanted.degraded, wanted.note
            if wanted.values is not None:
                served = self.tier
                found = self.dense.search(wanted.values, view, self.max_candidates)
                hits = _fuse(hits, found)
        summary_costs: dict[str, int] = {}
        candidates = self._candidates(
            hits, view, set(exclude_ids), detect(query), query, summary_costs
        )
        payloads, costs, unbuilt = self._materialize(candidates, view, budget_tokens)
        result = pack(costs, budget_tokens)
        reason = result.reason
        if not candidates:
            reason = "no_match" if terms or dense == "on" else "empty_query"
        elif not result.chosen:
            reason = "budget_too_small"
        chosen = [payloads[i] for i in result.chosen]
        summary_tokens = sum(summary_costs.get(str(item["id"]), 0) for item in chosen)
        return {
            "items": chosen,
            "used_tokens": result.used_tokens,
            "budget_tokens": budget_tokens,
            "omitted": result.omitted + unbuilt,
            "reason": reason,
            "tier": served,
            "dense": dense,
            "degraded": degraded,
            "query_vector": note,
        }, summary_tokens

    def _lexical(self, terms: list[str], view: Visibility) -> list[_Hit]:
        if not terms:
            return []
        stats = self.store.corpus_stats(view)
        hits = bm25(self.store.postings(terms, view), stats)[: self.max_candidates]
        return [_Hit(h.doc_id, h.kind, h.score, _why_terms(h)) for h in hits]

    def _remember_via(self, item_id: str, edge_id: str) -> None:
        with self._via_lock:
            self._via[item_id] = edge_id
            self._via.move_to_end(item_id)
            while len(self._via) > REINFORCEMENT_MEMORY:
                self._via.popitem(last=False)

    def _reinforce(self, item_ids: Sequence[str]) -> list[str]:
        """The expand signal (SPEC 5.3): every delivered item a recall reached
        through an edge reinforces that edge, once per recall."""
        reinforced: list[str] = []
        for item_id in item_ids:
            with self._via_lock:
                edge_id = self._via.pop(item_id, None)
            if edge_id is not None and self.store.reinforce_edge(edge_id) is not None:
                reinforced.append(edge_id)
        return reinforced

    def _materialize(
        self, candidates: list[Candidate], view: Visibility, budget_tokens: int
    ) -> tuple[list[dict[str, Any]], list[int], int]:
        """Build handles in rank order, a batch at a time, until the remaining
        budget cannot hold even an empty handle. The packer then decides on
        exactly what was built; everything after the cut-off is omitted (it
        could not have fit either), so the result equals a full first-fit pack."""
        payloads: list[dict[str, Any]] = []
        costs: list[int] = []
        remaining = budget_tokens
        cache: dict[str, Fact] = {}
        index = 0
        while index < len(candidates) and remaining >= self.min_handle_cost:
            batch = candidates[index : index + BATCH]
            wanted = [c.fact_id for c in batch if c.fact_id and c.fact_id not in cache]
            cache.update(self.store.facts_by_id(wanted, view))
            for candidate in batch:
                if remaining < self.min_handle_cost:
                    break
                index += 1
                payload = candidate.build(cache)
                if payload is None:  # vanished between ranking and building
                    continue
                cost = self.cost(payload)
                payloads.append(payload)
                costs.append(cost)
                if cost <= remaining:
                    remaining -= cost
        return payloads, costs, len(candidates) - index

    def _candidates(
        self,
        hits: list[_Hit],
        view: Visibility,
        exclude: set[str],
        language: str = "und",
        query: str = "",
        summary_costs: dict[str, int] | None = None,
    ) -> list[Candidate]:
        if not hits:
            return []
        # Postings and vectors already applied the view, so only the subject of each matched
        # fact is needed now; full rows are loaded in batches when handles are built.
        subjects = self.store.fact_subjects([h.doc_id for h in hits if h.kind == "fact"])
        node_ids = {n for n in subjects.values() if n} | {
            h.doc_id for h in hits if h.kind == "node"
        }
        nodes = self.store.nodes_by_id(sorted(node_ids), view)
        now = self._now()

        seen: set[str] = set(exclude)
        out: list[Candidate] = []
        seeds: dict[str, tuple[Node, float]] = {}

        def fact_candidate(fact_id: str, node: Node | None, why: str, score: float) -> None:
            if fact_id in seen:
                return
            seen.add(fact_id)

            def build(cache: dict[str, Fact]) -> dict[str, Any] | None:
                fact = cache.get(fact_id)
                if fact is None:
                    return None
                expansion = handles.fact_expansion(fact, node, 1, None)
                return handles.fact_handle(fact, node, why, now, self.cost(expansion))

            out.append(Candidate(score, fact_id, build, fact_id, node, why))

        def node_candidate(node: Node, why: str, score: float) -> None:
            if node.id in seen:
                return
            seen.add(node.id)

            def build(cache: dict[str, Fact]) -> dict[str, Any]:
                return self._node_handle(node, why, now, view, summary_costs)

            out.append(Candidate(score, node.id, build))

        def seed(node: Node, score: float) -> None:
            if score > seeds.get(node.id, (node, 0.0))[1]:
                seeds[node.id] = (node, score)

        for hit in hits:
            if hit.kind == "fact":
                if hit.doc_id not in subjects:
                    continue
                node_id = subjects[hit.doc_id]
                node = nodes.get(node_id) if node_id else None
                fact_candidate(hit.doc_id, node, hit.why, hit.score)
                if node is not None:
                    seed(node, hit.score)
            else:
                node = nodes.get(hit.doc_id)
                if node is None:
                    continue
                node_candidate(node, hit.why, hit.score)
                seed(node, hit.score)

        neighbours = graph.neighbours(self.store, seeds, view)[: self.max_neighbours]
        by_node = {n.node.id: n for n in neighbours}
        for fact in self.store.facts_top(list(by_node), view, self.neighbour_facts):
            neighbour = by_node[fact.node_id or ""]
            if fact.id in seen:
                continue
            self._remember_via(fact.id, neighbour.via.id)
            fact_candidate(fact.id, neighbour.node, f"edge path: {neighbour.path}", neighbour.score)
        out.sort(key=lambda c: (-c.score, c.id))
        if self.subject_anchoring and query.strip():
            out = _anchored(out, nodes.values(), query)
        return self._one_per_group(out, view, exclude, language, now)

    def _one_per_group(
        self,
        candidates: list[Candidate],
        view: Visibility,
        exclude: set[str],
        language: str,
        now: datetime,
    ) -> list[Candidate]:
        """SPEC 5.4: one fact per equivalence group, at the group's best rank, the
        budget never paying twice for one value; a group one of whose facts the
        harness already injected (``exclude_ids``) is not injected again."""
        roots = self.store.fact_roots([c.fact_id for c in candidates if c.fact_id])
        groups = self.store.equivalents(sorted(set(roots.values())), view)
        out: list[Candidate] = []
        done: set[str] = set()
        for candidate in candidates:
            root = roots.get(candidate.fact_id or "")
            members = groups.get(root or "", [])
            if root is None or len(members) <= 1:
                out.append(candidate)
                continue
            if root in done:
                continue
            done.add(root)
            if any(m.id in exclude for m in members):
                continue
            best = preferred(members, language)
            if best.id == candidate.fact_id:
                out.append(candidate)
                continue
            out.append(self._fact_candidate(best.id, candidate, now))
        return out

    def _fact_candidate(self, fact_id: str, like: Candidate, now: datetime) -> Candidate:
        """The group's preferred fact, in the place of the candidate that found it."""
        node = like.node

        def build(cache: dict[str, Fact]) -> dict[str, Any] | None:
            fact = cache.get(fact_id)
            if fact is None:
                return None
            expansion = handles.fact_expansion(fact, node, 1, None)
            return handles.fact_handle(fact, node, like.why, now, self.cost(expansion))

        return Candidate(like.score, fact_id, build, fact_id, node, like.why)

    def _node_handle(
        self,
        node: Node,
        why: str,
        now: datetime,
        view: Visibility,
        summary_costs: dict[str, int] | None,
    ) -> dict[str, Any]:
        """A node's handle; its summary tokens (SPEC 11.2) noted in ``summary_costs``."""
        expansion, count = self._node_expansion(node, view, 1)
        handle = handles.node_handle(node, why, now, self.cost(expansion), count, self.tokenizer)
        if summary_costs is not None and node.summary.strip():
            plain = {**handle, "line": handles.plain_line(node, count)}
            extra = self.cost(handle) - self.cost(plain)
            summary_costs[node.id] = min(max(extra, 0), handles.SUMMARY_LINE_TOKENS)
        return handle

    def _node_expansion(
        self, node: Node, view: Visibility, level: int
    ) -> tuple[dict[str, Any], int]:
        count = self.store.fact_counts([node.id], view).get(node.id, 0)
        top = self.store.facts([node.id], view, limit=NODE_EXPANSION_FACTS)
        return handles.node_expansion(node, top, level, count), count

    # ------------------------------------------------------------------ expand

    def expand(
        self,
        ids: Sequence[str],
        *,
        view: Visibility,
        budget_tokens: int,
        level: int = 1,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        start = _decode_cursor(cursor)
        requested = list(dict.fromkeys(ids))
        pending = requested[start:]
        expansions: list[tuple[str, dict[str, Any]]] = []
        missing: list[str] = []
        for record_id in pending:
            expansion = self._expansion(record_id, view, level)
            if expansion is None:
                missing.append(record_id)
            else:
                expansions.append((record_id, expansion))
        result = pack([self.cost(e) for _, e in expansions], budget_tokens)
        chosen = [expansions[i][1] for i in result.chosen]
        reinforced = self._reinforce([expansions[i][0] for i in result.chosen])
        # The cursor points just past the last delivered id, so an item that was
        # skipped for size comes back on a later page instead of being lost.
        next_cursor: str | None = None
        if result.chosen and result.omitted:
            last_id = expansions[result.chosen[-1]][0]
            position = start + pending.index(last_id) + 1
            if position < len(requested):
                next_cursor = _encode_cursor(position)
        return {
            "items": chosen,
            "missing": missing,
            "used_tokens": result.used_tokens,
            "budget_tokens": budget_tokens,
            "omitted": result.omitted,
            "reason": result.reason,
            "cursor": next_cursor,
            "level": level,
            "reinforced_edges": reinforced,
        }

    def _expansion(self, record_id: str, view: Visibility, level: int) -> dict[str, Any] | None:
        if record_id.startswith("fact_"):
            fact = self.store.facts_by_id([record_id], view).get(record_id)
            if fact is None:
                return None
            node = (
                self.store.nodes_by_id([fact.node_id], view).get(fact.node_id)
                if fact.node_id
                else None
            )
            excerpt = self._excerpt(fact, view) if level >= 2 else None
            return handles.fact_expansion(fact, node, level, excerpt)
        if record_id.startswith("node_"):
            node = self.store.nodes_by_id([record_id], view).get(record_id)
            if node is None:
                return None
            return self._node_expansion(node, view, level)[0]
        return None

    def _excerpt(self, fact: Fact, view: Visibility) -> str | None:
        if fact.source_id is None:
            return None
        source = self.store.source_if_visible(fact.source_id, view)
        if source is None or source.text is None:
            return None
        if fact.excerpt_start is not None and fact.excerpt_end is not None:
            return source.text[fact.excerpt_start : fact.excerpt_end]
        return source.text

    # ------------------------------------------------------------------ node view

    def node_view(
        self, node_id: str, *, view: Visibility, budget_tokens: int
    ) -> dict[str, Any] | None:
        node = self.store.nodes_by_id([node_id], view).get(node_id)
        if node is None:
            return None
        head: dict[str, Any] = {
            "id": node.id,
            "kind": node.kind,
            "name": node.name,
            "aliases": list(node.aliases),
            "summary": node.summary,
            "language": node.language,
            "flags": list(node.flags),
            "scopes": list(node.scopes),
            "external_id": node.external_id,
            "created_at": node.created_at,
            "updated_at": node.updated_at,
        }
        facts = [
            {"id": f.id, "statement": f.statement, "origin": f.origin, "valid_from": f.valid_from}
            for f in self.store.facts([node.id], view)
        ]
        edges = [
            {
                "id": e.id,
                "src": e.src,
                "dst": e.dst,
                "type": e.type,
                "status": e.status,
                "weight": e.weight,
            }
            for e in self.store.edges_of([node.id], view)
        ]
        # SPEC 4.4b: pending proposals visible in the view (P9), after everything else
        proposals = [
            {
                "id": p.id,
                "kind": p.kind,
                "value": p.value,
                "source_id": p.source_id,
                "confidence": p.confidence,
                "created_at": p.created_at,
            }
            for p in self.store.proposals([node.id], view)
        ]
        head_cost = self.cost(head)
        if head_cost > budget_tokens:
            return {
                "id": node.id,
                "facts": [],
                "edges": [],
                "proposals": [],
                "used_tokens": 0,
                "budget_tokens": budget_tokens,
                "omitted": 1 + len(facts) + len(edges) + len(proposals),
                "reason": "budget_too_small",
            }
        items = (*facts, *edges, *proposals)
        packed = pack([self.cost(x) for x in items], budget_tokens, reserved_tokens=head_cost)
        chosen = set(packed.chosen)
        first_proposal = len(facts) + len(edges)
        return {
            **head,
            "facts": [f for i, f in enumerate(facts) if i in chosen],
            "edges": [e for i, e in enumerate(edges) if i + len(facts) in chosen],
            "proposals": [p for i, p in enumerate(proposals) if i + first_proposal in chosen],
            "used_tokens": packed.used_tokens if packed.chosen else head_cost,
            "budget_tokens": budget_tokens,
            "omitted": packed.omitted if items else 0,
            "reason": None,
        }


def _names(node: Node, query: str) -> bool:
    """Does the question name ``node``: every word of its name, or of one of its
    aliases, named in the question by the naming rule of SPEC 4.2 (case, diacritics,
    elision, an inflected ending: "Emilie", "d'Émilie", "Карима")?"""
    for form in (node.name, *node.aliases):
        words = form.split()
        if words and all(names_subject(query, [word]) for word in words):
            return True
    return False


def _anchored(candidates: list[Candidate], nodes: Any, query: str) -> list[Candidate]:
    """SPEC 5.5: a question naming exactly one subject (a visible node the recall
    found) puts that node's facts first, each side keeping its order; the node's
    own handle keeps its place (it is costly: ahead, it pushed the answer out of
    small budgets). Naming none, two, or two homonyms changes nothing."""
    named = {node.id for node in nodes if _names(node, query)}
    if len(named) != 1:
        return candidates
    (subject,) = named

    def about(candidate: Candidate) -> bool:
        node = candidate.node
        return candidate.fact_id is not None and node is not None and node.id == subject

    return [c for c in candidates if about(c)] + [c for c in candidates if not about(c)]


def _why_terms(hit: LexicalHit) -> str:
    unique = list(dict.fromkeys(hit.terms))
    label = "query term" if len(unique) == 1 else "query terms"
    return f"{label} " + ", ".join(f"'{t}'" for t in unique)


def _encode_cursor(position: int) -> str:
    return base64.urlsafe_b64encode(str(position).encode()).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> int:
    if not cursor:
        return 0
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        return max(0, int(base64.urlsafe_b64decode(padded).decode()))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid cursor") from exc
