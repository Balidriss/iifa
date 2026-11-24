"""Tier-0 recall and expansion: FTS candidates, one graph hop, rank, pack (P1, P2, P9)."""

from __future__ import annotations

import base64
import dataclasses
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from iifa.ingest import credentials
from iifa.ingest.activity import matched_words
from iifa.ingest.checks import grounded, names_subject, unspaced_chars
from iifa.ingest.queue import WORD_KINDS
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
# SPEC 5.5: a request for an activity - activities matched, and what each block holds
MAX_ACTIVITIES = 3
_LIFECYCLE = WORD_KINDS  # P5: the sources a turn's words were extracted from
ACTIVITY_RULES = 10
USED_NODES = 3
USED_RULES = 5
METHOD = "method"
TASK = "task"
EPISODE = "episode"
USES = "uses"
INSTANCE_OF = "instance_of"
TIMELINE = 3  # SPEC 5.5: the episodes a family's block shows, the most recent first
BLOCK_SHARE = 2 / 3  # SPEC 5.5: the share of the budget the activity blocks may take


@dataclass(frozen=True)
class Candidate:
    """A ranked candidate whose handle is built only if it can still fit."""

    score: float
    id: str
    build: Callable[[dict[str, Fact]], dict[str, Any] | None]
    fact_id: str | None = None
    node: Node | None = None
    why: str = ""
    terms: frozenset[str] = frozenset()  # the query words the lexical side matched
    activity: bool = False  # SPEC 5.5: brought by an activity's block


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
    terms: frozenset[str] = frozenset()  # the query words the lexical side matched


def _fuse(lexical: list[_Hit], dense: list[DenseHit]) -> list[_Hit]:
    """Reciprocal rank fusion of the lexical and dense rankings (SPEC 6.1); the
    ``why`` keeps what each side saw."""
    if not dense:
        return lexical
    whys: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    terms: dict[str, frozenset[str]] = {}
    for hit in lexical:
        whys.setdefault(hit.doc_id, []).append(hit.why)
        kinds[hit.doc_id] = hit.kind
        terms[hit.doc_id] = hit.terms
    for found in dense:
        whys.setdefault(found.record_id, []).append(f"vector neighbour {found.score:.2f}")
        kinds[found.record_id] = found.kind
    fused = rrf([[h.doc_id for h in lexical], [d.record_id for d in dense]])
    return [
        _Hit(doc_id, kinds[doc_id], score, "; ".join(whys[doc_id]), terms.get(doc_id, frozenset()))
        for doc_id, score in fused
    ]


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
        observed_half_life_days: float = 7.0,
        activity_threshold: Callable[[str], float | None] | None = None,
        secrets_by_wording: bool = False,
    ) -> None:
        self.store = store
        self.secrets_by_wording = secrets_by_wording  # SPEC P5: detect_secrets_by_wording
        self.tokenizer = tokenizer
        self.max_candidates = max_candidates
        self.neighbour_facts = neighbour_facts
        self.max_neighbours = max_neighbours
        self.tier = tier
        self.dense = dense  # tier 3 (SPEC 6.1); None without an embeddings endpoint
        self.subject_anchoring = subject_anchoring  # SPEC 5.5
        self.observed_half_life_days = observed_half_life_days  # SPEC 4.2; 0: no decay
        # SPEC 5.5, tier 3: how close a vector neighbour matches an activity, per
        # embedding model (None: uncalibrated, vectors match no activity)
        self.activity_threshold = activity_threshold
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
        near: set[str] = set()  # SPEC 5.5: records close enough to match an activity
        if self.dense is not None:
            wanted = self.dense.query_vector(query, query_vector)
            dense, degraded, note = wanted.dense, wanted.degraded, wanted.note
            if wanted.values is not None:
                served = self.tier
                found = self.dense.search(wanted.values, view, self.max_candidates)
                hits = _fuse(hits, found)
                near = self._near(found)
        summary_costs: dict[str, int] = {}
        candidates = self._candidates(
            hits, view, set(exclude_ids), detect(query), query, summary_costs, near
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

    def _near(self, found: Sequence[DenseHit]) -> set[str]:
        """SPEC 5.5, tier 3: the node records at least as close to the request as
        the embedding model's activity threshold (none: no record)."""
        model = self.dense.model if self.dense is not None else None
        limit = self.activity_threshold(model) if self.activity_threshold and model else None
        if limit is None:
            return set()
        return {f.record_id for f in found if f.kind == "node" and f.score >= limit}

    def _lexical(self, terms: list[str], view: Visibility) -> list[_Hit]:
        if not terms:
            return []
        stats = self.store.corpus_stats(view)
        hits = bm25(self.store.postings(terms, view), stats)[: self.max_candidates]
        return [_Hit(h.doc_id, h.kind, h.score, _why_terms(h), frozenset(h.terms)) for h in hits]

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
        read: dict[str, set[str]] = {}  # SPEC 8.5: flags computed at read
        linked: dict[str, list[dict[str, str]]] = {}  # SPEC 8.5: what expand adds
        blocks, capped = 0, 0  # SPEC 5.5: what the activity blocks took, and left out
        index = 0
        while index < len(candidates) and remaining >= self.min_handle_cost:
            batch = candidates[index : index + BATCH]
            wanted = [c.fact_id for c in batch if c.fact_id and c.fact_id not in cache]
            loaded = self.store.facts_by_id(wanted, view)
            cache.update(loaded)
            read.update(self.store.read_flags(loaded.values(), view))
            linked.update(self._activities_of(list(loaded), view))
            for candidate in batch:
                if remaining < self.min_handle_cost:
                    break
                index += 1
                payload = candidate.build(cache)
                if payload is None:  # vanished between ranking and building
                    continue
                handles.with_read_flags(payload, read)
                if candidate.fact_id in linked and "expand_tokens" in payload:
                    extra = {"activities": linked[candidate.fact_id]}
                    payload["expand_tokens"] += self.tokenizer.count(handles.serialized(extra))
                cost = self.cost(payload)
                if candidate.activity and blocks + cost > budget_tokens * BLOCK_SHARE:
                    capped += 1  # SPEC 5.5: the blocks take BLOCK_SHARE of the budget at most
                    continue
                payloads.append(payload)
                costs.append(cost)
                if cost <= remaining:
                    remaining -= cost
                    blocks += cost if candidate.activity else 0
        return payloads, costs, len(candidates) - index + capped

    def _candidates(
        self,
        hits: list[_Hit],
        view: Visibility,
        exclude: set[str],
        language: str = "und",
        query: str = "",
        summary_costs: dict[str, int] | None = None,
        near: set[str] | None = None,
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

        def fact_candidate(
            fact_id: str,
            node: Node | None,
            why: str,
            score: float,
            terms: frozenset[str] = frozenset(),
        ) -> None:
            if fact_id in seen:
                return
            seen.add(fact_id)

            def build(cache: dict[str, Fact]) -> dict[str, Any] | None:
                fact = cache.get(fact_id)
                if fact is None:
                    return None
                expansion = handles.fact_expansion(fact, node, 1, None)
                return handles.fact_handle(fact, node, why, now, self.cost(expansion))

            out.append(Candidate(score, fact_id, build, fact_id, node, why, terms))

        def node_candidate(node: Node, why: str, score: float, terms: frozenset[str]) -> None:
            if node.id in seen:
                return
            seen.add(node.id)

            def build(cache: dict[str, Fact]) -> dict[str, Any]:
                return self._node_handle(node, why, now, view, summary_costs)

            out.append(Candidate(score, node.id, build, terms=terms))

        def seed(node: Node, score: float) -> None:
            if score > seeds.get(node.id, (node, 0.0))[1]:
                seeds[node.id] = (node, score)

        for hit in hits:
            if hit.kind == "fact":
                if hit.doc_id not in subjects:
                    continue
                node_id = subjects[hit.doc_id]
                node = nodes.get(node_id) if node_id else None
                fact_candidate(hit.doc_id, node, hit.why, hit.score, hit.terms)
                if node is not None:
                    seed(node, hit.score)
            else:
                node = nodes.get(hit.doc_id)
                if node is None:
                    continue
                node_candidate(node, hit.why, hit.score, hit.terms)
                seed(node, hit.score)

        neighbours = graph.neighbours(self.store, seeds, view)[: self.max_neighbours]
        by_node = {n.node.id: n for n in neighbours}
        for fact in self.store.facts_top(list(by_node), view, self.neighbour_facts):
            neighbour = by_node[fact.node_id or ""]
            if fact.id in seen:
                continue
            self._remember_via(fact.id, neighbour.via.id)
            fact_candidate(fact.id, neighbour.node, f"edge path: {neighbour.path}", neighbour.score)
        out = sorted(self._aged(out, now), key=lambda c: (-c.score, c.id))
        out = self._behind_their_facts(out, view)
        out = self._with_named_subject(out, list(nodes.values()), query, view, fact_candidate)
        # SPEC 5.5: a request for an activity brings its blocks ahead of all
        blocks = self._activity_blocks(hits, nodes, query, view, near or set())
        start = self._anchored_count(out, list(nodes.values()), query)
        return self._one_per_group(
            _ahead(out, seen, blocks, fact_candidate, start), view, exclude, language, now
        )

    def _anchored_count(self, out: list[Candidate], nodes: list[Node], query: str) -> int:
        """SPEC 5.5: how many candidates lead with the named subject's facts (the
        activity blocks go behind them)."""
        subject = _named_subject(nodes, query) if self.subject_anchoring else None
        if subject is None:
            return 0
        count = 0
        for candidate in out:
            node = candidate.node
            if candidate.fact_id is None or node is None or node.id != subject.id:
                break
            count += 1
        return count

    def _activity_blocks(
        self,
        hits: list[_Hit],
        nodes: dict[str, Node],
        query: str,
        view: Visibility,
        near: set[str],
    ) -> list[tuple[str, Node | None, str]]:
        """SPEC 5.5: the activities the request matches - a word of the name or of
        an alias written whole in it, or (tier 3) a close vector neighbour -, the
        most matched words first, at most ``MAX_ACTIVITIES``; for each, its rules,
        the rules of what it ``uses`` and its newest episode, as
        ``(fact id, the fact's node, why)``."""
        matched: list[tuple[int, int, Node]] = []
        for order, hit in enumerate(hits):
            node = nodes.get(hit.doc_id) if hit.kind == "node" else None
            if node is None or node.kind != TASK:
                continue
            words = max(matched_words(form, query) for form in (node.name, *node.aliases))
            if words or node.id in near:
                matched.append((-words, order, node))
        out: list[tuple[str, Fact, str]] = []
        families: list[str] = []
        for _, _, activity in sorted(matched, key=lambda m: (m[0], m[1])):
            root, family = self._family(activity, view)
            if root.id in families:
                continue  # two occurrences of one activity: one block
            families.append(root.id)
            out.extend(self._activity_block(root, family, view))
            if len(families) >= MAX_ACTIVITIES:
                break
        if not out:
            return []
        subjects = self.store.nodes_by_id(sorted({f.node_id for _, f, _ in out if f.node_id}), view)
        return [(fid, subjects.get(f.node_id or ""), why) for fid, f, why in out]

    def _family(self, activity: Node, view: Visibility) -> tuple[Node, list[str]]:
        """SPEC 5.5: the generic activity ``activity`` stands for (itself, or the
        live, visible task node it is an ``instance_of``) and the family's ids:
        that node and its live, visible instances, one hop."""
        ups = [
            e.dst
            for e in self.store.edges_of([activity.id], view)
            if e.src == activity.id and e.type == INSTANCE_OF
        ]
        parents = self.store.nodes_by_id(ups, view)
        root = next((parents[u] for u in ups if u in parents and parents[u].kind == TASK), activity)
        downs = [
            e.src
            for e in self.store.edges_of([root.id], view)
            if e.dst == root.id and e.type == INSTANCE_OF
        ]
        instances = self.store.nodes_by_id(downs, view)
        return root, [
            root.id,
            *(d for d in dict.fromkeys(downs) if d in instances and instances[d].kind == TASK),
        ]

    def _activity_block(
        self, activity: Node, family: Sequence[str], view: Visibility
    ) -> list[tuple[str, Fact, str]]:
        name = activity.name
        linked = self.store.activity_facts(family, view, rules=ACTIVITY_RULES, episodes=TIMELINE)
        rules = [f for f in linked if EPISODE not in f.flags]
        block = [(f.id, f, f"rule of the activity: {name}") for f in rules]
        used = [
            e.dst
            for e in self.store.edges_of(list(family), view)
            if e.src in family and e.type == USES
        ]
        targets = self.store.nodes_by_id(list(dict.fromkeys(used))[:USED_NODES], view)
        for node_id in list(dict.fromkeys(used))[:USED_NODES]:
            target = targets.get(node_id)
            if target is None:
                continue
            own = self.store.node_rules(node_id, view, USED_RULES)
            block.extend((f.id, f, f"rule of {target.name}, used for {name}") for f in own)
        episodes = [f for f in linked if EPISODE in f.flags]
        block.extend((f.id, f, f"episode of the activity: {name}") for f in episodes)
        return block

    def _aged(self, candidates: list[Candidate], now: datetime) -> list[Candidate]:
        """SPEC 4.2: an observed fact weighs less as it ages - its score times
        2^(-age / half-life), the age in days from ``observed_at``. Moved down,
        never removed; a lasting fact never ages."""
        if self.observed_half_life_days <= 0:
            return candidates
        days = self.store.observed_days([c.fact_id for c in candidates if c.fact_id])
        if not days:
            return candidates
        out: list[Candidate] = []
        for candidate in candidates:
            age = handles.age_days(now, days.get(candidate.fact_id or ""))
            factor = 2.0 ** (-age / self.observed_half_life_days) if age else 1.0
            out.append(dataclasses.replace(candidate, score=candidate.score * factor))
        return out

    def _with_named_subject(
        self,
        out: list[Candidate],
        nodes: list[Node],
        query: str,
        view: Visibility,
        fact_candidate: Callable[[str, Node | None, str, float], None],
    ) -> list[Candidate]:
        """SPEC 5.5: the facts of the one subject the question names join the
        candidates (``fact_candidate`` adds to ``out``), then that subject is
        anchored first."""
        if not self.subject_anchoring or not query.strip():
            return out
        subject = _named_subject(nodes, query)
        if subject is None:
            return out
        for fact in self._joined(subject, query, view):
            fact_candidate(fact.id, subject, _joined_why(subject), 0.0)
        return _anchored(out, subject.id)

    def _joined(self, subject: Node, query: str, view: Visibility) -> list[Fact]:
        """SPEC 5.5: the live, visible facts of a subject the question names
        exactly, in one query on the node (pinned first, then most recent), at
        most ``max_candidates``. None when it is named only by a near form, or
        when another visible node bears the form it is named by (a homonym the
        bounded search did not return: which one is meant cannot be told)."""
        forms = _exact_forms(subject, query)
        if not forms:
            return []
        for form in forms:
            if any(other["id"] != subject.id for other in self.store.lookup_name(form, view)):
                return []
        return self.store.facts([subject.id], view, limit=self.max_candidates)

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

    def _behind_their_facts(self, candidates: list[Candidate], view: Visibility) -> list[Candidate]:
        """SPEC 5.5: a node with no live fact of its own, which the question
        matched only on words a fact candidate matched too ("bureau de Osaka"
        beside "Karim travaille au bureau de Osaka."), moves to just after the
        last such fact; everything else keeps its order."""
        behind: dict[int, int] = {}  # a node's position -> the last fact matching its words
        for position, candidate in enumerate(candidates):
            if candidate.fact_id is not None or not candidate.terms:
                continue
            for index in range(len(candidates) - 1, position, -1):
                other = candidates[index]
                if other.fact_id is not None and candidate.terms <= other.terms:
                    behind[position] = index
                    break
        if not behind:
            return candidates
        # only now the store: most recalls have no node with such a fact below it
        counts = self.store.fact_counts([candidates[p].id for p in behind], view)
        after: dict[int, list[Candidate]] = {}  # index of a fact -> the nodes moved behind it
        moved: set[int] = set()
        for position, index in behind.items():
            if not counts.get(candidates[position].id, 0):
                after.setdefault(index, []).append(candidates[position])
                moved.add(position)
        if not moved:
            return candidates
        out: list[Candidate] = []
        for index, candidate in enumerate(candidates):
            if index not in moved:
                out.append(candidate)
            out.extend(after.get(index, ()))
        return out

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
        handle = handles.node_handle(
            node,
            why,
            now,
            self.cost(expansion),
            count,
            self.tokenizer,
            external_only=node.id in self.store.external_only([node.id], view),
        )
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
        self._flag_at_read([e for _, e in expansions], view)
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

    def _flag_at_read(self, expansions: list[dict[str, Any]], view: Visibility) -> None:
        """SPEC 8.5: the flags computed at read on the expanded facts and on the
        facts a node's expansion lists, in one lookup, before their cost is
        counted."""
        listed = [f for e in expansions if e["kind"] == "node" for f in e["facts"]]
        ids = [e["id"] for e in expansions if e["kind"] == "fact"] + [f["id"] for f in listed]
        if not ids:
            return
        facts = self.store.facts_by_id(list(dict.fromkeys(ids)), view)
        read = self.store.read_flags(facts.values(), view)
        for item in [*(e for e in expansions if e["kind"] == "fact"), *listed]:
            handles.with_read_flags(item, read)

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
            out = handles.fact_expansion(fact, node, level, excerpt)
            activities = self._activities_of([fact.id], view).get(fact.id)
            if activities:  # SPEC 8.5: the activities it was learnt during
                out["activities"] = activities
            return out
        if record_id.startswith("node_"):
            node = self.store.nodes_by_id([record_id], view).get(record_id)
            if node is None:
                return None
            return self._node_expansion(node, view, level)[0]
        return None

    def _activities_of(
        self, fact_ids: Sequence[str], view: Visibility
    ) -> dict[str, list[dict[str, str]]]:
        """SPEC 4.2, 8.5: each fact's activities (``{"id", "name"}``), each read
        through ``merged_into``, those the reader may see (P9); facts without one
        are left out."""
        stored = self.store.fact_activities(fact_ids)
        if not stored:
            return {}
        targets = {a: self.store.merge_target(a) for ids in stored.values() for a in ids}
        visible = self.store.nodes_by_id(sorted({t for t in targets.values() if t}), view)
        out: dict[str, list[dict[str, str]]] = {}
        for fact_id, ids in stored.items():
            shown = dict.fromkeys(t for a in ids if (t := targets[a]) in visible)
            if shown:
                out[fact_id] = [{"id": t, "name": visible[t].name} for t in shown]
        return out

    def _excerpt(self, fact: Fact, view: Visibility) -> str | None:
        if fact.source_id is None:
            return None
        source = self.store.source_if_visible(fact.source_id, view)
        if source is None or source.text is None:
            return None
        if fact.excerpt_start is not None and fact.excerpt_end is not None:
            return source.text[fact.excerpt_start : fact.excerpt_end]
        if source.kind in _LIFECYCLE:
            # P5: a turn may hold a secret only its extraction knew; a fact
            # without an excerpt shows none of it
            return None
        found = credentials.collect(source.text, [], wording=self.secrets_by_wording)
        return credentials.mask(source.text, found)

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
            "rev": node.rev,
        }
        facts = [
            handles.with_from(
                {
                    "id": f.id,
                    "statement": f.statement,
                    "origin": f.origin,
                    "valid_from": f.valid_from,
                    **({"observed_at": f.observed_at} if f.observed_at else {}),
                },
                f,
            )
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
                "rev": e.decision_rev,  # what POST /v1/edges/{id} compares (SPEC 4.4)
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


def _ahead(
    out: list[Candidate],
    seen: set[str],
    block: list[tuple[str, Node | None, str]],
    fact_candidate: Callable[[str, Node | None, str, float], None],
    start: int = 0,
) -> list[Candidate]:
    """SPEC 5.5: the activity blocks ahead of every other candidate but the
    first ``start`` (the named subject's facts, which stay there even when a
    block holds them too); a fact the search also found further down moves
    into its block. ``fact_candidate`` appends to ``out``, which is changed in
    place, and skips a fact already a candidate."""
    if not block:
        return out
    wanted = {fact_id for fact_id, _, _ in block}
    lead = out[:start]
    found = {c.fact_id for c in out[start:] if c.fact_id in wanted}
    out[:] = [c for c in out[start:] if c.fact_id not in found]
    seen.difference_update(found)
    rest = len(out)
    for fact_id, node, why in block:
        fact_candidate(fact_id, node, why, 0.0)
    added = [dataclasses.replace(c, activity=True) for c in out[rest:]]
    return lead + added + out[:rest]


def _names(node: Node, query: str) -> bool:
    """Does the question name ``node``: every word of its name, or of one of its
    aliases, named in the question by the naming rule of SPEC 4.2 (case, diacritics,
    elision, an inflected ending: "Emilie", "d'Émilie", "Карима")?"""
    for form in (node.name, *node.aliases):
        words = form.split()
        if words and all(names_subject(query, [word]) for word in words):
            return True
    return False


def _named_subject(nodes: Any, query: str) -> Node | None:
    """SPEC 5.5: the one visible node the recall found that the question names;
    None when it names none, two, or two homonyms."""
    if not query.strip():
        return None
    named = {node.id: node for node in nodes if _names(node, query)}
    return next(iter(named.values())) if len(named) == 1 else None


def _exact_forms(node: Node, query: str) -> list[str]:
    """``node``'s name and aliases the question holds whole, as consecutive whole
    words (case, Latin diacritics and elision folded, SPEC 4.4b's grounding).
    Never a piece of a word nor an inflected ending ("Karima" is not Karim), and
    no form in a script written without spaces (whole words cannot be told
    there): these decide what joins the candidates (SPEC 5.5)."""
    forms = (node.name, *node.aliases)
    return [f for f in forms if f.strip() and not unspaced_chars(f) and grounded(f, query)]


def _joined_why(subject: Node) -> str:
    return f"subject named in the question: {subject.name}"


def _anchored(candidates: list[Candidate], subject: str) -> list[Candidate]:
    """SPEC 5.5: the named subject's facts first (the searched ones, then the
    joined ones), each side keeping its order; the node's own handle keeps its
    place (it is costly: ahead, it pushed the answer out of small budgets)."""

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
