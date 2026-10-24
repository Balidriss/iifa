"""Entity resolution on the write path (SPEC 5.1): mention -> node.

In order, stopping at the first confident match:

1. an exact name or alias in scope (case, accents and diacritics folded);
2. whole words of exactly one known name (a first name against a full
   name), of a compatible kind - a match; names alike letter by letter (typos,
   near names) are only candidates for step 4;
3. embedding similarity (tier 3, SPEC 6.1): the extractor embeds the new
   mentions of a job in one call and hands the near nodes to
   ``set_semantic``; they only ever make a mention **ambiguous** (a candidate
   for step 4), never resolve it on their own;
4. one batched LLM adjudication for what is still ambiguous (tier 2).

An unresolved mention becomes a new node. Everything a resolver compares
against is what the writer's view may see (P9 holds on the write path too: a
mention never resolves to a node the scope cannot see).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from iifa.recall.tokenizer import normalize, tokenize
from iifa.scopes import Visibility
from iifa.store.repo import Store

EXACT, FUZZY, AMBIGUOUS, ADJUDICATED, NEW = "exact", "fuzzy", "ambiguous", "adjudicated", "new"
OTHER = "other"  # a node kind the model left open: compatible with any
DEFAULT_FUZZY_THRESHOLD = 0.86
CONTAINMENT_SCORE = 0.9
MIN_FUZZY_CHARS = 4
MIN_CONTAINMENT_CHARS = 3


@dataclass(frozen=True)
class Candidate:
    key: str  # a node id, or the key of a node this same answer proposes
    name: str
    kind: str
    score: float
    summary: str = ""


@dataclass
class Resolution:
    mention: str
    key: str | None
    method: str
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class _Entry:
    key: str
    name: str
    kind: str
    norms: tuple[str, ...]
    summary: str = ""
    raw: tuple[str, ...] = ()  # the forms as written: their words (a hyphen joins one)


def name_words(name: str) -> frozenset[str]:
    """The words of a name as written, folded: split on spaces only, so that
    "Jean-Pierre" and "Marie-Anne" stay one word each."""
    return frozenset(w for w in ("-".join(tokenize(chunk)) for chunk in name.split()) if w)


def part_of(mention: str, name: str) -> bool:
    """SPEC 5.1: every word of ``mention`` is a word of the longer ``name``
    ("Karim" of "Karim Benali"), never the reverse ("Karim Haddad" is not part
    of "Karim") and never a piece of a word ("Anne" of "Marie-Anne")."""
    words, whole = name_words(mention), name_words(name)
    return bool(words) and words < whole and len("".join(words)) >= MIN_CONTAINMENT_CHARS


def _kinds_agree(kind: str | None, other: str) -> bool:
    return kind is None or kind == other or OTHER in (kind, other)


def contained(a: str, b: str) -> bool:
    """One *normalised* name is whole words of the other (``karim`` in ``karim
    benali``), never a substring (``karim`` is not in ``karima``)."""
    if a == b or not a or not b:
        return False
    ta, tb = set(a.split()), set(b.split())
    shorter = a if len(a) <= len(b) else b
    return len(shorter) >= MIN_CONTAINMENT_CHARS and (ta <= tb or tb <= ta)


def similarity(a: str, b: str) -> float:
    """How alike two *normalised* names are: 1 when equal; ``CONTAINMENT_SCORE``
    when one is a whole-token part of the other (``karim`` in ``karim benali``);
    else the SequenceMatcher ratio, only for strings long enough to mean it."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if contained(a, b):
        return CONTAINMENT_SCORE
    if min(len(a), len(b)) < MIN_FUZZY_CHARS:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


class Resolver:
    """Resolves the mentions of one source against the nodes ``view`` sees, plus
    the nodes the same source proposes (so a fact can name a node created two
    lines earlier in the answer)."""

    def __init__(
        self,
        store: Store,
        view: Visibility,
        *,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        index_limit: int = 5000,
    ) -> None:
        self.store = store
        self.view = view
        self.fuzzy_threshold = fuzzy_threshold
        self._index_limit = index_limit
        self._entries: list[_Entry] | None = None
        self._pending: list[_Entry] = []
        self._cache: dict[tuple[str, str | None], Resolution] = {}
        # SPEC 5.1 step 4: what the adjudication attached stays attached, whatever
        # the answer proposes afterwards (a proposal clears ``_cache``)
        self._decided: dict[str, Resolution] = {}
        self._semantic: dict[str, list[Candidate]] = {}
        self._contested: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ the candidate pool

    def _index(self) -> list[_Entry]:
        if self._entries is None:
            self._entries = [
                _Entry(node_id, name, kind, (name_norm, *aliases), raw=(name,))
                for node_id, kind, name, name_norm, aliases in self.store.node_index(
                    self.view, limit=self._index_limit
                )
            ]
        return self._entries

    def known_names(self, limit: int) -> list[str]:
        """Display names of the most recently updated visible nodes (for the prompt)."""
        return [e.name for e in self._index()[:limit]]

    def known_nodes(self, limit: int) -> list[tuple[str, str]]:
        """``(node id, display name)`` of the same nodes, in the same order."""
        return [(e.key, e.name) for e in self._index()[:limit]]

    def propose(self, key: str, name: str, kind: str, aliases: Sequence[str] = ()) -> None:
        """A node the current answer proposes: resolvable from now on under ``key``."""
        norms = tuple(dict.fromkeys(n for n in (normalize(name), *map(normalize, aliases)) if n))
        self._pending.append(_Entry(key, name, kind, norms, raw=(name, *aliases)))
        self._cache.clear()

    def set_semantic(self, mention: str, candidates: Sequence[Candidate]) -> None:
        """The nodes close to ``mention`` by their name vectors (SPEC 5.1 step 3),
        computed by the caller in one batched embedding call."""
        self._semantic[mention] = list(candidates)
        self._forget(mention)

    def semantic_candidates(self, mention: str) -> list[Candidate]:
        """SPEC 5.1 step 3, tier 3: what ``set_semantic`` found for ``mention``."""
        return list(self._semantic.get(mention, ()))

    def contest(self, mention: str, keys: Sequence[str]) -> None:
        """SPEC 5.1, the source form of a subject: ``mention`` is never a match
        on its own; the nodes ``keys`` name join its candidates for the
        adjudication, beside whatever it resolves to."""
        self._contested[mention] = list(dict.fromkeys(keys))
        self._forget(mention)

    def _forget(self, mention: str) -> None:
        for key in [k for k in self._cache if k[0] == mention]:
            del self._cache[key]

    # ------------------------------------------------------------------ resolution

    def resolve(self, mention: str, *, kind: str | None = None) -> Resolution:
        """``mention`` -> node; ``kind``, when the answer gives one, keeps a
        containment match within compatible kinds (a city is not a person)."""
        decided = self._decided.get(mention)
        if decided is not None:
            return decided
        cached = self._cache.get((mention, kind))
        if cached is not None:
            return cached
        resolution = self._resolve(mention, kind)
        self._cache[(mention, kind)] = resolution
        return resolution

    def _resolve(self, mention: str, kind: str | None) -> Resolution:
        resolution = self._plain(mention, kind)
        contested = self._contested.get(mention)
        if contested is None:
            return resolution
        norm = normalize(mention)
        keys = [*contested]
        if resolution.key is not None:
            keys.append(resolution.key)
        keys.extend(c.key for c in resolution.candidates)
        found: dict[str, Candidate] = {}
        for entry in [*self._pending, *self._index()]:
            if entry.key in keys and entry.key not in found:
                score = max(similarity(norm, n) for n in entry.norms)
                found[entry.key] = Candidate(
                    entry.key, entry.name, entry.kind, score, entry.summary
                )
        if not found:
            return Resolution(mention, None, NEW)
        ranked = sorted(found.values(), key=lambda c: (-c.score, c.name))
        return Resolution(mention, None, AMBIGUOUS, ranked[:5])

    def _plain(self, mention: str, kind: str | None) -> Resolution:
        norm = normalize(mention)
        if not norm:
            return Resolution(mention, None, NEW)
        pool = [*self._pending, *self._index()]
        # 1. exact name or alias
        for entry in pool:
            if norm in entry.norms:
                return Resolution(mention, entry.key, EXACT)
        # 2. the mention's words inside exactly one known name of a compatible
        # kind; 3. anything else alike is a candidate only (SPEC 5.1: a wrong
        # attachment corrupts another person's memory, a split node is visible)
        scored: list[Candidate] = []
        containing: set[str] = set()
        for entry in pool:
            best = max(similarity(norm, n) for n in entry.norms)
            if best >= self.fuzzy_threshold:
                scored.append(Candidate(entry.key, entry.name, entry.kind, best, entry.summary))
                if _kinds_agree(kind, entry.kind) and any(part_of(mention, r) for r in entry.raw):
                    containing.add(entry.key)
        scored.sort(key=lambda c: (-c.score, c.name))
        if len(scored) == 1 and scored[0].key in containing:
            return Resolution(mention, scored[0].key, FUZZY, scored[:1])
        # 3. semantic neighbours join the candidates: never a match on their own
        known = {c.key for c in scored}
        scored.extend(c for c in self.semantic_candidates(mention) if c.key not in known)
        scored.sort(key=lambda c: (-c.score, c.name))
        if not scored:
            return Resolution(mention, None, NEW)
        return Resolution(mention, None, AMBIGUOUS, scored[:5])

    def decide(self, resolution: Resolution, choice: str | None) -> None:
        """Apply an adjudication (step 4): ``choice`` is a candidate key or ``new``."""
        if choice is not None and any(c.key == choice for c in resolution.candidates):
            resolution.key = choice
            resolution.method = ADJUDICATED
            self._decided[resolution.mention] = resolution
        else:
            resolution.key = None
            resolution.method = NEW
            self._decided.pop(resolution.mention, None)
        self._cache[(resolution.mention, None)] = resolution


__all__ = [
    "ADJUDICATED",
    "AMBIGUOUS",
    "EXACT",
    "FUZZY",
    "NEW",
    "Candidate",
    "Resolution",
    "Resolver",
    "contained",
    "name_words",
    "part_of",
    "similarity",
]
