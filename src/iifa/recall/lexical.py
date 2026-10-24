"""BM25 over postings that were already filtered by scope (P9).

Every statistic here - document frequency, corpus size, average length - is
computed from the visible corpus only, so an invisible record can neither
change a score nor reveal its existence through one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from iifa.store.repo import CorpusStats, Posting

K1 = 1.2
B = 0.75


@dataclass
class LexicalHit:
    doc_id: str
    kind: str
    score: float
    terms: list[str] = field(default_factory=list)


def bm25(postings: Iterable[Posting], stats: CorpusStats) -> list[LexicalHit]:
    """Score documents from scope-filtered postings. Sorted by score, ties by id."""
    by_term: dict[str, list[Posting]] = {}
    for posting in postings:
        by_term.setdefault(posting.term, []).append(posting)
    hits: dict[str, LexicalHit] = {}
    n = max(stats.documents, 1)
    avgdl = stats.avg_length or 1.0
    for term, term_postings in by_term.items():
        df = len(term_postings)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for p in term_postings:
            norm = K1 * (1 - B + B * p.ntok / avgdl)
            partial = idf * (p.tf * (K1 + 1)) / (p.tf + norm)
            hit = hits.get(p.doc_id)
            if hit is None:
                hit = hits[p.doc_id] = LexicalHit(p.doc_id, p.kind, 0.0)
            hit.score += partial
            hit.terms.append(term)
    return sorted(hits.values(), key=lambda h: (-h.score, h.doc_id))
