from __future__ import annotations

from iifa.recall.lexical import bm25
from iifa.store.repo import CorpusStats, Posting


def test_bm25_ranks_rarer_terms_and_shorter_documents_higher() -> None:
    postings = [
        Posting("solar", 1, "fact_a", "fact", 1, 10),
        Posting("solar", 2, "fact_b", "fact", 1, 40),
        Posting("the", 1, "fact_a", "fact", 1, 10),
        Posting("the", 2, "fact_b", "fact", 1, 40),
        Posting("the", 3, "fact_c", "fact", 1, 10),
    ]
    hits = bm25(postings, CorpusStats(documents=100, avg_length=20.0))
    assert [h.doc_id for h in hits] == ["fact_a", "fact_b", "fact_c"]
    assert hits[0].terms == ["solar", "the"]
    assert hits[2].terms == ["the"]
    assert hits[0].score > hits[1].score > hits[2].score > 0


def test_bm25_is_deterministic_on_ties() -> None:
    postings = [Posting("x", 1, "fact_b", "fact", 1, 5), Posting("x", 2, "fact_a", "fact", 1, 5)]
    hits = bm25(postings, CorpusStats(2, 5.0))
    assert [h.doc_id for h in hits] == ["fact_a", "fact_b"]
    assert bm25([], CorpusStats(0, 0.0)) == []
