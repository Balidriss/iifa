"""The Python query tokenizer and FTS5's unicode61 must agree on every script,
otherwise a query could miss what the index holds."""

from __future__ import annotations

import pytest

from iifa.recall.tokenizer import tokenize
from iifa.scopes import Visibility
from iifa.store.repo import Store
from tests.fixtures.corpus import SAMPLES


@pytest.mark.parametrize("language", sorted(SAMPLES))
def test_every_query_token_is_an_indexed_term(store: Store, language: str) -> None:
    text = SAMPLES[language]
    fact = store.add_fact(scopes=["user:alice"], statement=text, origin="explicit")
    indexed = store.indexed_terms(fact.id)
    for token in tokenize(text):
        assert token in indexed, (token, indexed)
    # and searching with those tokens finds the document
    view = Visibility.of_scope("user:alice")
    for token in tokenize(text):
        assert [p.doc_id for p in store.postings([token], view)] == [fact.id]


def test_document_length_agrees_with_fts5(store: Store) -> None:
    text = SAMPLES["mixed"]
    fact = store.add_fact(scopes=["user:alice"], statement=text, origin="explicit")
    assert len(store.indexed_terms(fact.id)) == len(set(tokenize(text)))
