"""The embed job's race (SPEC 6.1, P5): the job reads a facet's text, calls the
model, then writes the vector. Whatever changed in between - the record
forgotten, or its text rewritten - the write is discarded and counted, never
applied: no vector derived from forgotten content, no stale vector used
forever. A seam puts the change exactly between the call and the write (no
thread, no timing)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import iifa.ingest.embed as embed_module
from iifa import vectors
from iifa.api.app import create_app
from iifa.store.repo import SUMMARY, TEXT, PendingVector
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth, remember
from tests.model.test_tier3 import tier3_settings
from tests.security.test_p5_encryption_archive_forget import _raw_bytes
from tests.tier2 import run_all

pytestmark = pytest.mark.security

MARKER = [123.456, 654.321, 111.111, 222.222, 333.333, 444.444, 555.555, 666.666]


@pytest.fixture
def app(tmp_path: Path, fake_openai: FakeOpenAIServer, clock: FakeClock) -> Iterator[FastAPI]:
    settings = tier3_settings(tmp_path, fake_openai, plaintext_database_for_tests=True)
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client


def vector_rows_of(app: FastAPI, record_id: str) -> int:
    row = app.state.store.execute(
        "SELECT count(*) FROM vectors WHERE record_id = ?", (record_id,)
    ).fetchone()
    return int(row[0])


def embed_result(report: dict[str, object]) -> dict[str, int]:
    jobs = report["jobs"]
    assert isinstance(jobs, list)
    result = next(j for j in jobs if j["kind"] == "embed")["result"]
    assert isinstance(result, dict)
    return result


def test_p5_a_forget_during_the_embed_call_leaves_no_vector(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "Karim's secret fact that will be forgotten"
    fake_openai.set_vectors({secret: MARKER})
    fact = remember(client, secret)
    fact_id = fact["fact"]["id"]
    store = app.state.store

    def forget_now(pending: list[PendingVector]) -> None:
        assert (fact_id, TEXT) in {(p.record_id, p.facet) for p in pending}
        store.forget(fact_id)  # exactly between the model call and the write

    monkeypatch.setattr(embed_module, "_after_embed", forget_now)
    result = embed_result(run_all(client, app))
    assert vector_rows_of(app, fact_id) == 0
    assert result["vectors_discarded"] >= 1
    store.checkpoint("TRUNCATE")
    assert vectors.pack(MARKER) not in _raw_bytes(Path(app.state.settings.db_path))


def test_a_text_change_during_the_embed_call_leaves_no_stale_vector(
    client: TestClient,
    app: FastAPI,
    fake_openai: FakeOpenAIServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, new = "Karim fixes roofs", "Karim leads the solar project"
    karim = remember(
        client, "Karim exists.", node={"name": "Karim", "kind": "person", "summary": old}
    )
    node_id = karim["node"]["id"]
    store = app.state.store

    def rewrite_now(pending: list[PendingVector]) -> None:
        if (node_id, SUMMARY) in {(p.record_id, p.facet) for p in pending}:
            store.update_node(node_id, summary=new)  # the old text is being embedded

    monkeypatch.setattr(embed_module, "_after_embed", rewrite_now)
    result = embed_result(run_all(client, app))
    # the old text's vector was never written; the job's next round, seeing the
    # summary pending again, computed the new text's vector
    digest = store.execute(
        "SELECT text_sha256 FROM vectors WHERE record_id = ? AND facet = ?", (node_id, SUMMARY)
    ).fetchone()[0]
    assert digest == hashlib.sha256(new.encode("utf-8")).hexdigest()
    assert result["vectors_discarded"] == 1
    assert (node_id, SUMMARY) not in {
        (p.record_id, p.facet) for p in store.pending_vectors("fake-embed", limit=50)
    }
