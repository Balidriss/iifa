"""Settings of tier 3 (SPEC 6.1): the query embedding's timeout and switch,
and the model-dependent thresholds of semantic resolution, merges and links;
and the foreground priority of the gate (P11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.api.app import build_pipeline
from iifa.config import ConfigError, load_settings
from iifa.ingest.worker import Gate

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def test_defaults() -> None:
    settings = load_settings(ENV)
    assert settings.recall_dense is True and settings.query_timeout == 1.5
    # no global thresholds: they come from a per-model profile (SPEC 6.1)
    assert settings.resolve_threshold is None and settings.merge_threshold is None
    assert settings.link_threshold is None and settings.embed_batch == 32


def test_toml_and_environment(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[embeddings]
recall_dense = false
query_timeout = 0.8
resolve_threshold = 0.7
merge_threshold = 0.95
link_threshold = 0.6
batch = 8
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config)
    assert settings.recall_dense is False and settings.query_timeout == 0.8
    assert (settings.resolve_threshold, settings.merge_threshold) == (0.7, 0.95)
    assert settings.link_threshold == 0.6 and settings.embed_batch == 8
    pipeline = build_pipeline(settings)
    assert pipeline.resolve_threshold == 0.7 and pipeline.merge_threshold == 0.95
    assert pipeline.link_threshold == 0.6 and pipeline.embed_batch == 8
    env = load_settings(
        {**ENV, "IIFA_RECALL_DENSE": "off", "IIFA_EMBEDDINGS_QUERY_TIMEOUT": "2.5"}, config
    )
    assert env.recall_dense is False and env.query_timeout == 2.5


@pytest.mark.parametrize(
    "table",
    [
        "query_timeout = 0",
        "resolve_threshold = 1.5",
        "merge_threshold = 0.5\nresolve_threshold = 0.8",  # a merge must be stricter
        "link_threshold = 0",
        "batch = 0",
    ],
)
def test_bad_values_are_refused(tmp_path: Path, table: str) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(f"[embeddings]\n{table}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(ENV, config)


# --------------------------------------------------------------------------- the gate


def test_a_foreground_call_holds_new_background_calls() -> None:
    """P11 + SPEC 6.1: while the query embedding is in flight, no background call
    starts; one that already started is not interrupted."""
    gate = Gate({"chat": 1, "embeddings": 1})
    assert gate.acquire("chat", timeout=0)  # a background call already running
    with gate.foreground():
        assert gate.foreground_active() == 1
        assert not gate.acquire("embeddings", timeout=0)
        assert not gate.acquire("chat", timeout=0)
        with gate.foreground():  # nested foreground calls count
            assert gate.foreground_active() == 2
        assert gate.foreground_active() == 1
    gate.release("chat")
    assert gate.foreground_active() == 0
    assert gate.acquire("embeddings", timeout=0)
    gate.release("embeddings")


def test_foreground_ends_even_when_the_call_fails() -> None:
    gate = Gate({"chat": 1, "embeddings": 1})
    with pytest.raises(RuntimeError), gate.foreground():
        raise RuntimeError("the query embedding failed")
    assert gate.foreground_active() == 0 and gate.acquire("chat", timeout=0)
