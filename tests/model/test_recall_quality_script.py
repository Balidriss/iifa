"""The tier-3 bench (``bench/recall_quality.py``) runs end to end against the
fake embeddings server, so a comparison of real models starts from a script
known to run and to write every number with its command."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.fakes.openai_server import FakeOpenAIServer

SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "recall_quality.py"


def load_script() -> object:
    spec = importlib.util.spec_from_file_location("recall_quality", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["recall_quality"] = module
    spec.loader.exec_module(module)
    return module


def test_the_bench_compares_tiers_and_writes_every_number(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    module = load_script()
    target = tmp_path / "bench.json"
    argv = [
        *("--embeddings-url", fake_openai.base_url, "--model", "fake-embed"),
        *("--repeat", "1", "--json", str(target)),
    ]
    assert module.main(argv) == 0  # type: ignore[attr-defined]
    out = capsys.readouterr().out
    assert "recall@1" in out and "tier0" in out and "tier3" in out and "latency tier3" in out
    assert "same person" in out and "gap" in out
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["model"] == "fake-embed" and report["build"]["records"] == 16
    assert report["build"]["dimension"] == 8 and report["vector_path"] == "numpy"
    assert set(report["quality"]) == {"tier0", "tier3"}
    assert all(p["tier"] == 3 for p in report["quality"]["tier3"]["probes"])
    assert report["latency"]["tier3"]["n"] == 12
    assert report["command"].startswith("python bench/recall_quality.py --embeddings-url")
    assert set(report["names"]["same_person"]) == {"Карим", "كريم", "卡里姆", "カリム"}
    assert all(c.model == "fake-embed" for c in fake_openai.embedding_calls)
    assert not fake_openai.chat_calls
