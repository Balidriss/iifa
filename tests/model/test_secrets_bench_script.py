"""The secrets bench (``bench/secrets_bench.py``) runs end to end against the
fake chat server, so a run on a real model starts from a script whose counts
are known to be right.

The fake is a model at its worst for P5: it copies each turn into a fact and
names no secret. With the service as it ships, only a key's shape keeps a
value out, so every other secret leaks; with ``--wording`` the service reads
the wording too. Neither mode may erase the trap turns' ordinary words."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.fakes.openai_server import FakeOpenAIServer, Reply, Respond, chat_json

SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "secrets_bench.py"
SOURCE = re.compile(r"<<<SOURCE>>>\n(.*)\n<<<END>>>", re.DOTALL)
OWNER_LINE = re.compile(r"^Alice: (.*)$", re.MULTILINE)


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location("secrets_bench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["secrets_bench"] = module
    spec.loader.exec_module(module)
    return module


def copier(body: Any) -> Reply:
    """Extraction calls: one fact that copies the owner's words; anything else: nothing."""
    content = str(body["messages"][-1]["content"])
    source = SOURCE.search(content)
    owner = OWNER_LINE.search(source.group(1)) if source else None
    if owner is None:
        return chat_json({"nodes": [], "facts": [], "edges": []})
    said = owner.group(1).strip()
    fact = {
        "subject": "Alice",
        "statement": f"Alice said: {said}",
        "quote": said,
        "modality": "fact",
        "duration": "lasting",
    }
    return chat_json({"nodes": [], "facts": [fact], "edges": [], "frame": "sincere"})


def bench(module: Any, fake: FakeOpenAIServer, tmp_path: Path, *extra: str) -> dict[str, Any]:
    argv = [
        *("--chat-url", fake.base_url, "--model", "fake-chat", "--languages", "en"),
        *("--out", str(tmp_path), "--quiet", *extra),
    ]
    assert module.main(argv) == 0
    (target,) = [p for p in tmp_path.glob("*-secrets-*.json") if p.stat().st_size]
    report: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    target.unlink()
    return report


def rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in report["rows"]}


def test_the_bench_counts_leaks_and_erasures_in_both_modes(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    module = load_script()
    fake_openai.set_default(chat=Respond(copier))

    shipped = bench(module, fake_openai, tmp_path)
    assert shipped["detect_secrets_by_wording"] is False and not shipped["missing"]
    assert shipped["command"].startswith("python bench/secrets_bench.py --chat-url")
    by_key = rows(shipped)
    assert not by_key["en-key"]["leaked"]  # a key's shape is always read
    assert by_key["en-symbols"]["leaked"]  # nothing else, when the model names nothing
    assert "facts" in by_key["en-symbols"]["leaked"]["Kx7!pq2rWm"]
    assert by_key["en-word"]["leaked"]
    assert shipped["summary"]["total"]["erased_cases"] == 0
    assert "leaked:" in capsys.readouterr().out

    wording = bench(module, fake_openai, tmp_path, "--wording")
    assert wording["detect_secrets_by_wording"] is True
    by_key = rows(wording)
    assert not by_key["en-symbols"]["leaked"] and not by_key["en-digits"]["leaked"]
    assert not by_key["en-mixed"]["leaked"] and not by_key["en-mixed"]["erased"]
    # the copier wrote the password and the city in one fact: it went as a whole
    assert by_key["en-mixed"]["lost"] == ["Bristol"]
    assert by_key["en-word"]["leaked"]  # a plain word is the model's to name
    assert wording["summary"]["total"]["erased_cases"] == 0  # no trap erased
    assert by_key["en-page"]["erased"] == [] and by_key["en-page"]["lost"] == []  # P6
    for key, row in by_key.items():
        assert row["trace"] or row["kind"] == "page", key


def test_the_options_of_the_server_wrapper_are_accepted(tmp_path: Path) -> None:
    module = load_script()
    args = module.parse_args(
        [
            *("--chat-url", "http://127.0.0.1:1/v1", "--model", "m"),
            *("--embeddings-url", "http://127.0.0.1:1/v1", "--embeddings-model", "e"),
            *("--no-json-mode", "--extra", "reasoning_effort=none", "--repeat", "1"),
            *("--timeout", "600", "--out", str(tmp_path)),
        ]
    )
    assert args.extra == {"reasoning_effort": "none"} and args.json_mode is False
    with pytest.raises(SystemExit):
        module.parse_args(["--chat-url", "http://x/v1", "--model", "m", "--languages", "xx"])
