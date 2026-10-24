"""``bench/key_names.py`` (SPEC 5.4 step 4 calibration) runs end to end against
the fake embeddings server and says whether the two groups are apart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bench import key_names

from tests.fakes.openai_server import FakeOpenAIServer


def run(fake: FakeOpenAIServer, tmp_path: Path) -> dict[str, Any]:
    target = tmp_path / "keys.json"
    argv = ["--embeddings-url", fake.base_url, "--model", "fake-embed", "--json", str(target)]
    assert key_names.main(argv) == 0
    report: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return report


def groups() -> dict[str, int]:
    """Words joined by a "same" pair share a group; every trap joins two groups."""
    parent: dict[str, str] = {}

    def find(word: str) -> str:
        parent.setdefault(word, word)
        while parent[word] != word:
            word = parent[word]
        return word

    for a, b in key_names.SAME:
        parent[find(a)] = find(b)
    for a, b in key_names.TRAPS:
        find(a), find(b)
    roots = sorted({find(w) for w in parent})
    return {w: roots.index(find(w)) for w in parent}


def one_hot(index: int, size: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def test_the_script_writes_every_number_with_its_command(
    fake_openai: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = run(fake_openai, tmp_path)
    assert report["model"] == "fake-embed"
    assert report["command"].startswith("python bench/key_names.py --embeddings-url")
    assert len(report["same"]) == len(key_names.SAME)
    assert len(report["traps"]) == len(key_names.TRAPS)
    assert len(fake_openai.embedding_calls) == 1  # one call, key names only
    assert set(fake_openai.embedding_calls[0].body["input"]) == set(report["inputs"])
    assert "gap" in capsys.readouterr().out


def test_apart_groups_are_separated(fake_openai: FakeOpenAIServer, tmp_path: Path) -> None:
    group = groups()
    assert all(group[a] != group[b] for a, b in key_names.TRAPS)  # the traps are traps
    size = max(group.values()) + 1
    fake_openai.set_vectors({w: one_hot(g, size) for w, g in group.items()})
    report = run(fake_openai, tmp_path)
    assert report["lowest_same"] == pytest.approx(1.0)
    assert report["highest_trap"] == pytest.approx(0.0)
    assert report["separated"] is True


def test_overlapping_groups_are_not(fake_openai: FakeOpenAIServer, tmp_path: Path) -> None:
    fake_openai.set_vectors({w: one_hot(0, 8) for w in groups()})
    report = run(fake_openai, tmp_path)
    assert report["gap"] == pytest.approx(0.0) and report["separated"] is False
