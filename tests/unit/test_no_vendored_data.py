"""Public benchmark datasets are downloaded, never vendored (SPEC 11): git
ignores ``bench/data/``, and nothing under it is tracked - LoCoMo is CC BY-NC
4.0, which an Apache-2.0 repository must not carry."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_gitignore_keeps_downloaded_datasets_out() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "bench/data/" in lines


def _git(*args: str) -> str:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    done = subprocess.run(
        [git, "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )
    return done.stdout


def test_a_downloaded_dataset_would_be_ignored_and_none_is_tracked() -> None:
    ignored = _git("check-ignore", "bench/data/locomo/locomo10.json")
    assert ignored.strip() == "bench/data/locomo/locomo10.json"
    assert _git("ls-files", "bench/data").strip() == ""
