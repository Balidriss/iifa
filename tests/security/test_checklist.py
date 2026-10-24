"""SPEC 12 stays honest: every box is ticked, each names at least one test, and
each test named exists where the box says."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.security


def _checklist() -> list[tuple[str, list[str]]]:
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    section = spec.split("## 12. Security and privacy checklist", 1)[1].split("\n---", 1)[0]
    boxes: list[tuple[str, list[str]]] = []
    for line in section.splitlines():
        if line.startswith("- ["):
            boxes.append((line, []))
        elif line.startswith("  - `") and boxes:
            boxes[-1][1].append(line.strip()[3:-1])
    return boxes


def test_every_box_is_ticked_with_the_tests_that_prove_it() -> None:
    boxes = _checklist()
    assert len(boxes) >= 18
    for box, proofs in boxes:
        assert box.startswith("- [x] "), box
        assert proofs, f"no test named for: {box}"
        for proof in proofs:
            path, name = proof.split("::")
            source = (ROOT / path).read_text(encoding="utf-8")
            assert re.search(rf"^def {name}\(", source, re.MULTILINE), proof
