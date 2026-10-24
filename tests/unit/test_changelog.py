"""The changelog is for users: an "Unreleased" section, then one per release."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION_HEADING = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?: — .+)?")


def sections(path: Path) -> list[str]:
    return re.findall(r"^## (.+?)\s*$", path.read_text(encoding="utf-8"), re.M)


def test_the_changelog_holds_unreleased_then_releases_only() -> None:
    headings = sections(ROOT / "CHANGELOG.md")
    assert headings[0] == "Unreleased"
    assert headings[1:], "no release section"
    for heading in headings[1:]:
        assert VERSION_HEADING.fullmatch(heading), f"not a release section: {heading}"
