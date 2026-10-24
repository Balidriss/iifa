"""Every relative link in the docs points to a file, and to an anchor, that exist.

Anchors follow GitHub's rule: the heading's text, lower-cased, punctuation
removed, spaces turned into hyphens; a repeated heading gets -1, -2, ...
Links inside code (fenced blocks, inline spans) are not links and are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[2]

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
INLINE_CODE = re.compile(r"`[^`]*`")
TARGETS = [
    re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)"),  # [text](target)
    re.compile(r"^\s*\[[^\]]+\]:\s*<?(\S+?)>?(?:\s+\"[^\"]*\")?\s*$"),  # [ref]: target
    re.compile(r"(?:href|src)=\"([^\"]+)\""),  # raw HTML
]
EXPLICIT_ANCHOR = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"")
EXTERNAL = ("http://", "https://", "mailto:", "data:")


def documents() -> list[Path]:
    """The public pages a reader follows: README, changelog, docs/, the adapters'."""
    pages = [ROOT / "README.md", ROOT / "CHANGELOG.md"]
    pages += sorted((ROOT / "docs").rglob("*.md"))
    pages += sorted((ROOT / "adapters").glob("*/*.md"))
    return pages


def prose_lines(text: str) -> list[str]:
    """The lines outside fenced code blocks."""
    lines, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


def slug(heading: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # a link keeps its text
    text = text.replace("`", "").replace("*", "")
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    found: set[str] = set()
    for line in prose_lines(path.read_text(encoding="utf-8")):
        found.update(EXPLICIT_ANCHOR.findall(line))
        match = HEADING.match(line)
        if not match:
            continue
        base = slug(match.group(2))
        count = seen.get(base, 0)
        seen[base] = count + 1
        found.add(base if count == 0 else f"{base}-{count}")
    return found


def relative_links(path: Path) -> list[tuple[int, str, Path, str]]:
    """(line, target, destination file, anchor) for each relative link of a page."""
    links = []
    for number, raw in enumerate(prose_lines(path.read_text(encoding="utf-8")), 1):
        line = INLINE_CODE.sub("", raw)
        for pattern in TARGETS:
            for target in pattern.findall(line):
                if target.startswith(EXTERNAL):
                    continue
                file_part, _, anchor = target.partition("#")
                dest = (path.parent / unquote(file_part)).resolve() if file_part else path
                links.append((number, target, dest, anchor))
    return links


def broken_links(path: Path, root: Path = ROOT) -> list[str]:
    problems = []
    for number, target, dest, anchor in relative_links(path):
        where = f"{path.relative_to(root)} (line {number}): {target}"
        if not dest.exists():
            problems.append(f"{where}: no such file")
        elif anchor and dest.suffix != ".md":
            problems.append(f"{where}: an anchor into a file that is not Markdown")
        elif anchor and unquote(anchor) not in anchors(dest):
            problems.append(f"{where}: no such anchor")
    return problems


def test_the_docs_exist() -> None:
    names = {p.name for p in documents()}
    for expected in (
        "configuration.md",
        "api.md",
        "how-it-works.md",
        "docker.md",
        "security.md",
        "benchmarks.md",
    ):
        assert expected in names


@pytest.mark.parametrize("page", documents(), ids=lambda p: str(p.relative_to(ROOT)))
def test_every_relative_link_resolves(page: Path) -> None:
    assert broken_links(page) == []


# The checker itself: it must catch what it is for, so the guard cannot pass empty.


def test_slug_follows_github() -> None:
    assert slug("P1 — The budget is a contract") == "p1--the-budget-is-a-contract"
    assert slug("5.4 Contradictions (P7)") == "54-contradictions-p7"
    assert slug("The `recall` route") == "the-recall-route"
    assert slug("[Docker](docs/docker.md) image") == "docker-image"


def test_checker_catches_a_missing_file_and_a_missing_anchor(tmp_path: Path) -> None:
    (tmp_path / "other.md").write_text("# Title\n## Same\n## Same\n", encoding="utf-8")
    page = tmp_path / "page.md"
    lines = [
        "# Here",
        "[ok](other.md#title) [ok](other.md#same-1) [ok](#here) [web](https://x.y/#z)",
        "[gone](missing.md) [bad](other.md#nope) [bad](#nowhere) [bad](other.md#same-2)",
        "`[code](missing.md)`",
        "```",
        "[fenced](missing.md)",
        "```",
        '<img src="missing.png">',
    ]
    page.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = broken_links(page, root=tmp_path)
    assert len(problems) == 5, problems
    assert sum("no such file" in p for p in problems) == 2
    assert sum("no such anchor" in p for p in problems) == 3
