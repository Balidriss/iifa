"""The release workflow: a version tag publishes that version's changelog notes."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def notes_script() -> str:
    """The Python the workflow runs to cut the notes out of CHANGELOG.md."""
    text = WORKFLOW.read_text(encoding="utf-8")
    block = re.search(r"<<'PY'\n(.*?)\n\s*PY\n", text, re.S)
    assert block, "no notes script in the release workflow"
    lines = block.group(1).splitlines()
    indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return "\n".join(line[indent:] for line in lines) + "\n"


def run_notes(version: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", version],
        input=notes_script(),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def package_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_the_workflow_runs_on_version_tags_and_can_publish_an_existing_tag() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r'tags: \["v\*"\]', text)
    assert "workflow_dispatch:" in text
    assert "contents: write" in text


def test_every_release_in_the_changelog_has_notes() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = re.findall(r"^## (\d\S*)", changelog, re.M)
    assert package_version() in versions
    for version in versions:
        proc = run_notes(version, ROOT)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip()
        assert not re.search(r"^## ", proc.stdout, re.M), "notes run into another section"


def test_the_notes_stop_at_the_next_release(tmp_path: Path) -> None:
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\n"
        "## 1.0.0b2 — 2030-01-02\n\n### Added\n- second\n\n"
        "## 1.0.0b1 — 2030-01-01\n\n### Added\n- first\n",
        encoding="utf-8",
    )
    proc = run_notes("1.0.0b2", tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "### Added\n- second"


def test_a_version_without_notes_is_refused(tmp_path: Path) -> None:
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\n## 1.0.0 — 2030-01-01\n\n## 0.9.0\n\n- old\n",
        encoding="utf-8",
    )
    assert run_notes("1.0.0", tmp_path).returncode != 0
    assert run_notes("2.0.0", tmp_path).returncode != 0
    # "1.0" must not match the "1.0.0" section
    assert run_notes("1.0", tmp_path).returncode != 0
