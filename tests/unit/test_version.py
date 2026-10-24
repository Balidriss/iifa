"""One version everywhere: the package, its metadata, the API, the adapters, the docs."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

import iifa

ROOT = Path(__file__).resolve().parents[2]


def semver(version: str) -> str:
    """PEP 440 pre-release (0.1.0a1) as SemVer (0.1.0-alpha.1), for npm-style manifests."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?", version)
    assert match, f"not a release or pre-release version: {version}"
    base, tag, number = match.groups()
    if tag is None:
        return base
    label = {"a": "alpha", "b": "beta", "rc": "rc"}[tag]
    return f"{base}-{label}.{number}"


def test_semver_mapping() -> None:
    assert semver("0.1.0") == "0.1.0"
    assert semver("0.1.0a1") == "0.1.0-alpha.1"
    assert semver("1.2.3rc4") == "1.2.3-rc.4"


def test_the_package_is_the_pre_alpha() -> None:
    assert iifa.__version__ == "0.1.0a1"


def test_pyproject_matches_the_package() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == iifa.__version__
    assert "Development Status :: 2 - Pre-Alpha" in project["classifiers"]


def test_health_reports_the_package_version(client: TestClient) -> None:
    assert client.get("/v1/health").json()["version"] == iifa.__version__


def test_adapter_manifests_match_the_package() -> None:
    adapters = ROOT / "adapters"
    plugin = json.loads(
        (adapters / "claude-code" / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    package = json.loads((adapters / "opencode" / "package.json").read_text(encoding="utf-8"))
    hermes = (adapters / "hermes" / "iifa" / "plugin.yaml").read_text(encoding="utf-8")
    assert plugin["version"] == semver(iifa.__version__)
    assert package["version"] == semver(iifa.__version__)
    declared = re.search(r"^version: (\S+)$", hermes, re.M)
    assert declared and declared.group(1) == iifa.__version__


def test_changelog_and_readme_name_the_version() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    releases = re.findall(r"^## (\d\S*)", changelog, re.M)
    assert releases and releases[0] == iifa.__version__
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert iifa.__version__ in readme and "pre-alpha" in readme.lower()
