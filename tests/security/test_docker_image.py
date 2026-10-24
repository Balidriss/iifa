"""What the Docker image promises, read from the Dockerfile (SPEC 12, P5), on
every platform and without Docker. The running image is checked end to end by
``tests/e2e/docker_smoke.py`` in CI (the ``docker-e2e`` job)."""

from __future__ import annotations

import re
import shlex
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.security


def _instructions() -> list[tuple[str, str]]:
    """The final stage's instructions, continuation lines joined."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", " ", text)
    lines = [line.strip() for line in joined.splitlines()]
    steps = [tuple(line.split(None, 1)) for line in lines if line and not line.startswith("#")]
    final = max(i for i, step in enumerate(steps) if step[0].upper() == "FROM")
    return [(step[0].upper(), step[1] if len(step) > 1 else "") for step in steps[final:]]


def _env() -> dict[str, str]:
    env: dict[str, str] = {}
    for word, rest in _instructions():
        if word == "ENV":
            for item in shlex.split(rest):
                name, value = item.split("=", 1)
                env[name] = value
    return env


def test_p5_the_image_runs_as_a_non_root_user() -> None:
    (user,) = [rest for word, rest in _instructions() if word == "USER"]
    uid = user.split(":")[0]
    assert uid.isdigit() and int(uid) >= 1000  # numeric: Kubernetes' runAsNonRoot can check it


def test_p5_no_key_is_baked_into_the_image() -> None:
    env = _env()
    assert "IIFA_DB_KEY" not in env
    for word, rest in _instructions():
        if word in ("ARG", "ENV", "LABEL"):
            assert "IIFA_DB_KEY=" not in rest
        if word in ("COPY", "ADD"):
            assert "key" not in rest.lower(), rest


def test_p5_the_key_and_the_database_are_on_separate_paths() -> None:
    env = _env()
    db_dir = PurePosixPath(env["IIFA_DB_PATH"]).parent
    key_dir = PurePosixPath(env["IIFA_DB_KEY_FILE"]).parent
    assert db_dir != key_dir
    assert db_dir not in key_dir.parents and key_dir not in db_dir.parents
    (volumes,) = [rest for word, rest in _instructions() if word == "VOLUME"]
    assert str(db_dir) in volumes and str(key_dir) not in volumes  # the key is mounted, never kept


def test_the_default_bind_inside_the_container_is_loopback() -> None:
    assert _env()["IIFA_HOST"] == "127.0.0.1"


def test_the_build_context_holds_the_package_and_nothing_else() -> None:
    """Local files (a database, a key, notes, a virtualenv) never reach the image."""
    lines = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "*"
    allowed = {line for line in lines[1:] if line.startswith("!")}
    assert allowed == {"!pyproject.toml", "!README.md", "!LICENSE", "!src/"}


def test_ci_runs_the_end_to_end_check_on_the_image() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "docker build -t iifa:e2e ." in workflow
    assert "python tests/e2e/docker_smoke.py --image iifa:e2e" in workflow
