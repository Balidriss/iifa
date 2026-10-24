"""Settings for the speaker (SPEC 4.6): per-scope owners, inherited
by the scopes below, the nearest declaration winning; and the ``review``
permission (SPEC 4.4b, 8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.api.app import build_pipeline
from iifa.config import PERMISSIONS, ConfigError, load_settings
from iifa.ingest.extract import Owner, PipelineConfig

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def test_no_owner_by_default() -> None:
    settings = load_settings(ENV)
    assert settings.owners == ()
    assert build_pipeline(settings).owner(["user:alice"]) is None


def test_owners_from_the_file_and_the_environment(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[extraction.owners]
"user:alice" = { name = "Alice", aliases = ["Alicia", "Алиса"] }
"user:bob" = "Bob"
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config)
    assert settings.owners == (
        ("user:alice", "Alice", ("Alicia", "Алиса")),
        ("user:bob", "Bob", ()),
    )
    pipeline = build_pipeline(settings)
    assert pipeline.owner(["user:alice/agent:coder"]) == Owner(
        "user:alice", "Alice", ("Alicia", "Алиса")
    )
    env = load_settings({**ENV, "IIFA_OWNERS": '{"user:alice": {"name": "Alice"}}'}, config)
    assert env.owners == (("user:alice", "Alice", ()),)


def test_the_nearest_declaration_wins_and_a_split_turn_has_no_owner() -> None:
    pipeline = PipelineConfig(
        owners=(
            ("user:team", "Team lead", ()),
            ("user:team/agent:nadia", "Nadia", ()),
            ("user:bob", "Bob", ()),
        )
    )
    assert pipeline.owner(["user:team/agent:nadia/project:x"]) == Owner(
        "user:team/agent:nadia", "Nadia", ()
    )
    assert pipeline.owner(["user:team/agent:other"]) == Owner("user:team", "Team lead", ())
    assert pipeline.owner(["user:team/agent:nadia", "user:team/agent:nadia/x"]) is not None
    # two owners, or one scope without any: who speaks cannot be told
    assert pipeline.owner(["user:team/agent:nadia", "user:bob"]) is None
    assert pipeline.owner(["user:bob", "user:carol"]) is None
    assert pipeline.owner(["user:carol"]) is None


@pytest.mark.parametrize(
    "table",
    [
        '"not a scope!" = "X"',
        '"user:x" = ""',
        '"user:x" = { aliases = ["X"] }',
        '"user:x" = { name = "X", aliases = "X" }',
        '"user:x" = 3',
    ],
)
def test_bad_owners_are_refused(tmp_path: Path, table: str) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(f"[extraction.owners]\n{table}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(ENV, config)


def test_the_replacement_question_is_on_by_default(tmp_path: Path) -> None:
    """SPEC 5.4: on by default since the quotes (measured on qwen3.5:9b); off only
    by choice."""
    assert load_settings(ENV).replacement_question is True
    assert build_pipeline(load_settings(ENV)).replacement_question is True
    config = tmp_path / "iifa.toml"
    config.write_text("[extraction]\nreplacement_question = false\n", encoding="utf-8")
    assert build_pipeline(load_settings(ENV, config)).replacement_question is False
    env = load_settings({**ENV, "IIFA_REPLACEMENT_QUESTION": "off"})
    assert env.replacement_question is False


def test_review_is_a_permission_of_its_own() -> None:
    assert "review" in PERMISSIONS
