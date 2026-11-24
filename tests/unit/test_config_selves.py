"""The self: per scope, inherited by the scopes below, the nearest
declaration winning; never on ``*``. A self is the assistant of its scope:
its name labels the assistant part, and it obeys the assistant's rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.api.app import build_pipeline
from iifa.config import ConfigError, load_settings
from iifa.ingest.extract import PipelineConfig
from iifa.ingest.parts import turn_source

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def test_no_self_by_default_is_the_user_perspective() -> None:
    settings = load_settings(ENV)
    assert settings.selves == ()
    assert build_pipeline(settings).agent(["user:alice"]) is None


def test_selves_from_the_file_and_the_environment(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[extraction.selves]
"user:alice" = { name = "NOVA", aliases = ["Nova"] }
"user:bob" = "Coder"
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config)
    assert settings.selves == (("user:alice", "NOVA", ("Nova",)), ("user:bob", "Coder", ()))
    pipeline = build_pipeline(settings)
    agent = pipeline.agent(["user:alice/agent:coder"])
    assert agent is not None and (agent.scope, agent.name, agent.aliases) == (
        "user:alice",
        "NOVA",
        ("Nova",),
    )
    # the self is the assistant of its scope
    assert pipeline.assistant(["user:alice/agent:coder"]) == "NOVA"
    env = load_settings({**ENV, "IIFA_SELVES": '{"user:alice": {"name": "Nova"}}'}, config)
    assert env.selves == (("user:alice", "Nova", ()),)


def test_the_nearest_self_wins_and_a_split_source_has_none() -> None:
    pipeline = PipelineConfig(selves=(("user:team", "Helper", ()), ("user:team/agent:x", "X", ())))
    nearest = pipeline.agent(["user:team/agent:x/project:y"])
    assert nearest is not None and nearest.name == "X"
    assert pipeline.agent(["user:team/agent:x", "user:team/agent:z"]) is None
    assert pipeline.agent(["user:team", "user:carol"]) is None
    # a self nearer than an assistant name labels the part, and the reverse
    mixed = PipelineConfig(
        assistants=(("user:team", "Bot"),), selves=(("user:team/agent:x", "X", ()),)
    )
    assert mixed.assistant(["user:team/agent:x"]) == "X"
    assert mixed.assistant(["user:team/agent:y"]) == "Bot"


def test_the_self_labels_the_assistant_part() -> None:
    source = turn_source({"user": "Hi", "assistant": "Hello"}, "Alice", "NOVA")
    assert source.text == "Alice: Hi\n\nNOVA: Hello"


@pytest.mark.parametrize(
    "table",
    [
        '"*" = "NOVA"',
        '"not a scope!" = "X"',
        '"user:x" = ""',
        '"user:x" = 3',
        '"user:x" = { aliases = ["X"] }',
        '"user:x" = { name = "X", aliases = [""] }',
        '"user:x" = "**NOVA**"',
        '"user:x" = "User"',
        '"user:x" = { name = "X", aliases = ["Web"] }',
    ],
)
def test_a_malformed_self_is_refused(tmp_path: Path, table: str) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(f"[extraction.selves]\n{table}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(ENV, config)


def test_a_self_named_like_the_owner_or_another_assistant_is_refused(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        '[extraction.owners]\n"user:alice" = { name = "Alice", aliases = ["Ali"] }\n'
        '[extraction.selves]\n"user:alice/agent:x" = { name = "X", aliases = ["Ali"] }\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="owner"):
        load_settings(ENV, config)
    config.write_text(
        '[extraction.assistants]\n"user:alice" = "Bot"\n'
        '[extraction.selves]\n"user:alice" = "NOVA"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="assistant"):
        load_settings(ENV, config)
    # the same name on the same scope is no contradiction
    config.write_text(
        '[extraction.assistants]\n"user:alice" = "NOVA"\n'
        '[extraction.selves]\n"user:alice" = "NOVA"\n',
        encoding="utf-8",
    )
    assert load_settings(ENV, config).selves == (("user:alice", "NOVA", ()),)
