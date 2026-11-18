"""The assistant's name (SPEC 4.6): per scope, inherited by the scopes below,
the nearest declaration winning; ``Assistant`` when none is declared. It
labels the assistant part of a source and the ``from`` of its facts, never
one of the service's own labels nor the owner's name."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.api.app import build_pipeline
from iifa.config import ConfigError, load_settings
from iifa.ingest.extract import PipelineConfig
from iifa.ingest.parts import ASSISTANT, precompact_source, turn_source
from iifa.ingest.prompts import extraction_messages

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def test_no_assistant_name_by_default() -> None:
    settings = load_settings(ENV)
    assert settings.assistants == ()
    assert build_pipeline(settings).assistant(["user:alice"]) is None


def test_assistant_names_from_the_file_and_the_environment(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[extraction.assistants]
"user:alice" = "NOVA"
"user:bob" = { name = "Coder" }
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config)
    assert settings.assistants == (("user:alice", "NOVA"), ("user:bob", "Coder"))
    assert build_pipeline(settings).assistant(["user:alice/agent:coder"]) == "NOVA"
    env = load_settings({**ENV, "IIFA_ASSISTANTS": '{"user:alice": {"name": "Nova"}}'}, config)
    assert env.assistants == (("user:alice", "Nova"),)


def test_the_nearest_declaration_wins_and_a_split_source_has_none() -> None:
    pipeline = PipelineConfig(assistants=(("user:team", "Helper"), ("user:team/agent:x", "X")))
    assert pipeline.assistant(["user:team/agent:x/project:y"]) == "X"
    assert pipeline.assistant(["user:team/agent:z"]) == "Helper"
    assert pipeline.assistant(["user:team/agent:x", "user:team/agent:z"]) is None
    assert pipeline.assistant(["user:team", "user:carol"]) is None


@pytest.mark.parametrize(
    "table",
    [
        '"not a scope!" = "X"',
        '"user:x" = ""',
        '"user:x" = 3',
        '"user:x" = { aliases = ["X"] }',
        '"user:x" = "**NOVA**"',
        # the service's own labels: a quote under them could not be told apart
        '"user:x" = "User"',
        '"user:x" = "tool"',
        '"user:x" = "WEB"',
        '"user:x" = "Document"',
        '"user:x" = "External"',
    ],
)
def test_bad_assistant_names_are_refused(tmp_path: Path, table: str) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(f"[extraction.assistants]\n{table}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(ENV, config)


@pytest.mark.parametrize("name", ["Alice", "alice", "Алиса"])
def test_an_assistant_never_bears_the_owner_name_of_its_scope(tmp_path: Path, name: str) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        f"""
[extraction.owners]
"user:alice" = {{ name = "Alice", aliases = ["Алиса"] }}

[extraction.assistants]
"user:alice/agent:coder" = "{name}"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="owner"):
        load_settings(ENV, config)


def test_the_name_labels_the_assistant_part_and_rides_on_it() -> None:
    source = turn_source({"user": "hi", "assistant": "Hello."}, assistant="NOVA")
    assert source.text == "User: hi\n\nNOVA: Hello."
    reply = source.parts[1]
    assert (reply.provenance, reply.ref, reply.kind) == (ASSISTANT, "NOVA", None)
    assert source.part_of((reply.label_start or 0, reply.end)) == reply
    default = turn_source({"user": "hi", "assistant": "Hello."})
    assert default.text == "User: hi\n\nAssistant: Hello." and default.parts[1].ref is None
    compacted = precompact_source(
        {"messages": [{"role": "assistant", "content": "Done."}]}, assistant="NOVA"
    )
    assert compacted.text == "NOVA: Done." and compacted.parts[0].ref == "NOVA"


def test_the_prompt_names_the_assistant_only_when_one_is_declared() -> None:
    plain = extraction_messages("User: hi", languages=["en"])
    named = extraction_messages("User: hi", languages=["en"], assistant="NOVA")
    assert 'labelled "NOVA:"' in named[0]["content"]
    assert "NOVA" not in plain[0]["content"]
    paragraphs = named[0]["content"].split("\n\n")
    assert "\n\n".join(p for p in paragraphs if "NOVA" not in p) == plain[0]["content"]


def test_a_root_declaration_and_an_owner_of_another_name_are_accepted(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[extraction.owners]
"user:alice" = "Alice"

[extraction.assistants]
"*" = "NOVA"
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config)
    assert build_pipeline(settings).assistant(["user:alice"]) == "NOVA"


@pytest.mark.parametrize("value", ["{not json", '["NOVA"]'])
def test_a_bad_environment_value_is_refused(value: str) -> None:
    with pytest.raises(ConfigError, match="assistants"):
        load_settings({**ENV, "IIFA_ASSISTANTS": value})
