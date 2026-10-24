"""Settings for talking to a model (SPEC 7.4b): per-endpoint extra request parameters,
JSON mode on or off, and the answer cap."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.api.app import build_models, build_pipeline
from iifa.config import RESERVED_REQUEST_KEYS, ConfigError, load_settings
from iifa.ingest.worker import Gate

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}
CHAT = {"IIFA_CHAT_URL": "http://127.0.0.1:11434/v1", "IIFA_CHAT_MODEL": "qwen3.5:9b"}


def test_defaults_keep_json_mode_and_add_nothing() -> None:
    settings = load_settings(ENV)
    assert settings.chat_json_mode is True
    assert dict(settings.chat_extra) == {} and dict(settings.embeddings_extra) == {}
    assert settings.answer_max_tokens == 4096


def test_toml_configures_extras_json_mode_and_the_cap(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[models]
chat_url = "http://127.0.0.1:11434/v1"
chat_model = "qwen3.5:9b"
chat_json_mode = false

[models.chat_extra]
reasoning_effort = "none"
top_p = 0.9
options = { num_ctx = 8192 }

[models.embeddings_extra]
truncate = true

[extraction]
answer_max_tokens = 3000
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config)
    assert settings.chat_json_mode is False
    assert dict(settings.chat_extra) == {
        "reasoning_effort": "none",
        "top_p": 0.9,
        "options": {"num_ctx": 8192},
    }
    assert dict(settings.embeddings_extra) == {"truncate": True}
    assert settings.answer_max_tokens == 3000
    pipeline = build_pipeline(settings)
    assert pipeline.json_mode is False and pipeline.answer_max_tokens == 3000
    models = build_models(settings, Gate({"chat": 1, "embeddings": 1}))
    try:
        assert models.chat_endpoint is not None
        assert dict(models.chat_endpoint.extra)["reasoning_effort"] == "none"
    finally:
        models.close()


def test_environment_overrides_extras_and_json_mode() -> None:
    settings = load_settings(
        {
            **ENV,
            **CHAT,
            "IIFA_CHAT_EXTRA": '{"reasoning_effort": "none"}',
            "IIFA_EMBEDDINGS_EXTRA": '{"dimensions": 256}',
            "IIFA_CHAT_JSON_MODE": "off",
        }
    )
    assert dict(settings.chat_extra) == {"reasoning_effort": "none"}
    assert dict(settings.embeddings_extra) == {"dimensions": 256}
    assert settings.chat_json_mode is False


@pytest.mark.parametrize("key", sorted(RESERVED_REQUEST_KEYS))
def test_an_extra_may_not_replace_what_iifa_sets(key: str) -> None:
    with pytest.raises(ConfigError, match=key):
        load_settings({**ENV, **CHAT, "IIFA_CHAT_EXTRA": f'{{"{key}": 1}}'})


def test_reserved_keys_cover_the_request_bodies() -> None:
    assert {
        "model",
        "messages",
        "input",
        "stream",
        "max_tokens",
        "temperature",
        "response_format",
    } <= RESERVED_REQUEST_KEYS


@pytest.mark.parametrize(
    "value",
    ["[1, 2]", '"reasoning_effort"', "not json"],
)
def test_extras_must_be_a_json_object(value: str) -> None:
    with pytest.raises(ConfigError):
        load_settings({**ENV, **CHAT, "IIFA_CHAT_EXTRA": value})


def test_extras_must_be_serialisable(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text("[models.chat_extra]\nwhen = 2026-09-27T10:00:00Z\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        load_settings(ENV, config)


def test_the_cap_must_be_sane() -> None:
    with pytest.raises(ConfigError):
        load_settings({**ENV, "IIFA_ANSWER_MAX_TOKENS": "100"})
    with pytest.raises(ConfigError):
        load_settings({**ENV, "IIFA_CHAT_JSON_MODE": "maybe"})
