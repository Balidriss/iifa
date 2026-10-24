"""Settings of the background mode and concurrency, the model
endpoints (explicit model, configured hosts only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.config import BACKGROUND_MODES, ConfigError, Settings, load_settings

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def test_defaults_are_turnkey_and_tier_zero() -> None:
    settings = load_settings(ENV)
    assert settings.background_mode == "auto" and BACKGROUND_MODES[0] == "auto"
    assert (settings.chat_concurrency, settings.embeddings_concurrency) == (1, 1)
    assert settings.yield_timeout == 30.0 and settings.max_attempts == 3
    assert settings.retry_delay == 30.0 and settings.model_timeout == 120.0
    assert settings.chat_url is None and settings.chat_model is None
    assert settings.embeddings_url is None and settings.embeddings_model is None


def test_environment_configures_mode_and_endpoints() -> None:
    settings = load_settings(
        {
            **ENV,
            "IIFA_BACKGROUND_MODE": "paused",
            "IIFA_CHAT_CONCURRENCY": "2",
            "IIFA_EMBEDDINGS_CONCURRENCY": "4",
            "IIFA_CHAT_URL": "http://localhost:11434/v1/",
            "IIFA_CHAT_MODEL": "llama3.2",
            "IIFA_CHAT_API_KEY": "sk-local",
            "IIFA_EMBEDDINGS_URL": "http://localhost:11434/v1",
            "IIFA_EMBEDDINGS_MODEL": "nomic-embed-text",
            "IIFA_MODEL_TIMEOUT": "45",
        }
    )
    assert settings.background_mode == "paused"
    assert (settings.chat_concurrency, settings.embeddings_concurrency) == (2, 4)
    assert settings.chat_url == "http://localhost:11434/v1/" and settings.chat_model == "llama3.2"
    assert settings.chat_api_key == "sk-local" and "sk-local" not in repr(settings)
    assert settings.embeddings_model == "nomic-embed-text" and settings.model_timeout == 45.0


def test_toml_sections(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[background]
mode = "on_demand"
chat_concurrency = 3
yield_timeout = 5.5
max_attempts = 2
retry_delay = 1

[models]
chat_url = "http://127.0.0.1:8080/v1"
chat_model = "qwen"
chat_api_key = "file-key"
timeout = 10
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config_file=config)
    assert settings.background_mode == "on_demand" and settings.chat_concurrency == 3
    assert settings.yield_timeout == 5.5 and settings.max_attempts == 2
    assert settings.retry_delay == 1.0 and settings.model_timeout == 10.0
    assert settings.chat_url == "http://127.0.0.1:8080/v1" and settings.chat_api_key == "file-key"
    assert settings.embeddings_url is None
    # the environment wins over the file
    assert (
        load_settings({**ENV, "IIFA_BACKGROUND_MODE": "auto"}, config_file=config).background_mode
        == "auto"
    )


def test_validation(tmp_path: Path) -> None:
    base = {"db_path": tmp_path / "x.db", "db_key": "k"}
    with pytest.raises(ConfigError, match="background mode"):
        Settings(**base, background_mode="sometimes")
    with pytest.raises(ConfigError, match="chat_concurrency"):
        Settings(**base, chat_concurrency=0)
    with pytest.raises(ConfigError, match="max_attempts"):
        Settings(**base, max_attempts=0)
    with pytest.raises(ConfigError, match="yield_timeout"):
        Settings(**base, yield_timeout=0)
    with pytest.raises(ConfigError, match="retry_delay"):
        Settings(**base, retry_delay=-1)
    with pytest.raises(ConfigError, match="explicit"):
        Settings(**base, chat_url="http://localhost:11434/v1")
    with pytest.raises(ConfigError, match="explicit"):
        Settings(**base, embeddings_model="nomic")
    with pytest.raises(ConfigError, match="http"):
        Settings(**base, chat_url="localhost:11434", chat_model="m")
    with pytest.raises(ConfigError, match="background mode"):
        load_settings({**ENV, "IIFA_BACKGROUND_MODE": "off"})
