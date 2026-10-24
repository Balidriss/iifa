"""Settings of the background checkpoint interval and the MCP options."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.config import ConfigError, Settings, load_settings

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def test_defaults() -> None:
    settings = load_settings(ENV)
    assert settings.checkpoint_interval == 0.5
    assert settings.mcp_http is True
    assert settings.mcp_expose_forget is False
    assert settings.mcp_default_scope is None


def test_environment_flags_and_scope() -> None:
    settings = load_settings(
        {
            **ENV,
            "IIFA_MCP_HTTP": "off",
            "IIFA_MCP_EXPOSE_FORGET": "yes",
            "IIFA_MCP_DEFAULT_SCOPE": "user:alice/agent:coder",
        }
    )
    assert settings.mcp_http is False
    assert settings.mcp_expose_forget is True
    assert settings.mcp_default_scope == "user:alice/agent:coder"
    with pytest.raises(ConfigError, match="not a boolean"):
        load_settings({**ENV, "IIFA_MCP_HTTP": "maybe"})


def test_toml_sections(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        "[store]\ncheckpoint_interval = 0\n[mcp]\nhttp = false\nexpose_forget = true\n"
        'default_scope = "user:bob"\n',
        encoding="utf-8",
    )
    settings = load_settings(ENV, config_file=config)
    assert settings.checkpoint_interval == 0
    assert (settings.mcp_http, settings.mcp_expose_forget) == (False, True)
    assert settings.mcp_default_scope == "user:bob"
    # the environment wins over the file
    assert load_settings({**ENV, "IIFA_MCP_HTTP": "1"}, config_file=config).mcp_http is True


def test_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="checkpoint_interval"):
        Settings(db_path=tmp_path / "x.db", db_key="k", checkpoint_interval=-1)
    with pytest.raises(ConfigError, match="default_scope"):
        Settings(db_path=tmp_path / "x.db", db_key="k", mcp_default_scope="not a scope")
