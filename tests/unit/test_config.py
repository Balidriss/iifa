"""Settings: key supplied by the operator, tokens hashed, defaults safe (P5, SPEC 12)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from iifa.config import (
    PERMISSIONS,
    ConfigError,
    Settings,
    TokenConfig,
    hash_token,
    load_settings,
)


def test_settings_refuse_to_start_without_a_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="never generates"):
        Settings(db_path=tmp_path / "x.db")
    with pytest.raises(ConfigError, match="never generates"):
        load_settings(env={"IIFA_DB_PATH": str(tmp_path / "x.db")})


def test_key_from_env_or_file(tmp_path: Path) -> None:
    key_file = tmp_path / "db.key"
    key_file.write_text("  file-secret \n", encoding="utf-8")
    assert load_settings(env={"IIFA_DB_KEY_FILE": str(key_file)}).db_key == "file-secret"
    assert (
        load_settings(env={"IIFA_DB_KEY": "env-secret", "IIFA_DB_KEY_FILE": str(key_file)}).db_key
        == "env-secret"
    )
    with pytest.raises(ConfigError, match="not found"):
        load_settings(env={"IIFA_DB_KEY_FILE": str(tmp_path / "missing")})


def test_key_never_appears_in_repr() -> None:
    settings = Settings(db_path=Path("x.db"), db_key="hunter2")
    assert "hunter2" not in repr(settings)


def test_default_bind_is_loopback_and_port_validated() -> None:
    settings = Settings(db_path=Path("x.db"), db_key="k")
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    with pytest.raises(ConfigError, match="port"):
        Settings(db_path=Path("x.db"), db_key="k", port=0)


def test_budget_defaults_validated() -> None:
    with pytest.raises(ConfigError, match="default_budget_tokens"):
        Settings(db_path=Path("x.db"), db_key="k", default_budget_tokens=0)
    with pytest.raises(ConfigError, match="default_budget_tokens"):
        Settings(
            db_path=Path("x.db"), db_key="k", default_budget_tokens=99999, max_budget_tokens=10
        )


def test_tokens_are_stored_hashed_and_validated() -> None:
    token = TokenConfig.from_plaintext(
        "secret", name="t", scopes=("user:alice",), permissions=frozenset({"read"})
    )
    assert token.sha256 == hashlib.sha256(b"secret").hexdigest() == hash_token("secret")
    assert "secret" not in repr(token)
    with pytest.raises(ConfigError, match="unknown permissions"):
        Settings(
            db_path=Path("x.db"),
            db_key="k",
            tokens=(TokenConfig("t", "00", ("*",), frozenset({"root"})),),
        )
    with pytest.raises(ConfigError, match="no scope"):
        Settings(
            db_path=Path("x.db"),
            db_key="k",
            tokens=(TokenConfig("t", "00", (), frozenset({"read"})),),
        )
    with pytest.raises(ConfigError, match="invalid scope"):
        Settings(
            db_path=Path("x.db"),
            db_key="k",
            tokens=(TokenConfig("t", "00", ("nope",), frozenset({"read"})),),
        )


def test_load_from_toml_and_env_tokens(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[service]
host = "127.0.0.1"
port = 9999

[store]
path = "mem.db"
retain_sources = false

[recall]
default_budget_tokens = 250
tokenizer = "conservative"

[[tokens]]
name = "agent"
token = "agent-secret"
scopes = ["user:alice"]
permissions = ["read", "write"]

[[tokens]]
name = "ops"
token_sha256 = "AB"
scopes = "*"
permissions = "admin"
""",
        encoding="utf-8",
    )
    settings = load_settings(
        env={
            "IIFA_CONFIG": str(config),
            "IIFA_DB_KEY": "k",
            "IIFA_TOKEN": "root-secret",
            "IIFA_TOKENS": '[{"name": "extra", "token": "x", "scopes": ["user:bob"], "permissions": ["forget"]}]',
        }
    )
    assert settings.port == 9999
    assert settings.db_path == Path("mem.db")
    assert settings.retain_sources is False
    assert settings.default_budget_tokens == 250
    names = [t.name for t in settings.tokens]
    assert names == ["agent", "ops", "extra", "env"]
    assert settings.tokens[0].sha256 == hash_token("agent-secret")
    assert settings.tokens[1].sha256 == "ab" and settings.tokens[1].scopes == ("*",)
    assert settings.tokens[1].permissions == frozenset({"admin"})
    assert settings.tokens[3].permissions == PERMISSIONS
    with pytest.raises(ConfigError, match="needs 'token'"):
        load_settings(env={"IIFA_DB_KEY": "k", "IIFA_TOKENS": '[{"name": "bad"}]'})
