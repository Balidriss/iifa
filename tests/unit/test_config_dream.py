"""Settings of the dream (SPEC 5.6): its own model endpoint, the
probe tolerance, the default budget and the review policy per scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.config import ConfigError, load_settings
from tests.helpers import make_settings

ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def test_no_dream_model_by_default() -> None:
    settings = load_settings(ENV)
    assert (settings.dream_url, settings.dream_model) == (None, None)
    assert settings.dream_probe_tolerance == 0.02
    assert settings.dream_review == ()


def test_the_dream_section_of_the_file(tmp_path: Path) -> None:
    config = tmp_path / "iifa.toml"
    config.write_text(
        """
[dream]
url = "http://127.0.0.1:11434/v1"
model = "qwen3.5:9b"
answer_max_tokens = 1024
probe_tolerance = 0.05
extra = { think = false }

[dream.review]
"user:alice" = { medium = "review" }
"user:alice/project:solar" = { high = "auto", low = "review" }
""",
        encoding="utf-8",
    )
    settings = load_settings(ENV, config)
    assert (settings.dream_url, settings.dream_model) == ("http://127.0.0.1:11434/v1", "qwen3.5:9b")
    assert settings.dream_answer_max_tokens == 1024
    assert settings.dream_probe_tolerance == 0.05
    assert settings.dream_extra == {"think": False}
    assert settings.dream_review == (
        ("user:alice", "medium", "review"),
        ("user:alice/project:solar", "high", "auto"),
        ("user:alice/project:solar", "low", "review"),
    )


def test_the_dream_model_from_the_environment() -> None:
    settings = load_settings(
        {**ENV, "IIFA_DREAM_URL": "http://127.0.0.1:9/v1", "IIFA_DREAM_MODEL": "big"}
    )
    assert (settings.dream_url, settings.dream_model) == ("http://127.0.0.1:9/v1", "big")


def test_the_dream_endpoint_needs_both_url_and_model(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dream"):
        make_settings(tmp_path, dream_url="http://127.0.0.1:9/v1")
    with pytest.raises(ConfigError, match="dream"):
        make_settings(tmp_path, dream_model="big")


@pytest.mark.parametrize(
    "review",
    [
        (("user:alice", "huge", "auto"),),
        (("user:alice", "high", "sometimes"),),
        (("not a scope", "high", "auto"),),
    ],
)
def test_a_malformed_review_policy_is_refused(
    tmp_path: Path, review: tuple[tuple[str, str, str], ...]
) -> None:
    with pytest.raises(ConfigError, match="review"):
        make_settings(tmp_path, dream_review=review)


@pytest.mark.parametrize("tolerance", [-0.1, 1.5])
def test_the_probe_tolerance_is_a_fraction(tmp_path: Path, tolerance: float) -> None:
    with pytest.raises(ConfigError, match="tolerance"):
        make_settings(tmp_path, dream_probe_tolerance=tolerance)
