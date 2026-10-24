"""Per-model threshold profiles (SPEC 6.1): a cosine means something for one model
only, so thresholds come from a profile keyed by the embedding model, from the
configuration, or not at all (uncalibrated: semantic resolution, merges and
links off)."""

from __future__ import annotations

import pytest

from iifa.calibration import PROFILES, UNCALIBRATED, thresholds_for


@pytest.mark.parametrize("model", ["bge-m3", "bge-m3:latest", "Bge-m3:LATEST"])
def test_bge_m3_has_the_calibrated_profile(model: str) -> None:
    """bench/thresholds.py: resolve = 95th percentile of other entities, merge
    above every near-name trap (Bogdan / Bogdana 0.84); link provisional."""
    thresholds = thresholds_for(model)
    assert thresholds is not None
    assert (thresholds.resolve, thresholds.merge, thresholds.link) == (0.53, 0.84, 0.70)
    assert thresholds.source == "profile:bge-m3" and thresholds.provisional


def test_qwen3_embedding_has_its_own_profile() -> None:
    thresholds = thresholds_for("qwen3-embedding:0.6b")
    assert thresholds is not None and thresholds.provisional
    # its names sit higher: its traps reach 0.97 (Bogdan / Bogdana, Niger / Nigeria)
    assert (thresholds.resolve, thresholds.merge) == (0.84, 0.98)
    assert thresholds_for("qwen3-embedding:4b") is None  # another size, another geometry


def test_an_unknown_model_is_uncalibrated() -> None:
    assert thresholds_for("nomic-embed-text") is None
    assert UNCALIBRATED == "uncalibrated"


def test_configured_thresholds_override_or_enable() -> None:
    configured = thresholds_for("nomic-embed-text", resolve=0.7, merge=0.9, link=0.75)
    assert configured is not None and configured.source == "config"
    assert not configured.provisional
    # a partial override keeps the profile's other values
    partial = thresholds_for("bge-m3", merge=0.65)
    assert partial is not None and (partial.resolve, partial.merge, partial.link) == (
        0.53,
        0.65,
        0.70,
    )
    assert partial.source == "profile:bge-m3+config"
    # an unknown model needs all three
    assert thresholds_for("nomic-embed-text", resolve=0.7) is None


def test_every_profile_is_coherent() -> None:
    for name, profile in PROFILES.items():
        assert 0 < profile.resolve <= profile.merge <= 1, name
        assert 0 < profile.link <= 1 and profile.provisional, name
