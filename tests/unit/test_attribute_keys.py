"""Attribute keys (SPEC 5.4): canonical keys from a base vocabulary and a small
table of unambiguous variants, bare families that never supersede, and the
per-model threshold of the embedding match. When we cannot tell, we do not
supersede."""

from __future__ import annotations

from pathlib import Path

import pytest

from iifa.calibration import describe, key_threshold_for, thresholds_for
from iifa.config import ConfigError, load_settings
from iifa.ingest.keys import (
    BASE_KEYS,
    FAMILIES,
    VOCABULARY_TEXT,
    canonical_key,
    family_of,
    is_free_form,
    same_key,
    supersedable,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("home", "home"),
        ("Home", "home"),
        (" lives in ", "home"),
        ("lives_in", "home"),
        ("lives-in", "home"),
        ("residence", "home"),
        ("home_city", "home"),
        ("address", "home"),
        ("moved_to", "home"),
        ("workplace", "workplace"),
        ("work_location", "workplace"),
        ("place_of_work", "workplace"),
        ("role", "role"),
        ("occupation", "role"),
        ("profession", "role"),
        ("job_title", "role"),
        ("works_as", "role"),
        ("job", "role"),
        ("employer", "employer"),
        ("works_for", "employer"),
        ("company", "employer"),
        ("preference.drink", "preference.drink"),
        ("preference_drink", "preference.drink"),
        ("preferred_drink", "preference.drink"),
        ("drink_preference", "preference.drink"),
        ("favorite_drink", "preference.drink"),
        ("favourite drink", "preference.drink"),
        ("Preference.Answers", "preference.answers"),
        ("status.marital", "status.marital"),
        ("marital_status", "status.marital"),
        ("work_status", "status.work"),
        ("date.birth", "date.birth"),
        ("birth_date", "date.birth"),
        ("date_of_birth", "date.birth"),
        ("deadline_date", "date.deadline"),
        ("Rôle", "role"),
        ("hobby", "hobby"),
        ("team_name", "team_name"),
        ("Lieu de résidence", "lieu_de_residence"),
        ("город", "город"),
        ("職業", "職業"),
    ],
)
def test_canonical_keys(raw: str, canonical: str) -> None:
    assert canonical_key(raw) == canonical


@pytest.mark.parametrize(
    "ambiguous",
    [
        "location",
        "city",
        "current_city",
        "work",
        "title",
        "position",
        "works_at",
        "office",
        "based_in",
    ],
)
def test_ambiguous_variants_stay_free_form(ambiguous: str) -> None:
    """A home city or a work city, a job or a book title: we cannot tell, so the
    key stays its own and never closes a vocabulary key."""
    key = canonical_key(ambiguous)
    assert key == ambiguous and is_free_form(key)
    for vocabulary in ("home", "workplace", "role", "employer"):
        assert not same_key(ambiguous, vocabulary)


def test_where_one_lives_is_never_where_one_works() -> None:
    assert not same_key("home", "workplace")
    for work in ("workplace", "work_location", "office", "location", "city", "based_in"):
        assert not same_key("lives_in", work), work
    assert "home" in VOCABULARY_TEXT and "workplace" in VOCABULARY_TEXT
    assert "location" not in BASE_KEYS


@pytest.mark.parametrize("raw", [None, "", "   ", "---", "..."])
def test_no_key(raw: str | None) -> None:
    assert canonical_key(raw) is None


@pytest.mark.parametrize(
    "bare", ["preference", "prefers", "likes", "preferences", "status", "date"]
)
def test_a_bare_family_key_never_supersedes(bare: str) -> None:
    """ "prefers short answers" must not be closed by "prefers tea": a family
    key without a topic would make every preference one key."""
    key = canonical_key(bare)
    assert key in FAMILIES and family_of(key) == key
    assert not supersedable(key)
    assert not same_key("preference", "preference.drink")
    assert not same_key("preference.drink", "preference.answers")
    assert not same_key("status.marital", "status.work")


def test_what_may_supersede() -> None:
    assert supersedable("home") and supersedable("preference.drink") and supersedable("hobby")
    assert not supersedable(None)
    assert family_of("preference.drink") == "preference" and family_of("home") is None
    assert family_of(None) is None
    assert not is_free_form("home") and not is_free_form("preference.drink")
    assert not is_free_form("preference") and is_free_form("commute_mode")


def test_keys_compare_canonically_on_both_sides() -> None:
    assert same_key("lives_in", "Residence")
    assert same_key("job_title", "occupation")
    assert not same_key("home", "role")
    assert not same_key(None, "home") and not same_key(None, None)


def test_the_base_vocabulary_is_small_and_language_neutral() -> None:
    assert set(BASE_KEYS) == {"home", "workplace", "role", "employer", "team"}
    assert set(FAMILIES) == {"preference", "status", "date"}
    assert all(k.isascii() and k == k.lower() for k in (*BASE_KEYS, *FAMILIES))


def test_the_key_threshold_is_off_until_calibrated() -> None:
    """SPEC 5.4 step 4 and 6.1: no profile measures keys yet, so the step is off
    unless configured; configuring it does not calibrate the other thresholds."""
    assert key_threshold_for("bge-m3") is None
    assert key_threshold_for("nomic-embed-text") is None
    assert key_threshold_for("nomic-embed-text", configured=0.8) == 0.8
    assert key_threshold_for("bge-m3", configured=0.75) == 0.75
    assert thresholds_for("nomic-embed-text") is None
    assert describe(thresholds_for("bge-m3"), key=None)["key"] is None
    assert describe(None, key=0.8) == {"status": "uncalibrated", "key": 0.8}


ENV = {"IIFA_DB_KEY": "k", "IIFA_TOKEN": "t"}


def _toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "iifa.toml"
    path.write_text("[embeddings]\n" + body, encoding="utf-8")
    return path


def test_the_key_threshold_is_configured_and_validated(tmp_path: Path) -> None:
    assert load_settings(ENV).key_threshold is None
    settings = load_settings(ENV, _toml(tmp_path, "key_threshold = 0.8\n"))
    assert settings.key_threshold == 0.8
    with pytest.raises(ConfigError, match="key_threshold"):
        load_settings(ENV, _toml(tmp_path, "key_threshold = 1.5\n"))
