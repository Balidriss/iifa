"""``bench/thresholds.py`` (SPEC 6.1 calibration): the rule that picks
``merge`` and ``resolve`` from positive, negative and trap cosines, and the
script end to end against the fake embeddings server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bench import thresholds

from tests.fakes.openai_server import FakeOpenAIServer

# --------------------------------------------------------------------------- the rule


def test_merge_keeps_every_trap_out_and_at_most_two_percent_of_negatives() -> None:
    negatives = [i / 100 for i in range(100)]  # 0.00 .. 0.99
    chosen = thresholds.choose(positives=[0.995, 0.999], negatives=negatives, traps=[0.40])
    assert chosen["profile"] is True
    merge = chosen["merge"]
    assert merge > 0.40  # every trap below
    above = sum(1 for n in negatives if n >= merge) / len(negatives)
    assert above <= 0.02 and chosen["negatives_above_merge"] == pytest.approx(above)
    # the lowest such threshold, on a 0.01 grid
    assert sum(1 for n in negatives if n >= merge - 0.01) / len(negatives) > 0.02


def test_a_trap_above_every_negative_sets_the_merge_threshold() -> None:
    chosen = thresholds.choose(positives=[0.9], negatives=[0.1, 0.2, 0.3], traps=[0.71])
    assert chosen["merge"] == pytest.approx(0.72)
    assert chosen["traps_above_merge"] == 0


def test_a_trap_above_an_outlier_positive_still_gives_a_profile() -> None:
    """One low positive (a script pair the model barely knows) must not decide:
    merge goes above the trap and catches the positives above it."""
    chosen = thresholds.choose(positives=[0.26, 0.9], negatives=[0.1, 0.2], traps=[0.6])
    assert chosen["profile"] is True
    assert chosen["merge"] == pytest.approx(0.61) and chosen["traps_above_merge"] == 0
    assert chosen["positives_at_merge"] == pytest.approx(0.5)


def test_no_profile_only_when_merge_would_catch_no_positive() -> None:
    chosen = thresholds.choose(positives=[0.6, 0.7], negatives=[0.1, 0.2], traps=[0.7])
    assert chosen["profile"] is False
    assert chosen["merge"] is None and chosen["resolve"] is None


def test_resolve_is_the_95th_percentile_of_negatives_never_above_merge() -> None:
    negatives = [i / 100 for i in range(100)]
    chosen = thresholds.choose(positives=[0.999], negatives=negatives, traps=[0.1])
    assert chosen["resolve"] == pytest.approx(0.95)
    assert chosen["resolve"] <= chosen["merge"]
    capped = thresholds.choose(positives=[0.999], negatives=[0.1] * 99 + [0.9], traps=[0.2])
    assert capped["resolve"] <= capped["merge"]


def test_the_report_says_how_many_positives_each_threshold_catches() -> None:
    """The 2 % of negatives allowed above ``merge`` may push it above some
    positives: those become no merge proposal, and the report says how many."""
    negatives = [0.1] * 97 + [0.8, 0.8, 0.8]
    chosen = thresholds.choose(positives=[0.60, 0.90, 0.95, 0.97], negatives=negatives, traps=[0.2])
    assert chosen["profile"] is True and chosen["merge"] == pytest.approx(0.81)
    assert chosen["positives_at_merge"] == pytest.approx(0.75)
    assert chosen["positives_at_resolve"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- the lists


def test_the_calibration_list_shares_no_name_with_the_corpus() -> None:
    """Not circular: the thresholds are tuned on names the bench corpus does
    not hold, and verified on the corpus."""
    calibration = {form.casefold() for form in thresholds.calibration_forms()}
    corpus = {form.casefold() for form in thresholds.corpus_forms()}
    assert calibration and corpus and not calibration & corpus


def test_adapted_forms_are_among_the_positives() -> None:
    adapted = thresholds.ADAPTED
    assert ("Joseph", "Иосиф") in adapted and ("Vienna", "Вена") in adapted
    assert ("Alice", "Алиса") in thresholds.CORPUS_ADAPTED


# --------------------------------------------------------------------------- the script


def run(fake: FakeOpenAIServer, tmp_path: Path) -> dict[str, Any]:
    target = tmp_path / "thresholds.json"
    argv = ["--embeddings-url", fake.base_url, "--model", "fake-embed", "--json", str(target)]
    assert thresholds.main(argv) == 0
    report: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return report


def test_the_script_writes_calibration_and_verification_with_its_command(
    fake_openai: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = run(fake_openai, tmp_path)
    assert report["model"] == "fake-embed"
    assert report["command"].startswith("python bench/thresholds.py --embeddings-url")
    assert set(report) >= {"calibration", "verification", "chosen", "adapted", "corpus_adapted"}
    assert report["chosen"]["profile"] in (True, False)
    assert report["current"] == {}  # no built-in profile for this model
    assert 0.0 <= report["calibration"]["positives_above_highest_trap"] <= 1.0
    for pair in report["adapted"] + report["corpus_adapted"]:
        assert set(pair) == {"a", "b", "cosine", "resolve", "merge"}
    inputs = {t for call in fake_openai.embedding_calls for t in call.body["input"]}
    assert inputs == set(thresholds.calibration_forms()) | set(thresholds.corpus_forms())
    assert "merge" in capsys.readouterr().out
