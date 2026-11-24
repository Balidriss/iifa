"""The benchmark harness and its corpus, at CI scale against the fake
servers: the command runs, writes every number with its command, and the
numbers are well formed. Whether they are good is the manual run's business."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bench import run as harness
from bench.self_bench import CASES, SELF_SCOPE, add_self_turns
from bench.synthetic import (
    ALICE,
    BOB,
    CITIES,
    GOLD_KEYS,
    NAMES,
    NEAR_NAMES,
    NEAR_PEOPLE,
    OWNER,
    SERVICES,
    SOLAR,
    TRAP,
    Question,
    generate,
)

from iifa.ingest.schema import parse_answer
from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer
from tests.fakes.openai_server import FakeOpenAIServer, Reply, Respond

SMALL = ["--months", "3", "--chatter", "0.5", "--budgets", "150,600", "--repeat", "1", "--quiet"]


# --------------------------------------------------------------------------- the corpus


def test_the_corpus_is_a_function_of_its_seed() -> None:
    a, b, c = generate(7), generate(7), generate(8)
    assert [t.text for t in a.turns] == [t.text for t in b.turns]
    assert [(q.query, q.expected, q.stale) for q in a.questions] == [
        (q.query, q.expected, q.stale) for q in b.questions
    ]
    assert [t.text for t in a.turns] != [t.text for t in c.turns]


def test_the_corpus_is_multilingual_across_scripts_and_scopes() -> None:
    corpus = generate(7)
    summary = corpus.summary()
    assert set(summary["languages"]) == {"en", "fr", "ru", "ar", "zh", "ja"}
    assert set(corpus.scopes) == {ALICE, SOLAR, BOB}
    karim = {
        corpus.events[t.event].language
        for t in corpus.turns
        if t.event and corpus.events[t.event].person == f"karim@{ALICE}"
    }
    assert len(karim) >= 2, "Karim is told in one script only"
    assert summary["chatter_turns"] > 0


def test_people_move_change_roles_and_minds_in_one_language_or_another() -> None:
    corpus = generate(7)
    changed = [q for q in corpus.questions if q.changed]
    assert {q.attribute for q in changed} == {"city", "role", "drink"}
    assert any(q.same_language_change for q in changed)
    assert any(q.same_language_change is False for q in changed)
    for q in corpus.questions:
        chain = [corpus.events[e] for e in (*q.stale, *q.expected)]
        assert all(e.person == q.person and e.attribute == q.attribute for e in chain)
        assert [e.month for e in chain] == sorted(e.month for e in chain)
        assert all(e.scope == q.scope for e in chain)


def test_the_homonym_lives_in_another_scope_with_other_facts() -> None:
    corpus = generate(7)
    alice = [q for q in corpus.questions if q.person == f"karim@{ALICE}"]
    bob = [q for q in corpus.questions if q.person == f"karim@{BOB}"]
    assert alice and bob
    assert alice[0].query != bob[0].query  # asked in French for Alice, in English for Bob
    assert not {e for q in alice for e in q.expected} & {e for q in bob for e in q.expected}


def test_every_turn_states_at_most_one_fact_and_its_gold_answer_parses() -> None:
    corpus = generate(7)
    for turn in corpus.turns:
        if turn.event is None:
            continue
        answer = parse_answer(json.dumps(corpus.gold[turn.text], ensure_ascii=False))
        assert len(answer.facts) == 1 and answer.rejected.total == 0
        assert answer.facts[0].statement == turn.text
        attribute = corpus.events[turn.event].attribute
        assert answer.facts[0].attribute == GOLD_KEYS[attribute]  # the base vocabulary


def test_a_corpus_needs_two_months() -> None:
    with pytest.raises(ValueError):
        generate(7, months=1)


# --------------------------------------------------------------------------- metrics


def test_contradiction_outcomes_and_leaks_are_counted() -> None:
    question = Question(
        "q", ALICE, "fr", "Où habite Karim ?", "karim", "city", ("e2",), ("e1",), False
    )
    events = {"e1": "t1", "e2": "t2"}
    tally = harness.Tally()
    unit = harness.Unit
    tally.add(question, harness.Answer([unit("t2", (ALICE,))], 10, 0.001), events)
    tally.add(
        question,
        harness.Answer([unit("t1", (ALICE,)), unit("t2", (ALICE,), ("conflict",))], 20, 0.001),
        events,
    )
    tally.add(question, harness.Answer([unit("t1", (ALICE,))], 10, 0.001), events)
    tally.add(question, harness.Answer([unit(None, (BOB,))], 10, 0.001), events)
    out = tally.as_dict()
    row = out["contradictions"]["cross_language"]
    assert (
        row["correct"],
        row["both"],
        row["both_flagged"],
        row["stale_only"],
        row["neither"],
    ) == (1, 1, 1, 1, 1)
    assert out["contradictions"]["same_language"]["questions"] == 0
    assert out["contradictions"]["all"]["accuracy"] == 0.25
    assert out["supported"] == 2 and out["recall"] == 0.5
    assert out["leaks"] == 1  # Bob's record in Alice's view
    assert out["recall_per_1k_tokens"] == pytest.approx(0.5 / 12.5 * 1000, abs=1e-3)
    # the headline: current supported AND no stale one (only the first answer)
    assert out["correct"] == 1 and out["correct_rate"] == 0.25
    assert out["correct_per_1k_tokens"] == pytest.approx(0.25 / 12.5 * 1000, abs=1e-3)


def test_an_unchanged_attribute_is_correct_when_supported() -> None:
    question = Question("q", ALICE, "fr", "Où habite Léa ?", "lea", "city", ("e1",), (), None)
    tally = harness.Tally()
    tally.add(question, harness.Answer([harness.Unit("t1", (ALICE,))], 10, 0.0), {"e1": "t1"})
    assert tally.as_dict()["correct"] == 1


def test_percentile_is_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert harness.percentile(values, 0.5) == 50.0
    assert harness.percentile(values, 0.95) == 95.0
    assert harness.percentile([3.0], 0.95) == 3.0


@pytest.mark.parametrize(
    "argv",
    [
        ["--budgets", "0"],
        ["--budgets", "a,b"],
        ["--chat-url", "http://x/v1"],
        ["--embeddings-model", "m"],
        ["--extra", "model=x"],
        ["--resolve-threshold", "0.9", "--merge-threshold", "0.8"],
        ["--merge-threshold", "1.5"],
    ],
)
def test_bad_arguments_are_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        harness.parse_args(argv)


# --------------------------------------------------------------------------- the command


def _only_report(out: Path) -> dict[str, Any]:
    files = list(out.glob("*.json"))
    assert len(files) == 1, files
    report: dict[str, Any] = json.loads(files[0].read_text(encoding="utf-8"))
    return report


def test_oracle_run_without_embeddings(tmp_path: Path, fake_openai: FakeOpenAIServer) -> None:
    assert harness.main([*SMALL, "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    assert report["command"].startswith("python bench/run.py --months 3")
    assert report["extractor"] == {"name": "oracle", "model": None}
    corpus = report["corpus"]
    assert corpus["seed"] == 7 and corpus["months"] == 3
    ingestion = report["ingestion"]
    assert ingestion["processed"] == ingestion["turns"] == corpus["turns"]
    # one extraction call per turn, chatter too, plus the value adjudications
    assert ingestion["model_calls"] == ingestion["extraction"]["calls"] >= corpus["turns"]
    extraction = ingestion["extraction"]
    assert extraction["facts_written"] == corpus["event_turns"]
    assert extraction["facts_superseded"] > 0  # P7: same-language changes close the old fact
    # across languages the oracle settles every value it knows, the role
    # and preference changes included (the replacement question, on by default)
    assert extraction["facts_conflict"] == 0
    assert extraction["values_replacement_replaces"] > 0
    assert report["pipeline"] == {
        "replacement_question": True,
        "restatement_question": True,
        "value_norm": False,
        "read_intent": True,
        "forced_language": None,
        "thresholds": None,  # the embedding model's profile, untouched
    }
    assert extraction["keys_renamed"] == 0  # the oracle already writes canonical keys
    audit = ingestion["supersessions"]
    assert audit["total"] == audit["correct"] == extraction["facts_superseded"]
    assert audit["false"] == 0 and audit["unmatched"] == 0
    assert set(report["systems"]) == {"none", "truncated_history", "iifa_tier2"}
    assert set(report["skipped"]) == {"vector_topk", "iifa_tier3"}
    for name, per_budget in report["systems"].items():
        assert set(per_budget) == {"150", "600"}
        for budget, row in per_budget.items():
            assert row["leaks"] == 0, (name, budget)  # P9 in the bench itself
            assert row["questions"] == corpus["questions"]
            assert row["mean_tokens"] <= int(budget)  # P1 for every system
            assert row["contradictions"]["all"]["questions"] == corpus["changed_questions"]
    assert report["systems"]["none"]["600"]["recall"] == 0.0
    assert report["systems"]["none"]["600"]["mean_tokens"] == 0
    assert report["systems"]["iifa_tier2"]["600"]["recall"] > 0
    assert report["systems"]["iifa_tier2"]["600"]["tier"] == 2
    for scope in (ALICE, SOLAR, BOB):
        probes = report["probes"][scope]
        assert probes["probes"] > 0
        assert set(probes["iifa_tier2"]) == {"150", "600"}
    assert not fake_openai.calls  # the oracle never leaves the process


def test_the_oracle_falls_into_no_trap(tmp_path: Path) -> None:
    """With --traps an ideal model keys where one works apart from where one
    lives: every supersession is correct, the role and preference changes the
    replacement question settled included."""
    argv = ["--months", "12", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    argv += ["--replacement-question"]
    assert harness.main([*argv, "--traps", "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    assert report["corpus"]["traps"] is True
    assert report["pipeline"]["replacement_question"] is True
    audit = report["ingestion"]["supersessions"]
    assert audit["false"] == 0 and audit["correct"] == audit["total"] > 0
    assert report["ingestion"]["extraction"]["values_replacement_replaces"] > 0


def test_without_the_replacement_question_an_added_value_closes_one_still_true(
    tmp_path: Path,
) -> None:
    """Switched off, a value added in one language closes the one already there:
    the trap corpus shows it as false replaces, even with the oracle."""
    argv = ["--months", "12", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    argv += ["--no-replacement-question"]
    assert harness.main([*argv, "--traps", "--out", str(tmp_path)]) == 0
    audit = _only_report(tmp_path)["ingestion"]["supersessions"]
    assert audit["false_replaces"] > 0


def test_a_model_that_lumps_work_with_home_shows_false_supersessions(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """The metric bites: a fake model that keys the office as ``home`` closes
    homes that are still true, and the audit says so."""
    corpus = generate(7, months=3, chatter=0.5, traps=True)
    oracle = harness.Oracle(corpus)

    def lumping(body: Any) -> Reply:
        answer = json.loads(oracle.answer(body["messages"]))
        for fact in answer.get("facts", []):
            if fact.get("attribute") == GOLD_KEYS[TRAP]:
                fact["attribute"] = GOLD_KEYS["city"]
        return Reply(json.dumps(answer, ensure_ascii=False))

    fake_openai.set_default(chat=Respond(lumping))
    argv = [*SMALL, "--traps", "--out", str(tmp_path)]
    argv += ["--chat-url", fake_openai.base_url, "--model", "fake-chat"]
    assert harness.main(argv) == 0
    audit = _only_report(tmp_path)["ingestion"]["supersessions"]
    # added values closed by a "replaces" would count apart (false replaces)
    assert audit["false_other_attribute"] > 0
    assert audit["false_other_attribute"] + audit["false_replaces"] == audit["false"]
    # each false one is listed: which turns, about whom, what was stored
    cases = audit["false_cases"]
    assert 0 < len(cases) == min(audit["false"], 10)
    keys = {"closed", "by", "cause"}
    assert all(set(case) == keys for case in cases)
    for case in cases:
        for side in (case["closed"], case["by"]):
            assert set(side) == {"turn", "person", "attribute", "statement"}
    assert {c["closed"]["attribute"] for c in cases} >= {"city"}
    assert {c["by"]["attribute"] for c in cases} >= {TRAP}
    assert "other attribute" in {c["cause"] for c in cases}


def test_traps_leave_the_default_corpus_untouched() -> None:
    plain, trapped = generate(7), generate(7, traps=True)
    assert not any(e.attribute == TRAP for e in plain.events.values())
    same = sorted(
        (e for e in trapped.events.values() if e.attribute not in (TRAP, "team") and not e.added),
        key=str,
    )
    assert same == sorted(plain.events.values(), key=str)  # the same events, more of them
    offices = [e for e in trapped.events.values() if e.attribute == TRAP]
    assert len(offices) == len({e.person for e in plain.events.values()})
    for office in offices:
        first_home = next(
            e for e in plain.events.values() if e.person == office.person and e.attribute == "city"
        )
        assert office.value != first_home.value
    teams = len({e.person for e in trapped.events.values() if e.attribute == "team"})
    assert len(trapped.questions) == len(plain.questions) + len(offices) + teams


def test_tier3_and_the_vector_baseline_with_embeddings(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    argv = [
        *SMALL,
        "--out",
        str(tmp_path),
        "--embeddings-url",
        fake_openai.base_url,
        "--embeddings-model",
        "fake-embed",
    ]
    assert harness.main(argv) == 0
    report = _only_report(tmp_path)
    assert set(report["systems"]) == {
        "none",
        "truncated_history",
        "vector_topk",
        "iifa_tier2",
        "iifa_tier3",
    }
    assert report["skipped"] == {}
    assert report["systems"]["iifa_tier3"]["600"]["tier"] == 3
    assert report["embeddings"]["vector_build"]["texts"] > 0
    for per_budget in report["systems"].values():
        assert all(row["leaks"] == 0 for row in per_budget.values())
    assert fake_openai.embedding_calls and not fake_openai.chat_calls
    assert all(c.model == "fake-embed" for c in fake_openai.embedding_calls)
    assert set(report["probes"][ALICE]) == {"probes", "iifa_tier2", "iifa_tier3"}


def test_a_real_model_run_goes_through_the_chat_endpoint(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """The fake model answers like the oracle: the model path gives the oracle's numbers."""
    corpus = generate(7, months=3, chatter=0.5)
    oracle = harness.Oracle(corpus)
    fake_openai.set_default(chat=Respond(lambda body: Reply(oracle.answer(body["messages"]))))
    assert (
        harness.main(
            [
                *SMALL,
                "--out",
                str(tmp_path / "model"),
                "--chat-url",
                fake_openai.base_url,
                "--model",
                "fake-chat",
            ]
        )
        == 0
    )
    assert harness.main([*SMALL, "--out", str(tmp_path / "oracle")]) == 0
    model, reference = _only_report(tmp_path / "model"), _only_report(tmp_path / "oracle")
    assert model["extractor"]["name"] == "model" and model["extractor"]["model"] == "fake-chat"
    # one extraction call per turn, plus the value adjudications: the same as the oracle's
    assert len(fake_openai.chat_calls) == model["ingestion"]["model_calls"]
    assert model["ingestion"]["model_calls"] == reference["ingestion"]["model_calls"]
    assert model["ingestion"]["model_calls"] >= corpus.summary()["turns"]
    assert all(c.model == "fake-chat" for c in fake_openai.chat_calls)
    for name in ("truncated_history", "iifa_tier2"):
        for budget in ("150", "600"):
            assert (
                model["systems"][name][budget]["recall"]
                == reference["systems"][name][budget]["recall"]
            )


def test_an_interruption_reports_what_was_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run = harness.Worker.run
    calls = {"n": 0}

    def run_then_stop(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > 5:
            raise KeyboardInterrupt
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(harness.Worker, "run", run_then_stop)
    assert harness.main([*SMALL, "--out", str(tmp_path)]) == harness.INTERRUPTED
    report = _only_report(tmp_path)
    assert report["ingestion"]["interrupted"] is True
    assert report["ingestion"]["processed"] == 5
    assert report["systems"]["iifa_tier2"]["600"]["questions"] > 0


def test_keep_store_keeps_an_encrypted_store_to_read_afterwards(tmp_path: Path) -> None:
    kept = tmp_path / "kept" / "bench.db"
    argv = [*SMALL, "--out", str(tmp_path / "out"), "--keep-store", str(kept)]
    assert harness.main(argv) == 0
    report = _only_report(tmp_path / "out")
    assert report["kept_store"] == str(kept)
    with Store(kept, harness.STORE_KEY) as store:
        assert store.counts()["facts"] == report["corpus"]["event_turns"]
    with pytest.raises(SystemExit):  # never overwrites a store
        harness.parse_args([*argv])


# --------------------------------------------------------------------------- chatter, restatements, the owner


def test_rephrase_and_speaker_leave_the_default_corpus_untouched() -> None:
    plain = generate(7)
    assert not any(t.restates for t in plain.turns) and not any(t.speaker for t in plain.turns)
    for corpus in (generate(7, rephrase=True), generate(7, speaker=True)):
        same = sorted((e for e in corpus.events.values() if e.person != OWNER), key=str)
        assert same == sorted(plain.events.values(), key=str)


def test_a_restatement_tells_the_same_value_in_another_language_the_same_month() -> None:
    corpus = generate(7, rephrase=True)
    restating = [t for t in corpus.turns if t.restates]
    assert restating and corpus.summary()["rephrase_turns"] == len(restating)
    order = {t.key: i for i, t in enumerate(corpus.turns)}
    original = corpus.turn_of_event()
    for turn in restating:
        event = corpus.events[turn.restates]
        assert turn.event is None and turn.scope == event.scope and turn.month == event.month
        assert turn.language != event.language
        assert order[turn.key] > order[original[event.key]]
        assert turn.key in corpus.turns_of_event()[event.key]
        answer = parse_answer(json.dumps(corpus.gold[turn.text], ensure_ascii=False))
        assert len(answer.facts) == 1 and answer.facts[0].statement == turn.text


def test_the_owner_speaks_in_the_first_person_and_is_asked_in_the_first_person() -> None:
    corpus = generate(7, speaker=True)
    spoken = [t for t in corpus.turns if t.speaker]
    assert spoken and corpus.summary()["speaker_turns"] == len(spoken)
    assert {t.language for t in spoken} <= {"fr", "en", "ru"}
    for turn in spoken:
        assert turn.scope == ALICE and "Alice" not in turn.text and "Алиса" not in turn.text
        answer = parse_answer(json.dumps(corpus.gold[turn.text], ensure_ascii=False))
        assert len(answer.facts) == 1 and answer.facts[0].subject == "Alice"
        assert answer.facts[0].quote == turn.text
    asked = [q for q in corpus.questions if q.person == OWNER]
    assert {q.attribute for q in asked} == {"city", "role", "drink"}
    assert all(q.language == "fr" and "Alice" not in q.query for q in asked)


def test_restatements_count_as_support_and_two_units_of_one_event_as_a_duplicate() -> None:
    question = Question("q", ALICE, "fr", "Où habite Karim ?", "karim", "city", ("e1",), (), None)
    events = {"e1": ("t1", "t9")}
    tally = harness.Tally()
    unit = harness.Unit
    tally.add(question, harness.Answer([unit("t9", (ALICE,))], 10, 0.0), events)
    tally.add(
        question, harness.Answer([unit("t1", (ALICE,)), unit("t9", (ALICE,))], 20, 0.0), events
    )
    out = tally.as_dict()
    assert out["correct"] == 2 and out["duplicates"] == 1


def test_an_oracle_run_counts_chatter_and_no_speaker_node(tmp_path: Path) -> None:
    assert harness.main([*SMALL, "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    chatter = report["ingestion"]["chatter"]
    assert chatter["turns"] == report["corpus"]["chatter_turns"] > 0
    assert chatter["facts"] == chatter["nodes"] == chatter["turns_with_facts"] == 0
    assert report["ingestion"]["speaker_nodes"] == 0
    assert report["ingestion"]["extraction"]["empty_answers"] == chatter["turns"]
    assert report["ingestion"]["owner_turns"] == []  # no owner in the default corpus
    for row in report["systems"]["iifa_tier2"].values():
        assert row["duplicates"] >= 0


def test_a_chatty_model_shows_in_the_chatter_counts(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    corpus = generate(7, months=3, chatter=0.5)
    oracle = harness.Oracle(corpus)
    chatter = {f"{t.text}" for t in corpus.turns if t.event is None}

    def chatty(body: Any) -> Reply:
        source = harness.source_text(body["messages"])
        if source in chatter:
            fact = {
                "subject": "User",
                "statement": "The user said: " + source,
                "quote": source,
                "modality": "fact",
                "duration": "lasting",
            }
            weather = {"name": "The weather", "kind": "topic"}
            chat = {"nodes": [weather], "facts": [fact], "edges": [], "frame": "sincere"}
            return Reply(json.dumps(chat))
        return Reply(oracle.answer(body["messages"]))

    fake_openai.set_default(chat=Respond(chatty))
    argv = [*SMALL, "--out", str(tmp_path), "--chat-url", fake_openai.base_url, "--model", "m"]
    assert harness.main(argv) == 0
    ingestion = _only_report(tmp_path)["ingestion"]
    assert ingestion["chatter"]["facts"] > 0 and ingestion["chatter"]["nodes"] > 0
    assert ingestion["chatter"]["turns_with_facts"] > 0
    assert ingestion["speaker_nodes"] == 0  # "User" never became a node


def test_an_oracle_run_with_restatements_counts_links_and_self_closures(tmp_path: Path) -> None:
    """Restatements are measured, not judged here: a value told again in the same
    spelling is a duplicate (no fact), an equivalence is linked, and a value closed
    by its own restatement is counted apart (``restated``), never as false."""
    assert harness.main([*SMALL, "--rephrase", "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    assert report["corpus"]["rephrase"] is True
    restated = report["ingestion"]["restatements"]
    assert restated["turns"] == report["corpus"]["rephrase_turns"] > 0
    assert 0 < restated["facts"] <= restated["turns"]
    assert restated["linked"] > 0
    audit = report["ingestion"]["supersessions"]
    assert audit["false"] == 0
    assert audit["total"] == audit["correct"] + audit["restated"] + audit["unmatched"]


def test_the_comparison_form_links_without_asking_and_closes_nothing_more(
    tmp_path: Path,
) -> None:
    """With --value-norm the oracle gives ideal forms; values
    restated in another language are linked by them instead of adjudicated, and
    no supersession is added or lost."""
    argv = [*SMALL, "--rephrase"]
    off, on = tmp_path / "off", tmp_path / "on"
    assert harness.main([*argv, "--out", str(off)]) == 0
    assert harness.main([*argv, "--value-norm", "--out", str(on)]) == 0
    without, with_forms = _only_report(off), _only_report(on)
    assert with_forms["pipeline"]["value_norm"] is True
    before, after = without["ingestion"]["extraction"], with_forms["ingestion"]["extraction"]
    assert before["values_norm"] == 0 < after["values_norm"]

    def asked(extraction: dict[str, Any]) -> int:
        return sum(v for k, v in extraction.items() if k.startswith("values_adjudicated_"))

    assert asked(after) < asked(before)
    assert after["facts_superseded"] == before["facts_superseded"]
    assert after["facts_conflict"] <= before["facts_conflict"]
    assert with_forms["ingestion"]["supersessions"]["false"] == 0


def test_an_oracle_run_with_the_owner_attaches_every_owner_fact(tmp_path: Path) -> None:
    assert harness.main([*SMALL, "--speaker", "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    assert report["corpus"]["speaker"] is True
    extraction = report["ingestion"]["extraction"]
    assert extraction["facts_speaker_owner"] == report["corpus"]["speaker_turns"] > 0
    assert extraction.get("facts_speaker_free", 0) == 0 and extraction["facts_proposed"] == 0
    assert report["ingestion"]["owner"] == {"scope": ALICE, "name": "Alice"}
    # one record per owner turn: which sentence was written, lost, and why
    records = report["ingestion"]["owner_turns"]
    assert len(records) == report["corpus"]["speaker_turns"]
    for record in records:
        assert set(record) == {
            "turn",
            "language",
            "written",
            "proposed",
            "incomplete",
            "cross_script",
            "mixed_script",
            "language_mismatch",
            "duplicate",
            "retries",
            "nodes_created",
        }
        assert record["written"] == 1 and record["incomplete"] == 0
        assert record["language"] in {"fr", "en", "ru"}
    assert report["ingestion"]["supersessions"]["false"] == 0
    assert report["systems"]["iifa_tier2"]["600"]["questions"] == report["corpus"]["questions"]


def test_the_translation_check_is_measured_not_applied(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """A model that writes French preferences in short English ("Nadia prefers
    coffee.", undetermined, same script: P12 cannot see it) shows up in the
    bench's translation check, by turn language; the oracle shows nothing."""
    assert harness.main([*SMALL, "--out", str(tmp_path / "oracle")]) == 0
    check = _only_report(tmp_path / "oracle")["ingestion"]["translation_check"]
    assert check == {"candidates": 0, "flagged": {"turn_en": 0, "turn_other": 0}}

    corpus = generate(7, months=3, chatter=0.5)
    oracle = harness.Oracle(corpus)
    french = {t.text for t in corpus.turns if t.event and t.language == "fr"}

    def translating(body: Any) -> Reply:
        answer = json.loads(oracle.answer(body["messages"]))
        if harness.source_text(body["messages"]) in french:
            for fact in answer["facts"]:
                if fact["attribute"] == GOLD_KEYS["drink"]:
                    fact["statement"] = f"{fact['subject']} prefers coffee."
        return Reply(json.dumps(answer, ensure_ascii=False))

    fake_openai.set_default(chat=Respond(translating))
    argv = [*SMALL, "--out", str(tmp_path / "model"), "--chat-url", fake_openai.base_url]
    assert harness.main([*argv, "--model", "fake-chat"]) == 0
    report = _only_report(tmp_path / "model")
    check = report["ingestion"]["translation_check"]
    assert check["flagged"]["turn_other"] > 0 and check["flagged"]["turn_en"] == 0
    assert check["candidates"] >= check["flagged"]["turn_other"]
    assert report["ingestion"]["extraction"].get("facts_language_mismatch", 0) == 0  # the hole


# --------------------------------------------------------------------------- values added, not replaced


def test_traps_add_values_on_top_of_the_current_ones_in_the_last_month() -> None:
    plain, trapped = generate(7), generate(7, traps=True)
    added = [e for e in trapped.events.values() if e.added]
    assert {e.attribute for e in added} == {"role", "drink", "team"}
    assert all(e.month == trapped.months for e in added)
    assert not any(e.added for e in plain.events.values())
    order = {t.event: i for i, t in enumerate(trapped.turns) if t.event}
    for event in added:
        person_events = [
            e
            for e in trapped.events.values()
            if e.person == event.person and e.attribute == event.attribute and not e.added
        ]
        assert person_events, event  # something already true to add to
        assert all(order[e.key] < order[event.key] for e in person_events)
        assert event.value not in {e.value for e in person_events if e.attribute == "role"}
    teams = [q for q in trapped.questions if q.attribute == "team"]
    assert teams and all(len(q.expected) == 2 and not q.stale for q in teams)


def test_the_oracle_answers_both_for_an_added_value_and_closes_nothing_true(
    tmp_path: Path,
) -> None:
    argv = ["--months", "12", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    argv += ["--replacement-question"]
    assert harness.main([*argv, "--traps", "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    extraction = report["ingestion"]["extraction"]
    assert extraction["values_replacement_both"] > 0
    audit = report["ingestion"]["supersessions"]
    assert audit["false"] == 0 and audit["false_replaces"] == 0


def test_a_model_that_always_replaces_shows_false_replaces(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """The metric bites: a model that answers "replaces" to every replacement
    question closes values still true, and the audit says so."""
    corpus = generate(7, months=3, chatter=0.0, traps=True)
    oracle = harness.Oracle(corpus)

    def replacing(body: Any) -> Reply:
        answer = json.loads(oracle.answer(body["messages"]))
        for item in answer.get("replacements", []):
            item["verdict"] = "replaces"
        return Reply(json.dumps(answer, ensure_ascii=False))

    fake_openai.set_default(chat=Respond(replacing))
    argv = ["--months", "3", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    argv += ["--traps", "--replacement-question", "--out", str(tmp_path)]
    argv += ["--chat-url", fake_openai.base_url]
    assert harness.main([*argv, "--model", "fake-chat"]) == 0
    audit = _only_report(tmp_path)["ingestion"]["supersessions"]
    assert audit["false_replaces"] > 0 and audit["false"] >= audit["false_replaces"]


# --------------------------------------------------------------------------- the language question


def test_force_language_makes_every_scope_force_it(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """--force-language en: every extraction prompt asks for English whatever the
    source (P12's scope override), and the report says so."""
    fake_openai.set_default(chat=Reply('{"nodes": [], "facts": [], "edges": []}'))
    argv = [*SMALL, "--out", str(tmp_path), "--force-language", "en"]
    argv += ["--chat-url", fake_openai.base_url, "--model", "fake-chat"]
    assert harness.main(argv) == 0
    report = _only_report(tmp_path)
    assert report["pipeline"]["forced_language"] == "en"
    systems = [c.body["messages"][0]["content"] for c in fake_openai.chat_calls]
    assert systems and all(
        "TARGET LANGUAGE: write every statement, value and summary in English, whatever the "
        "language of the source" in s
        for s in systems
        if "<<<SOURCE>>>" not in s
    )


@pytest.mark.parametrize("code", ["english", "zz1", ""])
def test_force_language_needs_a_language_code(code: str) -> None:
    with pytest.raises(SystemExit):
        harness.parse_args(["--force-language", code])


def test_without_force_language_nothing_is_forced(tmp_path: Path) -> None:
    assert harness.main([*SMALL, "--out", str(tmp_path)]) == 0
    assert _only_report(tmp_path)["pipeline"]["forced_language"] is None


def test_every_answer_keeps_its_per_question_outcome(tmp_path: Path) -> None:
    """Two runs are compared question by question without the stores: keys and
    booleans only, consistent with the totals."""
    assert harness.main([*SMALL, "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    for per_budget in report["systems"].values():
        for row in per_budget.values():
            outcomes = row["per_question"]
            assert len(outcomes) == row["questions"]
            assert set(outcomes[0]) == {"question", "current", "stale", "correct"}
            assert sum(o["correct"] for o in outcomes) == row["correct"]
            assert sum(o["current"] for o in outcomes) == row["supported"]
            assert all(o["correct"] == (o["current"] and not o["stale"]) for o in outcomes)


# --------------------------------------------------------------------------- near names, name adjudications


def test_name_traps_add_near_named_people_and_leave_the_default_corpus_untouched() -> None:
    plain, near = generate(7), generate(7, name_traps=True)
    added = {p.key for p in NEAR_PEOPLE}
    same = sorted((e for e in near.events.values() if e.person not in added), key=str)
    assert same == sorted(plain.events.values(), key=str)
    assert {e.person for e in near.events.values()} - {
        e.person for e in plain.events.values()
    } == added
    assert added <= {q.person for q in near.questions}
    assert near.summary()["name_traps"] is True and plain.summary()["name_traps"] is False
    # each stands beside someone of the same scope whose name is near
    beside = {"karima": "karim", "ivana": "ivan", "nadine": "nadia"}
    people = {(p.name_key, p.scope) for p in near.people}
    for person in NEAR_PEOPLE:
        assert (beside[person.name_key], person.scope) in people
    told = [t.text for t in near.turns if t.event and near.events[t.event].person in added]
    assert told and all(t in near.gold for t in told)


def test_a_near_name_pair_and_its_answer_are_reported() -> None:
    """A mention and a candidate naming two different people are a
    near-name pair; attaching one to the other is a wrong attachment."""
    records = [
        {
            "turn": "t001",
            "mention": "Karima",
            "candidates": [{"name": "Karim", "score": 0.83, "semantic": True}],
            "chosen": None,
        },
        {
            "turn": "t002",
            "mention": "Ивана",
            "candidates": [{"name": "Ivan", "score": 0.81, "semantic": True}],
            "chosen": "Ivan",
        },
        {  # one person in two scripts: not a near-name pair
            "turn": "t003",
            "mention": "Карим",
            "candidates": [{"name": "Karim", "score": 0.9, "semantic": True}],
            "chosen": "Karim",
        },
        {  # a city: not a person
            "turn": "t004",
            "mention": "Лион",
            "candidates": [{"name": "Lyon", "score": 0.7, "semantic": True}],
            "chosen": "Lyon",
        },
    ]
    out = harness.name_adjudications(records)
    assert out["mentions"] == 4 and out["near_name_pairs"] == 2 and out["attached"] == 1
    assert [(p["mention"], p["candidate"], p["answer"]) for p in out["pairs"]] == [
        ("Karima", "Karim", "new"),
        ("Ивана", "Ivan", "attached"),
    ]
    assert out["pairs"][0]["semantic"] is True and out["pairs"][0]["score"] == 0.83


def _rename(answer: dict[str, Any], forms: dict[str, str]) -> dict[str, Any]:
    """A model that writes one person's facts under another's name, in its
    statement and its quote too: no near form left in a located quote, so nothing
    contests the attachment and the audit must catch it."""
    for node in answer.get("nodes", []):
        node["name"] = forms.get(node["name"], node["name"])
    for fact in answer.get("facts", []):
        if fact.get("subject") in forms:
            old, new = fact["subject"], forms[fact["subject"]]
            fact["subject"] = fact["subject_as_written"] = new
            for key in ("statement", "quote"):
                fact[key] = str(fact.get(key) or "").replace(old, new)
    return answer


def test_a_fact_written_on_another_persons_node_is_a_wrong_attachment(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    corpus = generate(7, months=3, chatter=0.0, name_traps=True)
    oracle = harness.Oracle(corpus)
    forms = {NEAR_NAMES["karima"][lang]: NAMES["karim"][lang] for lang in NAMES["karim"]}

    def confusing(body: Any) -> Reply:
        answer = json.loads(oracle.answer(body["messages"]))
        return Reply(json.dumps(_rename(answer, forms), ensure_ascii=False))

    fake_openai.set_default(chat=Respond(confusing))
    argv = [*SMALL, "--name-traps", "--out", str(tmp_path)]
    argv += ["--chat-url", fake_openai.base_url, "--model", "fake-chat"]
    assert harness.main(argv) == 0
    attachments = _only_report(tmp_path)["ingestion"]["attachments"]
    assert attachments["wrong"] > 0 and attachments["checked"] >= attachments["wrong"]
    assert any(
        w["person"].startswith("karima@") and w["node"] in NAMES["karim"].values()
        for w in attachments["examples"]
    )


def test_the_oracle_attaches_no_fact_to_a_near_named_person(tmp_path: Path) -> None:
    """An ideal model names Karima as Karima: nothing may file her facts on
    Karim's node (an exact, fuzzy or adjudicated match alike)."""
    argv = ["--months", "3", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    assert harness.main([*argv, "--name-traps", "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    assert report["corpus"]["name_traps"] is True
    attachments = report["ingestion"]["attachments"]
    assert attachments["checked"] > 0
    assert attachments["wrong"] == 0, attachments["examples"]


def test_near_names_reach_the_adjudication_and_the_report_says_what_it_answered(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """Tier 3: Karima's name vectors sit next to Karim's, so her mentions reach
    the adjudication (the oracle answers "new"): listed, none attached."""
    near = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fake_openai.set_vectors(
        {
            **{form: near for form in NAMES["karim"].values()},
            **{form: near for form in NEAR_NAMES["karima"].values()},
        }
    )
    argv = [*SMALL, "--name-traps", "--out", str(tmp_path)]
    argv += ["--embeddings-url", fake_openai.base_url, "--embeddings-model", "fake-embed"]
    argv += ["--resolve-threshold", "0.5", "--merge-threshold", "0.9"]
    assert harness.main(argv) == 0
    report = _only_report(tmp_path)
    assert report["pipeline"]["thresholds"] == {"resolve": 0.5, "merge": 0.9}
    adjudications = report["ingestion"]["name_adjudications"]
    assert adjudications["near_name_pairs"] > 0 and adjudications["attached"] == 0
    assert report["ingestion"]["extraction"]["adjudication_calls"] > 0
    assert report["ingestion"]["extraction"]["mentions_adjudicated"] >= adjudications["mentions"]


def test_the_oracle_answers_the_restatement_question_from_the_truth() -> None:
    """One value in two forms is the same, two values are different, a
    form the corpus does not know is unsure."""
    oracle = harness.Oracle(generate(7, months=3, chatter=0.5))
    pairs = [
        {"pair": 0, "a": {"value": "Lyon"}, "b": {"value": "\u041b\u0438\u043e\u043d"}},
        {"pair": 1, "a": {"value": "Lyon"}, "b": {"value": "Paris"}},
        {"pair": 2, "a": {"value": "120 o"}, "b": {"value": "120 octets"}},
    ]
    messages = [
        {"role": "system", "content": 'Answer {"restatements": [...]}'},
        {"role": "user", "content": f"<<<PAIRS>>>\n{json.dumps(pairs)}\n<<<END>>>"},
    ]
    answered = json.loads(oracle.answer(messages))["restatements"]
    assert [v["verdict"] for v in answered] == ["same", "different", "unsure"]


# --------------------------------------------------------------------------- neighbouring subjects


def test_neighbours_add_turns_of_two_projects_and_leave_the_default_corpus_untouched() -> None:
    plain, side = generate(7), generate(7, neighbours=True)
    told = [t for t in side.turns if t.neighbours]
    assert told and not any(t.neighbours or t.aside for t in plain.turns)
    added = [t for t in side.turns if t.neighbours or t.aside]
    assert [t.text for t in side.turns if t not in added] == [t.text for t in plain.turns]
    assert side.questions == plain.questions
    assert side.summary()["neighbour_turns"] == len(told) and plain.summary()["neighbours"] is False
    for turn in told:
        assert not side.is_chatter(turn) and turn.event is None
        gold = side.gold[turn.text]
        assert [f["subject"] for f in gold["facts"]] == [n["name"] for n in gold["nodes"]]
        assert all(f["quote"] in turn.text for f in gold["facts"])
        assert not parse_answer(json.dumps(gold)).rejected.total
        assert {f["duration"] for f in gold["facts"]} == {"lasting"}


def test_the_oracle_writes_each_projects_fact_on_its_own_node(tmp_path: Path) -> None:
    argv = ["--months", "4", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    assert harness.main([*argv, "--neighbours", "--out", str(tmp_path)]) == 0
    side = _only_report(tmp_path)["ingestion"]["neighbours"]
    assert side["turns"] > 0 and side["expected"] == 2 * side["turns"]
    assert side["right"] == side["expected"]
    assert side["wrong"] == side["missing"] == side["unjudged"] == side["proposed"] == 0
    asides = side["asides"]
    assert asides["told"] == side["turns"] and asides["kept"] == asides["told"]
    assert asides["closed"] == asides["conflict"] == asides["other_key"] == asides["missing"] == 0
    assert {(k["description"][0], k["aside"][0]) for k in asides["keys"]} == {
        ("description", "description")
    }


def test_each_neighbour_turn_is_followed_by_an_aside_under_the_same_key() -> None:
    """The web service told something unrelated, keyed ``description``
    with one value (a real model's slip on a generic key); no random draw."""
    corpus = generate(7, neighbours=True)
    turns = corpus.turns
    for index, turn in enumerate(turns):
        if not turn.neighbours:
            continue
        aside = turns[index + 1]
        assert aside.aside is not None and aside.aside in turn.text
        assert not corpus.is_chatter(aside) and aside.event is None
        (fact,) = corpus.gold[aside.text]["facts"]
        described = {f["subject"]: f for f in corpus.gold[turn.text]["facts"]}
        assert fact["attribute"] == described[aside.aside]["attribute"] == "description"
        assert fact["cardinality"] == "one" and fact["quote"] in aside.text
        assert not parse_answer(json.dumps(corpus.gold[aside.text])).rejected.total
    assert corpus.summary()["aside_turns"] == corpus.summary()["neighbour_turns"]


def test_a_model_answering_replaces_for_an_aside_closes_the_description(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """The probe measures what it claims: a replacement question answered
    ``replaces`` closes the description, and the report says so."""
    corpus = generate(7, months=4, chatter=0.0, neighbours=True)
    oracle = harness.Oracle(corpus)

    def replacing(body: Any) -> Reply:
        system = str(body["messages"][0]["content"])
        if '"replacements"' in system:
            user = str(body["messages"][-1]["content"])
            match = harness.PAIRS.search(user)
            pairs = json.loads(match.group(1)) if match else []
            verdicts = [{"pair": p["pair"], "verdict": "replaces"} for p in pairs]
            return Reply(json.dumps({"replacements": verdicts}))
        return Reply(oracle.answer(body["messages"]))

    fake_openai.set_default(chat=Respond(replacing))
    argv = ["--months", "4", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    argv += ["--neighbours", "--out", str(tmp_path)]
    argv += ["--chat-url", fake_openai.base_url, "--model", "fake-chat"]
    assert harness.main(argv) == 0
    asides = _only_report(tmp_path)["ingestion"]["neighbours"]["asides"]
    assert asides["told"] > 0 and asides["closed"] == asides["told"]
    assert asides["kept"] == 0


def test_an_aside_under_another_key_is_counted_apart(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """A model keying the aside otherwise leaves nothing to replace: counted as
    ``other_key``, never as kept or closed, and its keys are listed."""
    corpus = generate(7, months=4, chatter=0.0, neighbours=True)
    oracle = harness.Oracle(corpus)
    asides = {t.text for t in corpus.turns if t.aside}

    def rekeying(body: Any) -> Reply:
        answer = json.loads(oracle.answer(body["messages"]))
        if harness.source_text(body["messages"]) in asides:
            for fact in answer["facts"]:
                fact["attribute"] = "deployment"
        return Reply(json.dumps(answer, ensure_ascii=False))

    fake_openai.set_default(chat=Respond(rekeying))
    argv = ["--months", "4", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    argv += ["--neighbours", "--out", str(tmp_path)]
    argv += ["--chat-url", fake_openai.base_url, "--model", "fake-chat"]
    assert harness.main(argv) == 0
    side = _only_report(tmp_path)["ingestion"]["neighbours"]["asides"]
    assert side["told"] > 0 and side["other_key"] == side["told"]
    assert side["kept"] == side["closed"] == 0
    assert all(k["aside"] == ["deployment"] for k in side["keys"])


def test_a_model_putting_the_tools_sentence_on_the_service_is_caught(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """The tool's sentence attached to the web service is a proposal,
    never a fact on the service's node."""
    corpus = generate(7, months=4, chatter=0.0, neighbours=True)
    oracle = harness.Oracle(corpus)

    def confusing(body: Any) -> Reply:
        answer = json.loads(oracle.answer(body["messages"]))
        names = [n["name"] for n in answer.get("nodes", [])]
        services = [n for n in names if n.split("-")[0].casefold() in SERVICES]
        if len(names) == 2 and len(services) == 1:
            service = services[0]
            for fact in answer["facts"]:
                if fact["subject"] != service:
                    other = fact["subject"]
                    fact["subject"] = service
                    fact["statement"] = fact["statement"].replace(other, service)
        return Reply(json.dumps(answer, ensure_ascii=False))

    fake_openai.set_default(chat=Respond(confusing))
    argv = ["--months", "4", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    argv += ["--neighbours", "--out", str(tmp_path)]
    argv += ["--chat-url", fake_openai.base_url, "--model", "fake-chat"]
    assert harness.main(argv) == 0
    report = _only_report(tmp_path)["ingestion"]
    side = report["neighbours"]
    assert side["wrong"] == 0, side["examples"]
    assert side["proposed"] == side["turns"] > 0
    assert report["extraction"]["facts_neighbour"] == side["turns"]


# --------------------------------------------------------------------------- place nodes named after a value


def _with_office_nodes(answer: dict[str, Any], language: str) -> dict[str, Any]:
    """What a real model answered for an office turn: besides the gold fact, a
    place node named after the office ("bureau de Osaka"), a summary naming
    the person, and an edge to it; the office stays the value of the person's fact."""
    for fact in list(answer.get("facts", [])):
        if fact.get("attribute") != GOLD_KEYS[TRAP]:
            continue
        place = f"bureau de {fact['value']}" if language == "fr" else f"{fact['value']} office"
        answer["nodes"].append(
            {"name": place, "kind": "place", "aliases": [], "summary": fact["statement"]}
        )
        answer["edges"].append(
            {
                "src": fact["subject"],
                "dst": place,
                "type": "works_at",
                "quote": fact["quote"],
                "confidence": 0.9,
            }
        )
    return answer


def test_a_place_node_named_after_an_office_does_not_cost_the_office_question(
    tmp_path: Path, fake_openai: FakeOpenAIServer
) -> None:
    """The trap corpus at tier 2, 300 tokens: a model that also proposes place
    nodes "bureau de Osaka" for where people work gets them created, yet "Où se
    trouve le bureau de Karim ?" (q003) is still answered by Karim's own fact:
    such a node is ranked on its name only (never its summary) and
    the named subject's facts come first (subject anchoring). The
    ranking without anchoring is tested in tests/model/test_place_nodes.py."""
    corpus = generate(7, months=12, chatter=0.0, traps=True, name_traps=True)
    question = next(q for q in corpus.questions if q.key == "q003")
    assert question.query == "Où se trouve le bureau de Karim ?" and question.attribute == TRAP
    oracle = harness.Oracle(corpus)
    languages = {turn.text: turn.language for turn in corpus.turns}

    def with_office_nodes(body: Any) -> Reply:
        answer = json.loads(oracle.answer(body["messages"]))
        if "facts" in answer:
            source = harness.source_text(body["messages"]) or ""
            answer = _with_office_nodes(answer, languages.get(source, "en"))
        return Reply(json.dumps(answer, ensure_ascii=False))

    fake_openai.set_default(chat=Respond(with_office_nodes))
    kept = tmp_path / "kept" / "bench.db"
    argv = ["--months", "12", "--chatter", "0", "--budgets", "300", "--repeat", "1", "--quiet"]
    argv += ["--traps", "--name-traps", "--keep-store", str(kept), "--out", str(tmp_path / "out")]
    argv += ["--chat-url", fake_openai.base_url, "--model", "fake-chat"]
    assert harness.main(argv) == 0
    report = _only_report(tmp_path / "out")
    (told,) = (t for t in corpus.turns if t.event in question.expected)
    assert told.language == "fr"
    place = f"bureau de {corpus.gold[told.text]['facts'][0]['value']}"
    with Store(kept, harness.STORE_KEY) as store:
        node = store.find_node(place, Visibility.of_scope(question.scope))
        assert node is not None and node.kind == "place"  # the model's node was created
    outcomes = report["systems"]["iifa_tier2"]["300"]["per_question"]
    (outcome,) = (o for o in outcomes if o["question"] == "q003")
    assert outcome == {"question": "q003", "current": True, "stale": False, "correct": True}


# --------------------------------------------------------------------------- --intent, --worksession


def test_intent_and_worksession_leave_the_default_corpus_untouched() -> None:
    plain = generate(7)
    assert not any(t.intent or t.control or t.ends_session for t in plain.turns)
    for corpus in (generate(7, intent=True), generate(7, worksession=True)):
        assert sorted(corpus.events.values(), key=str) == sorted(plain.events.values(), key=str)
        assert [(q.query, q.expected) for q in corpus.questions] == [
            (q.query, q.expected) for q in plain.questions
        ]


def test_every_unmeant_turn_names_a_real_value_before_its_sincere_control() -> None:
    corpus = generate(7, months=4, intent=True)
    summary = corpus.summary()
    assert summary["intent_turns"] > 0 and summary["control_turns"] > 0
    assert summary["expected_episodes"] == 6  # a frame and a task persona, in three languages
    assert {t.language for t in corpus.turns if t.intent} == {"en", "fr", "ru"}
    for index, turn in enumerate(corpus.turns):
        if turn.intent is None or turn.intent == "pretend":
            continue
        answer = parse_answer(json.dumps(corpus.gold[turn.text], ensure_ascii=False))
        assert answer.rejected.total == 0 and len(answer.facts) == 1
        if turn.intent == "example":
            assert answer.frame == "example"
        else:
            assert answer.frame == "sincere" and answer.facts[0].modality == turn.intent
        # the sincere control comes next in the same session, on the same home
        control = next(t for t in corpus.turns[index + 1 :] if t.control)
        event = corpus.events[control.event or ""]
        assert control.session == turn.session and event.attribute == "city"
        assert answer.facts[0].value == CITIES[event.value][turn.language]


def test_a_persona_frame_spans_three_turns_and_leaves_one_episode() -> None:
    corpus = generate(7, months=3, intent=True)
    pretend = [t for t in corpus.turns if t.intent == "pretend"]
    assert len(pretend) == 12  # (open, inside, end) + a task persona, in three languages
    for turn in pretend:
        answer = parse_answer(json.dumps(corpus.gold[turn.text], ensure_ascii=False))
        assert answer.rejected.total == 0
        episodes = [f for f in answer.facts if f.episode]
        assert len(episodes) == int(turn.episode)
        for episode in episodes:
            assert "Dana" not in episode.statement and "Kazan" not in episode.statement
    assert len({t.session for t in pretend}) == 1


def test_an_oracle_intent_run_writes_nothing_unmeant_and_loses_no_control(
    tmp_path: Path,
) -> None:
    assert harness.main([*SMALL, "--intent", "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    assert report["corpus"]["intent"] is True
    intent = report["ingestion"]["intent"]
    unmeant = intent["unmeant"]
    assert unmeant["turns"] == report["corpus"]["intent_turns"] > 0
    assert unmeant["facts"] == unmeant["nodes"] == unmeant["turns_with_facts"] == 0
    assert unmeant["episodes"] == unmeant["expected_episodes"] == 6
    controls = intent["controls"]
    assert controls["turns"] == report["corpus"]["control_turns"] > 0
    assert controls["written"] == controls["turns"] and controls["lost"] == 0
    dropped = intent["dropped_by_intent"]
    assert dropped["frame"]["pretend"] > 0 and sum(dropped["modality"].values()) > 0
    assert report["ingestion"]["chatter"]["facts"] == 0  # unmeant turns are not chatter


def test_without_the_intent_reading_the_unmeant_turns_become_facts(tmp_path: Path) -> None:
    """The instrument sees the failure it is built for: the same oracle answers,
    read without intent, write the hypotheses and the persona as facts."""
    argv = [*SMALL, "--intent", "--no-read-intent", "--out", str(tmp_path)]
    assert harness.main(argv) == 0
    report = _only_report(tmp_path)
    assert report["pipeline"]["read_intent"] is False
    unmeant = report["ingestion"]["intent"]["unmeant"]
    assert unmeant["facts"] > 0 and unmeant["episodes"] == 0


def test_an_oracle_worksession_run_keeps_one_episode_and_every_lasting_fact(
    tmp_path: Path,
) -> None:
    assert harness.main([*SMALL, "--worksession", "--out", str(tmp_path)]) == 0
    report = _only_report(tmp_path)
    work = report["ingestion"]["worksession"]
    assert work["task_turns"] == 8 and work["task_facts_live"] == 0
    assert work["session_facts_left"] == 0 and work["episodes"] == 1
    assert work["lasting_events"] > 0 and work["lasting_written"] == work["lasting_events"]
    assert report["ingestion"]["intent"]["session_state"]["task"] == 8


def test_without_the_intent_reading_the_work_session_stays_in_memory(tmp_path: Path) -> None:
    argv = [*SMALL, "--worksession", "--no-read-intent", "--out", str(tmp_path)]
    assert harness.main(argv) == 0
    work = _only_report(tmp_path)["ingestion"]["worksession"]
    assert work["task_facts_live"] == work["task_turns"] and work["episodes"] == 0


def test_observed_adds_dated_turns_and_leaves_the_default_corpus_untouched() -> None:
    """Per thing, a lasting fact then observations in later months,
    each dated, under one key; the default corpus is the same."""
    plain, seen = generate(7), generate(7, observed=True)
    told = [t for t in seen.turns if t.observed]
    assert told and not any(t.observed for t in plain.turns)
    assert [t.text for t in seen.turns if t not in told] == [t.text for t in plain.turns]
    assert seen.questions == plain.questions
    assert seen.summary()["observed_turns"] == sum(1 for t in told if t.observation is not None)
    for name in {t.observed for t in told}:
        mine = [t for t in told if t.observed == name]
        assert mine[0].observation is None  # what it is for comes first
        days = [t.occurred_at for t in mine]
        assert days == sorted(days) and len(set(days)) == len(days)
        keys = set()
        for turn in mine:
            assert not seen.is_chatter(turn) and turn.event is None
            gold = seen.gold[turn.text]
            (fact,) = gold["facts"]
            assert fact["quote"] in turn.text and not parse_answer(json.dumps(gold)).rejected.total
            assert fact["duration"] == ("lasting" if turn.observation is None else "observed")
            keys.add(fact["attribute"])
        assert len(keys) == 1


def test_the_oracle_keeps_the_newest_observation_and_what_each_thing_is_for(
    tmp_path: Path,
) -> None:
    argv = ["--months", "4", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    assert harness.main([*argv, "--observed", "--out", str(tmp_path)]) == 0
    seen = _only_report(tmp_path)["ingestion"]["observed"]
    assert seen["things"] == 3 and seen["current"] == 3 and seen["kept"] == 3
    assert seen["stale"] == seen["closed"] == seen["missing"] == 0
    assert seen["recall"]["iifa_tier2"]["600"] == {"shown": 3, "stale_shown": 0, "stale_flagged": 0}
    # the same store read without the decay, beside it
    assert set(seen["recall"]) == {"iifa_tier2", "iifa_tier2_no_decay"}
    assert seen["recall"]["iifa_tier2_no_decay"]["600"]["shown"] == 3


def test_the_engine_without_decay_differs_from_its_engine_by_the_decay_alone(
    store: Store,
) -> None:
    engine = RecallEngine(
        store, ConservativeTokenizer(), tier=3, activity_threshold=lambda model: 0.65
    )
    twin = harness.without_decay(engine)
    assert twin.observed_half_life_days == 0 and engine.observed_half_life_days > 0
    changed = {k for k, v in vars(engine).items() if vars(twin)[k] is not v}
    assert changed == {"observed_half_life_days"}


def test_self_adds_a_scope_of_its_own_and_leaves_the_default_corpus_untouched() -> None:
    """The --self cases live in their own scope, each turn with the
    assistant's reply and its pages; the main corpus is the same."""
    plain = generate(7)
    agent = generate(7)
    add_self_turns(agent)
    told = [t for t in agent.turns if t.self_case]
    assert len(told) == len(CASES) and {t.scope for t in told} == {SELF_SCOPE}
    assert [t.text for t in agent.turns if t not in told] == [t.text for t in plain.turns]
    assert agent.questions == plain.questions and agent.summary()["self_turns"] == len(CASES)
    for turn in told:
        assert not agent.is_chatter(turn) and turn.source_key is not None
        gold = agent.gold[turn.source_key]
        assert not parse_answer(json.dumps(gold)).rejected.total
        for fact in gold["facts"]:
            assert fact["quote"] in turn.source_key  # every quote lies in the source


def test_the_oracle_writes_the_agents_memory_and_nothing_a_page_or_a_role_said(
    tmp_path: Path,
) -> None:
    argv = ["--months", "2", "--chatter", "0", "--budgets", "600", "--repeat", "1", "--quiet"]
    assert harness.main([*argv, "--self", "--out", str(tmp_path)]) == 0
    agent = _only_report(tmp_path)["ingestion"]["self"]
    for category, row in agent["categories"].items():
        assert row["written"] == row["expected_written"], category
        assert row["proposed"] == row["expected_proposed"], category
        assert row["echoes"] == row["expected_echoes"], category
    assert agent["hostile_written"] == agent["unmeant_written"] == 0
    assert agent["assistant_turns_not_sincere"] == 3
    assert agent["counters"] == {"self": 3, "method": 11, "self_proposed": 5, "self_echo": 1}
    assert agent["profile_commitments"] == {"held": 2, "in_profile": 2}
    probe = agent["probes"]["iifa_tier2"]["600"]
    assert probe["transfer_shown"] is True and probe["rules_total"] == 3
    # another map brings the map's rules first, within the blocks' share
    assert probe["activity_rules_total"] == 3 and probe["activity_rules_shown"] >= 1
    assert probe["activity_first"] is True
    # what the model named the activities, and the rules linked to each
    assert {a["name"]: a["rules"] for a in agent["activities"]}["drawing a map"] == 3
    (mapped,) = [f for f in agent["written_by_case"]["activity"] if f["key"] == "max_size"]
    assert mapped["activities"] == []  # a limit is not linked
    # four occurrences of one activity, ungrouped: three blocks at most
    assert agent["family"] == {"groups": {}, "rules_shown": {"iifa_tier2": {"600": "2/4"}}}
