"""The intent reading on its own: one answer and the frame its
session carries in, what goes on to the write path, what is proposed, what is
kept as session state, and the frame carried out."""

from __future__ import annotations

import json
from typing import Any

from iifa.ingest.intent import Carry, IntentReport, read
from iifa.ingest.schema import parse_answer


def answer(facts: list[dict[str, Any]], **top: Any) -> Any:
    return parse_answer(json.dumps({"nodes": [], "facts": facts, "edges": [], **top}))


def fact(statement: str, **extra: Any) -> dict[str, Any]:
    return {"subject": "User", "statement": statement, "quote": statement, **extra}


def speaker(subject: str | None) -> bool:
    return subject == "User"


def test_the_closed_words_are_read_folded_and_anything_else_is_rejected() -> None:
    parsed = answer(
        [fact("User lives in Porto.", modality=" Fact ", duration="LASTING")],
        frame="Pretend",
        frame_holds="yes",
    )
    assert parsed.frame == "pretend" and parsed.frame_holds is False
    assert parsed.rejected.keys == 1  # the flag that is not a boolean
    assert parsed.facts[0].modality == "fact" and parsed.facts[0].duration == "lasting"
    # a word outside the closed list is read as not given: a doubt, not a drop
    wrong = answer([fact("User lives in Porto.", modality="maybe", duration="forever")])
    assert wrong.rejected.facts == 0
    assert (wrong.facts[0].modality, wrong.facts[0].duration) == (None, None)


def test_a_carried_frame_reread_as_another_non_sincere_frame_keeps_its_episode() -> None:
    report = IntentReport()
    episode = fact("User tested a demo page.", episode=True)
    reading, carry = read(
        answer([episode], frame="staged"),
        Carry("pretend", True),
        about_speaker=speaker,
        report=report,
    )
    assert reading.kept.facts == [] and carry == Carry("staged", True)
    assert report.episodes["dropped"] == 1 and report.carried == 1


def test_a_frame_that_does_not_hold_is_not_carried() -> None:
    report = IntentReport()
    reading, carry = read(
        answer([fact("User is a genius.", modality="fact")], frame="irony"),
        Carry(),
        about_speaker=speaker,
        report=report,
    )
    assert carry == Carry() and reading.frame == "irony"
    assert report.dropped_frame == {"irony": 1}


def test_an_ended_frame_lets_the_same_answer_open_a_new_one() -> None:
    report = IntentReport()
    reading, carry = read(
        answer(
            [fact("User plays a pirate.", episode=True)],
            frame="pretend",
            frame_ends=True,
            frame_holds=True,
        ),
        Carry("staged", True),
        about_speaker=speaker,
        report=report,
        date="2026-03-14T10:00:00Z",
    )
    assert reading.ended and carry == Carry("pretend", True)
    (episode,) = reading.kept.facts
    assert episode.attribute == "activity" and episode.valid_from == "2026-03-14"
    assert report.ended == 1 and report.opened == 1


def test_an_unsure_frame_proposes_everything_and_carries_what_it_found() -> None:
    reading, carry = read(
        answer([fact("User lives in Porto."), fact("User likes tea.")], frame="unsure"),
        Carry("pretend", False),
        about_speaker=speaker,
        report=IntentReport(),
    )
    assert len(reading.proposed) == 2 and reading.kept.facts == []
    assert carry == Carry("pretend", False)


def test_in_a_sincere_frame_no_fact_is_an_episode_and_durations_split() -> None:
    report = IntentReport()
    reading, _ = read(
        answer(
            [
                fact("User lives in Porto.", modality="fact", duration="lasting", episode=True),
                fact("User's draft is two pages long.", modality="fact", duration="task"),
                fact("User feels tired.", modality="fact"),
            ],
            frame="sincere",
        ),
        Carry(),
        about_speaker=speaker,
        report=report,
    )
    assert [f.episode for f in reading.kept.facts] == [False]
    assert [f.duration for f in reading.session] == ["task", None]
    assert (report.session_task, report.session_unsure) == (1, 1)
    assert report.as_dict()["session"] == {"task": 1, "unsure": 1}


def test_an_edge_quoting_a_dropped_fact_goes_with_it_and_its_lone_node_too() -> None:
    raw = {
        "nodes": [{"name": "Karim"}, {"name": "Porto"}, {"name": "Lisbon"}],
        "facts": [
            {
                "subject": "Karim",
                "statement": "Karim lives in Porto.",
                "quote": "if Karim moved",
                "modality": "hypothesis",
            },
            {
                "subject": "Karim",
                "statement": "Karim works in Lisbon.",
                "quote": "works in Lisbon",
                "modality": "fact",
                "duration": "lasting",
            },
        ],
        "edges": [
            {"src": "Karim", "dst": "Porto", "type": "lives_in", "quote": "if Karim moved"},
            {"src": "Karim", "dst": "Lisbon", "type": "works_at", "quote": "works in Lisbon"},
        ],
        "frame": "sincere",
    }
    reading, _ = read(
        parse_answer(json.dumps(raw)), Carry(), about_speaker=speaker, report=IntentReport()
    )
    assert [e.dst for e in reading.kept.edges] == ["Lisbon"]
    assert [n.name for n in reading.kept.nodes] == ["Karim", "Lisbon"]


def test_an_episode_is_dated_only_by_a_date() -> None:
    """A precompact's reference is its trigger ("auto"), not a date."""
    reading, _ = read(
        answer([fact("User tried a demo.", episode=True)], frame="staged"),
        Carry(),
        about_speaker=speaker,
        report=IntentReport(),
        date="auto",
    )
    assert reading.kept.facts[0].valid_from is None


def test_an_edge_is_pruned_by_whole_words_only() -> None:
    raw = {
        "nodes": [],
        "facts": [
            {
                "subject": "Karim",
                "statement": "Karim likes art.",
                "quote": "art",
                "modality": "guess",
            },
        ],
        "edges": [{"src": "Karim", "dst": "Lisbon", "type": "knows", "quote": "the start"}],
        "frame": "sincere",
    }
    reading, _ = read(
        parse_answer(json.dumps(raw)), Carry(), about_speaker=speaker, report=IntentReport()
    )
    assert len(reading.kept.edges) == 1
