"""Ids: prefix plus ULID, and the ULID sequence is monotonic within a process,
so records written in a burst (same millisecond, or a clock that stepped back)
still sort in creation order; every query that breaks ties on the id depends
on it."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from itertools import pairwise

import pytest

from iifa import ids
from iifa.ids import decode_ulid, kind_of, new_id, new_ulid


@pytest.fixture
def fresh_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the process-wide sequence over, so a scripted clock is not seen as
    a clock that stepped back; monkeypatch restores the real state afterwards."""
    monkeypatch.setattr(ids._sequence, "timestamp", -1)
    monkeypatch.setattr(ids._sequence, "randomness", 0)


def clock(values: list[int], monkeypatch: pytest.MonkeyPatch) -> None:
    """``_now_ms`` answers ``values`` in order, then repeats the last one."""
    remaining: Iterator[int] = iter(values)
    last = values[-1]

    def now_ms() -> int:
        nonlocal last
        last = next(remaining, last)
        return last

    monkeypatch.setattr(ids, "_now_ms", now_ms)


def test_ids_have_prefix_and_ulid_shape() -> None:
    fact_id = new_id("fact")
    assert fact_id.startswith("fact_") and len(fact_id) == 31
    assert kind_of(fact_id) == "fact"
    assert kind_of("nope_" + "0" * 26) is None
    assert kind_of("fact_short") is None
    assert kind_of("fact_" + "U" * 26) is None  # U is not a Crockford digit


def test_ulids_sort_by_time_and_round_trip() -> None:
    assert new_ulid(1) < new_ulid(2)
    assert decode_ulid(new_ulid(123_456))[0] == 123_456
    timestamp, randomness = decode_ulid(new_ulid())
    assert timestamp > 1_600_000_000_000 and 0 <= randomness < (1 << 80)
    with pytest.raises(ValueError, match="not a ULID"):
        decode_ulid("short")
    assert len({new_ulid() for _ in range(50)}) == 50


def test_unknown_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown id prefix"):
        new_id("thing")


def test_thousands_of_ids_in_a_tight_loop_sort_in_creation_order() -> None:
    created = [new_ulid() for _ in range(20_000)]
    assert len(set(created)) == len(created)
    assert all(a < b for a, b in pairwise(created)), "an id was not greater than its predecessor"
    assert created == sorted(created)
    stamps = [decode_ulid(u)[0] for u in created]
    assert stamps == sorted(stamps)
    # the loop is faster than the clock: the same-millisecond path was exercised
    assert any(a == b for a, b in pairwise(stamps))
    prefixed = [new_id("fact") for _ in range(2_000)]
    assert prefixed == sorted(prefixed) and len(set(prefixed)) == 2_000


def test_same_millisecond_increments_the_previous_random_part(
    fresh_sequence: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock([1_000], monkeypatch)
    monkeypatch.setattr(ids, "_random_bits", lambda: 12_345)
    first, second, third = new_ulid(), new_ulid(), new_ulid()
    assert decode_ulid(first) == (1_000, 12_345)
    assert decode_ulid(second) == (1_000, 12_346)
    assert decode_ulid(third) == (1_000, 12_347)
    assert first < second < third
    # an explicit timestamp does not disturb the sequence
    assert decode_ulid(new_ulid(5))[0] == 5
    assert decode_ulid(new_ulid()) == (1_000, 12_348)


def test_a_clock_that_steps_back_keeps_the_order(
    fresh_sequence: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock([2_000, 1_500, 1_600, 2_001], monkeypatch)
    monkeypatch.setattr(ids, "_random_bits", lambda: 7)
    first, back, still_back, later = new_ulid(), new_ulid(), new_ulid(), new_ulid()
    assert decode_ulid(first) == (2_000, 7)
    assert decode_ulid(back) == (2_000, 8)  # the wall clock went back: the sequence did not
    assert decode_ulid(still_back) == (2_000, 9)
    assert decode_ulid(later) == (2_001, 7)
    assert first < back < still_back < later


def test_random_overflow_waits_for_the_next_millisecond(
    fresh_sequence: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock([3_000, 3_000, 3_001], monkeypatch)  # the first id, the overflow, the wait
    monkeypatch.setattr(ids, "_random_bits", lambda: ids._MAX_RANDOM)
    naps: list[float] = []
    monkeypatch.setattr(ids.time, "sleep", naps.append)
    first, second = new_ulid(), new_ulid()
    assert decode_ulid(first) == (3_000, ids._MAX_RANDOM)
    assert decode_ulid(second) == (3_001, ids._MAX_RANDOM)  # waited, then fresh bits
    assert first < second and naps == [ids._WAIT_SECONDS]


def test_threads_get_unique_ids_each_in_creation_order() -> None:
    per_thread: dict[int, list[str]] = {}

    def burst(index: int) -> None:
        per_thread[index] = [new_ulid() for _ in range(2_000)]

    threads = [threading.Thread(target=burst, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    everything = [u for created in per_thread.values() for u in created]
    assert len(set(everything)) == 16_000
    for created in per_thread.values():
        assert created == sorted(created) and all(a < b for a, b in pairwise(created))
