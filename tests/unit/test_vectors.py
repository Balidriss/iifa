"""Vector arithmetic for tier 3 (SPEC 6.1): unit-normalised float32 blobs, a
numpy path and a pure-Python path that rank identically, and reciprocal rank
fusion of the lexical and dense rankings."""

from __future__ import annotations

import math
import random
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iifa import vectors
from iifa.vectors import NUMPY, PYTHON, active_path, pack, python_path_limit, rank, rrf, unpack


def test_pack_normalises_to_unit_float32() -> None:
    blob = pack([3.0, 4.0])
    assert len(blob) == 8  # two float32
    values = unpack(blob)
    assert values[0] == pytest.approx(0.6) and values[1] == pytest.approx(0.8)
    assert math.fsum(v * v for v in values) == pytest.approx(1.0)


def test_a_zero_or_non_finite_vector_is_refused() -> None:
    with pytest.raises(ValueError):
        pack([0.0, 0.0])
    with pytest.raises(ValueError):
        pack([float("nan"), 1.0])
    with pytest.raises(ValueError):
        pack([])


def test_rank_orders_by_cosine_then_index() -> None:
    blobs = [pack([1.0, 0.0]), pack([0.0, 1.0]), pack([1.0, 1.0]), pack([1.0, 0.0])]
    for path in (PYTHON, NUMPY):
        ranked = rank([1.0, 0.0], blobs, path=path)
        assert [i for i, _ in ranked] == [0, 3, 2, 1]
        assert ranked[0][1] == pytest.approx(1.0) and ranked[2][1] == pytest.approx(
            0.7071, abs=1e-4
        )


def test_rank_skips_vectors_of_another_dimension() -> None:
    # stale vectors are never compared with fresh ones (SPEC 6)
    blobs = [pack([1.0, 0.0]), pack([1.0, 0.0, 0.0]), pack([0.5, 0.5])]
    for path in (PYTHON, NUMPY):
        assert [i for i, _ in rank([1.0, 0.0], blobs, path=path)] == [0, 2]


def test_rank_of_nothing_is_nothing() -> None:
    assert rank([1.0, 0.0], [], path=PYTHON) == []
    assert rank([1.0, 0.0], [], path=NUMPY) == []


@settings(max_examples=60, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=60),
    st.integers(min_value=2, max_value=48),
)
def test_numpy_and_pure_python_paths_rank_identically(seed: int, count: int, dim: int) -> None:
    """SPEC 6.1: the optional numpy extra changes the speed, never the result."""
    rng = random.Random(seed)
    blobs = []
    for _ in range(count):
        values = [rng.uniform(-1, 1) for _ in range(dim)]
        if not any(values):
            values[0] = 1.0
        blobs.append(pack(values))
    query = [rng.uniform(-1, 1) for _ in range(dim)] or [1.0]
    query[0] = query[0] or 1.0
    by_python = rank(query, blobs, path=PYTHON)
    by_numpy = rank(query, blobs, path=NUMPY)
    # same scores position by position; the same order unless two scores tie
    # closer than float32 noise
    for (i, a), (j, b) in zip(by_python, by_numpy, strict=True):
        assert a == pytest.approx(b, abs=1e-5)
        if i != j:
            assert dict(by_python)[j] == pytest.approx(a, abs=1e-5)


def test_the_active_path_and_the_python_limit_are_reported() -> None:
    assert active_path() == NUMPY  # the dev install carries the extra
    # the size past which the pure-Python path misses the 50 ms target (SPEC 7.4)
    limit = python_path_limit()
    assert 100 <= limit <= 10_000
    if sys.version_info < (3, 12):
        assert limit < 2000  # no math.sumprod: a slower loop


def test_without_numpy_the_python_path_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vectors, "_numpy", None)
    assert active_path() == PYTHON
    blobs = [pack([0.0, 1.0]), pack([1.0, 0.0])]
    assert [i for i, _ in rank([1.0, 0.0], blobs)] == [1, 0]


def test_rrf_fuses_rankings_by_reciprocal_rank() -> None:
    lexical = ["fact_a", "fact_b", "fact_c"]
    dense = ["fact_c", "fact_a", "fact_d"]
    fused = rrf([lexical, dense], k=60)
    scores = dict(fused)
    assert scores["fact_a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["fact_c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["fact_d"] == pytest.approx(1 / 63)
    assert [doc for doc, _ in fused] == ["fact_a", "fact_c", "fact_b", "fact_d"]


def test_rrf_breaks_ties_by_id_and_ignores_empty_rankings() -> None:
    assert rrf([["b", "a"], []], k=60) == [
        ("b", pytest.approx(1 / 61)),
        ("a", pytest.approx(1 / 62)),
    ]
    assert rrf([["a"], ["b"]], k=60) == [("a", pytest.approx(1 / 61)), ("b", pytest.approx(1 / 61))]
    assert rrf([], k=60) == []
