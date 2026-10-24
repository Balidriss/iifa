"""Vector arithmetic for tier 3 (SPEC 6.1).

Vectors are stored unit-normalised as float32 blobs, so a cosine is a dot
product. Two paths compute the same ranking: numpy when the optional
``vectors`` extra is installed (milliseconds for 10 000 vectors of dimension
1024), otherwise pure Python (``math.sumprod`` on 3.12, a slower loop on 3.11):
correct, only slower, and ``/v1/health`` says which one runs and past which
size the 50 ms target of SPEC 7.4 is missed without numpy (P10).

Reciprocal rank fusion (``rrf``) merges the lexical and dense rankings.
"""

from __future__ import annotations

import math
import sys
from array import array
from collections.abc import Sequence
from types import ModuleType

try:  # the optional extra iifa[vectors]
    import numpy as _numpy_module

    _numpy: ModuleType | None = _numpy_module
except ImportError:  # pragma: no cover - depends on the install
    _numpy = None

NUMPY, PYTHON = "numpy", "python"
RRF_K = 60

# Measured on a 16-core laptop: 10 000 dot products of dimension 1024 take
# ~124 ms with math.sumprod (3.12) and ~370 ms with a generator loop (3.11); the
# sizes below keep a scan under the 50 ms recall target at dimension 1024.
_PYTHON_LIMIT_SUMPROD = 4000
_PYTHON_LIMIT_LOOP = 1300


def active_path() -> str:
    """``numpy`` when the extra is installed, else ``python``."""
    return NUMPY if _numpy is not None else PYTHON


def python_path_limit() -> int:
    """How many vectors (dimension 1024) the pure-Python path scans within the
    50 ms target; more is still correct, only slower."""
    return _PYTHON_LIMIT_SUMPROD if sys.version_info >= (3, 12) else _PYTHON_LIMIT_LOOP


def normalise(values: Sequence[float]) -> list[float]:
    """The unit vector of ``values``; refuses empty, zero or non-finite vectors."""
    if not values:
        raise ValueError("an empty vector")
    floats = [float(v) for v in values]
    if not all(math.isfinite(v) for v in floats):
        raise ValueError("a vector with a non-finite value")
    norm = math.sqrt(math.fsum(v * v for v in floats))
    if norm == 0.0:
        raise ValueError("a zero vector has no direction")
    return [v / norm for v in floats]


def pack(values: Sequence[float]) -> bytes:
    """Unit-normalised float32 bytes, what the store keeps."""
    return array("f", normalise(values)).tobytes()


def unpack(blob: bytes) -> array[float]:
    return array("f", blob)


def dimension_of(blob: bytes) -> int:
    return len(blob) // 4


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    sumprod = getattr(math, "sumprod", None)
    if sumprod is not None:
        return float(sumprod(a, b))
    return math.fsum(x * y for x, y in zip(a, b, strict=True))  # pragma: no cover - 3.11


def rank(
    query: Sequence[float], blobs: Sequence[bytes], *, path: str | None = None
) -> list[tuple[int, float]]:
    """``(index, cosine)`` of every blob with the query's dimension, best first,
    ties by index. Blobs of another dimension (stale vectors) are skipped."""
    unit = normalise(query)
    size = len(unit) * 4
    indices = [i for i, blob in enumerate(blobs) if len(blob) == size]
    if not indices:
        return []
    chosen = path or active_path()
    if chosen == NUMPY:
        if _numpy is None:
            raise RuntimeError("the numpy path needs the iifa[vectors] extra")
        matrix = _numpy.frombuffer(b"".join(blobs[i] for i in indices), dtype=_numpy.float32)
        matrix = matrix.reshape(len(indices), len(unit))
        scores = (matrix @ _numpy.asarray(unit, dtype=_numpy.float32)).tolist()
    else:
        query32 = unpack(array("f", unit).tobytes())
        scores = [_dot(query32, unpack(blobs[i])) for i in indices]
    ranked = [(index, float(score)) for index, score in zip(indices, scores, strict=True)]
    ranked.sort(key=lambda pair: (-pair[1], pair[0]))
    return ranked


def rrf(rankings: Sequence[Sequence[str]], *, k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each ranking adds ``1 / (k + rank)`` (rank from 1)
    to its ids. Best first, ties by id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


__all__ = [
    "NUMPY",
    "PYTHON",
    "RRF_K",
    "active_path",
    "dimension_of",
    "normalise",
    "pack",
    "python_path_limit",
    "rank",
    "rrf",
    "unpack",
]
