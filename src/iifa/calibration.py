"""Per-model similarity thresholds (SPEC 6.1).

A cosine means something for one embedding model only: bge-m3 puts one
person's name across scripts at 0.64-0.69 and other people at 0.25-0.38,
qwen3-embedding:0.6b at 0.85-0.92 and 0.53-0.79. So there is no global default:
thresholds come from a built-in profile keyed by the model, from the
configuration (which overrides a profile, or calibrates an unknown model), or
not at all - then semantic resolution, merge proposals and semantic links are
off and ``/v1/health`` says ``uncalibrated``. Dense recall needs no threshold
(RRF fuses ranks).

``resolve`` and ``merge`` come from ``bench/thresholds.py``: ``resolve`` is the
95th percentile of two-entity pairs (it only sends a candidate to the
adjudication), ``merge`` sits above every near-name trap (Bogdan / Bogdana)
with at most 2 % of two-entity pairs above it. ``link`` has no ground truth to
calibrate on, so the profiles stay marked **provisional**.
"""

from __future__ import annotations

from dataclasses import dataclass

UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True)
class Thresholds:
    resolve: float  # a near node becomes a candidate for the adjudication
    merge: float  # two existing near nodes become a merge proposal
    link: float  # two close summaries get a proposed similar_to edge
    source: str = "config"
    provisional: bool = False


PROFILES: dict[str, Thresholds] = {
    # resolve catches 85 % of one entity's forms, merge 5 % (traps reach 0.84)
    "bge-m3": Thresholds(0.53, 0.84, 0.70, "profile:bge-m3", provisional=True),
    # resolve catches 52 %, merge 1 % (traps reach 0.97)
    "qwen3-embedding:0.6b": Thresholds(
        0.84, 0.98, 0.70, "profile:qwen3-embedding:0.6b", provisional=True
    ),
}


def profile_key(model: str) -> str:
    """``Bge-m3:LATEST`` -> ``bge-m3``: case and a ``:latest`` tag do not matter."""
    key = model.strip().lower()
    return key.removesuffix(":latest")


def thresholds_for(
    model: str,
    *,
    resolve: float | None = None,
    merge: float | None = None,
    link: float | None = None,
) -> Thresholds | None:
    """The thresholds for ``model``: its profile with any configured value
    overriding, or the three configured values for a model without a profile,
    or None (uncalibrated)."""
    profile = PROFILES.get(profile_key(model))
    configured = (resolve, merge, link)
    if profile is None:
        if resolve is None or merge is None or link is None:
            return None
        return Thresholds(resolve, merge, link)
    if all(value is None for value in configured):
        return profile
    return Thresholds(
        profile.resolve if resolve is None else resolve,
        profile.merge if merge is None else merge,
        profile.link if link is None else link,
        source=f"{profile.source}+config",
        provisional=profile.provisional,
    )


# SPEC 5.4 step 4: no profile measures attribute keys yet, so the embedding
# match of synonym keys is off unless the operator configures ``key_threshold``.
KEY_PROFILES: dict[str, float] = {}


def key_threshold_for(model: str, *, configured: float | None = None) -> float | None:
    """The cosine above which a new attribute key is aligned with an existing one
    (SPEC 5.4), independent of the three thresholds above; None: the step is off."""
    if configured is not None:
        return configured
    return KEY_PROFILES.get(profile_key(model))


def describe(thresholds: Thresholds | None, *, key: float | None = None) -> dict[str, object]:
    """What ``/v1/health`` shows."""
    if thresholds is None:
        return {"status": UNCALIBRATED, "key": key}
    status = "config" if thresholds.source == "config" else thresholds.source
    return {
        "status": status,
        "resolve": thresholds.resolve,
        "merge": thresholds.merge,
        "link": thresholds.link,
        "key": key,
        "provisional": thresholds.provisional,
    }


__all__ = [
    "KEY_PROFILES",
    "PROFILES",
    "UNCALIBRATED",
    "Thresholds",
    "describe",
    "key_threshold_for",
    "profile_key",
    "thresholds_for",
]
