"""Probe scores (SPEC 11.2): run a stored probe set through ``recall`` exactly as
an agent would, and compare two scores — what the dream measures before and
after a run (SPEC 5.6 rule 9). Scores carry ids and numbers only (P5)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from iifa.ids import kind_of
from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.models import ProbeSet
from iifa.store.repo import Store

OK, REGRESSED, INCONCLUSIVE = "ok", "regressed", "inconclusive"
DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    expected: tuple[str, ...]
    found: tuple[str, ...]
    missing: tuple[str, ...]
    used_tokens: int
    recall: float
    tier: int | None = None
    summary_tokens: int = 0  # SPEC 11.2: what summary lines cost above the plain line

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.probe_id,
            "expected": list(self.expected),
            "found": list(self.found),
            "missing": list(self.missing),
            "used_tokens": self.used_tokens,
            "summary_tokens": self.summary_tokens,
            "recall": self.recall,
        }


@dataclass(frozen=True)
class ProbeScore:
    set_id: str
    scope: str
    budget_tokens: int
    tier: int | None  # the tier that served every probe, None when they differ
    recall: float | None
    used_tokens: float
    recall_per_1k_tokens: float | None
    scored: int
    probes: tuple[ProbeResult, ...]
    summary_tokens: float = 0.0  # mean over scored probes (SPEC 11.2)
    compared_per_1k_tokens: float | None = None  # recall per 1k tokens, summaries aside

    @property
    def scored_ids(self) -> frozenset[str]:
        return frozenset(p.probe_id for p in self.probes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "scope": self.scope,
            "budget_tokens": self.budget_tokens,
            "tier": self.tier,
            "recall": self.recall,
            "used_tokens": self.used_tokens,
            "recall_per_1k_tokens": self.recall_per_1k_tokens,
            "summary_tokens": self.summary_tokens,
            "compared_per_1k_tokens": self.compared_per_1k_tokens,
            "scored": self.scored,
            "probes": [p.as_dict() for p in self.probes],
        }


def per_1k(recall: float, tokens: float) -> float | None:
    return recall / tokens * 1000 if tokens > 0 else None


def run_probe_set(engine: RecallEngine, store: Store, probe_set: ProbeSet) -> ProbeScore:
    """Ask every probe through ``recall`` (the set's scope and ancestors, the set's
    budget). An expected record is found when the recall returns it or what
    replaced it (P7: ``superseded_by``; SPEC 5.6: ``consolidated_into``,
    ``merged_into``)."""
    view = Visibility.of_scope(probe_set.scope)
    results: list[ProbeResult] = []
    tiers: set[int] = set()
    for probe in probe_set.probes:  # never empty: forget deletes a probe that lost an answer
        answer, summary_tokens = engine.measured_recall(
            probe.query, view=view, budget_tokens=probe_set.budget_tokens
        )
        tier = int(answer["tier"])
        tiers.add(tier)
        returned = {item["id"] for item in answer["items"]}
        found = tuple(e for e in probe.expected if _answered(store, e, returned))
        missing = tuple(e for e in probe.expected if e not in found)
        results.append(
            ProbeResult(
                probe.id,
                probe.expected,
                found,
                missing,
                int(answer["used_tokens"]),
                len(found) / len(probe.expected),
                tier,
                summary_tokens,
            )
        )
    recall = statistics.fmean(r.recall for r in results) if results else None
    tokens = statistics.fmean(r.used_tokens for r in results) if results else 0.0
    summaries = statistics.fmean(r.summary_tokens for r in results) if results else 0.0
    return ProbeScore(
        set_id=probe_set.id,
        scope=probe_set.scope,
        budget_tokens=probe_set.budget_tokens,
        tier=next(iter(tiers)) if len(tiers) == 1 else None,
        recall=recall,
        used_tokens=tokens,
        recall_per_1k_tokens=per_1k(recall, tokens) if recall is not None else None,
        scored=len(results),
        probes=tuple(results),
        summary_tokens=summaries,
        compared_per_1k_tokens=per_1k(recall, tokens - summaries) if recall is not None else None,
    )


def _answered(store: Store, expected: str, returned: set[str]) -> bool:
    """Found as itself, or as what replaced it: the fact that superseded it (P7)
    or that a dream consolidated it into, the node a dream merged it into."""
    if expected in returned:
        return True
    if kind_of(expected) not in ("fact", "node"):
        return False
    return any(successor in returned for successor in store.successors(expected))


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str | None
    recall_delta: float | None = None
    per_token_delta: float | None = None  # relative change of recall per 1k tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "recall_delta": self.recall_delta,
            "per_token_delta": self.per_token_delta,
        }


def _rates(before: ProbeScore, after: ProbeScore) -> tuple[float | None, float | None]:
    """The two rates a comparison reads: both with summaries aside, or both raw
    when either score lacks ``compared_per_1k_tokens`` (saved before summary
    tokens were counted) - never one of each, which would favour the dream."""
    if before.compared_per_1k_tokens is None or after.compared_per_1k_tokens is None:
        return before.recall_per_1k_tokens, after.recall_per_1k_tokens
    return before.compared_per_1k_tokens, after.compared_per_1k_tokens


def compare(
    before: ProbeScore, after: ProbeScore, *, tolerance: float = DEFAULT_TOLERANCE
) -> Verdict:
    """``regressed`` when recall drops by more than ``tolerance`` (absolute) or
    recall per 1k tokens, the summary tokens of both scores set aside
    (``compared_per_1k_tokens``; both raw if either score lacks it), by more
    than ``tolerance`` (relative);
    ``inconclusive`` when the two runs were served by different tiers or scored
    different probes."""
    if before.tier is None or after.tier is None or before.tier != after.tier:
        return Verdict(INCONCLUSIVE, "tier")
    if not before.scored_ids or before.scored_ids != after.scored_ids:
        return Verdict(INCONCLUSIVE, "probes")
    assert before.recall is not None and after.recall is not None
    recall_delta = after.recall - before.recall
    per_token_delta: float | None = None
    before_rate, after_rate = _rates(before, after)
    if before_rate:
        per_token_delta = (after_rate or 0.0) / before_rate - 1
    if recall_delta < -tolerance:
        return Verdict(REGRESSED, "recall", recall_delta, per_token_delta)
    if per_token_delta is not None and per_token_delta < -tolerance:
        return Verdict(REGRESSED, "recall_per_1k_tokens", recall_delta, per_token_delta)
    return Verdict(OK, None, recall_delta, per_token_delta)
