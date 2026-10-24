"""Review by risk (SPEC 5.6 rule 5).

Each risk class either applies automatically (``auto``) or goes to the review
queue (``review``). The default: only ``low`` applies; ``medium`` and ``high``
go to review (measured with qwen3.5:9b on the synthetic corpus: the one
consolidation proposed was wrong, no theme was ever proposed). The operator
declares a policy per scope (the nearest declaration of each risk class wins,
like a language override); a request may tighten it, never loosen it.
``merge_nodes`` always goes to review, whatever the policy says, permanently
("never merge silently").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from iifa.config import RISK_CLASSES
from iifa.dream.catalogue import RISK
from iifa.scopes import ROOT_SCOPE, within

AUTO, REVIEW = "auto", "review"
# Permanent, not pending a measurement: the model follows an identity claim
# found in memory, and between two namesakes the names fit - only a reviewer
# stops that merge (SPEC 5.6 rule 5).
ALWAYS_REVIEW: frozenset[str] = frozenset({"merge_nodes"})


class PolicyError(ValueError):
    """A request tried to loosen the configured policy."""


@dataclass(frozen=True)
class ReviewPolicy:
    low: str = AUTO
    medium: str = REVIEW
    high: str = REVIEW

    @staticmethod
    def default() -> ReviewPolicy:
        return ReviewPolicy()

    def action(self, risk: str) -> str:
        value: str = getattr(self, risk)
        return value

    def route(self, operation: str) -> str:
        """``auto`` or ``review`` for one operation of the catalogue."""
        if operation in ALWAYS_REVIEW:
            return REVIEW
        return self.action(RISK[operation])

    def tightened(self, requested: Mapping[str, str | None]) -> ReviewPolicy:
        """This policy with a request's changes, which may only send more to review."""
        changes: dict[str, str] = {}
        for risk, action in requested.items():
            if action is None:
                continue
            if risk not in RISK_CLASSES or action not in (AUTO, REVIEW):
                raise PolicyError(f"unknown review policy entry {risk}={action}")
            if action == AUTO and self.action(risk) == REVIEW:
                raise PolicyError(f"a request may not loosen the review policy: {risk} is reviewed")
            changes[risk] = action
        return replace(self, **changes)

    def as_dict(self) -> dict[str, str]:
        return {"low": self.low, "medium": self.medium, "high": self.high}

    @staticmethod
    def from_dict(data: Mapping[str, str]) -> ReviewPolicy:
        return ReviewPolicy(**{risk: str(data[risk]) for risk in RISK_CLASSES if risk in data})


def resolve_policy(scope: str, declared: Sequence[tuple[str, str, str]]) -> ReviewPolicy:
    """The policy of ``scope``: for each risk class, the declaration of the nearest
    scope at or above it (``*`` the farthest), else the default."""
    policy = ReviewPolicy.default()
    for risk in RISK_CLASSES:
        best: tuple[int, str] | None = None
        for declared_scope, declared_risk, action in declared:
            if declared_risk != risk:
                continue
            if declared_scope == ROOT_SCOPE:
                depth = 0
            elif within(scope, declared_scope):
                depth = declared_scope.count("/") + 1
            else:
                continue
            if best is None or depth >= best[0]:
                best = (depth, action)
        if best is not None:
            policy = replace(policy, **{risk: best[1]})
    return policy


__all__ = ["ALWAYS_REVIEW", "AUTO", "REVIEW", "PolicyError", "ReviewPolicy", "resolve_policy"]
