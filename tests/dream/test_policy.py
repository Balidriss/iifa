"""Review by risk (SPEC 5.6 rule 5): only low applies by default, medium and high
go to review (measured: one consolidation, wrong, and no theme); the operator
sets it per scope (nearest declaration wins); a request may only tighten it;
merge_nodes always goes to review."""

from __future__ import annotations

import pytest

from iifa.dream.policy import AUTO, REVIEW, PolicyError, ReviewPolicy, resolve_policy


def test_the_default_policy() -> None:
    policy = ReviewPolicy.default()
    assert policy.route("refresh_summary") == AUTO
    assert policy.route("flag_conflict") == AUTO
    assert policy.route("create_theme") == REVIEW
    assert policy.route("consolidate_facts") == REVIEW
    assert policy.route("supersede_fact") == REVIEW
    assert policy.route("merge_nodes") == REVIEW


def test_the_nearest_declaration_wins() -> None:
    declared = (
        ("user:alice", "medium", "auto"),
        ("user:alice/project:solar", "medium", "review"),
    )
    assert resolve_policy("user:alice", declared).route("create_theme") == AUTO
    assert resolve_policy("user:alice/agent:coder", declared).route("create_theme") == AUTO
    assert resolve_policy("user:alice/project:solar", declared).route("create_theme") == REVIEW
    assert resolve_policy("user:bob", declared).route("create_theme") == REVIEW


def test_merge_nodes_always_goes_to_review_whatever_the_policy() -> None:
    """'Never merge silently': a real model missed 18 of 22 true merges; no policy
    lifts it."""
    policy = resolve_policy("user:alice", (("user:alice", "high", "auto"),))
    assert policy.route("supersede_fact") == AUTO
    assert policy.route("merge_nodes") == REVIEW


def test_a_request_may_tighten_the_policy() -> None:
    policy = ReviewPolicy.default().tightened({"low": "review"})
    assert policy.route("refresh_summary") == REVIEW
    assert policy.route("supersede_fact") == REVIEW


def test_a_request_may_not_loosen_the_policy() -> None:
    with pytest.raises(PolicyError, match="high"):
        ReviewPolicy.default().tightened({"high": "auto"})
    with pytest.raises(PolicyError, match="medium"):
        ReviewPolicy.default().tightened({"medium": "auto"})


def test_a_request_repeating_the_policy_is_fine() -> None:
    policy = ReviewPolicy.default().tightened({"low": "auto", "high": "review"})
    assert policy == ReviewPolicy.default()


def test_the_policy_round_trips_through_the_journal() -> None:
    policy = ReviewPolicy.default().tightened({"low": "review"})
    assert ReviewPolicy.from_dict(policy.as_dict()) == policy
