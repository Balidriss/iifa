"""Scopes as sets, the three read modes, and authorization (SPEC 4.6, P9)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from iifa.scopes import (
    ScopeError,
    Visibility,
    ancestors,
    authorized,
    touches,
    validate_scope,
    within,
)

segment = st.from_regex(r"[a-z]{1,4}:[a-z0-9]{1,6}", fullmatch=True)
scope_path = st.lists(segment, min_size=1, max_size=4).map("/".join)


def test_validate_scope_accepts_paths_and_rejects_junk() -> None:
    assert validate_scope("user:alice/project:solar") == "user:alice/project:solar"
    for bad in ("", "alice", "user:", ":alice", "user:al ice", "user:alice//x", "a" * 600):
        with pytest.raises(ScopeError):
            validate_scope(bad)


def test_ancestors_closest_first() -> None:
    assert ancestors("user:alice/agent:coder/task:1") == [
        "user:alice/agent:coder/task:1",
        "user:alice/agent:coder",
        "user:alice",
    ]


def test_within_and_root() -> None:
    assert within("user:alice/agent:coder", "user:alice")
    assert within("user:alice", "user:alice")
    assert not within("user:alice2", "user:alice")
    assert not within("user:alice", "user:alice/agent:coder")
    assert within("anything:x", "*")


def test_authorized_needs_every_scope_inside_a_bound() -> None:
    assert authorized(["user:alice/x:1"], ["user:alice"])
    assert not authorized(["user:alice", "user:bob"], ["user:alice"])
    assert not authorized([], ["*"])
    assert touches(["user:bob", "user:alice/x:1"], ["user:alice"])
    assert not touches(["user:bob"], ["user:alice"])


def test_read_modes() -> None:
    default = Visibility.of_scope("user:alice/project:solar")
    assert default.includes("user:alice") and default.includes("user:alice/project:solar")
    assert not default.includes("user:alice/project:wind")
    explicit = Visibility.of_scopes(["user:alice/project:solar", "user:bob"])
    assert explicit.includes("user:bob") and not explicit.includes("user:alice")
    subtree = Visibility.of_subtree("user:alice")
    assert subtree.includes("user:alice") and subtree.includes("user:alice/project:wind/x:y")
    assert not subtree.includes("user:alice2") and not subtree.includes("user:bob")
    with pytest.raises(ScopeError):
        Visibility.of_scopes([])


def test_rank_prefers_the_requesters_own_scope() -> None:
    view = Visibility.of_scope("user:alice/agent:coder")
    assert view.rank(["user:alice/agent:coder"]) < view.rank(["user:alice"])
    assert view.rank(["user:bob"]) > view.rank(["user:alice"])
    assert Visibility.of_subtree("user:alice").rank(["user:alice/x:1"]) == 1


def test_sql_predicate_escapes_like_wildcards() -> None:
    sql, params = Visibility.of_subtree("user:al_ice%").sql("rs.scope")
    assert sql == "(rs.scope IN (?) OR rs.scope LIKE ? ESCAPE '\\')"
    assert params == ["user:al_ice%", "user:al\\_ice\\%/%"]
    assert Visibility().sql("x") == ("0", [])
    assert Visibility().is_empty


@given(
    record=st.lists(scope_path, min_size=1, max_size=3),
    view=st.lists(scope_path, min_size=1, max_size=3),
)
@settings(max_examples=100, deadline=None)
def test_p9_visibility_is_exactly_any_scope_in_the_view(record: list[str], view: list[str]) -> None:
    """A record is visible iff at least one of its scopes is in the view - for every mode."""
    explicit = Visibility.of_scopes(view)
    assert explicit.sees(record) == any(s in view for s in record)
    default = Visibility.of_scope(view[0])
    assert default.sees(record) == any(s in ancestors(view[0]) for s in record)
    subtree = Visibility.of_subtree(view[0])
    assert subtree.sees(record) == any(within(s, view[0]) for s in record)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        # a level without its kind: the mistake a client actually made
        (
            "user:alice/assistant",
            "segment 'assistant' needs kind:name, like mode:assistant",
        ),
        ("assistant", "segment 'assistant' needs kind:name, like mode:assistant"),
        ("user:alice//x:1", "empty segment"),
        ("user:alice/", "empty segment"),
        (":alice", "segment ':alice' needs a kind before ':'"),
        ("user:", "segment 'user:' needs a name after ':'"),
        ("us-er:alice", "kind 'us-er' takes letters, digits and _ only"),
        ("user:al ice", "segment 'user:al ice' contains whitespace"),
    ],
)
def test_a_bad_scope_names_the_offending_segment_and_the_expected_form(
    scope: str, expected: str
) -> None:
    with pytest.raises(ScopeError) as caught:
        validate_scope(scope)
    message = str(caught.value)
    assert expected in message
    assert repr(scope) in message  # the whole path too, to find where it came from
