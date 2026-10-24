"""Scopes (SPEC section 4.6): what a reader may see and what a writer may touch.

A record belongs to a *set* of scope paths and is visible when any of them is
visible. A reader's view is one of three modes:

* ``Visibility.of_scope(s)`` - ``s`` and its ancestors (the default);
* ``Visibility.of_scopes([...])`` - exactly those scopes;
* ``Visibility.of_subtree(root)`` - ``root`` and everything below (admin only).

The view is applied in SQL before any scoring or traversal happens (P9).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ROOT_SCOPE = "*"


class ScopeError(ValueError):
    """A malformed scope path."""


def validate_scope(scope: str) -> str:
    """A scope path: ``kind:name`` segments joined by ``/``, e.g. ``user:alice/project:solar``."""
    if not scope or len(scope) > 512:
        raise ScopeError("scope must be a non-empty path of at most 512 characters")
    for segment in scope.split("/"):
        problem = _segment_problem(segment)
        if problem:
            raise ScopeError(f"invalid scope {scope!r}: {problem}")
    return scope


def _segment_problem(segment: str) -> str | None:
    """What is wrong with one ``/`` level, said with the form it should take."""
    kind, sep, name = segment.partition(":")
    checks = (
        (not segment, "empty segment (two '/' in a row, or one at an end)"),
        (any(c.isspace() for c in segment), f"segment {segment!r} contains whitespace"),
        (not sep, f"segment {segment!r} needs kind:name, like mode:{segment}"),
        (not kind, f"segment {segment!r} needs a kind before ':', like mode{segment}"),
        (not name, f"segment {segment!r} needs a name after ':', like {kind}:default"),
        (
            not kind.replace("_", "").isalnum(),
            f"kind {kind!r} takes letters, digits and _ only (segment {segment!r})",
        ),
    )
    return next((message for failed, message in checks if failed), None)


def ancestors(scope: str) -> list[str]:
    """``user:alice/project:solar`` -> ``['user:alice/project:solar', 'user:alice']``."""
    parts = scope.split("/")
    return ["/".join(parts[:i]) for i in range(len(parts), 0, -1)]


def within(scope: str, bound: str) -> bool:
    """Is ``scope`` equal to or below ``bound``? ``*`` bounds everything."""
    return bound in (ROOT_SCOPE, scope) or scope.startswith(bound + "/")


def _like_pattern(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "/%"


@dataclass(frozen=True)
class Visibility:
    """The set of scopes a request may see: exact scopes plus whole subtrees."""

    exact: tuple[str, ...] = ()
    subtrees: tuple[str, ...] = ()

    @staticmethod
    def of_scope(scope: str) -> Visibility:
        return Visibility(exact=tuple(ancestors(validate_scope(scope))))

    @staticmethod
    def of_scopes(scopes: Iterable[str]) -> Visibility:
        unique = tuple(dict.fromkeys(validate_scope(s) for s in scopes))
        if not unique:
            raise ScopeError("at least one scope is required")
        return Visibility(exact=unique)

    @staticmethod
    def of_subtree(root: str) -> Visibility:
        validate_scope(root)
        return Visibility(exact=(root,), subtrees=(root,))

    @property
    def is_empty(self) -> bool:
        return not self.exact and not self.subtrees

    def includes(self, scope: str) -> bool:
        return scope in self.exact or any(scope.startswith(root + "/") for root in self.subtrees)

    def sees(self, scopes: Iterable[str]) -> bool:
        """Is a record with these scopes visible?"""
        return any(self.includes(s) for s in scopes)

    def rank(self, scopes: Iterable[str]) -> int:
        """Lower is closer to the requester: the position of the best-matching scope."""
        positions = [self.exact.index(s) for s in scopes if s in self.exact]
        if positions:
            return min(positions)
        return len(self.exact) if self.sees(scopes) else len(self.exact) + 1

    def sql(self, column: str) -> tuple[str, list[str]]:
        """A predicate on ``column`` (a scope column) plus its parameters."""
        clauses: list[str] = []
        params: list[str] = []
        if self.exact:
            clauses.append(f"{column} IN ({','.join('?' for _ in self.exact)})")
            params.extend(self.exact)
        for root in self.subtrees:
            clauses.append(f"{column} LIKE ? ESCAPE '\\'")
            params.append(_like_pattern(root))
        if not clauses:
            return "0", []
        return "(" + " OR ".join(clauses) + ")", params


def authorized(scopes: Sequence[str], bounds: Sequence[str]) -> bool:
    """May a token bound to ``bounds`` write every scope in ``scopes``?"""
    return bool(scopes) and all(any(within(s, b) for b in bounds) for s in scopes)


def touches(scopes: Iterable[str], bounds: Sequence[str]) -> bool:
    """Does a record with ``scopes`` have at least one scope a token bound to ``bounds`` owns?"""
    return any(any(within(s, b) for b in bounds) for s in scopes)
