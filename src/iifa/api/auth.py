"""Bearer tokens, permissions and scope authorization.

Tokens are stored as SHA-256 digests and compared in constant time, every
stored digest being compared on every request so timing reveals nothing about
which token, if any, matched. Every route declares the permission it needs in
``ROUTE_PERMISSIONS``; the permission matrix test walks that table.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request
from fastapi.security.utils import get_authorization_scheme_param

from iifa.config import PERMISSIONS, TokenConfig, hash_token
from iifa.scopes import Visibility, authorized, touches

ANY = "any"  # a valid token with any permission
ROUTE_PERMISSIONS: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class Principal:
    name: str
    scopes: tuple[str, ...]
    permissions: frozenset[str]

    def can(self, permission: str) -> bool:
        return permission == ANY or permission in self.permissions

    def may_write(self, scopes: Sequence[str]) -> bool:
        """May this token create records in (or read) every one of ``scopes``?"""
        return authorized(scopes, self.scopes)

    def may_touch(self, scopes: Sequence[str]) -> bool:
        """May this token archive, restore, forget or patch a record with ``scopes``?"""
        return touches(scopes, self.scopes)


class TokenRegistry:
    def __init__(self, tokens: Sequence[TokenConfig]) -> None:
        self._tokens = tuple(tokens)

    def authenticate(self, presented: str) -> Principal | None:
        digest = hash_token(presented)
        match: TokenConfig | None = None
        for token in self._tokens:  # no early exit: the loop runs the same for every input
            if hmac.compare_digest(digest, token.sha256) and match is None:
                match = token
        if match is None:
            return None
        return Principal(match.name, match.scopes, frozenset(match.permissions))


def _registry(request: Request) -> TokenRegistry:
    registry: TokenRegistry = request.app.state.registry
    return registry


def require(permission: str) -> Callable[[Request], Principal]:
    """A dependency that authenticates the bearer token and checks ``permission``."""
    if permission != ANY and permission not in PERMISSIONS:
        raise ValueError(f"unknown permission {permission!r}")

    def dependency(request: Request) -> Principal:
        scheme, credentials = get_authorization_scheme_param(request.headers.get("Authorization"))
        if scheme.lower() != "bearer" or not credentials:
            raise HTTPException(
                401, "bearer token required", headers={"WWW-Authenticate": "Bearer"}
            )
        principal = _registry(request).authenticate(credentials)
        if principal is None:
            raise HTTPException(401, "invalid token", headers={"WWW-Authenticate": "Bearer"})
        if not principal.can(permission):
            raise HTTPException(403, f"token lacks the '{permission}' permission")
        return principal

    return dependency


def declare(method: str, path: str, permission: str) -> Any:
    """Register the permission a route needs and return its ``Depends``."""
    ROUTE_PERMISSIONS[(method.upper(), path)] = permission
    return Depends(require(permission))


def resolve_view(
    principal: Principal,
    *,
    scope: str | None,
    scopes: Sequence[str] | None,
    subtree: str | None,
) -> Visibility:
    """Turn the three read modes into a ``Visibility`` the token is allowed to use."""
    modes = [m for m in (scope, scopes, subtree) if m]
    if len(modes) != 1:
        raise HTTPException(422, "give exactly one of scope, scopes or subtree")
    if subtree:
        if not principal.can("admin"):
            raise HTTPException(403, "reading a whole subtree needs the 'admin' permission")
        if not principal.may_write([subtree]):
            raise HTTPException(403, "subtree outside the token's scopes")
        return Visibility.of_subtree(subtree)
    if scopes:
        if not principal.may_write(list(scopes)):
            raise HTTPException(403, "scope outside the token's scopes")
        return Visibility.of_scopes(scopes)
    assert scope is not None
    if not principal.may_write([scope]):
        raise HTTPException(403, "scope outside the token's scopes")
    return Visibility.of_scope(scope)


def require_write_scopes(principal: Principal, scopes: Sequence[str]) -> list[str]:
    unique = list(dict.fromkeys(scopes))
    if not principal.may_write(unique):
        raise HTTPException(403, "scope outside the token's scopes")
    return unique


def require_touch(principal: Principal, record_scopes: Sequence[str]) -> None:
    if not principal.may_touch(record_scopes):
        raise HTTPException(403, "record outside the token's scopes")


Auth = Annotated[Principal, Depends(require(ANY))]
