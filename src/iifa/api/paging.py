"""Cursors of the list routes.

Positions are keyset (the last delivered item's unique sort key), never an
offset; ``iifa.recall.paging`` cuts the pages. A cursor is opaque to the client:
a small JSON position, signed with HMAC-SHA256 under a key kept in the encrypted
store, and bound to a digest of the request's context (the route, the view, the
sort and every filter). A cursor that was altered, or that is replayed with
another view, filter or sort, is refused with 400 before anything is read, so a
cursor obtained on one scope never pages through another.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import HTTPException

from iifa.scopes import Visibility

STATE_KEY = "cursor_key"
VERSION = 1
MAX_CURSOR_CHARS = 2048
_DOMAIN = b"iifa-cursor\x00"
_SIGNATURE_BYTES = 16


class CursorError(ValueError):
    """A cursor that is malformed, altered, or not made for this request."""


def view_context(view: Visibility) -> dict[str, list[str]]:
    """The view as a cursor binds it: its exact scopes and its subtrees, as sets
    (the order a request lists them in changes no list)."""
    return {"exact": sorted(view.exact), "subtrees": sorted(view.subtrees)}


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class Cursors:
    """Signs and checks cursors with one secret key."""

    def __init__(self, key: bytes) -> None:
        if len(key) < 16:
            raise ValueError("a cursor key needs at least 16 bytes")
        self._key = key

    @classmethod
    def for_store(cls, store: Any) -> Cursors:
        """The key kept in the store's service state (created on first use), so a
        cursor outlives a restart and the key lives under the store's encryption."""
        with store.lock:
            stored = store.get_state(STATE_KEY)
            if stored is None:
                stored = os.urandom(32).hex()
                store.set_state(STATE_KEY, stored)
        return cls(bytes.fromhex(stored))

    @staticmethod
    def _digest(context: Mapping[str, Any]) -> str:
        text = json.dumps(context, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _sign(self, payload: bytes) -> bytes:
        return hmac.new(self._key, _DOMAIN + payload, hashlib.sha256).digest()[:_SIGNATURE_BYTES]

    def encode(self, context: Mapping[str, Any], position: Sequence[Any]) -> str:
        body = {"v": VERSION, "c": self._digest(context), "p": list(position)}
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return f"{_b64(payload)}.{_b64(self._sign(payload))}"

    def decode(self, cursor: str | None, context: Mapping[str, Any]) -> list[Any] | None:
        """The position a cursor carries, ``None`` without one (the first page).
        Raises ``CursorError`` for anything this key did not sign for ``context``."""
        if not cursor:
            return None
        if len(cursor) > MAX_CURSOR_CHARS or cursor.count(".") != 1:
            raise CursorError("invalid cursor")
        head, tail = cursor.split(".")
        try:
            payload, signature = _unb64(head), _unb64(tail)
        except (binascii.Error, ValueError) as exc:
            raise CursorError("invalid cursor") from exc
        if not hmac.compare_digest(signature, self._sign(payload)):
            raise CursorError("invalid cursor")
        body = json.loads(payload.decode("utf-8"))  # signed by this key: our own JSON
        if body.get("c") != self._digest(context) or body.get("v") != VERSION:
            raise CursorError("the cursor belongs to another view, filter or sort")
        position: list[Any] = body["p"]
        return position


def read_cursor(
    cursors: Cursors,
    cursor: str | None,
    context: Mapping[str, Any],
    *types: type,
) -> tuple[Any, ...] | None:
    """The keyset position of ``cursor`` for ``context``, checked against the
    shape the route expects (one type per field); 400 for anything else."""
    try:
        position = cursors.decode(cursor, context)
    except CursorError as exc:
        raise HTTPException(400, str(exc)) from exc
    if position is None:
        return None
    if len(position) != len(types) or not all(
        type(value) is kind for value, kind in zip(position, types, strict=True)
    ):
        raise HTTPException(400, "invalid cursor")
    return tuple(position)


__all__ = ["CursorError", "Cursors", "read_cursor", "view_context"]
