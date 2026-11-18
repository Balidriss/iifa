"""The parts of a source and who they came from (P6).

A turn is what the owner said, what the assistant answered, and content from
outside the conversation - tool results, web pages, documents. The model reads
one text, each part under its label so that a quote stays locatable; the
service keeps the parts' offsets, and only the offsets - never a label written
in the text, which a hostile page can imitate - tell whose words a quote is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

OWNER, ASSISTANT, EXTERNAL = "owner", "assistant", "external"  # provenance of a part (P6)
_EXTERNAL_LABELS = {"tool": "Tool", "web": "Web", "document": "Document", "other": "External"}
REF_CHARS = 40  # the short form of a part's ref (P6)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SEPARATORS = re.compile(r"[/\\]")


def short_ref(ref: str | None) -> str | None:
    """The only form of a part's ``ref`` the service keeps (P6): a URL gives its
    host (without ``www.``), a path its last segment (without query or
    fragment), anything else itself; whitespace becomes ``_``; at most
    ``REF_CHARS`` characters. Never a whole URL: a query string may carry a
    token. None when nothing is left."""
    text = (ref or "").strip()
    if not text:
        return None
    if _SCHEME.match(text):
        url = urlsplit(text)
        host = (url.hostname or "").lower()
        if url.scheme.lower() != "file" and host:
            text = host.removeprefix("www.") or host
        else:
            text = url.path
    if "/" in text or "\\" in text:
        path = re.split(r"[?#]", text, maxsplit=1)[0]
        segments = [segment for segment in _SEPARATORS.split(path) if segment.strip()]
        text = segments[-1] if segments else ""
    short = "_".join(text.split())[:REF_CHARS]
    return short or None


@dataclass(frozen=True)
class Part:
    """One part of a source: its text's offsets, who it came from (P6), and
    where the label the service wrote in front of it starts (None: no label)."""

    start: int
    end: int
    provenance: str
    label_start: int | None = None
    kind: str | None = None  # an external part's kind: tool | web | document | other
    # an external part's short ref (``short_ref``); the assistant's declared name
    ref: str | None = None
    # an external part the host vouched for: the owner gave it or asked for it
    vouched: bool = False


@dataclass(frozen=True)
class SourceText:
    """The text the model reads - each part under its label, so a quote stays
    locatable - and the offsets of the parts, which only the service reads."""

    text: str
    parts: tuple[Part, ...] = ()

    def part_of(self, span: tuple[int, int] | None) -> Part | None:
        """The part holding ``span``: inside it, or starting in the part's own
        label and ending inside it (models quote "User: ..." with the sentence;
        the label is the service's text, a page cannot write it). None when the
        span is missing, on a label only, or across two parts (nobody's words)."""
        if span is None:
            return None
        start, end = span
        for part in self.parts:
            begin = part.start if part.label_start is None else part.label_start
            if begin <= start and part.start < end <= part.end:
                return part
        return None

    def provenance_of(self, span: tuple[int, int] | None) -> str | None:
        """The provenance of the part holding ``span`` (``part_of``)."""
        part = self.part_of(span)
        return part.provenance if part is not None else None

    def texts(self, provenance: str) -> list[str]:
        return [self.text[p.start : p.end] for p in self.parts if p.provenance == provenance]


_Piece = tuple[str, str, str, str | None, str | None, bool]


def _assemble(pieces: list[_Piece]) -> SourceText:
    """``(label, text, provenance, kind, ref, vouched)`` pieces, empty ones left out."""
    chunks: list[str] = []
    parts: list[Part] = []
    position = 0
    for label, raw, provenance, kind, ref, vouched in pieces:
        text = raw.strip()
        if not text:
            continue
        if chunks:
            position += 2  # the blank line between two parts
        head = f"{label}: "
        parts.append(
            Part(
                position + len(head),
                position + len(head) + len(text),
                provenance,
                position,
                kind,
                ref,
                vouched,
            )
        )
        chunks.append(head + text)
        position += len(head) + len(text)
    return SourceText("\n\n".join(chunks), tuple(parts))


def turn_source(
    payload: dict[str, Any], owner: str | None = None, assistant: str | None = None
) -> SourceText:
    """The source of a turn: what the owner said (labelled with the owner's name
    when one is declared, SPEC 4.2, else "User"), what the assistant answered
    (labelled with its declared name, SPEC 4.6, else "Assistant"), and the
    external parts - tool summaries, pages, documents (P6)."""
    pieces: list[_Piece] = [
        (owner or "User", str(payload.get("user") or ""), OWNER, None, None, False),
        (
            assistant or "Assistant",
            str(payload.get("assistant") or ""),
            ASSISTANT,
            None,
            assistant,
            False,
        ),
    ]
    pieces.extend(
        ("Tool", str(s), EXTERNAL, "tool", None, False) for s in payload.get("tool_summaries") or ()
    )
    for item in payload.get("external") or ():
        if isinstance(item, dict):
            kind = str(item.get("kind") or "other")
            kind = kind if kind in _EXTERNAL_LABELS else "other"
            ref = item.get("ref")
            text = str(item.get("text") or "")
            pieces.append(
                (
                    _EXTERNAL_LABELS[kind],
                    text,
                    EXTERNAL,
                    kind,
                    short_ref(str(ref or "")),
                    item.get("vouched") is True,
                )
            )
    return _assemble(pieces)


def turn_text(
    payload: dict[str, Any], owner: str | None = None, assistant: str | None = None
) -> str:
    return turn_source(payload, owner, assistant).text


def precompact_source(
    payload: dict[str, Any], owner: str | None = None, assistant: str | None = None
) -> SourceText:
    """The messages a harness is about to drop; their role gives their
    provenance: ``user`` the owner's, ``assistant`` the assistant's, anything
    else external (P6)."""
    pieces: list[_Piece] = []
    for message in payload.get("messages") or ():
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "message").strip()
        content = str(message.get("content") or "")
        if role.lower() == "user":
            pieces.append((owner or "User", content, OWNER, None, None, False))
        elif role.lower() == "assistant":
            pieces.append((assistant or "Assistant", content, ASSISTANT, None, assistant, False))
        else:
            kind = "tool" if role.lower() == "tool" else None
            pieces.append((role.capitalize(), content, EXTERNAL, kind, None, False))
    return _assemble(pieces)


def precompact_text(
    payload: dict[str, Any], owner: str | None = None, assistant: str | None = None
) -> str:
    return precompact_source(payload, owner, assistant).text


__all__ = [
    "ASSISTANT",
    "EXTERNAL",
    "OWNER",
    "Part",
    "SourceText",
    "precompact_source",
    "precompact_text",
    "short_ref",
    "turn_source",
    "turn_text",
]
