"""The parts of a source and who they came from (P6).

A turn is what the owner said, what the assistant answered, and content from
outside the conversation - tool results, web pages, documents. The model reads
one text, each part under its label so that a quote stays locatable; the
service keeps the parts' offsets, and only the offsets - never a label written
in the text, which a hostile page can imitate - tell whose words a quote is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OWNER, ASSISTANT, EXTERNAL = "owner", "assistant", "external"  # provenance of a part (P6)
_EXTERNAL_LABELS = {"tool": "Tool", "web": "Web", "document": "Document", "other": "External"}


@dataclass(frozen=True)
class Part:
    """One part of a source: its text's offsets, who it came from (P6), and
    where the label the service wrote in front of it starts (None: no label)."""

    start: int
    end: int
    provenance: str
    label_start: int | None = None


@dataclass(frozen=True)
class SourceText:
    """The text the model reads - each part under its label, so a quote stays
    locatable - and the offsets of the parts, which only the service reads."""

    text: str
    parts: tuple[Part, ...] = ()

    def provenance_of(self, span: tuple[int, int] | None) -> str | None:
        """The provenance of the part holding ``span``: inside it, or starting in
        the part's own label and ending inside it (models quote "User: ..." with
        the sentence; the label is the service's text, a page cannot write it).
        None when the span is missing, on a label only, or across two parts
        (nobody's words)."""
        if span is None:
            return None
        start, end = span
        for part in self.parts:
            begin = part.start if part.label_start is None else part.label_start
            if begin <= start and part.start < end <= part.end:
                return part.provenance
        return None

    def texts(self, provenance: str) -> list[str]:
        return [self.text[p.start : p.end] for p in self.parts if p.provenance == provenance]


def _assemble(pieces: list[tuple[str, str, str]]) -> SourceText:
    """``(label, text, provenance)`` pieces, empty ones left out."""
    chunks: list[str] = []
    parts: list[Part] = []
    position = 0
    for label, raw, provenance in pieces:
        text = raw.strip()
        if not text:
            continue
        if chunks:
            position += 2  # the blank line between two parts
        head = f"{label}: "
        parts.append(
            Part(position + len(head), position + len(head) + len(text), provenance, position)
        )
        chunks.append(head + text)
        position += len(head) + len(text)
    return SourceText("\n\n".join(chunks), tuple(parts))


def turn_source(payload: dict[str, Any], owner: str | None = None) -> SourceText:
    """The source of a turn: what the owner said (labelled with the owner's name
    when one is declared, SPEC 4.2, else "User"), what the assistant answered,
    and the external parts - tool summaries, pages, documents (P6)."""
    pieces = [
        (owner or "User", str(payload.get("user") or ""), OWNER),
        ("Assistant", str(payload.get("assistant") or ""), ASSISTANT),
    ]
    pieces.extend(("Tool", str(s), EXTERNAL) for s in payload.get("tool_summaries") or ())
    for item in payload.get("external") or ():
        if isinstance(item, dict):
            label = _EXTERNAL_LABELS.get(str(item.get("kind") or "other"), "External")
            pieces.append((label, str(item.get("text") or ""), EXTERNAL))
    return _assemble(pieces)


def turn_text(payload: dict[str, Any], owner: str | None = None) -> str:
    return turn_source(payload, owner).text


def precompact_source(payload: dict[str, Any], owner: str | None = None) -> SourceText:
    """The messages a harness is about to drop; their role gives their
    provenance: ``user`` the owner's, ``assistant`` the assistant's, anything
    else external (P6)."""
    pieces: list[tuple[str, str, str]] = []
    for message in payload.get("messages") or ():
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "message").strip()
        content = str(message.get("content") or "")
        if role.lower() == "user":
            pieces.append((owner or "User", content, OWNER))
        elif role.lower() == "assistant":
            pieces.append(("Assistant", content, ASSISTANT))
        else:
            pieces.append((role.capitalize(), content, EXTERNAL))
    return _assemble(pieces)


def precompact_text(payload: dict[str, Any], owner: str | None = None) -> str:
    return precompact_source(payload, owner).text


__all__ = [
    "ASSISTANT",
    "EXTERNAL",
    "OWNER",
    "Part",
    "SourceText",
    "precompact_source",
    "precompact_text",
    "turn_source",
    "turn_text",
]
