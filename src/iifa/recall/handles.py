"""Handles (P2): what a recall returns instead of content."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from iifa.ingest.checks import unspaced_chars
from iifa.store.models import Fact, Node
from iifa.tokens import Tokenizer

LINE_CHARS = 140
TITLE_CHARS = 80
SUMMARY_LINE_TOKENS = 24  # SPEC 5.6: the accepted price of a summary shown
ELLIPSIS = "…"


def one_line(text: str, limit: int = LINE_CHARS) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1].rstrip()
    return cut + "…"


def summary_line(summary: str, tokenizer: Tokenizer, limit: int = SUMMARY_LINE_TOKENS) -> str:
    """SPEC 5.6: the summary as a handle or a profile shows it - whole when it
    fits ``limit`` tokens, else its longest start that fits with "…", cut at a
    word boundary (a character boundary in a script written without spaces)."""
    flat = " ".join(summary.split())
    if tokenizer.count(flat) <= limit:
        return flat

    def fits(n: int) -> bool:
        return tokenizer.count(flat[:n].rstrip() + ELLIPSIS) <= limit

    # the longest start that fits, by bisection (``low`` only ever holds a start
    # that fits), then back to the last whole word when that fits too
    low, high = 0, len(flat)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    cut = flat[:low]
    space = cut.rfind(" ")
    cut_in_a_word = low < len(flat) and flat[low] != " "
    if cut_in_a_word and space > 0 and not unspaced_chars(cut[space:]) and fits(space):
        cut = cut[:space]  # a spaced word cut in two: drop it
    return cut.rstrip() + ELLIPSIS


def plain_line(node: Node, fact_count: int) -> str:
    """The line of a node without a summary."""
    return f"{node.kind}, {fact_count} live facts"


def age_days(now: datetime, stamp: str | None) -> int | None:
    if not stamp:
        return None
    try:
        then = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return max(0, (now - then).days)


def fact_expansion(
    fact: Fact, node: Node | None, level: int, excerpt: str | None
) -> dict[str, Any]:
    """Level 1: the fact. Level 2: plus the source excerpt it came from.

    ``node`` is None for a free-standing fact and for a fact whose node the
    reader may not see: an invisible subject is never named or identified (P9).
    """
    out: dict[str, Any] = {
        "id": fact.id,
        "kind": "fact",
        "node_id": node.id if node else None,
        "node_name": node.name if node else None,
        "statement": fact.statement,
        "attribute": fact.attribute,
        "value": fact.value,
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "origin": fact.origin,
        "confidence": fact.confidence,
        "pinned": fact.pinned,
        "flags": list(fact.flags),
        "created_at": fact.created_at,
    }
    if fact.conflicts_with is not None:
        out["conflicts_with"] = fact.conflicts_with
    if fact.language is not None:
        out["language"] = fact.language
    if level >= 2:
        out["source_id"] = fact.source_id
        out["excerpt"] = excerpt
    return out


NODE_EXPANSION_FACTS = 20


def node_expansion(node: Node, facts: list[Fact], level: int, fact_count: int) -> dict[str, Any]:
    """A node with its best ``NODE_EXPANSION_FACTS`` live facts (pinned first,
    then most recent) and the total count; the rest is one ``expand`` away."""
    out: dict[str, Any] = {
        "id": node.id,
        "kind": "node",
        "name": node.name,
        "node_kind": node.kind,
        "aliases": list(node.aliases),
        "summary": node.summary,
        "created_at": node.created_at,
        "fact_count": fact_count,
        "facts": [
            {"id": f.id, "statement": f.statement, "valid_from": f.valid_from, "origin": f.origin}
            for f in facts
        ],
    }
    if level >= 2:
        out["scopes"] = list(node.scopes)
    return out


def fact_handle(
    fact: Fact, node: Node | None, why: str, now: datetime, expand_tokens: int
) -> dict[str, Any]:
    title = f"{node.name} — {fact.statement}" if node else fact.statement
    return {
        "id": fact.id,
        "kind": "fact",
        "title": one_line(title, TITLE_CHARS),
        "line": one_line(fact.statement),
        "why": why,
        "age_days": age_days(now, fact.valid_from or fact.created_at),
        "expand_tokens": expand_tokens,
        "flags": sorted({*fact.flags, *(["pinned"] if fact.pinned else [])}),
    }


def node_handle(
    node: Node,
    why: str,
    now: datetime,
    expand_tokens: int,
    fact_count: int,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    summary = summary_line(node.summary, tokenizer)
    return {
        "id": node.id,
        "kind": "node",
        "title": one_line(node.name, TITLE_CHARS),
        "line": summary or plain_line(node, fact_count),
        "why": why,
        "age_days": age_days(now, node.updated_at),
        "expand_tokens": expand_tokens,
        "flags": [],
    }


def serialized(payload: Any) -> str:
    """The exact text an agent will receive for ``payload``: what the budget counts."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
