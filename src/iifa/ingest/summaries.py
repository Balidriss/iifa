"""Summary maintenance (SPEC 6, tier 2): one line per node, rewritten from its
live facts in the language of those facts (P12). A summary the model wrote in
another language is rejected and asked for again once; the second answer is
stored whatever its language, flagged ``language_mismatch`` on the node rather
than silently. A summary cut by its token cap is never stored (SPEC 7.4b): it
is asked for again once with twice the cap, then the summary is left as it was.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from iifa import log
from iifa.ingest.prompts import summary_messages
from iifa.language import UNDETERMINED, detect, writing_mismatch
from iifa.llm import TRUNCATED, ModelError
from iifa.scopes import Visibility
from iifa.store.models import Fact

if TYPE_CHECKING:
    from iifa.ingest.worker import JobContext

LANGUAGE_MISMATCH = "language_mismatch"
SUMMARY_CHARS = 300
SUMMARY_MAX_TOKENS = 160


def facts_language(facts: list[Fact], forced: str | None) -> str:
    """The language a node's summary must be written in: forced by the scope,
    else the most common detected language of its facts (ties to the newest)."""
    if forced:
        return forced
    counts: Counter[str] = Counter()
    for fact in facts:
        language = fact.language or detect(fact.statement)
        if language != UNDETERMINED:
            counts[language] += 1
    if not counts:
        joined = detect(" ".join(f.statement for f in facts))
        return joined
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return ordered[0][0]


def clean_summary(content: str) -> str:
    """The first non-empty line of the answer, unquoted, bounded."""
    for line in content.splitlines():
        text = " ".join(line.split()).strip().strip('"“”«» ').strip()
        if text:
            return text[:SUMMARY_CHARS]
    return ""


def refresh_summary(ctx: JobContext, node_id: str, view: Visibility) -> dict[str, Any]:
    """Rewrite one node's summary. Returns what happened, as counts and words."""
    store, config = ctx.store, ctx.config
    node = store.get_node(node_id)
    facts = store.facts([node_id], view, limit=config.summary_facts)
    if not facts:
        return {"node": node_id, "outcome": "no_facts"}
    forced = config.forced_language(node.scopes)
    language = facts_language(facts, forced)
    target = language if language != UNDETERMINED else "the language of the facts"
    statements = [f.statement for f in facts]

    calls, retried = 0, False

    def ask(note: str | None) -> str:
        nonlocal calls
        messages = summary_messages(
            node.name, node.kind, statements, language=target, retry_note=note
        )
        calls += 1
        try:
            result = ctx.models.chat(messages, max_tokens=SUMMARY_MAX_TOKENS)
        except ModelError as exc:
            if exc.reason != TRUNCATED:
                raise
            calls += 1  # asked again once with twice the cap; still cut, it goes up
            result = ctx.models.chat(messages, max_tokens=2 * SUMMARY_MAX_TOKENS)
        return clean_summary(result.content)

    try:
        summary = ask(None)
        if not summary:
            retried = True
            summary = ask("Answer with the summary sentence itself.")
            if not summary:
                return {"node": node_id, "outcome": "empty", "calls": calls}
        if writing_mismatch(summary, language):
            retried = True
            summary = ask(
                f"Your previous answer was not in the required language. Write it in the "
                f"language of the facts ({language}). Do not translate."
            )
            if not summary:
                return {"node": node_id, "outcome": "empty", "calls": calls}
    except ModelError as exc:
        if exc.reason != TRUNCATED:
            raise
        log.event("summary", node=node_id, calls=calls, truncated=True)
        return {"node": node_id, "outcome": "truncated", "calls": calls}
    written = detect(summary)
    flags = set(node.flags) - {LANGUAGE_MISMATCH}
    flagged = writing_mismatch(summary, language)
    if flagged:
        flags.add(LANGUAGE_MISMATCH)
    store.update_node(
        node_id,
        summary=summary,
        language=None if written == UNDETERMINED else written,
        flags=flags,
    )
    log.event(
        "summary",
        node=node_id,
        calls=calls,
        retried=retried,
        language_mismatch=flagged,
    )
    return {
        "node": node_id,
        "outcome": "language_mismatch" if flagged else "written",
        "calls": calls,
        "retried": retried,
    }


__all__ = ["LANGUAGE_MISMATCH", "clean_summary", "facts_language", "refresh_summary"]
