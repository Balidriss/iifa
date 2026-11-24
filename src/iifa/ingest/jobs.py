"""Job handlers, by kind.

The lifecycle jobs do the model-driven work of tier 2, each call going
through the job's ``ModelSession`` so that P11 holds whatever the job:

* ``turn`` and ``precompact`` extract nodes, facts and edges from the text they
  carry (``iifa.ingest.extract``), in the language of that text (P12), with
  provenance (P6) and contradictions superseded (P7); at tier 0 they run the
  deterministic part only (known names mentioned -> co-occurrence edges);
* ``session_end`` closes the session's state (``iifa.ingest.consolidate``:
  its work in progress consolidated, the rest expired), refreshes the
  summaries of the nodes the session touched (``iifa.ingest.summaries``) and
  applies edge decay in the session's scopes, with the store's clock;
* ``session_close`` closes the state of a session nobody ended, once it has
  been idle long enough (queued by the worker), and nothing else.

``model_noop`` makes real calls to the configured endpoint and discards the
answers, so the fake model server's call log can count them in P11 tests.
"""

from __future__ import annotations

from typing import Any

from iifa import log
from iifa.ingest.consolidate import close_session
from iifa.ingest.embed import EMBED, embed
from iifa.ingest.extract import extract_text, write_view
from iifa.ingest.parts import (
    ASSISTANT,
    EXTERNAL,
    OWNER,
    Part,
    SourceText,
    precompact_source,
    precompact_text,
    turn_source,
    turn_text,
)
from iifa.ingest.queue import Job
from iifa.ingest.sessions import SESSION_CLOSE, release_closing
from iifa.ingest.sessions import load as load_session
from iifa.ingest.summaries import refresh_summary
from iifa.ingest.worker import Handler, JobContext
from iifa.llm import CHAT, EMBEDDINGS

TURN = "turn"
PRECOMPACT = "precompact"
SESSION_END = "session_end"
NOOP = "noop"
MODEL_NOOP = "model_noop"

MAX_NOOP_CALLS = 100


def noop(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    """Nothing: the queue's own plumbing is what this exercises."""
    return None


def model_noop(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    """``payload.calls`` (default 1) calls to ``payload.endpoint`` (default
    ``chat``), answers discarded. Each call goes through the budget and the gate,
    so a run can stop in the middle of one job and resume it later."""
    endpoint = str(job.payload.get("endpoint", CHAT))
    calls = min(MAX_NOOP_CALLS, max(1, int(job.payload.get("calls", 1))))
    if endpoint not in (CHAT, EMBEDDINGS):
        raise ValueError(f"unknown endpoint {endpoint!r}")
    for index in range(calls):
        if endpoint == CHAT:
            ctx.models.chat(
                [{"role": "user", "content": f"ping {index + 1}/{calls}"}],
                max_tokens=1,
                accept_truncated=True,  # a ping: nothing of the answer is used
            )
        else:
            ctx.models.embed([f"ping {index + 1}/{calls}"])
    return {"calls": calls}


def _owner_name(ctx: JobContext, scopes: list[str]) -> str | None:
    owner = ctx.config.owner(scopes)
    return owner.name if owner is not None else None


def turn(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    """A recorded turn: one source, extracted."""
    scopes = list(job.scopes)
    source = turn_source(job.payload, _owner_name(ctx, scopes), ctx.config.assistant(scopes))
    if not source.text:
        return {"skipped": "empty"}
    external = job.payload.get("external_id")
    return extract_text(
        ctx,
        source.text,
        parts=source.parts,
        scopes=job.scopes,
        kind=TURN,
        session_id=job.session_id,
        reference=str(job.payload.get("occurred_at") or "") or None,
        external_id=str(external) if external else None,
    )


def precompact(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    """The messages a harness is about to drop: one source, extracted in chunks."""
    scopes = list(job.scopes)
    source = precompact_source(job.payload, _owner_name(ctx, scopes), ctx.config.assistant(scopes))
    if not source.text:
        return {"skipped": "empty"}
    trigger = job.payload.get("trigger")
    return extract_text(
        ctx,
        source.text,
        parts=source.parts,
        scopes=job.scopes,
        kind=PRECOMPACT,
        session_id=job.session_id,
        reference=str(trigger) if trigger else None,
    )


def session_end(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    """The session's flush: summaries of the nodes it touched are rewritten from
    their facts (tier 2), then unconfirmed edges in its scopes decay."""
    view = write_view(list(job.scopes))
    config = ctx.config
    closed = close_session(ctx, job.session_id, list(job.scopes), reason="end")
    summaries = {"refreshed": 0, "language_mismatch": 0, "truncated": 0, "skipped": 0, "calls": 0}
    if job.session_id and ctx.models.has_chat:
        touched = ctx.store.nodes_touched_by_session(job.session_id, view)
        for node_id in touched[: config.summary_nodes_per_session]:
            outcome = refresh_summary(ctx, node_id, view)
            summaries["calls"] += int(outcome.get("calls", 0))
            if outcome["outcome"] == "written":
                summaries["refreshed"] += 1
            elif outcome["outcome"] == "language_mismatch":
                summaries["refreshed"] += 1
                summaries["language_mismatch"] += 1
            elif outcome["outcome"] == "truncated":
                summaries["truncated"] += 1
            else:
                summaries["skipped"] += 1
    decay = ctx.store.decay_edges(
        half_life_days=config.edge_half_life_days, prune_below=config.edge_prune_below, view=view
    )
    log.event(
        "session_flushed",
        summaries=summaries["refreshed"],
        mismatches=summaries["language_mismatch"],
        decayed=decay["decayed"],
        pruned=decay["pruned"],
    )
    return {"session": closed, "summaries": summaries, "edges": decay}


def session_close(job: Job, ctx: JobContext) -> dict[str, Any] | None:
    """A session nobody ended, idle long enough: its state closed."""
    reason = str(job.payload.get("reason") or "idle")
    seen = job.payload.get("seen")
    if seen is not None:
        state = load_session(ctx.store, job.session_id, list(job.scopes))
        if state.id is not None and state.last_seen_at != seen:
            # a turn of the session came after the close was queued: still active
            release_closing(ctx.store, state.id)
            return {"session": {"reason": reason, "session": "active"}}
    return {"session": close_session(ctx, job.session_id, list(job.scopes), reason=reason)}


HANDLERS: dict[str, Handler] = {
    NOOP: noop,
    MODEL_NOOP: model_noop,
    TURN: turn,
    PRECOMPACT: precompact,
    SESSION_END: session_end,
    SESSION_CLOSE: session_close,
    EMBED: embed,
}

__all__ = [
    "ASSISTANT",
    "EXTERNAL",
    "HANDLERS",
    "MODEL_NOOP",
    "NOOP",
    "OWNER",
    "PRECOMPACT",
    "SESSION_CLOSE",
    "SESSION_END",
    "TURN",
    "Part",
    "SourceText",
    "model_noop",
    "noop",
    "precompact",
    "precompact_source",
    "precompact_text",
    "session_close",
    "session_end",
    "turn",
    "turn_source",
    "turn_text",
]
