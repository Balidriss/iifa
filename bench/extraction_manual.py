"""Run tier-2 extraction on the multilingual synthetic corpus against a real
OpenAI-compatible chat endpoint, and print what it wrote: the documented manual
run against a real local model (the test suite always fakes it).

    python bench/extraction_manual.py --chat-url http://127.0.0.1:11434/v1 --model llama3.2
    python bench/extraction_manual.py --chat-url http://127.0.0.1:8080/v1 --model qwen2.5:7b \\
        --samples en,fr,ru,ar,zh,ja,mixed --corpus 12 --verbose
    python bench/extraction_manual.py --chat-url http://127.0.0.1:11434/v1 --model qwen3.5:9b \\
        --no-json-mode --extra reasoning_effort=none

What it does, in a throwaway encrypted store in a temporary directory:

1. records one turn per fixture sample (``tests/fixtures/corpus.py``: English,
   French, Russian, Arabic, Chinese, Japanese, a mixed one) plus ``--corpus N``
   synthetic statements grouped three to a turn, all under one session;
2. runs the queue **on demand** through the real worker (P11: the gate, the
   budget and the retries are the service's own), one job at a time, printing
   each job's result as soon as it is done, then a ``session/end`` job so that
   summaries are written and edges decay;
3. prints every node with its summary and language, every fact with its
   language check (expected from the source sentence, detected from the
   statement, flagged or not), every edge; and the totals: model calls, tokens,
   wall time.

Ctrl+C stops the run: what was written so far is reported, and the exit code is
130. Thinking models: ``--no-json-mode`` (some model/server pairs never finish
in JSON mode) and ``--extra reasoning_effort=none`` (Ollama's OpenAI API
ignores ``think: false``); ``--max-tokens`` bounds every answer (SPEC 7.4b).

Nothing is kept: the store is deleted when the script ends. Content is printed
to the terminal on purpose (this is a manual run for the operator's eyes); the
service's own logs still carry ids and counts only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fixtures.corpus import SAMPLES, statements

from iifa.ingest.embed import EMBED
from iifa.ingest.extract import PipelineConfig, write_view
from iifa.ingest.jobs import HANDLERS, SESSION_END, TURN
from iifa.ingest.queue import Queue
from iifa.ingest.worker import DONE, FAILED, ON_DEMAND, Budget, Gate, Worker
from iifa.language import detect, language_of_span
from iifa.llm import CHAT, EMBEDDINGS, RESERVED_REQUEST_KEYS, Endpoint, ModelClient
from iifa.scopes import Visibility
from iifa.store.repo import Store

SCOPE = "user:bench"
SESSION = "bench-session"
STORE_KEY = "bench-throwaway-key"
INTERRUPTED = 130
FINAL = (DONE, FAILED)  # a "retry" or "released" attempt runs again later


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--chat-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1",
    )
    parser.add_argument("--model", required=True, help="model name, named explicitly on every call")
    parser.add_argument(
        "--api-key", default="", help="bearer key for the endpoint, if it needs one"
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="seconds to wait for one answer (default 180)"
    )
    parser.add_argument(
        "--samples",
        default="all",
        help="comma-separated fixture sample names (default: all of " + ",".join(SAMPLES) + ")",
    )
    parser.add_argument(
        "--corpus",
        type=int,
        default=6,
        help="synthetic statements to add, three per turn (default 6)",
    )
    parser.add_argument(
        "--max-calls", type=int, default=None, help="stop the run after this many model calls"
    )
    parser.add_argument(
        "--chunk-chars", type=int, default=12000, help="characters per extraction chunk"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="the bound of every extraction answer's max_tokens (default 4096)",
    )
    parser.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask for response_format json_object (default on)",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an extra request parameter (VALUE read as JSON when it parses), repeatable",
    )
    parser.add_argument(
        "--embeddings-url",
        default=None,
        help="tier 3: an OpenAI-compatible embeddings endpoint (turns then run one at a time,"
        " each followed by its embed job, as in a conversation)",
    )
    parser.add_argument("--embeddings-model", default=None, help="the embedding model (tier 3)")
    parser.add_argument(
        "--no-session-end", action="store_true", help="skip the session end (summaries, decay)"
    )
    parser.add_argument("--verbose", action="store_true", help="also print every job's full result")
    args = parser.parse_args(argv)
    if bool(args.embeddings_url) != bool(args.embeddings_model):
        parser.error("--embeddings-url and --embeddings-model go together")
    try:
        args.extra = extras(args.extra)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def extras(pairs: Sequence[str]) -> dict[str, Any]:
    """``["reasoning_effort=none", 'options={"num_ctx": 8192}']`` -> a dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--extra {pair!r}: expected KEY=VALUE")
        if key in RESERVED_REQUEST_KEYS:
            raise ValueError(f"--extra {key}: iifa sets it itself")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def turns(args: argparse.Namespace) -> list[tuple[str, str]]:
    names = (
        list(SAMPLES)
        if args.samples == "all"
        else [s.strip() for s in args.samples.split(",") if s.strip()]
    )
    out: list[tuple[str, str]] = []
    for name in names:
        if name not in SAMPLES:
            raise SystemExit(f"unknown sample {name!r}; known: {', '.join(SAMPLES)}")
        out.append((f"sample:{name}", SAMPLES[name]))
    extra = statements(args.corpus)
    for start in range(0, len(extra), 3):
        out.append((f"corpus:{start // 3}", " ".join(extra[start : start + 3])))
    return out


def language_check(store: Store, fact: Any) -> tuple[str, str, str]:
    """``(expected, detected, verdict)`` for one fact, as the pipeline judged it."""
    detected = fact.language or detect(fact.statement)
    expected = "?"
    if fact.source_id is not None:
        source = store.get_source(fact.source_id)
        if source.text is not None:
            if fact.excerpt_start is not None and fact.excerpt_end is not None:
                expected = language_of_span(source.text, fact.excerpt_start, fact.excerpt_end)
            else:
                expected = source.language or detect(source.text)
    if "language_mismatch" in fact.flags:
        verdict = "FLAGGED"
    elif "und" in (expected, detected) or expected == "?":
        verdict = "undetermined"
    else:
        verdict = "ok"
    return expected, detected, verdict


def job_line(job: dict[str, Any]) -> str:
    """One job's outcome and counts, on one line."""
    result = job["result"] or {}
    line = f"{job['kind']:<12} {job['outcome']:<9}"
    if job["kind"] == TURN and result:
        facts, nodes = result["facts"], result["nodes"]
        line += (
            f" languages={','.join(result.get('languages', [])) or '-'}"
            f" nodes+{nodes['created']}/={nodes['matched']}"
            f" facts+{facts['written']} dup={facts['duplicate']} sup={facts['superseded']}"
            f" conflict={facts['conflict']} lang_mismatch={facts['language_mismatch']}"
            f" incomplete={facts['incomplete']} cross_script={facts['cross_script']}"
            f" mixed_script={facts['mixed_script']}"
            f" aliases+{result['aliases']['kept']}/?{result['aliases']['proposed']}"
            f"/!{result['aliases']['invented']}"
            f" edges+{result['edges']['proposed']} adjudicated={result['mentions_adjudicated']}"
            f" calls={result['calls']}"
            f" retries={result['retries']} truncated={result['truncated']}"
            f" rejected={sum(result['rejected'].values())}"
        )
    elif job["kind"] == SESSION_END and result:
        line += f" summaries={result['summaries']} edges={result['edges']}"
    elif job["kind"] == EMBED and result:
        line += (
            f" embedded={result['embedded']} merge_proposals={result['merge_proposals']}"
            f" similar_edges={result['similar_edges']}"
        )
    return line


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="iifa-extraction-"))
    started = time.monotonic()
    try:
        return run(args, workdir, started)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run(args: argparse.Namespace, workdir: Path, started: float) -> int:
    gate = Gate({CHAT: 1, "embeddings": 1})
    endpoint = Endpoint(
        CHAT, args.chat_url, args.model, args.api_key, timeout=args.timeout, extra=args.extra
    )
    embeddings = (
        Endpoint(EMBEDDINGS, args.embeddings_url, args.embeddings_model, timeout=args.timeout)
        if args.embeddings_url
        else None
    )
    models = ModelClient(endpoint, embeddings, gate=gate)
    store = Store(workdir / "bench.db", STORE_KEY, checkpoint_interval=0)
    worker = Worker(
        store,
        Queue(store),
        mode=ON_DEMAND,
        gate=gate,
        models=models,
        handlers=HANDLERS,
        pipeline=PipelineConfig(
            chunk_chars=args.chunk_chars,
            answer_max_tokens=args.max_tokens,
            json_mode=args.json_mode,
        ),
    )
    view = write_view([SCOPE])
    budget = Budget(max_calls=args.max_calls)
    jobs: list[dict[str, Any]] = []
    try:
        print(
            f"endpoint {args.chat_url} model {args.model}"
            f" json_mode={'on' if args.json_mode else 'off'} max_tokens<={args.max_tokens}"
            f" extra={json.dumps(args.extra, ensure_ascii=False) if args.extra else '-'}"
        )
        if embeddings is not None:
            thresholds = worker.pipeline.thresholds(embeddings.model)
            print(
                f"embeddings {embeddings.base_url} model {embeddings.model}"
                f" calibration {thresholds.source if thresholds else 'uncalibrated'}"
                + (
                    f" (resolve {thresholds.resolve}, merge {thresholds.merge},"
                    f" link {thresholds.link})"
                    if thresholds
                    else ""
                )
            )
        planned = turns(args)
        total = len(planned) + (0 if args.no_session_end else 1)
        stopped_by = "queue_empty"
        if embeddings is None:
            for label, text in planned:
                enqueue_turn(worker, label, text)
            if not args.no_session_end:
                worker.enqueue(SESSION_END, {"reason": "bench"}, scopes=[SCOPE], session_id=SESSION)
            print(flush=True)
            stopped_by = drive(worker, budget, jobs, total, verbose=args.verbose)
        else:
            # tier 3: one turn at a time, each followed by the embed job it queues, so
            # that the next turn resolves against the vectors of the previous ones
            for label, text in planned:
                enqueue_turn(worker, label, text)
                stopped_by = drive(worker, budget, jobs, total, verbose=args.verbose)
                if stopped_by != "queue_empty":
                    break
            if stopped_by == "queue_empty" and not args.no_session_end:
                worker.enqueue(SESSION_END, {"reason": "bench"}, scopes=[SCOPE], session_id=SESSION)
                stopped_by = drive(worker, budget, jobs, total, verbose=args.verbose)
        interrupted = stopped_by == "interrupted"
        if interrupted:
            print(f"\ninterrupted after {finished(jobs)} of {total} jobs: what was written so far")
        else:
            print(f"\nrun stopped by {stopped_by}")
        report(store, view)
        status = models.status()[CHAT] or {}
        print(
            f"totals: {budget.calls} model calls ({status.get('failures', 0)} failed),"
            f" {budget.tokens} tokens, {time.monotonic() - started:.1f} s wall"
        )
        if interrupted:
            return INTERRUPTED
        return 1 if any(j["outcome"] == "failed" for j in jobs) else 0
    finally:
        worker.stop()
        models.close()
        store.close()


def enqueue_turn(worker: Worker, label: str, text: str) -> None:
    worker.enqueue(TURN, {"user": text, "assistant": ""}, scopes=[SCOPE], session_id=SESSION)
    print(f"queued {label}: {len(text)} chars, languages {detect(text)}", flush=True)


def finished(jobs: Sequence[dict[str, Any]]) -> int:
    """Planned jobs that reached a final outcome (a retried attempt is not one, an
    embed job the service queued itself is not planned)."""
    return sum(1 for job in jobs if job["outcome"] in FINAL and job["kind"] != EMBED)


def drive(
    worker: Worker, budget: Budget, jobs: list[dict[str, Any]], total: int, *, verbose: bool
) -> str:
    """Run the queue one job at a time and print each one as soon as it is done.
    Only a final outcome advances the ``[k/N]`` count; an attempt that goes back to
    the queue prints ``[retry]`` (the job runs again later, counted once).
    Returns why it stopped; ``interrupted`` on Ctrl+C (``jobs`` holds every attempt)."""
    try:
        while True:
            job_started = time.monotonic()
            run_report = worker.run(budget, max_jobs=1)
            for job in run_report.jobs:
                jobs.append(job)
                seconds = time.monotonic() - job_started
                if job["outcome"] not in FINAL:
                    position = "[retry]"
                elif job["kind"] == EMBED:
                    position = "[embed]"
                else:
                    position = f"[{finished(jobs)}/{total}]"
                print(f"{position} {job_line(job)} ({seconds:.1f} s)", flush=True)
                if verbose and job["result"]:
                    print(f"    {job['result']}", flush=True)
            if run_report.stopped_by != "max_jobs":
                return str(run_report.stopped_by)
    except KeyboardInterrupt:
        return "interrupted"


def report(store: Store, view: Visibility) -> None:
    """Every node with its summary, facts (and their language check), proposals and
    edges; then the language-check totals."""
    print()
    print("nodes:")
    mismatches = checked = 0
    for node_id, kind, name, _norm, aliases in reversed(store.node_index(view)):
        node = store.get_node(node_id)
        flags = f" flags={','.join(node.flags)}" if node.flags else ""
        alias_text = f" aliases={list(aliases)}" if aliases else ""
        print(f"  {name} ({kind}){alias_text} lang={node.language or '-'}{flags}")
        if node.summary:
            print(f"    summary: {node.summary}")
        for fact in store.facts([node_id], view, live_only=False):
            expected, detected, verdict = language_check(store, fact)
            if verdict != "undetermined":
                checked += 1
            if verdict == "FLAGGED":
                mismatches += 1
            closed = f" closed→{fact.superseded_by}" if fact.valid_to else ""
            attr = f" [{fact.attribute}={fact.value}]" if fact.attribute else ""
            print(
                f"    - {fact.statement}{attr} (expected {expected}, wrote {detected}: {verdict};"
                f" confidence {fact.confidence:.2f}{closed})"
            )
        for proposal in store.proposals([node_id], view):
            if proposal.kind == "merge":
                other = store.get_node(proposal.value).name
                print(f"    ? merge with {other!r} (similarity {proposal.confidence})")
            else:
                print(f"    ? {proposal.kind} {proposal.value!r} (proposed, not in its source)")
        for edge in store.edges_of([node_id], view):
            if edge.src == node_id:
                other = store.get_node(edge.dst).name
                print(
                    f"    -> {edge.type} {other} ({edge.status}, weight {edge.weight:.2f},"
                    f" {len(edge.evidence)} sources)"
                )
    print()
    print(f"language checks: {checked} determined, {mismatches} flagged")


if __name__ == "__main__":
    raise SystemExit(main())
