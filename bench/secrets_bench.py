"""What a model keeps out of memory when a turn says a secret (SPEC P5).

Each case of ``bench/secrets_corpus.py`` is one turn, recorded in a session of
its own and extracted by the chat model; then the store is read table by
table. Two failures are counted, never one without the other:

* **leaked**: a secret of the case found in any table but ``sources`` (facts,
  proposals, session facts, the turn's trace, the lexical index...);
* **erased**: an ordinary word of the owner's (``keep``) the turn's trace
  masked - the rule took it for a secret;
* **lost**: an ordinary word no fact of the turn holds any more while the
  extraction dropped a fact for carrying a secret - the model wrote it beside
  a secret, and the fact went with it. A word the model simply did not
  extract is not counted: that is extraction, not this rule.

By default the service reads secrets as it ships (``detect_secrets_by_wording``
off: the model's list, credential attributes and key shapes); ``--wording``
turns the heuristic on, to compare. The extraction runs with the reading of
intent on, as the service does.

    python bench/secrets_bench.py --chat-url http://127.0.0.1:11434/v1 --model qwen3.5:9b

The options of ``bench/run.py`` for the endpoints are accepted (``--extra``,
``--json-mode``, ``--timeout``...); the embeddings options are accepted and
ignored, so one wrapper can call both. Results go to ``--out`` as JSON, with
the command that made them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.secrets_corpus import CASES, LANGUAGES, MIXED, PAGE, SECRET, TOOL, TRAP, Case, cases

from iifa.ingest.extract import PipelineConfig
from iifa.ingest.jobs import HANDLERS, TURN
from iifa.ingest.queue import Queue
from iifa.ingest.worker import ON_DEMAND, Budget, Gate, Worker
from iifa.llm import CHAT, RESERVED_REQUEST_KEYS, Endpoint, ModelClient
from iifa.store.repo import Store

STORE_KEY = "bench-throwaway-key"
SCOPE = "user:alice"
OWNER = "Alice"
MASK = "[secret]"
DEFAULT_OUT = Path(__file__).resolve().parent / "results"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chat-url", required=True, help="the chat endpoint (OpenAI-compatible)")
    parser.add_argument("--model", required=True, help="the chat model, explicit on every call")
    parser.add_argument("--api-key", default="", help="bearer key for the endpoint, if needed")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per model call")
    parser.add_argument("--max-tokens", type=int, default=4096, help="bound of every answer")
    parser.add_argument(
        "--json-mode", action=argparse.BooleanOptionalAction, default=True, help="JSON mode"
    )
    parser.add_argument(
        "--extra", action="append", default=[], metavar="KEY=VALUE", help="extra chat parameter"
    )
    parser.add_argument(
        "--wording",
        action="store_true",
        help="turn detect_secrets_by_wording on (off by default, as the service ships)",
    )
    parser.add_argument(
        "--languages",
        default="",
        help=f"comma-separated language codes (default: all of {', '.join(LANGUAGES)})",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results directory")
    parser.add_argument("--quiet", action="store_true", help="no progress lines")
    # accepted so that one wrapper calls this and bench/run.py alike; unused here
    parser.add_argument("--embeddings-url", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--embeddings-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    extra: dict[str, Any] = {}
    for pair in args.extra:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip() or key.strip() in RESERVED_REQUEST_KEYS:
            parser.error(f"--extra {pair!r}: expected KEY=VALUE, not a key iifa sets itself")
        try:
            extra[key.strip()] = json.loads(raw)
        except ValueError:
            extra[key.strip()] = raw
    args.extra = extra
    args.languages = tuple(code.strip() for code in args.languages.split(",") if code.strip())
    unknown = sorted(set(args.languages) - set(LANGUAGES))
    if unknown:
        parser.error(f"--languages: no case in {', '.join(unknown)}")
    return args


# --------------------------------------------------------------------------- reading the store


def table_texts(store: Store) -> dict[str, list[str]]:
    """Every text cell of every table (FTS shadow tables included), by table."""
    out: dict[str, list[str]] = defaultdict(list)
    names = [
        r["name"]
        for r in store.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    ]
    for name in names:
        for row in store.execute(f'SELECT * FROM "{name}"').fetchall():
            for cell in row:
                text = cell.decode("utf-8", "replace") if isinstance(cell, bytes) else cell
                if isinstance(text, str) and text:
                    out[name].append(text.casefold())
    return out


def leaks(texts: dict[str, list[str]], value: str) -> list[str]:
    """The tables but ``sources`` that hold ``value`` (case folded)."""
    folded = value.casefold()
    return sorted(
        name
        for name, cells in texts.items()
        if name != "sources" and any(folded in cell for cell in cells)
    )


def trace_of(store: Store, source_id: str | None) -> str | None:
    if source_id is None:
        return None
    row = store.execute(
        "SELECT excerpt FROM session_turns WHERE source_id = ?", (source_id,)
    ).fetchone()
    return str(row["excerpt"]) if row is not None else None


def fact_texts(store: Store, source_id: str | None) -> list[str]:
    """What the facts and the fact proposals of one source say."""
    if source_id is None:
        return []
    rows = store.execute(
        "SELECT statement, value FROM facts WHERE source_id = ?", (source_id,)
    ).fetchall()
    texts = [f"{r['statement']} {r['value'] or ''}" for r in rows]
    texts.extend(
        str(r["payload"])
        for r in store.execute(
            "SELECT payload FROM proposals WHERE source_id = ?", (source_id,)
        ).fetchall()
    )
    return texts


# --------------------------------------------------------------------------- one case


def audit(
    store: Store, case: Case, result: dict[str, Any], texts: dict[str, list[str]]
) -> dict[str, Any]:
    source_id = result.get("source_id")
    secrets = result.get("secrets") or {}
    dropped = int(secrets.get("dropped") or 0)
    leaked = {value: leaks(texts, value) for value in case.secrets}
    trace = trace_of(store, source_id)
    held = " ".join(fact_texts(store, source_id)).casefold()
    erased: list[str] = []
    lost: list[str] = []
    for word in case.keep:
        folded = word.casefold()
        if trace is not None and folded not in trace.casefold() and MASK in trace:
            erased.append(word)
        elif dropped and folded not in held:
            lost.append(word)
    return {
        "key": case.key,
        "language": case.language,
        "kind": case.kind,
        "form": case.form,
        "leaked": {value: tables for value, tables in leaked.items() if tables},
        "erased": erased,
        "lost": lost,
        "found": int(secrets.get("found") or 0),
        "dropped": dropped,
        "facts": int((result.get("facts") or {}).get("written") or 0),
        "proposed": int((result.get("facts") or {}).get("proposed") or 0),
        "trace": trace is not None,
        "calls": int(result.get("calls") or 0),
    }


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per kind, per language and per form of the secret."""

    def tally(selected: Sequence[dict[str, Any]]) -> dict[str, int]:
        secret_rows = [r for r in selected if r["kind"] in (SECRET, MIXED, TOOL)]
        keep_rows = [r for r in selected if r["kind"] in (TRAP, MIXED, PAGE)]
        return {
            "cases": len(selected),
            "secret_cases": len(secret_rows),
            "leaked_cases": sum(1 for r in secret_rows if r["leaked"]),
            "keep_cases": len(keep_rows),
            "erased_cases": sum(1 for r in keep_rows if r["erased"]),
            "lost_cases": sum(1 for r in keep_rows if r["lost"]),
        }

    by_language = {
        language: tally([r for r in rows if r["language"] == language])
        for language in dict.fromkeys(r["language"] for r in rows)
    }
    forms = sorted({r["form"] for r in rows if r["form"]})
    by_form = {form: tally([r for r in rows if r["form"] == form]) for form in forms}
    by_kind = {
        kind: tally([r for r in rows if r["kind"] == kind])
        for kind in dict.fromkeys(r["kind"] for r in rows)
    }
    return {
        "total": tally(rows),
        "by_kind": by_kind,
        "by_language": by_language,
        "by_form": by_form,
    }


# --------------------------------------------------------------------------- the run


def run(
    args: argparse.Namespace,
    workdir: Path,
    *,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    selected = cases(args.languages)
    chat = Endpoint(
        CHAT, args.chat_url, args.model, args.api_key, timeout=args.timeout, extra=args.extra
    )
    gate = Gate({CHAT: 1})
    models = ModelClient(chat, None, gate=gate)
    store = Store(workdir / "secrets.db", STORE_KEY, checkpoint_interval=0)
    pipeline = PipelineConfig(
        answer_max_tokens=args.max_tokens,
        json_mode=args.json_mode,
        owners=((SCOPE, OWNER, ()),),
        intent=True,
        detect_secrets_by_wording=args.wording,
    )
    worker = Worker(
        store,
        Queue(store),
        mode=ON_DEMAND,
        gate=gate,
        models=models,
        handlers=HANDLERS,
        retry_delay=0.0,
        pipeline=pipeline,
    )
    results: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    try:
        for index, case in enumerate(selected, start=1):
            payload: dict[str, Any] = {
                "user": case.user,
                "assistant": case.assistant,
                "external": [dict(part) for part in case.external],
                "external_id": case.key,
            }
            worker.enqueue(TURN, payload, scopes=[SCOPE], session_id=case.key)
            report = worker.run(Budget())
            for job in report.jobs:
                outcomes[f"{job['kind']}:{job['outcome']}"] += 1
                if job["kind"] == TURN and job["outcome"] == "done":
                    results[case.key] = job.get("result") or {}
            say(f"{index}/{len(selected)} {case.key} ({time.perf_counter() - started:.0f} s)")
        store.checkpoint("TRUNCATE")
        texts = table_texts(store)
        rows = [audit(store, case, results.get(case.key, {}), texts) for case in selected]
    finally:
        store.close()
    return {
        "bench": "secrets",
        "model": args.model,
        "url": args.chat_url,
        "json_mode": args.json_mode,
        "extra": args.extra,
        "detect_secrets_by_wording": args.wording,
        "cases": len(selected),
        "languages": list(dict.fromkeys(case.language for case in selected)),
        "seconds": round(time.perf_counter() - started, 1),
        "jobs": dict(sorted(outcomes.items())),
        "missing": [case.key for case in selected if case.key not in results],
        "summary": summarise(rows),
        "rows": rows,
    }


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    total = summary["total"]
    mode = "on" if report["detect_secrets_by_wording"] else "off"
    print(
        f"\nmodel {report['model']}, detect_secrets_by_wording {mode}, "
        f"{report['cases']} cases in {report['seconds']} s"
    )
    print(
        f"leaked: {total['leaked_cases']}/{total['secret_cases']} turns with a secret; "
        f"erased: {total['erased_cases']}/{total['keep_cases']} turns with ordinary words "
        f"(lost beside a secret: {total['lost_cases']})"
    )
    for title, table in (("language", "by_language"), ("form", "by_form"), ("kind", "by_kind")):
        print(f"\nby {title}:")
        for name, t in summary[table].items():
            print(
                f"  {name:10} leaked {t['leaked_cases']}/{t['secret_cases']:<3} "
                f"erased {t['erased_cases']}/{t['keep_cases']}  lost {t['lost_cases']}"
            )
    bad = [r for r in report["rows"] if r["leaked"] or r["erased"] or r["lost"]]
    if bad:
        print("\ncases:")
        for r in bad:
            what = []
            if r["leaked"]:
                what.append(
                    "leaked in " + "; ".join(sorted({t for ts in r["leaked"].values() for t in ts}))
                )
            if r["erased"]:
                what.append("erased " + ", ".join(r["erased"]))
            if r["lost"]:
                what.append("lost beside a secret " + ", ".join(r["lost"]))
            print(f"  {r['key']}: {' / '.join(what)}")
    if report["missing"]:
        print(f"\nnot processed: {', '.join(report['missing'])}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    say: Callable[[str], None] = (lambda _line: None) if args.quiet else print
    workdir = Path(tempfile.mkdtemp(prefix="iifa-secrets-"))
    try:
        report = run(args, workdir, say=say)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    report["command"] = " ".join(
        ["python", "bench/secrets_bench.py", *(sys.argv[1:] if argv is None else argv)]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    mode = "wording" if args.wording else "default"
    target = args.out / f"{stamp}-secrets-{mode}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CASES", "audit", "leaks", "main", "parse_args", "run", "summarise", "table_texts"]
