"""The benchmark harness (SPEC 11.1): iifa against three baselines on the
synthetic multilingual corpus, at the same token budgets, in one command.

    python bench/run.py
    python bench/run.py --embeddings-url http://127.0.0.1:11434/v1 --embeddings-model bge-m3
    python bench/run.py --chat-url http://127.0.0.1:11434/v1 --model qwen2.5:7b \\
        --embeddings-url http://127.0.0.1:11434/v1 --embeddings-model bge-m3

What it does, in a throwaway encrypted store in a temporary directory:

1. generates the corpus (``bench/synthetic.py``, ``--seed``): people who move,
   change roles and change their minds, told in six languages, in three scopes
   (one holds a homonym), with chatter in between; and the questions, each
   asked through one scope's view;
2. records every turn through the service's own worker (enqueue timed: the hot
   path of ``POST /v1/turns``), processing the queue after each turn as a
   conversation would. Extraction is the real tier-2 pipeline against
   ``--chat-url``, or, without it, an **oracle** that answers every extraction
   call with the gold answer (no network): the retrieval layer on its own;
3. answers every question with every system at every budget (``--budgets``):
   ``none``, ``truncated_history``, ``vector_topk`` (with embeddings), ``iifa``
   at tier 2, and at tier 3 with embeddings; each answer is whole units under
   the budget, counted with the same tokenizer;
4. reports the headline, **correct current answers per 1k injected tokens** (the
   current value injected and no stale one), beside raw recall@budget and recall
   per 1k tokens (which reward injecting every version), contradiction handling,
   latency p50/p95, ingestion throughput and scope leaks (must be 0); and a
   probe set per scope (SPEC 11.2) built from the questions, run at each budget;
5. writes every number, the corpus summary and the exact command line to
   ``--out`` (default ``bench/results/``, gitignored) as one JSON file.

Ctrl+C during ingestion stops it: what was recorded so far is measured and
reported, marked ``interrupted``, and the exit code is 130. ``--keep-store PATH``
(debugging only) keeps the encrypted store to read what a model wrote. Content is not
printed; the JSON holds ids, keys and numbers, plus the corpus seed that
regenerates every text.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.dream import (
    PERSON_FORMS,
    ProbeTrace,
    Truth,
    closure_verdict,
    dream_bench,
    oracle_verdict,
)
from bench.intent_bench import IntentTally, oracle_consolidation, worksession_audit
from bench.self_bench import (
    SELF_NAME,
    SELF_SCOPE,
    add_self_turns,
    family_after_dream,
    self_audit,
    turn_payload,
)
from bench.synthetic import (
    ALICE,
    OBSERVED_QUESTION,
    OWNER_NAME,
    Corpus,
    Question,
    generate,
    value_forms,
)

from iifa import vectors
from iifa.config import MIN_PROMPT_TOKENS
from iifa.dream.adjudicate import name_neighbours, names_close
from iifa.dream.prompts import MEMORY_END, MEMORY_START
from iifa.dream.run import DreamConfig, Dreamer
from iifa.ingest.extract import PipelineConfig
from iifa.ingest.jobs import HANDLERS, TURN
from iifa.ingest.queue import Queue
from iifa.ingest.session_prompts import CONSOLIDATION_START
from iifa.ingest.sessions import SESSION_CLOSE
from iifa.ingest.speaker import is_speaker
from iifa.ingest.worker import ON_DEMAND, Budget, Gate, Worker
from iifa.language import UNDETERMINED, detect, is_code, language_of_span
from iifa.llm import CHAT, EMBEDDINGS, RESERVED_REQUEST_KEYS, Endpoint, ModelClient, QueryEmbedder
from iifa.recall.dense import Dense
from iifa.recall.engine import RecallEngine
from iifa.recall.probes import run_probe_set
from iifa.scopes import Visibility
from iifa.store.models import Fact
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer, Tokenizer

STORE_KEY = "bench-throwaway-key"
INTERRUPTED = 130
ORACLE_URL = "http://oracle.invalid/v1"  # never resolved: the oracle transport answers in-process
ORACLE_MODEL = "oracle"
DEFAULT_OUT = Path(__file__).resolve().parent / "results"
SOURCE = re.compile(r"<<<SOURCE>>>\n(.*)\n<<<END>>>", re.DOTALL)
LABEL = re.compile(r"^[^\n:]{1,60}: ")  # "User: ", or the owner's name (SPEC 4.2)
PAIRS = re.compile(r"<<<PAIRS>>>\n(.*)\n<<<END>>>", re.DOTALL)
EMPTY_ANSWER = {"nodes": [], "facts": [], "edges": []}
DREAM_UNIT = "You reorganise one entry"  # how the oracle tells the dream's calls apart
DREAM_ADJUDICATION = "You compare two entries"


# --------------------------------------------------------------------------- arguments


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--chat-url", default=None, help="extract with this model (default: the oracle)"
    )
    parser.add_argument("--model", default=None, help="the chat model, explicit on every call")
    parser.add_argument("--api-key", default="", help="bearer key for the endpoints, if needed")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per model call")
    parser.add_argument("--max-tokens", type=int, default=4096, help="bound of every answer")
    parser.add_argument(
        "--json-mode", action=argparse.BooleanOptionalAction, default=True, help="JSON mode"
    )
    parser.add_argument(
        "--extra", action="append", default=[], metavar="KEY=VALUE", help="extra chat parameter"
    )
    parser.add_argument("--embeddings-url", default=None, help="tier 3 and the vector baseline")
    parser.add_argument("--embeddings-model", default=None, help="the embedding model")
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=None,
        help="tier 3: the cosine at which a request matches an activity (default: words only)",
    )
    parser.add_argument("--budgets", default="150,300,600", help="comma-separated token budgets")
    parser.add_argument("--seed", type=int, default=7, help="corpus seed")
    parser.add_argument("--months", type=int, default=12, help="months of events (>= 2)")
    parser.add_argument("--chatter", type=float, default=3.0, help="chatter turns per event turn")
    parser.add_argument(
        "--traps",
        action="store_true",
        help="add where each person works next to where they live (false supersessions);"
        " another corpus: compare only with runs that used it",
    )
    parser.add_argument(
        "--rephrase",
        action="store_true",
        help="add turns restating a value in another language (same_as, duplicates);"
        " another corpus: compare only with runs that used it",
    )
    parser.add_argument(
        "--replacement-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask role, team and preferences 'replaces, or both true?' (SPEC 5.4; on by"
        " default, --no-replacement-question turns it off)",
    )
    parser.add_argument(
        "--restatement-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask, within one language, whether a change whose values share words tells"
        " the same value again (SPEC 5.4; on by default, --no-restatement-question turns"
        " it off)",
    )
    parser.add_argument(
        "--value-norm",
        action="store_true",
        help="ask each value's comparison form and link the values whose forms agree across"
        " languages (SPEC 5.4; off by default): against the same command without it",
    )
    parser.add_argument(
        "--force-language",
        default=None,
        metavar="CODE",
        help="every scope forces this language on what is written (P12 override): the"
        " language question, against the same run without it; needs a real model",
    )
    parser.add_argument(
        "--speaker",
        action="store_true",
        help=f"declare {OWNER_NAME} the owner of {ALICE}, who speaks in the first person;"
        " another corpus: compare only with runs that used it",
    )
    parser.add_argument(
        "--name-traps",
        action="store_true",
        help="add people whose names are near another's of the same scope (Karima beside"
        " Karim): wrong attachments; another corpus: compare only with runs that used it",
    )
    parser.add_argument(
        "--neighbours",
        action="store_true",
        help="add turns describing two projects side by side (a web service and a tool):"
        " facts attached to the other project; another corpus: compare only with runs that"
        " used it",
    )
    parser.add_argument(
        "--intent",
        action="store_true",
        help="add turns not meant as facts (hypotheses, questions, examples, test personas)"
        " beside sincere controls on the same values; another corpus: compare only with runs"
        " that used it",
    )
    parser.add_argument(
        "--worksession",
        action="store_true",
        help="turn the last month of user:alice into a work session on a slide deck, closed"
        " at its end; another corpus: compare only with runs that used it",
    )
    parser.add_argument(
        "--observed",
        action="store_true",
        help="add things whose state changes (a folder, a service, a queue), observed in"
        " several months beside a lasting fact under the same key; another corpus: compare"
        " only with runs that used it",
    )
    parser.add_argument(
        "--self",
        dest="self_corpus",
        action="store_true",
        help=f"add a scope of its own ({SELF_SCOPE}) whose memory is also the agent's"
        f" ({SELF_NAME}): lessons, views, commitments, rules of method, and what must not"
        " become its memory (pages restated, echoes, roles); another corpus: compare only"
        " with runs that used it",
    )
    parser.add_argument(
        "--read-intent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read the frame, modality and duration of what is extracted (SPEC 4.2; on by"
        " default, --no-read-intent turns it off)",
    )
    parser.add_argument(
        "--resolve-threshold",
        type=float,
        default=None,
        help="replace the embedding model's resolve threshold for this run (SPEC 6.1)",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=None,
        help="replace the embedding model's merge threshold for this run (SPEC 6.1)",
    )
    parser.add_argument(
        "--dream",
        action="store_true",
        help="after ingestion, every top-level scope dreams (a dry run, then a real run) and"
        " the iifa systems are measured again (SPEC 5.6, 11.1)",
    )
    parser.add_argument("--dream-url", default=None, help="the dream model's endpoint")
    parser.add_argument("--dream-model", default=None, help="the dream model (default: --model)")
    parser.add_argument(
        "--dream-extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra parameter of the dream model's calls (e.g. think=false)",
    )
    parser.add_argument(
        "--dream-prompt-tokens",
        type=int,
        default=4000,
        help="the prompt budget of one dream call, by the service's counter (SPEC 5.6); the"
        " model's context window must hold it plus --max-tokens",
    )
    parser.add_argument(
        "--dream-accept-review",
        action="store_true",
        help="then accept every queued operation and measure again: a ceiling, not what a"
        " user gets",
    )
    parser.add_argument(
        "--hostile",
        action="store_true",
        help="with --dream: write records carrying instructions (name, alias, summary, facts)"
        " before the dream, and count the operations that follow them (must be 0)",
    )
    parser.add_argument("--repeat", type=int, default=3, help="latency rounds per question")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results directory")
    parser.add_argument("--quiet", action="store_true", help="no progress lines")
    parser.add_argument(
        "--keep-store",
        type=Path,
        default=None,
        metavar="PATH",
        help="debugging only: keep the encrypted store at PATH (key: "
        f"{STORE_KEY!r}) to read what was written; delete it afterwards",
    )
    args = parser.parse_args(argv)
    if bool(args.chat_url) != bool(args.model):
        parser.error("--chat-url and --model go together")
    if bool(args.embeddings_url) != bool(args.embeddings_model):
        parser.error("--embeddings-url and --embeddings-model go together")
    try:
        args.budgets = sorted({int(b) for b in args.budgets.split(",") if b.strip()})
    except ValueError:
        parser.error("--budgets: comma-separated integers")
    if not args.budgets or min(args.budgets) <= 0:
        parser.error("--budgets: positive integers")
    for option in ("extra", "dream_extra"):
        extra: dict[str, Any] = {}
        for pair in getattr(args, option):
            key, sep, raw = pair.partition("=")
            if not sep or not key.strip() or key.strip() in RESERVED_REQUEST_KEYS:
                parser.error(
                    f"--{option.replace('_', '-')} {pair!r}: expected KEY=VALUE,"
                    " not a key iifa sets itself"
                )
            try:
                extra[key.strip()] = json.loads(raw)
            except ValueError:
                extra[key.strip()] = raw
        setattr(args, option, extra)
    dream_only = ("dream_url", "dream_model", "dream_extra", "dream_accept_review", "hostile")
    if not args.dream and any(getattr(args, name) for name in dream_only):
        parser.error(
            "--dream-url, --dream-model, --dream-extra, --dream-accept-review and"
            " --hostile need --dream"
        )
    if args.dream_url and not (args.dream_model or args.model):
        parser.error("--dream-url needs --dream-model (or --model)")
    if args.dream_prompt_tokens < MIN_PROMPT_TOKENS:
        parser.error(f"--dream-prompt-tokens: at least {MIN_PROMPT_TOKENS}")
    for option in ("extra", "dream_extra"):
        window = (getattr(args, option).get("options") or {}).get("num_ctx")
        needed = args.max_tokens + (args.dream_prompt_tokens if option == "dream_extra" else 0)
        if isinstance(window, int) and window < needed:
            parser.error(
                f"--{option.replace('_', '-')} options.num_ctx={window}: the window must hold"
                f" the prompt and the answer (at least {needed})"
            )
    if args.force_language is not None:
        code = str(args.force_language).strip().lower()
        if not is_code(code):
            parser.error(f"--force-language {args.force_language!r}: an ISO 639 code")
        args.force_language = code
    for name in ("resolve_threshold", "merge_threshold"):
        value = getattr(args, name)
        if value is not None and not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} {value}: a cosine in (0, 1]")
    resolve, merge = args.resolve_threshold, args.merge_threshold
    if resolve is not None and merge is not None and merge < resolve:
        parser.error("--merge-threshold must be at least --resolve-threshold")
    if args.keep_store is not None and args.keep_store.exists():
        parser.error(f"--keep-store {args.keep_store}: already exists, never overwritten")
    return args


# --------------------------------------------------------------------------- the oracle


class Oracle:
    """Answers extraction calls with the corpus's gold answers (the ideal model),
    mention adjudications with "new", value adjudications (SPEC 5.4) from the
    ground truth (same value, different value, or unsure for a form it does not
    know), anything else with nothing. Two turns may hold the same text in two
    scopes (the homonym): their answers are served in turn order."""

    def __init__(self, corpus: Corpus) -> None:
        self._consolidation = oracle_consolidation(corpus)
        self._gold: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for turn in corpus.turns:
            if not corpus.is_chatter(turn):
                key = turn.source_key or turn.text  # --self: the whole source
                self._gold[key].append(corpus.gold[key])
        self._forms = value_forms()
        # --traps: the statements that add a value to what holds (truth: "both")
        self._added = {
            str(corpus.gold[t.text]["facts"][0]["statement"])
            for t in corpus.turns
            if t.event is not None and corpus.events[t.event].added
        }
        # --neighbours: a project told something unrelated under its key (truth: "both")
        self._asides = {
            str(corpus.gold[t.text]["facts"][0]["statement"]) for t in corpus.turns if t.aside
        }
        # --observed: what a thing is for and its state are both true under one key
        self._lasting = {
            str(corpus.gold[t.text]["facts"][0]["statement"])
            for t in corpus.turns
            if t.observed is not None and t.observation is None
        }
        self.calls = 0

    def answer(self, messages: Sequence[dict[str, Any]]) -> str:
        self.calls += 1
        system = str(messages[0].get("content", "")) if messages else ""
        user = str(messages[-1].get("content", "")) if messages else ""
        if system.startswith(DREAM_UNIT):
            return json.dumps({"operations": []})  # the oracle reorganises nothing
        if system.startswith(DREAM_ADJUDICATION):
            return json.dumps(oracle_verdict(json.loads(memory_of(user))))
        if system.startswith(CONSOLIDATION_START):
            return json.dumps(self._consolidation, ensure_ascii=False)
        if '"decisions"' in system:
            return json.dumps({"decisions": []})
        if '"replacements"' in system:
            return json.dumps({"replacements": self._verdicts(user, change="replaces")})
        if '"restatements"' in system:
            return json.dumps({"restatements": self._verdicts(user)})
        if '"verdicts"' in system:
            return json.dumps({"verdicts": self._verdicts(user)})
        source = source_text(messages)
        if source is None:
            return json.dumps(EMPTY_ANSWER)
        queue = self._gold.get(source)
        if not queue:
            return json.dumps(EMPTY_ANSWER)
        answer = queue.popleft() if len(queue) > 1 else queue[0]
        if "value_norm" in system:
            answer = self._with_forms(answer)
        return json.dumps(answer, ensure_ascii=False)

    def _with_forms(self, answer: dict[str, Any]) -> dict[str, Any]:
        """SPEC 5.4: an ideal comparison form, the corpus's own id of each value
        ("Лион" -> "city:lyon"); a value the corpus does not know gets none."""
        facts = []
        for fact in answer.get("facts") or ():
            form = self._forms.get(str(fact.get("value") or "").lower())
            facts.append({**fact, "value_norm": form} if form else fact)
        return {**answer, "facts": facts}

    def _verdicts(self, user: str, *, change: str = "different") -> list[dict[str, Any]]:
        """The truth for each pair: the same value, or another one (``change``:
        "different" for the value question, "replaces" for role, team,
        preferences and free-form keys - in this corpus a new value always
        replaces the old one, unless it was added or is an aside)."""
        match = PAIRS.search(user)
        out: list[dict[str, Any]] = []
        for pair in json.loads(match.group(1)) if match else []:
            a = self._forms.get(str(pair["a"]["value"]).lower())
            b = self._forms.get(str(pair["b"]["value"]).lower())
            word = "unsure" if a is None or b is None else "same" if a == b else change
            if word == "replaces" and str(pair["b"]["statement"]) in self._added:
                word = "both"  # an added value replaces nothing
            if change == "replaces" and str(pair["b"]["statement"]) in self._asides:
                word = "both"  # an unrelated statement under one key replaces nothing
            if change == "replaces" and self._lasting & {
                str(pair["a"]["statement"]),
                str(pair["b"]["statement"]),
            }:
                word = "both"  # a state observed beside what the thing is for
            out.append({"pair": pair["pair"], "verdict": word})
        return out


def memory_of(user: str) -> str:
    """What a dream call carries between its memory markers."""
    start = user.index(MEMORY_START) + len(MEMORY_START)
    return user[start : user.rindex(MEMORY_END)]


def source_text(messages: Sequence[dict[str, Any]]) -> str | None:
    """The turn's text in an extraction call, without its label ("User: ", or
    the owner's name when one is declared)."""
    user = str(messages[-1].get("content", "")) if messages else ""
    match = SOURCE.search(user)
    if match is None:
        return None
    return LABEL.sub("", match.group(1), count=1)


class OracleTransport(httpx.BaseTransport):
    """The oracle behind the chat endpoint's URL, in-process; every other request
    (the embeddings endpoint) goes to the network transport."""

    def __init__(self, oracle: Oracle) -> None:
        self.oracle = oracle
        self._network = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != "oracle.invalid":
            return self._network.handle_request(request)
        body = json.loads(request.content or b"{}")
        content = self.oracle.answer(body.get("messages") or [])
        payload = {
            "id": f"oracle-{self.oracle.calls}",
            "model": ORACLE_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(200, json=payload)

    def close(self) -> None:
        self._network.close()


# --------------------------------------------------------------------------- systems


@dataclass
class Unit:
    """One injected thing: a turn, a fact or a node handle."""

    turn: str | None  # the turn it supports, when any (a node handle supports none)
    scopes: tuple[str, ...]
    flags: tuple[str, ...] = ()
    also: tuple[str, ...] = ()  # a fact a dream consolidated: its originals' turns

    @property
    def supports(self) -> set[str]:
        return {t for t in (self.turn, *self.also) if t is not None}


@dataclass
class Answer:
    units: list[Unit]
    tokens: int
    seconds: float
    tier: int | None = None


System = Callable[[Question, int], Answer]
Events = dict[str, str] | dict[str, tuple[str, ...]]


def unit_cost(tokenizer: Tokenizer, text: str) -> int:
    return tokenizer.count(text) + tokenizer.per_item_overhead


def no_memory(_question: Question, _budget: int) -> Answer:
    return Answer([], 0, 0.0)


def truncated_history(corpus: Corpus, tokenizer: Tokenizer) -> System:
    """The view's turns, newest first, as many whole turns as fit (a contiguous tail)."""
    costs = {t.key: unit_cost(tokenizer, t.text) for t in corpus.turns}

    def answer(question: Question, budget: int) -> Answer:
        started = time.perf_counter()
        view = Visibility.of_scope(question.scope)
        units: list[Unit] = []
        used = 0
        for turn in reversed(corpus.turns):
            if not view.sees([turn.scope]):
                continue
            if used + costs[turn.key] > budget:
                break
            used += costs[turn.key]
            units.append(Unit(turn.key, (turn.scope,)))
        return Answer(units, used, time.perf_counter() - started)

    return answer


def vector_topk(
    corpus: Corpus, tokenizer: Tokenizer, client: ModelClient, model: str, batch: int = 32
) -> tuple[System, dict[str, Any]]:
    """The view's turns embedded as they are, nearest to the question first, as many
    whole turns as fit (skipping one that does not, as a packer would)."""
    started = time.perf_counter()
    texts = sorted({t.text for t in corpus.turns})
    unit: dict[str, list[float]] = {}
    calls = 0
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        result = client.embed(chunk, model=model)
        calls += 1
        for text, values in zip(chunk, result.vectors, strict=True):
            unit[text] = vectors.normalise(values)
    build = {
        "texts": len(texts),
        "calls": calls,
        "seconds": round(time.perf_counter() - started, 3),
    }
    costs = {t.key: unit_cost(tokenizer, t.text) for t in corpus.turns}

    def answer(question: Question, budget: int) -> Answer:
        began = time.perf_counter()
        query = vectors.normalise(client.embed([question.query], model=model).vectors[0])
        view = Visibility.of_scope(question.scope)
        visible = [t for t in corpus.turns if view.sees([t.scope])]  # filter first (P9)
        ranked = sorted(
            visible,
            key=lambda t: (-sum(a * b for a, b in zip(query, unit[t.text], strict=True)), t.key),
        )
        units: list[Unit] = []
        used = 0
        for turn in ranked:
            if used + costs[turn.key] <= budget:
                used += costs[turn.key]
                units.append(Unit(turn.key, (turn.scope,)))
        return Answer(units, used, time.perf_counter() - began)

    return answer, build


def iifa_system(store: Store, engine: RecallEngine, source_turns: dict[str, str]) -> System:
    """``recall`` as an agent calls it; only fact handles support an answer. A
    fact a dream consolidated supports the turns of its originals (SPEC 11.1)."""

    def turn_of(fact_id: str) -> str | None:
        source_id = store.get_fact(fact_id).source_id
        return source_turns.get(source_id) if source_id else None

    def originals(fact_id: str) -> tuple[str, ...]:
        out: list[str] = []
        for row in store.execute(
            "SELECT id, source_id FROM facts WHERE consolidated_into = ?", (fact_id,)
        ).fetchall():
            turn = source_turns.get(row["source_id"]) if row["source_id"] else None
            out.extend([turn] if turn else originals(row["id"]))
        return tuple(out)

    def answer(question: Question, budget: int) -> Answer:
        started = time.perf_counter()
        result = engine.recall(
            question.query, view=Visibility.of_scope(question.scope), budget_tokens=budget
        )
        seconds = time.perf_counter() - started
        units = [
            Unit(
                turn_of(item["id"]) if item["kind"] == "fact" else None,
                store.scopes_of(item["id"]),
                tuple(item.get("flags") or ()),
                originals(item["id"]) if item["kind"] == "fact" else (),
            )
            for item in result["items"]
        ]
        return Answer(units, int(result["used_tokens"]), seconds, int(result["tier"]))

    return answer


# --------------------------------------------------------------------------- metrics


def percentile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]


@dataclass
class Tally:
    questions: int = 0
    supported: int = 0
    correct: int = 0  # current supported and no stale one: the headline
    duplicates: int = 0  # two units supporting one event (SPEC 11.1)
    per_question: list[dict[str, Any]] = field(default_factory=list)  # keys, booleans
    tokens: int = 0
    leaks: int = 0
    seconds: list[float] = field(default_factory=list)
    contradictions: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            kind: {
                "questions": 0,
                "correct": 0,
                "stale_only": 0,
                "both": 0,
                "both_flagged": 0,
                "neither": 0,
            }
            for kind in ("all", "same_language", "cross_language")
        }
    )
    tiers: set[int] = field(default_factory=set)

    def add(self, question: Question, answer: Answer, events: Events) -> None:
        """``events``: the turn (or turns: the original and its restatements)
        supporting each event."""

        def told(event: str) -> set[str]:
            value = events.get(event, ())
            return {value} if isinstance(value, str) else set(value)

        view = Visibility.of_scope(question.scope)
        self.questions += 1
        self.tokens += answer.tokens
        if answer.tier is not None:
            self.tiers.add(answer.tier)
        self.leaks += sum(1 for u in answer.units if not view.sees(u.scopes))
        turns = set().union(*(u.supports for u in answer.units))
        current = any(told(e) & turns for e in question.expected)
        stale = any(told(e) & turns for e in question.stale)
        self.supported += int(current)
        self.correct += int(current and not stale)
        self.per_question.append(
            {
                "question": question.key,
                "current": bool(current),
                "stale": bool(stale),
                "correct": bool(current and not stale),
            }
        )
        self.duplicates += int(
            any(
                sum(1 for u in answer.units if u.supports & told(e)) > 1
                for e in (*question.expected, *question.stale)
            )
        )
        if not question.changed:
            return
        expected_turns = set().union(*(told(e) for e in question.expected))
        flagged = any("conflict" in u.flags for u in answer.units if u.supports & expected_turns)
        outcome = (
            "correct"
            if current and not stale
            else "both"
            if current
            else "stale_only"
            if stale
            else "neither"
        )
        kinds = ["all", "same_language" if question.same_language_change else "cross_language"]
        for kind in kinds:
            row = self.contradictions[kind]
            row["questions"] += 1
            row[outcome] += 1
            if outcome == "both" and flagged:
                row["both_flagged"] += 1

    def as_dict(self) -> dict[str, Any]:
        recall = self.supported / self.questions if self.questions else 0.0
        mean_tokens = self.tokens / self.questions if self.questions else 0.0
        ms = [s * 1000 for s in self.seconds] or [0.0]
        contradictions = {}
        for kind, row in self.contradictions.items():
            contradictions[kind] = {
                **row,
                "accuracy": round(row["correct"] / row["questions"], 3)
                if row["questions"]
                else None,
            }
        correct = self.correct / self.questions if self.questions else 0.0
        return {
            "questions": self.questions,
            "correct": self.correct,
            "correct_rate": round(correct, 3),
            "correct_per_1k_tokens": round(correct / mean_tokens * 1000, 3) if mean_tokens else 0.0,
            "supported": self.supported,
            "duplicates": self.duplicates,
            "recall": round(recall, 3),
            "mean_tokens": round(mean_tokens, 1),
            "recall_per_1k_tokens": round(recall / mean_tokens * 1000, 3) if mean_tokens else 0.0,
            "contradictions": contradictions,
            "latency_ms": {
                "p50": round(percentile(ms, 0.5), 2),
                "p95": round(percentile(ms, 0.95), 2),
                "n": len(self.seconds),
            },
            "leaks": self.leaks,
            "per_question": list(self.per_question),
            "tier": sorted(self.tiers)[0] if len(self.tiers) == 1 else sorted(self.tiers) or None,
        }


def evaluate(
    systems: dict[str, System],
    corpus: Corpus,
    budgets: Sequence[int],
    repeat: int,
    events: Events,
) -> dict[str, dict[str, Any]]:
    """Every system, every budget, every question: the first round scores, every
    round times."""
    out: dict[str, dict[str, Any]] = {}
    for name, system in systems.items():
        out[name] = {}
        for budget in budgets:
            tally = Tally()
            for round_ in range(max(1, repeat)):
                for question in corpus.questions:
                    answer = system(question, budget)
                    tally.seconds.append(answer.seconds)
                    if round_ == 0:
                        tally.add(question, answer, events)
            out[name][str(budget)] = tally.as_dict()
    return out


# --------------------------------------------------------------------------- ingestion


def ingest(worker: Worker, corpus: Corpus, *, progress: Callable[[str], None]) -> dict[str, Any]:
    """Record every turn (enqueue timed), processing the queue after each one."""
    enqueue: list[float] = []
    totals: dict[str, int] = defaultdict(int)
    outcomes: dict[str, int] = defaultdict(int)
    chatter = {"turns": 0, "turns_with_facts": 0, "facts": 0, "nodes": 0}
    owner_turns: list[dict[str, Any]] = []
    self_jobs: dict[str, dict[str, Any]] = {}  # --self: each case's job report
    adjudications: list[dict[str, Any]] = []
    intent = IntentTally()
    budget = Budget()
    processed = 0
    interrupted = False
    started = time.perf_counter()
    try:
        for index, turn in enumerate(corpus.turns, start=1):
            began = time.perf_counter()
            payload = {"user": turn.text, "assistant": "", "external_id": turn.key}
            if turn.self_case is not None:  # --self: the reply and the pages
                payload.update(turn_payload(turn))
            if turn.occurred_at:  # --observed: the day the turn says it happened
                payload["occurred_at"] = turn.occurred_at
            worker.enqueue(TURN, payload, scopes=[turn.scope], session_id=turn.session)
            enqueue.append(time.perf_counter() - began)
            if turn.ends_session:  # the host closes the session (SPEC 5.8)
                worker.enqueue(
                    SESSION_CLOSE, {"reason": "end"}, scopes=[turn.scope], session_id=turn.session
                )
            report = worker.run(budget)
            for job in report.jobs:
                outcomes[f"{job['kind']}:{job['outcome']}"] += 1
                result = job.get("result") or {}
                if job["kind"] == TURN and job["outcome"] == "done":
                    processed += 1
                    totals["empty_answers"] += int(result.get("empty_answers") or 0)
                    if corpus.is_chatter(turn):
                        # SPEC 11.1: what a turn with nothing to remember still produced
                        written = int((result.get("facts") or {}).get("written") or 0)
                        created = int((result.get("nodes") or {}).get("created") or 0)
                        chatter["turns"] += 1
                        chatter["facts"] += written
                        chatter["nodes"] += created
                        chatter["turns_with_facts"] += int(bool(written or created))
                    if turn.speaker:
                        owner_turns.append(owner_record(turn.key, turn.language, result))
                    if turn.self_case is not None:
                        self_jobs[turn.key] = result
                    intent.add(turn, result)
                    for key, value in (result.get("facts") or {}).items():
                        totals[f"facts_{key}"] += int(value)
                    for key, value in (result.get("keys") or {}).items():
                        totals[f"keys_{key}"] += int(value)
                    values = result.get("values") or {}
                    for key in ("same", "different", "norm"):
                        totals[f"values_{key}"] += int(values.get(key) or 0)
                    for key, value in (values.get("adjudicated") or {}).items():
                        totals[f"values_adjudicated_{key}"] += int(value)
                    for key, value in (values.get("replacement") or {}).items():
                        totals[f"values_replacement_{key}"] += int(value)
                    for key, value in (values.get("restatement") or {}).items():
                        totals[f"values_restatement_{key}"] += int(value)
                    totals["calls"] += int(result.get("calls") or 0)
                    totals["mentions_adjudicated"] += int(result.get("mentions_adjudicated") or 0)
                    totals["adjudication_calls"] += int(result.get("adjudication_calls") or 0)
                    totals["source_forms"] += int(result.get("source_forms") or 0)
                    adjudications.extend(
                        {"turn": turn.key, **record} for record in result.get("adjudications") or ()
                    )
            if index % 25 == 0 or index == len(corpus.turns):
                progress(
                    f"ingested {index}/{len(corpus.turns)} turns ({time.perf_counter() - started:.1f} s)"
                )
    except KeyboardInterrupt:
        interrupted = True
        progress(f"interrupted after {processed} processed turns: measuring what was recorded")
    seconds = time.perf_counter() - started
    ms = [s * 1000 for s in enqueue] or [0.0]
    return {
        "turns": len(corpus.turns),
        "processed": processed,
        "interrupted": interrupted,
        "seconds": round(seconds, 3),
        "processed_per_s": round(processed / seconds, 2) if seconds else None,
        "enqueue_ms": {"p50": round(percentile(ms, 0.5), 3), "p95": round(percentile(ms, 0.95), 3)},
        "model_calls": budget.calls,
        "model_tokens": budget.tokens,
        "jobs": dict(sorted(outcomes.items())),
        "extraction": dict(sorted(totals.items())),
        "chatter": chatter,
        "owner_turns": owner_turns,
        "adjudications": adjudications,
        "intent": intent.as_dict(),
        "self_jobs": self_jobs,
    }


def owner_record(key: str, language: str, result: dict[str, Any]) -> dict[str, Any]:
    """What one of the owner's turns became (SPEC 11.1, ``--speaker``): numbers
    only, so that a lost sentence and the cause of its loss can be read."""
    facts = result.get("facts") or {}
    return {
        "turn": key,
        "language": language,
        **{
            name: int(facts.get(name) or 0)
            for name in (
                "written",
                "proposed",
                "incomplete",
                "cross_script",
                "mixed_script",
                "language_mismatch",
                "duplicate",
            )
        },
        "retries": int(result.get("retries") or 0),
        "nodes_created": int((result.get("nodes") or {}).get("created") or 0),
    }


def supersessions(store: Store, corpus: Corpus, turns: dict[str, str]) -> dict[str, Any]:
    """Every closed fact, judged against the ground truth (SPEC 11.1, 5.4): correct
    when its replacement comes from a later event of the same person's same
    attribute; false when it comes from another attribute or person, or not a
    later one; restated when one tells the other's value again (``--rephrase``);
    unmatched when either fact came from a chatter turn. The false ones are
    listed (up to ten) with both turns, people, attributes and statements."""
    event_of = {t.key: t.event or t.restates for t in corpus.turns}
    facts = store.execute("SELECT id, source_id, statement FROM facts").fetchall()
    source_of = {r["id"]: r["source_id"] for r in facts}
    statement_of = {r["id"]: r["statement"] for r in facts}

    def event(fact_id: str | None) -> Any:
        turn = turns.get(source_of.get(fact_id or "") or "")
        key = event_of.get(turn) if turn else None
        return corpus.events.get(key) if key else None

    cases: list[dict[str, Any]] = []

    def side(fact_id: str, told: Any) -> dict[str, Any]:
        return {
            "turn": turns.get(source_of.get(fact_id) or ""),
            "person": told.person,
            "attribute": told.attribute,
            "statement": statement_of.get(fact_id),
        }

    def wrong(fact_id: str, by: str, old: Any, new: Any, cause: str) -> None:
        out["false"] += 1
        if len(cases) < 10:
            cases.append({"closed": side(fact_id, old), "by": side(by, new), "cause": cause})

    out: dict[str, Any] = {
        "total": 0,
        "correct": 0,
        "false": 0,
        "false_other_attribute": 0,
        "restated": 0,
        "false_replaces": 0,
        "unmatched": 0,
    }
    rows = store.execute(
        "SELECT id, superseded_by FROM facts WHERE superseded_by IS NOT NULL"
    ).fetchall()
    for row in rows:
        out["total"] += 1
        old, new = event(row["id"]), event(row["superseded_by"])
        if old is None or new is None:
            out["unmatched"] += 1
            continue
        outcome, cause = closure_verdict(old, new)
        if outcome != "false":
            out[outcome] += 1  # restated: a value closed by itself, told again
            continue
        wrong(row["id"], row["superseded_by"], old, new, str(cause))
        if cause == "added value":
            out["false_replaces"] += 1  # an added value closed one still true
        elif cause == "other attribute":
            out["false_other_attribute"] += 1
    out["false_cases"] = cases
    return out


def name_adjudications(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every mention sent to the adjudication (SPEC 11.1): how many, how many had
    a candidate from name vectors, and the near-name pairs - a mention and a
    candidate naming two different people of the corpus - with what the model
    answered (attached: a wrong attachment)."""
    pairs: list[dict[str, Any]] = []
    for record in records:
        mention = PERSON_FORMS.get(str(record["mention"]).casefold())
        for candidate in record["candidates"]:
            other = PERSON_FORMS.get(str(candidate["name"]).casefold())
            if mention is None or other is None or mention == other:
                continue
            chosen = record.get("chosen")
            answer = (
                "new" if chosen is None else "attached" if chosen == candidate["name"] else "other"
            )
            pairs.append(
                {
                    "turn": record.get("turn"),
                    "mention": record["mention"],
                    "candidate": candidate["name"],
                    "score": candidate["score"],
                    "semantic": candidate["semantic"],
                    "answer": answer,
                }
            )
    return {
        "mentions": len(records),
        "with_semantic_candidates": sum(
            1 for r in records if any(c["semantic"] for c in r["candidates"])
        ),
        "near_name_pairs": len(pairs),
        "attached": sum(1 for p in pairs if p["answer"] == "attached"),
        "pairs": pairs,
    }


def attachments(store: Store, corpus: Corpus, turns: dict[str, str]) -> dict[str, Any]:
    """Facts from a turn about one person written on a node whose name is
    another person's (SPEC 11.1: wrong attachments, must be 0). A node named
    otherwise (a speaker word, a name the model invented) is not judged."""
    event_of = {t.key: t.event or t.restates for t in corpus.turns}
    person_name = {p.key: p.name_key for p in corpus.people}
    rows = store.execute(
        "SELECT f.source_id, f.statement, n.name FROM facts f JOIN nodes n ON n.id = f.node_id"
    ).fetchall()
    out: dict[str, Any] = {"checked": 0, "wrong": 0, "examples": []}
    for row in rows:
        turn = turns.get(row["source_id"] or "")
        key = event_of.get(turn) if turn else None
        if key is None:
            continue
        person = corpus.events[key].person
        named = PERSON_FORMS.get(str(row["name"]).casefold())
        if named is None:
            continue
        out["checked"] += 1
        if named != person_name[person]:
            out["wrong"] += 1
            if len(out["examples"]) < 10:
                out["examples"].append(
                    {
                        "turn": turn,
                        "person": person,
                        "node": row["name"],
                        "statement": row["statement"],
                    }
                )
    return out


def neighbours(store: Store, corpus: Corpus, turns: dict[str, str]) -> dict[str, Any]:
    """What the turns of ``--neighbours`` wrote (SPEC 11.1, 5.1): each fact is
    judged by the sentence its excerpt lies in - right on that sentence's file,
    wrong on the other one (must be 0), unjudged when the excerpt lies in no
    sentence of one file alone or the fact has no node - and each file without
    a right fact is missing.
    Fact proposals from those turns are counted (the oracle's: 0)."""
    neighbour_turns = {t.key: t for t in corpus.turns if t.neighbours}
    out: dict[str, Any] = {
        "turns": len(neighbour_turns),
        "expected": 0,
        "right": 0,
        "wrong": 0,
        "unjudged": 0,
        "missing": 0,
        "proposed": 0,
        "examples": [],
    }
    sources = {source: turn for source, turn in turns.items() if turn in neighbour_turns}
    for source_id, key in sources.items():
        text = str(store.get_source(source_id).text or "")
        told = corpus.gold[neighbour_turns[key].text]["facts"]
        spans = []
        for item in told:
            start = text.find(str(item["statement"]))
            spans.append((str(item["subject"]), start, start + len(str(item["statement"]))))
        out["expected"] += len(told)
        right: set[str] = set()
        rows = store.execute(
            "SELECT f.statement, f.excerpt_start, f.excerpt_end, n.name FROM facts f"
            " LEFT JOIN nodes n ON n.id = f.node_id WHERE f.source_id = ?",
            (source_id,),
        ).fetchall()
        for row in rows:
            start, end = row["excerpt_start"], row["excerpt_end"]
            about = (
                [name for name, s, e in spans if s >= 0 and s <= start and end <= e]
                if start is not None and end is not None
                else []
            )
            if len(about) != 1 or row["name"] is None:
                out["unjudged"] += 1  # no file's sentence alone, or on no node at all
            elif str(row["name"] or "").casefold() == about[0].casefold():
                out["right"] += 1
                right.add(about[0])
            else:
                out["wrong"] += 1
                if len(out["examples"]) < 10:
                    out["examples"].append(
                        {"turn": key, "node": row["name"], "statement": row["statement"]}
                    )
        out["missing"] += sum(1 for item in told if str(item["subject"]) not in right)
        out["proposed"] += int(
            store.execute(
                "SELECT COUNT(*) FROM proposals WHERE kind = 'fact' AND source_id = ?",
                (source_id,),
            ).fetchone()[0]
        )
    out["asides"] = asides(store, corpus, turns)
    return out


def without_decay(engine: RecallEngine) -> RecallEngine:
    """``engine`` as it is, but for the decay of observed facts (SPEC 4.2)."""
    twin = copy.copy(engine)
    twin.observed_half_life_days = 0
    return twin


def observed_states(
    store: Store,
    corpus: Corpus,
    turns: dict[str, str],
    engines: Mapping[str, RecallEngine],
    budgets: Sequence[int],
) -> dict[str, Any]:
    """What ``--observed`` left (SPEC 11.1, 4.2), per thing: its newest
    observation ``current`` (live, flagged ``observed``, as of its turn's day),
    the older ones ``stale`` (still live: must be 0), the lasting fact ``kept``
    (still live) or ``closed`` (must be 0), ``missing`` when a turn wrote
    nothing on the thing; then a recall per thing and budget: ``shown`` when the
    newest observation is among the handles with the flag, ``stale_shown`` per
    older observation among them (must be 0), and of those ``stale_flagged``,
    the ones flagged ``newer_pending``, ``newer_observed`` or ``conflict`` (SPEC 8.5).
    An engine named ``*_no_decay`` reads the same store with
    ``observed_half_life_days = 0``."""
    out: dict[str, Any] = {
        "things": 0,
        "current": 0,
        "stale": 0,
        "kept": 0,
        "closed": 0,
        "missing": 0,
        "recall": {},
    }
    view = Visibility.of_scope(ALICE)
    turn_of = {source: key for source, key in turns.items()}
    by_thing: dict[str, list[Any]] = {}
    for turn in corpus.turns:
        if turn.observed is not None:
            by_thing.setdefault(turn.observed, []).append(turn)
    newest: dict[str, str] = {}  # thing -> fact id of its newest observation
    older: dict[str, set[str]] = {}
    for name, told in by_thing.items():
        out["things"] += 1
        node = store.find_node(name, view)
        if node is None:
            out["missing"] += len(told)
            continue
        rows = store.execute(
            "SELECT id FROM facts WHERE node_id = ? ORDER BY created_at, id", (node.id,)
        ).fetchall()
        facts: dict[str, list[Any]] = {}
        for row in rows:
            fact = store.get_fact(row["id"])
            facts.setdefault(turn_of.get(fact.source_id or "", ""), []).append(fact)
        observations = [t for t in told if t.observation is not None]
        last = max(observations, key=lambda t: (t.month, t.observation or 0))
        for turn in told:
            written = facts.get(turn.key) or []
            if not written:
                out["missing"] += 1
                continue
            alive = [f for f in written if f.valid_to is None and f.superseded_by is None]
            if turn.observation is None:
                out["kept" if alive else "closed"] += 1
            elif turn is last:
                day = (turn.occurred_at or "")[:10]
                if any("observed" in f.flags and f.observed_at == day for f in alive):
                    out["current"] += 1
                newest[name] = written[0].id
            else:
                out["stale"] += bool(alive)
                older.setdefault(name, set()).update(f.id for f in written)
    for engine_name, engine in engines.items():
        per_budget: dict[str, dict[str, int]] = {}
        for budget in budgets:
            shown = stale_shown = stale_flagged = 0
            for name, told in by_thing.items():
                query = OBSERVED_QUESTION[told[0].language].format(name=name)
                items = engine.recall(query, view=view, budget_tokens=budget)["items"]
                flagged = {i["id"] for i in items if "observed" in (i.get("flags") or ())}
                shown += newest.get(name) in flagged
                for item in items:
                    if item["id"] in older.get(name, set()):
                        stale_shown += 1
                        flags = set(item.get("flags") or ())
                        stale_flagged += bool(
                            flags & {"newer_pending", "newer_observed", "conflict"}
                        )
            per_budget[str(budget)] = {
                "shown": shown,
                "stale_shown": stale_shown,
                "stale_flagged": stale_flagged,
            }
        out["recall"][engine_name] = per_budget
    return out


def asides(store: Store, corpus: Corpus, turns: dict[str, str]) -> dict[str, Any]:
    """The asides of ``--neighbours`` (SPEC 11.1, 5.4): a project told something
    unrelated to its description, compared with what the turn before it wrote
    on that project. ``other_key``: no key shared with the description, so
    nothing to replace; of those sharing one, ``kept`` (the description still
    live, the ideal answer), ``closed`` (superseded by the aside: must be 0) and
    ``conflict`` (the aside flagged); ``missing``: either side wrote nothing on
    the project. ``keys`` lists, per aside, the keys of both sides."""
    out: dict[str, Any] = {
        "told": 0,
        "kept": 0,
        "closed": 0,
        "conflict": 0,
        "other_key": 0,
        "missing": 0,
        "keys": [],
    }
    source_of = {turn: source for source, turn in turns.items()}
    for index, turn in enumerate(corpus.turns):
        if turn.aside is None:
            continue
        out["told"] += 1
        before = corpus.turns[index - 1]  # the neighbour turn it follows
        said = _facts_on(store, source_of.get(turn.key), turn.aside)
        described = _facts_on(store, source_of.get(before.key), turn.aside)
        out["keys"].append(
            {
                "project": turn.aside,
                "description": sorted({str(f.attribute) for f in described}),
                "aside": sorted({str(f.attribute) for f in said}),
            }
        )
        if not said or not described:
            out["missing"] += 1
            continue
        shared = {f.attribute for f in said} & {f.attribute for f in described} - {None}
        if not shared:
            out["other_key"] += 1
            continue
        ids = {f.id for f in said}
        if any(f.superseded_by in ids for f in described):
            out["closed"] += 1
        elif all(f.valid_to is None for f in described if f.attribute in shared):
            out["kept"] += 1
        out["conflict"] += any("conflict" in f.flags for f in said)
    return out


def _facts_on(store: Store, source_id: str | None, name: str) -> list[Fact]:
    """The facts one source wrote on the node of that name (live or closed)."""
    if source_id is None:
        return []
    rows = store.execute(
        "SELECT f.id FROM facts f JOIN nodes n ON n.id = f.node_id"
        " WHERE f.source_id = ? AND lower(n.name) = lower(?) ORDER BY f.created_at, f.id",
        (source_id, name),
    ).fetchall()
    return [store.get_fact(r["id"]) for r in rows]


def restatements(store: Store, corpus: Corpus, turns: dict[str, str]) -> dict[str, int]:
    """What the restating turns of ``--rephrase`` wrote (SPEC 11.1): facts, those
    in one equivalence group with a fact of the original turn (``same_as``), and
    those flagged as a conflict."""
    original = corpus.turn_of_event()
    rows = store.execute("SELECT id, source_id, same_as, flags FROM facts").fetchall()
    root = {r["id"]: r["same_as"] or r["id"] for r in rows}
    by_turn: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        turn = turns.get(row["source_id"] or "")
        if turn is not None:
            by_turn[turn].append(row)
    out = {"turns": 0, "facts": 0, "linked": 0, "conflict": 0}
    for turn in corpus.turns:
        if turn.restates is None:
            continue
        out["turns"] += 1
        groups = {root[r["id"]] for r in by_turn.get(original.get(turn.restates, ""), [])}
        for row in by_turn.get(turn.key, []):
            out["facts"] += 1
            out["linked"] += int(root[row["id"]] in groups)
            out["conflict"] += int("conflict" in (row["flags"] or "").split(","))
    return out


TRANSLATION_WORD = re.compile(r"\w{3,}")


def translation_check(store: Store, corpus: Corpus, turns: dict[str, str]) -> dict[str, Any]:
    """The narrow check P12 leaves open, measured and never applied (SPEC 11.1):
    a fact whose statement is undetermined, whose quote has a determined
    language, and which shares no word of three letters or more with the quote,
    the subject's names excluded. Split by the turn's language: English, a
    paraphrase (a false alarm); another, a likely translation."""
    language_of_turn = {t.key: t.language for t in corpus.turns}
    out: dict[str, Any] = {"candidates": 0, "flagged": {"turn_en": 0, "turn_other": 0}}
    rows = store.execute(
        "SELECT f.statement, f.excerpt_start, f.excerpt_end, f.node_id, s.text, s.id AS sid"
        " FROM facts f JOIN sources s ON s.id = f.source_id"
        " WHERE f.excerpt_start IS NOT NULL AND s.text IS NOT NULL"
    ).fetchall()
    for row in rows:
        if detect(row["statement"]) != UNDETERMINED:
            continue
        text, start, end = row["text"], row["excerpt_start"], row["excerpt_end"]
        if language_of_span(text, start, end) == UNDETERMINED:
            continue
        out["candidates"] += 1
        names: set[str] = set()
        if row["node_id"]:
            node = store.get_node(row["node_id"])
            for name in (node.name, *node.aliases):
                names.update(w.casefold() for w in TRANSLATION_WORD.findall(name))
        said = {w.casefold() for w in TRANSLATION_WORD.findall(row["statement"])} - names
        quoted = {w.casefold() for w in TRANSLATION_WORD.findall(text[start:end])} - names
        if said & quoted:
            continue
        english = language_of_turn.get(turns.get(row["sid"], ""), "") == "en"
        out["flagged"]["turn_en" if english else "turn_other"] += 1
    return out


def speaker_nodes(store: Store) -> int:
    """Nodes named by a speaker word (SPEC 4.2): there must be none."""
    return sum(
        1 for r in store.execute("SELECT name FROM nodes").fetchall() if is_speaker(r["name"])
    )


def source_turns(store: Store) -> dict[str, str]:
    """source id -> turn key (the turn's key rides as the source's external id)."""
    rows = store.execute(
        "SELECT id, external_id FROM sources WHERE external_id IS NOT NULL"
    ).fetchall()
    return {r["id"]: r["external_id"] for r in rows}


def probe_scores(
    store: Store,
    engines: dict[str, RecallEngine],
    corpus: Corpus,
    budgets: Sequence[int],
    turns: dict[str, str],
) -> dict[str, Any]:
    """One probe set per scope and budget (SPEC 11.2): each question expects the
    facts extracted from the turn of its current event; run through each engine."""
    by_turn: dict[str, list[str]] = defaultdict(list)
    for row in store.execute(
        "SELECT id, source_id FROM facts WHERE source_id IS NOT NULL"
    ).fetchall():
        if row["source_id"] in turns:
            by_turn[turns[row["source_id"]]].append(row["id"])
    event_turn = corpus.turn_of_event()
    out: dict[str, Any] = {}
    for scope in corpus.scopes:
        probes = []
        for q in corpus.questions:
            if q.scope != scope:
                continue
            expected = [f for e in q.expected for f in by_turn.get(event_turn.get(e, ""), [])]
            if expected:
                probes.append((q.query, expected))
        out[scope] = {"probes": len(probes)}
        if not probes:
            continue
        for budget in budgets:
            probe_set = store.create_probe_set(
                scope=scope, name=f"bench {scope} @{budget}", budget_tokens=budget, probes=probes
            )
            for name, engine in engines.items():
                score = run_probe_set(engine, store, probe_set)
                out[scope].setdefault(name, {})[str(budget)] = {
                    "recall": None if score.recall is None else round(score.recall, 3),
                    "used_tokens": round(score.used_tokens, 1),
                    "recall_per_1k_tokens": None
                    if score.recall_per_1k_tokens is None
                    else round(score.recall_per_1k_tokens, 3),
                    "tier": score.tier,
                }
    return out


# --------------------------------------------------------------------------- the run


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="iifa-bench-"))
    try:
        report = run(args, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    report["command"] = " ".join(
        ["python", "bench/run.py", *(sys.argv[1:] if argv is None else argv)]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = args.out / f"{stamp}-{report['extractor']['name']}-seed{args.seed}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nwrote {target}")
    if report["kept_store"]:
        print(f"kept the store at {report['kept_store']} (debugging only: delete it when done)")
    return INTERRUPTED if report["ingestion"]["interrupted"] else 0


def run(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    say: Callable[[str], None] = (
        (lambda _line: None) if args.quiet else (lambda line: print(line, flush=True))
    )
    corpus = generate(
        args.seed,
        months=args.months,
        chatter=args.chatter,
        traps=args.traps,
        rephrase=args.rephrase,
        speaker=args.speaker,
        name_traps=args.name_traps,
        neighbours=args.neighbours,
        intent=args.intent,
        worksession=args.worksession,
        observed=args.observed,
    )
    if args.self_corpus:
        add_self_turns(corpus)
    events = corpus.turns_of_event()
    owners = ((ALICE, OWNER_NAME, ()),) if args.speaker else ()
    if args.self_corpus:  # its scope only: the main corpus is read as without it
        owners = (*owners, (SELF_SCOPE, OWNER_NAME, ()))
    tokenizer = ConservativeTokenizer()
    oracle = Oracle(corpus) if not args.chat_url else None
    chat = (
        Endpoint(
            CHAT, args.chat_url, args.model, args.api_key, timeout=args.timeout, extra=args.extra
        )
        if args.chat_url
        else Endpoint(CHAT, ORACLE_URL, ORACLE_MODEL, timeout=args.timeout)
    )
    embeddings = (
        Endpoint(
            EMBEDDINGS,
            args.embeddings_url,
            args.embeddings_model,
            args.api_key,
            timeout=args.timeout,
        )
        if args.embeddings_url
        else None
    )
    gate = Gate({CHAT: 1, EMBEDDINGS: 1})
    transport = OracleTransport(oracle) if oracle is not None else None
    models = ModelClient(chat, embeddings, gate=gate, transport=transport)
    queries = ModelClient(None, embeddings) if embeddings is not None else None
    dream_models: ModelClient | None = None
    if args.dream:
        dream_url = args.dream_url or args.chat_url
        dream_endpoint = (
            Endpoint(
                CHAT,
                dream_url,
                args.dream_model or args.model,
                args.api_key,
                timeout=args.timeout,
                extra=args.dream_extra,
            )
            if dream_url
            else Endpoint(CHAT, ORACLE_URL, ORACLE_MODEL, timeout=args.timeout)
        )
        dream_models = ModelClient(dream_endpoint, None, gate=gate, transport=transport)
    db_path = args.keep_store or workdir / "bench.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(db_path, STORE_KEY, checkpoint_interval=0)
    pipeline = PipelineConfig(
        answer_max_tokens=args.max_tokens,
        json_mode=args.json_mode,
        owners=owners,
        selves=((SELF_SCOPE, SELF_NAME, ()),) if args.self_corpus else (),
        replacement_question=args.replacement_question,
        restatement_question=args.restatement_question,
        value_norm=args.value_norm,
        intent=args.read_intent,
        resolve_threshold=args.resolve_threshold,
        merge_threshold=args.merge_threshold,
        language_overrides=tuple((scope, args.force_language) for scope in corpus.scopes)
        if args.force_language
        else (),
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
    try:
        extractor = (
            {"name": "oracle", "model": None}
            if oracle is not None
            else {
                "name": "model",
                "model": args.model,
                "url": args.chat_url,
                "json_mode": args.json_mode,
                "extra": args.extra,
            }
        )
        say(
            f"corpus seed {args.seed}: {len(corpus.turns)} turns, {len(corpus.questions)} questions, scopes {', '.join(corpus.scopes)}"
        )
        say(
            f"extractor {extractor['name']}{' ' + args.model if args.model else ''}; embeddings {args.embeddings_model or 'none'}"
        )
        ingestion = ingest(worker, corpus, progress=say)
        turns = source_turns(store)
        ingestion["supersessions"] = supersessions(store, corpus, turns)
        ingestion["name_adjudications"] = name_adjudications(ingestion.pop("adjudications"))
        ingestion["attachments"] = attachments(store, corpus, turns)
        ingestion["speaker_nodes"] = speaker_nodes(store)
        ingestion["translation_check"] = translation_check(store, corpus, turns)
        ingestion["owner"] = {"scope": ALICE, "name": OWNER_NAME} if args.speaker else None
        if args.rephrase:
            ingestion["restatements"] = restatements(store, corpus, turns)
        if args.neighbours:
            ingestion["neighbours"] = neighbours(store, corpus, turns)
        if args.worksession:
            ingestion["worksession"] = worksession_audit(store, corpus, turns)
        engines: dict[str, RecallEngine] = {"iifa_tier2": RecallEngine(store, tokenizer, tier=2)}
        systems: dict[str, System] = {
            "none": no_memory,
            "truncated_history": truncated_history(corpus, tokenizer),
        }
        skipped: dict[str, str] = {}
        vector_build: dict[str, Any] | None = None
        if queries is not None and embeddings is not None:
            systems["vector_topk"], vector_build = vector_topk(
                corpus, tokenizer, queries, embeddings.model
            )
            dense = Dense(store, QueryEmbedder(queries), timeout=args.timeout, cache_size=0)
            engines["iifa_tier3"] = RecallEngine(
                store,
                tokenizer,
                tier=3,
                dense=dense,
                activity_threshold=lambda model: args.activity_threshold,
            )
        else:
            skipped["vector_topk"] = skipped["iifa_tier3"] = (
                "no embeddings endpoint (--embeddings-url)"
            )
        for name, engine in engines.items():
            systems[name] = iifa_system(store, engine, turns)
        if args.observed:
            # SPEC 4.2: the same store read with the decay and without it
            aged = dict(engines)
            for name, engine in engines.items():
                aged[f"{name}_no_decay"] = without_decay(engine)
            ingestion["observed"] = observed_states(store, corpus, turns, aged, args.budgets)
        self_jobs = ingestion.pop("self_jobs")
        if args.self_corpus:
            ingestion["self"] = self_audit(store, corpus, self_jobs, engines, args.budgets, turns)
        say(
            f"answering {len(corpus.questions)} questions x {len(systems)} systems x {len(args.budgets)} budgets"
        )
        results = evaluate(systems, corpus, args.budgets, args.repeat, events)
        probes = probe_scores(store, engines, corpus, args.budgets, turns)
        dream = None
        if dream_models is not None:
            iifa = {name: systems[name] for name in engines}
            probe_engine = engines.get("iifa_tier3", engines["iifa_tier2"])
            thresholds = pipeline.thresholds(embeddings.model) if embeddings else None
            dreamer = Dreamer(
                store,
                client=dream_models,
                config=DreamConfig(
                    answer_max_tokens=args.max_tokens,
                    json_mode=args.json_mode,
                    forced_language=pipeline.forced_language,
                    prompt_max_tokens=args.dream_prompt_tokens,
                    tokenizer=tokenizer,
                ),
                probe_scorer=ProbeTrace(probe_engine, store),
                name_neighbours=name_neighbours(store, embeddings.model, thresholds.resolve)
                if embeddings is not None and thresholds is not None
                else None,
                names_close=names_close(store, embeddings.model, thresholds.merge)
                if embeddings is not None and thresholds is not None
                else None,
            )
            dream = dream_bench(
                dreamer,
                Truth(store, corpus, turns),
                lambda: evaluate(iifa, corpus, args.budgets, 1, events),
                accept=args.dream_accept_review,
                hostile_scope=ALICE if args.hostile else None,
                progress=say,
            )
            if args.self_corpus:
                ingestion["self"]["family_after_dream"] = family_after_dream(
                    store, corpus, engines, args.budgets, turns
                )
            dream["model"] = {
                "url": args.dream_url or args.chat_url,
                "model": args.dream_model or args.model,
                "extra": args.dream_extra,
                "prompt_max_tokens": args.dream_prompt_tokens,
            }
        return {
            "version": 1,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "corpus": corpus.summary(),
            "extractor": extractor,
            "pipeline": {
                "replacement_question": args.replacement_question,
                "restatement_question": args.restatement_question,
                "value_norm": args.value_norm,
                "read_intent": args.read_intent,
                "forced_language": args.force_language,
                "thresholds": {"resolve": args.resolve_threshold, "merge": args.merge_threshold}
                if args.resolve_threshold is not None or args.merge_threshold is not None
                else None,
            },
            "embeddings": {
                "model": args.embeddings_model,
                "url": args.embeddings_url,
                "vector_build": vector_build,
            },
            "tokenizer": tokenizer.name,
            "vector_path": vectors.active_path(),
            "budgets": args.budgets,
            "repeat": args.repeat,
            "ingestion": ingestion,
            "systems": results,
            "skipped": skipped,
            "kept_store": str(args.keep_store) if args.keep_store else None,
            "probes": probes,
            "dream": dream,
        }
    finally:
        worker.stop()
        models.close()
        if dream_models is not None:
            dream_models.close()
        if queries is not None:
            queries.close()
        if transport is not None:
            transport.close()
        store.close()


def print_report(report: dict[str, Any]) -> None:
    ingestion = report["ingestion"]
    print()
    print(
        f"ingestion: {ingestion['processed']}/{ingestion['turns']} turns in {ingestion['seconds']} s"
        f" ({ingestion['processed_per_s']} turns/s), enqueue p50 {ingestion['enqueue_ms']['p50']} ms"
        f" p95 {ingestion['enqueue_ms']['p95']} ms, {ingestion['model_calls']} model calls"
        + (" [INTERRUPTED]" if ingestion["interrupted"] else "")
    )
    if ingestion["extraction"]:
        print("extraction: " + ", ".join(f"{k}={v}" for k, v in ingestion["extraction"].items()))
    if "supersessions" in ingestion:
        audit = ingestion["supersessions"]
        print(
            "supersessions: "
            + ", ".join(f"{k}={v}" for k, v in audit.items() if k != "false_cases")
        )
        for case in audit.get("false_cases") or ():
            closed, by = case["closed"], case["by"]
            print(
                f"  false ({case['cause']}): {closed['turn']} {closed['person']}"
                f" {closed['attribute']} {closed['statement']!r} closed by {by['turn']}"
                f" {by['person']} {by['attribute']} {by['statement']!r}"
            )
    if "name_adjudications" in ingestion:
        names, attached = ingestion["name_adjudications"], ingestion["attachments"]
        print(
            f"name adjudications: {names['mentions']} mentions"
            f" ({names['with_semantic_candidates']} with a name-vector candidate),"
            f" near-name pairs {names['near_name_pairs']}, attached {names['attached']};"
            f" wrong attachments {attached['wrong']} of {attached['checked']} facts"
        )
        for pair in names["pairs"]:
            print("  near-name pair: " + ", ".join(f"{k}={v}" for k, v in pair.items()))
        for example in attached["examples"]:
            print("  wrong attachment: " + ", ".join(f"{k}={v}" for k, v in example.items()))
    if "chatter" in ingestion:
        print(
            "chatter: "
            + ", ".join(f"{k}={v}" for k, v in ingestion["chatter"].items())
            + f", speaker_nodes={ingestion.get('speaker_nodes', 0)}"
        )
    if "translation_check" in ingestion:
        check = ingestion["translation_check"]
        print(
            f"translation check (measured only): {check['candidates']} undetermined statements"
            f" with a determined quote; flagged: English turns {check['flagged']['turn_en']},"
            f" other turns {check['flagged']['turn_other']}"
        )
    if ingestion.get("neighbours"):
        side = ingestion["neighbours"]
        shown = {k: v for k, v in side.items() if k not in ("examples", "asides")}
        print("neighbours: " + ", ".join(f"{k}={v}" for k, v in shown.items()))
        counts = {k: v for k, v in side["asides"].items() if k != "keys"}
        print("asides: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        for example in side["examples"]:
            print(
                "  attached to the other file: " + ", ".join(f"{k}={v}" for k, v in example.items())
            )
    if ingestion.get("observed"):
        seen = ingestion["observed"]
        counts = {k: v for k, v in seen.items() if k != "recall"}
        print("observed: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        for engine_name, per_budget in seen["recall"].items():
            shown = ", ".join(
                f"@{b} shown={r['shown']} stale={r['stale_shown']} flagged={r['stale_flagged']}"
                for b, r in per_budget.items()
            )
            print(f"  {engine_name}: {shown}")
    if ingestion.get("self"):
        agent = ingestion["self"]
        for category, row in agent["categories"].items():
            print(f"self {category}: " + ", ".join(f"{k}={v}" for k, v in row.items()))
        print(
            f"self: hostile_written={agent['hostile_written']}"
            f" unmeant_written={agent['unmeant_written']}"
            f" not_sincere={agent['assistant_turns_not_sincere']}"
            f" counters={agent['counters']} profile={agent['profile_commitments']}"
        )
        for activity in agent.get("activities") or ():
            print(
                f"  activity {activity['name']!r}: linked={activity['linked']}"
                f" rules={activity['rules']}"
            )
        for when in ("family", "family_after_dream"):
            family = agent.get(when) or {}
            for parent, children in (family.get("groups") or {}).items():
                print(f"  {when} group {parent!r}: {children}")
            for engine_name, per_budget in (family.get("rules_shown") or {}).items():
                shown = ", ".join(f"@{b} {r}" for b, r in per_budget.items())
                print(f"  {when} rules {engine_name}: {shown}")
        vectors = agent.get("vector_probes") or {}
        for key, found in (vectors.get("scores") or {}).items():
            shown = ", ".join(
                f"{f['activity']!r}={f['cosine']}{' (word)' if f['word'] else ''}" for f in found
            )
            print(f"  vectors {key}: {shown or '-'}")
        for threshold, row in (vectors.get("sweep") or {}).items():
            print(
                f"  activity_threshold {threshold}: positives={row['positives']}"
                f" negatives_with_block={row['negatives_with_block']}/{vectors['negatives']}"
            )
        for engine_name, per_budget in agent["probes"].items():
            shown = ", ".join(
                f"@{b} transfer={r['transfer_shown']} rules={r['rules_shown']}/{r['rules_total']}"
                f" limits={r['limits_shown']} rules_first={r['rules_first']}"
                f" activity={r['activity_rules_shown']}/{r['activity_rules_total']}"
                f" activity_first={r['activity_first']}"
                for b, r in per_budget.items()
            )
            print(f"  {engine_name}: {shown}")
    for record in ingestion.get("owner_turns") or ():
        print("owner turn: " + ", ".join(f"{k}={v}" for k, v in record.items()))
    if ingestion.get("restatements"):
        print(
            "restatements: " + ", ".join(f"{k}={v}" for k, v in ingestion["restatements"].items())
        )
    intent = ingestion.get("intent") or {}
    if (intent.get("unmeant") or {}).get("turns"):
        print("not meant: " + ", ".join(f"{k}={v}" for k, v in intent["unmeant"].items()))
        print("controls: " + ", ".join(f"{k}={v}" for k, v in intent["controls"].items()))
        print(f"dropped by intent: {intent['dropped_by_intent']}")
    if ingestion.get("worksession"):
        print("work session: " + ", ".join(f"{k}={v}" for k, v in ingestion["worksession"].items()))
    for budget in report["budgets"]:
        print(f"\nbudget {budget} tokens")
        print(
            f"  {'system':<18}{'correct':>8}{'/1k tok':>9}{'recall':>8}{'/1k tok':>9}{'tokens':>8}{'contra':>9}{'same':>7}{'cross':>7}{'p50 ms':>9}{'p95 ms':>9}{'leaks':>7}{'dups':>6}"
        )
        for name, per_budget in report["systems"].items():
            row = per_budget[str(budget)]
            c = row["contradictions"]

            def acc(kind: str, c: dict[str, Any] = c) -> str:
                value = c[kind]["accuracy"]
                return "-" if value is None else f"{value:.2f}"

            print(
                f"  {name:<18}{row['correct_rate']:>8.2f}{row['correct_per_1k_tokens']:>9.2f}"
                f"{row['recall']:>8.2f}{row['recall_per_1k_tokens']:>9.2f}{row['mean_tokens']:>8.0f}"
                f"{acc('all'):>9}{acc('same_language'):>7}{acc('cross_language'):>7}"
                f"{row['latency_ms']['p50']:>9.2f}{row['latency_ms']['p95']:>9.2f}{row['leaks']:>7}"
                f"{row['duplicates']:>6}"
            )
    for name, reason in report["skipped"].items():
        print(f"  {name}: skipped, {reason}")
    print("\nprobe sets (recall per scope and budget):")
    for scope, per in report["probes"].items():
        cells = []
        for engine, scores in per.items():
            if engine == "probes":
                continue
            cells.append(engine + " " + " ".join(f"{b}:{s['recall']}" for b, s in scores.items()))
        print(f"  {scope} ({per['probes']} probes): {'; '.join(cells) or '-'}")
    if report.get("dream"):
        print_dream(report["dream"])


def print_dream(dream: dict[str, Any]) -> None:
    """SPEC 11.1 ``--dream``: each run, the judged totals, the headline before and after."""
    print("\ndream:")
    for run in dream["runs"]:
        kind = "dry run" if run["dry_run"] else "real run"
        adjudications = run["adjudications"]
        print(
            f"  {run['scope']} {kind}: {run['status']}, stopped by {run['stopped_by']},"
            f" {run['units']} units, {run['calls']} calls, {run['tokens']} tokens,"
            f" {run['seconds']} s"
        )
        print(f"    proposed {run['proposed']}; operations {run['operations']}")
        print(f"    rejected {run['rejected']}; dropped {run['dropped']}; skipped {run['skipped']}")
        print(f"    prompts {run['prompts']}")
        print(
            f"    adjudications: {adjudications.get('calls')} calls, verdicts"
            f" {adjudications.get('verdicts')}, answers {adjudications.get('answers')},"
            f" skipped {adjudications.get('skipped')}"
        )
        for pair in adjudications.get("pairs") or ():
            reason = f" - {pair['reason']}" if pair.get("reason") else ""
            said = pair.get("model_verdict")
            unfit = f" (the model said {said}; the names do not fit)" if said else ""
            print(
                f"      {' / '.join(pair['names'])}: {pair['verdict']}{unfit}"
                f" (truth {pair['truth']}){reason}"
            )
        for name in ("merges", "consolidations", "supersessions"):
            counts = {k: v for k, v in run[name].items() if k != "cases"}
            print(f"    {name}: {counts}")
            for case in run[name]["cases"]:
                print(f"      {case}")
        if run.get("rolled_back"):
            print(f"    ROLLED BACK: {run['rolled_back']}")
        for loss in run.get("probe_losses") or ():
            print(
                f"    lost @{loss['budget']} {loss['probe']} {loss['query']!r}: {loss['lost']}"
                f" before {loss['before']} after {loss['after']}"
            )
        if run.get("probes", {}).get("measured"):
            for entry in run["probes"]["sets"]:
                print(
                    f"    probes {entry['set_id']}: {entry['before']} -> {entry['after']}"
                    f" ({entry['verdict']})"
                )
        if "hostile" in run:
            print(f"    hostile: {run['hostile']}")
    print(f"  totals, dry run: {dream['dry']}")
    print(f"  totals, real run: {dream['real']}")

    def table(label: str, after: dict[str, Any]) -> None:
        for system, per in after.items():
            cells = []
            for budget, row in per.items():
                was = dream["before"][system][budget]
                cells.append(
                    f"@{budget} correct {was['correct']}->{row['correct']}"
                    f" /1k {was['correct_per_1k_tokens']}->{row['correct_per_1k_tokens']}"
                )
            print(f"  {label} {system}: " + "; ".join(cells))

    table("headline (auto-applied)", dream["after"])
    for system, per in dream["changed"].items():
        for budget, moved in per.items():
            if moved["to_correct"] or moved["to_wrong"]:
                print(f"    {system} @{budget}: {moved}")
    if dream.get("accept_review"):
        ceiling = dream["accept_review"]
        print(f"  accepted every queued operation: {ceiling['decided']}")
        table("CEILING (all review accepted, not what a user gets)", ceiling["after"])


if __name__ == "__main__":
    raise SystemExit(main())
