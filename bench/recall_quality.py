"""Tier 3 against tier 0 on a small multilingual corpus (the delta tier 3 adds),
with a real OpenAI-compatible embeddings endpoint named on the command line.

    python bench/recall_quality.py --embeddings-url http://127.0.0.1:11434/v1 \\
        --model bge-m3 --json bench-bge-m3.json
    python bench/recall_quality.py --embeddings-url http://127.0.0.1:11434/v1 \\
        --model qwen3-embedding:0.6b --json bench-qwen3-embedding.json

What it does, in a throwaway encrypted store in a temporary directory:

1. writes the corpus below (facts in English, French, Russian, Arabic, Chinese
   and Japanese, plus distractors), then embeds every record in batches through
   the service's own client and store (the vector build time);
2. asks every probe (a question whose answer is written in another language, or
   names its subject in another script) at tier 0 (lexical + graph) and at tier
   3 (fused with dense), and reports recall@1/3/5 and MRR for each;
3. times recall at both tiers (``--repeat`` rounds, the query cache off, so
   every tier-3 recall pays its embedding call): p50 and p95;
4. prints the cosine between the names of one person across scripts (Karim,
   Карим, كريم, 卡里姆, カリム) and between different people, and the same for
   one project name across languages: what the thresholds of SPEC 6.1
   (resolve, merge, link) should sit between.

``--json`` writes every number with the exact command line, so a comparison
between models can be kept. Nothing else is kept. Content is printed on purpose
(a manual run for the operator's eyes).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iifa import vectors
from iifa.llm import EMBEDDINGS, Endpoint, ModelClient, QueryEmbedder
from iifa.recall.dense import Dense
from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

SCOPE = "user:bench"
KEY = "bench-throwaway-key"
KS = (1, 3, 5)

# key -> statement; every probe's answer is in a language or script other than the question's
CORPUS: dict[str, str] = {
    "karim_fr": "Karim dirige la planification de l'installation du projet solaire depuis mars 2026.",
    "karim_ru": "Карим руководит планированием установки солнечного проекта с марта 2026 года.",
    "karim_ar": "كريم يقود تخطيط تركيب المشروع الشمسي منذ مارس 2026.",
    "karim_zh": "卡里姆自2026年3月起负责太阳能项目的安装规划。",
    "emilie_fr": "Émilie préfère les réponses courtes et déteste les réunions tôt le matin.",
    "nadia_en": "Nadia manages the budget of the renovation and signs every invoice.",
    "ivan_ru": "Иван живёт в Лионе и ездит на работу на велосипеде.",
    "lea_fr": "Léa est allergique aux arachides et évite les restaurants thaïs.",
    "youssef_ar": "يوسف يتعلم اللغة اليابانية كل مساء.",
    "meiling_zh": "美玲每周六去游泳。",
    "kenji_ja": "健二は毎朝コーヒーを二杯飲む。",
    "sofia_en": "Sofia moved her team meeting to Thursday afternoons.",
    "noise_en": "The office printer is broken again and nobody knows why.",
    "noise_fr": "Le parking de l'immeuble ferme à vingt heures.",
    "noise_ru": "Погода в Лионе сегодня дождливая и холодная.",
    "noise_ja": "駅前の本屋は来月閉店する。",
}
KARIM = ("karim_fr", "karim_ru", "karim_ar", "karim_zh")

PROBES: list[tuple[str, tuple[str, ...]]] = [
    ("Who leads the solar installation planning?", KARIM),
    ("太阳能项目由谁负责？", KARIM),
    ("What does Emilie dislike in the morning?", ("emilie_fr",)),
    ("Who avoids early meetings?", ("emilie_fr",)),
    ("Qui gère le budget de la rénovation ?", ("nadia_en",)),
    ("Where does Ivan live?", ("ivan_ru",)),
    ("How does Ivan get to work?", ("ivan_ru",)),
    ("Which food is Lea allergic to?", ("lea_fr",)),
    ("Quelle langue Youssef apprend-il ?", ("youssef_ar",)),
    ("Who goes swimming on Saturdays?", ("meiling_zh",)),
    ("How much coffee does Kenji drink?", ("kenji_ja",)),
    ("Когда проходит собрание команды Софии?", ("sofia_en",)),
]

SAME_PERSON = ["Karim", "Карим", "كريم", "卡里姆", "カリム"]
OTHER_PEOPLE = ["Nadia", "Иван", "Émilie", "يوسف", "美玲"]
SAME_PROJECT = [
    "solar project",
    "projet solaire",
    "солнечный проект",
    "المشروع الشمسي",
    "太阳能项目",
]
OTHER_THINGS = ["office printer", "renovation budget", "team meeting"]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--embeddings-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="embedding model, explicit on every call")
    parser.add_argument("--api-key", default="", help="bearer key, if the endpoint needs one")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per embedding call")
    parser.add_argument("--batch", type=int, default=32, help="texts per embedding call")
    parser.add_argument("--repeat", type=int, default=5, help="latency rounds per probe")
    parser.add_argument("--json", type=Path, default=None, help="write every number here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="iifa-recall-quality-"))
    try:
        report = run(args, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    report["command"] = " ".join(["python", "bench/recall_quality.py", *(argv or sys.argv[1:])])
    if args.json is not None:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def run(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    endpoint = Endpoint(EMBEDDINGS, args.embeddings_url, args.model, args.api_key, args.timeout)
    client = ModelClient(None, endpoint)
    store = Store(workdir / "bench.db", KEY, checkpoint_interval=0)
    view = Visibility.of_scope(SCOPE)
    try:
        ids = {
            key: store.add_fact(scopes=[SCOPE], statement=text, origin="explicit").id
            for key, text in CORPUS.items()
        }
        build = build_vectors(store, client, args.model, args.batch)
        tokenizer = ConservativeTokenizer()
        tier0 = RecallEngine(store, tokenizer, tier=0)
        dense = Dense(store, QueryEmbedder(client), timeout=args.timeout, cache_size=0)
        tier3 = RecallEngine(store, tokenizer, tier=3, dense=dense)
        print(f"endpoint {args.embeddings_url} model {args.model}")
        print(
            f"vectors: {build['records']} records, dimension {build['dimension']},"
            f" {build['seconds']:.2f} s ({build['calls']} calls), path {vectors.active_path()}"
        )
        quality = {"tier0": score(tier0, view, ids), "tier3": score(tier3, view, ids)}
        print_quality(quality)
        latency = {
            "tier0": timing(tier0, view, args.repeat),
            "tier3": timing(tier3, view, args.repeat),
        }
        for tier, numbers in latency.items():
            print(f"latency {tier}: p50 {numbers['p50_ms']:.1f} ms, p95 {numbers['p95_ms']:.1f} ms")
        names = name_similarities(client, args.model)
        print_names(names)
        return {
            "model": args.model,
            "endpoint": args.embeddings_url,
            "vector_path": vectors.active_path(),
            "build": build,
            "quality": quality,
            "latency": latency,
            "names": names,
        }
    finally:
        client.close()
        store.close()


def build_vectors(store: Store, client: ModelClient, model: str, batch: int) -> dict[str, Any]:
    started = time.perf_counter()
    records = calls = dimension = 0
    while True:
        pending = store.pending_vectors(model, limit=batch)
        if not pending:
            break
        result = client.embed([p.text for p in pending], model=model)
        for item, values in zip(pending, result.vectors, strict=True):
            store.put_vector(item.record_id, item.facet, model=model, text=item.text, values=values)
        records += len(pending)
        calls += 1
        dimension = result.dimension
    return {
        "records": records,
        "calls": calls,
        "dimension": dimension,
        "seconds": round(time.perf_counter() - started, 3),
    }


def score(engine: RecallEngine, view: Visibility, ids: dict[str, str]) -> dict[str, Any]:
    """recall@k and MRR over the probes: a probe is found at the rank of the first
    expected fact among the recalled facts."""
    ranks: list[int | None] = []
    per_probe: list[dict[str, Any]] = []
    for query, expected in PROBES:
        result = engine.recall(query, view=view, budget_tokens=4000)
        facts = [item["id"] for item in result["items"] if item["kind"] == "fact"]
        wanted = {ids[key] for key in expected}
        rank = next((i + 1 for i, fact_id in enumerate(facts) if fact_id in wanted), None)
        ranks.append(rank)
        per_probe.append({"query": query, "rank": rank, "tier": result["tier"]})
    out: dict[str, Any] = {
        f"recall@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 3)
        for k in KS
    }
    out["mrr"] = round(statistics.fmean(1 / r if r else 0.0 for r in ranks), 3)
    out["probes"] = per_probe
    return out


def timing(engine: RecallEngine, view: Visibility, repeat: int) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(max(1, repeat)):
        for query, _expected in PROBES:
            started = time.perf_counter()
            engine.recall(query, view=view, budget_tokens=4000)
            samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    p95 = samples[min(len(samples) - 1, round(0.95 * (len(samples) - 1)))]
    return {
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(p95, 2),
        "n": len(samples),
    }


def name_similarities(client: ModelClient, model: str) -> dict[str, Any]:
    texts = [*SAME_PERSON, *OTHER_PEOPLE, *SAME_PROJECT, *OTHER_THINGS]
    result = client.embed(texts, model=model)
    unit = {text: vectors.normalise(v) for text, v in zip(texts, result.vectors, strict=True)}

    def cos(a: str, b: str) -> float:
        return round(sum(x * y for x, y in zip(unit[a], unit[b], strict=True)), 4)

    anchor, project = SAME_PERSON[0], SAME_PROJECT[0]
    same_person = {name: cos(anchor, name) for name in SAME_PERSON[1:]}
    other_people = {name: cos(anchor, name) for name in OTHER_PEOPLE}
    same_project = {name: cos(project, name) for name in SAME_PROJECT[1:]}
    other_things = {name: cos(project, name) for name in OTHER_THINGS}
    return {
        "anchor": anchor,
        "same_person": same_person,
        "other_people": other_people,
        "project_anchor": project,
        "same_project": same_project,
        "other_things": other_things,
        "gap_person": round(min(same_person.values()) - max(other_people.values()), 4),
        "gap_project": round(min(same_project.values()) - max(other_things.values()), 4),
    }


def print_quality(quality: dict[str, Any]) -> None:
    print()
    print(f"{'':8}" + "".join(f"{f'recall@{k}':>11}" for k in KS) + f"{'MRR':>8}")
    for tier, numbers in quality.items():
        cells = "".join(f"{numbers[f'recall@{k}']:>11.2f}" for k in KS)
        print(f"{tier:8}{cells}{numbers['mrr']:>8.2f}")
    print()
    for zero, three in zip(quality["tier0"]["probes"], quality["tier3"]["probes"], strict=True):
        print(
            f"  {zero['query']!s:48.48} tier0 rank {zero['rank']!s:>4}  tier3 rank {three['rank']!s:>4}"
        )
    print()


def print_names(names: dict[str, Any]) -> None:
    print()
    print(f"cosine to {names['anchor']!r}: same person {names['same_person']}")
    print(f"                    other people {names['other_people']}")
    print(f"  gap (lowest same - highest other): {names['gap_person']}")
    print(f"cosine to {names['project_anchor']!r}: same project {names['same_project']}")
    print(f"                    other things {names['other_things']}")
    print(f"  gap (lowest same - highest other): {names['gap_project']}")


if __name__ == "__main__":
    raise SystemExit(main())
