"""Calibrate the ``resolve`` and ``merge`` thresholds (SPEC 6.1) for one
embedding model, on names that are NOT in the bench corpus, then verify them
on the corpus's names.

    python bench/thresholds.py --embeddings-url http://127.0.0.1:11434/v1 \\
        --model bge-m3 --json thresholds-bge-m3.json

Cosines are taken between name vectors as the service embeds them (a node's
name). Positive pairs are one entity in two scripts, adapted forms included
(Joseph / Иосиф: what transliteration cannot do); negative pairs are two
entities of the same kind; traps are two entities with near names. The rule:
``merge`` keeps every trap and all but 2 % of the negatives below it (it only
makes a proposal, reviewed later); ``resolve`` is the 95th percentile of the
negatives, never above ``merge``; no profile only when ``merge`` would catch
no positive at all. ``link`` is not calibrated here (no ground truth
for related subjects).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.synthetic import CITIES, NAMES, TEAMS

from iifa import vectors
from iifa.calibration import PROFILES, profile_key
from iifa.llm import EMBEDDINGS, Endpoint, ModelClient

NEGATIVES_ABOVE_MERGE = 0.02  # a false merge proposal costs a review, not a wrong memory
RESOLVE_PERCENTILE = 0.95
BATCH = 64

# --------------------------------------------------------------------------- calibration names
# Not in the bench corpus (checked by a test): the thresholds must say something
# about a real user's names, not about the nine people they would be tuned on.

PEOPLE: dict[str, dict[str, str]] = {
    "olivia": {"en": "Olivia", "ru": "Оливия", "ar": "أوليفيا", "zh": "奥利维亚", "ja": "オリビア"},
    "dmitri": {
        "en": "Dmitri",
        "ru": "Дмитрий",
        "ar": "ديمتري",
        "zh": "德米特里",
        "ja": "ドミトリー",
    },
    "fatima": {"en": "Fatima", "ru": "Фатима", "ar": "فاطمة", "zh": "法蒂玛", "ja": "ファティマ"},
    "hiroshi": {"en": "Hiroshi", "ru": "Хироси", "ar": "هيروشي", "zh": "浩", "ja": "ヒロシ"},
    "lucas": {"en": "Lucas", "ru": "Лукас", "ar": "لوكاس", "zh": "卢卡斯", "ja": "ルーカス"},
    "zainab": {
        "en": "Zainab",
        "fr": "Zaynab",
        "ru": "Зейнаб",
        "ar": "زينب",
        "zh": "宰纳卜",
        "ja": "ザイナブ",
    },
    "mateo": {"en": "Mateo", "ru": "Матео", "ar": "ماتيو", "zh": "马特奥", "ja": "マテオ"},
    "ingrid": {
        "en": "Ingrid",
        "ru": "Ингрид",
        "ar": "إنغريد",
        "zh": "英格丽德",
        "ja": "イングリッド",
    },
    "omar": {"en": "Omar", "ru": "Омар", "ar": "عمر", "zh": "奥马尔", "ja": "オマール"},
    "svetlana": {
        "en": "Svetlana",
        "ru": "Светлана",
        "ar": "سفيتلانا",
        "zh": "斯维特兰娜",
        "ja": "スヴェトラーナ",
    },
}
# adapted forms of one name: what transliteration cannot do and embeddings should
ADAPTED_PEOPLE: list[tuple[str, ...]] = [
    ("Joseph", "Иосиф"),
    ("Mary", "Мария"),
    ("Peter", "Pierre", "Пётр"),
    ("Paul", "Павел"),
    ("George", "Георгий"),
    ("Catherine", "Екатерина"),
    ("Michael", "Михаил"),
    ("William", "Guillaume"),
    ("James", "Jacques"),
    ("Stephen", "Étienne"),
    ("Abraham", "إبراهيم"),
    ("David", "داود"),
]
PLACES: dict[str, dict[str, str]] = {
    "madrid": {"en": "Madrid", "ru": "Мадрид", "ar": "مدريد", "zh": "马德里", "ja": "マドリード"},
    "rome": {"en": "Rome", "ru": "Рим", "ar": "روما", "zh": "罗马", "ja": "ローマ"},
    "vienna": {
        "en": "Vienna",
        "fr": "Vienne",
        "ru": "Вена",
        "ar": "فيينا",
        "zh": "维也纳",
        "ja": "ウィーン",
    },
    "cairo": {
        "en": "Cairo",
        "fr": "Le Caire",
        "ru": "Каир",
        "ar": "القاهرة",
        "zh": "开罗",
        "ja": "カイロ",
    },
    "seoul": {
        "en": "Seoul",
        "fr": "Séoul",
        "ru": "Сеул",
        "ar": "سول",
        "zh": "首尔",
        "ja": "ソウル",
    },
    "lisbon": {
        "en": "Lisbon",
        "fr": "Lisbonne",
        "ru": "Лиссабон",
        "ar": "لشبونة",
        "zh": "里斯本",
        "ja": "リスボン",
    },
    "prague": {"en": "Prague", "ru": "Прага", "ar": "براغ", "zh": "布拉格", "ja": "プラハ"},
    "warsaw": {
        "en": "Warsaw",
        "fr": "Varsovie",
        "ru": "Варшава",
        "ar": "وارسو",
        "zh": "华沙",
        "ja": "ワルシャワ",
    },
}
GROUPS: dict[str, dict[str, str]] = {
    "phoenix": {
        "en": "Phoenix",
        "fr": "Phénix",
        "ru": "Феникс",
        "ar": "فينيكس",
        "zh": "凤凰",
        "ja": "フェニックス",
    },
    "aurora": {
        "en": "Aurora",
        "fr": "Aurore",
        "ru": "Аврора",
        "ar": "أورورا",
        "zh": "奥罗拉",
        "ja": "オーロラ",
    },
    "vega": {"en": "Vega", "fr": "Véga", "ru": "Вега", "ar": "فيغا", "zh": "维加", "ja": "ベガ"},
}
TRAPS: list[tuple[str, str]] = [  # two entities, near names: never one merge proposal
    ("Daniel", "Daniela"),
    ("Mario", "Maria"),
    ("Paolo", "Paola"),
    ("Alexander", "Alexandra"),
    ("Bogdan", "Bogdana"),
    ("Jan", "Jana"),
    ("Даниэль", "Daniela"),
    ("Александр", "Alexandra"),
    ("Austria", "Australia"),
    ("Iran", "Iraq"),
    ("Slovenia", "Slovakia"),
    ("Niger", "Nigeria"),
    ("Mali", "Malawi"),
]
ADAPTED: list[tuple[str, str]] = [
    *((a, b) for group in ADAPTED_PEOPLE for a, b in combinations(group, 2)),
    ("Vienna", "Вена"),
    ("Vienna", "ウィーン"),
    ("Cairo", "القاهرة"),
    ("Cairo", "Le Caire"),
    ("Rome", "Рим"),
    ("Warsaw", "Varsovie"),
]

# --------------------------------------------------------------------------- verification names (the corpus)

CORPUS_TRAPS: list[tuple[str, str]] = [
    ("Karim", "Karima"),
    ("Ivan", "Ivana"),
    ("Nadia", "Nadine"),
    ("Tunis", "Tunisia"),
]
CORPUS_ADAPTED: list[tuple[str, str]] = [
    ("Alice", "Алиса"),
    ("Émilie", "Эмили"),
    ("Youssef", "Юсеф"),
    ("Casablanca", "الدار البيضاء"),
    ("Osaka", "大阪"),
    ("Kenji", "健二"),
]


def _groups(table: dict[str, dict[str, str]]) -> list[list[str]]:
    return [list(dict.fromkeys(forms.values())) for forms in table.values()]


def calibration_groups() -> list[list[list[str]]]:
    """Kinds of entities, each a list of one entity's forms."""
    return [_groups(PEOPLE) + [list(g) for g in ADAPTED_PEOPLE], _groups(PLACES), _groups(GROUPS)]


def corpus_groups() -> list[list[list[str]]]:
    return [_groups(NAMES), _groups(CITIES), _groups(TEAMS)]


def _forms(kinds: list[list[list[str]]], traps: Iterable[tuple[str, str]]) -> list[str]:
    out = [form for kind in kinds for group in kind for form in group]
    out.extend(form for pair in traps for form in pair)
    return list(dict.fromkeys(out))


def calibration_forms() -> list[str]:
    return _forms(calibration_groups(), TRAPS)


def corpus_forms() -> list[str]:
    return _forms(corpus_groups(), CORPUS_TRAPS)


def _pairs(kinds: list[list[list[str]]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Positives (one entity, two forms) and negatives (two entities of a kind)."""
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []
    for kind in kinds:
        for group in kind:
            positives.extend(combinations(group, 2))
        for left, right in combinations(kind, 2):
            negatives.extend((a, b) for a in left for b in right)
    return positives, negatives


# --------------------------------------------------------------------------- the rule


def _grid_above(value: float) -> float:
    """The next 0.01 step strictly above ``value``."""
    return (math.floor(value * 100 + 1e-9) + 1) / 100


def choose(
    *, positives: Sequence[float], negatives: Sequence[float], traps: Sequence[float]
) -> dict[str, Any]:
    """``merge``: every trap and all but 2 % of the negatives below it; ``resolve``:
    the 95th percentile of the negatives, never above ``merge``; no profile only
    when ``merge`` would catch no positive (SPEC 6.1)."""
    lowest, worst_trap = min(positives), max(traps)
    ordered = sorted(negatives)
    out: dict[str, Any] = {
        "lowest_positive": round(lowest, 4),
        "highest_trap": round(worst_trap, 4),
        "highest_negative": round(ordered[-1], 4),
    }
    allowed = math.floor(NEGATIVES_ABOVE_MERGE * len(ordered))
    floor_negative = ordered[len(ordered) - allowed - 1]
    merge = _grid_above(max(worst_trap, floor_negative))
    if not any(p >= merge for p in positives):
        return {**out, "profile": False, "merge": None, "resolve": None}
    resolve = round(
        min(ordered[min(len(ordered) - 1, int(RESOLVE_PERCENTILE * len(ordered)))], merge), 2
    )
    return {
        **out,
        "profile": True,
        "merge": merge,
        "resolve": resolve,
        **rates(positives, negatives, traps, merge, resolve),
    }


def rates(
    positives: Sequence[float],
    negatives: Sequence[float],
    traps: Sequence[float],
    merge: float,
    resolve: float,
) -> dict[str, Any]:
    def share(values: Sequence[float], threshold: float) -> float:
        return round(sum(1 for v in values if v >= threshold) / len(values), 4) if values else 0.0

    return {
        "negatives_above_merge": share(negatives, merge),
        "negatives_above_resolve": share(negatives, resolve),
        "positives_at_merge": share(positives, merge),
        "positives_at_resolve": share(positives, resolve),
        "traps_above_merge": sum(1 for t in traps if t >= merge),
    }


# --------------------------------------------------------------------------- the command


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--embeddings-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="embedding model, explicit on every call")
    parser.add_argument("--api-key", default="", help="bearer key, if the endpoint needs one")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per call")
    parser.add_argument("--json", type=Path, default=None, help="write every number here")
    return parser.parse_args(argv)


def embed_all(client: ModelClient, model: str, texts: Sequence[str]) -> dict[str, list[float]]:
    unit: dict[str, list[float]] = {}
    for start in range(0, len(texts), BATCH):
        chunk = list(texts[start : start + BATCH])
        result = client.embed(chunk, model=model)
        unit.update({t: vectors.normalise(v) for t, v in zip(chunk, result.vectors, strict=True)})
    return unit


def measure(client: ModelClient, model: str) -> dict[str, Any]:
    texts = sorted(set(calibration_forms()) | set(corpus_forms()))
    unit = embed_all(client, model, texts)

    def cos(a: str, b: str) -> float:
        return round(sum(x * y for x, y in zip(unit[a], unit[b], strict=True)), 4)

    def stats(kinds: list[list[list[str]]], traps: list[tuple[str, str]]) -> dict[str, Any]:
        positives, negatives = _pairs(kinds)
        return {
            "positives": [cos(a, b) for a, b in positives],
            "negatives": [cos(a, b) for a, b in negatives],
            "traps": {f"{a} ~ {b}": cos(a, b) for a, b in traps},
            "worst_negatives": sorted(
                ({"pair": f"{a} ~ {b}", "cosine": cos(a, b)} for a, b in negatives),
                key=lambda p: -p["cosine"],
            )[:10],
            "worst_positives": sorted(
                ({"pair": f"{a} ~ {b}", "cosine": cos(a, b)} for a, b in positives),
                key=lambda p: p["cosine"],
            )[:10],
        }

    calibration = stats(calibration_groups(), TRAPS)
    verification = stats(corpus_groups(), CORPUS_TRAPS)
    chosen = choose(
        positives=calibration["positives"],
        negatives=calibration["negatives"],
        traps=list(calibration["traps"].values()),
    )
    merge, resolve = chosen["merge"], chosen["resolve"]

    def caught(pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        out = []
        for a, b in pairs:
            value = cos(a, b)
            out.append(
                {
                    "a": a,
                    "b": b,
                    "cosine": value,
                    "resolve": resolve is not None and value >= resolve,
                    "merge": merge is not None and value >= merge,
                }
            )
        return out

    checked: dict[str, Any] = {}
    if chosen["profile"]:
        checked = rates(
            verification["positives"],
            verification["negatives"],
            list(verification["traps"].values()),
            merge,
            resolve,
        )
    profile = PROFILES.get(profile_key(model))
    current: dict[str, Any] = {}
    for name, section in (("calibration", calibration), ("verification", verification)):
        trap_values = list(section["traps"].values())
        # the most a merge threshold above every trap could catch
        section["positives_above_highest_trap"] = round(
            sum(1 for p in section["positives"] if p > max(trap_values))
            / len(section["positives"]),
            4,
        )
        if profile is not None:
            current[name] = rates(
                section["positives"],
                section["negatives"],
                trap_values,
                profile.merge,
                profile.resolve,
            )
    if profile is not None:
        current["profile"] = {"resolve": profile.resolve, "merge": profile.merge}
    for section in (calibration, verification):
        section["positive_count"] = len(section.pop("positives"))
        section["negative_count"] = len(section.pop("negatives"))
    return {
        "model": model,
        "calibration": calibration,
        "verification": {**verification, "at_chosen": checked},
        "chosen": chosen,
        "current": current,
        "adapted": caught(ADAPTED),
        "corpus_adapted": caught(CORPUS_ADAPTED),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    endpoint = Endpoint(EMBEDDINGS, args.embeddings_url, args.model, args.api_key, args.timeout)
    client = ModelClient(None, endpoint)
    try:
        report = measure(client, args.model)
    finally:
        client.close()
    report["command"] = " ".join(
        ["python", "bench/thresholds.py", *(sys.argv[1:] if argv is None else argv)]
    )
    chosen = report["chosen"]
    print(f"model {report['model']}")
    print(
        f"calibration: lowest positive {chosen['lowest_positive']}, highest trap"
        f" {chosen['highest_trap']}, highest negative {chosen['highest_negative']}"
    )
    if chosen["profile"]:
        print(
            f"chosen: merge {chosen['merge']} (negatives above {chosen['negatives_above_merge']},"
            f" positives at {chosen['positives_at_merge']}), resolve {chosen['resolve']}"
            f" (positives at {chosen['positives_at_resolve']})"
        )
        print(f"verification on the corpus: {report['verification']['at_chosen']}")
    else:
        print(
            "chosen: NO profile - merge above every trap catches no positive (merge and resolve off)"
        )
    for title, key in (("adapted forms", "adapted"), ("corpus adapted forms", "corpus_adapted")):
        print(title)
        for pair in report[key]:
            print(
                f"  {pair['cosine']:6.3f}  {pair['a']} ~ {pair['b']}"
                f"  resolve={pair['resolve']} merge={pair['merge']}"
            )
    if args.json is not None:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
