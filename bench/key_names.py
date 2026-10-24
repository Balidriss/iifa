"""Calibrate ``[embeddings] key_threshold`` (SPEC 5.4 step 4) for one embedding
model: the cosine between attribute-key texts exactly as step 4 embeds them (a
free-form key's words, a family key's topic only), for pairs that name the same
aspect and for traps that must never be aligned.

    python bench/key_names.py --embeddings-url http://127.0.0.1:11434/v1 \\
        --model bge-m3 --json key-names-bge-m3.json

A threshold is only worth setting when the two groups are clearly apart: the
lowest "same" cosine above the highest "trap" one. Otherwise step 4 stays off
(when we cannot tell, we do not supersede). The inputs are key names only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iifa import vectors
from iifa.llm import EMBEDDINGS, Endpoint, ModelClient

SAME: list[tuple[str, str]] = [  # should be aligned
    ("commute mode", "transport mode"),
    ("hair colour", "hair color"),
    ("hometown", "home town"),
    ("shoe size", "shoe size eu"),
    ("drink", "beverage"),  # preference topics
    ("music", "music genre"),
    ("answers", "answer length"),
    ("marital", "relationship"),  # status topics
    ("health", "health condition"),
]
TRAPS: list[tuple[str, str]] = [  # must never be aligned
    ("residence city", "office city"),
    ("city", "office"),
    ("location", "workplace"),
    ("home city", "work location"),
    ("marital", "work"),  # status.marital vs status.work
    ("drink", "music"),  # preference topics
    ("drink", "food"),
    ("answers", "questions"),
    ("birth", "deadline"),  # date topics
    ("commute mode", "shoe size"),
]


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


def measure(client: ModelClient, model: str) -> dict[str, Any]:
    texts = sorted({t for pair in SAME + TRAPS for t in pair})
    result = client.embed(texts, model=model)
    unit = {t: vectors.normalise(v) for t, v in zip(texts, result.vectors, strict=True)}

    def cos(a: str, b: str) -> float:
        return round(sum(x * y for x, y in zip(unit[a], unit[b], strict=True)), 4)

    same = {f"{a} ~ {b}": cos(a, b) for a, b in SAME}
    traps = {f"{a} ~ {b}": cos(a, b) for a, b in TRAPS}
    lowest, highest = min(same.values()), max(traps.values())
    return {
        "model": model,
        "inputs": texts,
        "same": same,
        "traps": traps,
        "lowest_same": lowest,
        "highest_trap": highest,
        "gap": round(lowest - highest, 4),
        "separated": lowest > highest,
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
        ["python", "bench/key_names.py", *(sys.argv[1:] if argv is None else argv)]
    )
    for title, table in (("same (should be aligned)", "same"), ("traps (must not)", "traps")):
        print(title)
        for pair, value in sorted(report[table].items(), key=lambda kv: -kv[1]):
            print(f"  {value:6.3f}  {pair}")
    print(
        f"lowest same {report['lowest_same']}, highest trap {report['highest_trap']},"
        f" gap {report['gap']}: "
        + ("separated" if report["separated"] else "NOT separated: leave key_threshold unset")
    )
    if args.json is not None:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
