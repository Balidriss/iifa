"""The OpenAPI document is a contract: a change is a deliberate diff.

Regenerate on purpose with ``IIFA_UPDATE_SNAPSHOT=1 pytest tests/api/test_openapi_snapshot.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI

SNAPSHOT = Path(__file__).parent / "openapi_snapshot.json"


def test_openapi_matches_snapshot(app: FastAPI) -> None:
    current = json.dumps(app.openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if os.environ.get("IIFA_UPDATE_SNAPSHOT") == "1" or not SNAPSHOT.exists():
        SNAPSHOT.write_text(current, encoding="utf-8")
    assert current == SNAPSHOT.read_text(encoding="utf-8"), (
        "OpenAPI changed; if intended, run with IIFA_UPDATE_SNAPSHOT=1 and commit the snapshot"
    )
