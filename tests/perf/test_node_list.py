"""The node list at small scale: a page of 100 out of 1000 nodes,
by name and by last change, stays well under the threshold, and a whole walk
reads every node once."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from iifa.ids import new_id
from iifa.scopes import Visibility
from iifa.store.repo import Store
from tests.fixtures.corpus import people

pytestmark = pytest.mark.perf

NODES = 1000
FACTS_PER_NODE = 2
PAGE = 100
P95_MS = 150.0


def test_node_list_pages_fast_at_small_scale(tmp_path: Path) -> None:
    with Store(tmp_path / "perf.db", "k") as store:
        names = people(50)
        for i in range(NODES):
            made = store.create_node(
                scopes=["user:perf"], name=f"{names[i % len(names)]} {i:04d}", kind="person"
            )
            for j in range(FACTS_PER_NODE):
                store.add_fact(
                    scopes=["user:perf"],
                    node_id=made.id,
                    statement=f"fact {j} about {made.name}",
                    origin="extracted",
                )
            if i % 10 == 0:
                store.add_proposal(scopes=["user:perf"], kind="alias", node_id=made.id, value="x")
        view = Visibility.of_scope("user:perf/agent:bench")
        mark = new_id("node").split("_", 1)[1]
        for sort, key in (("name", "name_norm"), ("changed", "changed_at")):
            latencies: list[float] = []
            seen: list[str] = []
            after: tuple[str, str] | None = None
            while True:
                started = time.perf_counter()
                rows = store.node_list(
                    view,
                    sort=sort,
                    prefix=None,
                    include_archived=False,
                    mark=mark,
                    after=after,
                    limit=PAGE,
                )
                latencies.append(time.perf_counter() - started)
                seen.extend(r["id"] for r in rows)
                if len(rows) < PAGE:
                    break
                after = (rows[-1][key], rows[-1]["id"])
            assert len(seen) == len(set(seen)) == NODES
            p95 = statistics.quantiles(latencies, n=20)[-1] * 1000
            assert p95 < P95_MS, f"{sort}: p95 {p95:.1f} ms over {len(latencies)} pages"
