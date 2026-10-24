"""P9 for the profile block: an invisible scope's pinned facts never appear, a
pinned fact whose node is invisible is returned without its node, and a node's
counts include only visible facts and edges with visible far nodes."""

from __future__ import annotations

import pytest

from iifa.recall.engine import RecallEngine
from iifa.recall.profile import profile_block
from iifa.scopes import Visibility
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

pytestmark = pytest.mark.security


def test_p9_profile_filters_before_counting(store: Store) -> None:
    engine = RecallEngine(store, ConservativeTokenizer())
    alice, bob = ["user:alice"], ["user:bob"]
    karim = store.create_node(scopes=alice, name="Karim", kind="person")
    secret = store.create_node(scopes=bob, name="Secret contact", kind="person")
    shared = store.create_node(scopes=alice, name="Solar project", kind="project")
    store.add_fact(scopes=alice, node_id=karim.id, statement="visible", origin="explicit")
    store.add_fact(scopes=bob, node_id=karim.id, statement="bob's note on Karim", origin="explicit")
    store.add_fact(scopes=alice, node_id=karim.id, statement="pin", origin="explicit", pinned=True)
    store.add_fact(
        scopes=bob, node_id=secret.id, statement="bob pinned", origin="explicit", pinned=True
    )
    # a pinned fact alice may see, about a node she may not
    store.add_fact(
        scopes=alice, node_id=secret.id, statement="about someone", origin="explicit", pinned=True
    )
    store.add_edge(scopes=alice, src=karim.id, dst=shared.id, type="works_on", origin="explicit")
    store.add_edge(scopes=alice, src=karim.id, dst=secret.id, type="knows", origin="explicit")
    store.add_edge(scopes=bob, src=karim.id, dst=shared.id, type="funds", origin="explicit")

    block = profile_block(engine, view=Visibility.of_scope("user:alice"), budget_tokens=5000)
    facts = [i for i in block["items"] if i["kind"] == "fact"]
    assert [f["statement"] for f in facts] == ["pin", "about someone"]
    assert facts[1]["node_id"] is None and facts[1]["node_name"] is None
    nodes = {i["name"]: i for i in block["items"] if i["kind"] == "node"}
    assert set(nodes) == {"Karim", "Solar project"}  # never "Secret contact"
    assert nodes["Karim"]["facts"] == 2  # bob's note is not counted
    assert nodes["Karim"]["edges"] == 1  # the edge to the invisible node is not, nor bob's edge
    assert nodes["Solar project"]["edges"] == 1
    text = str(block)
    assert "Secret contact" not in text and "bob" not in text.lower()

    bobs = profile_block(engine, view=Visibility.of_scope("user:bob"), budget_tokens=5000)
    assert [i["statement"] for i in bobs["items"] if i["kind"] == "fact"] == ["bob pinned"]
    assert profile_block(engine, view=Visibility(), budget_tokens=500)["items"] == []
