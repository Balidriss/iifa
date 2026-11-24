"""One review queue: pending alias and fact proposals and the
dream's queued operations in one listing, under the one ``review`` permission;
accepting a queued operation re-checks it against memory as it is now."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.dream import _item_fact_ids, _item_node_ids, _name_nodes
from iifa.dream.apply import Applier
from tests.dream.conftest import (
    SCOPE,
    Memory,
    build_memory,
    dream,
    dream_model,
    new_run,
    submit,
)
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth


def _queue_a_merge(client: TestClient, fake: FakeOpenAIServer, memory: Memory) -> dict[str, Any]:
    # the pending merge proposal makes the pair a candidate; the adjudication's
    # "same" queues the merge
    store = client.app.state.store  # type: ignore[attr-defined]
    store.add_proposal(scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id)
    fake.set_default(
        chat=dream_model(
            lambda unit: [], lambda pair: {"verdict": "same", "reason": "the same person"}
        )
    )
    return dream(client)


def _review(client: TestClient, token: str = "review", scope: str = SCOPE) -> Any:
    return client.get(
        "/v1/review", params={"scope": scope, "budget_tokens": 16000}, headers=auth(token)
    )


def test_the_review_queue_lists_proposals_and_queued_operations_together(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    alias = store.add_proposal(scopes=[SCOPE], kind="alias", node_id=memory.karim.id, value="Kaka")
    fact = store.add_proposal(
        scopes=[SCOPE],
        kind="fact",
        node_id=None,
        value="The user lives in Moscow.",
        payload={"provenance": "external"},
    )
    merge = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    run = _queue_a_merge(dream_client, fake_openai, memory)
    response = _review(dream_client)
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    kinds = {(i["kind"], i["id"]) for i in items}
    queued = run["report"]["review"]
    assert kinds == {("alias", alias.id), ("fact", fact.id), ("dream_operation", queued[0])}
    assert merge.id not in {i["id"] for i in items}  # its queued merge_nodes stands for it
    operation = next(i for i in items if i["kind"] == "dream_operation")
    assert operation["op"] == "merge_nodes"
    assert operation["risk"] == "high"
    assert operation["reason_given"] == "the same person"
    assert operation["run_id"] == run["run_id"]
    assert operation["proposal_id"] == merge.id
    adjudication = operation["adjudication"]
    assert (adjudication["verdict"], adjudication["reason"]) == ("same", "the same person")
    assert adjudication["run_id"] == run["run_id"]


def test_the_review_queue_needs_the_review_permission(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    for token in ("dream", "read", "write", "admin", "alice", "alice-dream"):
        assert _review(dream_client, token).status_code == 403, token
    assert _review(dream_client, "alice-review").status_code == 200


def test_the_review_queue_shows_only_what_the_reviewer_sees(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    _queue_a_merge(dream_client, fake_openai, memory)
    assert len(_review(dream_client, "alice-review").json()["items"]) == 1
    response = _review(dream_client, "alice-review", scope="user:bob")
    assert response.status_code == 403
    assert _review(dream_client, "review", scope="user:bob").json()["items"] == []


def test_accepting_a_queued_operation_applies_it(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "applied"
    assert store.get_node(memory.kiki.id).merged_into == memory.karim.id
    again = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert again.status_code == 409
    assert _review(dream_client).json()["items"] == []


def test_rejecting_a_queued_operation_changes_nothing(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    before = store.revisions([memory.karim.id, memory.kiki.id])
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "reject"}, headers=auth("review")
    )
    assert (response.status_code, response.json()["status"]) == (200, "rejected")
    assert store.revisions([memory.karim.id, memory.kiki.id]) == before


def test_a_queued_operation_whose_inputs_moved_is_dropped_on_accept(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    store.update_node(memory.kiki.id, add_aliases=["Kiki B."])  # memory moved meanwhile
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert response.status_code == 409
    assert "stale" in response.text
    assert store.get_node(memory.kiki.id).merged_into is None


def test_a_dream_token_cannot_approve_what_its_dream_proposed(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    for token in ("dream", "alice-dream", "write"):
        response = dream_client.post(
            f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth(token)
        )
        assert response.status_code == 403, token


def test_a_reviewer_bound_elsewhere_does_not_see_the_operation(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    op_id = _queue_a_merge(dream_client, fake_openai, memory)["report"]["review"][0]
    response = dream_client.post(
        f"/v1/review/{op_id}", json={"decision": "accept"}, headers=auth("bob-review")
    )
    assert response.status_code == 404


def test_proposals_are_decided_through_the_review_queue_too(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    alias = store.add_proposal(scopes=[SCOPE], kind="alias", node_id=memory.karim.id, value="Kaka")
    fact = store.add_proposal(
        scopes=[SCOPE],
        kind="fact",
        node_id=memory.karim.id,
        value="Karim plays golf on Sundays.",
    )
    for proposal in (alias, fact):
        response = dream_client.post(
            f"/v1/review/{proposal.id}", json={"decision": "accept"}, headers=auth("review")
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "accepted"
    assert "Kaka" in store.get_node(memory.karim.id).aliases


def test_accepting_a_merge_proposal_in_the_review_queue_merges_the_two_nodes(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    """A reviewer's accept is a person's decision; Karim has the more
    live facts, so Karim is kept."""
    store = dream_app.state.store
    memory = build_memory(store)
    merge = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.kiki.id, value=memory.karim.id
    )
    response = dream_client.post(
        f"/v1/review/{merge.id}", json={"decision": "accept"}, headers=auth("review")
    )
    assert response.status_code == 200, response.text
    out = response.json()
    assert (out["status"], out["keep"], out["absorb"]) == (
        "accepted",
        memory.karim.id,
        memory.kiki.id,
    )
    assert out["correction_id"].startswith("corr_")
    assert store.get_node(memory.kiki.id).merged_into == memory.karim.id
    assert store.get_proposal(merge.id).status == "accepted"
    assert _review(dream_client).json()["items"] == []


def test_an_unknown_review_item_is_not_found(dream_client: TestClient) -> None:
    for item in ("dop_" + "0" * 26, "prop_" + "0" * 26, "node_" + "0" * 26, "x"):
        response = dream_client.post(
            f"/v1/review/{item}", json={"decision": "accept"}, headers=auth("review")
        )
        assert response.status_code == 404, item


def test_a_review_item_names_its_nodes_with_their_first_facts(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    store.add_fact(
        scopes=[SCOPE],
        node_id=memory.kiki.id,
        statement="Kiki plays the guitar.",
        origin="extracted",
        language="en",
    )
    merge = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    alias = store.add_proposal(
        scopes=[SCOPE], kind="alias", node_id=memory.lyon.id, value="Lugdunum"
    )
    free = store.add_proposal(
        scopes=[SCOPE], kind="fact", node_id=None, value="The user lives in Moscow."
    )
    items = {i["id"]: i for i in _review(dream_client).json()["items"]}
    karim, kiki = items[merge.id]["nodes"]
    assert (karim["id"], karim["name"], karim["kind"]) == (memory.karim.id, "Karim", "person")
    assert (kiki["name"], kiki["aliases"]) == ("Kiki", ["K."])
    assert [f["statement"] for f in kiki["facts"]] == ["Kiki plays the guitar."]
    # pinned first, then newest, never more than three
    assert [f["id"] for f in karim["facts"]] == [
        memory.pinned.id,
        memory.green_tea.id,
        memory.tea.id,
    ]
    assert [n["name"] for n in items[alias.id]["nodes"]] == ["Lyon"]
    assert "nodes" not in items[free.id]


def test_a_queued_merge_names_the_node_kept_then_the_node_absorbed(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    memory = build_memory(dream_app.state.store)
    _queue_a_merge(dream_client, fake_openai, memory)
    (operation,) = _review(dream_client).json()["items"]
    payload = operation["payload"]
    assert [n["id"] for n in operation["nodes"]] == [payload["keep"], payload["absorb"]]
    assert {n["name"] for n in operation["nodes"]} == {"Karim", "Kiki"}


def test_a_review_item_shows_only_the_nodes_and_facts_the_view_sees(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    hidden = f"{SCOPE}/agent:coder"
    coder_karim = store.create_node(scopes=[hidden], name="Karim B.", kind="person")
    secret = store.add_fact(
        scopes=[hidden],
        node_id=memory.karim.id,
        statement="Karim is moving abroad.",
        origin="extracted",
        language="en",
    )
    merge = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=coder_karim.id
    )
    narrow = {i["id"]: i for i in _review(dream_client).json()["items"]}
    (karim,) = narrow[merge.id]["nodes"]
    assert karim["id"] == memory.karim.id
    assert secret.id not in {f["id"] for f in karim["facts"]}
    wide = dream_client.get(
        "/v1/review",
        params={"scopes": [SCOPE, hidden], "budget_tokens": 16000},
        headers=auth("review"),
    )
    assert wide.status_code == 200, wide.text
    item = next(i for i in wide.json()["items"] if i["id"] == merge.id)
    assert [n["id"] for n in item["nodes"]] == [memory.karim.id, coder_karim.id]
    assert item["nodes"][0]["facts"][1]["id"] == secret.id


def test_naming_the_nodes_never_costs_an_item_its_place_in_the_page(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    merge = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    # 400 is the service's default budget: the bare item fits it, two named
    # nodes with their facts do not
    for budget in (400, 600, 16000):
        page = dream_client.get(
            "/v1/review", params={"scope": SCOPE, "budget_tokens": budget}, headers=auth("review")
        ).json()
        assert [i["id"] for i in page["items"]] == [merge.id], (budget, page["reason"])
    full = page["items"][0]["nodes"]
    assert [len(n["facts"]) for n in full] == [3, 0]


def test_naming_one_item_never_pushes_the_next_off_the_page(
    dream_client: TestClient, dream_app: FastAPI, fake_openai: FakeOpenAIServer
) -> None:
    store = dream_app.state.store
    memory = build_memory(store)
    first = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    second = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.lyon.id, value=memory.kiki.id
    )
    page = dream_client.get(
        "/v1/review", params={"scope": SCOPE, "budget_tokens": 400}, headers=auth("review")
    ).json()
    assert [i["id"] for i in page["items"]] == [first.id, second.id]
    assert page["used_tokens"] <= 400


def test_the_facts_go_first_then_the_names() -> None:
    def item() -> dict[str, Any]:
        return {"kind": "merge", "node_id": "node_a", "value": "node_b"}

    briefs = {
        "node_a": {"id": "node_a", "name": "Ann", "facts": [{"id": "f", "statement": "s" * 40}]},
        "node_b": {"id": "node_b", "name": "Bea", "facts": []},
    }

    def cost(record: Any) -> int:
        return len(str(record))

    names = [{**b, "facts": []} for b in briefs.values()]
    full_extra = cost({**item(), "nodes": list(briefs.values())}) - cost(item())
    names_extra = cost({**item(), "nodes": names}) - cost(item())
    for spare, expected in ((full_extra, list(briefs.values())), (names_extra, names)):
        page = [item()]
        _name_nodes(page, briefs, spare, cost)
        assert page[0]["nodes"] == expected
    page = [item()]
    _name_nodes(page, briefs, names_extra - 1, cost)
    assert "nodes" not in page[0]
    # the spare is shared in page order: the second item gets what is left
    page = [item(), item()]
    _name_nodes(page, briefs, full_extra + names_extra, cost)
    assert [len(i["nodes"][0]["facts"]) for i in page] == [1, 0]


def _queue(app: FastAPI, op: dict[str, Any]) -> Any:
    """One operation submitted under the default policy, as a dream would."""
    store, journal = app.state.store, app.state.dreamer.journal
    return submit(store, Applier(store, journal), new_run(journal), op)


def test_a_queued_grouping_names_its_members(dream_client: TestClient, dream_app: FastAPI) -> None:
    store = dream_app.state.store
    names = ("painting a portrait of the baker", "painting a portrait of the mayor")
    members = [store.create_node(scopes=[SCOPE], name=n, kind="task") for n in names]
    told = store.add_fact(
        scopes=[SCOPE],
        node_id=members[0].id,
        statement="The baker's portrait starts with the eyes.",
        origin="extracted",
        language="en",
    )
    entry = _queue(
        dream_app,
        {
            "op": "group_activities",
            "name": "painting a portrait",
            "members": [m.id for m in members],
        },
    )
    assert entry.status == "queued"
    (item,) = _review(dream_client).json()["items"]
    assert (item["op"], item["payload"]["name"]) == ("group_activities", "painting a portrait")
    assert [(n["id"], n["name"], n["kind"]) for n in item["nodes"]] == [
        (m.id, m.name, "task") for m in members
    ]
    assert [f["id"] for f in item["nodes"][0]["facts"]] == [told.id]
    assert "facts" not in item  # the operation names no fact


def test_a_queued_consolidation_states_the_facts_it_replaces(
    dream_client: TestClient, dream_app: FastAPI
) -> None:
    memory = build_memory(dream_app.state.store)
    replaced = [memory.tea, memory.green_tea]
    entry = _queue(
        dream_app,
        {
            "op": "consolidate_facts",
            "node": memory.karim.id,
            "facts": [f.id for f in replaced],
            "statement": "Karim drinks green tea every morning.",
        },
    )
    assert entry.status == "queued"
    (item,) = _review(dream_client).json()["items"]
    assert [n["name"] for n in item["nodes"]] == ["Karim"]
    assert item["facts"] == [{"id": f.id, "statement": f.statement} for f in replaced]


def test_the_facts_under_the_nodes_go_first_then_the_named_facts_then_the_names() -> None:
    node, fact = "node_" + "0" * 26, "fact_" + "0" * 26

    def item() -> dict[str, Any]:
        payload = {"op": "consolidate_facts", "node": node, "facts": [fact], "statement": "s"}
        return {"kind": "dream_operation", "op": "consolidate_facts", "payload": payload}

    brief = {"id": node, "name": "Ann", "facts": [{"id": "f", "statement": "s" * 40}]}
    name = {**brief, "facts": []}
    stated = {"id": fact, "statement": "t" * 40}

    def cost(record: Any) -> int:
        return len(str(record))

    def extra(**added: Any) -> int:
        return cost({**item(), **added}) - cost(item())

    full = extra(nodes=[brief], facts=[stated])
    without_node_facts = extra(nodes=[name], facts=[stated])
    names_only = extra(nodes=[name])
    for spare, nodes, facts in (
        (full, [brief], [stated]),
        (full - 1, [name], [stated]),
        (without_node_facts - 1, [name], None),
        (names_only - 1, None, None),
    ):
        page = [item()]
        _name_nodes(page, {node: brief}, spare, cost, {fact: stated})
        assert (page[0].get("nodes"), page[0].get("facts")) == (nodes, facts), spare


def test_an_item_names_only_the_nodes_and_facts_its_payload_names() -> None:
    node, fact = "node_" + "0" * 26, "fact_" + "0" * 26
    operation = {"kind": "dream_operation", "op": "supersede_fact", "payload": None}
    assert (_item_node_ids(operation), _item_fact_ids(operation)) == ([], [])
    operation["payload"] = {"op": "supersede_fact", "fact": fact, "by": fact, "reason": node}
    assert (_item_node_ids(operation), _item_fact_ids(operation)) == ([], [fact])
    orphan = {"kind": "merge", "node_id": None, "value": node}
    assert (_item_node_ids(orphan), _item_fact_ids(orphan)) == ([], [])
