"""The store is the one place every name and alias goes through (SPEC 5.1):
extraction, ``remember``, an accepted proposal, a dream's theme or merge. It
strips Markdown and refuses what is no name, so no path can write debris."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from iifa.dream.catalogue import parse_operation
from iifa.ingest.schema import ProposedNode, parse_answer
from iifa.names import InvalidName
from iifa.scopes import Visibility
from iifa.store.repo import Store

SCOPES = ["user:alice"]
VIEW = Visibility.of_scope("user:alice")
DEBRIS = "Le Orme (Orme Vallon"
NODE = "node_01J8Z3A4Q5R6S7T8V9W0X1Y2Z3"
NODE2 = "node_01J8Z3A4Q5R6S7T8V9W0X1Y2Z4"


def test_a_node_is_created_with_its_plain_name_and_aliases(store: Store) -> None:
    node = store.create_node(
        scopes=SCOPES, name="**Le Orme**", aliases=["`Orme`", DEBRIS, "[Vallon](https://x.y)"]
    )
    assert node.name == "Le Orme"
    assert node.aliases == ("Orme", "Vallon")
    assert store.find_node("le orme", VIEW) is not None


def test_a_node_name_that_is_no_name_is_refused(store: Store) -> None:
    for name in (DEBRIS, "**", "- "):
        with pytest.raises(InvalidName):
            store.create_node(scopes=SCOPES, name=name)
    assert store.execute("SELECT count(*) FROM nodes").fetchone()[0] == 0


def test_added_aliases_are_plain_and_debris_is_left_out(store: Store) -> None:
    node = store.create_node(scopes=SCOPES, name="Le Orme")
    rev = node.rev
    node = store.update_node(node.id, add_aliases=["réunion du **mardi 3 mars", DEBRIS])
    assert node.aliases == ("réunion du mardi 3 mars",)
    assert store.update_node(node.id, add_aliases=[DEBRIS]).aliases == node.aliases
    assert node.rev > rev


def test_an_alias_proposal_is_plain_or_refused(store: Store) -> None:
    node = store.create_node(scopes=SCOPES, name="Le Orme")
    proposal = store.add_proposal(scopes=SCOPES, kind="alias", node_id=node.id, value="**Orme**")
    assert proposal.value == "Orme"
    with pytest.raises(InvalidName):
        store.add_proposal(scopes=SCOPES, kind="alias", node_id=node.id, value=DEBRIS)


def test_a_stored_debris_proposal_cannot_be_accepted(store: Store) -> None:
    """A stored proposal that is no name: accepting it would add debris, so it is
    refused and stays pending, for a reviewer to reject."""
    node = store.create_node(scopes=SCOPES, name="Le Orme")
    proposal = store.add_proposal(scopes=SCOPES, kind="alias", node_id=node.id, value="Orme")
    store.execute("UPDATE proposals SET value = ? WHERE id = ?", (DEBRIS, proposal.id))
    with pytest.raises(InvalidName):
        store.decide_proposal(proposal.id, accept=True)
    assert store.get_proposal(proposal.id).status == "pending"
    assert store.get_node(node.id).aliases == ()
    assert store.decide_proposal(proposal.id, accept=False).status == "rejected"


def test_remember_strips_markdown_and_refuses_debris(client: TestClient) -> None:
    body = {"scope": "user:alice", "statement": "Le Orme ouvre mardi."}
    created = client.post(
        "/v1/remember", json={**body, "node": {"name": "**Le Orme**", "aliases": ["*Orme*"]}}
    )
    assert created.status_code == 201, created.text
    assert created.json()["node"]["name"] == "Le Orme"
    for node in ({"name": DEBRIS}, {"name": "Le Orme", "aliases": [DEBRIS]}, {"name": "``"}):
        refused = client.post("/v1/remember", json={**body, "node": node})
        assert refused.status_code == 422, node
    kept = client.post("/v1/remember", json={**body, "node": {"name": "Lyon (bureau)"}})
    assert kept.json()["node"]["name"] == "Lyon (bureau)"


def test_accepting_a_debris_proposal_through_the_api_is_refused(client: TestClient) -> None:
    store: Store = client.app.state.store  # type: ignore[attr-defined]
    node = store.create_node(scopes=SCOPES, name="Le Orme")
    proposal = store.add_proposal(scopes=SCOPES, kind="alias", node_id=node.id, value="Orme")
    store.execute("UPDATE proposals SET value = ? WHERE id = ?", (DEBRIS, proposal.id))
    response = client.post(f"/v1/proposals/{proposal.id}", json={"decision": "accept"})
    assert response.status_code == 422
    assert store.get_proposal(proposal.id).status == "pending"


def test_a_dream_theme_is_named_plainly_or_refused() -> None:
    allowed = frozenset({"create_theme"})
    theme = {"op": "create_theme", "members": [NODE, NODE2], "reason": "cluster"}
    assert parse_operation({**theme, "name": "**Thé**"}, allowed=allowed)["name"] == "Thé"
    with pytest.raises(ValueError, match="schema"):
        parse_operation({**theme, "name": "Thé (vert"}, allowed=allowed)


def test_the_alias_counts_survive_a_second_validation() -> None:
    node = ProposedNode.model_validate({"name": "Le Orme", "aliases": [None, DEBRIS, "Orme"]})
    again = ProposedNode.model_validate(node)
    assert (again.unknown_aliases, again.malformed_aliases) == (1, 1)
    assert again.aliases == ["Orme"]


def test_the_alias_bound_holds_on_what_the_model_wrote() -> None:
    answer = parse_answer(
        '{"nodes": [{"name": "Le Orme", "aliases": [' + "null, " * 25 + '"Orme"]}]}'
    )
    assert answer.nodes == [] and answer.rejected.nodes == 1
