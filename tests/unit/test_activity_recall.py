"""A request for an activity: a request that matches an activity
- a word of its name written whole in the request, or at tier 3 a close
vector neighbour - brings that activity's rules first, then the rules of what
it uses, then its newest episode, ahead of the facts that only share words.
A store without activities ranks as before."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest

from iifa.recall.dense import ON, Dense, DenseHit, QueryVector
from iifa.recall.engine import RecallEngine
from iifa.scopes import Visibility
from iifa.store.corrections import merge_nodes
from iifa.store.repo import Store
from iifa.tokens import ConservativeTokenizer

SCOPE = "user:alice"
ALICE = Visibility.of_scope(SCOPE)
REQUEST = "Make me a map of the balcony."


@dataclass
class Memory:
    activity: str
    guide: str
    rules: list[str]  # most recent first
    guide_rule: str
    episode: str
    decoys: list[str]


def _memory(store: Store, *, scope: str = SCOPE) -> Memory:
    activity = store.create_node(scopes=[scope], name="drawing a map", kind="task")
    guide = store.create_node(scopes=[scope], name="Verdant mapping guide", kind="object")
    flat = store.create_node(scopes=[scope], name="the flat", kind="place")

    def add(node: str, statement: str, **kw: Any) -> str:
        return store.add_fact(
            scopes=[scope], node_id=node, statement=statement, origin="extracted", **kw
        ).id

    # facts that share the request's words and are no rule
    decoys = [
        add(flat.id, "The balcony map hangs in the hall of the flat."),
        add(flat.id, "The balcony of the flat gets the afternoon sun."),
    ]
    # rules never resemble the request; the first one hangs on the guide (a
    # model's choice), linked to the activity beside it
    older = add(
        guide.id,
        "Verdant mapping guide: draw north at the top.",
        attribute="method",
        value="north at the top",
        activities=[activity.id],
    )
    newer = add(
        activity.id,
        "Drawing a map: label each bed with its sowing date.",
        attribute="method",
        value="label each bed with its sowing date",
        activities=[activity.id],
    )
    guide_rule = add(
        guide.id,
        "Verdant mapping guide: use one colour per crop.",
        attribute="method",
        value="one colour per crop",
    )
    store.add_edge(scopes=[scope], src=activity.id, dst=guide.id, type="uses", origin="extracted")
    episode = add(
        activity.id,
        "NOVA drew the garden map with Alice.",
        attribute="activity",
        flags=["episode"],
        activities=[activity.id],
    )
    return Memory(activity.id, guide.id, [newer, older], guide_rule, episode, decoys)


def _items(engine: RecallEngine, query: str = REQUEST, **kw: Any) -> list[dict[str, Any]]:
    result: dict[str, Any] = engine.recall(query, view=kw.pop("view", ALICE), budget_tokens=4000)
    return list(result["items"])


def test_a_request_for_the_activity_brings_its_rules_first(store: Store) -> None:
    memory = _memory(store)
    items = _items(RecallEngine(store, ConservativeTokenizer()))
    ids = [i["id"] for i in items]
    block = [*memory.rules, memory.guide_rule, memory.episode]
    assert ids[: len(block)] == block
    assert set(memory.decoys) <= set(ids[len(block) :])  # still there, behind
    whys = [i["why"] for i in items[: len(block)]]
    assert whys == [
        "rule of the activity: drawing a map",
        "rule of the activity: drawing a map",
        "rule of Verdant mapping guide, used for drawing a map",
        "episode of the activity: drawing a map",
    ]


def test_a_named_subject_keeps_its_rules_that_a_block_also_holds(store: Store) -> None:
    """The request names the guide: its rules lead as the named subject's facts
    and the activity's block holds them too; they stay at the front, once."""
    memory = _memory(store)
    query = "Use the Verdant mapping guide to make me a map."
    ids = [i["id"] for i in _items(RecallEngine(store, ConservativeTokenizer()), query)]
    guide_rules = {memory.rules[1], memory.guide_rule}
    assert set(ids[:2]) == guide_rules
    assert memory.rules[0] in ids
    assert len(ids) == len(set(ids))


def test_a_request_matching_only_function_words_brings_no_block(store: Store) -> None:
    """ "a" is in "drawing a map": the search may find the rules on it, but no
    block brings them."""
    _memory(store)
    items = _items(RecallEngine(store, ConservativeTokenizer()), "Feed a cat.")
    assert not [i for i in items if "activity" in i["why"] or "used for" in i["why"]]


def test_without_activities_the_ranking_is_unchanged(store: Store) -> None:
    memory = _memory(store)
    engine = RecallEngine(store, ConservativeTokenizer())
    before = [i["id"] for i in _items(engine)]
    store.execute("DELETE FROM fact_activities")
    store.execute("UPDATE nodes SET kind = 'object' WHERE id = ?", (memory.activity,))
    after = [i["id"] for i in _items(engine)]
    assert after.index(memory.decoys[0]) < after.index(memory.rules[0])
    assert before != after


def test_the_activity_with_more_matched_words_comes_first_and_three_at_most(
    store: Store,
) -> None:
    _memory(store)
    rules: dict[str, str] = {}
    for name in ("balcony map drawing", "map colouring", "map folding", "map printing"):
        node = store.create_node(scopes=[SCOPE], name=name, kind="task")
        rules[name] = store.add_fact(
            scopes=[SCOPE],
            node_id=node.id,
            statement=f"{name.capitalize()}: keep the legend small.",
            origin="extracted",
            attribute="method",
            value="keep the legend small",
            activities=[node.id],
        ).id
    items = _items(RecallEngine(store, ConservativeTokenizer()))
    assert items[0]["id"] == rules["balcony map drawing"]  # two matched words
    prefix = "rule of the activity: "
    brought = {i["why"][len(prefix) :] for i in items if i["why"].startswith(prefix)}
    assert len(brought) == 3 and "balcony map drawing" in brought  # three at most


def test_rules_of_a_node_merged_into_the_activity_come_with_it(store: Store) -> None:
    memory = _memory(store)
    old = store.create_node(scopes=[SCOPE], name="making a plan", kind="task")
    merged = store.add_fact(
        scopes=[SCOPE],
        node_id=old.id,
        statement="Making a plan: measure twice.",
        origin="extracted",
        attribute="method",
        value="measure twice",
        activities=[old.id],
    ).id
    keep, absorb = store.get_node(memory.activity), store.get_node(old.id)
    with store.lock, store.transaction():
        merge_nodes(store, keep, absorb, store.node_fact_ids(old.id), store.node_edge_ids(old.id))
    items = _items(RecallEngine(store, ConservativeTokenizer()))
    (item,) = [i for i in items if i["id"] == merged]
    assert item["why"] == "rule of the activity: drawing a map"
    assert items.index(item) < 3  # among the activity's own rules, most recent first


def test_p9_a_block_never_brings_what_the_reader_cannot_see(store: Store) -> None:
    memory = _memory(store)
    hidden = store.add_fact(
        scopes=["user:bob"],
        statement="Bob's secret: never share the balcony keys.",
        origin="extracted",
        attribute="method",
        value="never share",
        activities=[memory.activity],
    ).id
    ids = [i["id"] for i in _items(RecallEngine(store, ConservativeTokenizer()))]
    assert hidden not in ids
    bob = [
        i["id"]
        for i in _items(
            RecallEngine(store, ConservativeTokenizer()), view=Visibility.of_scope("user:bob")
        )
    ]
    assert not set(bob) & {*memory.rules, memory.guide_rule, memory.episode}


# --------------------------------------------------------------------------- tier 3


class _Vectors:
    """A dense side that always finds the activity at a given closeness."""

    def __init__(self, record: str, score: float) -> None:
        self.record, self.score = record, score
        self.model = "bge-m3"

    def query_vector(self, query: str, supplied: Any = None) -> QueryVector:
        return QueryVector([1.0], ON)

    def search(self, values: Sequence[float], view: Visibility, limit: int) -> list[DenseHit]:
        return [DenseHit(self.record, "node", self.score)]


@pytest.mark.parametrize(
    ("score", "threshold", "brought"),
    [(0.60, 0.53, True), (0.50, 0.53, False), (0.99, None, False)],
)
def test_at_tier_3_a_close_vector_neighbour_matches_an_activity(
    store: Store, score: float, threshold: float | None, brought: bool
) -> None:
    """A request in other words ("Sketch the balcony for me" shares no word
    with "drawing a map") matches the activity when its vector is as close as
    the model's resolve threshold; an uncalibrated model matches nothing."""
    memory = _memory(store)
    engine = RecallEngine(
        store,
        ConservativeTokenizer(),
        tier=3,
        dense=cast(Dense, _Vectors(memory.activity, score)),
        activity_threshold=lambda model: threshold,
    )
    ids = [i["id"] for i in _items(engine, "Sketch the balcony for me.")]
    assert (ids[: len(memory.rules)] == memory.rules) is brought


# --------------------------------------------------------------------------- the blocks' bounds


def test_the_named_subjects_facts_stay_ahead_of_the_blocks(store: Store) -> None:
    """A word of an activity is often a common word ("notes", "map"): the
    subject the question names keeps the first places."""
    memory = _memory(store)
    karim = store.create_node(scopes=[SCOPE], name="Karim", kind="person")
    hers = store.add_fact(
        scopes=[SCOPE], node_id=karim.id, statement="Karim keeps the map.", origin="explicit"
    ).id
    ids = [
        i["id"]
        for i in _items(RecallEngine(store, ConservativeTokenizer()), "Where is Karim's map?")
    ]
    assert ids[0] == hers
    assert ids[1:3] == memory.rules  # the block right behind the subject


def test_the_blocks_take_two_thirds_of_the_budget_at_most(store: Store) -> None:
    memory = _memory(store)
    engine = RecallEngine(store, ConservativeTokenizer())
    result = engine.recall(REQUEST, view=ALICE, budget_tokens=300)
    block = {*memory.rules, memory.guide_rule, memory.episode}
    spent = sum(engine.cost(i) for i in result["items"] if i["id"] in block)
    assert 0 < spent <= 200
    assert any(i["id"] not in block for i in result["items"])  # room left for the rest


def test_a_handle_counts_the_activities_its_expand_adds(store: Store) -> None:
    memory = _memory(store)
    engine = RecallEngine(store, ConservativeTokenizer())
    (handle,) = [i for i in _items(engine) if i["id"] == memory.rules[0]]
    expanded = engine.expand([memory.rules[0]], view=ALICE, budget_tokens=4000)["items"][0]
    assert expanded["activities"] == [{"id": memory.activity, "name": "drawing a map"}]
    assert handle["expand_tokens"] == engine.cost(expanded)


def test_the_rules_of_a_used_node_come_whatever_its_other_facts(store: Store) -> None:
    memory = _memory(store)
    for n in range(30):
        store.add_fact(
            scopes=[SCOPE],
            node_id=memory.guide,
            statement=f"Verdant mapping guide page {n} lists a symbol.",
            origin="extracted",
        )
    engine = RecallEngine(store, ConservativeTokenizer(), max_candidates=10)
    whys = {i["id"]: i["why"] for i in _items(engine)}
    assert whys[memory.guide_rule] == "rule of Verdant mapping guide, used for drawing a map"


# --------------------------------------------------------------------------- families and timeline


def _occurrence(store: Store, name: str, rule: str) -> tuple[str, str, str]:
    """A task node named with its object, one rule and one episode linked to it."""
    node = store.create_node(scopes=[SCOPE], name=name, kind="task")
    rule_id = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement=f"{name.capitalize()}: {rule}.",
        origin="extracted",
        attribute="method",
        value=rule,
        activities=[node.id],
    ).id
    episode = store.add_fact(
        scopes=[SCOPE],
        node_id=node.id,
        statement=f"NOVA was {name} with Alice.",
        origin="extracted",
        attribute="activity",
        flags=["episode"],
        activities=[node.id],
    ).id
    return node.id, rule_id, episode


def _grouped(store: Store, name: str, members: Sequence[str]) -> str:
    parent = store.create_node(scopes=[SCOPE], name=name, kind="task")
    for member in members:
        store.add_edge(
            scopes=[SCOPE], src=member, dst=parent.id, type="instance_of", origin="dream"
        )
    return parent.id


OCCURRENCES = (
    ("painting a portrait of the baker", "paint in daylight"),
    ("painting a portrait of the gardener", "keep each sitting under two hours"),
    ("painting a portrait of the cook", "sign the lower corner"),
    ("painting a portrait of the dog", "sketch the outline in pencil first"),
)


def test_a_family_brings_every_occurrences_rules_and_its_timeline(clock: Any, store: Store) -> None:
    """Four occurrences named with their objects: ungrouped, a request for a
    portrait of someone else matches three at most (three blocks); grouped,
    it brings the family's four rules, then its episodes, the most recent
    first, three at most."""
    made = []
    for name, rule in OCCURRENCES:
        made.append(_occurrence(store, name, rule))
        clock.advance(days=1)
    request = "Paint a portrait of the mayor in the garden."
    engine = RecallEngine(store, ConservativeTokenizer())
    before = _items(engine, request)
    blocked = {i["why"] for i in before if i["why"].startswith("rule of the activity: ")}
    assert len(blocked) == 3  # one occurrence left out
    _grouped(store, "painting a portrait", [node for node, _, _ in made])
    items = _items(engine, request)
    rules = {rule for _, rule, _ in made}
    episodes = [episode for _, _, episode in reversed(made)][:3]  # the last time first
    assert [i["id"] for i in items[:4]] == [r for _, r, _ in reversed(made)]
    assert set(i["id"] for i in items[:4]) == rules
    assert [i["id"] for i in items[4:7]] == episodes
    assert {i["why"] for i in items[:7]} == {
        "rule of the activity: painting a portrait",
        "episode of the activity: painting a portrait",
    }


def test_an_instance_of_a_hidden_or_non_task_node_stands_for_itself(store: Store) -> None:
    (node, rule, _) = _occurrence(store, "painting a portrait of the baker", "paint in daylight")
    hidden = store.create_node(scopes=["user:bob"], name="painting a portrait", kind="task")
    store.add_edge(scopes=[SCOPE], src=node, dst=hidden.id, type="instance_of", origin="dream")
    items = _items(RecallEngine(store, ConservativeTokenizer()), "Paint a portrait of the mayor.")
    (item,) = [i for i in items if i["id"] == rule]
    assert item["why"] == "rule of the activity: painting a portrait of the baker"
