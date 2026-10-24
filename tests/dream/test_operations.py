"""Each operation of the catalogue (SPEC 5.6): what the service checks before
applying it, what applying does, and that undo reverts it exactly. One test
per row of the table, one per reason an operation is dropped."""

from __future__ import annotations

from typing import Any

import pytest

from iifa.dream.apply import Applier, says_one_thing
from iifa.dream.journal import Journal
from iifa.dream.policy import ReviewPolicy
from iifa.scopes import Visibility
from iifa.store.models import Fact
from iifa.store.repo import Store
from tests.dream.conftest import SCOPE, Memory, new_run, submit

AUTO_ALL = ReviewPolicy(low="auto", medium="auto", high="auto")
# an operator who lets consolidations and themes apply (the default reviews them)
MEDIUM_AUTO = ReviewPolicy(medium="auto")


def _status(entry: Any) -> tuple[str, str | None]:
    return entry.status, entry.reason


# --------------------------------------------------------------------------- refresh_summary


def test_refresh_summary_applies_and_undoes(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    store.update_node(memory.karim.id, summary="Old line.", language="en")
    run = new_run(journal)
    entry = submit(
        store,
        applier,
        run,
        {
            "op": "refresh_summary",
            "node": memory.karim.id,
            "summary": "Karim lives in Lyon and likes tea.",
        },
        ids=[memory.karim.id],
    )
    assert _status(entry) == ("applied", None)
    node = store.get_node(memory.karim.id)
    assert node.summary == "Karim lives in Lyon and likes tea."
    assert node.language == "en"
    applier.undo_op(entry.id)
    node = store.get_node(memory.karim.id)
    assert (node.summary, node.language) == ("Old line.", "en")
    assert journal.get_op(entry.id).status == "undone"


def test_refresh_summary_needs_live_facts(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "refresh_summary", "node": memory.kiki.id, "summary": "Kiki is someone."},
    )
    assert _status(entry) == ("dropped", "no_facts")


def test_refresh_summary_in_another_language_is_dropped_not_flagged(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "refresh_summary", "node": memory.karim.id, "summary": "Карим живёт в Лионе."},
        ids=[memory.karim.id],
    )
    assert _status(entry) == ("dropped", "language_mismatch")
    assert store.get_node(memory.karim.id).summary == ""


# --------------------------------------------------------------------------- propose_edge


def test_propose_edge_adds_a_proposed_edge_and_undo_deletes_it(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "propose_edge", "src": memory.karim.id, "dst": memory.lyon.id, "type": "lives in"},
    )
    assert _status(entry) == ("applied", None)
    edges = store.edges_between([memory.karim.id, memory.lyon.id], Visibility.of_scope(SCOPE))
    assert [(e.type, e.status, e.origin, e.scopes) for e in edges] == [
        ("lives_in", "proposed", "dream", (SCOPE,))
    ]
    assert edges[0].weight < 1.0
    applier.undo_op(entry.id)
    assert store.edges_between([memory.karim.id, memory.lyon.id], Visibility.of_scope(SCOPE)) == []


def test_propose_edge_drops_a_duplicate(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    store.add_edge(
        scopes=[SCOPE], src=memory.lyon.id, dst=memory.karim.id, type="lives_in", origin="test"
    )
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "propose_edge", "src": memory.karim.id, "dst": memory.lyon.id, "type": "lives_in"},
    )
    assert _status(entry) == ("dropped", "duplicate")


def test_propose_edge_needs_two_nodes(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "propose_edge", "src": memory.karim.id, "dst": memory.karim.id, "type": "knows"},
    )
    assert _status(entry) == ("dropped", "same_record")


def test_propose_edge_between_scope_sets_that_differ_is_dropped(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    solar = store.create_node(scopes=[f"{SCOPE}/project:solar"], name="Solar", kind="project")
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "propose_edge", "src": memory.karim.id, "dst": solar.id, "type": "works_on"},
    )
    assert _status(entry) == ("dropped", "scope_mismatch")


# --------------------------------------------------------------------------- create_theme


def test_create_theme_adds_a_topic_and_undo_deletes_it(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal, policy=MEDIUM_AUTO),
        {
            "op": "create_theme",
            "name": "Family",
            "members": [memory.karim.id, memory.kiki.id],
            "summary": "The family.",
        },
    )
    assert _status(entry) == ("applied", None)
    theme = store.find_node("Family", Visibility.of_scope(SCOPE))
    assert theme is not None
    assert (theme.kind, theme.scopes, theme.summary) == ("topic", (SCOPE,), "The family.")
    edges = store.edges_of([theme.id], Visibility.of_scope(SCOPE))
    assert sorted((e.src, e.type, e.status) for e in edges) == sorted(
        [(memory.karim.id, "about", "confirmed"), (memory.kiki.id, "about", "confirmed")]
    )
    applier.undo_op(entry.id)
    assert store.find_node("Family", Visibility.of_scope(SCOPE)) is None
    assert store.edges_of([memory.karim.id], Visibility.of_scope(SCOPE)) == []


def test_create_theme_drops_a_duplicate_topic(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    store.create_node(scopes=[SCOPE], name="Family", kind="topic")
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "create_theme", "name": "family", "members": [memory.karim.id, memory.kiki.id]},
    )
    assert _status(entry) == ("dropped", "duplicate")


def test_create_theme_drops_a_name_mixing_scripts(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "create_theme", "name": "Семьяfamily", "members": [memory.karim.id, memory.kiki.id]},
    )
    assert _status(entry) == ("dropped", "mixed_script")


def test_create_theme_over_scope_sets_that_differ_is_dropped(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    solar = store.create_node(scopes=[f"{SCOPE}/project:solar"], name="Solar", kind="project")
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "create_theme", "name": "Work", "members": [memory.karim.id, solar.id]},
    )
    assert _status(entry) == ("dropped", "scope_mismatch")


# --------------------------------------------------------------------------- consolidate_facts


def _consolidate(memory: Memory, facts: list[str], statement: str) -> dict[str, Any]:
    return {
        "op": "consolidate_facts",
        "node": memory.karim.id,
        "facts": facts,
        "statement": statement,
    }


def test_consolidate_closes_and_references_and_undo_reopens(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal, policy=MEDIUM_AUTO),
        _consolidate(
            memory, [memory.tea.id, memory.green_tea.id], "Karim drinks green tea every morning."
        ),
    )
    assert _status(entry) == ("applied", None)
    tea, green = store.get_fact(memory.tea.id), store.get_fact(memory.green_tea.id)
    assert tea.valid_to is not None and green.valid_to is not None
    assert tea.consolidated_into == green.consolidated_into
    assert tea.superseded_by is None  # not a change of value
    new = store.get_fact(tea.consolidated_into or "")
    assert (new.statement, new.origin, new.node_id, new.scopes, new.valid_to) == (
        "Karim drinks green tea every morning.",
        "extracted",
        memory.karim.id,
        (SCOPE,),
        None,
    )
    assert new.language == "en"
    assert (new.attribute, new.value) == ("routine", "morning drink")  # one key, one value
    applier.undo_op(entry.id)
    tea, green = store.get_fact(memory.tea.id), store.get_fact(memory.green_tea.id)
    assert (tea.valid_to, tea.consolidated_into, green.valid_to) == (None, None, None)
    assert store.revisions([new.id]) == {}  # what the operation created is gone


def test_consolidations_and_themes_go_to_review_by_default(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    """Measured with qwen3.5:9b: the one consolidation proposed was wrong, and no
    theme was ever proposed, so both wait for a reviewer unless the operator
    says auto."""
    run = new_run(journal)
    folded = submit(
        store,
        applier,
        run,
        _consolidate(memory, [memory.tea.id, memory.green_tea.id], "Karim drinks green tea."),
    )
    theme = submit(
        store,
        applier,
        run,
        {"op": "create_theme", "name": "Family", "members": [memory.karim.id, memory.kiki.id]},
    )
    assert (_status(folded), _status(theme)) == (("queued", None), ("queued", None))
    assert store.get_fact(memory.tea.id).valid_to is None
    assert store.find_node("Family", Visibility.of_scope(SCOPE)) is None


def test_consolidate_keeps_a_key_and_value_every_original_shares(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    again = store.add_fact(
        scopes=[SCOPE],
        node_id=memory.karim.id,
        statement="Karim has his home in Lyon.",
        origin="extracted",
        attribute="home",
        value="lyon",
        language="en",
    )
    entry = submit(
        store,
        applier,
        new_run(journal, policy=MEDIUM_AUTO),
        _consolidate(memory, [memory.home.id, again.id], "Karim lives in Lyon."),
    )
    new = store.get_fact(store.get_fact(memory.home.id).consolidated_into or "")
    assert entry.status == "applied"
    assert (new.attribute, new.value) == ("home", "Lyon")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("team", "Vega"), ("team", "Lyra")),  # two values of a multi-valued key
        (("home", "Lyon"), ("workplace", "Lyon")),  # one value under two keys
        ((None, None), (None, None)),  # no key: nothing says they are one thing
        (("team", "Vega"), (None, None)),
    ],
)
def test_a_consolidation_must_say_one_thing_again(
    store: Store,
    applier: Applier,
    journal: Journal,
    memory: Memory,
    first: tuple[str | None, str | None],
    second: tuple[str | None, str | None],
) -> None:
    """SPEC 5.6 rule 14: the service checks the originals share one key and an
    equivalent value. Measured: qwen3.5:9b folded two team facts into "... in
    the Atlas and Orion teams." - a dream may not join two values in one."""
    ids = [
        store.add_fact(
            scopes=[SCOPE],
            node_id=memory.karim.id,
            statement=statement,
            origin="extracted",
            attribute=key,
            value=value,
            language="en",
        ).id
        for statement, (key, value) in (
            ("Karim works on the Vega team.", first),
            ("Karim is a member of the Lyra team.", second),
        )
    ]
    entry = submit(
        store,
        applier,
        new_run(journal, policy=MEDIUM_AUTO),
        _consolidate(memory, ids, "Karim is in the Vega and Lyra teams."),
    )
    assert _status(entry) == ("dropped", "other_value")
    assert all(store.get_fact(i).valid_to is None for i in ids)


def test_a_consolidation_takes_one_value_written_two_ways(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    """Equivalent is one spelling up to case and accents (`same_spelling`), not
    the one-letter tolerance: a wrong SAME here would lose a value."""
    ids = [
        store.add_fact(
            scopes=[SCOPE],
            node_id=memory.karim.id,
            statement=statement,
            origin="extracted",
            attribute="workplace",
            value=value,
            language="en",
        ).id
        for statement, value in (
            ("Karim works at the Montréal office.", "Montréal"),
            ("Karim works in Montreal.", "montreal"),
            ("Karim works in Montrea.", "Montrea"),
        )
    ]
    run = new_run(journal, policy=MEDIUM_AUTO)
    statement = "Karim works at the Montréal office."
    one_letter = submit(store, applier, run, _consolidate(memory, ids, statement))
    assert _status(one_letter) == ("dropped", "other_value")
    entry = submit(store, applier, run, _consolidate(memory, ids[:2], statement))
    assert entry.status == "applied"
    new = store.get_fact(store.get_fact(ids[0]).consolidated_into or "")
    assert (new.attribute, new.value) == ("workplace", "Montréal")


def _bare_fact(fact_id: str, attribute: str | None, value: str | None, **kw: Any) -> Fact:
    return Fact(
        id=fact_id,
        node_id="n1",
        statement=f"{attribute} {value}",
        attribute=attribute,
        value=value,
        valid_from=None,
        valid_to=None,
        origin="extracted",
        confidence=0.9,
        source_id=None,
        excerpt_start=None,
        excerpt_end=None,
        superseded_by=None,
        pinned=False,
        flags=(),
        scopes=(SCOPE,),
        external_id=None,
        created_at="",
        updated_at="",
        archived_at=None,
        **kw,
    )


def test_one_same_as_group_counts_as_one_value_in_rule_14() -> None:
    # rule 13 keeps groups out of a consolidation today, so _consolidate never
    # reaches this; rule 14 still holds on its own (defense in depth)
    lyon = _bare_fact("f1", "home", "Lyon")
    lione = _bare_fact("f2", "home", "Lione", same_as="f1")
    lione_too = _bare_fact("f3", "home", "Lyon city", same_as="f1")
    assert says_one_thing([lyon, lione, lione_too])  # one group, three spellings
    assert says_one_thing([lione, lione_too])  # two members, their root left out
    # the same spelling outside the group is not enough
    assert not says_one_thing([lyon, _bare_fact("f4", "home", "Lione")])
    # one stranger spoils the whole set
    assert not says_one_thing([lyon, lione, _bare_fact("f5", "home", "Rome")])
    # a group never joins two keys, nor makes keyless facts say one thing
    assert not says_one_thing([lyon, _bare_fact("f6", "workplace", "Lyon", same_as="f1")])
    assert not says_one_thing(
        [_bare_fact("f7", None, None), _bare_fact("f8", None, None, same_as="f7")]
    )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("explicit", "explicit"),
        ("pinned", "pinned"),
        ("conflict", "conflict"),
        ("same_as", "same_as"),
        ("languages", "language"),
        ("other_node", "wrong_node"),
        ("closed", "not_live"),
        ("scopes", "scope_mismatch"),
        ("fragment", "incomplete"),
        ("unnamed", "unnamed"),
        ("wrong_language", "language_mismatch"),
        ("mixed", "mixed_script"),
    ],
)
def test_consolidate_drops_what_it_may_not_touch(
    store: Store,
    applier: Applier,
    journal: Journal,
    memory: Memory,
    case: str,
    reason: str,
) -> None:
    facts = [memory.tea.id, memory.green_tea.id]
    statement = "Karim drinks tea every morning."

    def extra(**kwargs: Any) -> str:
        values: dict[str, Any] = {
            "scopes": [SCOPE],
            "node_id": memory.karim.id,
            "statement": "Karim enjoys a cup of tea.",
            "origin": "extracted",
            "attribute": "routine",
            "value": "morning drink",
            "language": "en",
        }
        values.update(kwargs)
        return store.add_fact(**values).id

    if case == "explicit":
        facts = [memory.tea.id, memory.pinned.id]
    elif case == "pinned":
        facts = [memory.tea.id, extra(pinned=True)]
    elif case == "conflict":
        facts = [memory.tea.id, extra(flags=["conflict"])]
    elif case == "same_as":
        facts = [memory.tea.id, extra(same_as=memory.green_tea.id)]
    elif case == "languages":
        facts = [memory.tea.id, extra(statement="Karim aime le thé vert.", language="fr")]
    elif case == "other_node":
        facts = [memory.tea.id, extra(node_id=memory.kiki.id, statement="Kiki likes tea a lot.")]
    elif case == "closed":
        store.close_fact(memory.green_tea.id)
    elif case == "scopes":
        facts = [memory.tea.id, extra(scopes=[SCOPE, f"{SCOPE}/project:solar"])]
    elif case == "fragment":
        statement = "Tea."
    elif case == "unnamed":
        statement = "Someone drinks tea every morning."
    elif case == "wrong_language":
        statement = "Карим пьёт чай каждое утро."
    elif case == "mixed":
        statement = "Karim drinks чайtea every morning."
    entry = submit(store, applier, new_run(journal), _consolidate(memory, facts, statement))
    assert _status(entry) == ("dropped", reason)
    assert store.get_fact(memory.tea.id).consolidated_into is None


# --------------------------------------------------------------------------- merge_nodes


def _with_kiki_history(store: Store, memory: Memory) -> tuple[str, str, str]:
    fact = store.add_fact(
        scopes=[SCOPE],
        node_id=memory.kiki.id,
        statement="Kiki plays the violin.",
        origin="extracted",
        language="en",
    ).id
    edge = store.add_edge(
        scopes=[SCOPE], src=memory.kiki.id, dst=memory.lyon.id, type="lives_in", origin="test"
    ).id
    loop = store.add_edge(
        scopes=[SCOPE], src=memory.karim.id, dst=memory.kiki.id, type="knows", origin="test"
    ).id
    return fact, edge, loop


def test_merge_nodes_goes_to_review_by_default(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "merge_nodes", "keep": memory.karim.id, "absorb": memory.kiki.id},
    )
    assert _status(entry) == ("queued", None)
    assert store.get_node(memory.kiki.id).archived_at is None


def test_merge_nodes_goes_to_review_even_when_the_policy_says_auto(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    run = new_run(journal, policy=AUTO_ALL)
    entry = submit(
        store,
        applier,
        run,
        {"op": "merge_nodes", "keep": memory.karim.id, "absorb": memory.kiki.id},
    )
    assert entry.status == "queued"


def test_an_accepted_merge_moves_everything_and_undo_moves_it_back(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    fact, edge, loop = _with_kiki_history(store, memory)
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.karim.id, value=memory.kiki.id
    )
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "merge_nodes", "keep": memory.karim.id, "absorb": memory.kiki.id},
    )
    decided = applier.decide(entry.id, accept=True)
    assert decided.status == "applied"
    kiki, karim = store.get_node(memory.kiki.id), store.get_node(memory.karim.id)
    assert kiki.archived_at is not None
    assert kiki.merged_into == memory.karim.id
    assert "Kiki" in karim.aliases and "K." in karim.aliases
    assert store.get_fact(fact).node_id == memory.karim.id
    moved = store.get_edge(edge)
    assert (moved.src, moved.dst) == (memory.karim.id, memory.lyon.id)
    assert store.get_edge(loop).valid_to is not None  # it would join Karim to himself
    assert store.get_proposal(proposal.id).status == "accepted"

    applier.undo_op(entry.id)
    kiki, karim = store.get_node(memory.kiki.id), store.get_node(memory.karim.id)
    assert (kiki.archived_at, kiki.merged_into) == (None, None)
    assert "Kiki" not in karim.aliases and "K." in karim.aliases
    assert store.get_fact(fact).node_id == memory.kiki.id
    back = store.get_edge(edge)
    assert (back.src, back.dst) == (memory.kiki.id, memory.lyon.id)
    restored_loop = store.get_edge(loop)
    assert (restored_loop.src, restored_loop.dst, restored_loop.valid_to) == (
        memory.karim.id,
        memory.kiki.id,
        None,
    )
    assert store.get_proposal(proposal.id).status == "pending"


def test_a_rejected_merge_rejects_its_proposal(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    proposal = store.add_proposal(
        scopes=[SCOPE], kind="merge", node_id=memory.kiki.id, value=memory.karim.id
    )
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "merge_nodes", "keep": memory.karim.id, "absorb": memory.kiki.id},
    )
    assert applier.decide(entry.id, accept=False).status == "rejected"
    assert store.get_proposal(proposal.id).status == "rejected"
    assert store.get_node(memory.kiki.id).archived_at is None


@pytest.mark.parametrize(("case", "reason"), [("kinds", "kind_mismatch"), ("same", "same_record")])
def test_merge_nodes_drops_what_it_cannot_tell(
    store: Store, applier: Applier, journal: Journal, memory: Memory, case: str, reason: str
) -> None:
    absorb = memory.lyon.id if case == "kinds" else memory.karim.id
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "merge_nodes", "keep": memory.karim.id, "absorb": absorb},
    )
    assert _status(entry) == ("dropped", reason)


# --------------------------------------------------------------------------- supersede_fact


def _paris(store: Store, memory: Memory, **kwargs: Any) -> str:
    values: dict[str, Any] = {
        "scopes": [SCOPE],
        "node_id": memory.karim.id,
        "statement": "Karim lives in Paris.",
        "origin": "extracted",
        "attribute": "home",
        "value": "Paris",
        "language": "en",
        "valid_from": "2026-09-01",
    }
    values.update(kwargs)
    return store.add_fact(**values).id


def test_supersede_fact_goes_to_review_by_default(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    paris = _paris(store, memory)
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "supersede_fact", "fact": memory.home.id, "by": paris},
    )
    assert _status(entry) == ("queued", None)


def test_supersede_fact_closes_and_undo_reopens(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    paris = _paris(store, memory)
    run = new_run(journal, policy=AUTO_ALL)
    entry = submit(
        store, applier, run, {"op": "supersede_fact", "fact": memory.home.id, "by": paris}
    )
    assert _status(entry) == ("applied", None)
    home = store.get_fact(memory.home.id)
    assert (home.valid_to, home.superseded_by) == ("2026-09-01", paris)
    applier.undo_op(entry.id)
    home = store.get_fact(memory.home.id)
    assert (home.valid_to, home.superseded_by) == (None, None)


@pytest.mark.parametrize(
    ("case", "cause"),
    [
        ("explicit", "explicit"),
        ("same_value", "same_value"),
        ("several", "several"),
        ("key", "key"),
        ("earlier", "earlier"),
        ("external", "external"),
        ("unproven", "unproven"),
        ("node", "node"),
    ],
)
def test_supersede_fact_the_write_path_would_not_make_is_downgraded_to_a_conflict(
    store: Store, applier: Applier, journal: Journal, memory: Memory, case: str, cause: str
) -> None:
    old = memory.home.id
    if case == "explicit":
        old = store.add_fact(
            scopes=[SCOPE],
            node_id=memory.karim.id,
            statement="Karim lives in Lyon for sure.",
            origin="explicit",
            attribute="home",
            value="Lyon",
            language="en",
        ).id
        by = _paris(store, memory)
    elif case == "same_value":
        by = _paris(store, memory, statement="Karim lives in Lyon, France.", value="lyon")
    elif case == "several":
        old = store.add_fact(
            scopes=[SCOPE],
            node_id=memory.karim.id,
            statement="Karim is a teacher.",
            origin="extracted",
            attribute="role",
            value="teacher",
            language="en",
        ).id
        by = _paris(
            store,
            memory,
            statement="Karim is a photographer.",
            attribute="role",
            value="photographer",
        )
    elif case == "key":
        by = _paris(store, memory, attribute="workplace")
    elif case == "earlier":
        store.close_fact(old)  # make room: an older fact stated later in the store
        old = _paris(store, memory, valid_from="2026-09-01")
        by = store.add_fact(
            scopes=[SCOPE],
            node_id=memory.karim.id,
            statement="Karim lives in Lyon again.",
            origin="extracted",
            attribute="home",
            value="Lyon",
            language="en",
            valid_from="2020-01-01",
        ).id
    elif case == "external":
        store.close_fact(old)
        old = _paris(
            store,
            memory,
            statement="Karim lives in Nice.",
            value="Nice",
            provenance="owner",
            valid_from="2025-01-01",
        )
        by = _paris(store, memory, provenance="external")
    elif case == "unproven":  # across languages, a value not proven different
        by = _paris(store, memory, statement="Карим живёт в Париже.", value="Париж", language="ru")
    else:
        by = store.add_fact(
            scopes=[SCOPE],
            node_id=memory.kiki.id,
            statement="Kiki lives in Paris.",
            origin="extracted",
            attribute="home",
            value="Paris",
            language="en",
            valid_from="2026-09-01",
        ).id
    run = new_run(journal, policy=AUTO_ALL)
    entry = submit(store, applier, run, {"op": "supersede_fact", "fact": old, "by": by})
    assert entry.op == "flag_conflict"
    assert entry.status == "applied"
    assert entry.reason == f"downgraded:{cause}"
    assert store.get_fact(old).valid_to is None  # nothing closed
    flagged = store.get_fact(by)
    assert "conflict" in flagged.flags
    assert flagged.conflicts_with == old


# --------------------------------------------------------------------------- flag_conflict


def test_flag_conflict_is_allowed_on_an_explicit_fact_and_undo_unflags(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {"op": "flag_conflict", "fact": memory.pinned.id, "with": memory.tea.id},
    )
    assert _status(entry) == ("applied", None)
    pinned = store.get_fact(memory.pinned.id)
    assert ("conflict" in pinned.flags, pinned.conflicts_with) == (True, memory.tea.id)
    assert pinned.statement == "Karim is allergic to peanuts."  # nothing else changed
    applier.undo_op(entry.id)
    pinned = store.get_fact(memory.pinned.id)
    assert (pinned.flags, pinned.conflicts_with) == ((), None)


def test_flag_conflict_drops_a_duplicate_and_a_self_conflict(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    store.flag_fact(memory.tea.id, "conflict", conflicts_with=memory.green_tea.id)
    dup = submit(
        store,
        applier,
        new_run(journal),
        {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id},
    )
    assert _status(dup) == ("dropped", "duplicate")
    same = submit(
        store,
        applier,
        new_run(journal),
        {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.tea.id},
    )
    assert _status(same) == ("dropped", "same_record")


# --------------------------------------------------------------------------- common rules


def test_an_operation_naming_a_record_outside_its_unit_is_dropped(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    op = {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id}
    entry = applier.submit(
        new_run(journal), op, store.revisions([memory.tea.id]), unit_ids={memory.tea.id}
    )
    assert _status(entry) == ("dropped", "unknown_record")
    assert store.get_fact(memory.tea.id).flags == ()


def test_an_operation_on_a_forgotten_record_is_dropped(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    op = {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id}
    snapshot = store.revisions([memory.tea.id, memory.green_tea.id])
    store.forget(memory.green_tea.id)
    entry = applier.submit(new_run(journal), op, snapshot)
    assert _status(entry) == ("dropped", "stale")


def test_a_record_outside_the_dreams_scope_is_never_changed(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    """Rule 12: a dream on user:alice/agent:coder reads user:alice as context and
    changes none of it."""
    run = new_run(journal, scope=f"{SCOPE}/agent:coder")
    entry = submit(
        store,
        applier,
        run,
        {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id},
    )
    assert _status(entry) == ("dropped", "outside_scope")
    assert store.get_fact(memory.tea.id).flags == ()


def test_a_dry_run_plans_and_changes_nothing(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    before = store.revisions([memory.tea.id, memory.green_tea.id])
    entry = submit(
        store,
        applier,
        new_run(journal, dry_run=True),
        {"op": "flag_conflict", "fact": memory.tea.id, "with": memory.green_tea.id},
    )
    assert _status(entry) == ("planned", None)
    assert store.revisions([memory.tea.id, memory.green_tea.id]) == before


def test_a_dry_run_still_drops_what_would_be_dropped(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal, dry_run=True),
        {"op": "refresh_summary", "node": memory.kiki.id, "summary": "Kiki is someone."},
    )
    assert _status(entry) == ("dropped", "no_facts")


def test_the_journal_keeps_the_models_reason_and_the_records(
    store: Store, applier: Applier, journal: Journal, memory: Memory
) -> None:
    entry = submit(
        store,
        applier,
        new_run(journal),
        {
            "op": "flag_conflict",
            "fact": memory.tea.id,
            "with": memory.green_tea.id,
            "reason": "two drinks",
        },
    )
    stored = journal.get_op(entry.id)
    assert stored.reason_given == "two drinks"
    assert set(stored.records) >= {memory.tea.id, memory.green_tea.id}
    assert stored.risk == "low"
