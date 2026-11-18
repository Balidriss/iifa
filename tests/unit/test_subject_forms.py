"""One subject in two forms (SPEC 5.1): a file name or slug and a title are
candidates of each other for the adjudication, never a match; how they are
written, never a list of words, decides."""

from __future__ import annotations

import pytest

from iifa.ingest.resolve import (
    ADJUDICATED,
    AMBIGUOUS,
    EXACT,
    FUZZY,
    NEW,
    Resolver,
    form_duplicate,
    form_words,
    same_form,
    written_as_slug,
)
from iifa.scopes import Visibility
from iifa.store.repo import Store

ALICE = Visibility.of_scope("user:alice")


def test_form_words_split_on_separators_and_drop_a_file_extension() -> None:
    assert form_words("le-jardin-du-nord.pdf") == {"le", "jardin", "du", "nord"}
    assert form_words("le_jardin_du_nord") == {"le", "jardin", "du", "nord"}
    assert form_words("Résumé : Le jardin du nord") == {"resume", "le", "jardin", "du", "nord"}
    # a dot followed by digits only, or a name with spaces, keeps its last word
    assert form_words("release-2.0") == {"release", "2", "0"}
    assert form_words("Notes du 3. pdf") == {"notes", "du", "3", "pdf"}
    assert form_words("Jardin du nord.pdf") == {"jardin", "du", "nord"}  # spaces too
    assert form_words("") == frozenset()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("le jardin du nord", "Résumé jardin du nord"),  # one word apart on each side
        ("le-jardin-du-nord.pdf", "Résumé : Le jardin du nord"),  # file name and title
        ("jardin-du-nord.pdf", "Résumé jardin du nord"),
        ("mon_post_du_lundi", "Post du lundi"),
        ("tokyo-trip-notes-2026", "東京 trip notes 2026"),  # words of any script
        ("le-jardin-du-nord.pdf", "le jardin du nord"),  # the same words, written apart
    ],
)
def test_two_forms_of_one_subject(a: str, b: str) -> None:
    assert same_form(a, b) and same_form(b, a)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Projet nord bleu", "Projet nord vert"),  # two words in common, two apart
        ("Karim", "Karima"),  # one word: letters are the near-name rule's business
        ("Karim Benali", "Karim Haddad"),
        ("Jean Pierre Martin", "Jean Pierre Dupont"),
        ("rapport annuel du service", "rapport mensuel du service public"),  # two on one side
        ("tokyo-trip-notes", "東京 trip notes"),  # two in common, two apart
        ("", "le jardin du nord"),
    ],
)
def test_not_two_forms_of_one_subject(a: str, b: str) -> None:
    assert not same_form(a, b)


@pytest.mark.parametrize(
    ("name", "other", "slug"),
    [
        ("le-jardin-du-nord", "Résumé jardin du nord", True),  # words joined without spaces
        ("jardin.pdf", "Résumé jardin", True),  # a file name
        ("Jardin du nord.pdf", "Résumé jardin du nord", True),  # with spaces too
        ("Le_Jardin_Du_Nord", "Résumé jardin du nord", True),  # an underscore joins
        ("Jean-Pierre", "Jean-Pierre Martin", False),  # a hyphenated name is a name
        ("Saint-Étienne", "Saint-Étienne Métropole", False),
        ("le jardin du nord", "Résumé jardin du nord", True),  # no capital where the other has
        ("Le jardin du nord", "Résumé jardin du nord", False),
        ("Photo jardin du nord", "Résumé jardin du nord", False),
        ("le jardin du nord", "résumé jardin du nord", False),  # neither has a capital
        ("東京", "Tokyo", False),  # a word with no case is not lower case
        ("Lyon", "lyon", False),  # one word: not joined, not a file
    ],
)
def test_written_as_slug(name: str, other: str, slug: bool) -> None:
    assert written_as_slug(name, other) is slug


def test_a_title_and_its_slug_are_candidates_never_a_match(store: Store) -> None:
    north = store.create_node(scopes=["user:alice"], name="le jardin du nord", kind="object")
    resolver = Resolver(store, ALICE)
    for mention in ("Résumé jardin du nord", "le_jardin_du_nord.pdf", "jardin-du-nord.pdf"):
        resolution = resolver.resolve(mention)
        assert resolution.key is None and resolution.method == AMBIGUOUS, mention
        assert [c.key for c in resolution.candidates] == [north.id]
    # normalisation already makes separators spaces and folds case
    exact = resolver.resolve("Le-Jardin-Du-Nord")
    assert exact.key == north.id and exact.method == EXACT


def test_a_form_candidate_never_turns_a_containment_into_a_match(store: Store) -> None:
    """Whole-word containment still attaches when the mention's words are all
    words of exactly one name; a second node in another form makes it ambiguous."""
    resume = store.create_node(scopes=["user:alice"], name="Résumé jardin du nord")
    resolver = Resolver(store, ALICE)
    assert resolver.resolve("jardin du nord").key == resume.id
    assert resolver.resolve("jardin du nord").method == FUZZY
    photo = store.create_node(scopes=["user:alice"], name="Photo jardin du nord")
    both = Resolver(store, ALICE).resolve("jardin du nord")
    assert both.method == AMBIGUOUS and {c.key for c in both.candidates} == {resume.id, photo.id}


def test_words_in_common_are_not_enough(store: Store) -> None:
    store.create_node(scopes=["user:alice"], name="Projet nord bleu")
    assert Resolver(store, ALICE).resolve("Projet nord vert").method == NEW


def test_a_form_candidate_is_found_through_an_alias_too(store: Store) -> None:
    node = store.create_node(scopes=["user:alice"], name="Clip", aliases=["le jardin du nord"])
    resolution = Resolver(store, ALICE).resolve("Résumé du jardin du nord")
    assert resolution.method == AMBIGUOUS and [c.key for c in resolution.candidates] == [node.id]


def test_form_nodes_name_the_known_nodes_in_another_form(store: Store) -> None:
    """What a new node may be a duplicate of: known nodes only (not the ones the
    same answer proposes), in two forms of one subject, one written as a slug."""
    slug = store.create_node(scopes=["user:alice"], name="le-jardin-du-nord.pdf", kind="object")
    store.create_node(scopes=["user:alice"], name="Photo jardin du nord", kind="object")
    store.create_node(scopes=["user:alice"], name="Karima", kind="person")
    store.create_node(scopes=["user:bob"], name="jardin_du_nord")  # invisible (P9)
    resolver = Resolver(store, ALICE)
    resolver.propose("new:0", "jardin-du-nord", "object")
    assert resolver.form_nodes("Résumé jardin du nord", "object") == [slug.id]
    assert resolver.form_nodes("Résumé jardin du nord", "other") == [slug.id]
    assert resolver.form_nodes("Résumé jardin du nord", "person") == []
    assert resolver.form_nodes("Karim", "person") == []
    assert resolver.form_nodes("Résumé Jardin du parc", "object") == []


def test_a_form_candidate_never_attaches_whatever_the_threshold(store: Store) -> None:
    """A containment the letter score leaves below the threshold stays out of
    step 2: the form rule makes a candidate, never a match."""
    resume = store.create_node(scopes=["user:alice"], name="Résumé jardin du nord")
    strict = Resolver(store, ALICE, fuzzy_threshold=0.95).resolve("jardin du nord")
    assert strict.key is None and strict.method == AMBIGUOUS
    assert [c.key for c in strict.candidates] == [resume.id]
    assert strict.method != ADJUDICATED


@pytest.mark.parametrize(
    ("name", "kind", "other", "other_kind", "paired"),
    [
        ("Résumé jardin du nord", "object", "le-jardin-du-nord.pdf", "object", True),
        ("Résumé jardin du nord", "other", "Jardin du nord.pdf", "object", True),
        ("Jean-Pierre Martin", "person", "Jean-Pierre", "person", False),
        ("Saint-Étienne Métropole", "place", "Saint-Étienne", "place", False),
        ("Résumé jardin du nord", "person", "le-jardin-du-nord", "object", False),
        ("le-jardin-du-nord", "object", "Le jardin du nord", "object", False),  # one name
    ],
)
def test_form_duplicate(name: str, kind: str, other: str, other_kind: str, paired: bool) -> None:
    assert form_duplicate(name, kind, other, other_kind) is paired
