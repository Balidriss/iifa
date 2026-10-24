"""Checks on what a model writes, for the failures real local models showed
(SPEC 4.2, 5.1, 5.2). Pure functions over text, every script, standard library
only.

* ``is_complete`` / ``names_subject``: an extracted statement is a sentence
  that names its subject, never a fragment ("dirige", "déteste"). A fragment
  says nothing to a reader and carries too little text for the language check
  of P12.
* ``grounded``: an alias is kept only when the source holds it (a model that
  attaches "Карим" to Иван would otherwise merge two people for good).
* ``cooccurring``: subjects are linked by co-occurrence only within one
  sentence, never merely because they share a turn.

Scripts written without spaces (Chinese, Japanese, Thai) have no words to count
or to match whole: there, length counts characters and names match as
substrings. Known limitation, on purpose not guessed: Arabic attached prefixes
(و، ب، ل، لا) are not split, so "وكريم" does not name كريم.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from iifa.language import script_counts, sentences
from iifa.recall.tokenizer import normalize, tokenize

MIN_WORDS = 3
MIN_UNSPACED_CHARS = 6
LONG_NAME_WORD = 4  # a word of a multi-word name this long names it on its own

_UNSPACED_SCRIPTS = ("han", "kana", "thai")
# straight, typographic (U+2019), modifier letter (U+02BC), grave and acute apostrophes
_APOSTROPHES = "'" + chr(0x2019) + chr(0x02BC) + "`" + chr(0x00B4)
# French and Italian elided articles, pronouns and conjunctions (l'avis, d'Émilie,
# qu'il, dell'anno): set aside before counting words or matching a name.
_ELIDED = "l|d|j|m|n|s|t|c|qu|lorsqu|puisqu|jusqu|dell|dall|nell|sull|all|un"
_ELISION = re.compile(
    rf"\b(?:{_ELIDED})[{_APOSTROPHES}]",
    re.IGNORECASE,
)


def _unelided(text: str) -> str:
    return _ELISION.sub(" ", text)


def words(text: str) -> list[str]:
    """The folded words of ``text``, elided articles set aside
    (``"l'installation d'Émilie"`` -> ``["installation", "emilie"]``)."""
    return tokenize(_unelided(text))


def unspaced_chars(text: str) -> int:
    """Letters of the scripts written without spaces between words."""
    counts = script_counts(text)
    return sum(counts[script] for script in _UNSPACED_SCRIPTS)


def is_complete(statement: str) -> bool:
    """A sentence, not a fragment: at least three words, or at least six letters
    of a script written without spaces."""
    if unspaced_chars(statement) >= MIN_UNSPACED_CHARS:
        return True
    spaced = [w for w in words(statement) if not unspaced_chars(w)]
    return len(spaced) >= MIN_WORDS


def _word_names(word: str, name_word: str) -> bool:
    """Does ``word`` of a statement stand for ``name_word`` of a name, allowing an
    inflected ending (Карим -> Карима, Иван -> Иваном) but not a longer, different
    word (Karim -> Karimova)?"""
    if word == name_word:
        return True
    length = len(name_word)
    if length < 4:
        return False
    stem = name_word if length == 4 else name_word[:-1]
    return word.startswith(stem) and len(word) <= length + 2


def _one_name(statement_words: Sequence[str], folded: str, name: str) -> bool:
    if unspaced_chars(name):
        return normalize(name).replace(" ", "") in folded
    name_words = words(name)
    if not name_words:
        return False

    def present(name_word: str) -> bool:
        return any(_word_names(w, name_word) for w in statement_words)

    long_words = [w for w in name_words if len(w) >= LONG_NAME_WORD]
    if len(name_words) > 1 and long_words:
        return any(present(w) for w in long_words)
    return all(present(w) for w in name_words)


def shares_script(statement: str, names: Iterable[str]) -> bool:
    """Is one of ``names`` written in the statement's script? When none is, a
    statement cannot name its subject whatever it says (the cause the retry
    note spells out, SPEC 4.2)."""
    written = script_of(statement)
    return any(n.strip() and script_of(n) == written for n in names)


def scripts_in(text: str) -> set[str]:
    """Every script ``text`` has letters in (Chinese characters and kana as one)."""
    counts = script_counts(text)
    found = {script for script, n in counts.items() if n}
    if found & {"han", "kana"}:
        found = (found - {"han", "kana"}) | {"cjk"}
    return found


def script_of(text: str) -> str | None:
    """The main script of ``text``; Chinese characters and kana count as one
    (a Japanese sentence mixes them), ``None`` for a text with no letters."""
    counts = script_counts(text)
    counts["cjk"] = counts.pop("han", 0) + counts.pop("kana", 0)
    counts = +counts
    if not counts:
        return None
    return max(sorted(counts), key=lambda script: counts[script])


# scripts written with spaces: two of them inside one word is a malformed word
_SPACED_SCRIPTS = frozenset({"latin", "cyrillic", "greek", "arabic", "hebrew", "devanagari"})
_LETTERS = re.compile(r"[^\W\d_]+")  # a run of letters: digits, marks and punctuation split


def mixed_script_words(statement: str) -> list[str]:
    """The words of ``statement`` whose letters belong to two scripts written with
    spaces ("كazan": Arabic then Latin), in order (SPEC 4.2). What cannot be told apart is
    never reported: a Latin word inside Chinese or Japanese (no spaces there),
    Chinese characters with kana, digits, marks, letters of other scripts."""
    out: list[str] = []
    for word in _LETTERS.findall(statement):
        scripts = {s for s, n in script_counts(word).items() if n} & _SPACED_SCRIPTS
        if len(scripts) > 1:
            out.append(word)
    return out


def names_subject(statement: str, names: Iterable[str]) -> bool:
    """Does ``statement`` name its subject by one of ``names`` (its name or an
    alias)? Case, Latin diacritics and elision are forgiven ("d'Émilie" names
    Émilie), so is an inflected ending; a name of several words is named by any
    of its longer words ("Karim" names Karim Benali).

    Only the names written in the statement's script count: "Karim" does not
    name the subject of "Карим переехал в Лион." (matching across scripts needs
    the model's help, an alias, or embeddings at tier 3). A translated statement is
    the caller's to exempt (P12 judges it)."""
    written = script_of(statement)
    comparable = [n for n in names if n.strip() and script_of(n) == written]
    if not comparable:
        return False
    statement_words = words(statement)
    folded = "".join(tokenize(statement))
    return any(_one_name(statement_words, folded, name) for name in comparable)


_WORD = re.compile(r"[^\W_]+")


def source_spelling(alias: str, text: str) -> str | None:
    """The span of the source ``text`` that holds ``alias`` - as whole words (case
    and Latin diacritics folded), or as a substring in a script written without
    spaces - exactly as the source writes it; None when the source does not hold
    it. An alias is stored with this spelling, never the model's copy."""
    wanted = tokenize(alias)
    if not wanted:
        return None
    spans = [(m.start(), m.end(), normalize(m.group())) for m in _WORD.finditer(text)]
    if unspaced_chars(alias):
        needle = "".join(wanted)
        for start, end, token in spans:
            at = token.find(needle)
            if at >= 0 and len(token) == end - start:  # folding kept the length
                return text[start + at : start + at + len(needle)]
        return None
    for i in range(len(spans) - len(wanted) + 1):
        if [token for _, _, token in spans[i : i + len(wanted)]] == wanted:
            return text[spans[i][0] : spans[i + len(wanted) - 1][1]]
    return None


def grounded(alias: str, text: str) -> bool:
    """Does the source ``text`` hold ``alias`` (see ``source_spelling``)?"""
    return source_spelling(alias, text) is not None


def _mentions(sentence: str, forms: Iterable[str]) -> bool:
    return any(grounded(form, sentence) for form in forms)


def cooccurring(
    text: str,
    forms: Mapping[str, Iterable[str]],
    spans: Mapping[str, Sequence[tuple[int, int]]],
    *,
    order: Sequence[str],
    limit: int,
) -> list[tuple[str, str]]:
    """The pairs of subjects that share a sentence of ``text`` (SPEC 5.2), in
    ``order``, at most ``limit`` subjects per sentence, each pair once.

    A subject is in a sentence when one of its ``forms`` (name, aliases, the
    mentions that resolved to it) appears there, or when one of its ``spans``
    (the quotes of its facts and edges, as offsets into ``text``) overlaps it."""
    materialised = {key: [f for f in values if f] for key, values in forms.items()}
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for start, end in sentences(text):
        sentence = text[start:end]
        present = [
            key
            for key in order
            if _mentions(sentence, materialised.get(key, ()))
            or any(s < end and e > start for s, e in spans.get(key, ()))
        ][:limit]
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                pair = frozenset((a, b))
                if a != b and pair not in seen:
                    seen.add(pair)
                    pairs.append((a, b))
    return pairs


__all__ = [
    "MIN_UNSPACED_CHARS",
    "MIN_WORDS",
    "cooccurring",
    "grounded",
    "is_complete",
    "names_subject",
    "script_of",
    "scripts_in",
    "shares_script",
    "source_spelling",
    "unspaced_chars",
    "words",
]
