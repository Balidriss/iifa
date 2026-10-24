"""Language detection for P12, with the standard library only.

Memory keeps the language it was told in: a fact or a summary is written in the
language of its source, the target language is stated in every writing prompt,
and a written item whose language differs from its source's is rejected,
retried once, then stored flagged. All of that needs a detector that works
offline, never downloads anything (P4) and costs nothing to ship. This one
looks at two things:

* the **script** of the letters (Latin, Cyrillic, Arabic, Han, kana, Hangul,
  Greek, Hebrew, Devanagari, Thai): for a non-Latin script the language follows
  from the script plus a few telltale letters (Ukrainian ``ї``, Persian ``پ``,
  kana beside Han for Japanese);
* for Latin text, **function words**: the short closed-class words every
  sentence needs (``the``, ``of``, ``le``, ``des``, ``und``, ``nicht``...),
  which a proper noun or a technical term never contains, with accented
  letters as a tie-break. Words shared by two languages (``a``, ``on``,
  ``plus``) are ignored: they prove nothing.

The detector answers ``und`` (undetermined) rather than guessing when the text
holds no evidence, for instance ``Karim prefers tea``. An undetermined language
is neither a match nor a mismatch: the check is skipped, never faked. Mixed
sources are handled per sentence (``sentence_languages``): a fact is compared
with the language of the sentence it was taken from, not of the whole source.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

UNDETERMINED = "und"

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "fa": "Persian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "th": "Thai",
}

# --------------------------------------------------------------------------- scripts

_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "latin"),
    (0x0061, 0x007A, "latin"),
    (0x00C0, 0x024F, "latin"),
    (0x1E00, 0x1EFF, "latin"),
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x052F, "cyrillic"),
    (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
    (0x1100, 0x11FF, "hangul"),
    (0x3130, 0x318F, "hangul"),
    (0xAC00, 0xD7AF, "hangul"),
    (0x3040, 0x309F, "kana"),
    (0x30A0, 0x30FF, "kana"),
    (0x3400, 0x4DBF, "han"),
    (0x4E00, 0x9FFF, "han"),
    (0xF900, 0xFAFF, "han"),
)

_SCRIPT_LANGUAGE = {
    "greek": "el",
    "hebrew": "he",
    "devanagari": "hi",
    "thai": "th",
    "hangul": "ko",
    "kana": "ja",
    "han": "zh",
}

_UKRAINIAN_LETTERS = set("іїєґ")
_PERSIAN_LETTERS = set("پچژگ")


def _script_of(char: str) -> str | None:
    code = ord(char)
    for low, high, script in _SCRIPT_RANGES:
        if low <= code <= high:
            return script
    return None


def script_counts(text: str) -> Counter[str]:
    """How many letters of each script ``text`` holds (digits and punctuation ignored)."""
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        script = _script_of(char)
        if script is None:
            script = "other"
        counts[script] += 1
    return counts


# --------------------------------------------------------------------------- Latin function words


def _words(*lines: str) -> frozenset[str]:
    return frozenset(word for line in lines for word in line.split())


_STOPWORDS: dict[str, frozenset[str]] = {
    "en": _words(
        "the and of to in is are was were be been has have had that this it its he she",
        "they we you his her their not from by as but who which will would can since than",
        "there about into does did do when what where my our your an at for with or",
    ),
    "fr": _words(
        "le la les une des du de et ou à au aux en dans sur pour avec est sont été être",
        "ont qui que quoi ce cette ces il elle ils elles nous vous je tu sa ses leur leurs",
        "ne pas très mais depuis comme chez où l d j qu c n m aussi mon ma mes notre votre",
        "tout tous toute toutes ça son",
    ),
    "de": _words(
        "der die das und nicht ist sind mit auf für von zu den dem ein eine einen einer",
        "ich er sie es wir nach bei aus über auch wird werden hat haben seit im am zum",
        "zur des dass oder aber wenn noch schon in",
    ),
    "es": _words(
        "el los las una y o en es son fue por para con del al que se su sus lo no pero",
        "desde como muy este esta estos estas nosotros ellos ella ellas usted también",
    ),
    "it": _words(
        "il lo gli le una e o di che è sono per con del della dei delle nel nella al",
        "alla non ma da come molto questo questa questi anche suo sua loro noi voi in",
        "sul sulla sui sulle dal dalla nei negli",
    ),
    "pt": _words(
        "o os as um uma e ou em é são foi por para com do da dos das no na nos nas que",
        "se seu sua não mas desde como muito este esta nós eles elas também ao",
    ),
    "nl": _words(
        "het een en of van is zijn was waren met voor op te niet dat dit ik hij zij",
        "wij jullie ze hun naar bij uit over ook wordt worden heeft hebben sinds maar",
        "als in",
    ),
}

_shared = {
    word
    for words in _STOPWORDS.values()
    for word in words
    if sum(word in other for other in _STOPWORDS.values()) > 1
}
# A word two languages share proves nothing; a few are unique enough to keep.
_KEEP_SHARED = {
    "des": "fr",
    "est": "fr",
    "les": "fr",
    "le": "fr",
    "il": "fr",
    "of": "en",
    "el": "es",
}
_UNIQUE: dict[str, frozenset[str]] = {
    lang: frozenset(w for w in words if w not in _shared or _KEEP_SHARED.get(w) == lang)
    for lang, words in _STOPWORDS.items()
}

_ACCENT_HINTS: dict[str, frozenset[str]] = {
    "fr": frozenset("éèêëàâçùûôîïœ"),
    "de": frozenset("äöüß"),
    "es": frozenset("ñ¿¡"),
    "pt": frozenset("ãõ"),
    "it": frozenset("ìò"),
}

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_APOSTROPHES = ("'", "\u2019", "\u02bc")  # ', right single quotation mark, modifier apostrophe


def _latin_words(text: str) -> list[str]:
    flat = text.lower()
    for mark in _APOSTROPHES:
        flat = flat.replace(mark, " ")
    return [w for w in _WORD.findall(flat) if _script_of(w[0]) == "latin"]


def _latin_language(text: str) -> str:
    words = _latin_words(text)
    if not words:
        return UNDETERMINED
    scores: dict[str, float] = {}
    for lang, unique in _UNIQUE.items():
        hits = sum(1 for w in words if w in unique)
        accents = _ACCENT_HINTS.get(lang)
        hint = 0.0
        if accents:
            hint = min(2.0, 0.5 * sum(1 for ch in text.lower() if ch in accents))
        score = hits + hint
        if score > 0:
            scores[lang] = score
    if not scores:
        return UNDETERMINED
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best, best_score = ranked[0]
    if best_score < 1.0:
        return UNDETERMINED
    if len(ranked) > 1 and ranked[1][1] >= best_score:
        return UNDETERMINED
    return best


# --------------------------------------------------------------------------- detection


def detect(text: str) -> str:
    """The language of ``text`` as an ISO 639-1 code, or ``und`` when the text
    holds no evidence (a name, a number, a word that belongs to several
    languages)."""
    counts = script_counts(text)
    if not counts:
        return UNDETERMINED
    if counts["kana"]:
        return "ja"
    script, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if script == "latin":
        return _latin_language(text)
    if script == "cyrillic":
        lowered = text.lower()
        return "uk" if any(ch in _UKRAINIAN_LETTERS for ch in lowered) else "ru"
    if script == "arabic":
        return "fa" if any(ch in _PERSIAN_LETTERS for ch in text) else "ar"
    return _SCRIPT_LANGUAGE.get(script, UNDETERMINED)


_SENTENCE_END = re.compile("[.!?\u3002\uff01\uff1f\n]+")  # Latin and CJK sentence ends


def sentences(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each sentence of ``text``, split on sentence-ending
    punctuation (Latin and CJK) and line breaks, whitespace trimmed."""
    out: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        piece = (start, match.start())
        start = match.end()
        trimmed = _trim(text, *piece)
        if trimmed is not None:
            out.append(trimmed)
    trimmed = _trim(text, start, len(text))
    if trimmed is not None:
        out.append(trimmed)
    return out


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def sentence_languages(text: str) -> list[tuple[int, int, str]]:
    """Each sentence with its detected language: the unit P12 compares a fact with
    in a mixed-language source."""
    return [(start, end, detect(text[start:end])) for start, end in sentences(text)]


def language_of_span(text: str, start: int, end: int) -> str:
    """The language a fact taken from ``text[start:end]`` must be written in: the
    span's own language when it is determined, else the language of the
    sentences the span overlaps, else the language of the whole text."""
    start, end = max(0, start), min(len(text), end)
    if end > start:
        own = detect(text[start:end])
        if own != UNDETERMINED:
            return own
    overlapping = [s for s in sentences(text) if s[0] < end and s[1] > start]
    if overlapping:
        joined = " ".join(text[s:e] for s, e in overlapping)
        found = detect(joined)
        if found != UNDETERMINED:
            return found
    return detect(text)


def languages_present(text: str) -> list[str]:
    """The determined languages of the text's sentences, most frequent first,
    the whole text's own language first when it is determined."""
    counts: Counter[str] = Counter()
    for _, _, lang in sentence_languages(text):
        if lang != UNDETERMINED:
            counts[lang] += 1
    whole = detect(text)
    if not counts:
        return [] if whole == UNDETERMINED else [whole]
    return [
        lang
        for lang, _ in sorted(
            counts.items(), key=lambda kv: (-kv[1], 0 if kv[0] == whole else 1, kv[0])
        )
    ]


def mismatch(written: str, expected: str) -> bool:
    """Do two detected languages disagree? Only when both are determined."""
    return UNDETERMINED not in (written, expected) and written != expected


# the script each language is written in; a language missing here is never
# judged by script (P12: when we cannot tell, nothing is flagged)
_LATIN_LANGUAGES = frozenset({"en", "fr", "de", "es", "it", "pt", "nl"})
_LANGUAGE_SCRIPT = {
    **{code: "latin" for code in _LATIN_LANGUAGES},
    "ru": "cyrillic",
    "uk": "cyrillic",
    "ar": "arabic",
    "fa": "arabic",
    "el": "greek",
    "he": "hebrew",
    "hi": "devanagari",
    "th": "thai",
    "ko": "hangul",
    "ja": "cjk",
    "zh": "cjk",
}


def main_script(text: str) -> str | None:
    """The main script of ``text``, Chinese characters and kana counted as one
    (a Japanese sentence mixes them); None for a text with no letters."""
    counts = script_counts(text)
    counts["cjk"] = counts.pop("han", 0) + counts.pop("kana", 0)
    counts = +counts
    if not counts:
        return None
    return max(sorted(counts), key=lambda script: counts[script])


def writing_mismatch(text: str, expected: str) -> bool:
    """P12 for something written (a statement, a summary): its detected language
    differs from ``expected``; or, undetermined, it is written in another script
    than ``expected``'s. Within one script an undetermined text is not judged."""
    written = detect(text)
    if written != UNDETERMINED:
        return mismatch(written, expected)
    script, wanted = main_script(text), _LANGUAGE_SCRIPT.get(expected)
    return script is not None and wanted is not None and script != wanted


def language_name(code: str) -> str:
    """``fr`` -> ``French``; an unknown code names itself."""
    return LANGUAGE_NAMES.get(code, code)


def describe(codes: Iterable[str]) -> str:
    """``['fr', 'ar']`` -> ``French and Arabic`` for a prompt."""
    names = [language_name(c) for c in codes]
    if not names:
        return "the language of the source"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def is_code(value: str) -> bool:
    """A plausible ISO 639 code: two or three lower-case letters."""
    return bool(re.fullmatch(r"[a-z]{2,3}", value))


def fold(text: str) -> str:
    """Lower-cased, NFC-normalised text (for comparing statements across answers)."""
    return unicodedata.normalize("NFC", text).casefold()


__all__ = [
    "LANGUAGE_NAMES",
    "UNDETERMINED",
    "describe",
    "detect",
    "fold",
    "is_code",
    "language_name",
    "language_of_span",
    "languages_present",
    "main_script",
    "mismatch",
    "script_counts",
    "sentence_languages",
    "sentences",
    "writing_mismatch",
]
