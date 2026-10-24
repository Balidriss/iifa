"""Values in other forms and other languages (SPEC 5.4): a deterministic
comparison of two values of one key that says ``same``, ``different`` or
``unknown`` - never a guess. Dates are read into ISO (month names of several
languages and of the Arabic Syriac series), numbers in any digits, names are
transliterated **for comparison only** (never stored). A name is never proved
different: exonyms (Londres / London) look nothing alike. Months of another
calendar (Hijri) are never converted. Standard library only."""

from __future__ import annotations

import re
import unicodedata

from iifa.ingest.checks import script_of

SAME, DIFFERENT, UNKNOWN = "same", "different", "unknown"

# --------------------------------------------------------------------------- months

_MONTH_NAMES: dict[int, tuple[str, ...]] = {
    1: (
        "january",
        "jan",
        "janvier",
        "janv",
        "enero",
        "januar",
        "jänner",
        "gennaio",
        "janeiro",
        "январь",
        "января",
        "يناير",
        "كانون الثاني",
    ),
    2: (
        "february",
        "feb",
        "fevrier",
        "fev",
        "febrero",
        "februar",
        "febbraio",
        "fevereiro",
        "февраль",
        "февраля",
        "فبراير",
        "شباط",
    ),
    3: ("march", "mar", "mars", "marzo", "marz", "marco", "март", "марта", "مارس", "آذار"),
    4: ("april", "apr", "avril", "abril", "aprile", "апрель", "апреля", "أبريل", "ابريل", "نيسان"),
    5: ("may", "mai", "mayo", "maggio", "maio", "май", "мая", "مايو", "أيار", "ايار"),
    6: (
        "june",
        "jun",
        "juin",
        "junio",
        "juni",
        "giugno",
        "junho",
        "июнь",
        "июня",
        "يونيو",
        "حزيران",
    ),
    7: (
        "july",
        "jul",
        "juillet",
        "julio",
        "juli",
        "luglio",
        "julho",
        "июль",
        "июля",
        "يوليو",
        "تموز",
    ),
    8: ("august", "aug", "aout", "agosto", "август", "августа", "أغسطس", "اغسطس", "آب"),
    9: (
        "september",
        "sep",
        "sept",
        "septembre",
        "septiembre",
        "settembre",
        "setembro",
        "сентябрь",
        "сентября",
        "سبتمبر",
        "أيلول",
        "ايلول",
    ),
    10: (
        "october",
        "oct",
        "octobre",
        "octubre",
        "oktober",
        "ottobre",
        "outubro",
        "октябрь",
        "октября",
        "أكتوبر",
        "اكتوبر",
        "تشرين الأول",
    ),
    11: (
        "november",
        "nov",
        "novembre",
        "noviembre",
        "novembro",
        "ноябрь",
        "ноября",
        "نوفمبر",
        "تشرين الثاني",
    ),
    12: (
        "december",
        "dec",
        "decembre",
        "diciembre",
        "dezember",
        "dicembre",
        "dezembro",
        "декабрь",
        "декабря",
        "ديسمبر",
        "كانون الأول",
    ),
}
_HIJRI = ("محرم", "صفر", "ربيع", "جمادى", "رجب", "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة")


_KANA_VOICING = frozenset("゙゚")  # dakuten, handakuten: letters, not accents


def _fold(text: str) -> str:
    """Lower case, diacritics off (``fevrier``, ``marz``, ``aout``; Arabic and
    Hebrew vowel points too), then recomposed: Hangul syllables and voiced kana
    stay whole (``년``, ``ブ``)."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    kept = "".join(c for c in decomposed if not unicodedata.combining(c) or c in _KANA_VOICING)
    return unicodedata.normalize("NFC", kept)


def _ascii_digits(text: str) -> str:
    """Digits of any script (Arabic-Indic, Persian, full-width, Devanagari) as ASCII."""
    return "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)


_MONTHS: list[tuple[str, int]] = sorted(
    ((_fold(name), month) for month, names in _MONTH_NAMES.items() for name in names),
    key=lambda pair: -len(pair[0]),  # "kanun al-thani" before "kanun"
)
_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_ISO = re.compile(r"^\s*(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\s*$")
_MY = re.compile(r"^\s*(\d{1,2})[/.](\d{4})\s*$")
_DMY = re.compile(r"^\s*(\d{1,2})[/.](\d{1,2})[/.](\d{4})\s*$")
_CJK = re.compile(r"^\s*(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월](?:\s*(\d{1,2})\s*[日일])?\s*$")
_DAY = re.compile(r"(?<!\d)(\d{1,2})(?!\d)")


def parse_date(value: str) -> tuple[int, int | None, int | None] | None:
    """``(year, month, day)`` read from ``value``, or None when it is not a date
    this module can read for sure."""
    text = _ascii_digits(_fold(value))
    if not text or any(name in text for name in map(_fold, _HIJRI)):
        return None
    for pattern in (_ISO, _CJK):
        match = pattern.match(text)
        if match:
            year, month, day = match.groups()
            return _checked(int(year), int(month), int(day) if day else None)
    match = _MY.match(text)
    if match:
        return _checked(int(match.group(2)), int(match.group(1)), None)
    match = _DMY.match(text)
    if match:
        first, second, year = (int(g) for g in match.groups())
        if first <= 12 and second <= 12 and first != second:
            return None  # 03/04/2026: day/month order cannot be told
        day, month = (first, second) if second <= 12 else (second, first)
        return _checked(year, month, day)
    return _named_date(text)


def _named_date(text: str) -> tuple[int, int | None, int | None] | None:
    years = _YEAR.findall(text)
    if len(years) != 1:
        return None
    month = next(
        (
            m
            for name, m in _MONTHS
            if re.search(rf"(?<![^\W\d_]){re.escape(name)}(?![^\W\d_])", text)
        ),
        None,
    )
    if month is None:
        return None
    rest = _YEAR.sub(" ", text)
    days = _DAY.findall(rest)
    if len(days) > 1:
        return None
    leftover = re.sub(r"[\d\s,.\-/]+", " ", rest)
    for name, _ in _MONTHS:
        leftover = re.sub(rf"(?<![^\W\d_]){re.escape(name)}(?![^\W\d_])", " ", leftover)
    # what may surround a date: "года" (of the year), "of", "le", "de"...
    if re.sub(r"\b(года|год|г|of|the|le|de|del|du|el|am|den)\b", " ", leftover).strip():
        return None
    return _checked(int(years[0]), month, int(days[0]) if days else None)


def _checked(year: int, month: int, day: int | None) -> tuple[int, int | None, int | None] | None:
    if not 1 <= month <= 12 or (day is not None and not 1 <= day <= 31):
        return None
    return (year, month, day)


def precision(value: str) -> int:
    """How precise a value is, to prefer the fuller one in recall: 3 for a day,
    2 for a month, 0 for anything that is not a date."""
    date = parse_date(value)
    if date is None:
        return 0
    return 3 if date[2] is not None else 2


def _compare_dates(
    a: tuple[int, int | None, int | None], b: tuple[int, int | None, int | None]
) -> str:
    for left, right in zip(a, b, strict=True):
        if left is None or right is None:
            return SAME  # one refines the other
        if left != right:
            return DIFFERENT
    return SAME


# --------------------------------------------------------------------------- numbers

_NUMBER = re.compile(r"^\s*(?P<number>\d+(?:[.,]\d+)?)\s*(?P<rest>.*?)\s*$")


def _number(value: str) -> tuple[float, str] | None:
    text = _ascii_digits(unicodedata.normalize("NFKC", value))
    match = _NUMBER.match(text)
    if match is None or re.search(r"\d", match.group("rest")):
        return None
    return float(match.group("number").replace(",", ".")), _fold(match.group("rest"))


# --------------------------------------------------------------------------- names

_CYRILLIC = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюяіїєґ",
        [
            "a",
            "b",
            "v",
            "g",
            "d",
            "e",
            "e",
            "zh",
            "z",
            "i",
            "i",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "r",
            "s",
            "t",
            "u",
            "f",
            "kh",
            "ts",
            "ch",
            "sh",
            "shch",
            "",
            "y",
            "",
            "e",
            "yu",
            "ya",
            "i",
            "i",
            "ye",
            "g",
        ],
        strict=True,
    )
)
_GREEK = dict(
    zip(
        "αβγδεζηθικλμνξοπρσςτυφχψω",
        [
            "a",
            "v",
            "g",
            "d",
            "e",
            "z",
            "i",
            "th",
            "i",
            "k",
            "l",
            "m",
            "n",
            "x",
            "o",
            "p",
            "r",
            "s",
            "s",
            "t",
            "i",
            "f",
            "ch",
            "ps",
            "o",
        ],
        strict=True,
    )
)
_KANA_BASE = {
    "ア": "a",
    "イ": "i",
    "ウ": "u",
    "エ": "e",
    "オ": "o",
    "カ": "ka",
    "キ": "ki",
    "ク": "ku",
    "ケ": "ke",
    "コ": "ko",
    "ガ": "ga",
    "ギ": "gi",
    "グ": "gu",
    "ゲ": "ge",
    "ゴ": "go",
    "サ": "sa",
    "シ": "shi",
    "ス": "su",
    "セ": "se",
    "ソ": "so",
    "ザ": "za",
    "ジ": "ji",
    "ズ": "zu",
    "ゼ": "ze",
    "ゾ": "zo",
    "タ": "ta",
    "チ": "chi",
    "ツ": "tsu",
    "テ": "te",
    "ト": "to",
    "ダ": "da",
    "デ": "de",
    "ド": "do",
    "ナ": "na",
    "ニ": "ni",
    "ヌ": "nu",
    "ネ": "ne",
    "ノ": "no",
    "ハ": "ha",
    "ヒ": "hi",
    "フ": "fu",
    "ヘ": "he",
    "ホ": "ho",
    "バ": "ba",
    "ビ": "bi",
    "ブ": "bu",
    "ベ": "be",
    "ボ": "bo",
    "パ": "pa",
    "ピ": "pi",
    "プ": "pu",
    "ペ": "pe",
    "ポ": "po",
    "マ": "ma",
    "ミ": "mi",
    "ム": "mu",
    "メ": "me",
    "モ": "mo",
    "ヤ": "ya",
    "ユ": "yu",
    "ヨ": "yo",
    "ラ": "ra",
    "リ": "ri",
    "ル": "ru",
    "レ": "re",
    "ロ": "ro",
    "ワ": "wa",
    "ヲ": "o",
    "ン": "n",
    "ー": "",
}
# hiragana are katakana shifted by 0x60
_KANA = {**_KANA_BASE, **{chr(ord(k) - 0x60): v for k, v in _KANA_BASE.items() if k != "ー"}}
# abjads leave vowels out: compare consonant skeletons
_ARABIC = {
    "ا": "",
    "أ": "",
    "إ": "",
    "آ": "",
    "ء": "",
    "ى": "",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "",
    "غ": "gh",
    "ف": "f",
    "ق": "k",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ة": "",
    "و": "w",
    "ي": "y",
    "ی": "y",
    "پ": "p",
    "چ": "ch",
    "گ": "g",
    "ژ": "zh",
}
_HEBREW = dict(
    zip(
        "אבגדהוזחטיכךלמםנןסעפףצץקרשת",
        [
            "",
            "b",
            "g",
            "d",
            "h",
            "w",
            "z",
            "h",
            "t",
            "y",
            "k",
            "k",
            "l",
            "m",
            "m",
            "n",
            "n",
            "s",
            "",
            "p",
            "p",
            "ts",
            "ts",
            "k",
            "r",
            "sh",
            "t",
        ],
        strict=True,
    )
)
_ABJADS = (_ARABIC, _HEBREW)
_VOWELS = re.compile(r"[aeiouyw]")
MIN_SKELETON = 3  # a shorter consonant skeleton ("ln") matches too much
LONG_NAME = 6  # one letter apart on a name this long is still the same name


def transliterate(value: str) -> str:
    """A plain Latin form of ``value`` for comparison only (never stored):
    Cyrillic, Greek and kana transliterated, Latin folded (case, diacritics),
    Arabic and Hebrew letters as consonants."""
    out: list[str] = []
    for char in _fold(value):
        for table in (_CYRILLIC, _GREEK, _KANA, _ARABIC, _HEBREW):
            if char in table:
                out.append(table[char])
                break
        else:
            out.append(char if char.isascii() else "")
    return re.sub(r"[^a-z]", "", "".join(out))


def _comparable(form: str) -> str:
    """Spelling variants that transliteration cannot settle: y/i, c/k, doubles."""
    form = form.replace("y", "i").replace("c", "k")
    return re.sub(r"(.)\1+", r"\1", form)


def _is_abjad(value: str) -> bool:
    return any(c in table for c in value for table in _ABJADS)


def _has_unknown_letters(value: str) -> bool:
    """Letters no table covers (Chinese characters): nothing to compare."""
    tables = (_CYRILLIC, _GREEK, _KANA, _ARABIC, _HEBREW)
    return any(
        c.isalpha() and not c.isascii() and not any(c in t for t in tables)
        for c in _fold(value)
        if not unicodedata.combining(c)
    )


def _compare_names(a: str, b: str, *, one_letter: bool = True) -> str:
    if _has_unknown_letters(a) or _has_unknown_letters(b):
        return UNKNOWN
    left, right = transliterate(a), transliterate(b)
    if not left or not right:
        return UNKNOWN
    if _is_abjad(a) or _is_abjad(b):
        left, right = _VOWELS.sub("", left), _VOWELS.sub("", right)
        return SAME if left == right and len(left) >= MIN_SKELETON else UNKNOWN
    left, right = _comparable(left), _comparable(right)
    if left == right:
        return SAME
    if one_letter and min(len(left), len(right)) >= LONG_NAME and _one_apart(left, right):
        return SAME
    return UNKNOWN  # never different: exonyms look nothing alike


def same_name(a: str, b: str) -> bool:
    """Are ``a`` and ``b`` one name, in any script, by the comparison of SPEC
    5.4 (equal when folded, or transliterated alike)? Never a guess: an
    adapted form ("Алиса" for Alice) is not the same name here."""
    if not a.strip() or not b.strip():
        return False
    return _fold(a) == _fold(b) or _compare_names(a, b) == SAME


def same_spelling(a: str, b: str) -> bool:
    """One spelling in two scripts (SPEC 5.1, the source form of a subject):
    ``same_name`` without its one-letter tolerance, which would take Daniela
    for Daniel."""
    if not a.strip() or not b.strip():
        return False
    return _fold(a) == _fold(b) or _compare_names(a, b, one_letter=False) == SAME


NEAR_STEM = 3  # letters two near words must share before an ending
NEAR_ENDING = 2  # the longest ending a near word may add or swap ("Иван" -> "Иваном")


def near_form(a: str, b: str) -> bool:
    """SPEC 5.1: is ``a`` a near form of ``b`` - not one spelling but, word by
    word, one letter or an ending apart ("Ивана" of "Иван", "Карим" of
    "Карима", "Karim" of "Карима")? Two words of one script are compared as
    written; across scripts, the same name (``same_spelling``: Ivan / Иван) is
    no near form, else they are compared transliterated and only one letter
    counts - an exonym's ending is no inflection (Paris / Париж). A far nickname (Kiki
    for Karim), a longer word (Karimova), another word count (Karim Benali)
    are not near forms; nor are words with letters no table covers (Chinese
    characters, Hangul): that is not judged."""
    left, right = a.split(), b.split()
    if not left or len(left) != len(right):
        return False
    near = False
    for x, y in zip(left, right, strict=True):
        if _fold(x) == _fold(y):
            continue
        if _has_unknown_letters(x) or _has_unknown_letters(y):
            return False
        if script_of(x) == script_of(y):
            if not _letter_or_ending(_fold(x), _fold(y)):
                return False
        elif same_spelling(x, y):
            continue
        elif not _one_letter(_comparable(transliterate(x)), _comparable(transliterate(y))):
            return False
        near = True
    return near


def _one_letter(a: str, b: str) -> bool:
    return a != b and min(len(a), len(b)) >= NEAR_STEM and _one_apart(a, b)


def _letter_or_ending(a: str, b: str) -> bool:
    if a == b or min(len(a), len(b)) < NEAR_STEM:
        return False
    if _one_apart(a, b):
        return True
    short = min(len(a), len(b))
    stem = next((i for i in range(short) if a[i] != b[i]), short)
    return stem >= NEAR_STEM and max(len(a), len(b)) - stem <= NEAR_ENDING


def _one_apart(a: str, b: str) -> bool:
    """Edit distance exactly one (a letter replaced, added or dropped)."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


# --------------------------------------------------------------------------- the comparison


def compare_values(a: str, b: str) -> str:  # noqa: PLR0911 - one return per rule of SPEC 5.4
    """``same``, ``different`` or ``unknown`` for two values of one key."""
    if not a.strip() or not b.strip():
        return UNKNOWN
    if _fold(a) == _fold(b):
        return SAME
    left_date, right_date = parse_date(a), parse_date(b)
    if left_date is not None and right_date is not None:
        return _compare_dates(left_date, right_date)
    if left_date is not None or right_date is not None:
        return UNKNOWN
    left_number, right_number = _number(a), _number(b)
    if left_number is not None and right_number is not None:
        if left_number[1] != right_number[1]:
            return UNKNOWN  # another unit or wording
        return SAME if left_number[0] == right_number[0] else DIFFERENT
    if left_number is not None or right_number is not None:
        return UNKNOWN
    return _compare_names(a, b)


__all__ = [
    "DIFFERENT",
    "SAME",
    "UNKNOWN",
    "compare_values",
    "parse_date",
    "precision",
    "transliterate",
]
