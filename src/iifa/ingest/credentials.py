"""A secret said in a conversation is never memory (P5).

A password, a passcode or PIN, a one-time or verification code, a recovery
code, an API key or a token typed in a turn must not outlive the turn as a
fact, a fact proposal, a session fact or a turn trace: a recalled secret is
one the agent uses again, and no host can filter a secret it was never told.
The extractor is the only place that reads a turn before it becomes memory,
so this is where the values are found and kept out.

The values of one source are the union of three readings:

* the service's own (``detect``): the shapes of well-known keys and tokens,
  always; and only with ``detect_secrets_by_wording`` (off by default, at the
  operator's risk: a heuristic that knows a few languages and may take an
  ordinary word for a secret, dropping the facts that hold it) a credential
  word followed in the same clause by a copula, a colon or an arrow and a
  value that looks like a secret, or a code word followed by digits;
* what the model lists in ``secrets``, kept only when the source
  writes it whole;
* the value of a fact whose attribute names a credential (``password``,
  ``api_key``...), when it looks like a secret and the source writes it whole.

A model reading a hostile page may list a harmless word as a secret: that
only removes from memory what the same model could already leave out of its
answer, so it gives a page nothing it did not have.

What carries a value is dropped (a fact, a node, an edge, an activity), a
quote loses it (the longest part of the quote without it is kept, so the
excerpt stored beside a fact never shows it) and a turn's trace masks it. The
source keeps the turn as said, when sources are retained: that is
what ``forget`` is for.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable, Sequence

from iifa.ingest.schema import Answer, ProposedFact

MASK = "[secret]"
MIN_CHARS = 4  # shorter: too common to be told apart from ordinary words
MIN_UNSPACED_CHARS = 2  # in a script without spaces, a character is nearly a word
INLINE_CHARS = 200  # the longest value read after a credential word
MAX_CHARS = 8192  # the longest secret kept (a private key block)
PASSPHRASE_WORDS = 4  # a credential value this long without an event verb is a passphrase
MIN_QUOTE_CHARS = 8  # a quote left shorter than this once a value is cut out says nothing
WINDOW = 80  # characters after a credential word where its value is looked for

_CREDENTIAL = re.compile(
    r"(?<![^\W_])(?:"
    r"pass(?:word|wd|code|phrase)s?|pw|(?:pwd|pass|secret)(?=\s*[:=])|pin(?:\s+code)?|otp|2fa(?:\s+code)?"
    r"|one[\s-]time\s+(?:pass)?(?:code|password)s?"
    r"|(?:verification|security|confirmation|login|sign[\s-]?in|access|recovery|backup|reset"
    r"|door|gate|alarm)[\s_-]+codes?"
    r"|(?:api|access|secret|private|licen[cs]e)[\s_-]?keys?"
    r"|(?:(?:access|auth|bearer|refresh|api|session)[\s_-]?)?tokens?"
    r"|client[\s_-]?secrets?"
    r"|mots?\s+de\s+passe|mdp|codes?\s+(?:pin|secret|de\s+(?:v[ée]rification|confirmation"
    r"|s[ée]curit[ée]|r[ée]cup[ée]ration|connexion)|[àa]\s+usage\s+unique)"
    r"|cl[ée]s?\s+(?:d['\u2019]\s*)?api|cl[ée]s?\s+secr[èe]tes?|jetons?"
    r"|contrase[ñn]as?|passwort|kennwort|wachtwoord|senhas?|парол[ьяию]"
    # an identifier may go on after the word before its = or : (DB_PASSWORD_PROD=)
    r")(?:(?:_[A-Za-z0-9]+)+(?=\s*[:=])|(?!\w))",
    re.IGNORECASE,
)
# a bare code word ("the door code is 4071"): its value is digits only, and
# never the code of a place, a language or a product (``_NOT_SECRET_CODE``)
_CODE = re.compile(r"(?<![^\W_])(?:code|codes|код)(?![^\W_])", re.IGNORECASE)
_NOT_SECRET_CODE = re.compile(
    r"(?:zip|post|postal|area|country|project|dial(?:l?ing)?|phone|source|dress|iso|language|error"
    r"|status|exit|return|product|bar|qr|tax|colou?r|airport|station|course|class|promo"
    r"|discount|coupon|voucher|cheat|morse|genetic|machine|program|python|sample|example)$",
    re.IGNORECASE,
)
_NOT_SECRET_CODE_AFTER = re.compile(
    r"\s*(?:postal|pays|iso|couleur|source|promo)(?!\w)", re.IGNORECASE
)
# P5: a credential word that only qualifies another noun ("password
# manager", "token limit", "gestionnaire de mots de passe") introduces no value
_COMPOUND_AFTER = re.compile(
    r"[\s_-]*(?:managers?|vaults?|generators?|polic(?:y|ies)|rules?|requirements?|strength"
    r"|length|hints?|resets?|rotation|fields?|prompts?|screens?|pages?|forms?|boxes?|limits?"
    r"|counts?|budgets?|windows?|usage|costs?|prices?|rates?|economy|summits?|sales?|holders?"
    r"|counters?|totals?|sizes?|stores?|services?|safes?|book|business)(?!\w)",
    re.IGNORECASE,
)
# what an identifier's segment measures about a credential (TOKEN_TTL, PASSWORD_MAX_LENGTH)
_MEASURE = re.compile(
    r"ttl|expiry|expires?|expiration|max|min|length|len|limit|size|count|timeout|lifetime|age"
    r"|duration|seconds|secs|minutes|days|bits|bytes",
    re.IGNORECASE,
)
_COMPOUND_BEFORE = re.compile(
    r"(?:(?:gestionnaire|g[ée]n[ée]rateur|coffre|politique|limite|nombre|taille|r[èe]gles?"
    r"|longueur)s?\s+(?:de\s+|d['\u2019]\s*)"
    r"|(?<!\w)(?:max|maximum|min|minimum|context|input|output|usage|total|prompt|completion"
    r"|reasoning|thinking)\s+)$",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"[\s:#-]{0,3}(\d{3,12}(?:[ -]\d{3,6})*)(?!\w)(?![-/.]\d)")
_CODE_LIKE = re.compile(r"code|pin|otp|2fa", re.IGNORECASE)
_YEAR = re.compile(r"(?:19|20)\d\d")
# a path names where something is kept, not what it is ("the password file
# is /etc/shadow")
_PATH = re.compile(r"(?:/|~/|\./|\.\./|[A-Za-z]:[\\/])")
_PREPOSITIONS = frozenset({"to", "en", "a", "auf", "на"})
_CHANGE_VERB = re.compile(
    r"(?<!\w)(?:chang(?:e|ed|ing)|set(?:ting)?|reset|updated?|switched|moved|renamed"
    r"|chang[ée]e?s?|modifi[ée]e?s?|r[ée]initialis[ée]e?s?|mis|ge[äa]ndert|gesetzt"
    r"|cambi[ée]|cambiado|puesto|изменил[аи]?|поменял[аи]?)(?!\w)",
    re.IGNORECASE,
)
# digits a count or an amount: "the code is 1500 lines" holds no secret
_UNIT = re.compile(
    r"\s*(?:lines?|users?|people|persons?|words?|pages?|files?|characters?|chars?|bytes?"
    r"|[kmg]b|euros?|dollars?|usd|eur|€|\$|%|percent|times|items?|rows?|tests?|lignes?"
    r"|mots?|fichiers?|personnes?|utilisateurs?|caract[èe]res?|octets?|fois)(?!\w)",
    re.IGNORECASE,
)
_COUNTED = re.compile(r"tokens?|jetons?")
_CODE_DIGITS = re.compile(r"\d{4,12}|\d{3,4}(?:[ -]\d{3,4})+")
_COPULA = re.compile(
    r"(?<!\w)(?:is|was|are|will\s+be|to|est|était|sera|en|ist|war|wird|auf|es|era|será|a"
    r"|è|é|это|будет|на)(?!\w)\s*"
    r"|\s*(?:[:=](?!//)|->|\u2192)\s*",  # a URL's scheme is no colon
    re.IGNORECASE,
)
_FILLER = re.compile(
    r"(?:(?:now|still|just|simply|maintenant|d[ée]sormais|dor[ée]navant|toujours|simplement"
    r"|jetzt|nun|ahora|agora|adesso|ora|теперь)\s+)*",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)|\n")
# a value is looked for in the credential word's own clause: "I changed my
# password and my flight is AF1234" gives none (a comma does not end it:
# "my password changed, the new one is ...")
_CLAUSE_END = re.compile(
    r";|\s(?:and|but|or|while|whereas|et|mais|ou|alors|und|aber|oder|y|pero|e|ma|и|\u0430|но)\s",
    re.IGNORECASE,
)
_QUOTES = {'"': '"', "'": "'", "`": "`", "\u201c": "\u201d", "\u2018": "\u2019", "\u00ab": "\u00bb"}
_TRAILING = ".,;:!?)]}\u00bb\"'`\u2019\u201d"
_WRAPPERS = "*_>~"
_LEADING = "*_([{<~"
# what separates the values of a list ("codes: A B, C and D")
_LIST_GAP = re.compile(r"\s*,\s*|\s+(?:and|et|und|y|e|и)\s+|\s+", re.IGNORECASE)
MAX_LISTED = 50
_PLURAL = re.compile(r"(?:s|es|и|ы)$", re.IGNORECASE)
_SHAPES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<!\w)gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<!\w)github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?<!\w)glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<!\w)xox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?<!\w)AKIA[0-9A-Z]{16}(?!\w)"),
    re.compile(r"(?<!\w)eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
)
_BEARER = re.compile(r"(?<!\w)Bearer\s+([A-Za-z0-9._~+/-]{16,}=*)")
_URL_PASSWORD = re.compile(r"://[^\s:/@]+:([^\s/@]+)@")

# an attribute whose last word names a credential holds one as value
_CREDENTIAL_KEYS = (
    "password",
    "passwd",
    "passcode",
    "passphrase",
    "pin",
    "pin_code",
    "pin_number",
    "card_pin",
    "bank_pin",
    "sim_pin",
    "atm_pin",
    "phone_pin",
    "access_code",
    "door_code",
    "gate_code",
    "alarm_code",
    "login_code",
    "confirmation_code",
    "2fa_code",
    "license_key",
    "licence_key",
    "session_token",
    "otp",
    "one_time_code",
    "verification_code",
    "security_code",
    "recovery_code",
    "recovery_codes",
    "backup_code",
    "backup_codes",
    "api_key",
    "apikey",
    "access_key",
    "secret_key",
    "private_key",
    "token",
    "access_token",
    "auth_token",
    "refresh_token",
    "api_token",
    "bearer_token",
    "client_secret",
    "mot_de_passe",
    "mdp",
    "contraseña",
    "passwort",
    "kennwort",
    "пароль",
)


_WHOLE_ONLY = frozenset({"pin", "token"})


def credential_key(attribute: str | None) -> bool:
    """Does ``attribute`` name a credential (``password``, ``shop_password``,
    ``api.key``)? The last word decides, so ``password_changed`` (an event) is
    not one."""
    if not attribute:
        return False
    key = re.sub(r"[\s.\-]+", "_", attribute.strip().lower()).strip("_")
    # "pin" and "token" end ordinary keys (map_pin, game_token): whole key only
    return any(
        key == word or (key.endswith("_" + word) and word not in _WHOLE_ONLY)
        for word in _CREDENTIAL_KEYS
    )


def _plausible(value: str, quoted: bool) -> bool:
    """Does ``value`` look like a secret rather than a word: a digit, a symbol
    (a hyphen or an apostrophe is a word's), a capital inside it, or quotes."""
    if not MIN_CHARS <= len(value) <= INLINE_CHARS:
        return False
    if quoted:
        return True
    return (
        any(c.isdigit() for c in value)
        or any(not c.isalnum() and not c.isspace() and c not in "-'\u2019" for c in value)
        # a capital inside a word that has lowercase letters (camel case); an
        # acronym (IKEA, NASA) is a name
        or (any(c.isupper() for c in value[1:]) and any(c.islower() for c in value))
    )


def _values_at(text: str, position: int) -> list[str]:
    """The value written at ``position``: inside quotes when it opens with one,
    else the next run of non-space characters, kept both whole and without its
    trailing punctuation (``Hunter22!`` may be the password, or ``Hunter22``
    followed by an exclamation); empty when it does not look like a secret."""
    if position >= len(text):
        return []
    opener = text[position]
    if opener in _QUOTES:
        close = _closer(text, position)
        if close > position + 1:
            inner = text[position + 1 : close].strip()
            if "\n" not in inner and _plausible(inner, quoted=True):
                return [inner]
        # a closer inside the value ('Don't4get!'): read it as a bare run
        position += 1
    match = re.compile(r"\S+").match(text, position)
    if match is None:
        return []
    whole = match.group()
    # Markdown or brackets around a value (**Hunter22**, (Hunter22)) are not
    # part of it; a comma, a period or a colon after it ends the clause
    value = whole.lstrip(_LEADING).rstrip(_TRAILING + _WRAPPERS)
    if not _plausible(value, quoted=False) or _PATH.match(value):
        return []
    forms = [value]
    field = re.split(r";|&(?=\w+=)", value, maxsplit=1)[0]
    if field != value and _plausible(field, quoted=False):
        forms.append(field)  # password=Hunter22;user=bob: the field may be the password
    kept = whole.lstrip(_LEADING).rstrip(".,;:" + _WRAPPERS + ")]}")
    if kept != value and len(kept) <= INLINE_CHARS:
        forms.append(kept)  # Hunter22! may be the password
    raw = whole.rstrip(".,;:")
    if raw.startswith(("_", "*", "~")) and raw not in forms and len(raw) <= INLINE_CHARS:
        forms.append(raw)  # _Secret1 may be the password, not Markdown
    return forms


def _closer(text: str, position: int) -> int:
    """Where the quote opened at ``position`` closes, or -1. A straight
    apostrophe closes only before a space, punctuation or the end, so the
    one of "Hunter's2day" stays inside the value."""
    opener = text[position]
    closing = _QUOTES[opener]
    limit = min(len(text), position + 2 + INLINE_CHARS)
    index = text.find(closing, position + 1, limit)
    while opener == "'" and index != -1:
        after = text[index + 1 : index + 2]
        if not after or not after.isalnum():
            break
        index = text.find(closing, index + 1, limit)
    return index


def _run_end(text: str, position: int) -> int:
    """Where the value read at ``position`` ends (a quoted one at its closer)."""
    if position < len(text) and text[position] in _QUOTES:
        close = _closer(text, position)
        if close > position:
            return close + 1
    match = re.compile(r"\S+").match(text, position)
    return match.end() if match is not None else position


def _copulas(text: str, start: int) -> list[tuple[int, str]]:
    """Where a value may start after a credential word ending at ``start``:
    after each copula, colon or arrow within the same sentence and ``WINDOW``
    characters, past the fillers ("now"), with the copula itself."""
    end = start + WINDOW
    for boundary in (_SENTENCE_END, _CLAUSE_END):
        stop = boundary.search(text, start, end)
        if stop is not None:
            end = min(end, stop.start())
    positions = []
    for copula in _COPULA.finditer(text, start, end):
        filler = _FILLER.match(text, copula.end())
        position = filler.end() if filler is not None else copula.end()
        positions.append((position, copula.group().strip().casefold()))
    return positions


def _compound(text: str, start: int, end: int) -> bool:
    """Does the credential word at ``start:end`` only qualify another noun
    ("password manager", "max tokens", ``TOKEN_LIMIT=``)?"""
    segments = text[start:end].split("_")[1:]
    return bool(
        _COMPOUND_AFTER.match(text, end)
        or _COMPOUND_BEFORE.search(text[max(0, start - 30) : start])
        or any(_COMPOUND_AFTER.fullmatch(s) or _MEASURE.fullmatch(s) for s in segments)
    )


def _after_credential(text: str, word: str, start: int) -> list[str]:
    """The value the credential ``word`` ending at ``start`` introduces:
    digits written right after a code word ("verification code 482913"),
    else, after each copula in turn, the first value that looks like a secret."""
    digits = _DIGITS.match(text, start) if _CODE_LIKE.search(word) else None
    if (
        digits is not None
        and _CODE_DIGITS.fullmatch(digits.group(1))
        and not _UNIT.match(text, digits.end())
    ):
        return [digits.group(1), *_listed(text, word, digits.end())]
    copulas = _copulas(text, start)
    # "the password to iCloud is X": a copula's value first, then a
    # preposition's, and that only after a change verb ("changed it to X")
    before = text[max(0, start - len(word) - 40) : start - len(word)]
    before = re.split(r"[.!?\n]", before)[-1]  # the word's own sentence
    changed = bool(_CHANGE_VERB.search(before))
    ordered = [c for c in copulas if c[1] not in _PREPOSITIONS]
    if changed:
        ordered += [c for c in copulas if c[1] in _PREPOSITIONS]
    for position, copula in ordered:
        values = [
            v
            for v in _values_at(text, position)
            if _fits(word, v)
            and _after_copula(word, copula, v)
            and not (v.isdigit() and _UNIT.match(text, _run_end(text, position)))
        ]
        if values:
            return values + _listed(text, word, _run_end(text, position))
    return []


def _after_copula(word: str, copula: str, value: str) -> bool:
    """After a preposition ("to", "en", "a"), a value is a secret only when it
    is not where the credential goes (an address, a link) nor a year or an
    amount, unless the word is a code: "sent the password to alice@x.org",
    "changé en 2019" give none."""
    if "://" in value or value.casefold().startswith("www."):
        return False  # a link names where something is, after any copula
    if copula not in _PREPOSITIONS:
        return True
    if "@" in value:
        return False
    return not value.rstrip(_TRAILING).isdigit() or bool(_CODE_LIKE.search(word))


def _listed(text: str, word: str, position: int) -> list[str]:
    """The values that follow a first one in the same sentence after a
    plural word ("recovery codes: A B, C and D"), as long as each looks like
    a secret."""
    if not _PLURAL.search(word):
        return []  # one password, one value
    stop = _SENTENCE_END.search(text, position)
    end = stop.start() if stop is not None else len(text)
    found: list[str] = []
    while len(found) < MAX_LISTED:
        gap = _LIST_GAP.match(text, position, end)
        if gap is None or gap.end() >= end:
            break
        values = [v for v in _values_at(text, gap.end()) if _fits(word, v)]
        if not values:
            break
        found.extend(values)
        position = _run_end(text, gap.end())
    return found


def _fits(word: str, value: str) -> bool:
    """A "token" counts things ("the max tokens is 4096", "API tokens are
    20000 per month"): its value is never digits alone; a bare "pin" marks
    places too ("meet at the pin: 10am"): its value is digits only."""
    bare = word.casefold()
    if _COUNTED.fullmatch(re.split(r"[\s_-]+", bare)[-1]):
        return not value.rstrip(_TRAILING).isdigit()
    if bare == "pin":
        return value.rstrip(_TRAILING).isdigit()
    return True


def _after_code(text: str, start: int, end: int) -> list[str]:
    """A bare code word's value: digits only, and never a zip, area or
    language code (by the word in front, or after: "code postal")."""
    before = text[max(0, start - 20) : start].split()
    if before and _NOT_SECRET_CODE.fullmatch(before[-1].strip("-_")):
        return []
    if _NOT_SECRET_CODE_AFTER.match(text, end):
        return []
    word = text[start:end]
    runs = [_DIGITS.match(text, end)]
    # the copula right after the word only: "code review is 2024" is no code
    near = re.compile(r"\s*").match(text, end)
    copula = _COPULA.match(text, near.end() if near is not None else end)
    if copula is not None and copula.group().strip().casefold() not in _PREPOSITIONS:
        runs.append(_DIGITS.match(text, copula.end()))
    for run in runs:
        if run is None or _UNIT.match(text, run.end()):
            continue
        value = run.group(1)
        # a year after a bare "code" is a title or a date ("Advent of Code 2024")
        if _CODE_DIGITS.fullmatch(value) and not _YEAR.fullmatch(value):
            return [value, *_listed(text, word, run.end())]
    return []


_SHORT_WORDS = frozenset({"pwd", "pass", "secret"})
_OWNERS = re.compile(
    r"(?:my|your|our|his|her|their|its|mon|ma|ton|ta|votre|notre|son|sa|mein|meine"
    r"|mi|tu|su)\s+$",
    re.IGNORECASE,
)


def _short_named(text: str, start: int) -> bool:
    """A bare "pass", "pwd" or "secret" before its colon names a credential at
    the start of a line, inside an identifier (``DB_PASS=``) or after a
    possessive ("my pass: ..."); "Boarding pass: AF1234" is a ticket."""
    before = text[max(0, start - 12) : start]
    if not before.strip() or before.endswith(("_", "\n")) or before.rstrip(" \t").endswith("\n"):
        return True
    return bool(_OWNERS.search(before))


def detect(text: str, *, wording: bool = False) -> list[str]:
    """The secrets ``text`` writes, by the service's own reading: the shapes
    of keys and tokens always; with ``wording`` (``detect_secrets_by_wording``,
    off by default) also what its words introduce as one, a heuristic that
    knows a few languages and may take an ordinary word for a secret."""
    found: list[str] = []
    if wording:
        found.extend(_by_wording(text))
    for shape in _SHAPES:
        found.extend(m.group() for m in shape.finditer(text))
    found.extend(m.group(1) for m in _BEARER.finditer(text))
    found.extend(m.group(1) for m in _URL_PASSWORD.finditer(text))
    return found


def _by_wording(text: str) -> list[str]:
    """What a credential or a code word introduces as a value."""
    found: list[str] = []
    for match in _CREDENTIAL.finditer(text):
        if match.group().casefold() in _SHORT_WORDS and not _short_named(text, match.start()):
            continue
        if not _compound(text, match.start(), match.end()):
            found.extend(_after_credential(text, match.group(), match.end()))
    for match in _CODE.finditer(text):
        if not _compound(text, match.start(), match.end()):
            found.extend(_after_code(text, match.start(), match.end()))
    return found


# a secret is matched whole: never inside a longer word, an underscore being
# no letter (_Hunter22_ in Markdown is Hunter22). A script written without
# spaces (Chinese, Japanese, Korean particles, Thai...) has no word to be
# inside: its characters never stop a match ("密码是向日葵。", "密码是Yk2#pm9Wt")
_UNSPACED = (
    "[\u0e00-\u0eff\u1000-\u109f\u1100-\u11ff\u1780-\u17ff\u3040-\u30ff"
    "\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f"
    "\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f]"
)
_LETTER = "(?!" + _UNSPACED + r")[^\W_]"  # a letter or a digit of a spaced script
_BEFORE, _AFTER = "(?<!" + _LETTER + ")", "(?!" + _LETTER + ")"


@functools.lru_cache(maxsize=256)
def _any(secrets: tuple[str, ...]) -> re.Pattern[str]:
    """One pattern for a set of secrets, longest first, that also matches the
    masks already there (so a mask is never masked again)."""
    ordered = sorted(set(secrets), key=len, reverse=True)
    words = "|".join(_whole(s) for s in ordered)
    return re.compile(re.escape(MASK) + "|" + words, re.IGNORECASE)


def _whole(secret: str) -> str:
    """``secret`` as a pattern, never inside a longer word: a boundary on a
    side only where its own edge is a letter or a digit (``#abc1`` is masked
    in ``x#abc1``), and none against a script written without spaces."""
    before = _BEFORE if secret[:1].isalnum() else ""
    after = _AFTER if secret[-1:].isalnum() else ""
    return before + re.escape(secret) + after


def _shortest(value: str) -> int:
    """The fewest characters a secret may have: fewer in a script written
    without spaces (向日葵 is three)."""
    return MIN_UNSPACED_CHARS if re.search(_UNSPACED, value) else MIN_CHARS


def _pattern(secret: str) -> re.Pattern[str]:
    return _any((secret,))


def written_in(text: str, secret: str) -> bool:
    """Is ``secret`` written whole in ``text`` (never inside a longer word)?"""
    return bool(secret) and any(m.group() != MASK for m in _pattern(secret).finditer(text))


def collect(
    text: str,
    answers: Iterable[Answer],
    *,
    read: str | None = None,
    wording: bool = False,
) -> tuple[str, ...]:
    """Every secret of a source: the service's reading of ``text`` (or of
    ``read``, the part of it the service may read), the values the model
    listed and the values of credential facts, each kept only when ``text``
    writes it whole; longest first, so that a mask never leaves the end of a
    longer one."""
    candidates = detect(text if read is None else read, wording=wording)
    for answer in answers:
        candidates.extend(answer.secrets)
        # a credential fact's value, unless it is the event ("changed"): the
        # fact itself is dropped anyway
        candidates.extend(
            word
            for f in answer.facts
            if f.value is not None and credential_key(f.attribute)
            for word in _credential_values(f.value)
        )
    kept: dict[str, str] = {}
    for candidate in candidates:
        value = candidate.strip()
        if _shortest(value) <= len(value) <= MAX_CHARS and written_in(text, value):
            kept.setdefault(value.casefold(), value)
    return tuple(sorted(kept.values(), key=lambda v: (-len(v), v)))


# what a model writes as the value of a credential attribute when it means the
# event, not the value: never taken for a secret
_EVENT_WORDS = frozenset(
    [
        "changed",
        "reset",
        "updated",
        "renewed",
        "new",
        "old",
        "same",
        "expired",
        "revoked",
        "set",
        "unset",
        "unknown",
        "forgotten",
        "lost",
        "leaked",
        "compromised",
        "required",
        "needed",
        "wrong",
        "right",
        "correct",
        "incorrect",
        "valid",
        "invalid",
        "strong",
        "weak",
        "refused",
        "rejected",
        "accepted",
        "saved",
        "stored",
        "shared",
        "sent",
        "received",
        "none",
        "null",
        "true",
        "false",
        "changé",
        "changée",
        "modifié",
        "modifiée",
        "réinitialisé",
        "réinitialisée",
        "nouveau",
        "nouvelle",
        "ancien",
        "ancienne",
        "oublié",
        "oubliée",
        "perdu",
        "perdue",
        "expiré",
        "expirée",
        "refusé",
        "refusée",
        "valide",
        "invalide",
        "inconnu",
        "inconnue",
        "geändert",
        "neu",
        "cambiado",
        "nuevo",
        "изменён",
        "изменен",
        "новый",
    ]
)


# the words that make a value of several words the event ("changed today"),
# not a passphrase ("correct horse battery staple")
_EVENT_VERBS = frozenset(
    """changed reset updated renewed expired revoked set unset forgotten lost leaked
    compromised refused rejected accepted saved stored shared sent received changé changée
    modifié modifiée réinitialisé réinitialisée oublié oubliée perdu perdue expiré expirée
    refusé refusée geändert cambiado изменён изменен""".split()  # noqa: SIM905 - a word list
)


_DESCRIPTION_START = frozenset(
    """too very not quite so still always never same new old unknown none the a an my his
    her their its our your expires expired trop très pas même nouveau nouvelle ancien
    ancienne le la les un une mon ma mes son sa ses zu sehr nicht der die das ein eine
    demasiado muy no el los una mismo nuevo""".split()  # noqa: SIM905 - a word list
)


def _eventful(value: str | None) -> bool:
    """Is a credential attribute's value the event rather than a value: none,
    one event word ("changed", "reset."), or several words with an event verb
    and nothing that looks like a secret ("changed today")? Such a fact is
    kept without its key. Several plain words without one are a passphrase."""
    if value is None:
        return True
    words = [w.casefold().rstrip(_TRAILING) for w in value.strip().split()]
    if not words:
        return True
    if _held_words(value):
        return False
    if len(words) == 1:
        return words[0] in _EVENT_WORDS
    # several words: a description when it opens like one ("too weak", "same
    # as email", "new one") or names a change ("changed today"); four words
    # or more without an event verb are a passphrase ("the old oak tree")
    if any(w in _EVENT_VERBS for w in words):
        return True
    return len(words) < PASSPHRASE_WORDS and words[0] in _DESCRIPTION_START


def _held_words(value: str) -> list[str]:
    """The words of a credential attribute's value that look like secrets
    ("changed to Xy7#kq9" holds one)."""
    words = [w.rstrip(_TRAILING) for w in value.split()]
    return [w for w in words if _plausible(w, quoted=False)]


def _credential_values(value: str) -> list[str]:
    """What a credential attribute's value gives to mask: its words that look
    like secrets, and a passphrase whole. A single plain word gives nothing
    (it may be an ordinary word of other facts)."""
    held = _held_words(value)
    words = value.strip().rstrip(_TRAILING).split()
    if len(words) > 1 and not held and not _eventful(value):
        return [" ".join(words), *held]
    return held


def distinct(secrets: Sequence[str]) -> int:
    """How many secrets ``secrets`` holds, a value and its form with trailing
    punctuation counted once (both are masked)."""
    return len({s.rstrip(_TRAILING).casefold() or s.casefold() for s in secrets})


def carries(text: str | None, secrets: Sequence[str]) -> bool:
    if not text or not secrets:
        return False
    return any(m.group() != MASK for m in _any(tuple(secrets)).finditer(text))


def mask(text: str, secrets: Sequence[str]) -> str:
    """``text`` with every secret replaced by ``MASK``, in one pass (longest
    first) that also matches the masks already there, so a mask is never
    masked again."""
    if not secrets:
        return text
    return _any(tuple(secrets)).sub(MASK, text)


def trim_quote(quote: str | None, secrets: Sequence[str]) -> str | None:
    """A quote without its secrets: the longest piece left between them, or
    None when nothing worth quoting is left."""
    if quote is None or not carries(quote, secrets):
        return quote
    pieces = _any(tuple(secrets)).split(quote)
    best = max((p.strip(" \t\n" + _TRAILING + "`\u201c\u2018\u00ab=") for p in pieces), key=len)
    return best if len(best) >= MIN_QUOTE_CHARS else None


def _fact_carries(fact: ProposedFact, secrets: Sequence[str]) -> bool:
    fields = (
        fact.statement,
        fact.value,
        fact.value_norm,
        fact.subject,
        fact.subject_as_written,
        fact.removes,
        fact.attribute,
    )
    return any(carries(field, secrets) for field in fields)


def shaped(value: str) -> bool:
    """Is ``value`` a key or a token by its shape alone (``sk-...``, a JWT)?"""
    return any(shape.fullmatch(value) for shape in _SHAPES)


def claimed(answers: Iterable[Answer]) -> set[str]:
    """The values the answers name as secrets themselves (listed, or under a
    credential attribute), case folded."""
    out: set[str] = set()
    for answer in answers:
        out.update(s.strip().casefold() for s in answer.secrets)
        out.update(
            word.casefold()
            for f in answer.facts
            if f.value is not None and credential_key(f.attribute)
            for word in _credential_values(f.value)
        )
    return out


def _scrub_facts(
    answer: Answer,
    secrets: Sequence[str],
    strong: Sequence[str],
    owned: Callable[[ProposedFact], bool] | None,
) -> int:
    """The facts part of ``scrub``: a fact the owner's words hold is checked
    against the ``strong`` secrets only."""
    dropped = 0
    facts: list[ProposedFact] = []
    for fact in answer.facts:
        if credential_key(fact.attribute):
            if not _eventful(fact.value):
                dropped += 1
                continue
            value = (fact.value or "").strip()
            if " " in value and (
                written_in(fact.statement, value) or written_in(fact.quote or "", value)
            ):
                dropped += 1  # the words of the value are in the fact: it may hold them
                continue
            fact.attribute, fact.value, fact.value_norm = None, None, None
        mine = owned is not None and owned(fact)
        held = strong if mine else secrets
        if _fact_carries(fact, held):
            dropped += 1
            continue
        # every secret leaves the quote; an owned fact left without one keeps
        # its words but the strong secrets' (a page cannot empty it)
        quote = trim_quote(fact.quote, secrets)
        if mine and fact.quote is not None and quote is None:
            quote = trim_quote(fact.quote, held)
        if fact.quote is not None and quote is None:
            dropped += 1  # nothing of its words is left to show (P5)
            continue
        fact.quote = quote
        facts.append(fact)
    answer.facts = facts
    return dropped


def scrub(
    answer: Answer,
    secrets: Sequence[str],
    *,
    weak: Sequence[str] = (),
    owned: Callable[[ProposedFact], bool] | None = None,
) -> int:
    """Remove ``secrets`` from one answer, in place; returns how many items
    (facts, nodes, edges, an activity) were dropped for carrying one. A fact
    with a credential attribute is dropped when its value is one; an event
    there ("changed") keeps the fact without its key. ``weak``: the values the
    service read only in content from outside the conversation (P6): a page
    cannot erase the owner's words with them, so they drop only a fact that
    ``owned`` does not give to the owner, and never a node or an edge."""
    strong = [s for s in secrets if s not in weak]
    dropped = _scrub_facts(answer, secrets, strong, owned)
    if not secrets:
        return dropped
    # a name is the owner's word as much as the page's: only a strong secret
    # drops it; a summary, an alias or a quote loses every secret
    nodes = []
    for node in answer.nodes:
        if carries(node.name, strong):
            dropped += 1
            continue
        node.aliases = [a for a in node.aliases if not carries(a, secrets)]
        if carries(node.summary, secrets):
            node.summary = ""
        nodes.append(node)
    answer.nodes = nodes
    edges = []
    for edge in answer.edges:
        if carries(edge.src, strong) or carries(edge.dst, strong) or carries(edge.type, strong):
            dropped += 1
            continue
        edge.quote = trim_quote(edge.quote, secrets)
        edges.append(edge)
    answer.edges = edges
    activity = answer.activity
    if activity is not None:
        quote = trim_quote(activity.quote, secrets) or trim_quote(activity.quote, strong)
        if carries(activity.name, strong) or quote is None:
            answer.activity = None
            dropped += 1
        else:
            activity.quote = quote
    return dropped


def named_values(answer: Answer) -> list[str]:
    """What an answer itself names as secrets: its list, and the words of its
    credential facts' values that look like secrets."""
    return [
        *answer.secrets,
        *(
            word
            for f in answer.facts
            if f.value is not None and credential_key(f.attribute)
            for word in _credential_values(f.value)
        ),
    ]


__all__ = [
    "MASK",
    "carries",
    "claimed",
    "collect",
    "credential_key",
    "detect",
    "distinct",
    "mask",
    "named_values",
    "scrub",
    "shaped",
    "trim_quote",
    "written_in",
]
