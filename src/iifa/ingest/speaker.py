"""Speaker words (SPEC 4.2): the subjects that, whole, name the person speaking.

In an assistant the person speaking is the owner of the memory. A model that
reads "User: I moved to Lyon." writes the subject "User", "the user",
"l'utilisateur" or "Пользователь"; such a subject is never a node of its own. It
means the scope's declared owner, or no node at all (a free-standing fact).

The rule is narrow on purpose: the whole subject, after folding case and
diacritics and dropping one leading article or elision, must be in the table.
"User Group Lyon" is an ordinary name, and pronouns are not in the table.
"""

from __future__ import annotations

import re
import unicodedata

_WORDS = (
    # English, French, Spanish, Portuguese, German, Italian
    "user",
    "utilisateur",
    "utilisatrice",
    "usuario",
    "usuaria",
    "utilizador",
    "utilizadora",
    "benutzer",
    "benutzerin",
    "nutzer",
    "nutzerin",
    "utente",
    # Russian, Arabic
    "пользователь",
    "пользовательница",
    "المستخدم",
    "مستخدم",
    # Chinese, Japanese, Korean
    "用户",
    "用戶",
    "使用者",
    "ユーザー",
    "ユーザ",
    "사용자",
)
_KANA_MARKS = (
    "\N{COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK}"
    "\N{COMBINING KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK}"
)
_ARTICLE = re.compile(
    r"^(?:l'|the\s+|le\s+|la\s+|el\s+|lo\s+|il\s+|o\s+|a\s+|der\s+|die\s+|das\s+)"
)


def _fold(text: str) -> str:
    """Case and diacritics folded, the typographic apostrophe made plain, spaces
    collapsed (letters of every script kept)."""
    decomposed = unicodedata.normalize("NFKD", text.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'"))
    # the kana voicing marks are letters, not accents: "ザ" stays "ザ"
    bare = "".join(c for c in decomposed if not unicodedata.combining(c) or c in _KANA_MARKS)
    return " ".join(unicodedata.normalize("NFC", bare).casefold().split())


_FOLDED = frozenset(_fold(w) for w in _WORDS)


def speaker_words() -> frozenset[str]:
    """The table, folded."""
    return _FOLDED


def is_speaker(subject: str | None) -> bool:
    """Does ``subject``, whole, name the person speaking?"""
    if not subject:
        return False
    folded = _fold(subject)
    return folded in _FOLDED or _ARTICLE.sub("", folded, count=1) in _FOLDED


__all__ = ["is_speaker", "speaker_words"]
