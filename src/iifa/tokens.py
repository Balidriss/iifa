"""Token counting (SPEC section 7.3).

The budget is a contract (P1), so the counter must never say "fewer" than the
agent's real tokenizer will. The default is a conservative estimate that needs
no model file: it over-counts by design. Exact counters are optional extras and
must be pre-cached by the operator, since iifa downloads nothing at runtime (P4).
"""

from __future__ import annotations

import math
import string
from typing import Protocol

__all__ = ["ConservativeTokenizer", "Tokenizer", "get_tokenizer"]


class Tokenizer(Protocol):
    """Counts tokens. ``per_item_overhead`` covers the JSON wrapper around one item."""

    name: str
    per_item_overhead: int

    def count(self, text: str) -> int: ...


class ConservativeTokenizer:
    """Characters / 3.5, rounded up, plus a per-item overhead, with refinements
    that keep the estimate conservative outside lower-case English prose. Each
    rate is at or above what byte-pair tokenizers spend, measured against
    ``cl100k_base`` in CI on English, French, Russian, Arabic, Chinese,
    Japanese and mixed samples:

    * lower-case ASCII letters and whitespace: 1 token per 3.5 characters
      (the SPEC formula);
    * upper-case letters and digits: 1 token each (identifiers such as
      ``fact_01J8Z3M4...`` and dates tokenize close to one piece per character);
    * other ASCII (punctuation, JSON syntax): 1 token per 2 characters;
    * every non-ASCII character: its UTF-8 length minus one, so 1 for two-byte
      scripts (accented Latin, Cyrillic, Arabic, Hebrew, Greek), 2 for CJK and
      3 for emoji.

    It over-counts, which costs budget, never correctness (P1).
    """

    name = "conservative"
    per_item_overhead = 8
    _prose_bytes = bytes(string.ascii_lowercase + string.whitespace, "ascii")
    _unit_bytes = bytes(string.ascii_uppercase + string.digits, "ascii")

    def count(self, text: str) -> int:
        ascii_only = text.encode("ascii", "ignore")
        ascii_chars = len(ascii_only)
        prose = ascii_chars - len(ascii_only.translate(None, self._prose_bytes))
        units = ascii_chars - len(ascii_only.translate(None, self._unit_bytes))
        other = ascii_chars - prose - units
        non_ascii = len(text.encode("utf-8")) - len(text)  # bytes minus characters
        return math.ceil(prose / 3.5 + units + other / 2 + non_ascii)


class TiktokenTokenizer:
    """Exact counts through ``tiktoken`` (optional extra ``iifa[exact-tokens]``).

    The encoding file must already be in ``TIKTOKEN_CACHE_DIR``: iifa never
    downloads at runtime (P4), and a missing cache is a configuration error.
    """

    per_item_overhead = 4

    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "tokenizer 'tiktoken' needs: pip install 'iifa[exact-tokens]'"
            ) from exc
        self.name = f"tiktoken:{encoding}"
        self._encoding = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


def get_tokenizer(name: str) -> Tokenizer:
    """``conservative`` (default) or ``tiktoken:<encoding>``."""
    if name == "conservative":
        return ConservativeTokenizer()
    if name.startswith("tiktoken:"):
        return TiktokenTokenizer(name.split(":", 1)[1])
    raise ValueError(f"unknown tokenizer {name!r}")
