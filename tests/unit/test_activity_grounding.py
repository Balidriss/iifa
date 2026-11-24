"""An activity is kept only when a word of its name is written whole
in the owner's words that ask for it."""

from __future__ import annotations

import pytest

from iifa.ingest.activity import grounded_activity


@pytest.mark.parametrize(
    ("name", "quote", "expected"),
    [
        ("dessiner une carte", "fais une carte du verger", True),  # the noun, folded
        ("dessiner une carte", "fais une liste du verger", False),  # "une" is a function word
        ("fixing the bug", "fix this bug", True),  # three letters are enough
        ("planning a garden", "Plan the beds", False),  # "plan" is not "planning"
        ("to do it with her", "do it with her", False),  # short and function words only
        ("绘制地图", "帮我画一张地图", True),  # two letters in a row, no spaces
        ("绘制地图", "帮我写信", False),
        ("Ivan's garden", "the gardening club", False),  # a whole word, never a piece
    ],
)
def test_a_word_of_the_name_must_be_written_whole_in_the_quote(
    name: str, quote: str, expected: bool
) -> None:
    assert grounded_activity(name, quote) is expected
