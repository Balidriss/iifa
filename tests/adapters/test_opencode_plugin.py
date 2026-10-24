"""The opencode adapter's TypeScript, held to iifa's Python where the two must
agree (SPEC 10.4). Its behaviour is tested under ``node --test`` against the
same fake service (adapters/opencode/test/plugin.test.ts, the CI's node job);
these checks need no Node: the P8 wrapper and the budget reserves are the same
strings and numbers, so what the plugin injects fits the budget the way the
Claude Code hook's does (P1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from iifa.recalled import NOTICE, as_recalled_data
from iifa.tokens import ConservativeTokenizer

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "adapters" / "opencode"
CORE = (ADAPTER / "plugin" / "iifa" / "core.ts").read_text(encoding="utf-8")

pytestmark = pytest.mark.security


def _constant(name: str) -> str:
    match = re.search(rf"export const {name} =\s*(.+?)\n(?=\S)", CORE, re.S)
    assert match, name
    return match.group(1).strip()


def _number(name: str) -> int:
    return int(_constant(name).split("//")[0].replace("_", ""))


def test_p8_the_notice_and_the_wrapper_are_iifas() -> None:
    pieces = re.findall(r'"((?:[^"\\]|\\.)*)"', _constant("NOTICE"))
    assert "".join(pieces) == NOTICE
    template = re.search(r"return `(<recalled-data[^`]+)`", CORE)
    assert template
    expected = as_recalled_data({"a": 1}, kind="recall").replace('{"a":1}', "${pyJson(payload)}")
    expected = expected.replace('kind="recall"', 'kind="${kind}"').replace(NOTICE, "${NOTICE}")
    assert template.group(1).replace("\\n", "\n") == expected


def test_p1_the_reserves_are_the_other_adapters_and_cover_the_wrapper() -> None:
    hook = (ROOT / "adapters" / "claude-code" / "hooks" / "iifa_hook.py").read_text(
        encoding="utf-8"
    )
    for name in ("WRAPPER_RESERVE_TOKENS", "PROFILE_RESERVE_TOKENS"):
        python_value = re.search(rf"^{name} = (\d+)", hook, re.M)
        assert python_value and _number(name) == int(python_value.group(1)), name
    tokenizer = ConservativeTokenizer()
    for kind in ("recall", "profile"):
        wrapper = tokenizer.count(as_recalled_data({}, kind=kind)) - tokenizer.count("{}")
        assert wrapper < _number("WRAPPER_RESERVE_TOKENS")


def test_the_limits_are_the_apis() -> None:
    assert _number("MAX_QUERY_CHARS") == 8000
    assert _number("MAX_EXCLUDE_IDS") == 2000
    assert _number("MAX_TURN_CHARS") == 100_000


def test_no_dependency_and_a_node_job_in_ci() -> None:
    package = json.loads((ADAPTER / "package.json").read_text(encoding="utf-8"))
    assert "dependencies" not in package and "devDependencies" not in package
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "actions/setup-node" in ci
    assert "node --test adapters/opencode/test/plugin.test.ts" in ci
