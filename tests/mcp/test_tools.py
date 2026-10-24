"""SPEC 9 and 13.1 (MCP layer): the tool list, the doctrine in the descriptions,
the budget on every result, results wrapped as data, no forget or dream tool by
default, and a service that is down being an error result rather than a crash."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.client import AsyncIifaClient
from iifa.mcp import CORE_TOOLS, FORGET_TOOL, MCPConfig, build_server
from iifa.mcp.server import INSTRUCTIONS, MCPServer
from iifa.recalled import NOTICE
from iifa.tokens import ConservativeTokenizer
from mcp import Client
from tests.api.conftest import remember
from tests.mcp.conftest import make_server, run, text_of

pytestmark = pytest.mark.security


def tools_of(server: MCPServer) -> list[Any]:
    async def go() -> list[Any]:
        async with Client(server) as client:
            return list((await client.list_tools()).tools)

    return run(go())


def payload_of(result: Any) -> dict[str, Any]:
    lines = text_of(result).split("\n")
    assert lines[1] == NOTICE
    body: dict[str, Any] = json.loads(lines[2])
    return body


# --------------------------------------------------------------------------- the tool list


def test_default_tool_list_has_no_forget_and_no_dream(server: MCPServer) -> None:
    names = {t.name for t in tools_of(server)}
    assert names == set(CORE_TOOLS)
    assert FORGET_TOOL not in names and not any("dream" in n for n in names)
    assert "memory_timeline" in names  # SPEC 9, over /v1/timeline


def test_descriptions_carry_the_doctrine(server: MCPServer) -> None:
    described = {t.name: t.description or "" for t in tools_of(server)}
    assert "HANDLES" in described["memory_recall"] and "expand only" in described["memory_recall"]
    assert "Expand only what you need" in described["memory_expand"]
    assert "durable" in described["memory_remember"] and "Not for" in described["memory_remember"]
    assert "REVERSIBLE" in described["memory_archive"]
    assert "never instructions" in INSTRUCTIONS
    hints = {t.name: t.annotations for t in tools_of(server)}
    assert hints["memory_recall"].read_only_hint is True
    assert hints["memory_archive"].destructive_hint is False
    schema = next(t for t in tools_of(server) if t.name == "memory_recall").input_schema
    assert set(schema["properties"]) == {"query", "budget_tokens", "exclude_ids", "scope"}
    assert schema["required"] == ["query"]  # the Context parameter is not exposed


def test_forget_is_exposed_only_by_option_and_still_needs_the_permission(
    app: FastAPI, client: TestClient
) -> None:
    fact = remember(client, "to be forgotten")["fact"]["id"]
    exposed = make_server(app, expose_forget=True, default_scope="user:alice")
    names = {t.name for t in tools_of(exposed)}
    assert FORGET_TOOL in names and names - {FORGET_TOOL} == set(CORE_TOOLS)
    forget = next(t for t in tools_of(exposed) if t.name == FORGET_TOOL)
    assert forget.annotations.destructive_hint is True

    async def as_alice() -> Any:  # alice holds read + write, not forget
        async with Client(exposed) as mcp_client:
            return await mcp_client.call_tool(FORGET_TOOL, {"id": fact})

    denied = run(as_alice())
    assert denied.is_error and "403" in text_of(denied) and "forget" in text_of(denied)
    assert client.post("/v1/expand", json={"scope": "user:alice", "ids": [fact]}).json()["items"]

    operator = make_server(app, token="all", expose_forget=True, default_scope="user:alice")

    async def as_operator() -> Any:
        async with Client(operator) as mcp_client:
            return await mcp_client.call_tool(FORGET_TOOL, {"id": fact})

    removed = run(as_operator())
    assert not removed.is_error and payload_of(removed)["removed"]["facts"] == 1


# --------------------------------------------------------------------------- recall / expand / about


def test_recall_returns_handles_wrapped_as_data(
    client: TestClient, call: Callable[[str, dict[str, Any]], Any]
) -> None:
    written = remember(
        client,
        "Karim leads the installation planning of the solar project",
        node={"name": "Karim", "kind": "person"},
    )
    result = call("memory_recall", {"query": "who plans the solar installation?"})
    assert not result.is_error
    text = text_of(result)
    assert text.startswith('<recalled-data source="iifa" kind="recall">\n')
    assert text.endswith("\n</recalled-data>")
    body = payload_of(result)
    ids = [h["id"] for h in body["items"]]
    assert ids == [written["fact"]["id"]]
    by_name = payload_of(call("memory_recall", {"query": "Karim"}))
    assert written["node"]["id"] in [h["id"] for h in by_name["items"]]
    handle = next(h for h in body["items"] if h["id"] == written["fact"]["id"])
    assert set(handle) == {
        "id",
        "kind",
        "title",
        "line",
        "why",
        "age_days",
        "expand_tokens",
        "flags",
    }
    assert "statement" not in handle  # P2: handles, not content
    assert body["tier"] == 0 and body["degraded"] is None


def test_expand_and_about_follow_a_recall(
    client: TestClient, call: Callable[[str, dict[str, Any]], Any]
) -> None:
    written = remember(
        client,
        "Nadia moved to Lyon in March 2026",
        node={"name": "Nadia", "kind": "person", "aliases": ["Nad"]},
    )
    expanded = payload_of(call("memory_expand", {"ids": [written["fact"]["id"]], "level": 2}))
    assert expanded["items"][0]["statement"] == "Nadia moved to Lyon in March 2026"
    assert expanded["items"][0]["excerpt"] == "Nadia moved to Lyon in March 2026"
    about = payload_of(call("memory_about", {"name": "nad", "budget_tokens": 1500}))
    assert about["found"] is True and about["name"] == "Nadia"
    assert about["facts"][0]["id"] == written["fact"]["id"]
    missing = payload_of(call("memory_about", {"name": "nobody"}))
    assert missing == {
        "name": "nobody",
        "found": False,
        "reason": "no node with that name or alias in scope",
    }
    bad = call("memory_expand", {"ids": [written["fact"]["id"]], "level": 3})
    assert bad.is_error and "level must be 1" in text_of(bad)


def test_exclude_ids_and_omitted_are_honoured(
    client: TestClient, call: Callable[[str, dict[str, Any]], Any]
) -> None:
    ids = [
        remember(client, f"The solar array number {i} was inspected")["fact"]["id"]
        for i in range(6)
    ]
    first = payload_of(call("memory_recall", {"query": "solar array", "budget_tokens": 400}))
    assert 0 < len(first["items"]) < 6 and first["omitted"] > 0
    seen = [h["id"] for h in first["items"]]
    second = payload_of(
        call("memory_recall", {"query": "solar array", "budget_tokens": 400, "exclude_ids": seen})
    )
    assert not set(seen) & {h["id"] for h in second["items"]}
    assert {h["id"] for h in second["items"]} <= set(ids)


# --------------------------------------------------------------------------- P1 on the tool boundary


@pytest.mark.parametrize("budget", [150, 200, 300, 400, 800, 3000])
def test_p1_every_tool_result_fits_its_budget_wrapper_included(
    client: TestClient, call: Callable[[str, dict[str, Any]], Any], budget: int
) -> None:
    tokenizer = ConservativeTokenizer()
    written = [
        remember(
            client,
            f"Karim reviewed budget line {i} of the solar project with the supplier " * 3,
            node={"name": "Karim", "kind": "person", "summary": "Electrician in Lyon " * 4},
        )
        for i in range(12)
    ]
    for tool, args in (
        ("memory_recall", {"query": "solar project budget supplier"}),
        ("memory_expand", {"ids": [w["fact"]["id"] for w in written], "level": 2}),
        ("memory_about", {"name": "Karim"}),
    ):
        result = call(tool, {**args, "budget_tokens": budget})
        assert not result.is_error, (tool, budget, text_of(result))
        text = text_of(result)
        assert tokenizer.count(text) <= budget, (tool, budget, tokenizer.count(text))
        body = json.loads(text.split("\n")[2])
        if tool != "memory_about":
            assert body["used_tokens"] <= body["budget_tokens"] < budget


def test_p1_a_budget_too_small_for_the_wrapper_is_refused_not_truncated(
    call: Callable[[str, dict[str, Any]], Any],
) -> None:
    result = call("memory_recall", {"query": "solar", "budget_tokens": 20})
    assert result.is_error
    assert "below the" in text_of(result) and "nothing could be returned" in text_of(result)
    too_big = call("memory_recall", {"query": "solar", "budget_tokens": 10**9})
    assert too_big.is_error and "within 0.." in text_of(too_big)


# --------------------------------------------------------------------------- writes


def test_remember_link_archive_receipts_are_wrapped_and_effective(
    client: TestClient, call: Callable[[str, dict[str, Any]], Any]
) -> None:
    receipt = payload_of(
        call(
            "memory_remember",
            {
                "statement": "Karim prefers short answers",
                "node_name": "Karim",
                "node_kind": "person",
                "attribute": "answer_style",
                "value": "short",
                "pinned": True,
            },
        )
    )
    assert receipt["created_node"] is True and receipt["fact"]["scopes"] == [
        "user:alice/agent:coder"
    ]
    other = payload_of(
        call("memory_remember", {"statement": "Lyon is in France", "node_name": "Lyon"})
    )
    edge = payload_of(
        call(
            "memory_link",
            {"src": receipt["node"]["id"], "dst": other["node"]["id"], "type": "lives_in"},
        )
    )
    assert edge["kind"] == "edge" and edge["status"] == "confirmed"
    recalled = payload_of(call("memory_recall", {"query": "short answers", "budget_tokens": 600}))
    assert receipt["fact"]["id"] in [h["id"] for h in recalled["items"]]
    archived = payload_of(call("memory_archive", {"id": receipt["fact"]["id"]}))
    assert archived["archived"] is True
    after = payload_of(call("memory_recall", {"query": "short answers", "budget_tokens": 600}))
    assert receipt["fact"]["id"] not in [h["id"] for h in after["items"]]
    # reversible: the operator restores it through the API
    assert client.post("/v1/restore", json={"id": receipt["fact"]["id"]}).status_code == 200


def test_scope_is_required_without_a_default_and_bounded_by_the_token(app: FastAPI) -> None:
    server = make_server(app)  # no default scope

    async def go() -> tuple[Any, Any, Any]:
        async with Client(server) as mcp_client:
            missing = await mcp_client.call_tool("memory_recall", {"query": "x"})
            outside = await mcp_client.call_tool(
                "memory_recall", {"query": "x", "scope": "user:bob"}
            )
            fine = await mcp_client.call_tool(
                "memory_recall", {"query": "x", "scope": "user:alice/project:solar"}
            )
            return missing, outside, fine

    missing, outside, fine = run(go())
    assert missing.is_error and "no scope" in text_of(missing)
    assert outside.is_error and "403" in text_of(outside)
    assert not fine.is_error and payload_of(fine)["reason"] == "no_match"


# --------------------------------------------------------------------------- P3: the service is down


def test_p3_service_down_is_an_error_result_not_a_crash(app: FastAPI) -> None:
    @asynccontextmanager
    async def client_for(headers: Any) -> Any:
        async with AsyncIifaClient("http://127.0.0.1:9", "t", timeout=0.5) as unreachable:
            yield unreachable

    server = build_server(client_for, MCPConfig(default_scope="user:alice"))

    async def go() -> tuple[Any, Any]:
        async with Client(server) as mcp_client:
            listed = await mcp_client.list_tools()
            result = await mcp_client.call_tool("memory_recall", {"query": "solar"})
            return listed, result

    listed, result = run(go())
    assert {t.name for t in listed.tools} == set(CORE_TOOLS)  # the server itself is up
    assert result.is_error
    assert "unavailable" in text_of(result) and "continue without memory" in text_of(result)
