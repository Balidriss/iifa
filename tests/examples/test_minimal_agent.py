"""The reference loop (SPEC 10.6) against the fake model server and the real
service: it injects recalled data before a turn, remembers on demand, keeps
under 200 lines, and never lets memory block the turn."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from iifa.client import IifaClient
from iifa.recalled import NOTICE
from tests.fakes.iifa_service import Fail, FakeIifaService
from tests.fakes.live_service import LiveService
from tests.fakes.openai_server import FakeOpenAIServer, Reply

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "minimal_agent.py"


@pytest.fixture(scope="module")
def example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("minimal_agent", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveService]:
    with LiveService(tmp_path_factory.mktemp("agent-live")) as service:
        yield service


def make_agent(example: ModuleType, fake_openai: FakeOpenAIServer, url: str, token: str) -> Any:
    env = {
        "OPENAI_BASE_URL": fake_openai.base_url,
        "OPENAI_API_KEY": "x",
        "MODEL": "fake-chat",
        "IIFA_URL": url,
        "IIFA_TOKEN": token,
        "IIFA_SCOPE": "user:demo/agent:minimal",
    }
    return example.build(env)


def test_under_200_lines() -> None:
    assert len(EXAMPLE.read_text(encoding="utf-8").splitlines()) < 200


def test_recall_is_injected_as_data_before_the_turn(
    example: ModuleType, fake_openai: FakeOpenAIServer, live: LiveService
) -> None:
    agent = make_agent(example, fake_openai, live.url, live.token())
    assert agent.memory.session_start() == ""  # moment 1: connected, nothing pinned yet
    assert agent.command("/remember Karim leads the solar installation planning").startswith(
        "remembered as fact_"
    )
    fake_openai.script(chat=[Reply("Karim does.")])
    assert agent.turn("who plans the solar installation?") == "Karim does."  # moment 2 + 3
    sent = fake_openai.chat_calls[-1].messages
    system_blocks = [m["content"] for m in sent if m["role"] == "system"]
    assert system_blocks[0].startswith("You are a helpful assistant")
    recalled = system_blocks[1]
    assert recalled.startswith('<recalled-data source="iifa" kind="recall">\n' + NOTICE)
    assert "solar installation planning" in recalled
    assert sent[-1] == {"role": "user", "content": "who plans the solar installation?"}
    # a second turn does not resend what was already injected
    fake_openai.script(chat=[Reply("Yes.")])
    agent.turn("does Karim plan the solar installation?")
    again = [m for m in fake_openai.chat_calls[-1].messages if m["role"] == "system"]
    assert len(again) == 1
    # expansion on demand, as data too
    fact_id = agent.memory.injected[0]
    assert agent.command(f"/expand {fact_id}").startswith(
        '<recalled-data source="iifa" kind="expand">'
    )
    assert agent.memory.notices == []


MOMENTS = (
    "/v1/session/start",
    "/v1/recall",
    "/v1/turns",
    "/v1/remember",
    "/v1/expand",
    "/v1/precompact",
    "/v1/session/end",
)


def test_p3_memory_down_never_blocks_the_turn(
    example: ModuleType, fake_openai: FakeOpenAIServer
) -> None:
    """The service answers 503 on every route (the fake's fail mode, not a closed
    port: nothing here depends on how a refused connection behaves)."""
    with FakeIifaService() as service:
        for path in MOMENTS:
            service.script(path, [Fail(503, "service unavailable (scripted)")] * 3)
        agent = make_agent(example, fake_openai, service.url, "t")
        _memory_down_turn(agent, fake_openai)
        assert {path for path in MOMENTS if service.calls_to(path)} == set(MOMENTS)


def _memory_down_turn(agent: Any, fake_openai: FakeOpenAIServer) -> None:
    assert agent.memory.session_start() is None
    fake_openai.script(chat=[Reply("Hello!")])
    assert agent.turn("hello") == "Hello!"
    assert agent.command("/remember something") == "not remembered: memory unavailable"
    assert agent.command("/expand fact_x").startswith("nothing to show")
    assert agent.memory.before_compact([{"role": "user", "content": "x"}]) is None
    agent.memory.session_end()
    assert [n.split(":")[0] for n in agent.memory.notices] == [
        "memory unavailable at session start",
        "memory unavailable at recall",
        "memory unavailable at turn",
        "memory unavailable at remember",
        "memory unavailable at expand",
        "memory unavailable at precompact",
        "memory unavailable at session end",
    ]
    only_system_and_user = fake_openai.chat_calls[-1].messages
    assert [m["role"] for m in only_system_and_user] == ["system", "user"]


def test_history_is_compacted_after_the_threshold(
    example: ModuleType, fake_openai: FakeOpenAIServer, live: LiveService
) -> None:
    agent = make_agent(example, fake_openai, live.url, live.token())
    dropped: list[list[dict[str, str]]] = []
    agent.memory.before_compact = dropped.append
    fake_openai.set_default(chat=Reply("ok"))
    for i in range(example.COMPACT_AFTER // 2 + 2):
        agent.turn(f"message {i} with no memory match zzqx")
    assert len(agent.history) == example.COMPACT_AFTER
    assert dropped and dropped[0][0]["content"] == "message 0 with no memory match zzqx"


def test_the_profile_block_and_the_recorded_turns_reach_the_service(
    example: ModuleType, fake_openai: FakeOpenAIServer, live: LiveService
) -> None:
    with IifaClient(live.url, live.token(), scope="user:demo/agent:minimal") as api:
        api.remember("Always answer in French", pinned=True)
        api.set_background(mode="on_demand")  # keep the queue observable
        agent = make_agent(example, fake_openai, live.url, live.token())
        agent.profile = agent.memory.session_start()  # moment 1
        assert agent.profile.startswith('<recalled-data source="iifa" kind="profile">')
        assert "Always answer in French" in agent.profile
        fake_openai.script(chat=[Reply("Bonjour.")])
        assert agent.turn("say hi zzqx") == "Bonjour."  # moments 2 and 3
        sent = fake_openai.chat_calls[-1].messages
        assert sent[1]["content"] == agent.profile  # the profile leads every turn
        pending = live.app.state.queue.jobs(status="pending")
        assert [j.kind for j in pending] == ["turn"]
        assert pending[0].payload["user"] == "say hi zzqx"
        assert pending[0].payload["assistant"] == "Bonjour."
        kept = agent.memory.before_compact([{"role": "user", "content": "old"}])  # moment 4
        assert kept is not None and "Always answer in French" in kept
        agent.memory.session_end()  # moment 5
        assert [j.kind for j in live.app.state.queue.jobs(status="pending")] == [
            "turn",
            "precompact",
            "session_end",
        ]
        assert agent.memory.notices == []
        api.run_background()
        api.set_background(mode="auto")


def test_the_service_sees_what_the_loop_remembered(
    example: ModuleType, fake_openai: FakeOpenAIServer, live: LiveService
) -> None:
    agent = make_agent(example, fake_openai, live.url, live.token())
    agent.command("/remember Nadia moved to Lyon")
    with IifaClient(live.url, live.token(), scope="user:demo/agent:minimal") as api:
        assert api.recall("Lyon")["items"][0]["line"] == "Nadia moved to Lyon"
    with pytest.raises(httpx.HTTPStatusError):  # the model endpoint's own errors do surface
        fake_openai.script(
            chat=[__import__("tests.fakes.openai_server", fromlist=["Fail"]).Fail(500)]
        )
        agent.turn("hello")
