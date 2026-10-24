"""A small chat loop with iifa memory on the OpenAI Agents SDK.

    pip install ".[openai-agents]"            # from a checkout of iifa
    export OPENAI_API_KEY=...                 # or OPENAI_BASE_URL for a compatible server
    export IIFA_TOKEN=agent-secret IIFA_SCOPE=user:alice/mode:assistant
    python adapters/openai-agents/example.py

Type a message, read the answer; an empty line ends the conversation.
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from agents import Agent, Runner, set_tracing_disabled
from iifa_agents import IifaMemory


async def main() -> None:
    # The SDK exports traces (instructions, the profile block included) to OpenAI by
    # default; memory stays local unless you decide otherwise (P4).
    set_tracing_disabled(True)
    memory = IifaMemory.from_env(session_id=f"example-{uuid.uuid4().hex[:8]}")
    agent = Agent(
        name="assistant",
        instructions=memory.instructions(
            "You are a helpful assistant with long-term memory. Use the memory_* tools "
            "to look things up and to store durable facts the user tells you."
        ),
        tools=memory.tools(),  # the model: the SDK's default, or OPENAI_DEFAULT_MODEL
    )
    try:
        while prompt := (await asyncio.to_thread(input, "> ")).strip():
            result = await Runner.run(agent, prompt)
            answer = str(result.final_output)
            print(answer)
            memory.record_turn_soon(prompt, answer)  # the next prompt does not wait for it
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        await memory.end_session()
        await memory.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
