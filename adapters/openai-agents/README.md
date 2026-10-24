# iifa memory with the OpenAI Agents SDK (example)

An example to copy into your own application, not a library: `iifa_agents.py`
(under 400 lines) gives an agent long-term memory under a token budget from a
running iifa service, and `example.py` is a small chat loop using it.

It is **not** a `Session` implementation. The SDK's `Session` protocol
(`get_items` / `add_items`) stores the conversation **history**; iifa is
**memory** — facts, subjects and links that outlive a conversation — and enters
through two other doors:

- **Tools** — `memory_recall`, `memory_expand`, `memory_remember`,
  `memory_archive`, built with `@function_tool`; the model calls them when it
  needs to. Results are wrapped as recalled data (P8) under a budget (P1).
  `memory_about`, `memory_timeline` and `memory_link` are on iifa's MCP server,
  which the SDK can load too (`MCPServerStdio` running `python -m iifa.mcp`).
- **The profile block** — the agent's **dynamic instructions** append it
  (pinned facts and the most central subjects, under budget) to your own
  instructions, fetched once per conversation. The SDK's `RunHooks` /
  `AgentHooks` only observe a run: they cannot change the prompt, dynamic
  instructions can.

After a run, `record_turn_soon(prompt, answer)` hands the turn to the service
(one enqueue there) without making the caller wait; `end_session()` tells the
service the conversation is over.

```python
from agents import Agent, Runner
from iifa_agents import IifaMemory

memory = IifaMemory.from_env(user_id="alice", session_id="chat-42")
agent = Agent(
    name="assistant",
    instructions=memory.instructions("You are a helpful assistant."),
    tools=memory.tools(),
)
result = await Runner.run(agent, prompt)
memory.record_turn_soon(prompt, str(result.final_output))
```

**Fails open** (P3): when the service is down, slow (past the deadline: 1.5 s
for the profile block, 5 s for a tool), answers 4xx/5xx or garbage, or the code
below hits a bug, the instructions are your own, a tool answers "iifa memory
unavailable; continue without it.", nothing is raised into the SDK, and one
line goes to the `iifa.openai_agents` logger (event, reason, duration; never
memory content). The run goes on.

## Requirements and setup

```bash
pip install ".[openai-agents]"       # from a checkout of iifa; the SDK is an optional extra
export IIFA_URL=http://127.0.0.1:8765
export IIFA_TOKEN=agent-secret
export IIFA_SCOPE=user:alice/mode:assistant
export OPENAI_API_KEY=...              # or OPENAI_BASE_URL for a compatible server
python adapters/openai-agents/example.py
```

A scope is `kind:name` at **every** `/` level: `user:alice/mode:assistant`,
never `user:alice/assistant` (the service refuses that with a 422 naming the
segment). `IifaMemory` checks the scope when it is built; a bad one turns memory
off with one log line saying which segment is wrong. Without `IIFA_SCOPE` the
scope is `user:<user_id>/agent:openai-agents` (spaces and `/` in the id
replaced by `_`), or `user:local/agent:openai-agents`.

Optional: `IIFA_HOOK_BUDGET_TOKENS` (default 400), `IIFA_HOOK_DEADLINE_MS`
(default 1500), `IIFA_TOOL_DEADLINE_MS` (default 5000).

The example turns the SDK's tracing off: by default the SDK exports traces —
instructions, the profile block included — to OpenAI. Memory stays local
unless you decide otherwise (P4).

## Built against

`openai-agents` **0.24.0** (PyPI):

- The documentation at `openai.github.io/openai-agents-python/` (agents and
  their `instructions`, tools, context, sessions, lifecycle hooks), checked
  against the package's source: `Agent.instructions` accepts a callable
  `(context, agent)` returning a string or an awaitable one
  (`agents/agent.py`); `RunHooksBase` / `AgentHooksBase` callbacks receive the
  context and the agent but return nothing and cannot change the system prompt
  (`agents/lifecycle.py`); `function_tool` builds the schema from the signature
  and docstring (`agents/tool.py`); `Session` is a history protocol
  (`agents/memory/session.py`); `RunConfig(tracing_disabled=True)` and
  `set_tracing_disabled`; `agents.testing.ScriptedModel` drives the tests.

These APIs move (the SDK is below 1.0): the example follows the version above.

## Tests

`python -m pytest tests/adapters/test_openai_agents.py` (skipped without the
extra) runs real `Agent` and `Runner` objects with the SDK's `ScriptedModel`:
no model, no network. It covers the profile block in the instructions (once per
conversation), each tool's call and wrapped result, the turn recorded without
waiting; fail-open on a stopped service, hangs past the deadline, 422/500,
malformed and wrongly shaped answers and a bug below — the run completes every
time; default and derived scopes valid, a bad scope turning memory off; budget
(P1), hostile memory (P8) and a content-free log (P5) against the real
service; and that iifa's core never imports the SDK.
