# iifa memory for Hermes Agent: reference

What each provider method does, how it was built, and how it is tested. To
install it, see [README.md](README.md).

## What each call does

| Hermes calls | What the provider does |
|---|---|
| `system_prompt_block()` | `POST /v1/session/start` once per session: the **profile block** (pinned facts, the most central subjects, a short status) as a `recalled-data` block of kind `profile`, kept for the session. Stable while memory does not change, as Hermes wants static text there. |
| `prefetch(query, session_id=…)` | `POST /v1/recall` with the upcoming message: handles under budget, as a `recalled-data` block of kind `recall`. Ids already injected in the session are sent as `exclude_ids`. Empty messages and slash commands are not recalled. |
| `sync_turn(user, assistant, …)` | **One in-process enqueue**, then it returns. A worker thread (made with Hermes' `spawn_context_thread`, so it stays in the right profile) posts the turn to `POST /v1/turns`, which only enqueues on the service side too. Nothing waits on extraction. |
| `on_pre_compress(messages)` | Hands the last user and assistant messages (20 at most, 32 k characters, tool output dropped) to `POST /v1/precompact`; what must survive comes back under budget and goes into Hermes' **summary prompt**, wrapped as data. |
| `on_session_end(messages)` | `POST /v1/session/end`, queued behind the session's turns so the service sees them first. |
| `on_session_switch(new_id, reset=…)` | Rebinds the session id; on a reset, ids injected under that id may be sent again. |
| `get_tool_schemas()` / `handle_tool_call()` | `memory_recall`, `memory_expand`, `memory_remember`, `memory_archive`: one call each, results wrapped as recalled data, always a JSON string (`{"result": …}` or `{"error": …}`). |

Subagents, cron runs and flushes (`agent_context` other than `primary`) read
memory but write nothing on their own. `memory_about`, `memory_timeline` and
`memory_link` are on iifa's MCP server, which Hermes can load as well
(`python -m iifa.mcp`).

**Fail open.** When the service is down, slow, answers 4xx/5xx or garbage, or
the provider itself hits a bug, nothing is injected, nothing is raised into
Hermes, one line goes to the `iifa.hermes` logger, and the turn proceeds.
Every call on the turn's path stops at the provider's own deadline (1.5 s by
default, well under the 8 s Hermes allows an external `prefetch`). The log
carries event names, ids, counts, reasons and durations, never memory
content.

Hermes can spill an oversized `prefetch` result to a file and hand the model a
preview and a path (`memory.prefetch_spill_enabled`, off by default). You do
not need it: the budget keeps every block small, and `memory_expand` is the
native way to read more.

## Built against

Hermes Agent **v0.22.4** (git tag):

- *Building a Memory Provider Plugin*
  (`website/docs/developer-guide/memory-provider-plugin.md` at that tag, served
  at `hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin`):
  directory layout and discovery (`$HERMES_HOME/plugins/<name>/`, found by
  `MemoryProvider` / `register_memory_provider` in `__init__.py`),
  `register(ctx)` returning a new instance per load, `plugin.yaml`, the
  config schema (`secret` + `env_var` go to `.env`, the rest to
  `save_config`), the threading contract (`sync_turn` non-blocking, threads
  through `spawn_context_thread`), profile isolation, the single-provider
  rule, opt-in prefetch spilling, the pre-compress checkpoint API (this
  provider stays on version 1, best effort).
- `agent/memory_provider.py`: the ABC's exact signatures — `prefetch(query, *,
  session_id="")`, `sync_turn(user_content, assistant_content, *, session_id="",
  messages=None, turn_author=None)`, `on_session_switch(new_session_id, *,
  parent_session_id="", reset=False, rewound=False, **kwargs)`,
  `handle_tool_call(tool_name, args, **kwargs) -> str` (a JSON string),
  `get_tool_schemas()` returning `{"name", "description", "parameters"}`, and
  the `initialize` keywords (`hermes_home`, `platform`, `agent_context`,
  `user_id`, …).
- `agent/memory_manager.py`: the external `prefetch` bound (8 s), `sync_turn`
  run on the manager's background worker, `on_pre_compress` results joined into
  the summary prompt, tool calls routed with `hermes_home` and friends as
  keywords.
- `plugins/memory/__init__.py`: user providers imported under a synthetic
  module name (so the directory name `iifa` does not shadow the `iifa`
  package).

Not yet run inside a live Hermes process: the tests drive the provider the way
`MemoryManager` does, against a scripted fake service and the real one. These
APIs move: the provider follows the version above.

## Tests

`python -m pytest tests/adapters/test_hermes_provider.py` — fail-open on a
stopped service, on hangs past the deadline, on 401/422/500/503, on bodies that
are not JSON or not the expected shape, and on a bug below the provider; the
deadline unit-tested with an injected clock; `sync_turn` proven to return
before any network call (a seam holds the worker, no clock involved) and not to
wait on a hanging service; the bounded queue; session end queued behind the
turns; the data wrapper and the scope check held identical to iifa's; default
and derived scopes valid for any user id; the budget, hostile memory and a log
without memory content against the real service.
