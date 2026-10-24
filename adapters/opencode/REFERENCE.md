# iifa memory for opencode: reference

What each hook does, how the plugin was built, and how it is tested. To
install it, see [README.md](README.md).

## What each moment does

| opencode moment | What the plugin does |
|---|---|
| `experimental.chat.system.transform` | `POST /v1/session/start` the first time a session asks: the **profile block** (pinned facts, the most central subjects, a short status) is added to the system prompt as a `recalled-data` block of kind `profile`, and kept for the session (opencode calls this hook before every model request). |
| `chat.message` | `POST /v1/recall` with the prompt: handles under budget, added to the user message as a **synthetic** text part (sent to the model, hidden in the interface). Ids already injected in the session are sent as `exclude_ids`. Prompts under three characters and slash commands are not recalled. |
| events `message.updated`, `message.part.updated`, `session.idle` | The answer's text parts are collected as they stream; on `session.idle` the prompt and the answer go to `POST /v1/turns` (one enqueue on the service side). |
| `experimental.session.compacting` | The session's recent turns (20 messages, 32 k characters at most) go to `POST /v1/precompact`; what must survive comes back under budget and is appended to the compaction prompt (`output.context`), wrapped as data. opencode's own prompt is kept. |
| event `session.deleted`, `dispose` | `POST /v1/session/end`, best effort: opencode has no session-end moment. |
| `tool` | `memory_recall`, `memory_expand`, `memory_remember`, `memory_archive`: one call each, results wrapped as recalled data. |

A subagent's session (one with a parent) reads memory but records nothing.

**Fail open.** opencode awaits each hook with no timeout and does not catch a
hook's error, so the plugin bounds every call itself (1.5 s by default on the
turn's path, 5 s for a tool) and never throws: when the service is down,
slow, answers 4xx/5xx or garbage, or the plugin hits a bug, nothing is
injected, a tool answers "iifa memory unavailable (…); continue without it.",
and one line goes to opencode's log (service `iifa`). The log carries event
names, counts, reasons and durations, never memory content.

The plugin has no dependency of its own: it uses opencode's `tool` helper,
which opencode provides to every plugin directory. opencode loads every file
of `.opencode/plugins/` as a plugin, and not its subdirectories, which is why
the logic sits in `iifa/core.ts`.

## Built against

opencode **1.19.4** (`opencode-ai` and `@opencode-ai/plugin` 1.19.4 on npm,
git tag `v1.19.4` of `anomalyco/opencode`):

- *Plugins* (`opencode.ai/docs/plugins/`): local
  plugin directories, plugins from npm, the `Plugin` signature, `tool()` with
  `tool.schema` (zod), `client.app.log`, the event list,
  `experimental.session.compacting` (`output.context`, `output.prompt`).
- `@opencode-ai/plugin` 1.19.4 `dist/index.d.ts` and `dist/tool.d.ts`: the
  `Hooks` interface (`chat.message` input and output, `experimental.chat.system
  .transform` with an optional `sessionID`, `event`, `dispose`, `tool`), the
  `PluginModule` `{ id, server }` default export, `ToolContext`; and
  `@opencode-ai/sdk` 1.19.4's `TextPart` (`synthetic`, `ignored`), `Session`
  (`parentID`), `EventSessionIdle`, `EventMessagePartUpdated`.
- The source at that tag: `session/prompt.ts` (where `chat.message` fires, after
  part ids are assigned: an added part needs its own `prt` id),
  `session/llm/request.ts` (the system transform before every request),
  `session/compaction.ts`, `session/status.ts` (`session.idle` after a run),
  `plugin/index.ts` (hooks awaited in sequence with no timeout and no catch;
  events delivered without waiting), `plugin/shared.ts` and
  `config/plugin.ts` (the `{plugin,plugins}/*.{ts,js}` scan, the v1 default
  export), `tool/registry.ts` (plugin tool arguments as zod),
  `config/config.ts` (`@opencode-ai/plugin` installed into each `.opencode`
  directory).

Not yet run inside a live opencode process: the tests drive the hooks the way
opencode does, against iifa's scripted fake service. These APIs move (several
of the hooks are `experimental.`): the plugin follows the version above.

## Tests

From the repository root, with iifa's dev dependencies installed in the
interpreter `IIFA_TEST_PYTHON` names (default `python`, `python3` outside
Windows); the tests start `tests/fakes/iifa_service.py` with it:

```bash
node --test adapters/opencode/test/plugin.test.ts
```

Node 22.18 or later runs the TypeScript directly; no package is installed.
They cover: fail-open for every hook on a stopped service, hangs past the
deadline, 401/422/500, bodies that are not JSON or not the expected shape, a
bug below and a broken log; the deadline taken from the settings; the scope
check agreeing with iifa's message for message, default and derived scopes
valid, a bad scope turning memory off with one log line; the profile block once
per session; injected handles never resent; turns recorded on `session.idle`,
none for a subagent; compaction; session end; the four tools; hostile memory
kept as data; a log without memory content; part ids in opencode's format.
