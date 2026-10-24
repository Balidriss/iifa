# iifa memory for opencode

An opencode plugin backed by a running iifa service. TypeScript, no
dependency of its own.

- **once per session** it adds the profile block (pinned facts, the most
  central subjects) to the system prompt;
- **before each prompt** it adds the handles that match it, hidden in the
  interface but sent to the model;
- **after each answer** it records the turn;
- **before a compaction** it adds what must survive to the compaction prompt;
- **four tools**: `memory_recall`, `memory_expand`, `memory_remember`,
  `memory_archive`.

It **fails open**: opencode waits on its plugins with no timeout, so the
plugin bounds every call itself and never throws. If the service is down,
slow or answering badly, nothing is injected and one line goes to opencode's
log.

What each hook does, and the opencode version it follows:
[REFERENCE.md](REFERENCE.md).

## Install

You need a running iifa service (see the
[quickstart](../../README.md#quickstart)) and opencode 1.19.4 or later.

1. Copy the entry file **and** the `iifa/` directory next to it into your
   project's `.opencode/plugins/` (or `~/.config/opencode/plugins/` for every
   project):

   ```bash
   mkdir -p .opencode/plugins
   cp -r adapters/opencode/plugin/iifa.ts adapters/opencode/plugin/iifa .opencode/plugins/
   ```

2. Give it the service and a token holding `read` and `write` only, in the
   environment opencode runs in:

   ```bash
   export IIFA_URL=http://127.0.0.1:8765
   export IIFA_TOKEN=agent-secret
   export IIFA_SCOPE=user:alice/mode:assistant
   ```

## Configure

| Variable | Option | Default | Meaning |
|---|---|---|---|
| `IIFA_URL` | `url` | – | the service |
| `IIFA_TOKEN` | – | – | the bearer token (environment only) |
| `IIFA_SCOPE` | `scope` | `user:local/agent:opencode` | whose memory: `kind:name` at every level |
| `IIFA_HOOK_BUDGET_TOKENS` | `budgetTokens` | 400 | the most one injected block may cost, wrapper included |
| `IIFA_HOOK_DEADLINE_MS` | `deadlineMs` | 1500 | how long a prompt waits before going on without memory |
| `IIFA_TOOL_DEADLINE_MS` | `toolDeadlineMs` | 5000 | the same for a tool call |
| – | `projectScope` | false | one memory per project, below yours |

Options go in `opencode.json`, loading the plugin by path; the environment
wins over them. Use this **or** the copy in `.opencode/plugins/`, not both: a
plugin in that directory loads on its own.

```json
{
  "plugin": [
    ["./.opencode/plugins/iifa.ts", { "scope": "user:alice/mode:coder", "projectScope": true }]
  ]
}
```

`projectScope: true` appends `/project:<folder name>`, so
`user:alice/mode:coder/project:solar` sees its own facts and those of
`user:alice/mode:coder` and `user:alice`. It keys on the folder's **name**
only: `~/work/solar` and `~/old/solar` share one memory. Rename one, or set
`scope` per project.

The other memory tools (`memory_about`, `memory_timeline`, `memory_link`) are
on iifa's MCP server:

```json
{
  "mcp": {
    "iifa": {
      "type": "local",
      "command": ["python", "-m", "iifa.mcp"],
      "environment": { "IIFA_URL": "http://127.0.0.1:8765", "IIFA_TOKEN": "agent-secret", "IIFA_SCOPE": "user:alice/mode:assistant" }
    }
  }
}
```

## Verify

1. Start opencode and ask it to remember something explicit.
2. Start a new session and ask about it.
3. opencode's log shows lines from the service `iifa`.

## Troubleshoot

- **Memory is off from the start.** The scope is malformed: `kind:name` at
  every `/` level, so `user:alice/mode:assistant`, never
  `user:alice/assistant`. The plugin checks it when it loads, turns memory
  off, and logs which segment is wrong.
- **The plugin does not load.** opencode loads the files of
  `.opencode/plugins/`, not its subdirectories: `iifa.ts` must sit there,
  with `iifa/` beside it.
- **A tool answers "iifa memory unavailable".** The service is down or slower
  than `IIFA_TOOL_DEADLINE_MS`; the session goes on without memory.
- **Subagents record nothing.** On purpose: only the main session does.
