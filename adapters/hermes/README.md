# iifa memory for Hermes Agent

A Hermes Agent **memory provider** backed by a running iifa service. One
standard-library file: nothing to install in Hermes' Python.

- **once per session** it adds the profile block (pinned facts, the most
  central subjects) to the system prompt;
- **before each turn** it injects the handles that match the message;
- **after each turn** it records it, without making Hermes wait;
- **before a compression** it hands what must survive to Hermes' summary;
- **four tools**: `memory_recall`, `memory_expand`, `memory_remember`,
  `memory_archive`.

It **fails open**: if the service is down, slow or answering badly, nothing is
injected, nothing is raised into Hermes, and one line goes to the
`iifa.hermes` logger.

What each provider method does, and the Hermes version it follows:
[REFERENCE.md](REFERENCE.md).

## Install

You need a running iifa service (see the
[quickstart](../../README.md#quickstart)) and Hermes Agent v0.22.4 or later.

1. Copy the provider into your Hermes profile (`$HERMES_HOME` is `~/.hermes`
   unless you use profiles):

   ```bash
   cp -r adapters/hermes/iifa "$HERMES_HOME/plugins/iifa"
   ```

2. Give it the service and a token holding `read` and `write` only, in the
   environment Hermes runs in or in `$HERMES_HOME/.env`:

   ```bash
   IIFA_URL=http://127.0.0.1:8765
   IIFA_TOKEN=agent-secret
   IIFA_SCOPE=user:alice/mode:assistant
   ```

3. Select it in `config.yaml` (Hermes runs one memory provider at a time):

   ```yaml
   memory:
     provider: iifa
   ```

## Configure

| Variable | Default | Meaning |
|---|---|---|
| `IIFA_URL` | – | the service |
| `IIFA_TOKEN` | – | the bearer token |
| `IIFA_SCOPE` | `user:<Hermes user id>/agent:hermes` | whose memory: `kind:name` at every level |
| `IIFA_HOOK_BUDGET_TOKENS` | 400 | the most one injected block may cost, wrapper included |
| `IIFA_HOOK_DEADLINE_MS` | 1500 | how long a turn waits before going on without memory |
| `IIFA_TOOL_DEADLINE_MS` | 5000 | the same for a tool call |

Without `IIFA_SCOPE`, the user id is the gateway's `user_id` (spaces and `/`
replaced by `_`), or `local` without one. `hermes memory setup` asks for the
token (written to `.env`), the URL and the scope (written to
`$HERMES_HOME/iifa.json`); the environment wins over that file.

## Verify

1. Start Hermes and ask it to remember something explicit.
2. Start a new session and ask about it.
3. `hermes logs` shows the provider's `iifa …` lines.

## Troubleshoot

- **Memory is off from the start.** The scope is malformed: `kind:name` at
  every `/` level, so `user:alice/mode:assistant`, never
  `user:alice/assistant`. The provider checks it at start, turns memory off,
  and logs which segment is wrong.
- **Nothing is injected.** Check `memory.provider: iifa` and that the token
  is in the environment Hermes runs in; then read `hermes logs`.
- **Subagents and cron runs write nothing.** On purpose: only the primary
  agent records turns.
- **The other memory tools.** `memory_about`, `memory_timeline` and
  `memory_link` are on iifa's MCP server, which Hermes can load too
  (`python -m iifa.mcp`).
