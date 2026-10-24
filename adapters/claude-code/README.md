# iifa memory for Claude Code

A Claude Code plugin that gives every session long-term memory, under a token
budget:

- **at session start** it injects the profile block (pinned facts, the most
  central subjects);
- **before each prompt** it injects the handles that match it;
- **after each answer** it records the turn, without making Claude wait;
- **before a compaction** it hands over what is about to be dropped, and
  injects what must survive;
- **the MCP tools** (`memory_recall`, `memory_remember`, …) let Claude ask
  memory itself.

It **fails open**: if the service is down, slow or answering badly, the turn
goes on without memory, and the plugin writes one line to its log. It never
delays Claude Code beyond its own deadline (1.5 s by default).

What each hook does in detail, and the Claude Code version it follows:
[REFERENCE.md](REFERENCE.md).

## Install

You need a running iifa service (see the
[quickstart](../../README.md#quickstart)), Python 3.11+ with `iifa` installed
(`pip install .` from a checkout), and Claude Code 2.1.283 or later.

1. Export three variables in the shell that starts Claude Code. Give it a
   token holding `read` and `write` only:

   ```bash
   export IIFA_URL=http://127.0.0.1:8765
   export IIFA_TOKEN=agent-secret          # a token the service is configured with
   export IIFA_SCOPE=user:alice/agent:claude-code
   ```

2. Activate the virtual environment where iifa is installed, so `python`
   resolves to it. The hooks and the MCP server run `python` directly, with
   no shell; on macOS and Linux without a virtual environment, `python` may
   not exist even when `python3` does.

3. Load the plugin:

   ```bash
   claude --plugin-dir /path/to/iifa/adapters/claude-code
   ```

   Or add the repository as a marketplace and install `iifa-memory` from it
   once it is published (`/plugin` → Marketplaces → Add).

## Configure

| Variable | Default | Meaning |
|---|---|---|
| `IIFA_URL` | – | the service |
| `IIFA_TOKEN` | – | the bearer token; without it the hooks stay silent |
| `IIFA_SCOPE` | – | whose memory: `kind:name` at every level |
| `IIFA_HOOK_BUDGET_TOKENS` | 400 | the most one injected block may cost, wrapper included |
| `IIFA_HOOK_DEADLINE_MS` | 1500 | how long a hook waits before the turn goes on without memory |
| `IIFA_HOOK_LOG` | – | a file for the hook's log |

To use the service's MCP endpoint over HTTP instead of the stdio server:

```bash
claude mcp add --transport http iifa http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $IIFA_TOKEN"
```

Pass `scope` per call, or set `[mcp] default_scope` on the service.

## Verify

1. Start a session and run `/mcp`: `iifa` should be connected.
2. Ask Claude to remember something explicit.
3. Start a new session and ask about it: Claude answers from memory.

## Troubleshoot

- **Where the log is.** One line per hook run: on stderr (visible with
  `claude --debug`), in `IIFA_HOOK_LOG` when set, and in `iifa-hook.log`
  under the plugin's data directory (`CLAUDE_PLUGIN_DATA`). It holds event
  names, ids, counts and durations, never memory content.
- **Nothing is injected.** Check `IIFA_TOKEN` is exported in the shell that
  started `claude`; without it the hooks stay silent.
- **The MCP server does not start.** It exits with a message naming the
  missing variable, or `python` is not the interpreter where iifa is
  installed.
- **A 422 on every call.** The scope is malformed: `user:alice/agent:claude-code`,
  never `user:alice/claude-code`.
- **Why environment variables, not plugin options.** Plugin options are not
  applied to a plugin loaded with `--plugin-dir`; see
  [REFERENCE.md](REFERENCE.md#why-environment-variables-and-not-plugin-options).
