# iifa memory for Claude Code: reference

What each hook does, how the plugin was built, and how it is tested. To
install it, see [README.md](README.md).

## What each moment does

| Claude Code event | What the plugin does |
|---|---|
| `SessionStart` (`startup`, `resume`, `clear`, `compact`) | `POST /v1/session/start` with the configured scope, the session id and `budget_tokens` minus a reserve; injects a `recalled-data` block of kind `profile`: a short status (service, tier, scope) plus the pinned facts and the most central subjects, under budget. The block is deterministic and stable while memory does not change. On `clear` the session's state is reset. |
| `UserPromptSubmit` | `POST /v1/recall` with the prompt as query, the configured scope, `budget_tokens` minus the wrapper's reserve, and the ids already injected this session as `exclude_ids`. Handles come back as a `recalled-data` block of kind `recall`. Prompts under three characters and slash commands are not recalled. The prompt is kept in the session's state file for `Stop`. |
| `Stop` | `POST /v1/turns` with the prompt kept at `UserPromptSubmit` and `last_assistant_message` from the hook input: one enqueue, answered in milliseconds. Prints **nothing**: any output on `Stop` would continue the conversation. A slash command's answer is recorded without a prompt; when there is neither, nothing is sent. |
| `PreCompact` (`manual` and `auto`) | Reads the last messages of `transcript_path` (bounded: 20 messages, 32 k characters; tool calls and results dropped) and posts them with `trigger` to `POST /v1/precompact`. The turns were already recorded one by one at `Stop`; this is the safety net for anything missed. What must survive comes back under budget and is injected as a `recalled-data` block of kind `profile` when it is not empty. |
| `SessionEnd` (`clear`, `resume`, `logout`, `prompt_input_exit`, `other`) | `POST /v1/session/end` with the reason: the service flushes and extracts in the background. On `clear` the session's state file is removed. Prints nothing. |

Everything injected is `additionalContext` wrapped as recalled *data*, under
the budget you set. Recording is one enqueue on the service side: nothing is
extracted while Claude waits.

The hooks run in **exec form** (`command` + `args`), so paths with spaces need
no quoting and no shell is involved on any platform. Inside Claude Code the
tools are named `mcp__plugin_iifa-memory_iifa__memory_recall` and so on.

The per-session state file (under `CLAUDE_PLUGIN_DATA/sessions/`, or a
per-user temporary directory without it) holds the ids injected this session
and the prompt of the turn in progress, because Claude Code hands `Stop` the
answer but not the prompt. It is the harness's own data directory, next to
the transcripts Claude Code keeps there itself.

The MCP server (`python -m iifa.mcp`, over stdio) offers `memory_recall`,
`memory_expand`, `memory_about`, `memory_remember`, `memory_link` and
`memory_archive`. No forget tool, no dream tool.

## Why environment variables and not plugin options

Claude Code's `userConfig` options are substituted into hook commands and MCP
configs as `${user_config.KEY}`, but a plugin loaded with `--plugin-dir` is
never asked for them and the declared defaults are not applied (observed with
2.1.283: the `SessionStart` hook failed with *Plugin option "python" isn't
set* and the MCP server did not start). Environment variables work in every
loading mode.

## Validation

Validated end to end with Claude Code 2.1.283 and a tier-0 service: in one
session Claude stored a fact through `memory_remember`; in a new session the
`UserPromptSubmit` hook injected the matching handle and Claude answered from
it, naming the hook as its source. The three hooks added later (`Stop`,
`PreCompact`, `SessionEnd`) were validated against the documented hook inputs
with the fake and the real service in the test suite; the manifest still
passes `claude plugin validate` with 2.1.283.

## Built against

The official Claude Code documentation at `code.claude.com/docs` for Claude
Code **2.1.283** (`claude --version`); the plugin manifest passes `claude plugin validate`
with that version. Pages and facts relied on:

- *Hooks reference* (`/docs/en/hooks`): `SessionStart` input `source` (`startup`, `resume`,
  `clear`, `compact`, `fork`) and its `hookSpecificOutput.additionalContext`;
  `UserPromptSubmit` input `prompt`, its 30 s default timeout, and the rule
  that a `command` hook reaching its timeout is cancelled **and its output
  discarded**, which is why the hook enforces its own shorter deadline;
  `Stop`: fires when Claude finishes responding, no matcher, input
  `last_assistant_message` (the documented way to get the final answer, the
  transcript file "may lag the in-memory conversation") and
  `stop_hook_active`, exit code 2 or a `decision: "block"` **continues the
  conversation**, so the plugin's `Stop` hook prints nothing at all;
  `PreCompact`: matchers `manual` and `auto`, input `trigger` and
  `custom_instructions`, it may block compaction (exit 2) or add
  `additionalContext`, and the messages about to be dropped are not in its
  input (hence the transcript tail); `SessionEnd`: input `reason` (`clear`,
  `resume`, `logout`, `prompt_input_exit`, `other`), cannot block, and its
  hooks share a **1.5 s budget** that a longer per-hook `timeout` raises up to
  60 s, which is why `hooks.json` sets `timeout: 10` on it while the hook keeps
  its own 1.5 s deadline; default `command` hook timeout 600 s; plain stdout is
  context only on `UserPromptSubmit`, `UserPromptExpansion`, `SessionStart`
  and `PostModelSwitch`, elsewhere it is debug output; exec form vs shell form;
  `${CLAUDE_PLUGIN_ROOT}`, `${CLAUDE_PLUGIN_DATA}`, `CLAUDE_PLUGIN_OPTION_<KEY>`.
- *Connect Claude Code to tools via MCP* (`/docs/en/mcp`): stdio servers,
  plugin-provided servers in `.mcp.json`, tool names of plugin servers
  (`mcp__plugin_iifa-memory_iifa__memory_recall`), `${VAR}` expansion,
  `claude mcp add --transport http`.
- *Plugin manifest reference* (`/docs/en/plugins/manifest-reference`):
  `.claude-plugin/plugin.json`, `hooks/hooks.json`, `.mcp.json`, the standard
  layout, and `userConfig` / `${user_config.KEY}`, which this plugin does not
  use for the reason given above.

These APIs move: the plugin follows the version above.

## Tests

`python -m pytest tests/adapters` runs the hook script as Claude Code runs it,
for all five events, against a scripted fake service (down, hanging past the
deadline, HTTP errors, malformed answers) and against the real service on a
loopback port: fail-open for every event, the budget for the profile and
recall blocks, the recalled-data wrapper, a log without memory content, the
prompt handed from `UserPromptSubmit` to `Stop`, the bounded transcript tail,
and the three jobs (`turn`, `precompact`, `session_end`) landing in the
service's persisted queue.
