# API reference

iifa has three ways in, all served by the same process:

- the **HTTP API**, under `/v1`, for harnesses and your own code;
- **MCP tools**, for the model itself (streamable HTTP at `/mcp`, or stdio);
- the **Python SDK**, a thin client of the HTTP API.

- [Conventions](#conventions)
- [Views](#views)
- [Scopes](#scopes)
- [Routes](#routes)
- [Dream, review and probes](#dream-review-and-probes)
- [MCP tools](#mcp-tools)
- [Python SDK](#python-sdk)

## Conventions

```bash
H="Authorization: Bearer $(cat ~/.config/iifa/token)"
J='Content-Type: application/json'
curl -s -H "$H" -H "$J" localhost:8765/v1/recall -d '{
  "scope": "user:alice", "query": "who works on the solar project?", "budget_tokens": 300
}'
```

- JSON in, JSON out; every call carries a bearer token.
- Every call that returns content takes `budget_tokens` (default 400) and
  never returns more. The answer's shape is shown in
  [How it works](how-it-works.md#recall-under-a-budget-handles-first).
- A recall also takes `exclude_ids`: ids already injected, not to send again.
- Errors are HTTP status codes with a JSON `detail`. A malformed request is a
  422 that names the field.

## Views

A read sees a **view**. Give exactly one of:

| Field | Sees |
|---|---|
| `scope` | that scope and every scope above it (`user:alice/mode:assistant` also sees `user:alice`) |
| `scopes` | an explicit list of scopes |
| `subtree` | a scope and everything below it (`admin` only) |

## Scopes

A scope is a path with `kind:name` at **every** level. A segment without its
kind is refused:

```bash
curl -s -H "$H" -H "$J" localhost:8765/v1/recall -d '{"scope": "user:alice/assistant", "query": "solar"}'
```

```json
{"detail": [{"type": "value_error", "loc": ["body", "scope"],
  "msg": "Value error, invalid scope 'user:alice/assistant': segment 'assistant' needs kind:name, like mode:assistant", ...}]}
```

Who sees what: [How it works](how-it-works.md#scopes-who-sees-what).

## Routes

All under `/v1`.

### Lifecycle

These are the moments a harness calls on its own (the adapters do it for you).

| Route | Permission | What |
|---|---|---|
| `POST /session/start` | read | the **profile block**: pinned facts, then the most central nodes, under budget. Stable while nothing changes |
| `POST /turns` | write | record a turn: the owner's words (`user`), the `assistant` answer, `tool_summaries` and `external` parts. One insert into the queue, answered in milliseconds; read in the background |
| `POST /precompact` | write (+ read for the block) | the messages a harness is about to drop go to the queue; what must survive comes back under budget |
| `POST /session/end` | write | the session's flush: summaries of what it touched, edge decay, in the background |

```bash
curl -s -H "$H" -H "$J" localhost:8765/v1/turns -d '{
  "scope": "user:alice", "session_id": "s1",
  "user": "I moved to Lyon last month.",
  "assistant": "Noted. How do you like it?",
  "external": [{"kind": "web", "text": "Lyon is the third-largest city of France."}]
}'
```

Send tool results, pages and documents as `external` parts: a fact about the
owner is then written only from the owner's own words. See
[Who said it](how-it-works.md#3-who-said-it).

### Reading

| Route | Permission | What |
|---|---|---|
| `POST /recall` | read | a question → handles under budget |
| `POST /expand` | read | ids → content at level 1 (facts) or 2 (+ the source excerpt), paged by cursor |
| `GET /nodes/{id}` | read | a node with its live facts, edges and pending proposals, under budget |
| `GET /lookup?external_id=` or `?name=` | read | records carrying your own id, or nodes with that name or alias |
| `POST /timeline` | read | fact changes (recorded, closed) of a node or of the whole view in a date range, newest first, under budget |
| `POST /at` | read | what was known about a node on a date: recorded by then, not closed by then |
| `POST /path` | read | how two nodes are related: the shortest path over visible edges, at most `max_hops` long, whole or not at all |
| `GET /archive` | read | archived records, under budget |
| `GET /proposals` | read | pending proposals in the view (optionally one `kind`), under budget |

### Writing

| Route | Permission | What |
|---|---|---|
| `POST /remember` | write | an explicit fact, optionally about a node (created on first mention) |
| `POST /link` | write | an explicit, confirmed edge between two visible nodes |
| `PATCH /facts/{id}` | write | attach a free-standing fact to a node, detach it, pin it |
| `POST /proposals/{id}` | write (+ **review** for a fact proposal) | `decision: accept` (an alias proposal becomes an alias, a fact proposal an explicit fact that keeps its source) or `reject` |
| `POST /archive`, `POST /restore` | write | hide a record everywhere, and bring it back unchanged |
| `POST /forget` | **forget** | delete for good, with counts of what went |

```bash
curl -s -H "$H" -H "$J" localhost:8765/v1/remember -d '{
  "scope": "user:alice",
  "statement": "Karim leads the installation planning of the solar project",
  "node": {"name": "Karim", "kind": "person"}
}'
```

### Operations

| Route | Permission | What |
|---|---|---|
| `GET /health` | any token | version, tier, configured backends, queue depth and oldest pending item, background mode |
| `GET /stats` | admin | counts only, never content |
| `GET` / `PUT /background` | admin | the background mode (saved) and the concurrency per model endpoint |
| `POST /background/run` | admin | process the queue now under a budget (`calls`, `tokens`, `seconds`, `max_jobs`); refused while paused; answers with what each job did, as counts |
| `POST /background/yield`, `POST /background/resume` | write | a foreground turn starts / ends |

## Dream, review and probes

What the dream is: [How it works](how-it-works.md#the-dream).

| Route | Permission | What |
|---|---|---|
| `POST /dream` | **dream** | start a dream over a scope (`operations`, `budget`, `dry_run`, a stricter `review_policy`). Queued at once (202), run in the background; 409 while one runs on the same scope, an ancestor or a descendant |
| `GET /dream/{id}`, `GET /dream/{id}/plan` | **dream** | status and report (counts and ids); every operation with what became of it and why, paged, under budget |
| `POST /dream/{id}/cancel`, `POST /dream/{id}/undo` | **dream** | stop cleanly; revert a run (all or nothing) or one operation (`op_id`), refused when a later write depends on it |
| `GET /review`, `POST /review/{id}` | **review** | the one review queue: alias, fact and merge proposals and queued dream operations; a merge is shown with the dream's latest verdict and reason. Accept or reject one (a merge proposal can only be rejected: only its reviewed `merge_nodes` merges) |
| `POST /probes`, `DELETE /probes/{id}` | **dream** | create or delete a probe set: questions and the ids of the records that answer them, in one scope |
| `GET /probes?scope=`, `GET /probes/{id}` | read | the sets of a scope; one set with its questions |
| `POST /probes/{id}/run` | read | run a set through recall at its budget: recall, tokens, recall per 1k tokens, tier; ids and numbers only |

```bash
curl -s -H "$H" -H "$J" localhost:8765/v1/dream -d '{"scope": "user:alice", "dry_run": true}'
curl -s -H "$H" localhost:8765/v1/dream/<run id>/plan
```

## MCP tools

**Over streamable HTTP**, at `/mcp`, behind the same tokens as the API:

```bash
claude mcp add --transport http iifa http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $(cat ~/.config/iifa/token)"
```

Pass `scope` per call, or set `[mcp] default_scope` (`IIFA_MCP_DEFAULT_SCOPE`).

**Over stdio**, for harnesses that start their MCP servers themselves. The
process talks to the running service:

```bash
IIFA_URL=http://127.0.0.1:8765 IIFA_TOKEN="$(cat ~/.config/iifa/token)" IIFA_SCOPE=user:alice/agent:coder \
  python -m iifa.mcp          # or: iifa mcp
```

It reads `IIFA_URL`, `IIFA_TOKEN` (required), `IIFA_SCOPE` (the default scope
of every call), `IIFA_DEFAULT_BUDGET_TOKENS`, `IIFA_MCP_EXPOSE_FORGET` and
`IIFA_TIMEOUT` (seconds toward the service). A service that is down at start
is not an error: tools answer with an error result until it comes back.

| Tool | What |
|---|---|
| `memory_recall` | handles under budget |
| `memory_expand` | the content behind chosen handles |
| `memory_about` | one subject, by name or alias |
| `memory_timeline` | what changed about a subject, or across the scope, in a date range |
| `memory_remember` | one explicit, durable fact |
| `memory_link` | an explicit link between two subjects |
| `memory_archive` | hide a record (reversible) |

- Every result is wrapped as recalled data and fits its `budget_tokens`,
  wrapper included.
- `memory_forget` is not offered unless the operator sets
  `IIFA_MCP_EXPOSE_FORGET=1`, and even then it needs a token with `forget`.
- There is no dream tool, on purpose: a model should not start a long, costly
  job in the middle of a conversation.

## Python SDK

```python
import os

from iifa.client import IifaClient, IifaUnavailable
from iifa.recalled import as_recalled_data

token = os.environ["IIFA_TOKEN"]
with IifaClient("http://127.0.0.1:8765", token, scope="user:alice/agent:demo") as memory:
    try:
        handles = memory.recall("who works on the solar project?", budget_tokens=300)
    except IifaUnavailable:
        handles = None  # fail open: the turn goes on without memory
    if handles and handles["items"]:
        prompt_block = as_recalled_data(handles, kind="recall")  # data, never instructions
    memory.remember("Karim leads the solar installation planning", node="Karim")
```

| Group | Methods |
|---|---|
| the five lifecycle moments | `session_start()` (the profile block), `recall()`, `turn(user, assistant)` (queued, answers in milliseconds), `precompact(messages)`, `session_end()` |
| time and relations | `timeline(node_id=..., since=..., until=...)`, `at(node_id, date)`, `path(src, dst)` |
| background | `background()`, `set_background(mode=...)`, `run_background(...)`, `yield_background()`, `resume_background()` |

`AsyncIifaClient` has the same methods, awaitable. A complete agent loop with
the five moments, in under 200 lines:
[examples/minimal_agent.py](../examples/minimal_agent.py).

**Fail open.** Catch `IifaUnavailable` and carry on: memory down must never
stop a turn. **Wrap what you inject** with `as_recalled_data`, so the model
reads memory as data, not as instructions.
