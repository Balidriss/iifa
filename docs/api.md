# API reference

iifa has three ways in, all served by the same process:

- the **HTTP API**, under `/v1`, for harnesses and your own code;
- **MCP tools**, for the model itself (streamable HTTP at `/mcp`, or stdio);
- the **Python SDK**, a thin client of the HTTP API.

- [Conventions](#conventions)
- [Lists and cursors](#lists-and-cursors)
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

## Lists and cursors

The routes that return a list — `GET /nodes`, `POST /timeline`,
`GET /proposals`, `GET /archive`, `GET /review`, `GET /dreams`, `GET /dream/{id}/plan` —
answer a page at a time. They take:

| Field | What |
|---|---|
| `limit` | at most this many items (default 100, at most 1000; the timeline's default is 200) |
| `budget_tokens` | at most this many tokens, as everywhere |
| `cursor` | the `next_cursor` of the previous page; none for the first page |

A page stops at whichever of `limit` and `budget_tokens` it reaches first. An
item that does not fit the budget is not skipped: it opens the next page, and
`omitted` says how many items of the `limit` window were left for it.
`next_cursor` picks up exactly after the last item returned, and is `null` once
the list is done. A page that returns nothing because the budget is too small
for its first item says `"reason": "budget_too_small"` with `next_cursor: null`:
ask again with the same cursor and a larger budget.

```bash
page=$(curl -s -H "$H" "localhost:8765/v1/nodes?scope=user:alice&limit=50&budget_tokens=8000")
next=$(echo "$page" | jq -r .next_cursor)
curl -s -H "$H" "localhost:8765/v1/nodes?scope=user:alice&limit=50&budget_tokens=8000&cursor=$next"
```

A walk never shows an item twice nor misses one because of a write between two
pages: new items join the end of an oldest-first list and the head of a
newest-first one (start a new walk to see them). The cursor is opaque and
signed: it only works with the route, view, sort and filters that produced it.
Anything else — a changed scope, sort or `prefix`, another route, an edited
cursor — is a 400.

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
| `POST /session/start` | read | the **profile block**: pinned facts, then the agent's own commitments and preferences when a self is declared (commitments first, the newest first; at most a third of the budget ahead of the nodes, the others after them), then the most central nodes, under budget. Stable while nothing changes |
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
[Who said it](how-it-works.md#3-who-said-it). When a scope has a self (see
[Configuration](configuration.md#owners-and-languages)), mark with
`"vouched": true` the parts the owner gave or asked for: the agent learns its
rules of method from them at once, instead of leaving them for review.

### Reading

| Route | Permission | What |
|---|---|---|
| `POST /recall` | read | a question → handles under budget |
| `POST /expand` | read | ids → content at level 1 (facts) or 2 (+ the source excerpt), paged by cursor |
| `GET /nodes` | read | the subjects memory knows about, a page at a time (see below) |
| `GET /nodes/{id}` | read | a node with its live facts, edges and pending proposals, under budget; each edge carries the `rev` that `POST /edges/{id}` takes |
| `GET /lookup?external_id=` or `?name=` | read | records carrying your own id, or nodes with that name or alias |
| `POST /timeline` | read | fact changes (recorded, closed) of a node or of the whole view in a date range, newest first, paged |
| `POST /at` | read | what was known about a node on a date: recorded by then, not closed by then |
| `POST /path` | read | how two nodes are related: the shortest path over visible edges, at most `max_hops` long, whole or not at all |
| `GET /archive` | read | archived records, most recently archived first, paged, each with what it said: a node's `name` and `node_kind`, a fact's `statement` and `node_name`, an edge's `type`, `status`, `src_name` and `dst_name` (null for a node your view does not see) |
| `GET /proposals` | read | pending proposals in the view (optionally one `kind`), oldest first, paged |

`GET /nodes` lists the nodes of a view, for a person to browse:

| Parameter | What |
|---|---|
| `sort` | `name` (default: case and Latin accents folded, then id) or `changed` (most recent change first) |
| `prefix` | only the nodes whose name, or one of whose aliases, starts with it — ignoring case and Latin accents, never in the middle of a word: `kar` finds Karim and Karima, `bo` finds Robert when Bob is his alias |
| `include_archived` | `true` also lists archived and merged nodes (`archived_at`, `merged_into` say which); off by default |

```json
{"items": [{"id": "node_01J…", "kind": "person", "name": "Karim", "aliases": ["K."],
            "live_facts": 4, "pending_proposals": 1,
            "changed_at": "2026-03-14T08:00:00.000000+00:00",
            "archived_at": null, "merged_into": null}],
 "sort": "name", "next_cursor": "eyJ2Ijox…", "used_tokens": 52, "budget_tokens": 8000,
 "omitted": 0, "reason": null}
```

`live_facts` and `pending_proposals` count what the view sees; `changed_at` is
when memory last recorded a fact about the node in the view's scopes (archived
and closed facts count), or created it. A walk lists
the nodes that existed at its first page, with `changed_at` as of then, so the
order holds while you page; a new walk shows what came since.

### Writing

| Route | Permission | What |
|---|---|---|
| `POST /remember` | write | an explicit fact, optionally about a node (created on first mention); with `about_self: true`, a fact about the agent itself when a self is declared (said again, it writes nothing and answers `echo: true`) |
| `POST /link` | write | an explicit, confirmed edge between two visible nodes (201). When the pair already has that edge (same type and direction), it is confirmed and returned instead of a second one (200, `created: false`) |
| `POST /edges/{id}` | write | decide a proposed edge: `decision: confirm` (it stops decaying) or `reject` (it is archived); optional `rev`, the edge's `rev` as the node view or `/link` gave it (409 `stale` with the current one). An edge's `rev` moves only when its status, life, ends or type change: a recall reinforcing it or a session's decay does not make your decision stale. 409 `not_proposed` for a confirmed edge, `not_live` for a closed or archived one. Answers with the edge and its `correction_id` |
| `PATCH /facts/{id}` | write | attach a free-standing fact to a node, detach it, pin it, or replace its scope set (`scopes`: only scopes the token owns may be added or removed, and the fact's node must stay visible — else 409: detach it in the same call) |
| `POST /proposals/{id}` | write (+ **review** for a fact proposal) | `decision: accept` (an alias proposal becomes an alias, a fact proposal an explicit fact that keeps its source) or `reject`. A merge proposal is accepted in review only (409 here) |
| `PATCH /nodes/{id}` | write | correct a node: `name` (the old name stays an alias unless `keep_old_name: false`), `kind`, `add_aliases`, `remove_aliases`; `rev` = the revision you read (409 `stale` when it moved). Answers with the node and its `correction_id` |
| `POST /nodes/merge` | **review** | merge two nodes you know are one: `keep`, `absorb`, optional `revs` (node id → rev). Same kind, same scopes, both live — else 409 with the `reason`. The absorbed node's facts, aliases and edges move to the kept one; it is archived with `merged_into` |
| `GET /corrections/{id}` | read | one correction: what was asked, by which token, when, whether it was undone |
| `POST /corrections/{id}/undo` | write (**review** for a merge) | revert a correction exactly (an edge decision: the edge is proposed again, or live again); 409 `depends` naming the records when a later write depends on it (undo the later one first) |
| `POST /archive`, `POST /restore` | write | hide a record everywhere, and bring it back unchanged |
| `POST /forget` | **forget** | delete for good, with counts of what went; also takes, as `id`, the `job_id` (`job_...`) of a turn or a compaction that has not run yet (the job and its text are deleted; `started: true` when it ran before, failed or interrupted, and may have written; 409 once it runs or has run: forget its source) |

```bash
curl -s -H "$H" -H "$J" localhost:8765/v1/remember -d '{
  "scope": "user:alice",
  "statement": "Karim leads the installation planning of the solar project",
  "node": {"name": "Karim", "kind": "person"}
}'
```

A person corrects what extraction got wrong. Every correction is journalled
with the name of the token that made it, and can be undone:

```bash
curl -s -X PATCH -H "$H" -H "$J" localhost:8765/v1/nodes/<node id> -d '{
  "name": "Atelier Lumen", "keep_old_name": false,
  "remove_aliases": ["rendez-vous du **mardi 3 mars"]
}'
# {"id": "node_…", "name": "Atelier Lumen", "aliases": [], "rev": 7, "correction_id": "corr_…", ...}
curl -s -X POST -H "$H" localhost:8765/v1/corrections/<correction id>/undo
```

### Operations

| Route | Permission | What |
|---|---|---|
| `GET /health` | any token | version, tier, configured backends, queue depth and oldest pending item, background mode, and `background.run`: the run in progress, or `null` (see below) |
| `GET /stats` | admin | counts only, never content |
| `GET` / `PUT /background` | admin | the background mode (saved) and the concurrency per model endpoint |
| `POST /background/run` | admin | process the queue now under a budget (`calls`, `tokens`, `seconds`, `max_jobs`); answers with what each job did, as counts. With `"wait": false`: 202 and the run id at once. 409 while paused, and while another run is in progress (the message names it) |
| `GET /background/run/{id}` | admin | a run's report, live while it runs; the current run and the last 20 are kept in memory (a restart forgets them: 404) |
| `POST /background/run/{id}/cancel` | admin | stop the run in progress early, as at a budget limit: no new job starts, a job in flight goes back to the queue untouched at its next model call (one that makes no call finishes; a call already waiting at a yielded gate is still made), the run ends with `stopped_by: "cancelled"`; the mode is unchanged. 404 for an unknown run, 409 for a finished one |
| `POST /background/yield`, `POST /background/resume` | write | a foreground turn starts / ends |

One run at a time. A run started by one client is visible to all of them:
`/health` (any token) carries it as ids and counts, never content.

```json
"background": {
  "mode": "on_demand", "yielded": false, "concurrency": {"chat": 1, "embeddings": 1},
  "worker": "running",
  "run": {
    "run_id": "run_01J…", "started_by": "host-bridge",
    "started_at": "2026-03-14T08:00:00.000000+00:00",
    "budget": {"calls": 40, "tokens": null, "seconds": 900, "max_jobs": null},
    "jobs_done": 7, "jobs_left": 12, "yielded": false, "cancel_requested": false
  }
}
```

`started_by` is the name of the token that started the run, so a client tells
its own runs from others'. `jobs_done` counts jobs taken to an end (done or
failed); `jobs_left` is the pending jobs, capped by `max_jobs` — an estimate,
since turns keep arriving and a budget may stop the run first. `yielded` says
the background is yielded: the run starts no new model call until the
foreground turn ends (`/background/resume` or the yield's timeout).
`cancel_requested` says someone asked `/background/run/{id}/cancel`: the run
is stopping, and `jobs_left` is 0.

Long runs need not hold a request open:

```sh
curl -s -H "$H" -H "$J" localhost:8765/v1/background/run -d '{"wait": false, "budget": {"seconds": 900}}'
# 202 {"run_id": "run_01J…", "status": "running"}
curl -s -H "$H" localhost:8765/v1/background/run/run_01J…
# {"status": "running", "jobs_done": 3, ...}, later "status": "done"
```

The report has the synchronous answer's fields (`jobs`, `processed`, `calls`,
`tokens`, `seconds`, `stopped_by`, `queue`) plus `status`, `started_by`,
`started_at`, `finished_at`, `budget`, `jobs_done`, `current_job`,
`cancel_requested`, `cancelled_by` (the token that cancelled it), and, while
it runs, `jobs_left` and `yielded` (`null` once done). `stopped_by` is `null`
while running, then `queue_empty`, `calls`, `tokens`, `seconds`, `max_jobs`,
`paused`, `shutdown`, `cancelled`, or `error` (an internal failure; the run
still ends).
`finished_at` is `null` when the run ended without the store (a shutdown).

## Dream, review and probes

What the dream is: [How it works](how-it-works.md#the-dream).

| Route | Permission | What |
|---|---|---|
| `POST /dream` | **dream** | start a dream over a scope (`operations`, `budget`, `dry_run`, a stricter `review_policy`). Queued at once (202), run in the background; 409 while one runs on the same scope, an ancestor or a descendant |
| `GET /dreams?scope=` | **dream** | the dreams on a scope and below it, newest first, paged, whichever client started them: `status` (optional filter), `dry_run`, `started_by` (the token's name) and their times |
| `GET /dream/{id}`, `GET /dream/{id}/plan` | **dream** | status and report (counts and ids); every operation with what became of it and why, paged (`cursor`, or `after` with the plain `seq` that `next` gives) |
| `POST /dream/{id}/cancel`, `POST /dream/{id}/undo` | **dream** | stop cleanly; revert a run (all or nothing) or one operation (`op_id`), refused when a later write depends on it |
| `GET /review`, `POST /review/{id}` | **review** | the one review queue, oldest first, paged: alias, fact and merge proposals and queued dream operations; a merge is shown with the dream's latest verdict and reason. Accept or reject one; accepting a merge proposal merges the two nodes as your decision (optional `keep`: the node to keep; default, the one with more live facts, else the older) and answers with its `correction_id` |
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
| browsing | `nodes(prefix=..., sort=..., cursor=...)`, `node(node_id)`, `lookup(name=...)`, `archived(cursor=...)` — pass a page's `next_cursor` back as `cursor` |
| time and relations | `timeline(node_id=..., since=..., until=..., cursor=...)`, `at(node_id, date)`, `path(src, dst)` |
| corrections | `patch_node(node_id, name=..., kind=..., add_aliases=..., remove_aliases=..., rev=...)`, `merge_nodes(keep, absorb)` (review), `decide_edge(edge_id, "confirm" \| "reject")`, `correction(id)`, `undo_correction(id)` |
| background | `background()`, `set_background(mode=...)`, `run_background(..., wait=False)`, `background_run(run_id)`, `yield_background()`, `resume_background()` |

`AsyncIifaClient` has the same methods, awaitable. A complete agent loop with
the five moments, in under 200 lines:
[examples/minimal_agent.py](../examples/minimal_agent.py).

**Fail open.** Catch `IifaUnavailable` and carry on: memory down must never
stop a turn. **Wrap what you inject** with `as_recalled_data`, so the model
reads memory as data, not as instructions.
