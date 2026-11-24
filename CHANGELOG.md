# Changelog

What changed in each release of iifa, for people who use it. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/), with Python's spelling for
pre-releases (0.1.0a1).

## Unreleased

## 0.1.0a4 — 2026-11-24

### Added
- `detect_secrets_by_wording` (`[extraction]`, off by default): the service
  also reads secrets from their wording, at the operator's risk.
- `POST /v1/forget` takes as `id` the `job_id` of a turn or a compaction that has
  not run yet: the job is deleted with its text, so a turn recorded by
  mistake never becomes memory (409 once the job runs or has run).

### Changed
- A rule the agent holds as `preference.<topic>` and states again as
  `commitment.<topic>` (or the reverse) is one rule: the same value writes
  nothing, another value is kept beside the first with a visible `conflict`,
  in a turn, in `remember` with `about_self` and when accepted from review.
- The profile block lists the agent's rules commitments first, the newest
  first; ahead of the central nodes they take at most a third of the budget,
  and the others follow the nodes, so many rules no longer crowd the nodes out.
- A password, a one-time or recovery code, an API key or a token said in a
  turn stays out of memory: memory keeps the event without the value, and the
  extraction report counts `secrets.found` and `secrets.dropped`.

## 0.1.0a3 — 2026-11-18

### Added
- `GET /v1/review` names the nodes an item is about (`nodes`: name, kind,
  aliases and the first three facts of each node the view sees), so a merge
  question reads as two subjects instead of two ids.
- `GET /v1/review` names the nodes of every queued operation, not only a
  merge's, and states the facts an operation names (`facts`: id and
  statement), so a grouping or a consolidation reads without looking up an id.
- `review_self_rules` under `[extraction]` (`IIFA_REVIEW_SELF_RULES`): what a
  declared self commits to or prefers waits in the review queue (`why:
  self:rule`) instead of being written by a turn. A rule said again is asked
  about once, a rejected one is not asked about again, and an accepted
  commitment replaces the one it changes. Off by default. A rule said again
  in other words is recognised by the chat model, which compares it with
  the rules already held, waiting or refused (one more call per such turn).
- `GET /v1/review` states what accepting a fact proposal would replace
  (`replaces`: id and statement of each live fact the accept closes).
- `[extraction.selves]` (`IIFA_SELVES`) makes a scope's memory also the
  agent's own: its lessons, views and commitments are kept on its node, its
  rules of method on the task they serve; what a page read in the same turn
  would teach it waits in the review queue instead, unless the host marks
  every external part of the turn `vouched` (given or asked for by the owner).
- With a self declared, each turn names the activity the owner asks for, a
  session keeps it until the owner asks for another, and the rules of method
  learnt meanwhile are linked to it (`activity` in the turn's report); the
  session's closing episode is linked to every activity of the session, a
  `uses` edge joins an activity to the tool or guide used, and `expand`
  shows a fact's `activities`.
- Recall: a request for an activity the agent did before brings that
  activity's rules first, then the rules of what it used and its last
  episode (`why`: `rule of the activity: …`), behind the subject the question
  names and within two thirds of the budget; with `embeddings.activity_threshold`
  set, a request in other words can match the activity too.
- The dream can group the agent's activities done on different objects
  under one generic activity (`group_activities`, reviewed by default); a
  request for it then brings every occurrence's rules and its timeline,
  the last time first.
- With a self declared, a session's closing episode is the agent's own, the
  profile block carries its commitments and its preferences, and `remember` (`about_self`, also
  in `memory_remember`, the Python clients and the Hermes, opencode and
  OpenAI Agents adapters) writes about the agent itself.
- `POST /v1/background/run/{id}/cancel` (admin) stops a background run early,
  as at a budget limit (`stopped_by: "cancelled"`); `/health` and the run's
  report show `cancel_requested`, the report `cancelled_by` (also
  `cancel_background_run(id)` in the Python client).
- `GET /v1/nodes` lists the subjects memory knows about, by name or by last
  change, with a name or alias prefix filter, their live facts and pending
  proposals (also `nodes()` in the Python client).
- Every list (`/nodes`, `/timeline`, `/proposals`, `/archive`, `/review`, a
  dream's plan) pages with `cursor` and `limit` and answers with
  `next_cursor`; a cursor only works with the scope and filters that made it.
- `PATCH /v1/nodes/{id}` renames a node, adds or removes aliases and changes its kind.
- `POST /v1/nodes/merge` merges two nodes you know are one (`review` permission).
- Every correction names the token that made it and can be undone (`POST /v1/corrections/{id}/undo`).
- Python client: `patch_node`, `merge_nodes`, `correction` and `undo_correction`.
- `POST /v1/edges/{id}` confirms or rejects a proposed edge, journalled and
  undoable like a node's correction (also `decide_edge()` in the Python client).
- Each edge of `GET /v1/nodes/{id}` and of `/link`'s answer carries `rev`, the
  revision `POST /v1/edges/{id}` checks: it moves when the edge's status, life,
  ends or type change, never when a recall reinforces it or a session decays it.
- `GET /v1/dreams?scope=` (dream) lists the dreams of a scope, newest first,
  with who started them, whichever client did.
- `[extraction] value_norm` (off by default): the model gives each value a
  comparison form, never shown, so the same value told in another language
  is linked without a second model call and without a conflict.
- `POST /v1/turns` takes `"record": false`: the turn is not written at all.
- `bench/run.py --intent`, `--worksession` and `--neighbours`: corpora that
  measure what unmeant turns, work sessions and two subjects side by side
  still write (and `--no-read-intent`, `--value-norm`).
- Every release appears on the GitHub releases page with its notes from
  this changelog and the wheel, ready for `pip install`.
- A recalled fact says where it came from when it is not the owner's own
  words (`from`: `assistant`, or `web example.com 2026-03-14` for a page
  read that day), on every read route; a node known only from what was read
  is flagged `external_only`.
- `[extraction.assistants]` (or `IIFA_ASSISTANTS`) names the assistant of a
  scope, `Assistant` by default: its answers are labelled with that name and
  the facts taken from them say it in `from`.
- What was observed (a stock level, a service's status, anything read
  from a tool) is kept as true at a date: flagged `observed`, its age counted
  from the day it was seen, ranked lower as it ages
  (`[recall] observed_half_life_days`, 7), and replaced by a newer
  observation of the same thing.
- A work session whose progress was read only as observations still
  leaves its episode when it closes.
- An observation with a newer one waiting in review is flagged
  `newer_pending` in recall and `expand`, so it is not read as today's state.
- An observation of a thing seen again since under another key is flagged
  `newer_observed` in recall and `expand`.
- An open conflict shows on both facts: the one a newer fact contradicts is
  flagged `conflict` too when it is read.
- A session's episode that names a known subject the session never named is
  only proposed for review.
- A language override on `"*"` now holds for every scope without a nearer
  one; an owner declared on `"*"` is refused when the configuration loads.
- A name written as it is inside a statement in another script (a file name
  in a Russian sentence) names its subject: the fact is kept, not dropped.
- `bench/run.py --self`: a corpus where the memory is also the agent's own,
  measuring what it keeps, what pages and roles fail to plant, and whether its
  lessons and rules of method come back.
- `bench/run.py --observed`: a corpus of things whose state changes between
  sessions.

- Docs: how to start over with an empty memory while keeping the key and
  the configuration ([Configuration](docs/configuration.md#starting-over-with-an-empty-memory),
  [Docker](docs/docker.md#starting-over)).

### Changed
- Accepting a merge proposal between two kinds of node needs `keep`
  (422 `keep_required`): the kept node's kind is the merged subject's.
- A reviewer may merge two nodes of different kinds (`POST /v1/review/{id}`,
  `POST /v1/nodes/merge`): the kept node keeps its kind. The dream's merges
  still need one kind (409, `kind_mismatch`).
- `POST /v1/link` between two nodes that already have that edge confirms it
  (200, `created: false`) instead of adding a second one.
- `GET /v1/archive` items say what each record was: a node's name and kind,
  a fact's statement and node, an edge's type and ends.
- With a chat model, each turn is read for how it was meant before what it
  says: tests, role-play, examples, irony, questions, hypotheses, guesses and
  plans are not stored as facts (a test leaves one `activity` fact flagged
  `episode`), and a doubt becomes a fact proposal; the job report counts what
  was set aside (`dropped_by_intent`). `extraction.intent = false` turns it off.
- Facts that only hold for the work in progress (and one-off requests) are
  kept as session state instead of memory; when the session ends, or after
  `extraction.session_idle_minutes` (120) without a turn, one model call
  reads those notes and what the person said in the session, leaves at most
  one line about what was done, and the rest expires.
- The same value told again in other words or units ("120 o", then
  "120 B"; a list restated) keeps the fact already held instead of
  replacing it; the job report counts it as `facts.kept_same`
  (`[extraction] restatement_question`, on by default).
- A list told one element at a time (the members of a club, the ingredients of a recipe) keeps
  every element instead of each one replacing the previous; removing or correcting
  one element closes that element only, and a free-form value whose cardinality the
  model does not give is kept beside the old one as a conflict rather than replacing it.
- A fact that a turn shows to be about a neighbouring subject, or that mixes
  two subjects, becomes a fact proposal for review instead of being attached.
- A file or slug and its title are proposed as one subject (adjudicated, or
  a merge proposal); a correction of one flags the other's stale value.
- Names and aliases are stored without Markdown debris: formatting is stripped, a
  name with an unbalanced bracket or quote is refused (`remember` answers
  422), and a name the source does not give is never an alias or a proposal.
- Accepting a merge proposal in the review queue merges the two nodes
  (optionally naming the one to keep) instead of being refused;
  `GET /v1/nodes/{id}` shows the node's `rev`.
- A list page stops at the first item that does not fit the budget instead of
  skipping it, so the next page starts there; `/archive` is now newest first
  across nodes, facts and edges, and the lists default to 100 items a page.
- Recall ranks a node that holds no fact of its own (a place the model named
  after someone's workplace, say) after the facts that matched the same
  words of the question, so its handle no longer pushes the answer out of a
  small budget.
- A question that names exactly one subject, by its name or a whole alias,
  now also brings that subject's facts that share no word with the question
  ("Where does Kiki live?" finds Karim's move), still within the budget.

### Fixed
- A pause or a shutdown that lands just as a background run takes its next job
  now stops the run before that job starts.
- A fact a dream consolidates in a scope with a forced language is no longer
  labelled with that language when its text cannot be identified as written
  in it.
- A fact is no longer attached on its own to a known subject that neither its
  turn nor its session names (another document spoken of by the same words): the
  adjudication decides, and what it does not attach becomes a fact proposal
  (`facts.unnamed` in the job report).
- Two unrelated statements about one subject under the same free-form key
  (`content`, `info`) no longer replace each other: a changed free-form value
  goes to the replacement question ("replaces, or both true?"), and without a
  clear answer both stay, flagged as a conflict. Another number in the same
  form ("12 jars", then "13 jars") still replaces without a question.

## 0.1.0a2 — 2026-10-30

### Added
- `PATCH /v1/facts/{id}` takes `scopes` to share a fact with a parent scope or
  with a chosen list of scopes (also `patch_fact(..., scopes=...)` in the
  Python client); a token adds or removes only scopes it owns, and a fact
  never leaves the scopes of its node (detach it in the same call).
- `GET /v1/health` shows the background run in progress, whoever started it:
  its id, the token that started it, when, its budget, jobs done and left,
  and whether it is yielded.
- `POST /v1/background/run` takes `"wait": false` to answer 202 with the run
  id at once, and `GET /v1/background/run/{id}` follows the run (also
  `run_background(wait=False)` and `background_run(id)` in the Python client).

### Changed
- A background run refused because another is in progress names that run.
- Accepting a merge proposal through `POST /v1/proposals/{id}` answers 409
  with a message that says how a merge is applied (a dream's `merge_nodes`,
  approved in review).

## 0.1.0a1 — pre-alpha release notes

The first release, a pre-alpha: everything below is built and tested, but
the API may still change.

iifa is a self-hosted memory service for an agent you run yourself: it
keeps what is known about people, projects, places and topics, links them,
notices when something changes, and hands it back **under a token budget
you set** — handles first, content on demand. One encrypted
SQLite file, no external database, no cloud, no telemetry. Start with the
README's quickstart (pip or Docker, a few minutes, no model needed).

### What works
- **The service**: an HTTP API (`iifa`, port 8765, loopback by default) with
  MCP at `/mcp`, an MCP server over stdio (`iifa mcp`), a Python SDK, and a
  Docker image (non-root, database on a volume, key mounted apart).
- **Recall under a budget** (P1, P2): `recall` returns handles — id, title,
  one line, why it matched — never more than `budget_tokens`, never an item cut
  in half; `expand` gives the content of the handles you pick, under a budget
  too. A profile block (pinned facts, the most central subjects) opens a
  session.
- **Writing**: explicit facts (`remember`), links, and recorded turns that the
  background turns into nodes, facts and edges when a model is configured.
  Facts keep their provenance (P6) and the language they were told in (P12);
  a contradiction closes the old fact instead of overwriting it, and
  `/v1/at` answers what was known at any date (P7). Text from a web page or
  a tool never becomes a fact about you on its own: at most a proposal that a
  reviewer accepts or rejects.
- **Scopes** (P9): `kind:name` paths (`user:alice/mode:assistant`); tokens are
  bound to scopes and permissions (`read`, `write`, `forget`, `dream`,
  `review`, `admin`); filtering happens before any scoring.
- **Archive, restore, forget** (P5): archive is reversible and hides a record
  everywhere; forget is real (rows, index, vectors, queue, dream journal) and
  needs its own permission.
- **Dream** (optional, offline): a model proposes reorganisations from a
  closed catalogue (summaries, merges, consolidations, themes, links); the
  service checks, journals and applies them, a dream can be undone whole, and
  one that lowers the recall probes rolls itself back. Merges and
  consolidations wait for a reviewer by default.
- **Your model's time** (P11): background work runs `auto`, `paused` or
  `on_demand` (only inside a budgeted run), one call at a time per endpoint by
  default, and yields to your agent on request. Nothing calls a model while a
  turn is waiting (P3).

### Tiers (P10)
| Tier | Needs | You get |
|---|---|---|
| 0 | nothing | lexical recall (BM25) and one hop over links, `remember`, names and aliases, the profile block, turns queued and known subjects linked |
| 2 | an OpenAI-compatible chat endpoint | extraction from every turn in the source's language, entity resolution, contradictions, summaries, the dream |
| 3 | + an embeddings endpoint | dense recall fused with lexical, semantic resolution and merge proposals; served at tier 2 when embeddings are down, and it says so |

Measured numbers (recall latency, the benchmark against no memory, truncated
history and naive vector top-k) are in [docs/benchmarks.md](docs/benchmarks.md)
with the command that reproduces each.

### Adapters
| Harness | What | Built against |
|---|---|---|
| Claude Code | plugin: hooks for the five lifecycle moments + the MCP tools | Claude Code 2.1.283 |
| Hermes Agent | memory provider (standard library only) | Hermes Agent v0.22.4 |
| opencode | plugin (TypeScript, no dependency) | opencode 1.19.4 |
| OpenAI Agents SDK | example: tools + dynamic instructions | openai-agents 0.24.0 (optional extra `openai-agents`) |

Every adapter is thin, fails open (service down, slow, or answering garbage:
the turn proceeds without memory) and injects memory only as a labelled
`recalled-data` block that says it is data, not instructions (P8).

### Known limits
- Not on PyPI yet: install from a clone (`pip install .`) or build the image.
- The Claude Code, Hermes, opencode and Agents SDK adapters are tested
  against a scripted fake service, not yet inside a running harness. These
  harness APIs move; each adapter's README names the version it follows.
- Tested on Linux and Windows; macOS is expected to work but untested.
- The opencode plugin's TypeScript runs under Node's type stripping and is not
  type-checked by a compiler.
- Quality at tiers 2 and 3 depends on the model;
  [docs/benchmarks.md](docs/benchmarks.md) reports what was measured with
  qwen2.5:7b, qwen3.5:9b and bge-m3, misses included. At tier 2, common words
  of a question can pull other subjects' facts ahead of the one asked about,
  and an alias in the question reorders results without adding that
  subject's facts.
- Near forms of a name are never merged on their own: in Russian a female
  name can also be the male name's genitive (Ивана, Иван), so such pairs are
  left as proposals for a reviewer. Arabic attached prefixes are not split.
- A change told in another language than the previous value does not
  supersede it: both facts stay, flagged `conflict`. A single-owner scope can
  force one language to avoid it.
- Plain HTTP: for access from another machine, put a TLS reverse proxy in
  front. The Docker image is checked on `linux/amd64` only.
- Without the `vectors` extra (numpy), tier-3 vectors use pure Python;
  `/v1/health` says past which size that gets slow.
- One process and one file: no replication, no multi-node (by design).
