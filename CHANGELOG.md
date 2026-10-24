# Changelog

What changed in each release of iifa, for people who use it. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/), with Python's spelling for
pre-releases (0.1.0a1).

## Unreleased

### Changed
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
