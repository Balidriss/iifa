# Configuration

iifa reads a TOML file (`--config path` or `IIFA_CONFIG=path`) and then
environment variables, which win over the file. With no file at all it runs at
tier 0 (no model) on `127.0.0.1:8765`, once it has a database key and a token.

- [The three common setups](#the-three-common-setups)
- [Environment variables](#environment-variables)
- [The configuration file](#the-configuration-file)
- [Tokens and permissions](#tokens-and-permissions)
- [Owners and languages](#owners-and-languages)
- [Background](#background)
- [Running with a local model](#running-with-a-local-model)
- [Embedding thresholds](#embedding-thresholds)
- [Dream](#dream)

## The three common setups

**No model** (tier 0): only the key, the file and a token.

```bash
export IIFA_DB_KEY_FILE=~/.config/iifa/db.key
export IIFA_DB_PATH=./memory.db
export IIFA_TOKEN="$(cat ~/.config/iifa/token)"
iifa
```

**With a chat model** (tier 2): the service reads every turn and writes the
memory itself. Any OpenAI-compatible server works; here, Ollama:

```bash
export IIFA_CHAT_URL=http://127.0.0.1:11434/v1
export IIFA_CHAT_MODEL=qwen3.5:9b
```

**With embeddings too** (tier 3): search by meaning.

```bash
export IIFA_EMBEDDINGS_URL=http://127.0.0.1:11434/v1
export IIFA_EMBEDDINGS_MODEL=bge-m3
```

From Docker, a model server on the host is
`http://host.docker.internal:11434/v1`. What each tier adds is explained in
[How it works](how-it-works.md#tiers-what-each-model-adds).

## Environment variables

| Variable | Meaning |
|---|---|
| `IIFA_CONFIG` | the TOML file to read (same as `--config`) |
| `IIFA_DB_PATH` | the encrypted SQLite file (default `iifa.db`) |
| `IIFA_DB_KEY` / `IIFA_DB_KEY_FILE` | the database passphrase, or a file holding it. **Required**; iifa never makes one for you. Lose it and the memory is gone |
| `IIFA_TOKEN` | one token with every permission on every scope. Good for a first try |
| `IIFA_TOKENS` | a JSON list of `{name, token \| token_sha256, scopes, permissions}` |
| `IIFA_HOST` / `IIFA_PORT` | bind address, default `127.0.0.1:8765`. Binding elsewhere is your explicit choice |
| `IIFA_DEFAULT_BUDGET_TOKENS` | budget used when a call gives none (default 400) |
| `IIFA_TOKENIZER` | `conservative` (default) or `tiktoken:<encoding>`; see [Token counting](security.md#token-counting) |
| `IIFA_MCP_HTTP` | serve MCP at `/mcp` (default on) |
| `IIFA_MCP_EXPOSE_FORGET` | expose `memory_forget` over MCP (default off) |
| `IIFA_MCP_DEFAULT_SCOPE` | scope used by MCP calls over HTTP that give none |
| `IIFA_BACKGROUND_MODE` | `auto` (default), `paused` or `on_demand`. A mode set at runtime through the API is saved and wins after a restart |
| `IIFA_CHAT_CONCURRENCY` / `IIFA_EMBEDDINGS_CONCURRENCY` | calls in flight toward each model endpoint (default 1) |
| `IIFA_CHAT_URL` + `IIFA_CHAT_MODEL` | an OpenAI-compatible chat endpoint (`…/v1`) and the model named on every call. Both or neither |
| `IIFA_EMBEDDINGS_URL` + `IIFA_EMBEDDINGS_MODEL` | the embeddings endpoint and model, same rule |
| `IIFA_CHAT_API_KEY` / `IIFA_EMBEDDINGS_API_KEY` | bearer keys for those endpoints, when they need one |
| `IIFA_MODEL_TIMEOUT` | seconds to wait for a model answer (default 120) |
| `IIFA_CHAT_EXTRA` / `IIFA_EMBEDDINGS_EXTRA` | a JSON object of extra request parameters, like `{"reasoning_effort": "none"}`. It may not replace `model`, `messages`, `input`, `stream`, `max_tokens`, `temperature` or `response_format` |
| `IIFA_CHAT_JSON_MODE` | `off` to stop asking for `response_format: json_object` (default on) |
| `IIFA_ANSWER_MAX_TOKENS` | the upper bound of every extraction answer's `max_tokens` (default 4096, at least 512) |
| `IIFA_RECALL_DENSE` | `off` to keep recall on keywords and links at tier 3 (useful on single-slot model servers); vectors still serve resolution and links (default on) |
| `IIFA_EMBEDDINGS_QUERY_TIMEOUT` | seconds a recall waits for its question's embedding before answering at tier 2 (default 1.5) |
| `IIFA_LANGUAGE_OVERRIDES` | a JSON object like `{"user:alice": "fr"}`: the language a scope, and everything below it, forces on what is written into it |
| `IIFA_OWNERS` | a JSON object like `{"user:alice": {"name": "Alice", "aliases": ["Алиса"]}}` (or just `{"user:bob": "Bob"}`): who speaks in a scope; see [Owners and languages](#owners-and-languages) |
| `IIFA_REPLACEMENT_QUESTION` | `off` to stop asking the model whether a new role, team or preference replaces the old one (default on) |
| `IIFA_DREAM_URL` + `IIFA_DREAM_MODEL` | the dream's own chat endpoint and model; see [Dream](#dream) |
| `IIFA_DREAM_API_KEY` / `IIFA_DREAM_EXTRA` | its bearer key, and extra request parameters as a JSON object |

## The configuration file

Every section is optional. A full example:

```toml
# iifa.toml
[store]
path = "/var/lib/iifa/memory.db"
key_file = "/etc/iifa/db.key"
checkpoint_interval = 0.5          # seconds between background WAL checkpoints; 0 = SQLite's own

[background]
mode = "auto"                      # auto | paused | on_demand
chat_concurrency = 1
embeddings_concurrency = 1
yield_timeout = 30.0               # a yield without a resume ends after this many seconds
max_attempts = 3                   # attempts before a job with a passing failure is marked failed
retry_delay = 30.0                 # seconds before the next attempt, times the attempt number

[models]
chat_url = "http://127.0.0.1:11434/v1"
chat_model = "llama3.2"            # named on every call: no server default is relied on
embeddings_url = "http://127.0.0.1:11434/v1"
embeddings_model = "nomic-embed-text"
timeout = 120
chat_json_mode = true              # false for model/server pairs that never finish in JSON mode

[models.chat_extra]                # added to every chat request; never replaces what iifa sets
reasoning_effort = "none"          # e.g. a thinking model behind Ollama's OpenAI API

[extraction]
chunk_chars = 12000                # a long precompact batch is read in pieces of this size
answer_max_tokens = 4096           # upper bound of every answer's max_tokens
confirm_after_sources = 2          # independent sources that confirm a proposed edge
edge_half_life_days = 30.0         # unconfirmed edges lose half their weight in this many days
edge_prune_below = 0.1             # ...and are closed (never deleted) under this weight
summary_nodes_per_session = 20     # at most this many summaries rewritten per session end
replacement_question = true        # ask "replaces, or both true?" for role, team and
                                   # preference changes (one model call per change)

[extraction.language_overrides]
"user:alice/agent:notes" = "fr"    # a scope may force the language of what is written into it

[extraction.owners]                # who speaks in a scope (and every scope below it)
"user:alice" = { name = "Alice", aliases = ["Алиса"] }   # or just "user:bob" = "Bob"

[embeddings]
recall_dense = true                # false: recall stays on keywords and links
query_timeout = 1.5                # past it, the recall is served at tier 2 and says so
batch = 32                         # texts per background embedding call
# resolve_threshold, merge_threshold, link_threshold, key_threshold:
# see "Embedding thresholds" below

[mcp]
http = true
expose_forget = false

[[tokens]]
name = "coder-agent"
token_sha256 = "…"                 # sha256 hex of the token you hand the agent
scopes = ["user:alice/agent:coder"]
permissions = ["read", "write"]    # no forget, no admin

[[tokens]]
name = "operator"
token = "…"                        # plaintext is hashed at load and dropped
scopes = ["*"]
permissions = ["read", "write", "forget", "dream", "review", "admin"]
```

## Tokens and permissions

Each client presents a bearer token. A token is bound to scopes (it reads and
writes only at or below them) and carries permissions:

| Permission | Allows |
|---|---|
| `read` | recall, expand, node views, lookup, timeline, `at`, `path`, the profile block, running a probe set |
| `write` | remember, link, edit facts, record turns, precompact, session end, archive and restore, yield and resume, decide alias proposals |
| `forget` | irreversible deletion |
| `dream` | start, cancel and undo a dream; manage probe sets |
| `review` | accept or reject proposals and what a dream queued |
| `admin` | stats, the background mode and runs, reading a whole subtree |

Any valid token may call `GET /v1/health`.

- Give an agent `read` and `write` only.
- `review` decides fact proposals (facts about the owner that were not the
  owner's own words) and queued dream operations. Give it to the person's own
  token, never to an agent's: the agent may be the one a page fooled.
- `dream` starts dreams but does not approve them.
- Tokens are stored as SHA-256 digests and compared in constant time. In a
  file, prefer `token_sha256`; a plaintext `token` is hashed at load.

The full route-by-permission table is in [the API reference](api.md#routes).

## Owners and languages

**Declare an owner** for a scope with a single person behind it. The owner's
words are then labelled with the owner's name, and facts like "I moved to
Lyon" go on the owner's node, where a later move closes the old one.

```toml
[extraction.owners]
"user:alice" = { name = "Alice", aliases = ["Алиса"] }
```

**Force a language** for that same scope, the owner's:

```toml
[extraction.language_overrides]
"user:alice" = "fr"
```

Keep the default (each fact in its source's language) for shared or
multilingual scopes. Why: [How it works](how-it-works.md#1-the-language-comes-first).
The numbers: [Benchmarks](benchmarks.md#choosing-a-storage-language).

## Background

The `[background]` section sets how the worker uses your model. The modes,
yield and resume are explained in
[How it works](how-it-works.md#background-work-and-your-models-time). At runtime:

```bash
curl -s -H "$H" -H "$J" -X PUT localhost:8765/v1/background -d '{"mode": "on_demand"}'
curl -s -H "$H" -H "$J" localhost:8765/v1/background/run -d '{"budget": {"calls": 20, "seconds": 120}}'
```

## Running with a local model

- **Thinking models behind Ollama.** Ollama's OpenAI API ignores
  `think: false`. Send `reasoning_effort = "none"` instead, in
  `[models.chat_extra]` or `IIFA_CHAT_EXTRA`.
- **JSON mode.** Some model and server pairs never finish an answer in JSON
  mode. Set `chat_json_mode = false`: the JSON object is then read out of the
  plain answer, with reasoning blocks removed.
- **Answer length.** Every call that writes caps `max_tokens`, sized from its
  source and bounded by `answer_max_tokens`. A cut answer is asked for again
  once with twice the cap, then the job fails.
- **Context window.** Ollama ignores a per-request `num_ctx`. For the dream,
  whose prompts are larger, use a model variant with `PARAMETER num_ctx 8192`
  (for example a `qwen3.5:9b-8k` you create), and set the window above
  `prompt_max_tokens` plus the answer.
- **One slot.** On a server that runs one request at a time, keep the
  concurrency at 1, consider `recall_dense = false`, and use `yield` /
  `resume` or `on_demand` so the background never makes your turn wait.

To check a model before trusting it, run the manual extraction script; see
[Benchmarks](benchmarks.md#extraction-against-a-real-model).

## Embedding thresholds

Thresholds are cosine similarities, and a value means something for one model
only:

| Setting | When a pair passes it |
|---|---|
| `resolve_threshold` | a near node becomes a candidate for the model to decide on |
| `merge_threshold` | two existing near nodes become a merge proposal (same scopes only) |
| `link_threshold` | two close summaries get a proposed `similar_to` edge |
| `key_threshold` | a new attribute name close to one the subject already has takes it (off unless set) |

`bge-m3` (recommended) and `qwen3-embedding:0.6b` have built-in values. For any
other model, set `resolve_threshold`, `merge_threshold` and `link_threshold`
together; until then its semantic resolution, merge proposals and links stay
off, and `/v1/health` says `uncalibrated`.

```toml
[embeddings]
resolve_threshold = 0.53
merge_threshold = 0.84
link_threshold = 0.70
```

`bench/thresholds.py` calibrates `resolve` and `merge` for a model on a list of
names kept apart from the test corpus. The built-in values and how they were
measured: [Benchmarks](benchmarks.md#embedding-models).

## Dream

The dream has its own model; nobody waits on it, so it can be larger and
slower than the extraction model. What a dream does:
[How it works](how-it-works.md#the-dream).

```toml
[dream]
url = "http://127.0.0.1:11434/v1"
model = "qwen3.5:9b"               # none: only decay_edges and reembed run
api_key = ""                       # when the endpoint needs one
# [dream.extra] takes extra request parameters, like [models.chat_extra]
answer_max_tokens = 2048
probe_tolerance = 0.02             # a probe score drop past this rolls the run back
adjudications_per_unit = 3         # merge candidates judged per node; 0: no merges
prompt_max_tokens = 4000           # one call's prompt; the model's window must hold it + the answer

[dream.review]                     # auto | review per risk class; the nearest scope wins
                                   # default: low = auto, medium and high = review
"user:alice" = { medium = "auto" } # merge_nodes is always reviewed, whatever this says
```
