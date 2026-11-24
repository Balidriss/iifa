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
- [Starting over with an empty memory](#starting-over-with-an-empty-memory)

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
| `IIFA_ASSISTANTS` | a JSON object like `{"user:alice": "NOVA"}`: the name of the assistant answering in a scope (default `Assistant`); see [Owners and languages](#owners-and-languages) |
| `IIFA_SELVES` | a JSON object like `{"user:alice": {"name": "NOVA", "aliases": ["Nova"]}}`: the memory of a scope is also the agent's own; see [Owners and languages](#owners-and-languages) |
| `IIFA_REVIEW_SELF_RULES` | `on` to make the agent's own commitments and preferences wait in the review queue instead of being written by a turn (default off); see [Owners and languages](#owners-and-languages) |
| `IIFA_DETECT_SECRETS_BY_WORDING` | `on` to have the service itself also read passwords, codes and keys from their wording ("my password is ...", `DB_PASSWORD=`), at your own risk (default off); see [Security](security.md#storage) |
| `IIFA_REPLACEMENT_QUESTION` | `off` to stop asking the model whether a new role, team or preference replaces the old one (default on) |
| `IIFA_RESTATEMENT_QUESTION` | `off` to stop asking the model whether a changed value that shares words with the one held tells it again (default on) |
| `IIFA_VALUE_NORM` | `on` to ask the model for each value's comparison form, so that a value restated in another language is linked without a second call (default off) |
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

[recall]
observed_half_life_days = 7.0      # an observed fact's recall score halves every this many
                                   # days (0: observations never move down the ranking)

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
restatement_question = true        # ask whether a change that shares words with the
                                   # value held tells it again ("120 o" / "120 octets")
value_norm = false                 # ask each value's comparison form (never shown);
                                   # links "Londres" and "London" without a call
intent = true                      # read how each turn is meant before what it says:
                                   # tests, examples, questions, plans and work in
                                   # progress are not stored as lasting facts
session_idle_minutes = 120         # a session nobody ends is closed (and its work in
                                   # progress consolidated) after this idle time; 0: never
review_self_rules = false          # true: what a declared self commits to or prefers
                                   # waits in the review queue before it is memory
detect_secrets_by_wording = false  # true: the service also reads secrets from their
                                   # wording, at your risk (English, French, German,
                                   # Spanish, a little Russian; may drop ordinary facts)

[extraction.language_overrides]
"user:alice/agent:notes" = "fr"    # a scope may force the language of what is written into it

[extraction.owners]                # who speaks in a scope (and every scope below it)
"user:alice" = { name = "Alice", aliases = ["Алиса"] }   # or just "user:bob" = "Bob"

[extraction.assistants]            # the assistant's name in a scope (default "Assistant")
"user:alice" = "NOVA"

[extraction.selves]                # the memory of a scope is also the agent's own
"user:alice" = { name = "NOVA", aliases = ["Nova"] }

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
| `write` | remember, link, edit facts, correct a node (rename, aliases, kind), record turns, precompact, session end, archive and restore, yield and resume, decide alias proposals |
| `forget` | irreversible deletion |
| `dream` | start, cancel and undo a dream; manage probe sets |
| `review` | accept or reject proposals and what a dream queued; merge two nodes |
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

**Name the assistant** of that scope, if you like. Its answers are then
labelled with that name for the model, and a fact taken from them says
`"from": "NOVA"` when recalled instead of `"assistant"`. Nothing else
changes: what the assistant says is still never taken as the owner's words.
The name cannot be one of iifa's own labels (`User`, `Tool`, `Web`,
`Document`, `External`) nor the owner's name or alias.

```toml
[extraction.assistants]
"user:alice" = "NOVA"
```

**Let the memory be the agent's own**, too, by declaring a self instead of
(or under the same name as) the assistant:

```toml
[extraction.selves]
"user:alice" = { name = "NOVA", aliases = ["Nova"] }
```

What the agent says about itself is then memory on its own node: what it
learnt, a view it holds, a commitment, what it did. A rule of method ("to
export a report, do this") goes to the task it serves, a node of kind
`task`, so it comes back when the task does. Each turn also names the
activity you asked the agent for ("drawing a map"), in your own words and
without what it is of or for; within a session it stays the current
activity until you ask for something else (a map, then labels for the map:
two activities), and every rule learnt meanwhile is linked to it, whatever
subject the model gave the rule. The tool or guide used for it is linked
by a `uses` edge, the session's closing episode is linked to every activity
of the session, and `expand` on a linked fact shows its `activities`. A
later request that names the same kind of thing ("make me a map of the
balcony") brings that activity's rules first, then the rules of what it
used, then its last episodes (the last time first; with the dream's
`group_activities`, those of every occasion it was done on another object), ahead of facts that only share words; with
embeddings (tier 3) and `activity_threshold` set, a request in other words
can match it too. Those rules never push out the subject you ask about, and
take two thirds of the budget at most. Only your
words set the activity: never the agent's reply nor a page. Its politeness is still not
kept. Three guards hold:

- a page, a tool result or a document in the same turn never gives the
  agent a belief or a method, even restated in other words: such a fact
  waits in the review queue for you to accept. This holds for what the
  model reads as the agent's own (about itself, or a rule of method); a
  rule the model writes as a plain fact about a thing is kept like any
  fact the agent states. The exception is content the
  host marks `vouched` (a guide you attached, the pages of a search you
  asked for): when every external part of the turn is vouched, the agent
  learns from it at once, and those facts are flagged `vouched` so you can
  find them. Vouching is trust you give: a hostile page among vouched ones
  would teach the agent too. It never covers facts about you. The guard
  reads one turn at a time: what the agent restates in a later turn, with no
  page in it, is kept as its own;
- the agent saying again what memory already holds about it adds nothing;
- the agent never closes what the owner said about it, nor the reverse: a
  correction ("no, it was not fixed") stays visible beside the agent's fact
  for a reviewer to settle;
- a rule the model files as a preference in one turn and as a commitment in
  the next is one rule: the same value on the same topic writes nothing,
  another value is kept beside the first with a visible `conflict`, and
  neither is closed.

A work session's closing episode is then the agent's own ("NOVA planned
the spring garden with Alice"), the profile block at the start of a session
carries the agent's commitments and preferences, and the agent can store something about
itself with `remember` and `about_self: true` (`memory_remember` takes the
same flag). In the profile the agent's rules leave room for the central
nodes: ahead of them they use at most a third of the budget (commitments
first, the newest first), and the others follow the nodes if room is left.

The memory may also be rooted at the agent: declare the owner and the self on
a scope such as `agent:nova` and the owner is a node inside it. No rule reads
the kind of a scope segment.

**Read the agent's rules before it follows them.** A commitment is carried
into every later session. With `review_self_rules = true` under
`[extraction]`, what the agent commits to and how it prefers to behave is
never written by a turn, whether you asked for it or the agent took it on:
it waits in the review queue, and becomes memory when a `review` token
accepts it. A rule said again is asked about once; one you rejected is not
asked about again. Said again in other words, it is the chat model that
recognises it: the rule is compared with those held, waiting or refused,
and is asked about unless the model is sure it is one of them. An accepted
commitment replaces the one it changes, and the review queue states which
(`replaces`) before you decide.
What the agent did or learnt, and its rules of method, are kept as before.

Without a self, nothing of this applies. A self cannot be declared on
`"*"`, and it obeys the assistant's naming rules.

**Force a language** for that same scope, the owner's:

```toml
[extraction.language_overrides]
"user:alice" = "fr"
```

A language declared on `"*"` holds for every scope that declares none
nearer. An owner cannot be declared on `"*"`: one person would then speak
in every scope, shared ones included, so the configuration refuses it.

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
| `activity_threshold` | a request matches an activity the agent did by meaning, not only by a shared word (off unless set; `IIFA_ACTIVITY_THRESHOLD`; with `bge-m3`, 0.60 to 0.65 measured, see below) |

`bge-m3` (recommended) and `qwen3-embedding:0.6b` have built-in values. For any
other model, set `resolve_threshold`, `merge_threshold` and `link_threshold`
together; until then its semantic resolution, merge proposals and links stay
off, and `/v1/health` says `uncalibrated`.

`activity_threshold` has no built-in value. Measured with `bge-m3` on the
`--self` bench corpus (`python bench/run.py --self --embeddings-url ...
--embeddings-model bge-m3`, the report's `vector_probes`, 27 requests):
twelve requests in other words reached their activity at 0.52 to 0.75,
fifteen unrelated requests stayed at 0.58 or below, so no value separates
the two. 0.60 matched six of the twelve and no unrelated request; 0.65
matched four and none; 0.55 matched ten and two unrelated ones. The
requests left out are those sharing neither a word nor its domain with
the activity ("the update text for the app store" for release notes).
A threshold here trades half of the reworded requests for no wrong
match: set it knowingly.

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

## Starting over with an empty memory

The whole memory is one encrypted file: nodes, facts, links, vectors, the
dream's journal, the review queue and the runtime background mode all live in
it. The configuration, the tokens and the key live elsewhere and are kept.
To forget everything:

1. Stop the service.
2. Delete the database file (`IIFA_DB_PATH`, else `[store] path`, else
   `iifa.db` in the working directory) and its `-wal` and `-shm` files:

   ```bash
   rm -f memory.db memory.db-wal memory.db-shm
   ```

3. Start the service. It creates an empty database with the same key.

Delete the files only while the service is stopped: a running service keeps
writing to the file it holds open. Deleting does not overwrite the bytes on
disk; they stay encrypted, so the old memory cannot be read without the key.
To be sure it never can be, also give the new database a new key. Backups of
the file, if you keep any, still hold the old memory.

With Docker, see [Docker](docker.md#starting-over).
