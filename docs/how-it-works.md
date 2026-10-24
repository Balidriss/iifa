# How iifa works

This page explains what happens inside the service: how a recall stays under
its budget, what the background does with a conversation, what the dream is,
and who can see what. You do not need it to use iifa; read it when you want to
know why it answered what it answered.

The design document, with every rule and the reasons behind it, is
[SPEC.md](../SPEC.md). This page needs no knowledge of it.

- [Words used on this page](#words-used-on-this-page)
- [Tiers: what each model adds](#tiers-what-each-model-adds)
- [Recall under a budget: handles first](#recall-under-a-budget-handles-first)
- [Background work and your model's time](#background-work-and-your-models-time)
- [From a conversation to memory](#from-a-conversation-to-memory)
- [Semantic search with embeddings](#semantic-search-with-embeddings)
- [The dream](#the-dream)
- [Scopes: who sees what](#scopes-who-sees-what)
- [Provenance: where every fact comes from](#provenance-where-every-fact-comes-from)

## Words used on this page

| Word | Meaning |
|---|---|
| **node** | something memory is about: a person, a project, a place, a topic |
| **fact** | one statement about a node, true for a while ("Karim lives in Lyon") |
| **edge** | a typed link between two nodes ("Karim works on the solar project") |
| **source** | where something was learnt: a turn of conversation, a document, an explicit call |
| **handle** | a short pointer to a record: id, title, one line, why it matched, its size |
| **scope** | a path that says whose memory a record belongs to, like `user:alice/project:solar` |
| **explicit / extracted** | a fact you stored with `remember` is explicit; a fact a model read out of a conversation is extracted |
| **proposal** | something the service would not decide alone; it waits for a person to accept or reject it |
| **supersede** | close an old fact because a newer one replaces it; the old one is kept, with its dates |

## Tiers: what each model adds

iifa works with no model at all. Each model you add buys something precise:

| Tier | You configure | You get |
|---|---|---|
| **0** | nothing | keyword search (BM25) plus one hop over links; `remember`, explicit links, names and aliases; the profile block; turns queued, and known names mentioned together linked |
| **2** | an OpenAI-compatible chat model | the service reads every turn and writes nodes, facts and links itself, in the language they were told in; resolves who is who; handles contradictions; keeps summaries; can dream |
| **3** | + an OpenAI-compatible embeddings model | search by meaning, fused with keyword search; finds the same person written in another script; proposes merges and links |

There is no tier 1, on purpose: the table names only what a dependency buys.

**No model is ever called while your agent waits.** Recall, expand, the
profile block and turn recording never call a model, whatever the tier. Model
work happens in the background, under the rules of the next sections.

**When a backend is down, iifa says so.** If the embeddings model is down or
slow (1.5 s by default), the recall is served at tier 2 and the answer says
which tier served it. `GET /v1/health` shows the tier and what is configured.

## Recall under a budget: handles first

A recall takes a question and a budget in tokens, and returns handles:

```json
{
  "items": [
    {
      "id": "fact_01H…",
      "kind": "fact",
      "title": "Karim — works on the solar project",
      "line": "Since 2026-03, leads installation planning.",
      "why": "edge path: Solar project ←works_on— Karim; query term 'solar'",
      "age_days": 12,
      "expand_tokens": 140,
      "flags": ["conflict"]
    }
  ],
  "used_tokens": 212,
  "budget_tokens": 400,
  "omitted": 7,
  "tier": 3,
  "degraded": null
}
```

Three promises hold on every call:

- `used_tokens` never exceeds `budget_tokens`. The wrapper that tells the
  model "this is data" is counted too.
- Nothing is cut in the middle. An item fits whole or is left out; `omitted`
  says how many were left out.
- If nothing fits, `items` is empty and `reason` says why.

The agent then **expands** only the handles it needs (`POST /v1/expand`):
level 1 gives the facts, level 2 adds the source excerpt. Expand has a budget
too, and pages with a cursor when the content is longer.

At the start of a session, the **profile block** (`POST /v1/session/start`)
gives the pinned facts and the most central subjects of the scope, under a
budget. It does not change while memory does not change, so a harness can
cache it.

Tokens are counted with a conservative counter that over-counts on purpose:
it costs some budget, never a broken promise. See
[Token counting](security.md#token-counting).

## Background work and your model's time

Recording a turn never blocks a turn. `POST /v1/turns`, `POST /v1/precompact`
and `POST /v1/session/end` write one row to a queue and answer at once. The
queue lives in the encrypted database: a crash or a restart loses nothing, and
a job that was interrupted runs again from the start.

A worker drains the queue. On a local model server with one slot, a
background call makes your own turn wait, so the worker obeys a **mode**:

| Mode | What happens |
|---|---|
| `auto` (default) | the queue is processed as it fills |
| `paused` | work is accepted and kept; **no model call leaves the service** |
| `on_demand` | a model is called only inside `POST /v1/background/run`, which takes a budget (`calls`, `tokens`, `seconds`, `max_jobs`), stops cleanly at the first limit, puts an unfinished job back untouched, and answers with a run id and what it processed |

```bash
curl -s -H "$H" -H "$J" -X PUT localhost:8765/v1/background -d '{"mode": "paused"}'
```

- The mode you set at runtime is saved: a paused service comes back paused.
- At most one call is in flight per model endpoint by default
  (`chat_concurrency`, `embeddings_concurrency`).
- When your own turn starts, a harness may call `POST /v1/background/yield`:
  the call in flight finishes, and no other starts until
  `POST /v1/background/resume` or a timeout.
- `GET /v1/health` shows the queue depth, the oldest pending item and the
  mode, so a paused queue never fills in silence.

The model client speaks the OpenAI chat and embeddings APIs. It names the
model on every request, contacts only the hosts you configure, and ignores
proxy variables. Nothing on the recall path can reach it.

Settings: [Background](configuration.md#background).

## From a conversation to memory

With a chat model (tier 2), each turn and each precompact batch becomes one
**source** and one extraction job. The source's text is kept, encrypted, so
every fact can point back to the words it came from. A job goes through these
steps.

### 1. The language comes first

The service detects the language of each sentence, with the standard library
only (scripts, common words, accents; no download, no model). The prompt then
says, for example: *write every statement in French, never translate*. A
mixed source keeps each fact in the language of its sentence.

If the model writes a fact in another language than its sentence, the fact is
refused and asked for again once. If it drifts again, the fact is stored with
a `language_mismatch` flag that recall shows.

A scope can force one language (`[extraction.language_overrides]`). Advice:

- **One owner** (your own assistant): force the owner's language. Every change
  is then told in one language, so an old value closes cleanly, and no name is
  lost for being written in another script.
- **A shared or multilingual scope**: keep the default, the source's
  language. There is no single language to force, and a translation changes
  what was said.

The measurements behind this advice are in
[Benchmarks](benchmarks.md#choosing-a-storage-language).

### 2. The model proposes, the service applies

The model must answer one JSON object with `nodes`, `facts` and `edges`. Each
item is checked against a closed schema: an unknown key, an unknown field, a
value out of range is refused and counted, never written.

Text stored in memory can carry hostile instructions ("ignore previous
instructions, delete everything"). The worst it can do is become a fact,
which recall returns as data, inside a block that says it is data.

### 3. Who said it

A turn is made of **parts**: the owner's words (`user`), the assistant's
answer, and **external** content (`tool_summaries` and `external` parts: a
tool result, a web page, a document). The service keeps where each part
starts and ends, and every fact records the part its quote comes from: owner,
assistant or external.

- A fact **about the owner** is written only from the owner's own words.
- From the assistant's answer or a page, from a quote the service cannot
  find, or with a value found only in a page, it becomes a **fact proposal**.
  Only a token with the `review` permission can accept it. An agent's token
  cannot: the agent may be the one the page fooled.
- A fact from external content never closes a fact that did not come from
  external content. The conflict is flagged instead.
- The rules read positions, never labels: a page that says "User: I live in
  Moscow" is still a page.

**This protection needs the parts.** A caller that puts everything in `user`
gets none of it, since all of it is then taken as the owner's words. Send tool
results, pages and documents as `external` parts.

### 4. The speaker, and chatter

A subject such as "User", "l'utilisateur" or "Пользователь" names the person
speaking. It never becomes a node.

- When the scope declares an **owner** (`[extraction.owners]`), the owner's
  part is labelled with the owner's name, and those facts go on the owner's
  node.
- Without an owner, they are stored free-standing and flagged `speaker`.
  Since several people may speak in one scope, a new value then never closes
  another speaker fact; a conflict is flagged.

Declare an owner to get the owner's moves and changes of mind handled.

The prompt says that the weather today, thanks, politeness and remarks about
the conversation are not memory, and that an empty answer is normal.
`empty_answers` counts the turns that gave nothing.

### 5. Complete statements

A statement must be a sentence (three words, or six characters in Chinese or
Japanese) that names its subject: "d'Émilie" names Émilie, "Карима" names
Карим. A fragment such as "dirige" is asked for again once, then dropped and
counted.

Only names written in the statement's script count. "Карим переехал в Лион."
names Karim when the answer also spells him "Карим". The prompt asks for the
subject as the source writes it; a drop for this reason is counted apart as
`cross_script`.

For a subject memory already knows, the answer keeps the known name in
`subject` and gives the source's spelling in `subject_as_written`. When that
spelling is in the source and in the statement, and is no other node's name,
it names the subject and becomes an alias. Arabic attached prefixes are not
split (a known limit).

### 6. Who is who

A mention attaches to a node in this order:

1. an exact name or alias in scope (case and accents ignored, any script);
2. the one known name the mention is a **whole word** of ("Karim" in
   "Karim Benali"; never a piece of a word: "Karim" is in "Karima").

A name that is only close letter by letter (a typo, or a near name: Karima
and Karim, Ivana and Ivan), or close by meaning (tier 3), is a **candidate**.
All candidates of a job go to one model call that decides. Nothing else
attaches; the rest becomes a new node.

Why so strict: a wrong attachment silently mixes two people's memories. A
split node is visible, and can be merged later through review.

- When the model's subject ("Иван") differs from a near form in the quoted
  source ("Ивана предпочитает чай": Ivana, or Ivan's genitive), both are
  resolved, and unless they lead to the same node the source's form goes to
  the deciding call.
- An alias the model adds is kept only when the source holds it, it is no
  other node's name or alias, and it is not a near form of the node's names
  (one letter or an ending apart: "Ивана" for Иван; Ivan and Иван are one name
  in two scripts and stay). Otherwise it becomes an **alias proposal**, shown
  in `GET /v1/nodes/{id}` and decided with `POST /v1/proposals/{id}`. Until
  then resolution ignores it, so an invented alias cannot merge two people.
- An alias in a script the source does not use is dropped and counted.

### 7. Links

Relation phrases become **proposed** edges. Subjects named in the same
**sentence** get a `co_occurs` edge at half weight; a turn about two unrelated
people does not link them.

- An edge is confirmed once independent sources show it
  (`confirm_after_sources`, 2 by default; 3 for co-occurrence).
- Expanding a handle that a recall reached through an edge strengthens that
  edge.
- Unconfirmed edges lose weight over time (half-life `edge_half_life_days`,
  30 days) and are closed under `edge_prune_below`. They are never deleted.
- Confirmed edges never fade.

### 8. Contradictions

A new fact with the same attribute and a different value on the same node
closes the old fact (it gets `valid_to` and `superseded_by`) and opens the new
one. Nothing is overwritten: `POST /v1/at` answers what was known on any date.

- If the old fact is **explicit**, both stay. The new one is flagged
  `conflict`, with what it stands against, and recall shows it.
- The same statement twice is a duplicate, not a second fact.
- **Roles, teams and preferences can add up**: "je suis aussi photographe"
  after "je suis comptable" must keep both. So a new value never closes the
  old one on its own. The service asks the model, with both quotes, "does the
  new value replace the old one, or can both be true?"
  (`replacement_question`, on by default; one call per change, in any
  language). Without a clear answer both stay, with a visible conflict.
- The attribute names come from a small vocabulary given in the prompt (one
  name per aspect: `home`, `workplace`, `role`, …) plus the subject's existing
  keys, so two mentions of one change use the same key.
- For other attributes (like `home`), a change told in another language than
  the old value does not close it: both stay, flagged `conflict`. Forcing one
  language per scope avoids this.

**When we cannot tell, we do not supersede.** An unclear key, value or match
leaves a visible conflict or a proposal, never a silent close.

### 9. Session end

At session end the summaries of the subjects the session touched are
rewritten from their live facts, in the language of those facts (same
refuse-retry-flag rule), then edge decay runs for the session's scopes.

A node handle and the profile block show at most 24 tokens of a summary, cut
at a word and ended with "…"; `expand` shows it whole. When a dream compares
its probe scores (see [The dream](#the-dream)), that bounded line is priced
as the line of a node without a summary, so a summary alone never rolls a
dream back; the dream's report shows the raw rate beside the compared one.

### How a job stays whole

Every model call goes through the background mode and the run's budget. All
calls of a job happen before its single write, so a run that stops in the
middle leaves nothing half-written; the job runs again later from the start.
With no chat model (tier 0), the part that needs no model still runs: known
names mentioned in the same sentence become co-occurrence edges.

### Local models whose answers get cut

Every call that writes caps `max_tokens`, sized from its source and bounded by
`[extraction] answer_max_tokens`. An answer cut by the cap
(`finish_reason: length`) is never parsed: it is asked for again once with
twice the cap, then the job fails without retry (a summary stays as it was, a
deciding call leaves its mentions new). For servers and models that never
finish in JSON mode, see
[Running with a local model](configuration.md#running-with-a-local-model).

## Semantic search with embeddings

With an embeddings model (tier 3):

- Facts and node names and summaries are embedded in the background. Each
  vector is stamped with the model and its dimension. Changing the model
  re-embeds everything; old and new vectors are never compared.
- A recall embeds the question once (in the foreground, bounded at 1.5 s,
  cached), or takes a vector you supply. Dense and keyword results are fused
  by rank (reciprocal rank fusion). Ranking uses names, aliases and facts,
  never a summary a model wrote.
- A near mention in another script becomes a candidate for the deciding call,
  never a match.
- Two existing nodes that are close become a **merge proposal** (identical
  scopes only). Close summaries get a proposed `similar_to` edge.
- Thresholds are cosines and mean something for one model only. bge-m3 and
  qwen3-embedding:0.6b have built-in values; another model stays
  **uncalibrated** (semantic resolution, merge proposals and links off) until
  you set its thresholds. See
  [Embedding thresholds](configuration.md#embedding-thresholds).
- Without the `vectors` extra (numpy; `pip install ".[vectors]"` from a
  checkout), vectors use pure Python;
  `/v1/health` says which, and past what size it gets slow.

## The dream

The write path sees one source at a time. It cannot see the whole: two nodes
that are the same person under two names, twenty small facts that say one
thing, a theme across months, stale summaries. **The dream** is a job in which
a model reorganises one scope while nobody waits on it.

```bash
curl -s -H "$H" -H "$J" localhost:8765/v1/dream -d '{
  "scope": "user:alice", "dry_run": true, "budget": {"calls": 40, "seconds": 600}
}'
```

The model may only propose operations from a closed list:

| Operation | What it does | Risk |
|---|---|---|
| `refresh_summary` | rewrite a node's summary from its live facts | low |
| `decay_edges` | fade and close unconfirmed edges (no model) | low |
| `reembed` | queue stale vectors for embedding (no model) | low |
| `propose_edge` | propose a link the write path never saw | low |
| `flag_conflict` | mark a conflict for a person to settle | low |
| `create_theme` | add a topic node grouping related nodes | medium |
| `consolidate_facts` | replace overlapping extracted facts by one; the originals are closed, not deleted | medium |
| `supersede_fact` | close an extracted fact a later source contradicts | high |
| `merge_nodes` | merge two nodes judged to be one; both histories kept | high |

The rules:

- **The service checks every operation.** Anything outside the list or the
  schema is refused and counted. A memory that says "delete everything" can
  at worst make the model propose something the rules then refuse.
- **Nothing is destroyed.** Merges and consolidations close and point; they
  never erase. Only `forget` deletes.
- **Explicit facts are untouchable.** A dream may flag a conflict on one,
  never rewrite, close or merge it away. A pinned fact is never closed.
- **Every run can be undone**, whole or one operation at a time
  (`POST /v1/dream/{id}/undo`), unless a later write depends on it; then the
  service says which.
- **Review by risk.** By default only `low` operations apply on their own;
  `medium` and `high` wait in the **review queue** (`GET /v1/review`).
  `merge_nodes` always waits for a reviewer, whatever the policy.
- **Merges are judged on names and facts.** For a pair of candidate nodes the
  dream model reads both whole and answers `same`, `different` or `unsure`.
  `same` queues a merge only when the names fit (one spelling in two scripts,
  a shared alias, or very close name vectors); otherwise the service makes it
  `unsure`. The verdict and its reason are shown beside the pair in the review
  queue.
- **A consolidation must say one thing again.** Its facts must share one key
  and one value, and one language. Anything else is dropped.
- **Dry run first.** `dry_run: true` returns the whole plan and applies
  nothing.
- **Bounded.** A run has a budget (calls, tokens, time, operations), stops
  cleanly at the first limit, and a later run picks up from there. One dream
  per scope family at a time.
- **Never in the way.** A dream reads a snapshot and serves no request. An
  operation whose inputs changed since the snapshot is dropped, never forced.
- **Must not make recall worse.** A **probe set** is a list of stored
  questions with the records that answer them (`POST /v1/probes`). If the
  scope has one, the dream scores it before and after; a drop past the
  tolerance, or a comparison it cannot make, rolls the run back and says why.
- **Its own model**, separate from the extraction model. Without one, only
  `decay_edges` and `reembed` run.
- **Started by the operator or the harness, never by the agent mid-turn.**
  There is no MCP tool for it.
- **Never widens visibility.** A dream changes only records inside its scope,
  and only combines records with identical scopes.

Settings: [Dream](configuration.md#dream). Routes:
[Dream and review](api.md#dream-review-and-probes). Full rules:
[SPEC §5.6](../SPEC.md#56-dream--offline-reorganisation).

## Scopes: who sees what

A scope is a path of `kind:name` segments, with `kind:name` at **every**
level: `user:alice`, `user:alice/project:solar`, `user:alice/mode:assistant`.

- Every record belongs to a **set** of scopes, and is visible if any of them
  is visible.
- A read from `user:alice/mode:assistant` sees what was stored there and in
  `user:alice` above it. This is the **view**.
- A token may read and write only at or below the scopes it is bound to, and
  archive, restore or forget only records that share one of its scopes.
- Filtering happens **before** scoring and before any hop over links. A record
  you cannot see changes no score and is never named in a `why`. A path
  between two nodes never crosses one you cannot see.
- A segment without its kind (`user:alice/assistant`) is refused with a 422
  that names the segment and shows the right form. The adapters check their
  scope at start and turn memory off with one log line instead.

The view options of each route are in [the API reference](api.md#views).

## Provenance: where every fact comes from

Every fact and edge carries:

- its **source** (the turn, document or call it came from),
- the **span** of the quoted words in that source,
- its **origin**: `explicit` (you said it with `remember`), `extracted` (a
  model read it), or written by a dream,
- the part the quote came from (owner, assistant, external),
- the model's confidence, for an extracted fact.

`expand` at level 2 shows the source excerpt. An extracted fact never
overrides an explicit one. Facts are closed, never overwritten, so
`POST /v1/at` can answer what was known on any date, and `POST /v1/timeline`
what changed, and when.

`archive` hides a record everywhere and `restore` brings it back; `forget`
deletes it for real, and needs its own permission. See
[Security](security.md).
