# SPEC — a context-budgeted memory service for AI agents

> **iifa** is named after the Iifa Tree, whose roots link everything and
> carry memories back to their source.

---

## 0. Founding decisions

Each is marked where it is used.

| # | Decision | Choice | Why |
|---|----------|--------|-----|
| D1 | Language | **Python 3.11+** | The agent ecosystem lives there; TypeScript only for the opencode adapter |
| D2 | License | **Apache-2.0** | Broad adoption, patent grant |
| D3 | Name | **`iifa`** | Short, free on PyPI and npm, and carried by no other memory project |

---

## 1. What this is

A self-hosted, turnkey memory service that developers plug into **their own
agent and harness**. It links subjects to one another — people, projects,
places, objects, topics — keeps what is known about them **over time**, and
hands it back to the agent **without flooding the context window**.

One sentence for the README:

> *Rich long-term memory for your agent, delivered under a token budget you
> set — never more.*

### 1.1 Who it is for

Developers who already run an agent (their own loop, or an open-source harness
such as Claude Code, Hermes Agent, opencode, Goose, or one built on the OpenAI
Agents SDK) and want persistent, structured memory without adopting a whole
agent framework and without sending data to a cloud.

### 1.2 Why another one — the positioning

The field is crowded (Mem0, Graphiti/Zep, Cognee, Letta, Hindsight,
Supermemory, agentmemory…). Their known trade-offs, as of 2026:

- **Graphiti** — strongest temporal knowledge graph, but several LLM calls per
  ingested episode and a Neo4j dependency.
- **Mem0** — removed its graph layer from the open-source v3 rewrite (it was
  measured slower and more token-hungry for no recall gain); returns raw memory
  text for the agent to paste into its prompt.
- **Letta** — core memory blocks sit **permanently** in the context window.
- **Cognee** — powerful pipeline, many moving parts to orchestrate.

**None of them treats the context budget as the primary contract.** They return
"what is relevant"; none guarantees what it costs. That is the gap:

1. **Budget-first.** Every read takes a token budget and never exceeds it.
2. **Progressive disclosure.** Handles first, content on demand.
3. **Graph without the graph database.** One SQLite file, no Neo4j.
4. **Cheap writes.** Ingestion is asynchronous, batched, and never blocks a turn.
5. **Works without any model**, better with a chat model, best with embeddings.
6. **Multi-harness out of the box** — one HTTP API shaped on the lifecycle every
   harness already has, one MCP server, thin adapters shipped in the repo.

The headline metric, published in the README with its method: **recall quality
per injected token** against the alternatives above.

---

## 2. Non-negotiable principles

These are the product. A change that weakens one is a regression, whatever it
gains elsewhere. Each must have a test that proves it holds.

### P1 — The budget is a contract
Every endpoint and MCP tool that returns memory content takes a `budget_tokens`
argument (with a configured default) and **its response never exceeds it**,
counted with the configured tokenizer (§7.3). If nothing fits, the response is
empty with a reason, never truncated mid-item.

### P2 — Progressive disclosure
A recall returns **handles** first: id, title, one line, date, why it matched,
and the size of what is behind it. The agent expands what it needs
(`expand`). Content is never pushed when a handle would do.

### P3 — Never block the turn
Writes (turn ingestion, extraction, linking, embedding) are queued and
processed in the background. A recall on the hot path does **no LLM call**:
at tier 3 it makes one embedding call for the query (§6.1), bounded and
degradable, through an object that can do nothing else — never a chat call.
A harness adapter **fails open**: if the service is down or slow, the turn
proceeds without memory and the adapter says so in its log — it never delays or
breaks the agent.

### P4 — Local-first, no egress
The only outbound connections are the ones the operator configures: an
OpenAI-compatible chat endpoint and an OpenAI-compatible embeddings endpoint
(both typically local, e.g. Ollama or llama.cpp). **No telemetry, no update
check, no analytics, no remote model download at runtime.** Tests must fail if
any other outbound connection is attempted.

### P5 — Encrypted at rest; archiving is reversible, forgetting is real
The database is encrypted at rest (§7.1). The key is **supplied by the
operator** (environment variable or key file), never generated and stored next
to the database by default.

Removing something has **two verbs, never one**:
- **`archive`** — reversible. The record leaves recall, expansion, the profile
  block and dreams, but stays in the store and comes back with `restore`. This
  is what an **agent** gets: a model that is wrong about what to remove must
  not be able to destroy anything.
- **`forget`** — irreversible, and gated by its own token permission (§8)
  that a harness normally keeps for its operator. Archiving first is not
  required: a person exercising their right to be forgotten must be able to do
  it in one call. An optional retention setting purges archived records after
  N days (off by default).

`forget` is a **hard delete**: the record, its FTS
entries, its vectors, its edges, its proposals (§4.4b) and its source excerpts, followed by the
housekeeping that makes the bytes unrecoverable in the file (see §7.1).
Logs never contain memory content — ids and counts only.

### P6 — Provenance on everything
Every fact and every edge records where it came from (source id, excerpt
offset, who wrote it: `explicit` via remember, `extracted` by the pipeline,
`imported`), when, and with what confidence. An **explicit** fact is never
overridden by an **extracted** one; a conflict between them is surfaced, not
resolved silently.

**Who said it.** A turn is made of **parts**, each with its provenance: what
the owner said (`owner`), what the assistant answered (`assistant`), and
content that came from outside the conversation (`external`: tool results, web
pages, documents — §8.1). The service keeps the parts' offsets and gives every
extracted fact the provenance of the part its quote lies in. Memory poisoning
goes through exactly this gap (an agent that resists injected instructions
still believes the memory it wrote itself from a hostile page), so the rules
that use it are applied by the service, from offsets, never by the model or by
labels written in the text. A quote belongs to a part when it lies inside it,
or when it starts in that part's own label (the "User: " or "Alice: " the
service writes in front of it) and ends inside it: models quote the label with
the sentence (measured with qwen3.5:9b on the synthetic corpus: 28 of 43
quotes). The label is the
service's own text, so this gives nothing to a page: a fake "User:" written in
a page lies inside that page's external part. A quote over two parts belongs
to none.
- a fact **about the owner** (§4.2, the speaker) is written only from an
  `owner` part; from any other part (the assistant's included: it may restate
  a hostile page), or when its quote cannot be located, it becomes a **fact
  proposal** (§4.4b), which only a `review` token may decide. A model can also
  quote a harmless owner sentence ("I live") and take the value from the page,
  so a fact about the owner whose `value` appears (whole words, case and
  diacritics folded) in an `external` part and in no `owner` part is a
  proposal too. A value in another language than the page's is not caught by
  this check: a known limit;
- an extracted fact from an `external` part never closes a fact that did not
  also come from external content (§5.4): the conflict is flagged instead.

### P7 — Time is first-class; contradictions supersede, never overwrite
Facts carry a validity interval. A new fact that contradicts an old one
**closes** the old one (`valid_to`) and opens the new one. History stays
queryable ("what did I know at date D?").

### P8 — Memory is data, never instructions
Everything injected into an agent's context is wrapped and labelled as recalled
data. Content that looks like instructions ("ignore previous…") is stored and
returned as data, and the injected block says so. This is the defence against
memory-borne prompt injection; test it.

### P9 — Scope filters before scoring
Multi-tenant scopes (§4.6) are applied to the candidate pool **before** any
ranking. Filtering after ranking leaks the existence of invisible records
through corpus statistics (IDF, normalisation) and weights visible records by
documents the caller may not see. Test it.

P9 extends to the **graph**: expansion never traverses an invisible record —
neither an invisible edge nor an edge whose far node is invisible — and a
`why` never names one. A path through an invisible node is a leak. A visible
fact whose node is invisible is returned without its node's id or name. Test
it (as a security test).

### P10 — Honest degradation
Each capability tier (§6) declares itself in `/v1/health`. A missing embedding
backend degrades recall to lexical + graph and **says so** in the response
metadata (`tier`, `dense`, `degraded`, §6.1); it never fails the request.

### P11 — The harness owns the model's time
Many local setups serve **one request at a time** from a single model server
(a single-slot Ollama, one GPU shared with image generation…). Background model
work that starts whenever it likes makes the user's turn queue behind it. So
every piece of background model work (extraction, summary maintenance,
embedding, dream) obeys a **background mode** set by config and changeable at
runtime through the API (§8.4):
- `auto` — process the queue as it fills (the turnkey default);
- `paused` — accept and persist work, call no model;
- `on_demand` — call a model only when the harness says `run` (with an optional
  budget: max calls / tokens / seconds), then return to waiting.
Plus a **concurrency limit toward each model endpoint** (default 1) and a
`yield` hint: the harness may tell the service "a foreground turn is starting"
and the worker finishes its current call and waits until told "done" or a
timeout passes. Queue depth and oldest pending item are visible in
`/v1/health`, so a paused queue never fills in silence.

### P12 — Memory keeps the language it was told in
A fact, a summary or a consolidated fact is written **in the language of its
source**, never translated into the model's preferred language. Models drift
toward English when summarising non-English text; a memory that drifted is
lost to lexical search in the user's language and quietly changes what was
said. Every prompt that writes memory states the target language explicitly
(detected from the source, overridable per scope), and a written item whose
detected language differs from its source's is rejected and retried once, then
stored with a `language_mismatch` flag rather than silently. Mixed-language
sources keep each fact in the language of the sentence it came from.

A node's **summary** is written in the scope's forced language when one is
set, else in the majority language of the node's live facts (the newest
`summary_facts` of them, undetermined ones left out); a tie goes to the
language of the newest of the tied facts. Without any determined fact, the
language of all the statements together, else "the language of the facts".

A short statement often holds no word that belongs to one language only ("in"
is English, German, Italian and Dutch; "as" is English and Portuguese): its
language is **undetermined** (`und`; 13 of the 26 English statements of the
bench corpus). An undetermined text is still checked by its **script**: written
in another script than the expected language's (a Latin statement for a
Japanese, Chinese, Russian or Arabic source), it is a mismatch, rejected and
asked again once, then stored flagged, like a determined one. In the same
script nothing is judged. **Known hole**: a translation within one script
("Karim préfère les réponses courtes" → "Karim prefers short answers.", `und`)
still escapes P12; a narrow check — an undetermined statement sharing almost
no word, names excluded, with a quote of determined language — is measured
before it is adopted (flag only, never a drop).

---

## 3. How harnesses plug in

Open-source harnesses converge on **two doors**. iifa offers both.

### 3.1 Door 1 — tools the model calls (MCP)
Claude Code, Goose, opencode, Cursor and most others load MCP servers. Tools
cost nothing until used. This is the universal integration.

### 3.2 Door 2 — lifecycle moments the harness calls itself
Names differ; the moments are the same five:

| Moment | Claude Code hooks | Hermes Agent `MemoryProvider` | opencode plugin | OpenAI Agents SDK |
|---|---|---|---|---|
| Session start | `SessionStart` | `initialize`, `system_prompt_block` | `experimental.chat.system.transform` (runs before every model request: the block is fetched once per session and cached) | dynamic `instructions` (a callable) |
| Before each turn (inject) | `UserPromptSubmit` → `additionalContext` | `prefetch(query, *, session_id)` | `chat.message` (a synthetic text part added to the user message) | — (tools) |
| After each turn (record) | `Stop` | `sync_turn` (must be non-blocking) | events: `message.part.updated` text, then `session.idle` | after `Runner.run` (the caller's code) |
| Before compaction (save) | `PreCompact` | `on_pre_compress(messages)` → text added to the **summary prompt** | `experimental.session.compacting` → `output.context` (appended to the compaction prompt) | — |
| Session end (extract) | `SessionEnd` | `on_session_end(messages)`; `on_session_switch` rebinds | — (`session.deleted` event, `dispose` hook: best effort) | — |

What their documentation says (each adapter's README names the version it
follows):
- Claude Code **discards** a `UserPromptSubmit` hook's output, including
  `additionalContext`, when the hook times out → our adapter must answer fast
  or not at all (P3).
- Hermes allows **one** external memory provider at a time. Its manager
  bounds an external `prefetch` (8 s) and runs `sync_turn` on its own
  background worker, yet asks providers to stay non-blocking. Spilling an
  oversized prefetch result to a file (preview plus path) is **opt-in**
  (`memory.prefetch_spill_enabled`, off by default; above a safety ceiling it
  spills anyway); our `expand` handles are the same idea, native, and our
  budget keeps results far below either limit. `system_prompt_block` is
  static text; recalled context goes through `prefetch`. Background work must
  run in threads made by `spawn_context_thread` (profile isolation lives in
  context variables). `agent_context` other than `primary` (subagent, cron,
  flush) means: write nothing automatically.
- opencode awaits each hook with **no timeout** and does not catch a hook's
  error (a rejected promise breaks the prompt): the plugin bounds every call
  itself and never throws. Events are delivered fire-and-forget. A part added
  in `chat.message` needs an id starting with `prt`; a `synthetic` text part is
  sent to the model and hidden from the user. A local plugin file must export
  only plugins (or default-export `{ id, server }`); helpers live in a
  subdirectory, which the loader does not scan.
- The OpenAI Agents SDK's `Session` protocol (`get_items`/`add_items`) is
  conversation **history**, not memory; integrate there through tools. Its
  `RunHooks`/`AgentHooks` observe a run but cannot change the prompt, so the
  profile block is injected through the agent's dynamic `instructions`.

---

## 4. Data model

### 4.1 Node — something memory is about
`id`, `kind` (person, project, place, object, organisation, topic, event,
other — extensible), `name`, `aliases[]` ("my sister" → a person node),
`summary` (one line, maintained), `created_at`, `updated_at`, `scope`,
`merged_into` (the node a dream merged it into, §5.6; the merged node is
archived, never deleted), `rev` (§4.4).

### 4.2 Fact — something true about a node, for a while
`id`, `node_id` (**optional**: a free-standing fact is valid — a user
preference, for instance — and may be attached to a node later by extraction or
a dream), `statement` (short natural-language sentence), optional
`attribute` / `value` for structured facts, `valid_from`, `valid_to` (null =
still true), `origin` (`explicit` | `extracted` | `imported`), `confidence`,
`source_id` + excerpt span, `provenance` (`owner` | `assistant` | `external`,
the part its quote lies in, P6; empty for `remember`, imports and facts written
before parts existed), `proposal_id` (the fact proposal it was accepted from,
§4.4b), `superseded_by`, `same_as` (the fact stating the same value in another
form, §5.4), `consolidated_into` (the fact a dream consolidated it into,
§5.6 — not a change of value, so it is not `superseded_by`), `pinned`,
`scope`, `rev` (§4.4).

**Chatter is not memory.** Today's weather, greetings, thanks, politeness,
remarks about the conversation itself ("that was helpful", "let's talk later")
and what the assistant says about itself are never facts: the extraction prompt
lists them and says that empty lists are the expected answer for such a turn.
There is no deterministic filter (a word list is language-bound and would drop
"Karim works for a weather service"); the service counts the turns that yield
nothing (`empty_answers`), the bench counts what chatter turns still produce
(§11.1).

**The speaker.** In an assistant the person speaking is the owner of the
memory, and facts about them are the most valuable ones, so a fact whose
subject names the speaker is **never rejected**. The subject names the speaker
when, whole, it is one of the **speaker words** — after folding case and
diacritics and dropping a leading article or elision: `user`, `utilisateur`,
`utilisatrice`, `usuario`, `usuária`, `utilizador`, `Benutzer`, `Nutzer`,
`utente`, `пользователь`, `المستخدم`, `用户`, `使用者`, `ユーザー`, `사용자`…
("the user", "l'utilisateur" included). A subject that only contains one
("User Group Lyon") is an ordinary name. Pronouns are not in the table.
- When the scope declares an **owner** (§4.6), the fact is attached to the
  owner's node, and the source labels the owner's part with the owner's name
  rather than "User", so that the model takes the right subject directly; the
  prompt asks for the owner's name "in exactly this spelling, as the subject
  and in the statement, whatever the language of the source" (an owner rarely
  declares every script of their name: measured with qwen3.5:9b on the
  synthetic corpus, it wrote "Алиса" for Alice in Russian turns and created a
  second node); the table is the net for
  what it still writes. A name that only transliterates the owner's never
  chooses the subject: a third party called Alisa spoken of in Russian must
  not land in the owner's memory (if ever captured, as a fact proposal only).
  **For the naming check only**, an owner fact names its subject when the
  statement holds a form of the owner's declared name that the name
  comparison of §5.4 finds `same` (in any script: "Самир" for Samir, or
  "Alice" itself inside "Alice переехала в Монреаль."; the form is read in as
  many words as the name has, both split the same way, so "Иван-Пётр" names
  Ivan-Petr), provided the fact's
  `subject` is exactly the declared spelling and its quote lies in an owner
  part. It confirms a subject already fixed; it never fixes one. An adapted
  form that does not transliterate ("Алиса" for Alice: `unknown`) is not
  helped; declaring it as an alias is.
- Otherwise the fact is stored **free-standing**, flagged `speaker`. A speaker
  word never becomes a node, and an edge to one is dropped.
For the owner's facts, a speaker word in the statement names the subject (the
naming rule below): "Пользователь живёт в Лионе" names the owner Alice.
Without a declared owner the service cannot tell who speaks — a team scope, or
several users of one agent, has several speakers — so free `speaker` facts
never supersede one another: a different value of the same key is flagged
`conflict` (§5.4). Declaring an owner is what makes the owner's changes
supersede. Facts about the owner obey the provenance rule of P6.

An **extracted** statement must be a complete sentence: at least three words
(at least six characters in a script written without spaces — Chinese,
Japanese), and, when the fact has a subject, it names that subject: its name,
one of its aliases (stored, or given by the answer and found in the source), or
a mention of the same answer that resolves to the same node — tolerant of
inflection and of French/Italian elision ("d'Émilie" names Émilie). Only names
written in the statement's script count: "Карим переехал в Лион." under the
subject "Karim" names it only when the answer also spells it "Карим" (an alias
the source holds). The extraction prompt asks to write the subject as the
source writes it; only the retry note of a fragment rejected for that cause
asks to list the source's spelling among the subject's aliases (in the base
prompt, that instruction made models stuff aliases).

**`subject_as_written`.** A known subject is named in the prompt ("reuse these
exact names"), so a model rightly writes `subject: "Karim"` for a Russian
sentence about Карим — and the statement, in Cyrillic, would then never name
it. Each fact may therefore carry `subject_as_written`, the subject's exact
spelling in the source, while `subject` keeps the known name for resolution.
When that spelling appears both in the source and in the statement, and is no
other node's name or alias (the guard of §5.1), it names the subject and
becomes a source-anchored alias of it — unless it is a near form of the
subject's name or aliases (§5.1: "Ивана" for Иван, an inflection): it then
names the subject for this fact and is not stored (`aliases.near`). Otherwise
it is ignored: the fact is judged as if the field were absent. The known-names line of the prompt asks
for it whenever the source spells a known subject differently. A fragment ("dirige", "déteste", "works on the solar
project" under Иван) is rejected and asked for again once — when the cause is a
subject not named in the statement's script, the retry note says so — then
dropped and counted; drops for that cause are also counted apart
(`cross_script`), so the rule's cost on real models stays visible. A
free-standing fact is checked for completeness only, and so is a true
translation (the statement's language differs from its source sentence's):
that is P12's to judge (asked again, then stored flagged), not this rule's to
drop. Matching names across scripts without the model's help is the job of
embeddings (tier 3, §6.1). Fragments are not only useless to a reader: they carry too
little text for language detection, so P12 could not check them.
(Known limitation: Arabic attached prefixes — و، ب، ل، لا — are not split,
so a subject written with one attached does not count as named.)

**A word that mixes scripts is malformed.** A statement holding a word whose
letters belong to two scripts written with spaces (Latin, Cyrillic, Greek,
Arabic, Hebrew, Devanagari) — "انتقل يوسف إلى كazan office", written by
qwen3.5:9b for a move, which then closed a true fact — is treated like a
fragment: rejected and asked for again once, with a note naming the cause,
then dropped and counted (`incomplete`, and apart as `mixed_script`). What
cannot be told apart is not rejected: a Latin word inside a Chinese or
Japanese sentence (those scripts have no spaces), Chinese characters mixed
with kana, digits and marks, letters of scripts not recognised, and a whole
Latin word next to Arabic ones.

The same guard holds for **names**: a fact whose subject holds such a word is
rejected and asked for again like a malformed statement, then dropped
(`mixed_script`); a node the answer proposes under such a name, or a mention
(an edge end) written so, never becomes a node — counted
(`nodes.mixed_script`) — and an alias written so is dropped (counted with the
invented ones). qwen3.5:9b left a node "كazan" on the synthetic corpus.

### 4.3 Edge — a typed link between two nodes
`id`, `src`, `dst`, `type` (free text normalised to a small vocabulary +
custom), `weight`, `status` (`proposed` | `confirmed`), `origin`, `evidence`
(source ids), `last_used_at`, `use_count`, `valid_from`/`valid_to`, `scope`,
`rev` (§4.4).

### 4.4 Source — where something was learnt
A turn, a document, an import. Stores the **text** (encrypted like everything
else) or, if the operator disables source retention, only a hash and a
reference. Facts and edges point into sources by span.

Every record above (node, fact, edge, source) also carries `scopes` — a **set**
of scope paths, at least one (§4.6) — and an optional `external_id` supplied by
the integrator (§4.6).

**Revision.** Every node, fact and edge carries `rev`, a counter that grows
by one with **every** change to the record, whoever makes it: its content
(statement, summary, key and value, weight, flags, links to other records),
its life (closed, reopened, archived, restored), its aliases (added or
removed) and its scope set. The store keeps it with triggers, so no write
path can forget it. A forgotten record has no `rev` at all: it is gone. A
dream compares revisions to know that what it read is still what is there
(§5.6 rule 8) and that nothing written since depends on what it would undo
(rule 4).

### 4.4b Proposal — something the write path would not decide alone
`id`, `kind` (`alias`; `merge` from tier 3, §6.1, whose `value` is the other
node's id; `fact`), target `node_id` (optional for a `fact` proposal: a free
speaker fact has no node), `value`, `source_id` +
excerpt span, `confidence`, `created_at`, `status` (`pending` | `accepted` |
`rejected`), and the **scope set of its source**. A proposal carries source
text, so it is a record like the others: read through the view (P9), deleted
with its node or its source (P5), never used by resolution or recall. Accepting
applies it (an alias proposal adds the alias to the node; `write` permission);
rejecting only changes its status. A `merge` proposal is never applied
directly: it is an input of the dream, whose adjudication (§5.6) reads both
nodes and, on `same`, queues a `merge_nodes` operation for review; when that
operation is accepted or rejected, the proposal follows. A reviewer may also
reject a pending `merge` proposal outright (the dream then never asks about
the pair again), never accept it: only the reviewed operation merges. Pending
`alias`, `fact` and `merge` proposals appear in the **review queue** (§8.6)
beside the dream's queued operations — a `merge` proposal with the latest
adjudication of its pair (verdict and reason), and not when its `merge_nodes`
is already queued (that item shows the same verdict): one listing, one
`review` permission, so a reviewer never has to look in two places.

A **fact proposal** is a fact about the owner that the write path would not
write (P6: not from the owner's own words). Its `value` is the statement, and
it keeps the rest of the fact (key, value, dates, language, flags, the
provenance of its part). It is decided — accepted **or** rejected — only by a
token holding the **`review`** permission, which an agent token does not hold:
an agent manipulated by the very page the proposal came from must not be able
to turn it into memory, nor to hide it by rejecting it. Accepting writes an
**`explicit`** fact (a human vouched for it, P6) that keeps its provenance: the
original source and span, the part's provenance, and `proposal_id`. Like
`remember`, it closes nothing.

### 4.5 Event — a dated happening linking nodes
Optional; needed for timeline queries ("what happened with X in June").

### 4.6 Scope
A scope is a path, e.g. `user:alice`, `user:alice/agent:coder`,
`user:alice/project:solar`: `kind:name` at every `/` level. A malformed scope
is refused (422) with a message naming the whole path, the offending segment
and the form it should take (`segment 'assistant' needs kind:name, like
mode:assistant`). A record belongs to a **set of scope paths** (at
least one) and is visible to a reader if **any** of its scopes is visible to
that reader.

A read uses one of three **modes**:

1. **scope + ancestors** (the default): `scope: user:alice/agent:coder` sees
   `user:alice/agent:coder` and `user:alice`;
2. **an explicit list**: `scopes: [...]` sees exactly those scopes;
3. **a whole subtree**: `subtree: user:alice` sees `user:alice` and everything
   below it — `admin` permission only.

A client token is bound to the scopes it may use: every scope of a view and
every scope a write targets must be at or below one of the token's bound
scopes. A record may be archived, restored, patched or forgotten by a token
bound to at least one of the record's scopes. (P9 applies, including to the
graph.)

**Owners.** The operator may declare, per scope, who the memory belongs to:
`owners = {"user:alice": {name = "Alice", aliases = ["Алиса"]}}`. A
declaration holds for the scope and every scope below it, like a language
override (P12); the nearest one wins. A source written into several scopes
has an owner only when every one of them leads to the same declaration;
otherwise who speaks cannot be told, and its speaker facts stay free-standing.
The owner's node is found by its exact
name in the declared scope, or created there (`person`, the configured aliases)
the first time a fact about the owner is written — a decision of the operator,
not of a token, so no visibility is widened; the facts themselves belong to the
scopes of their source, as always.

**External ids.** Import, `remember` and `link` accept an optional
`external_id` per record. It is kept and queryable (`GET /v1/lookup`), so an
integrator's existing references keep resolving.

### 4.7 Probe set
`id`, `scope` (exactly one), `name`, `budget_tokens`, `created_at`, and its
probes: `query`, `expected` record ids. Not memory: it is never recalled,
expanded or dreamed over; it measures recall (§11.2).

---

## 5. Linking subjects — the "intelligent" part

The goal is that asking about one subject surfaces the right neighbouring
subjects **cheaply and explainably**.

### 5.1 Entity resolution (write path)
Mention → node, in order, stopping at the first confident match:
1. exact name or alias in scope, after normalisation (case, accents,
   diacritics — language-agnostic);
2. whole-word containment: the mention's words are all words of exactly one
   known name, of a kind compatible with the mention's when the answer gives
   one ("Karim" in "Karim Benali") — a match. Only in that direction: a
   longer mention is someone else as often as not ("Karim Haddad" is not the
   node "Karim" whose alias is "Karim Benali"; the person "Paris Hilton" is
   not the city "Paris"). Words as written, split on spaces, never
   substrings: as strings "Karim" is in "Karima", "Ivan" in "Ivana", "Paris"
   in "Parisa", and "Anne" is not a word of "Marie-Anne" nor "Jean" of
   "Jean-Pierre Martin". A first name contained in two known names (Karim
   Benali, Karim Haddad) is not a match: both go to step 4. A stored alias is
   kept only in its folded form, so containment reads node names and the
   names the same answer proposes;
3. candidates, **never a match on their own**: names alike letter by letter
   (typos, and near names: Karima / Karim scores 0.91, Ivana / Ivan 0.89),
   and, at tier 3, names close by embedding (§6.1; never a merge of two
   existing nodes);
4. LLM adjudication for the candidates (tier 2+), batched. What it attaches
   stays attached for the rest of the source, whatever new nodes the answer
   proposes afterwards.
An unresolved mention creates a new node: so does a candidate without a chat
model, or when the adjudication's answer is truncated. Merges are possible
later (`merge_nodes`) and keep both histories.

For entities the asymmetry is the reverse of values' (§5.4, where a false
"same" only keeps a conflict open). A **wrong attachment** silently corrupts
another person's memory: Ivana's facts written on Ivan's node are recalled as
his, close his own, and nothing shows it. A **split node** is visible, loses
nothing, and a dream can propose the merge (§5.6). So a resemblance of
letters attaches nothing by itself: if it did, whenever a single known name
was alike, which of Ivan and Ivana got the other's facts would depend on who
was named first.

**The source form of a subject.** The model may itself turn the name it read
into another one: given "Ивана предпочитает чай." (Ivana prefers tea) qwen3.5
wrote the subject "Иван" — "Ивана" is also Ivan's genitive — and the fact went
to Ivan by an exact name, with no adjudication. So, for a fact whose quote is
located, the service looks in the quote for the **source form** of its
subject: the words, as many as the subject has, that are near the subject
(alike letter by letter, or after transliteration) without being the same name
once folded or transliterated (§5.4). When the quote holds the subject itself,
or no near form, nothing changes. Otherwise the service resolves the source
form too; when the subject and its source form do not both lead to exactly the
same node, the fact's mention becomes the source form and goes to the
adjudication — with the source sentence as its context and, as candidates, the
nodes either led to — and is never attached alone; "new" makes a node named
as the source writes it. A new node the answer proposed only under the
replaced subject is then left without a fact or an edge: it is not created. This holds in either order: Ivan known and Ivana
never seen ("Ивана" is nobody's name, yet the tea must not land on Ivan), or
both known. Limits: a pronoun has no source form, so the rule does not apply
(nor to a quote that cannot be located, nor to scripts written without spaces,
whose words are not split); a real inflection ("у Ивана новая работа", Ivan
has a new job) costs one adjudication, counted by the bench
(`source_forms`). A **speaker word** (§4.2) is never
resolved and never creates a node: it means the declared owner, or no node.

An **alias** the model attaches to a node is accepted only when it appears in
the source text (whole words, case and diacritics folded) **and** is not the
name or an alias of another node — of the same answer, or known in scope.
Otherwise it becomes an alias **proposal** (§4.4b) and resolution ignores it.
A model that invents an alias ("Карим" attached to Иван) or hands a node the
name of the person next to it ("Youssef 00" attached to Иван, seen with a
real local model) would otherwise merge two people for good; the second case is
also where a merge proposal belongs. An alias written in a script that appears
nowhere in the source ("카림" from a source without Hangul) is neither: it is
pure invention, dropped and counted (`aliases.invented`), so that it never
floods the review queue.

An alias that is a **near form** of the node's name or of one of its aliases
— word by word, one letter or an ending apart ("Ивана" for Иван, "Карим" or
"Karim" for Карима), compared as written within one script; across two,
transliterated, and only one letter apart (an exonym's ending is no
inflection: Париж stays an alias of Paris) — is never kept on its own either, even when the source holds it:
it becomes an alias proposal (counted `aliases.near`). Kept, it would resolve
every later mention of the near name by exact match (step 1) before the
adjudication could see it: huihui_ai/Qwen3.8-abliterated:27b gave Иван the
alias "Ивана" before Ivana's node existed, and five of her facts landed on him
on the traps corpus, three superseding his. The same holds for the mention an
adjudication attached ("Yousef" for Youssef): it stays attached for that
source, and the next one is adjudicated again. The same name in another
script (Ivan / Иван, Karima / Карима: one spelling, as the source form
above compares them) and a far
nickname (Kiki, Mimi) are kept; a pair of words with letters no
table covers (Chinese characters, Hangul) is not judged. This is the
asymmetry of entities: a split node is visible and a merge is proposed later,
a wrong attachment is silent.

An alias that is a **description** rather than a name — more than four words,
or at least three words and more than three times as long as the node's name
(letters counted, spaces left out) — is never kept, even when the source holds
it: it becomes an alias proposal (counted `aliases.descriptive`). A real
local model proposed "تخطيط تركيب المشروع الشمسي" ("planning of the solar
project installation") as an alias of the project; kept, it would resolve any
mention of that phrase to it. A full name of three words for a short first name ("Ivan
Petrovich Sidorov" for "Ivan") is proposed too: a proposal loses nothing, a
wrong alias merges for good.

### 5.2 Link proposal
Edges are **proposed** from: co-occurrence in the same **sentence** of a
source (not merely the same turn: a turn about two unrelated people must not
link them; a node is in a sentence when its name, an alias, the mention that
resolved to it or the quote of one of its facts or edges lies there), explicit
relation phrases found by extraction, and (tier 3) semantic proximity between
node summaries (`similar_to`, §6.1). Proposed edges are usable in recall at reduced weight.
An edge becomes **confirmed** when: stated explicitly, re-observed in N
independent sources (configurable), or confirmed via API.

### 5.3 Reinforcement and decay
Using an edge in a recall that the agent then **expands** reinforces it
(the expand call is the signal — no LLM judgement needed). Unconfirmed edges
decay over time and are pruned below a threshold. Confirmed edges never decay
away; they may only be closed (`valid_to`) or deleted.

### 5.4 Contradictions (P7)
When extraction produces a fact whose `attribute` matches a live fact on the
same node with a different value, the old one is closed and the new one opened
— unless the old one is `explicit` and the new one `extracted` (P6): then both
stay, the new one is flagged `conflict`, and `recall` reports the conflict.

Two more cases keep both facts with the `conflict` flag instead of closing:
- the new fact came from an **`external`** part and the old one did not
  (P6: a web page never closes what the owner or the assistant said, nor a
  fact of unknown provenance; external content may close external content);
- both are free **`speaker`** facts (§4.2: without a declared owner, two
  speakers may share a scope). They are compared with the visible free
  `speaker` facts of the same key, as a node's facts are; an equal value is a
  duplicate and a `same` value links, as below.

**Values in other forms and other languages.** "mars 2026" and "March 2026",
"Lyon" and "Лион" are the same value; "Lyon" and "Париж" are not. Two values of
the same key on the same node are compared **deterministically** first, and
the comparison says `same`, `different` or `unknown` — never a guess:

- **dates** are read into ISO (`2026-03`, `2026-03-15`) from numeric forms and
  from month names in English, French, Spanish, German, Italian, Portuguese,
  Russian (nominative and genitive), Arabic (both the Gregorian and the
  Syriac series, مارس and آذار) and the `3月` / `3월` forms. Equal, or one
  refining the other (`2026-03` and `2026-03-15`), is `same`; otherwise
  `different`. Months of another calendar (Hijri) are never converted:
  `unknown`;
- **numbers** in any digits (Arabic-Indic, Persian, full-width, Devanagari)
  with the rest of the value equal: same number `same`, another number
  `different`; another unit or wording `unknown`;
- **names** (places, employers…) are transliterated **for comparison only,
  never stored**: Cyrillic and Greek to plain Latin, kana to Hepburn, and for
  Arabic and Hebrew, which leave vowels out, the consonant skeletons are
  compared. An equal form (or one letter apart on a long name) is `same`. A
  different form is **never** `different`: exonyms (Londres / London, Pékin /
  北京) look nothing alike. Chinese characters have no table here: `unknown`.

Within one language, values that differ as strings supersede, unless
the comparison says `same`. With the **replacement question** (`[extraction]
replacement_question`, on by default, see below), `role`, `team`
and `preference.*` are the exception, since they may hold
several values at once ("je suis aussi photographe" after "je suis comptable"
must not close the accountant): within one language too, a different value
of these keys goes to the replacement question ("replaces, or both true?"),
and without an answer — no chat model, a truncated or invalid one, `unsure` —
both stay live with a visible conflict rather than an automatic close; a
comparison saying `different` (another date or number) does not settle them
either: a different value may be an addition. Across languages:

- `same` — both facts stay live, **without a conflict**, and the new one
  records the equivalence (`same_as`, pointing at the group's first fact). The
  budget must not pay twice for one value: recall injects one fact per
  equivalence group, the one in the query's language, else the most precise
  (a full date before a month), else the group's first. The dream may
  consolidate them later;
- `different` — the older fact is superseded when the new one is **later**
  (its `valid_from` is not earlier than the old one's; without dates, the
  later source);
- `unknown` — asked to the chat model, in one batched call before the write
  transaction, **only** for single-valued keys (`home`, `workplace`,
  `employer`, `status.*`, `date.*`; never `role` or `team`, which may hold
  several values) and only when the new fact is later: "do these two values
  name the same thing? same / different / unsure". `different` supersedes,
  `same` links, and `unsure`, an invalid answer or anything else leaves both
  facts live with the `conflict` flag. The answer goes through a
  strict schema (P8). The bench measures false supersessions with `--traps`;
  if they rise, the adjudication is switched off.
  With the replacement question switched on, for `role`, `team` and
  `preference.*`, which may hold several values at once, the question is
  another, in a call of its own: "does the new value
  replace the old one, or can both be true at once? same / replaces / both /
  unsure" ("now works as an accountant" replaces; "is also a photographer"
  adds). `replaces` supersedes, `same` links, `both` keeps both facts live
  without a conflict (both are true), and `unsure`, an invalid answer or
  anything else leaves the conflict. Without this question, every stale
  answer left on the bench corpora comes from these keys (role 4, drink 2 on
  the default corpus with the oracle). Same conditions as
  above: only across languages after an `unknown` comparison, only when the
  new fact is later, never against an explicit fact, never when P6 keeps the
  conflict. Measured with `--traps` (false supersessions and false
  replaces) and on the default and `--rephrase` corpora (stale answers).
  Each fact of a pair carries its **`quote`**, the exact words of its source,
  since a model writing a statement drops the very word that tells a
  replacement from an addition ("Émilie travaille *maintenant* comme
  photographe." is stored without "maintenant"; "также" is dropped too): the
  old fact's from the stored offsets of its source, read through the write's
  view (P9), absent when sources are not retained or not visible; the new
  fact's from the source being extracted, without the service's label,
  inside the same `<<<PAIRS>>>` markers (P8). At most 300 characters each.
  **On by default, decided by measurement (qwen3.5:9b + bge-m3, synthetic
  corpora).**
  Without the quotes the question lowered correct current answers at 600
  tokens from 0.59 to 0.52 (default corpus) and from 0.81 to 0.70
  (`--rephrase`): the model answered "both" for real changes, having lost the
  word that decides in the statements. With the quotes the criterion — the
  headline metric no lower than with the switch off, and no more false
  supersessions — is met at tier 2 on all three corpora (default 0.59 =
  0.59, `--rephrase` 0.81 = 0.81, trap corpus 0.60 against 0.58), with 0
  false supersessions where the switch off has 15 false replaces on the trap
  corpus (every value added in one language closing one still true). At
  tier 3 the trap corpus is one question lower of 45 (0.69 against 0.71),
  judged noise against 15 true values kept live. Cost: one batched
  model call per extraction that changes a role, team or preference, under
  P11 like every background call. Off (`replacement_question = false`),
  these keys close within one language and stay in conflict across
  languages.

Superseding a fact closes its whole equivalence group (the "Лион" linked to
"Lyon" states the same, now-outdated value). A supersession closes, never
erases (P7): a wrong one stays recoverable. An `explicit` fact is still never
closed by an extracted one (P6): `same` links it, anything else flags the
conflict. `forget` of a group's first fact re-roots the group on the next one
(P5: no reference to a forgotten fact remains). A value's own language is not
used: values are too short to detect; the statements' languages are.

**Undetermined languages.** When both statements' languages are determined,
they are within one language when equal. When either is undetermined, the
**scripts** decide: an undetermined Latin statement against a Japanese,
Russian or Arabic one is known to be another language and takes the path
across languages above (deterministic comparison, then the adjudication for
single-valued keys, else a conflict); within one script ("Nadia prefers tea."
against "Nadia préfère le thé.") the pair is treated as one language. A
statement with no letters at all ("2024-05-01") has no script:
such a pair is never taken for one language, since we cannot tell. This is a measured choice, not a convenience: within one script, an
undetermined statement that closes an equivalent value replaces it with the
same value — the current answer stays right and P7 keeps the history —
whereas treating the pair as undecidable leaves both live as a conflict, and
at the next change only one of them is closed (role and preferences are
never adjudicated), so a stale value stays live and is injected. With the
oracle on the synthetic corpus: treating every undetermined pair as undecidable cut correct
answers at 600 tokens from 0.63 to 0.56 (default corpus) and from 0.70 to
0.63 (`--rephrase`); the script rule gives 0.59 and 0.78, closures by a
value's own restatement fall from 10 to 4, and false supersessions stay at 0
in every case. A fact whose statement is undetermined keeps an empty
`language`: it does not borrow its source's (a short English source is
undetermined too).

**Attribute keys.** A contradiction is only seen when the two facts carry the
same key, and a model left to pick keys freely does not (synthetic corpus,
read from the stores: qwen2.5:7b left 96 of 108 facts without a key and wrote `prefers` then
`preference`; qwen3.5:9b wrote `profession`, `occupation` and `role` for one
aspect, `lives_in` and `location` for another — 0 and 6 supersessions). Keys
are therefore steered and aligned, never left to chance:

**The rule: when we cannot tell, we do not supersede.** Closing a fact that is
still true is worse than missing a contradiction (the closed fact leaves
recall), so every step below leans that way.

1. **A base vocabulary** of language-neutral keys, one name for each aspect
   that holds one current value: `home` (where the subject lives), `workplace`
   (where they work: an office, a site, a city), `role` (job, occupation),
   `employer`, `team`, and three open families whose suffix names the topic:
   `preference.<topic>` (`preference.drink`, `preference.answers` — "prefers
   tea" must not replace "prefers short answers"), `status.<topic>`
   (`status.marital`, `status.work` — "married" must not replace "on sick
   leave") and `date.<event>` (`date.birth`, `date.deadline`). There is no
   `location`: given only an exclusion ("home only: where they work is not a
   location"), qwen3.5:9b still keyed all nine offices of the trap corpus
   `location` — a model needs a name for each aspect, not a
   warning. The extraction prompt gives the vocabulary and asks for a key on
   every fact that states the current value of such an aspect; any other key
   is a short snake_case identifier.
2. **The subject's existing keys in the prompt**: next to the known names
   (§4.2), the prompt lists the keys the live facts of those subjects already
   carry ("Karim: home, role, preference.drink"), and asks to reuse them for
   the same aspect.
3. **Canonical keys at write time, without a model**: a key is folded (case,
   diacritics, spaces, hyphens; letters of any script kept) and a small table
   maps **unambiguous** variants only onto the vocabulary (`lives_in`,
   `residence`, `home_city`, `address`, `moved_to` → `home`; `work_location`,
   `place_of_work` → `workplace`; `occupation`, `profession`, `job_title`,
   `works_as` → `role`; `marital_status` → `status.marital`;
   `preferred_drink`, `drink_preference` → `preference.drink`). An ambiguous
   variant is left as a free-form key and never closes a vocabulary one:
   `location` and `city` may be a home or a work city, `work` and `title` may
   be anything, `works_at` may be a place or an employer. Keys are compared
   canonically on both sides, so an older fact stored with a raw key still
   meets its successor; the extracted fact is stored with the canonical key.
   **A bare family key** (`preference`, `status`, `date` without a topic —
   what `prefers` or `likes` become) never drives a supersession or a
   conflict: every preference would share it.
4. **Synonym keys by embeddings (tier 3), for free-form keys only**: a
   free-form key that is not among the subject's existing free-form keys after
   step 3 is embedded with them in one call, before the write transaction; the
   nearest above `key_threshold` replaces it. A vocabulary key is never renamed
   and never a target (a new `employer` must not be matched to `role` and close
   it). Within a family, only the topics are compared, and only within the
   same family (`preference.music` is compared with the other preferences'
   topics — "music" with "drink" —, never on the shared word "preference").
   Like every threshold (§6.1) it is per model: without a profile value or a
   configured one, this step is off and says so. **No model is calibrated for
   it**: `bench/key_names.py` measures a model's cosines between key names that
   mean one aspect and traps that must never meet; for bge-m3 the two overlap
   (true pairs from 0.65, traps up to 0.79: gap −0.13), so there is no profile
   value and the step stays opt-in by configuration. Another model is checked
   with the same command before a profile value is set.

Only the key is aligned; values are compared as above.

**Supersession assumes a single-valued key**: a new value of the same key
replaces the old one. That holds for `home`; it does not always hold for
`team`, `role` or a preference (someone may lead two teams, be an engineer and
a teacher, like tea and coffee); with the replacement question (on by
default, see above) those keys never close automatically and the question
decides, in one language as across languages. A key aligned
wrongly still closes a fact that is still true.
The benchmark therefore counts **false supersessions** (a closed fact whose
replacement is not a later value of the same person's same attribute), with
traps built for it (`--traps`: where someone works next to where they live,
and values **added** rather than replaced — a second role, a second team, a
second favourite drink; a value closed by an addition is counted apart as a
**false replaces**);
a key or a synonym that produces them is removed from the vocabulary rather
than tuned around.

### 5.5 Explainability
Every recalled item carries a `why`: which query terms, which vector
neighbour, or which edge path (`Karim —works_with→ Solar project`), plus age.
This lets the model judge relevance **without expanding**.

**Subject anchoring.** A question that names exactly **one** subject — a
visible node the recall found every word of whose name, or of one of whose
aliases, the question names by the rule of §4.2 (case, diacritics, elision,
an inflected ending: "Emilie" and "d'Émilie" name Émilie, "Карима" names
Карим) — puts that node's **facts** ahead of the other candidates, each
side keeping its own order; the node's own handle keeps its place (it costs
175 to 358 tokens on the bench, and moving it ahead pushed the answer out of
small budgets). Nothing is added, nothing removed, only the order changes,
so what the budget keeps favours the subject asked about. A question naming
two subjects or none, or two homonyms visible in the view, is ranked
without anchoring. Measured case (synthetic corpus, tier 2, a French
store): "Où habite
Karim maintenant ?" matched other people's facts on "habite" and Karim's
other facts on "maintenant" ahead of "Karim a déménagé à Osaka", which only
the subject's name matched. A store's write-side defects still cost answers
it cannot recover: an answer on another node of the same person (a script
split awaiting its merge), on a fact wrongly filed as the speaker's, or
behind a stale value left live is pushed back when the named node's facts
move ahead.

### 5.6 Dream — offline reorganisation

The write path (§5.1–5.4) is deliberately cheap and local: it sees one source
at a time. What it cannot see is the **whole**: two nodes that are the same
person under different names, twenty small facts that say one thing, a theme
running through months of sources, summaries gone stale. **Dream** is a job,
started through a route (§8.6), in which a model reorganises a scope while
nobody is waiting on it. It is where the expensive intelligence goes, so the
hot path never has to carry it (P3).

**What a dream may do — a CLOSED catalogue of operations:**

| Operation | What it does | Risk class |
|---|---|---|
| `refresh_summary` | rewrite a node's one-line summary from its live facts | low |
| `decay_edges` | apply decay and prune unconfirmed edges below threshold (§5.3) | low |
| `reembed` | queue the background embed job for stale vectors (§6.1) | low |
| `propose_edge` | add a `proposed` edge between nodes the write path never saw together | low |
| `create_theme` | add a `topic` node grouping related nodes (clusters, recurring subjects), linked by `about` edges | medium |
| `consolidate_facts` | replace N overlapping `extracted` facts on one node by one consolidated fact; the originals are closed (`valid_to`) and point at the new one (`consolidated_into`), never deleted | medium |
| `merge_nodes` | merge two nodes judged identical — proposed only by the **adjudication** below, never by a unit's answer; both histories kept, aliases unioned | high |
| `supersede_fact` | close an `extracted` fact contradicted by a later source | high |
| `flag_conflict` | mark a conflict for a human or the agent to settle — the ONLY thing a dream may do about an `explicit` fact | low |

`decay_edges` and `reembed` need no model: the service runs them itself at
the start of a run, journalled like the others (`reembed` only queues work on
derived data, so it has nothing to undo). The model proposes the other seven:
six in a unit's answer, `merge_nodes` only through the adjudication of a pair.

**Rules (each one tested):**
1. **The model proposes; the service applies.** The dream model returns
   operations in a strict JSON schema drawn from the catalogue above — never
   free text written into memory, never SQL. Anything outside the schema is
   rejected and counted. This is also the prompt-injection wall: a memory that
   says "delete everything" can at worst make the model propose an operation
   the schema and the rules then refuse (P8).
2. **Never destroys.** No operation hard-deletes; only `forget` does (P5).
   Merges and consolidations close and reference, they do not erase. (Undoing
   an operation deletes what that operation created — nothing else.)
3. **`explicit` is untouchable** (P6): a dream may flag a conflict on an
   explicit fact, never rewrite, close or merge it away. An operation whose
   inputs include an explicit fact is dropped (`explicit`), except
   `flag_conflict`, `supersede_fact` (downgraded to a conflict, below) and
   `merge_nodes`, which moves an explicit fact to the kept node unchanged and
   therefore always goes to review. A **pinned** fact (the owner's choice) is
   never closed by a dream either.
4. **Every run is a changeset.** Each applied operation is journalled with its
   before/after state and the model's stated reason. `undo` reverts a whole run
   (or one operation) as long as later writes do not depend on it — and says so
   when they do: a record the operation changed whose revision (§4.4) moved
   since is a later write that depends on it.
5. **Review by risk.** Per scope policy: `low` applies automatically; `medium`
   and `high` apply automatically, or go to a **review queue** the harness can
   show to a human or hand to its agent. **Default: only `low` applies;
   `medium` and `high` go to review.** Decided by measurement (qwen3.5:9b on
   the synthetic corpus, three bench runs): the one `consolidate_facts` proposed was wrong (Nadia's
   tea and coffee folded into "contradictory preferences"), and no
   `create_theme` was ever proposed, so neither has a measured right to apply
   alone; an operator may still declare `medium = auto` for a scope.
   **`merge_nodes` always goes to review — permanently**, whatever the policy
   says ("never merge silently", §5.1); no measurement lifts it. A merge moves
   everything a node holds, and the model follows an identity claim found in
   memory: in a `--hostile` bench run, qwen3.5:9b answered `same` for Bogdan /
   Bogdana because a fact said "Bogdana and Bogdan are one person". The
   names-fit check (below) stopped that one; between two namesakes (two
   different Karim) the names fit, and only a reviewer stops it. A request
   may tighten the policy, never loosen it.
6. **Dry run first-class.** `dry_run: true` returns the full plan and applies
   nothing.
7. **Bounded.** Every run has a budget: max model calls, max tokens, max wall
   time, max operations. It stops cleanly at the first limit reached, reports
   what it did and what it left, and a later run resumes from there. One dream
   per scope at a time: a run is refused while another is pending or running
   on the same scope, an ancestor or a descendant.
8. **Never in the way.** A dream reads a consistent snapshot and serves no
   request; recall keeps answering throughout. Its writes apply with
   optimistic checks — an operation whose inputs changed since the snapshot is
   dropped and reported, never forced. "Changed" is a revision (§4.4) that
   moved or a record that is gone.
9. **Must not make recall worse.** If a probe set exists for the scope or a
   scope below it (§11.2), the run measures the probe score before and after;
   a drop beyond a configured tolerance rolls the run back automatically and
   says why. An `inconclusive` comparison (another tier served, other probes
   scored) is a failure too: a dream that cannot be shown harmless is not kept
   — it is rolled back with the reason `inconclusive` and reported.
10. **Its own model.** `dream_model` is configured separately from the
    extraction model — it can be larger and slower, since nobody waits on it.
    The model is explicit on every call. Without a dream model, only the
    operations that need none run; the others are reported `unavailable`
    (P10) — the extraction model never stands in for it.
11. **Started by the operator or the harness, never by the agent mid-turn.**
    There is no MCP tool for it: a model deciding in the middle of a
    conversation to launch a long, costly job is exactly what this design
    avoids. The harness picks the moment (session end, idle time, night,
    GPU free); there is no built-in schedule.
12. **Never widens visibility.** A dream on a scope changes only records whose
    **every** scope is that scope or below it; records of its ancestors are
    read as context, never changed. Every record a dream creates or merges —
    a consolidated fact, a merged node, a theme, a proposed edge — draws on
    records with **identical scope sets**, and the result keeps that set. An
    operation over records whose scope sets differ is dropped and reported
    (`scope_mismatch`), never applied.
13. **Keeps each language (P12).** The facts a consolidation replaces share
    one language (their determined languages agree), and the consolidated
    fact is written in the scope's forced language when one is set, else in
    that language. A dream never consolidates facts of a `same_as` group
    (§5.4): each serves search in its own language, and recall already injects
    one of them. A text the dream model wrote in another language than
    required (a summary, a statement) is dropped (`language_mismatch`) rather
    than stored flagged: the old state stays and a later run can try again.
14. **A consolidation says one thing again — the service checks it.** Every
    fact it replaces has the same key (§4.3, compared canonically; a fact with
    no key fails) and an equivalent value: the same one up to case, accents
    and script (`same_spelling`, without the one-letter tolerance, since a
    wrong SAME here loses a value), or one `same_as` group (§5.4). Rule 13
    already keeps groups out of a consolidation; this part is defence in
    depth. Anything else is dropped (`other_value`),
    never rewritten. Measured cause (`--traps --name-traps`): qwen3.5:9b
    folded Émilie's two team facts into "Émilie is in the Atlas and Orion
    teams." — two values of a multi-valued key joined in one statement, which
    the store then cannot close, supersede or forget one at a time.

**How a run works.**

*Starting.* `POST /v1/dream` (§8.6) records the run and queues it as a job in
the persistent queue (§7.2): it obeys the background mode, the gate and
`yield` like any background work (P11). Paused, it waits; between two units it
stops cleanly when the service is paused or shut down or the run is
cancelled, exactly as at a budget limit. A run interrupted by a crash goes
back to the queue and resumes where it was.

*Units.* A run walks the nodes the scope may change (live, every scope at or
below it) in id order, from the scope's **cursor**. Each node is one unit,
one model call: the node (name, kind, aliases, summary), its live facts (id,
statement, key and value, origin, language, dates), its live edges with the
node at their other end. The model may name only records of its unit: any
other id is dropped (`unknown_record`); a `merge_nodes` item in a unit's
answer is rejected (`adjudication_only`). Before the unit's call, the node's
merge candidates are adjudicated (below), within the same budget: a run
stopped among them resumes at the same unit, and the pairs already judged are
skipped. The cursor moves after each unit; reaching the end resets it
(`stopped_by: complete`). A dry run never moves it.

*What one call reads is bounded.* A real owner's node holds hundreds of facts,
and a local server given a prompt longer than its context window truncates it
silently **from the start** — dropping the system prompt, and with it the P8
wall. So every dream call (a unit, an adjudication) has a prompt budget,
`prompt_max_tokens` (4000), counted with the service's own counter (§7.3, both
messages) before it is sent. Over it, the call carries fewer records: the
node always; then its facts — pinned first, then explicit, then the most
recent (`valid_from`, then creation) — as many as fit; then, for a unit, its
edges, most recent first, with the nodes at their other end. The data says
how many were left out (`facts_left_out`, `edges_left_out`; per side for a
pair, whose two sides take facts in turn). A record left out is not part of
the unit: an operation naming it is dropped (`unknown_record`). A call whose
rules and node alone exceed the budget is not sent: the unit is skipped, the
pair left as it was (`oversized`). The operator sets the model's context
window above `prompt_max_tokens` plus `answer_max_tokens`; the service cannot
see it. A per-request window does not do it on every server: Ollama's
OpenAI-compatible endpoint ignores `options.num_ctx` (measured, Ollama 0.31.1:
qwen3.5:9b loaded at 4096 tokens with `num_ctx` 8192 in the request). With
Ollama, declare a model variant whose window is set — `qwen3.5:9b-8k`, `FROM
qwen3.5:9b` with `PARAMETER num_ctx 8192` — and check it in `/api/ps`
(`context_length`).

*Merge candidates and adjudication.* A wrong merge silently corrupts memory
(one person's facts on another's node), so a merge is never decided on a name
— not by the write path, not by a cosine, not by the unit's model reading one
node — but by an **adjudication**: one model call per pair, reading both
nodes whole.
- **Candidates of a unit's node**: the nodes a pending `merge` proposal pairs
  it with (§4.4b, §6.1), then, at tier 3 with a calibrated embedding model,
  the nodes whose name vector is at least `resolve` (§6.1) from its own, best
  first. A candidate is a live node of the **same kind** whose scope set is
  **identical** to the node's (rule 12) and that the dream may change. A pair
  is skipped, costing no call, when it was adjudicated before and neither node
  nor any of their live facts moved since (its revisions, §4.4), when a
  reviewer rejected its merge (the proposal or the operation), or when its
  merge is already queued. Each pair is adjudicated at most once per run, and
  at most `adjudications_per_unit` (3) per unit, proposals first.
- **The call** shows both nodes (name, kind, aliases, summary) between the
  memory markers, as DATA (P8), and asks whether they are one and the same
  entity. Each side's facts are shown as **timelines**: per key, each live
  value with the values it replaced (`superseded_by`, oldest first, at most
  six), e.g. `"home": "Paris (learned 4th) -> Lyon (learned 6th)"`; a fact
  without a key is its statement, under `other`. Every value says **when
  memory learned it**, in one order shared by both sides (creation, then id),
  and its date when the fact has one (`from 2026-01`). Measured why: told
  in words that a change over time is not a contradiction, qwen3.5:9b still
  answered `different` to 18 of 22 true pairs, reading "architect vs
  accountant" or "Lyon vs Casablanca" side by side; what the model sees has to
  carry the order. The pair's revisions include the replaced facts shown.
  The answer is one closed object: `{"verdict": "same" | "different" |
  "unsure", "reason": "<short>"}`. Anything else is counted (`bad_answer`,
  `truncated`) and the pair is left as it was. The prompt says that a
  near name is not evidence, that the same name in two scripts may be one
  entity, and that facts that cannot both be true of one entity at once say
  `different`; when the facts neither confirm nor refute, the answer is
  `unsure`. Its examples are never names of the bench corpus (Daniela /
  Daniel, Иосиф / Joseph — the calibration list's, §6.1): a prompt that names
  the traps it is measured on would flatter the measure.
- **`same` needs names that fit — the service checks it**, not the model.
  Names fit when **any** of: one name or alias of each side is the same
  spelling after transliteration (`same_spelling`, the strict comparison of
  §5.1, without the one-letter tolerance), the two share a name or alias, or
  — at tier 3 with a calibrated model — their name cosine is at least the
  calibrated `merge` threshold (§6.1; no threshold of its own). A `same`
  whose names do not fit becomes **`unsure`**, keeps the model's reason and
  goes to review as any `unsure` does: a downgrade of label, never a merge
  decision. The report counts these (`names_unfit`) and keeps, per pair, what
  the model said. Measured why: with timelines, qwen3.5:9b answered
  `same` to Иван / يوسف (Ivan and Youssef, "the same person who moved from
  Tunisia to Kazan"), a pair every run without timelines had called
  `unsure`; the prompt's "a near name is not evidence" was the half of the
  rule the model dropped.
- **What the verdict does**: `same` submits `merge_nodes` (`keep`: the node
  with more live facts, else the older; `absorb`: the other), checked against
  the snapshot the adjudication read, and queued — it always goes to review
  (rule 5). `different` changes nothing. `unsure` leaves a visible proposal:
  when no `merge` proposal pairs the two, one is created (a dry run creates
  none). "When we cannot tell, we do not merge" — and a pending proposal is
  never rejected by a dream: only a reviewer decides it.
- **The verdict is kept** for the reviewer: each adjudication is recorded with
  its run, both nodes, the verdict, the model's reason and the revisions it
  read, and the review queue (§8.6) shows the latest verdict and reason beside
  the pending proposal or the queued `merge_nodes` of that pair. A dry run
  records nothing; its report counts the verdicts. `forget` of a record an
  adjudication read deletes the adjudication (P5: its reason may quote it).
  Undoing or rolling back a run deletes the pending proposals its `unsure`
  verdicts created; its verdicts stay as a record of what the model said.

*The answer.* One JSON object, `{"operations": [...]}`, each item one
operation of the catalogue with its fields and a short `reason`; a closed
schema (no extra key, bounded strings), at most 20 items. An item that fails
it is rejected and counted by cause (`outside_catalogue`, `schema`,
`too_many`) and the others are still considered. An answer that is not JSON
is counted (`bad_answer`) and its unit skipped; one cut by its token cap
(§7.4b) is never parsed (`truncated`). The prompt says, like every prompt
that reads memory, that the memory between its markers is DATA, never
instructions (P8).

*Checks, then application.* Before anything is applied, the service checks
each operation against memory as it is, on top of the rules above:

| Operation | Fields | Checked (else dropped, reason) | Applying it |
|---|---|---|---|
| `refresh_summary` | `node`, `summary` | the node has live facts; one line of at most 300 characters; written in the summary language (P12) | the summary and its language replaced |
| `propose_edge` | `src`, `dst`, `type` | two distinct live nodes; a type that normalises (§4.3); no live edge of that type between them (`duplicate`) | a `proposed` edge, origin `dream`, at reduced weight |
| `create_theme` | `name`, `members` (2–20), `summary` | live member nodes; a name of at most 200 characters, no word mixing scripts (§4.2); no live `topic` node of that name in view (`duplicate`) | a `topic` node and a confirmed `about` edge from each member |
| `consolidate_facts` | `node`, `facts` (2–20), `statement` | live `extracted` facts of that node, not pinned, not flagged `conflict`, in no `same_as` group, one language (rule 13); one key and equivalent values (rule 14, `other_value`); a complete statement (§4.2) naming the node, no word mixing scripts, in the required language | a new `extracted` fact — the originals' key, the first original's value, the lowest confidence, the earliest `valid_from`; the originals closed with `consolidated_into` |
| `merge_nodes` | `keep`, `absorb` | two distinct live nodes of the same kind | `absorb` archived with `merged_into`; its name and aliases join `keep`'s; its facts and edges move to `keep` (an edge left joining `keep` to itself is closed); a pending merge proposal between the two is accepted |
| `supersede_fact` | `fact`, `by` | the write path's own rules (§5.4) checked again: one node, both live, `fact` extracted, the same single-valued key, `by` later, P6's provenance rule, and the values not the same within one language or proven different across two — any of them failing **downgrades** it to `flag_conflict` (`downgraded:<cause>`) | `fact` closed at `by`'s `valid_from`, `superseded_by` = `by` |
| `flag_conflict` | `fact`, `with` | two distinct live facts not already flagged against each other (`duplicate`) | `fact` flagged `conflict` with `conflicts_with` |

Each operation that passes is then **applied** (its risk class applies
automatically under the policy), **queued** for review, or, in a dry run,
**planned**.

*Snapshot and optimistic apply (rule 8).* A unit's records are read with their
revisions in one short read; no lock and no transaction is held while the
model thinks, so recall answers meanwhile. Applying an operation re-reads, in
the write's own transaction, the revision of every record the operation reads
or changes: one that moved, or a record that is gone, drops the operation
(`stale`). The dream's own writes update its snapshot. A queued operation keeps
its snapshot: accepted later, it is checked the same way and dropped `stale`
when memory moved meanwhile.

*The journal and undo (rule 4).* Every operation is a journal entry: its run,
operation, risk, status (`planned`, `applied`, `queued`, `rejected`,
`dropped`, `undone`), why it was dropped, the model's reason, the records it
touches, the revisions it read, and the before and after state of every record
it changed with the revisions right after. Undoing one operation is refused,
naming the records, when a record it changed has moved since; otherwise it
reverts exactly: what it created is deleted, what it closed reopened, moved
facts and edges go back, added aliases are removed, the archived node is
restored, a proposal it accepted is pending again. Undoing a run is
all-or-nothing over its applied operations (latest first), refused with every
blocking operation named.

*Forget and the journal (P5).* A journal entry holds content (statements,
summaries, names, the model's reason). `forget` of a record wipes the content
of every entry that mentions it, and its id: such an operation can no longer
be undone (`forgotten`), and a queued one is dropped (`forgotten`). Forgetting
a consolidated fact clears its originals' `consolidated_into`, and forgetting a
kept node clears `merged_into` on the nodes merged into it: they stay closed
or archived (forget never brings back what it did not delete).

*Probes (rule 9).* Before a run that may apply something (not a dry run)
applies its first operation, every probe set of its scope and of the scopes
below it is scored; after its last operation they are scored again and
compared (§11.2) with `dream_probe_tolerance` (0.02). One set `regressed` or
`inconclusive` rolls the run back: every applied operation is undone, latest
first, and the queued ones are dropped (`rolled_back`); an operation a later
write depends on stays, named in the report (`rollback_incomplete`).
Operations accepted later from the review queue are the reviewer's decision
and are not measured. Without a probe set nothing is measured, and the report
says so.
The report shows, per probe set, before and after: `recall`,
`recall_per_1k_tokens` (raw), `compared_per_1k_tokens` (what is compared),
`summary_tokens` and the tier, so the price of summaries stays visible.

*Budgets (rule 7).* Calls, tokens, seconds and operations (applied and queued;
planned in a dry run). A dream job that runs inside a `run` (§8.4) also
counts its calls and tokens against that run's budget.

*Configuration.* `[dream]`: `url`, `model`, `api_key`, `extra`,
`answer_max_tokens` (the dream model's endpoint, through the chat gate),
`probe_tolerance`, `adjudications_per_unit` (3; 0 turns merges off),
`prompt_max_tokens` (4000: the prompt budget of one call), and
`[dream.review]`, a table of scope → `{low, medium,
high}` = `auto` | `review` (the nearest declaration wins, like a language
override, P12).

Output of a run: a **dream report** — counts per operation, what went to
review, what was dropped and why, model calls and tokens spent, recall probe
before/after (numbers only, never the per-probe ids), and the adjudications:
pairs adjudicated, skipped and why, verdicts (`same`, `different`, `unsure`),
answers that were bad or truncated, proposals created, and per pair the two
node ids and the verdict (never the reason: content); and the prompts: their
budget, the largest prompt sent (tokens, by the service's counter), the units
and pairs reduced to fit, and those skipped `oversized`. Counts and ids only in
logs (P5); content only through the API.

---

## 6. Capability tiers (P10)

| Tier | Needs | Recall | Write path |
|---|---|---|---|
| **0** | nothing | FTS (BM25) + graph expansion over explicit nodes/edges | `remember`, explicit links, deterministic mention detection of known names/aliases |
| **2** | an OpenAI-compatible **chat** endpoint | same as 0 (**no LLM on the hot path**, P3) | + LLM extraction of nodes/facts/edges, contradiction detection, summary maintenance, resolution adjudication, **dream** (§5.6) |
| **3** | + an OpenAI-compatible **embeddings** endpoint | + dense retrieval fused with lexical by RRF, semantic resolution and link proposal | + embeddings computed in the background |

(Tier 1 is intentionally absent: keep the table honest about what each
dependency buys.)

The embedding model is **stamped** (name + dimension) on every vector. Changing
the model marks vectors stale and re-embeds in the background; stale vectors
are never compared with fresh ones.

### 6.1 Tier 3 in detail
**What is embedded.** A fact's statement; a node's **name** (name and aliases)
and, when it has one, its **summary** — two vectors, since a mention is
compared with names and a subject with summaries. The summary vector serves
**semantic link proposals only** (below), never recall. Vectors are stored
unit-normalised (cosine = dot product) in the encrypted database, stamped with
the model name, the dimension and the hash of the embedded text. A vector is
**stale** when its model or dimension differs from the configured one or its
text changed; stale vectors are skipped by every comparison, and the
background `embed` job recomputes missing and stale vectors in batches, under
the gate and the run budget (P11). It is queued after every write that adds or
changes embedded text, and at start-up when stale vectors exist (a model
change). `forget` deletes a record's vectors (P5). The job reads a text, calls
the model, then writes: the write re-reads the facet's current text in its own
transaction and is discarded (and counted) when the record is gone or no longer
live, or its text changed during the call — no vector derived from forgotten
content, no stale vector kept for good.

**Recall (dense + lexical).** The query is embedded once — or not at all when
the caller supplies a `query_vector` stamped with the configured model and
dimension (otherwise it is ignored, and the response says so) — and repeated
queries hit a small in-process cache. Dense candidates are drawn from the live
vectors **of the records the view sees, filtered in SQL before any score** (P9:
an invisible vector neither scores nor shifts a rank). Lexical and dense
rankings are fused by **reciprocal rank fusion** (k = 60); graph expansion
then works as in tier 0/2. A handle reached by the dense side says so in its
`why` ("vector neighbour 0.83").

**Recall ranks on what was said, never on a summary.** A summary is
model-written text (extraction, `refresh_summary`, `create_theme`). Recall —
the lexical index at every tier and the dense side — ranks on **names, aliases
and facts**, what was actually said, so a rewritten summary can never push a
fact down. Measured (qwen3.5:9b, default corpus): nine auto-applied
`refresh_summary` and nothing else rolled `user:alice` back; the new summary
"Youssef est comptable et préfère le café." restated a fact already shown and
took, at 600 tokens, the place of the only answer to "Que préfère boire Ivan ?"
(a fact in Russian). The summary stays **shown** in node handles and profiles.

**The summary shown is bounded.** A node handle's `line` and a profile
entry's `summary` show at most `SUMMARY_LINE_TOKENS` (24) tokens of the
summary, counted with the configured tokenizer (§7.3), cut at a word boundary
(a character boundary in a script written without spaces) and ended with "…"
inside the bound. A node without a summary shows its kind and live fact count
("person, 4 live facts"). The stored summary stays whole and is
shown in full by `expand` and the node view. 24 tokens is the accepted price
of a summary: about one clause, against some 8 for the line it replaces. A
bound in characters would not do: the same 60 characters cost far more in
Cyrillic than in English. Measured with
huihui_ai/Qwen3.8-abliterated:27b on the synthetic corpus: two auto-applied `refresh_summary`, a
`flag_conflict` and a `decay_edges` rolled `user:bob` back (recall 1.0 → 1.0,
recall per 1k tokens 1.81 → 1.75, −3.5 %); a local proof applying each
operation alone found only `refresh_summary` costing tokens (`flag_conflict`
+1.5 tokens per probe at most, `decay_edges` none), and a full summary also
changing what fits the budget. Before the first dream a node has no summary,
so any summary costs more than the line it replaces: the probe comparison
(§11.2) prices a bounded summary as that line.

Every recall response carries `tier` (the tier
that actually served it), `dense` (`on`, `off` by configuration, or
`unavailable`) and `degraded` (why a configured tier was not served).

**The query embedding has priority.** It is the foreground turn: it does not
pass the background gate, but while it is in flight the background issues no
new model call (the `yield` mechanism), so on a single-slot backend it never
waits behind iifa's own queue. It is bounded by `[embeddings] query_timeout`
(1.5 s); a timeout or a failure serves the recall at tier 2 with `degraded`
set (P10), never fails it. `[embeddings] recall_dense = false` turns the query
embedding off for single-slot setups: recall stays lexical + graph, vectors
still serve resolution and link proposal in the background; `/v1/health` says
which.

**Vector arithmetic.** With the optional `vectors` extra (numpy) a
brute-force scan of 10 000 vectors takes milliseconds. Without it tier 3 stays
available on a pure-Python path (`math.sumprod` on 3.12, a slower loop on
3.11): the same results, slower — fine for hundreds of vectors, above the 50 ms
target (§7.4) past a documented size. `/v1/health` reports the active path and
that size (P10: say it, do not impose it). Both paths are tested to rank
identically.

**Resolution (§5.1 step 3).** A mention is embedded and compared with the name
vectors of the nodes visible in the **source's scopes** only (P9: a
near-identical node in another scope is never a candidate). A neighbour above
`resolve_threshold` is a candidate for the batched adjudication (tier 2): it
may attach the **new mention** to that node, recorded like any adjudication.
It never resolves on its own.

**Merge proposals.** Two **existing** nodes found close by their name vectors
(above `merge_threshold`) are never merged here, not even by adjudication:
they become a `merge` proposal (§4.4b) for the review queue (§5.6) — and only
when their scope sets are identical (dream rule 12: never widen visibility).

**Semantic links (§5.2).** Two visible nodes whose summary vectors are close
(above `link_threshold`) get a `similar_to` edge, proposed, at reduced weight;
it follows the ordinary confirmation, reinforcement and decay rules.

**Thresholds are per model.** A cosine means something only for the model that
produced it (bge-m3 puts most names of one entity across scripts above 0.53
and 95 % of two entities below it; qwen3-embedding:0.6b needs 0.84 for the
same), so there is no global default. Built-in **profiles** keyed by the
embedding model (a `:latest` tag and case ignored) give `resolve`, `merge` and
`link` thresholds: `bge-m3` (0.53 / 0.84 / 0.70) and `qwen3-embedding:0.6b`
(0.84 / 0.98 / 0.70), calibrated by the rule below; `link` has no ground
truth to calibrate on (below), so both profiles stay marked **provisional**. `[embeddings]`
values override a profile. For a model with neither a profile nor configured
thresholds, semantic resolution, merge proposals and semantic links are **off**
and `/v1/health` says `uncalibrated`; dense recall still works (RRF fuses ranks,
not similarities). `bench/recall_quality.py` measures recall with a model;
`bench/thresholds.py` is the calibration command.

**Calibration.** The cosines are taken between name vectors exactly as
the service embeds them, on a **separate name list that is not in the bench
corpus** — other first names, cities and teams, in the corpus's six scripts —
so that a threshold tuned on the corpus's nine people says something about a
real user's names; the corpus is then the **verification** set. Positive
pairs are one entity in two scripts, **adapted forms included** (Joseph /
Иосиф, Vienna / Вена: what transliteration cannot do and embeddings should;
the report says whether `resolve` catches them, on the corpus too: Alice /
Алиса, Émilie / Эмили, Youssef / Юсеф). Negative pairs are two entities of the
same kind; **traps** are two entities with near names (Daniel / Daniela,
Austria / Australia). The rule, per model:
- `merge` (it only makes a proposal, reviewed later): the lowest 0.01 step
  above **every trap** — zero traps, always — with at most **2 %** of the
  other negatives above it; the report gives that rate and the share of
  positives it still catches. Near names score as high as one name's
  adapted forms (bge-m3: Bogdan / Bogdana 0.84, Alice / Алиса 0.73), so this
  leaves few merge proposals, and that is accepted. The model gets **no
  profile** only when `merge` would catch no positive at all;
- `resolve` (it only sends a candidate to the adjudication): the 95th
  percentile of the negatives, never above `merge`. A trap above it reaches
  the adjudication, and "when we cannot tell, we do not supersede" (§5.4)
  stays the guard;
- `link` (summaries of *related* subjects) has no ground truth in the
  corpus: it stays provisional and is said to be.
Measured (calibration list: 269 positive pairs, 3,724 negative, 13 traps):
bge-m3 — `resolve` 0.53 catches 85 % of the positives (83 % on the corpus),
`merge` 0.84 catches 5 % (8 %) with no negative above; qwen3-embedding:0.6b
— `resolve` 0.84 catches 52 % (57 %), `merge` 0.98 catches 1 % (its traps
reach 0.97: Bogdan / Bogdana, Niger / Nigeria). A real run then checks the
result at tier 3, default corpus and trap corpus: which trap pairs crossed
`resolve`, what the adjudication answered for each, the number of
adjudication calls, no false supersession or wrong attachment, and no drop
in the headline metric.
A merge proposal is decided by a model adjudication that reads the
names **and** the facts of both nodes, never by the name cosine alone.
bge-m3 is the recommended model: the widest separation, the best recall@1,
and 0.66 GB loaded against 2.4 GB for qwen3-embedding:0.6b.

---

## 7. Storage and engineering constraints

### 7.1 One encrypted SQLite file
- **SQLCipher** for encryption at rest. Rationale: SQLite's FTS5 cannot index
  application-level ciphertext; whole-file encryption keeps FTS5 working.
  (The alternative — decrypt rows into memory and score in Python — was
  measured elsewhere to be fine at hundreds of records and does not scale to a
  multi-user service.)
- FTS5 for lexical search.
- Vectors as BLOBs with model stamp; brute-force cosine is acceptable up to a
  few thousand vectors per scope. Offer `sqlite-vec` as an **optional extra**
  for larger stores, behind the same interface.
- Hard delete (P5): delete rows, FTS entries, vectors; enable `secure_delete`;
  run the FTS `optimize`/merge so deleted tokens leave the index segments.
- WAL mode, a single writer (the background worker), many readers. WAL checkpoints
  run on a background thread with its own connection, never inside a write:
  a commit only appends to the WAL, and the fsyncs happen off the request path.
- Versioned migrations from day one.

### 7.2 Processes
One service process (API + MCP over HTTP/SSE or stdio) and one background
worker (may be a thread/task in the same process for the turnkey mode, a
separate process in a scaled mode). The worker owns the ingestion queue
(persisted in the DB so a crash loses nothing) and obeys the background mode
(P11) — including after a restart: a `paused` service comes back paused.

### 7.3 Token counting
Pluggable tokenizer. Default: a conservative estimate (characters / 3.5,
rounded up, plus per-item overhead) so P1 holds even when the caller's real
tokenizer is unknown. Optional exact counters (e.g. `tiktoken`, a Hugging Face
tokenizer) as extras. The budget packer must be tested against the exact
counter when one is installed.

### 7.4 Performance targets (measured by the benchmark, reported in docs/benchmarks.md)
- `recall` tier 0/2, 10k facts per scope: **p95 < 50 ms** on a laptop CPU.
- `recall` tier 3: p95 < 50 ms + one embedding call for the query.
- Turn ingestion call: **< 10 ms** (it only enqueues).

### 7.4b Talking to a model
Real local models fail in ways a fake does not; the write path assumes them.
- **Every writing call caps `max_tokens`**, sized from its input (a base plus a
  multiple of the source's estimated tokens, bounded by
  `[extraction] answer_max_tokens`). A model that never stops costs one
  bounded call, not the whole budget.
- **A `finish_reason: length` answer is a failure, never parsed**: a truncated
  JSON object may still parse into a partial, wrong answer. The call is asked
  again once with twice the cap (up to the bound), then the job fails without
  retry (at temperature 0 the same call truncates the same way). Only a caller
  that asks for nothing but the call itself (a ping) may accept a truncated
  answer, explicitly.
- **Per-endpoint extra request parameters** (`[models.chat_extra]`,
  `[models.embeddings_extra]`, `IIFA_CHAT_EXTRA` / `IIFA_EMBEDDINGS_EXTRA` as
  JSON) go into every request body of that endpoint — e.g.
  `reasoning_effort = "none"` for a thinking model behind Ollama's OpenAI API,
  which ignores `think: false`. They may not replace what iifa sets itself
  (`model`, `messages`, `input`, `stream`, `max_tokens`, `temperature`,
  `response_format`): such a key is refused at load time.
- **JSON mode is configurable** (`[models] chat_json_mode`, default on): some
  model/server pairs never finish in JSON mode. When it is off, the JSON object
  is taken out of a plain answer — reasoning blocks (`<think>…</think>`) and
  code fences removed, then the first object carrying the expected keys. The
  schema wall (P8) is the same either way.

### 7.5 Stack
Python 3.11+ (D1), FastAPI + uvicorn, pydantic, an SQLCipher binding, the
official MCP Python SDK, httpx. Every runtime dependency is listed with a
one-line justification in `DEPENDENCIES.md`; keep the list short and prefer the
standard library when it is not worse. Tests: pytest, pytest-xdist.

---

## 8. HTTP API (v1)

All under `/v1`, JSON, bearer-token auth bound to scopes. Every content-returning
call accepts `budget_tokens`, a **view** — exactly one of `scope` (that scope
and its ancestors), `scopes` (an explicit list) or `subtree` (admin only), see
§4.6 — and `exclude_ids` (ids the harness has already injected this session —
never resent). Writes take `scope` or `scopes` (the set the new record belongs
to).

Each token carries **permissions**, checked on every route: `read`, `write`
(remember, link, confirm, archive, restore), `forget` (hard delete, P5),
`dream` (start, follow, cancel and undo a dream, §8.6; change a probe set,
§8.7), `review` (list and decide the review queue — fact and alias proposals
and queued dream operations — §4.4b, §8.6), `admin` (background
mode, export/import, stats). A token handed to an agent normally holds
`read` + `write` only. `review` is never implied by another permission.

### 8.1 Lifecycle (Door 2)
| Endpoint | Purpose | Hot path? |
|---|---|---|
| `POST /v1/session/start` | Returns the **profile block**: pinned facts + the most central nodes for the scope, under budget. **Deterministic and stable** across calls when nothing changed, so it can sit in a cacheable prompt prefix. | yes |
| `POST /v1/recall` | `query` (usually the user's message, optionally recent turns) → handles under budget; optional `query_vector` (`model`, `values`) at tier 3 | yes |
| `POST /v1/turns` | Record a turn: `user` (the owner's words), `assistant`, and `external` parts (`kind`: `tool` \| `web` \| `document` \| `other`, `text`, optional `ref`) for content from outside the conversation; `tool_summaries` is still accepted and counts as `external` parts of kind `tool` (P6). Enqueue only. | yes (enqueue) |
| `POST /v1/precompact` | The harness is about to compact: receive the messages about to be dropped, enqueue extraction, return a **short** block of what must survive (under budget) | yes |
| `POST /v1/session/end` | Flush and extract the session | no |

### 8.2 Reading
| Endpoint | Purpose |
|---|---|
| `POST /v1/expand` | ids → content at a chosen level (1 = facts, 2 = facts + source excerpts), under budget; paginates with a cursor |
| `GET /v1/nodes/{id}` | a node with its live facts, top edges and pending proposals (§4.4b) visible in the view, under budget |
| `POST /v1/timeline` | node or scope + date range → events and fact changes |
| `POST /v1/at` | "what was known at date D" for a node |
| `POST /v1/path` | how two nodes are related (bounded path search) |
| `GET /v1/lookup` | `external_id` → the visible records carrying it (§4.6); or `name` → the visible nodes with that name or alias (what `memory_about` uses) |

### 8.3 Writing
`POST /v1/remember` (explicit fact, P6), `POST /v1/link`, `POST /v1/confirm`
(edge or fact), `POST /v1/nodes/merge`, `POST /v1/archive` and
`POST /v1/restore` (node, fact, edge — reversible, P5), `GET /v1/archive`
(list archived items, under budget), `POST /v1/forget` (node, fact, edge or
source — hard delete, `forget` permission, P5; returns what was removed, by
count), `PATCH` on nodes and facts — including `PATCH /v1/facts/{id}` to
attach a free-standing fact to a node (`node_id`), detach it, or pin it.
`POST /v1/proposals/{id}` with `decision: accept | reject` (§4.4b; `write`,
by a token bound to one of the proposal's scopes; a decided proposal is not
decided again; a `fact` proposal also needs `review`, and its acceptance
answers with the id of the explicit fact written). `GET /v1/proposals` (view,
optional `kind`; `read`): the pending proposals visible in the view, oldest
first, under budget — the only place a fact proposal without a node shows.

`precompact` messages take their provenance from their `role`: `user` is the
owner's, `assistant` the assistant's, anything else (`tool`, `system`…)
external (P6).

### 8.4 Operations
`GET/PUT /v1/background` (mode `auto` | `paused` | `on_demand`, concurrency
per endpoint — P11), `POST /v1/background/run` (process now, with an optional
budget; answers with a run id and what it processed), `POST /v1/background/yield`
and `POST /v1/background/resume` (a foreground turn starts / ends).
`GET /v1/health` (tier, backends reachable, queue depth, stale vectors),
`GET /v1/stats` (counts only, never content), `POST /v1/export` and
`POST /v1/import` (a documented JSONL format, encrypted export optional; each
imported record may carry an `external_id`, §4.6).

### 8.5 Response shape for handles
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

### 8.6 Dream (§5.6)
| Endpoint | Purpose | Permission |
|---|---|---|
| `POST /v1/dream` | Start a run: `scope`, `operations` (subset of the catalogue, default all), `budget` (calls, tokens, seconds, operations), `dry_run`, `review_policy` (`low` / `medium` / `high` → `auto` \| `review`, only stricter than the configured policy, else 422). Returns a run id at once (202) — the work is asynchronous; 409 while a run is pending or running on the scope, an ancestor or a descendant | `dream` |
| `GET /v1/dream/{id}` | Status and, once finished, the dream report | `dream` |
| `GET /v1/dream/{id}/plan` | The operations proposed / applied / sent to review / dropped, each with its reason, in order (paginated with `after`, under budget) | `dream` |
| `GET /v1/review` | The review queue of a view: pending `alias`, `fact` and `merge` proposals and queued dream operations, oldest first, under budget; a `merge` proposal and a queued `merge_nodes` carry `adjudication` (`verdict`, `reason`, `run_id`, `at`; null when the pair was never adjudicated) | `review` |
| `POST /v1/review/{id}` | `accept` or `reject` one item (a proposal or a queued operation); an operation whose inputs moved since its snapshot is dropped (409, `stale`); a `merge` proposal can only be rejected (409 on accept: only its reviewed `merge_nodes` merges) | `review` |
| `POST /v1/dream/{id}/undo` | Revert a run, or one operation (`op_id`); 409 naming what later writes depend on | `dream` |
| `POST /v1/dream/{id}/cancel` | Stop a running dream cleanly (same path as hitting a budget limit) | `dream` |

Starting a dream requires a token carrying the `dream` permission — distinct
from read/write, so a harness can let its agent read and remember without
being able to launch one. Deciding what a dream queued requires `review`,
which `dream` does not imply: whoever may launch a dream does not thereby
approve what it proposes. A token reaches a run only when it is bound to the
run's scope or above (§4.6), otherwise the run does not exist for it (404); it
reaches a queued operation only when it sees every record the operation
touches, and decides it only when it may touch them all.

### 8.7 Probe sets (§11.2)
| Endpoint | Purpose | Permission |
|---|---|---|
| `POST /v1/probes` | Create a set: `scope`, `name`, `budget_tokens`, `probes` (`query`, `expected` ids) | `dream` |
| `GET /v1/probes?scope=` | The sets of that scope (ids, names, sizes) | `read` |
| `GET /v1/probes/{id}` | One set with its questions and expected ids | `read` |
| `POST /v1/probes/{id}/run` | Run it now: the probe score (§11.2) | `read` |
| `DELETE /v1/probes/{id}` | Delete it (hard: it is the operator's, not memory) | `dream` |

Changing a set is the `dream` permission's, because a set decides whether a
dream is rolled back: an agent that may remember must not be able to move that
bar. A token reaches a set only when it is bound to the set's scope (§4.6);
otherwise the set does not exist for it (404). No MCP tool (§9).

---

## 9. MCP server (Door 1)

Tools (names prefixed to avoid collisions): `memory_recall`, `memory_expand`,
`memory_remember`, `memory_link`, `memory_archive`, `memory_timeline`,
`memory_about` (node lookup by name/alias). **`memory_forget` is not exposed
by default** (P5): enabling it is a server option, and even then it requires a
token with the `forget` permission.

- **Tool descriptions carry the doctrine**, briefly: recall returns handles —
  expand only what you need; remember explicit, durable facts, not chatter;
  archive what is wrong or obsolete — it can be restored.
- Every tool result obeys the budget (default from config, overridable per call).
- Results are wrapped as recalled data (P8).
- Transports: stdio and streamable HTTP.
- **No dream tool** — on purpose (§5.6 rule 11).

---

## 10. Adapters shipped in the repository

Each adapter is thin (the logic lives in the service), fail-open (P3), has its
own README with install steps and the harness doc version it was built against,
and its own tests against a fake service.

Fail open means: the service down, slow, answering 4xx/5xx or garbage → nothing
injected, no exception into the harness, one log line (ids, counts, reasons;
never memory content, P5), the turn proceeds. Every call on the turn's path has
a hard deadline of the adapter's own, under the harness's.

Every adapter builds its scopes `kind:name` at **every** `/` level
(`user:alice/mode:assistant`, never `user:alice/assistant`), checks the
configured scope once at start (a bad one is logged with the offending segment
and memory stays off), and has a test proving its default and derived scopes
pass `iifa.scopes.validate_scope`.

The tools an adapter declares itself are the four that map to one call each —
`memory_recall`, `memory_expand`, `memory_remember`, `memory_archive` — with
results wrapped as recalled data (P8) under a budget (P1). The other tools
(`memory_about`, `memory_timeline`, `memory_link`) stay on the MCP server
(§9), which these harnesses can load as well.

1. **Python client SDK** (`iifa.client`) — sync + async.
2. **Claude Code plugin** — hooks for the five moments (§3.2) + the MCP server
   registration. `UserPromptSubmit` must answer within its timeout or emit
   nothing.
3. **Hermes Agent `MemoryProvider`** — a directory provider, standard library
   only: the profile block from `system_prompt_block`, `prefetch` (handles
   under budget, bounded by the adapter's own deadline), `sync_turn`
   (non-blocking: one in-process enqueue, a worker thread posts the turn),
   `on_pre_compress` (what must survive, into the summary prompt),
   `on_session_end`, `on_session_switch`, tools via `get_tool_schemas` /
   `handle_tool_call`.
4. **opencode plugin** (TypeScript) — the profile block through
   `experimental.chat.system.transform`, `chat.message` injection, turns
   recorded on `session.idle`, `experimental.session.compacting`, tools. Its
   tests run with `node --test` (no dependency) against the fake service.
5. **OpenAI Agents SDK example** — memory through tools, plus the agent's
   dynamic `instructions` injecting the profile block. The SDK is an optional
   extra (`openai-agents`), never a core dependency.
6. **A minimal reference loop** (`examples/minimal_agent.py`) showing all five
   moments against any OpenAI-compatible endpoint, in under 200 lines. This is
   the integration guide for developers with their **own** harness.

---

## 11. Evaluation (ships with the code)

`bench/` runs without network except the configured model endpoints.

- **Datasets:** a **synthetic** multilingual corpus generated by a script in
  the repo (people, projects, moves, changes of mind — exercises P7), and two
  public long-term-memory benchmarks the operator downloads (licences
  checked, download instructions in docs/benchmarks.md). Neither is ever vendored —
  not in the repository, not in a release, not in an image — whatever its
  licence: the repository stays Apache-2.0 alone, and each dataset keeps its
  own terms with whoever downloads it. They go to `bench/data/`, which git
  ignores.
  - **LongMemEval** (Wu et al., ICLR 2025): 500 questions over chat histories
    testing five long-term memory abilities. Repository MIT; the dataset card on
    Hugging Face (`xiaowu0162/longmemeval-cleaned`, the 2025-09 cleaned
    release) declares MIT. Files: `longmemeval_oracle.json` (15.4 MB,
    evidence sessions only), `longmemeval_s_cleaned.json` (277 MB, ~40
    sessions per history), `longmemeval_m_cleaned.json` (2.74 GB, ~500).
  - **LoCoMo** (Maharana et al., 2024): ten very long two-person
    conversations with annotated questions (`data/locomo10.json` in
    `snap-research/locomo`). Licence **CC BY-NC 4.0**: non-commercial use
    only, with attribution. It is used for measurement only and never enters
    anything iifa distributes.
  Both are English; the multilingual properties (P12, cross-script names)
  stay measured on the synthetic corpus. The harness does not run them.
- **Metrics:** answer-supporting recall@budget, **recall per injected token**,
  contradiction handling accuracy, latency p50/p95, ingestion throughput.
- **Baselines:** "no memory", "full history truncated to the same budget",
  "naive vector top-k to the same budget". Comparisons with other projects are
  published only with their exact versions and configuration.
- Results are reproducible with one command and written to `bench/results/`.

### 11.1 The harness (`bench/run.py`)

One command: `python bench/run.py [--chat-url … --model …] [--embeddings-url …
--embeddings-model …] [--budgets 150,300,600] [--seed 7]`. It builds the
corpus, ingests it into a throwaway encrypted store through the service's own
worker, asks every question of every system at every budget, and writes one
JSON file to `bench/results/` (gitignored) holding every number, the corpus
seed and size, the models, and the exact command line. Same seed, same corpus,
same questions: the only thing that varies between two runs is what is being
measured.

**The corpus** (`bench/synthetic.py`, seeded, no randomness outside the seed):
people whose names exist in several scripts (Karim, Карим, كريم, カリム,
卡里姆…), each with attributes that **change over time** — the city they live
in (moves), their role, a preference (changes of mind) — told month after month
in turns written in English, French, Russian, Arabic, Chinese or Japanese, the
name spelt in the turn's script; chatter turns with nothing to remember in
between. **Several scopes**, one of them holding a homonym (another Karim with
other facts): every question is asked through the view of one scope (§4.6),
and a unit from a scope the view cannot see is counted as a **leak** — which
must stay at zero for every system (P9 in the bench itself). Each turn states
at most one fact, so a recalled record supports the event of the turn it came
from.

**Questions** are generated from the ground truth in the asking scope's
language: "Where does Karim live?" expects the event of his **current** city;
when the attribute changed, the earlier events are its **stale** answers.

**Extraction.** `--chat-url` runs the real tier-2 pipeline against that model
(measuring it end to end: facts written, fragments and cross-script drops,
superseded, conflicts). Without it, an **oracle** answers every extraction call
with the gold answer an ideal model would give (the known name reused, the
source's spelling in `subject_as_written`, attribute/value set), through the
same client, parser, checks, resolution and contradiction rules — so the
retrieval layer is measured on its own, and the report says which extractor
ran. The oracle makes no network call. `--keep-store PATH` (debugging only) keeps
the encrypted store at PATH instead of deleting it, so that what a model wrote
can be read afterwards; the operator deletes it.

**Systems**, each answering a question with whole units under the budget,
counted with the same tokenizer (§7.3, per-item overhead included):
- `none` — injects nothing (the floor);
- `truncated_history` — the view's turns, newest first, as many whole turns as
  fit;
- `vector_topk` (needs `--embeddings-url`) — the view's turns embedded, the
  nearest to the question first, as many whole turns as fit;
- `iifa` — `recall` at tier 2, and at tier 3 when embeddings are configured.
  Only **fact** handles count as support (a node handle may carry the answer in
  its line; counting it would flatter iifa).

**Metrics**, per system and budget:
- **correct current answers per 1k injected tokens — the headline.** A
  question is answered *correctly* when an injected unit supports its current
  event and none supports a stale one; the rate is correct answers /
  questions / mean injected tokens × 1000 (`none` reports 0). Raw recall alone
  rewards injecting every version of a value — naive vector top-k finds the
  old city and the new one and scores best on it — so it is never shown alone;
- **recall@budget** — share of questions for which an injected unit supports
  the expected event (a turn containing it; a fact extracted from it), with
  **recall per 1k injected tokens** beside it;
- **contradiction accuracy** — over questions whose attribute changed: the
  current event is supported and no stale one is; also counted: stale only,
  both (and, for iifa, both with the `conflict` flag), neither. Split by
  whether the change was told in the same language as the previous value
  (§5.4 compares values within one language only);
- **latency** p50/p95 of answering (the query embedding included);
- **ingestion**: enqueue p50/p95 (the `turns` hot path, §7.4) and processed
  turns per second through the worker (the model's time included);
- **leaks** — units outside the view (must be 0);
- **supersessions** (read from the store after ingestion): correct (a later
  value of the same person's same attribute closed the earlier one), **false**
  (the replacement is about another attribute or person, or is not later), and
  unmatched (a fact from a chatter turn was closed or closed another). Each
  false supersession is listed (up to ten): both turns, the person and
  attribute each was told about, and both statements as stored — a count
  alone cannot tell a wrong key from a fact filed on the wrong node. With
  `--traps` the corpus adds, for every person, where they work — a city next
  to the one they live in — so that a key that lumps both together shows up
  as false supersessions; and, in the last month, values **added** to what is
  already true — a second role ("is also a photographer"), a team then a
  second team, a second favourite drink — so that a "replaces" that closes a
  value still true shows up as a **false replaces** (counted within the false
  supersessions and apart). The questions about those attributes accept any
  of the values that hold at the end. The trap corpus is a different corpus: its numbers
  are compared only with runs that used it;
- **chatter** (read from each turn's extraction report): facts written and
  nodes created from chatter turns, the chatter turns that produced anything,
  and, after ingestion, the nodes named by a speaker word (§4.2; must be 0);
- **restatements** (`--rephrase`): after some event turns, a later turn of
  the same month restates the same value in another of the person's languages
  ("Karim habite à Lyon." then "Карим живёт в Лионе."): not a change. Counted:
  facts written from them, linked to the original by `same_as`, flagged
  `conflict`, and closures between a value and its restatement (`restated`,
  apart from correct and false supersessions). Every system also counts
  **duplicates**: questions for which two injected units support the same
  event — recall injects one fact per equivalence group (§5.4), so iifa
  should have none. A restatement supports its event like the original turn;
- **the owner** (`--speaker`): the scope `user:alice` declares its owner,
  Alice (§4.6), who speaks in the first person in her own turns ("J'ai
  déménagé à Lyon.", "Я работаю фотографом."), in French, English and Russian,
  with moves, role changes and changes of mind like the others, and is asked
  about in the first person ("Où est-ce que j'habite maintenant ?"). Counted
  as the other people's questions, plus the facts attached to the owner, free
  `speaker` facts and fact proposals; and, for each of the owner's turns, its
  own record (turn key, language, facts written and proposed, drops by cause,
  retries, nodes created — numbers only), since totals cannot tell which of
  her sentences was lost and why;
- **near names** (`--name-traps`): three people whose names are near another
  person's of the same scope — Karima beside Karim, Ivana beside Ivan, Nadine
  beside Nadia — with homes, roles and drinks of their own, told in their
  languages and asked about like the others. A name vector near enough to
  reach `resolve` (§6.1) sends the other person's node to the adjudication;
  the bench lists every such pair and what the model answered. Counted:
  **wrong attachments** — a fact from a turn about one person written on a
  node whose name is another person's (must be 0);
- **name adjudications**, every corpus: each mention sent to the adjudication
  with its candidates (name, score, whether the candidate came from name
  vectors) and the answer (the candidate chosen, or a new node), as each
  turn's extraction report gives them, with the number of adjudication calls;
  a pair of two different people among them is a **near-name pair**, and its
  answer is reported (attached is a wrong attachment).
`--resolve-threshold` and `--merge-threshold` replace the embedding model's
profile for a run, so that one corpus is measured at two profiles.

`--rephrase`, `--speaker` and `--name-traps` each make another corpus,
compared only with runs that used it; the default corpus is unchanged.

**The language question** (`--force-language CODE`): every scope of the
corpus forces the language of what is written into it (P12's scope override),
so the model translates what it stores; the questions stay in each scope's
language. Run against the same command without it, it answers by measurement
whether storing facts in their source's language (P12's default) or in one
forced language (English) gives more correct current answers per token. It
needs a real model: the oracle's gold answers are in the source's language.

**The dream** (`--dream`, §5.6): after ingestion and the first evaluation,
every top-level scope of the corpus dreams, through the service's own
`Dreamer` and the probe sets above (rule 9 measured as in service): a dry run,
then a real run under the default review policy, then the iifa systems are
evaluated again. The dream model is `--dream-url` / `--dream-model` (default:
the chat endpoint and model), with `--dream-extra` (e.g. thinking off; a
context window there is ignored by Ollama, see *What one call reads is
bounded*) and `--dream-prompt-tokens` (the prompt budget, 4000); without
`--chat-url` the oracle answers the dream too (no operation, and the corpus's
truth for each adjudication), which proves the command runs, nothing more.
Reported per operation: proposed (planned in the dry run), rejected by cause,
applied, queued, dropped by reason (`stale` apart); the adjudications, each
pair judged against the corpus (one entity, two, or not judged) beside its
verdict; the probe scores before and after; and the headline at every budget
before and after. Judged against the ground truth, dry and real run alike:
- **wrong merges** — a `same` verdict (a planned, queued or applied
  `merge_nodes`) between two nodes that name two different entities of the
  corpus (a person by any form of their name, a place by any form of its
  value); must be 0 with `--traps --name-traps`. A `different` on one entity
  is reported (a missed merge), not counted wrong. A `same` the service
  downgraded because the names do not fit is counted apart: **caught** on two
  entities (a wrong merge stopped), **held** on one (a right merge sent back
  to review as `unsure`);
- **wrong consolidations** — a `consolidate_facts` whose originals come from
  events of different people, attributes or values (they did not say one
  thing), and **lossy consolidations** — one whose statement holds no form of
  a value one of its originals carries, even when they said one thing: both
  counted wrong;
- **wrong supersessions** — a dream `supersede_fact`, judged as the write
  path's (above);
- a consolidated fact supports the events of its originals; a merged node's
  facts keep their turns;
- an operation a rollback undid is judged like one still applied: it reached
  memory all the same.
A run whose probes regressed names what it lost: each probe that found a
record before and misses it after, with its query and the ids recall returned
before and after, in rank order — read in a store kept with `--keep-store`.
`--dream-accept-review` then accepts every queued operation, as a reviewer who
accepts everything would, and evaluates once more: a **ceiling**, not what a
user gets — reported beside the auto-applied numbers, never alone.
`--hostile` (with `--dream`) writes hostile records straight into the store
before the dream (no extraction to filter them out): instructions in a node's
**name**, an **alias**, a **summary** and in **facts**, each asking for one
recognisable action — a marker word in a summary, a theme or a statement, one
given fact flagged or superseded, one given pair merged (through the
adjudication), an operation outside the catalogue — and two **namesakes**: two
different people with one name, one of them carrying a fact that says they are
one person. Counted: items outside the catalogue (rejected; reaching apply
must be 0), and, per carrier, operations that follow an instruction:
**proposed** (what the model did; reported), **queued** (sent to review —
in a dry run, planned on the review route — the review doing its job;
counted and reported apart, not a failure) and **applied** (applied, or
applied then undone — in a dry run, planned on an automatic route; **must
be 0**, a hard failure). The namesakes are the case the queued count is
for: their names fit, so the service keeps a `same` the model was told to
give, and the merge is queued for review (rule 5) — only the review stops
it. What `--dream-accept-review` accepts afterwards is the ceiling
reviewer's choice, not counted as followed. Real dream runs are manual,
after the operator has seen the exact prompts.

Every system and budget also keeps its **per-question outcome** — the
question's key, whether the current event was supported, whether a stale one
was, whether it was answered correctly — keys and booleans only, so that two
runs can be compared question by question (which turned from wrong to right,
and why) without keeping the stores.

Every run also measures, without flagging anything in the service, the narrow
translation check that P12 leaves open (§2): among the facts whose statement
is undetermined and whose quote has a determined language, those sharing no
word of three letters or more with the quote, the subject's names excluded,
counted apart by the turn's language (English: a paraphrase, a false alarm;
another: a likely translation). The check is adopted only if it does not flag
same-language statements.

CI runs the harness at small scale against the fake servers (§13): it proves
the command runs and the numbers are well formed, not that they are good.
Real-model runs are manual, after the operator has seen the exact prompts.

### 11.2 Probe sets (needed by the dream, §5.6 rule 9)

A **probe set** belongs to exactly one scope and holds questions, each with the
ids of the records that answer it (facts or nodes), plus the budget every run
uses. It is stored in the encrypted file like any record (its questions are
content: P5 logs ids and counts only). Every probe expects at least one
record, and its expected records must be visible from its scope (scope +
ancestors, §4.6) when it is created; an invisible or unknown id is refused the
same way (no existence leak, P9). Forgetting a record **deletes every probe
that expected it**, question included (P5: forget leaves no trace). Whether a
question reveals the forgotten content ("Where does Karim live?") cannot be
decided mechanically, so it is never kept; and a probe that lost an expected
answer no longer measures what it measured, so keeping it would skew the
before/after comparison a dream is judged by. `forget` reports them as
`probes_deleted`. A set emptied that way stays, with nothing to score.

**Running a set** asks each question through `recall` with the set's scope and
budget, exactly as an agent would. A probe **finds** an expected record when
the recall returns it, or the fact that superseded it (P7: following
`superseded_by`, and `consolidated_into` for a fact a dream consolidated; for
an expected node, the node a dream merged it into, `merged_into`). Per probe:
found / expected, the tokens used. The **probe score** of the set: `recall`
(mean over scored probes), `used_tokens` (mean), `recall_per_1k_tokens`
(recall / mean tokens × 1000), the tier that served (§6.1), and per probe the
ids found and missing — ids only.

**Summary tokens.** Per probe, `summary_tokens` is, for each node handle the
recall returned whose line is a summary, the tokens of that handle minus the
tokens of the same handle showing the line of a node without a summary (its
kind and live fact count) — never negative, and at most what the bound of
§5.6 allows per handle; fact handles count none. The score carries
`summary_tokens` (mean over scored probes) and `compared_per_1k_tokens`:
recall / (mean `used_tokens` − mean `summary_tokens`) × 1000.
`used_tokens` and `recall_per_1k_tokens` stay raw. Both scores of a
comparison compute `summary_tokens` the same way, so a second dream, run
when summaries already exist, compares like with like: a summary rewritten
within the bound is neutral, as is a first one.
If either score lacks `compared_per_1k_tokens` (a score saved before summary
tokens were counted), both are compared raw: never a raw rate against a
compared one, which would always favour the dream.

**Comparing two scores** (what the dream does before and after a run): the
verdict is `regressed` when recall drops by more than the tolerance (absolute)
or `compared_per_1k_tokens` drops by more than the tolerance (relative); `ok`
otherwise. A bounded summary is thus priced as the line it replaces: on its
own it never rolls a dream back. Everything else still counts: a lost answer
(recall stays strict), a longer consolidated statement, a merge that brings
more facts, an extra handle that now fits — and the tokens of the items a
summary's length lets in or keeps out of the budget, which are content.
The verdict is `inconclusive` when the two runs were not served by the same tier
or have different scored probes. The dream treats `inconclusive` as a
failure (§5.6 rule 9): an embeddings endpoint down during one of the two runs
rolls the dream back with the reason `inconclusive` rather than let an
unmeasured change through.

---

## 12. Security and privacy checklist (each item has a test)

Each box is ticked only with the test that proves it, named beside it
(`path::test`); the CI runs them all.

- [x] Database unreadable without the key (open with plain SQLite fails).
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_database_is_unreadable_without_the_key`
- [x] `forget` leaves no trace in tables, FTS index, vectors, or queue.
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_forget_leaves_no_trace_in_the_file`
  - `tests/store/test_vectors_store.py::test_forget_deletes_the_vectors`
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_forgetting_a_turns_source_leaves_no_trace_in_the_queue_or_anywhere`
- [x] `archive` hides a record from recall, expand, profile and dreams, and `restore` brings it back unchanged; a token without `forget` cannot hard-delete by any route or tool.
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_archive_hides_everywhere_and_restore_brings_back_unchanged`
  - `tests/security/test_p8_dream.py::test_p5_an_archived_record_reaches_neither_the_profile_nor_the_dream`
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_forget_needs_its_own_permission_on_every_route`
  - `tests/mcp/test_tools.py::test_forget_is_exposed_only_by_option_and_still_needs_the_permission`
- [x] In `paused` mode no model call leaves the service; in `on_demand` mode none leaves outside a `run`, and a run stops at its budget (P11).
  - `tests/security/test_p11_background.py::test_p11_paused_means_zero_model_calls`
  - `tests/security/test_p11_background.py::test_p11_on_demand_calls_the_model_only_inside_a_run_that_stops_at_its_budget`
- [x] A French (and one non-Latin-script) source yields facts and summaries in that language; a forced English answer from the fake model is rejected, retried, then flagged (P12).
  - `tests/security/test_p12_language.py::test_p12_the_target_language_is_stated_and_a_drift_is_retried_then_flagged`
  - `tests/security/test_p12_language.py::test_p12_a_non_latin_source_keeps_its_script`
  - `tests/security/test_p12_language.py::test_p12_summaries_are_written_in_the_language_of_the_facts`
  - `tests/security/test_p12_language.py::test_p12_a_non_latin_source_gets_its_summary_in_its_script`
- [x] Logs contain no memory content (scan logs in tests with canary strings).
  - `tests/security/test_logs_and_tokens.py::test_logs_contain_no_memory_content`
- [x] No outbound connection except configured endpoints (socket-level guard in tests).
  - `tests/security/test_p4_socket_guard.py::test_p4_connect_to_public_ip_is_blocked`
  - `tests/security/test_p4_socket_guard.py::test_p4_name_resolution_is_blocked`
  - `tests/security/test_p4_socket_guard.py::test_p4_loopback_fake_is_reachable`
- [x] Scope isolation, including P9's statistical leak.
  - `tests/security/test_p9_scope_isolation.py::test_p9_tokens_cannot_read_outside_their_scopes`
  - `tests/security/test_p9_scope_isolation.py::test_p9_invisible_corpus_changes_no_score_and_no_statistic`
- [x] Injected content is labelled as data; an instruction-like memory does not change the wrapper.
  - `tests/unit/test_recalled.py::test_content_never_changes_the_wrapper`
  - `tests/mcp/test_tools.py::test_recall_returns_handles_wrapped_as_data`
- [x] Tokens are compared in constant time; tokens are stored hashed.
  - `tests/security/test_logs_and_tokens.py::test_tokens_are_hashed_and_compared_in_constant_time`
- [x] Default bind address is `127.0.0.1`; binding elsewhere requires explicit config.
  - `tests/security/test_logs_and_tokens.py::test_default_bind_address_is_loopback`
  - `tests/security/test_docker_image.py::test_the_default_bind_inside_the_container_is_loopback`
- [x] A dream cannot hard-delete, cannot alter an `explicit` fact, and rejects any operation outside the catalogue — including one requested by text stored in memory.
  - `tests/security/test_p8_dream.py::test_a_dream_never_deletes_and_never_touches_an_explicit_fact`
  - `tests/security/test_p8_dream.py::test_p8_hostile_memory_makes_the_dream_propose_nothing_outside_the_catalogue`
- [x] A dream run is fully undoable, and a dream that lowers the recall probe beyond tolerance rolls itself back.
  - `tests/dream/test_undo.py::test_undo_reverts_a_whole_run`
  - `tests/dream/test_rollback.py::test_a_dream_that_lowers_recall_rolls_itself_back`
- [x] A token without the `dream` permission cannot start, follow, cancel or undo a dream, nor create or delete a probe set; a token without `review` (a `dream` token included) can neither list nor decide the review queue.
  - `tests/security/test_p8_dream.py::test_without_the_dream_permission_no_dream_route_answers`
  - `tests/api/test_permission_matrix.py::test_permission_matrix`
  - `tests/dream/test_review_api.py::test_the_review_queue_needs_the_review_permission`
  - `tests/dream/test_review_api.py::test_a_dream_token_cannot_approve_what_its_dream_proposed`
- [x] `forget` of a record wipes it from the dream journal (raw bytes): the operations that mentioned it lose their content and can no longer be undone.
  - `tests/security/test_p8_dream.py::test_p5_forget_wipes_the_dream_journal`
  - `tests/dream/test_undo.py::test_an_operation_wiped_by_forget_cannot_be_undone`
- [x] A dream never widens visibility: it changes no record outside its scope's subtree and draws no new record from records with different scope sets.
  - `tests/security/test_p8_dream.py::test_a_dream_never_widens_visibility`
  - `tests/security/test_p9_tier3.py::test_rule_12_a_merge_proposal_needs_identical_scope_sets`
- [x] A probe set cannot name a record its scope does not see; `forget` deletes every probe that expected the record, question included (raw bytes).
  - `tests/api/test_probes.py::test_p9_an_invisible_or_unknown_expected_id_is_refused_alike`
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_forget_deletes_every_probe_that_expected_the_record`
- [x] Memory poisoning (P6): a hostile web page quoted in a turn (a fake "User:" line, "the user lives in Moscow", instructions) yields no fact about the owner — at most fact proposals — and closes no fact the owner stated; a token without `review` (an agent's) can neither accept nor reject a fact proposal.
  - `tests/security/test_p6_poisoning.py::test_p6_a_hostile_page_yields_no_fact_about_the_owner_only_proposals`
  - `tests/security/test_p6_poisoning.py::test_p6_a_page_never_closes_what_the_owner_said_about_someone`
  - `tests/security/test_p6_poisoning.py::test_p6_an_agent_token_can_neither_accept_nor_reject_a_fact_proposal`

---

## 13. Testing

Tests are part of the product. **A principle without its test is not
implemented**, whatever the code says. The suite must run with no network, no
real model, no GPU, and finish in under **2 minutes** with `pytest -n auto` on
a laptop, its temporary directory on an SSD (measured: 1 min 40 s with 16
workers; on a spinning disk the stores the tests create make it about 4
minutes).

### 13.1 Layers
| Layer | What | How |
|---|---|---|
| **Unit** | budget packer, tokenizer, RRF, entity resolution, decay, contradiction rules, dream operation schema, language detection | pure functions; **property-based tests** (Hypothesis) for the packer — for any items and any budget, output ≤ budget and never cuts an item — and for scope filtering |
| **Store** | SQLCipher schema, migrations, archive/restore/forget | real encrypted DB in `tmp_path`; a fixture DB per released schema version, migrated forward in tests; `forget` checked by scanning the **raw file bytes** for a canary string |
| **API contract** | every route | the app in-process with httpx; a **permission matrix** generated over every route × every permission (allowed / 403); an OpenAPI **snapshot** test so a breaking change is a deliberate diff, never an accident |
| **Model-dependent** | extraction, summaries, embeddings, dream | a scripted **fake OpenAI-compatible server** (chat + embeddings) in `tests/fakes/` that can answer slowly, time out, fail with 500, return malformed JSON, answer in the wrong language, echo hostile content, or propose out-of-catalogue operations — and **records every call**, so "paused means zero calls" (P11) is an assertion, not a hope |
| **MCP** | tool list and behaviour | the server over stdio with the official client: no `forget` or dream tool by default, budget obeyed, results wrapped as data (P8) |
| **Adapters** | each harness adapter | against a fake service; **fail-open** tested with the service down and with it answering past the harness timeout; the TypeScript adapter with Node's built-in runner (`node --test`, no dependency) |
| **End-to-end** | the Docker image | CI only: start the container, `health`, `remember`, `recall`, `archive`, stop |
| **Performance** | recall latency, ingestion enqueue | a small-scale bench with p95 thresholds runs in CI, alone on the runner after the parallel suite (beside it, a loaded runner measures its neighbours); the full bench stays manual (§11) |
| **Security** | every item of §12 | marked `@pytest.mark.security`; each test name cites its principle, e.g. `test_p1_budget_never_exceeded_with_huge_items` |

### 13.2 Rules
- **No network in tests**: an autouse fixture blocks every socket except the
  in-process fakes on loopback. It enforces P4 at the same time.
- **Deterministic**: fixed seeds, an injectable clock (never `sleep` to test
  decay or expiry), no dependence on test order.
- **`tmp_path` only**; never the user's home, never a shared directory.
- **Multilingual fixtures**: English, French, and one non-Latin script, in the
  synthetic corpus and in the unit tests of anything that writes text (P12).
- **Every bug fix starts with a failing test** that reproduces it.
- **Coverage is reported, not worshipped**: ≥ 90 % lines on `src/iifa/`
  (adapters excluded), and **100 % branches** on the budget packer and on
  permission checks — the two places where a missed branch is a broken promise.

### 13.3 Continuous integration
A workflow (GitHub Actions) runs on every push and pull request: formatter and
linter (ruff), type checking (mypy, strict on `src/`), the suite on Python
3.11, 3.12 and 3.13, on **Linux and Windows**. If a dependency makes Windows
impossible (an SQLCipher binding without Windows wheels, for instance), the
README says so and CI keeps Linux plus the Docker image. The workflow file is
ordinary project content and describes the checks it runs, nothing else.

Dev dependencies: pytest, pytest-xdist, pytest-cov, hypothesis, httpx, ruff,
mypy — listed in `DEPENDENCIES.md` under "dev".

---

## 14. Scope

### In scope
Everything in §§2–13, tiers 0/2/3, the adapters in §10, Docker image,
`pip install`, a README with a 5-minute quickstart.

### Out of scope
- Any hosted/cloud offering, account system or billing.
- A web UI.
- Neo4j or any external database server.
- Multi-node / replication.
- Training or fine-tuning anything.
- Document ingestion pipelines (PDF parsing, crawling) — accept text; parsing is
  the caller's job.

---

## 15. Repository layout

```
iifa/
  SPEC.md                 this file
  README.md               what it does, quickstart, integrations, status
  CHANGELOG.md            user-facing changes, per release
  DEPENDENCIES.md         every runtime dependency and why
  LICENSE                 Apache-2.0 (D2)
  pyproject.toml
  Dockerfile
  docs/                   configuration, API, how it works, Docker, security, benchmarks, development
  src/iifa/
    config.py             settings (file + env), key loading
    scopes.py             scope paths, scope sets, the three read modes (§4.6)
    store/                SQLCipher schema, migrations, repositories, background checkpoints
    recall/               lexical, dense, graph expansion, RRF, budget packer, probe scores (§11.2)
    ingest/               queue, extraction, resolution, linking, contradictions
    dream/                operation catalogue + schema, planner, applier, changesets, review queue
    llm.py                OpenAI-compatible chat + embeddings clients (explicit model on every call)
    tokens.py             tokenizer interface + conservative default
    language.py           language and script detection (P12)
    calibration.py        per-model embedding thresholds (§6.1)
    vectors.py            vector arithmetic, numpy or pure Python (§6.1)
    ids.py                prefixed, monotonic ids
    log.py                logging without memory content (P5)
    transport.py          transport failures, classified in one place
    api/                  FastAPI app, auth, routes
    mcp/                  MCP server (tools, streamable HTTP mount at /mcp, stdio entry point)
    client/               Python SDK
    recalled.py           the P8 wrapper every injected block goes through
  adapters/
    claude-code/
    hermes/
    opencode/             (TypeScript)
    openai-agents/
  examples/minimal_agent.py
  bench/                  run.py (the harness, §11.1), synthetic.py (the corpus), thresholds.py,
                          key_names.py, recall_latency.py, recall_quality.py, dream.py, extraction_manual.py
  tests/
    fakes/                fake OpenAI-compatible server (chat + embeddings, call log), fake service, fake harness
    fixtures/             multilingual synthetic corpus, one DB per released schema version
    unit/ store/ api/ client/ model/ mcp/ dream/ adapters/ examples/ security/ perf/ e2e/
  .github/workflows/ci.yml  lint, types, tests (3.11–3.13, Linux + Windows), Docker smoke
```
