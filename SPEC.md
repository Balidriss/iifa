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
- in the agent perspective (§4.2, §4.6), a **self-side** fact — about the
  agent itself, or a rule of method, as the model reads it (§4.2) — is
  written only from an `owner` part
  (under the owner's rule above), or from the `assistant` part of a source
  whose `external` parts, if any, are all **vouched** — the host's statement
  that the owner gave that content or asked for it (§8.1) — or from a
  vouched part itself in such a source; otherwise it is a fact proposal,
  which a reviewer accepts. The rule reads the parts and the host's
  statement, not the words, so a page paraphrased in the reply does not pass
  it, and nothing the answer or a page says (that the owner asked for it to
  be kept, that it is vouched) changes it;
- an extracted fact from an `external` part never closes a fact that did not
  also come from external content (§5.4): the conflict is flagged instead.

**The reader is told who said it.** An agent weighs its memory only if a
recalled line says whether the owner said it, the assistant did, or a page
was read. So every fact a read route returns carries `from` (§8.5) whenever
it is not the owner's own words: the assistant's declared name
(`assistant` when none is declared, §4.6), or for external content the
part's kind, the short form of its `ref` and the day it was read
(`"web example.com 2026-03-14"`). An extracted fact keeps, beside its
provenance, its external part's `kind` and short `ref` and that day, or its
assistant part's declared name (§4.2).
The short form is made when the turn is read and is the only form kept: a
URL gives its host (without `www.`), a path its last segment (without query
or fragment), anything else itself; whitespace becomes `_`, at most 40
characters. Never a whole URL: its query string may carry a token. The day
read is the turn's `occurred_at` (the day it was recorded when the harness
gives none), or for `precompact` the day the service read the messages.

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
(detected from the source, overridable per scope - a scope and every scope
below it, the nearest override winning, and one on `*` everywhere), and a written item whose
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
task — the agent perspective's rules of method, §4.2 —, other — extensible), `name`, `aliases[]` ("my sister" → a person node),
`summary` (one line, maintained), `created_at`, `updated_at`, `scope`,
`merged_into` (the node a reviewed dream operation or a person merged it
into, §5.6, §5.7; the merged node is archived, never deleted), `rev` (§4.4).
A person corrects a node — its name, aliases and kind — and merges two nodes
through §5.7, never by writing the record directly.

### 4.2 Fact — something true about a node, for a while
`id`, `node_id` (**optional**: a free-standing fact is valid — a user
preference, for instance — and may be attached to a node later by extraction or
a dream), `statement` (short natural-language sentence), optional
`attribute` / `value` for structured facts, `valid_from`, `valid_to` (null =
still true), `origin` (`explicit` | `extracted` | `imported`), `confidence`,
`source_id` + excerpt span, `provenance` (`owner` | `assistant` | `external`,
the part its quote lies in, P6; empty for `remember` and imports), and for an
`external` one `part_kind` (`tool` | `web` | `document` | `other`, empty when
the harness gave none), `part_ref` (the short form of the part's `ref`, P6)
and `read_at` (the day it was read, P6); for an `assistant` one, `part_ref`
holds the assistant's declared name (§4.6), when one was declared — a fact
proposal, the fact its acceptance writes and a session note keep all four; a
dream's consolidation keeps the kind and the ref when every fact it restates
shares them, and the latest day read, `observed_at` (for an
**observed** fact, below, the day it was last observed), `proposal_id` (the fact proposal it was accepted from,
§4.4b), `superseded_by`, `same_as` (the fact stating the same value in another
form, §5.4), `consolidated_into` (the fact a dream consolidated it into,
§5.6 — not a change of value, so it is not `superseded_by`), its
**activities** (in the agent perspective, the activities it was learnt or
done during, below — links beside the subject, never the subject: one for
a rule, every activity of its session for an episode; a fact proposal and
the fact its acceptance writes keep them), `pinned`,
`scope`, `rev` (§4.4). With the comparison form switched on (§5.4), an
extracted fact also keeps its value's **comparison form**, which is not
part of the record: no route, tool, search or prompt ever reads it.

**Chatter is not memory.** Today's weather, greetings, thanks, politeness,
remarks about the conversation itself ("that was helpful", "let's talk later")
and what the assistant says about itself (in the user perspective; in the
agent perspective, below, only its politeness and its remarks about the
conversation) are never facts: the extraction prompt
lists them and says that empty lists are the expected answer for such a turn.
There is no deterministic filter (a word list is language-bound and would drop
"Karim works for a weather service"); the service counts the turns that yield
nothing (`empty_answers`), the bench counts what chatter turns still produce
(§11.1).

**Intent before content.** The extraction reads how something was meant
before what it says, for every fact, whatever its subject. The rule is a
question, not a list of cases: was it meant, as a fact, for good? There is no
word list ("test", "imagine", "si" are language-bound, and "I test cars for a
living" is a fact). The answer carries, in closed words the service checks:
- a **frame** for the source as a whole, read once per answer (read per fact,
  a model contradicts itself inside one turn): `sincere`, `pretend` (a test, a
  role-play, a persona the person or the task sets), `example`, `irony`,
  `voice` (written for someone else, a draft, a translation), `staged` (a test
  page, a demo, a fictitious document), or `unsure`; with `frame_holds` (the
  frame governs the turns that follow) and `frame_ends` (this source ends a
  frame an earlier turn opened);
- a **modality** per fact: `fact`, `question`, `hypothesis`, `plan` (an
  outcome planned or considered, not reached), `guess`, `proposal` (what the
  assistant suggested and the person did not accept), or `unsure`. A decision
  or a plan may be a `fact` about the intention itself ("plans to move in
  March"), never about the outcome;
- a **duration** per fact: `lasting`, `task` (a request about the task at
  hand, the state of something being built, a step, an intermediate result, a passing mood —
  "I'm fed up with my job", said one bad evening), `observed` (the state of
  something that changes without anyone telling the memory, seen at one
  moment by a tool, a page or the assistant: a stock level, a service's
  status, a queue's length, what a page shows — never a count or a result of
  the work in progress, which is `task`), or `unsure`.

The model reads; the service decides. Only a `sincere` frame and a `fact`
modality write a fact. A non-sincere frame writes nothing from its content —
no fact, no node, no edge, not even the co-occurrence of the known names it
mentions — but its activity is true, so it may leave **one
episode**: a sincere fact about the person speaking ("rehearsed a job interview
with the assistant as the recruiter"), marked `episode` in the answer, stored under the
key `activity` with no value (an episode never replaces another), flagged
`episode`, dated by its source; never more than one per frame, and one that
holds a value of the content it reframes (whole words, folded) is only
proposed. An episode about anyone else is dropped. A frame or a modality that
is `unsure`, outside the closed list, or not given at all makes a **fact
proposal** (§4.4b), never a fact — a wrong fact costs more than a missing one —
on the node its subject already names, never creating one. A modality other
than `fact` drops the fact, and an edge quoting the same words goes with it; a
node named only by what was dropped is not created. Words reported or quoted
are a fact about whoever said them (the prompt says so). A `task` or `unsure`
duration keeps the fact as **session state** (§5.8), never as memory; an
`unsure` duration is not a proposal (a work session would fill the review
queue), the session's consolidation decides.

**What was observed is true at a date.** An `observed` fact is memory, not
session state, but it holds only as of the day it was seen: "the item is
out of stock", read by a tool a week ago, says nothing about the stock today. The
model reads it (the `observed` duration), and the service reads it from
provenance (P6): a lasting fact whose quote lies in an `external` part of kind
`tool` is observed whatever the model said — a tool's output is a snapshot.
Other external kinds (`web`, `document`, `other`) follow the model's reading
(a page's author does not change every day); a `task` or `unsure` fact stays
session state wherever it lies; with the intent reading off, the provenance
rule alone applies. An observed fact is written like a lasting one, flagged
`observed`, with `observed_at` the day its source was read (P6's day read). A
fact proposal keeps both, and its acceptance writes them. In a session the host
named, a fact read as observed (by the model, or by the provenance rule for a
quote in a `tool` part) also leaves a **session note** with the duration
`observed` (§5.8): the work of a session may be read as observations only, and
its close must still say what was done. The note is never kept by the close,
is not counted as expired, and goes with the session.
- **Seen again.** A later observation that states a held observed fact's
  value again (the same statement, or a value the comparison of §5.4 finds
  `same`) writes nothing new and moves the held fact's `observed_at` to the
  later day. Only an observation does: the owner saying it again leaves the
  date as it is.
- **Replaced by a newer one.** A newer observation of the same key on the
  same subject closes the older one (§5.4, **Observations**).
- **Weighs less as it ages.** Recall ranks an observed fact by its score
  times 2^(−age / `recall.observed_half_life_days`), the age counted in days
  from `observed_at` (7 by default; 0 turns it off): it moves down the ranking,
  is never removed from it, and is never closed by age. A lasting fact never
  ages in the ranking.
- **Shown with its date.** Its handles carry the flag `observed`, and their
  `age_days` counts from `observed_at` (§8.5).
- **Consolidated.** A dream's consolidation of facts that are all observed
  is observed, as of the latest of their days; otherwise it is lasting.
The job report counts the observed facts written (`facts.observed`, those
the provenance rule made so apart, `facts.observed_by_tool`) and the ones seen
again (`facts.observed_again`).

A frame that holds is carried by the session (§5.8) to its next turns, shown
in their prompt, and ends when a turn says so or the session closes. A turn
read as `sincere` while a carried frame is open, without ending it, is a
disagreement, so a doubt: its facts are proposed. The job report counts what
the reading did: `dropped_by_intent` per frame and per modality, the facts it
proposed, the facts kept as session state, the episodes kept, proposed and
dropped, the frames opened, carried and ended. What a real model reads is
measured by the `--intent` corpus (§11.1), not promised by the mechanics;
`extraction.intent = false` turns the reading off (every fact is then read as
sincere, a fact, lasting, and the prompt does not ask).

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

**The agent's own memory.** A scope keeps the **user perspective** unless the
operator declares a **self** for it (§4.6): the memory is then also the
agent's own, the **agent perspective**. In the user perspective the agent is
the reader of the memory, never its subject. In the agent perspective:
- the assistant part is labelled with the self's name, and the prompt says
  it is the self speaking in the first person: a fact about itself takes the
  self's name, in exactly the declared spelling, as its subject. A lesson
  learnt, a view held, a commitment, what was done, the relation with the
  owner are memory; its politeness and its remarks about the conversation
  stay chatter; a text it writes for someone else, a translation, a role the
  owner set and a suggestion the owner did not take are read by the intent
  reading above, as for anyone;
- a **rule of method** ("to do X, do Y": a step, an order to follow, a
  format or a convention to apply, something to always or never do) is a
  lesson about a **task**, whatever its source — learnt by doing, or read
  in a guide and taken up: one fact per rule, its subject the task it
  serves, a node of kind `task` named by the activity ("filing a quarterly
  report"), never the self, the guide it was read in nor the object it was
  applied to, under the key `method`, an element of a set (the service
  reads `method` as `many` whatever the answer says), lasting. A limit or a
  property of a thing (a maximum size) is not a rule: it is a fact about
  that thing. The prompt says all of this; the service cannot check it;
- the **activity**: what the owner asks the agent to do is what a later
  request repeats in other words, and a rule never resembles that request
  ("label each bed with its sowing date" against "make me a map of the
  balcony"). So the answer
  names the activity the owner asks for in the source, if any:
  `"activity": {"name": "...", "quote": "..."}` or null, its name in the
  owner's words and language, a short phrase that keeps the noun of what
  is made ("drawing a map": a conjugated verb does not match a later
  request, the noun does) and leaves out what it is made of or for ("drawing
  a map", never "drawing a map of the garden": the next map is of something
  else, and two names would split one activity). A new thing to make is a
  new activity, even when it serves the one before (a garden plan, then a
  map of the plan, then labels for the map: three activities).
  The service checks it, else drops it and counts
  it (`activity.refused`): the answer's frame is sincere (above); the quote
  lies in an owner part (P6: neither the reply nor a page sets the
  activity); a word of the name is written whole in the quote — a word of
  three letters or more that is not a function word of a language the
  service detects (P12), or two letters in a row for a script written
  without spaces; the name obeys the name rule (§5.1), mixes no two
  scripts in one word and is not the word for the person speaking (no
  node bears such a name, §4.2). A name that is,
  normalised (§5.1 step 1), the name or an alias of exactly one visible,
  live node of kind `task` is that node; held by a node of another kind or
  by two `task` nodes, it is refused (when we cannot tell, nothing
  attaches); otherwise a `task` node is created in the source's scopes. A
  near or a containing name never attaches: an activity split in two is
  visible and a merge proposal joins them, a rule hung on the wrong
  activity is silent (§5.1, the asymmetry of entities);
- in a session the host named, the activity becomes the session's
  **current activity** (§5.8) until a later source of the session names
  another, so a rule read three turns later belongs to what the owner asked
  first; a source without a session id keeps its activity for itself. The
  prompt shows the current activity as a known subject and says that a rule
  learnt or read during it takes it as subject. The check for a known
  subject the source does not name (§5.1) never applies to the current
  activity: the service set it from the owner's own words;
- a `method` fact written or proposed from a source with an activity (its
  own, else the session's current one) is linked to that activity, whatever
  subject the model gave the rule (the guide, a tool): the subject is never
  moved, so a wrong link harms no subject and shows. A link to a merged
  activity is read through `merged_into`; forgetting an activity's node
  deletes the links to it, and forgetting a fact its own (P5). The
  session's episode (§5.8) is linked to every activity the session named;
  a rule kept at the close is linked to none, since which activity was
  current at its turn is not kept (a missing link shows, a wrong one does
  not); a dream's
  consolidation (§5.6) keeps the activities every original shares;
  `expand` shows them (§8.5);
- the tool, the guide or the service used for an activity is linked to it
  by an edge `uses` from the activity, which the prompt asks for and which
  is written like any edge (§4.3, §5.2): proposed, gaining evidence, never
  a fact. It joins the activity to what it was done with, so that a later
  request for the activity reaches them one hop away;
- a fact about the self and a `method` fact on any subject are **self-side**.
  A self-side fact is written only when its quote lies in an owner part
  (and its value does not come from outside alone, P6), or in the assistant
  part or a vouched external part of a source whose external parts are all
  vouched (none at all included). Which facts are self-side is the model's
  reading — the subject it names and the key it gives: a rule a model writes
  as an ordinary fact about an object (`format` on "the card") is not
  self-side, and is written like any fact the assistant states (measured
  with a Qwen3.8 27B variant on the `--self` corpus: both rules of a page the
  agent found on its own, keyed `format`). That is a known limit, not a
  guarantee. Otherwise it is a fact proposal (`why:
  self:outside`), even when the owner asked in that turn for it to be kept:
  the service cannot tell such a request from any other sentence of the
  owner, and a model fooled by the page would cite one; the owner approves
  it in review. **Vouching** is how the agent learns from what it reads for
  the owner: a guide the owner attached, the pages of a search the owner
  asked for. It is trust the host gives, not something iifa can check: a
  hostile page inside a vouched search teaches the agent like any other.
  So a self-side fact written from a source holding a vouched part is
  flagged `vouched`, which a reviewer can list and archive; vouching never
  changes the owner's rule above, and a source holding one part that is not
  vouched gives no self-side fact at all. The rule is per
  source: what a page said in an earlier turn and the reply restates later
  is not seen (a known limit, as for the owner's facts);
- **an echo is not evidence**: a self-side fact from the assistant part that
  states again a live self-side fact of its node (the same statement, or the
  same value of the same key) writes nothing, closes nothing and moves no
  `observed_at` — otherwise the agent's own mistakes ("I fixed it"), recalled
  into its context and said again, would confirm themselves;
- a self-side fact closes only self-side facts of its own provenance: the
  owner's correction of the agent ("no, it was not fixed") stays beside the
  agent's fact as a visible conflict, the agent never closes what the owner
  said about it, and a reviewer settles it;
- an edge touching the self's node gains evidence (§5.2) only from a quote in
  an owner part, and the self's node takes no co-occurrence: its label stands
  in front of every assistant sentence;
- a **commitment** of the self ("from now on I confirm the date before
  booking a table") takes a key of the family `commitment.<topic>`, one current value
  per topic like `status.*`: a commitment changed or withdrawn on the same
  topic replaces the old one, within the provenance rule above. The profile
  (§8.1) carries the self's live rules: its commitments and its preferences
  (`preference.*` on the self's node), which a later session must start with
  whatever its first query is;
- **a self's rules may wait for a reviewer** (`review_self_rules`, off by
  default). A rule is a fact about the self whose key is of the family
  `commitment` or `preference`: what the agent commits to, how it prefers to
  behave. With the setting on, a turn never writes one, whoever said it —
  the owner asking for it or the agent taking it on: it is a fact proposal
  (`why: self:rule`), and only a reviewer makes it memory. An operator turns
  it on when a rule the agent follows in every later session must be one a
  person has read. A rule from outside keeps its own reason
  (`self:outside`). The agent states its rules again in most replies, so a
  rule is asked about **once**: one that restates a live fact of the self's
  node, or a pending or a rejected proposal about it (the same statement, or
  the same value of the same key, or of the same topic under the other rule
  family: a model files one rule as `commitment.<topic>` in one turn and as
  `preference.<topic>` in the next), writes and proposes nothing, counted as
  an echo — a rule the reviewer refused is not asked about again, another
  value of the same topic is. The model that extracts words a rule
  differently from one turn to the next (another statement, another value,
  another topic), which no comparison of words finds: a rule that restates
  none word for word goes to the chat model, in one batched call before the
  write transaction, with the known rules of the self's node — its live
  facts of a rule family, the pending and the rejected rule proposals, read
  through the write's view (P9) — at most five per rule, the nearest by
  embeddings when there are more (without an embeddings endpoint, or when
  it fails, the held ones first, then the waiting, then the refused;
  embeddings only rank, no threshold decides). "The same rule?":
  only `same` makes it an echo; `different`, `unsure`, an invalid or a
  truncated answer leave it asked about — a new rule the reviewer never sees
  costs more than one asked twice (counted `self_rules.compared`). An
  accepted commitment with a topic replaces the live one of that topic, as
  it would have without review: the reviewer's accept is the decision, and
  the review queue names what it replaces (§4.4b). An accepted preference
  closes nothing (two preferences on a topic may both be true). What the agent did or
  learnt, and its rules of method, are written as without the setting;
- the session's close writes its episode about the self (§5.8), and the
  agent may write about itself through `remember` (§8.3).
The job report counts `facts.self` (written about the self), `facts.method`
(rules written), `facts.self_proposed` and `facts.self_echo`, and `activity`:
`named` (an activity checked and set by the source), `carried` (the
session's current one used), `refused` and `linked` (facts linked to an
activity). Nothing of this applies in the user perspective: its prompt
is unchanged, and an `activity` in its answer is refused as an unknown key.

An **extracted** statement must be a complete sentence: at least three words
(at least six characters in a script written without spaces — Chinese,
Japanese), and, when the fact has a subject, it names that subject: its name,
one of its aliases (stored, or given by the answer and found in the source), or
a mention of the same answer that resolves to the same node — tolerant of
inflection and of French/Italian elision ("d'Émilie" names Émilie). Only names
written in the statement's script count: "Карим переехал в Лион." under the
subject "Karim" names it only when the answer also spells it "Карим" (an alias
the source holds). A name written in the statement as it is, whole words
(case and diacritics folded), names its subject whatever the statement's main
script: a file, a product or a handle spelt in Latin letters inside a Russian,
Arabic or Greek sentence ("report-2.pdf — корректный документ, страниц: 4."). The extraction prompt asks to write the subject as the
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
reference. Facts and edges point into sources by span. A turn's source also
records what external content it held (`external`: `none`, `vouched` when
every external part was vouched, else `unvouched`, P6): the parts' offsets
are not kept, and the agent perspective's rule (§4.2) must still be read
when a session's close writes from the source later (§5.8).

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
(rule 4); a person's correction does the same (§5.7).

An edge also carries a **decision revision**, kept by the same triggers, that
grows only when what a person decides about the edge changes: its status, its
life (`valid_to`, archived or restored), its ends, its type or its scope set —
never its weight, its use counters or its evidence (reinforcement, decay that
leaves it open, a new source). Through the API, an edge's `rev` is always this
decision revision (the node view, `/link`, `/edges/{id}` and its 409 `stale`);
no route shows the other under that name. The reason: an edge is reinforced
by every recall the agent expands and weakened by every session's decay, so a
person who read an edge would otherwise be told it moved when nothing they
decide on did. The dream and the journal's `after_revs` keep the full `rev`:
undoing a decay or a merge must still see a reinforcement made since, or it
would write an old weight over it.

### 4.4b Proposal — something the write path would not decide alone
`id`, `kind` (`alias`; `merge` from tier 3, §6.1, or from one subject named
in two forms, §5.1, whose `value` is the other node's id; `fact`), target
`node_id` (optional for a `fact` proposal: a free speaker fact has no node),
`value`, `source_id` +
excerpt span, `confidence`, `created_at`, `status` (`pending` | `accepted` |
`rejected`), and the **scope set of its source**. A proposal carries source
text, so it is a record like the others: read through the view (P9), deleted
with its node or its source (P5), never used by resolution or recall. Accepting
applies it (an alias proposal adds the alias to the node; `write` permission);
rejecting only changes its status. A `merge` proposal is never applied by
the write path nor by a `write` token: it is an input of the dream, whose
adjudication (§5.6) reads both nodes and, on `same`, queues a `merge_nodes`
operation for review; when that operation is accepted or rejected, the
proposal follows. A reviewer may also decide a pending `merge` proposal
directly in the review queue: rejecting it (the dream then never asks about
the pair again), or accepting it, which merges the two nodes as a person's
decision (§5.7). Pending
`alias`, `fact` and `merge` proposals appear in the **review queue** (§8.6)
beside the dream's queued operations — a `merge` proposal with the latest
adjudication of its pair (verdict and reason), and not when its `merge_nodes`
is already queued (that item shows the same verdict): one listing, one
`review` permission, so a reviewer never has to look in two places. An item
about nodes (an `alias`, a `fact` on a node, a `merge` and a queued
operation, whatever it is) names them: each node the view sees, by name, kind
and aliases with its first three facts (pinned first, then newest), so a
question about several subjects reads without opening them; a queued
operation also states the facts it names (the facts a consolidation replaces,
the two facts of a supersession). A `fact` proposal whose accept would close
live facts (a commitment of the self, §4.2) states them (`replaces`): the
topic a rule is filed under is the extraction's reading, and a rule filed
under another rule's topic would replace it — the reviewer reads what goes
before deciding. The list is what the accept closes, less what the reading
view does not see (P9): a view narrower than the proposal's own scopes and
their ancestors may be shown fewer facts than the accept closes. A
node or a fact the view does not see is left out (P9). Naming never costs an
item its place in a page: the page holds the items it would hold without
names, and the names share what it leaves, in page order; where that runs
short the facts shown under an item's nodes are left out first, then the
facts it names, then its nodes' names.

A **fact proposal** is a fact the write path would not write on its own:
about the owner and not from the owner's own words (P6), attached to a
subject its source shows another subject for (§5.1, neighbouring subjects),
about a known subject neither its source nor its session names that the
adjudication did not attach (§5.1), or about anyone when its intent is in doubt (§4.2; its `why` says
`intent:frame:…`, `intent:modality:…` or `intent:episode`). Its `value` is
the statement, and
it keeps the rest of the fact (key, value, dates, language, flags, the
provenance of its part, its activity). It is decided — accepted **or** rejected — only by a
token holding the **`review`** permission, which an agent token does not hold:
an agent manipulated by the very page the proposal came from must not be able
to turn it into memory, nor to hide it by rejecting it. Accepting writes an
**`explicit`** fact (a human vouched for it, P6) that keeps its provenance: the
original source and span, the part's provenance, and `proposal_id`. Like
`remember`, it closes nothing.

### 4.5 Event — a dated happening linking nodes
Optional; needed for timeline queries ("what happened with X in June"). An
**episode** (§4.2, §5.8) — a test that took place, a session's work — is a
fact under the key `activity`, not an event, so that recall and the timeline,
which list facts, both answer "did we already test this?".

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

**Re-scoping a fact.** `PATCH /v1/facts/{id}` with `scopes` replaces a fact's
scope set — to share a fact learnt in `user:alice/agent:coder` with every
agent of `user:alice`, or with a chosen few. A token changes only the scopes it
is bound to: every scope the patch **adds** and every scope it **removes** must
be at or below one of the token's bound scopes, so a scope the token does not
own is neither granted nor taken away, and is kept as it is. The new set is
never empty. The fact stays where its node is: archiving or forgetting a node
reaches its facts, so a new set from which the fact's node is not visible
(sharing none of its scopes, the rule for attaching) is refused with 409
unless the same patch detaches the fact or names a node visible from the new
set. Otherwise re-scoping moves the fact alone: its source and the facts it
points to (`conflicts_with`, `superseded_by`, `same_as`) keep their own
scopes, and the usual rules show them — a reader who sees the fact but not its
source gets it without the excerpt. The lexical statistics follow the new set at once, so
the fact counts for the scopes it now belongs to and no longer for the ones it
left (P9); the revision grows (§4.4), so a dream that read the fact before
the change applies nothing on it.

**Owners.** The operator may declare, per scope, who the memory belongs to:
`owners = {"user:alice": {name = "Alice", aliases = ["Алиса"]}}`. A
declaration holds for the scope and every scope below it, like a language
override (P12); the nearest one wins. A declaration on `*` is refused when
the configuration loads: it would make one person the speaker of every scope,
a shared one included. A source written into several scopes
has an owner only when every one of them leads to the same declaration;
otherwise who speaks cannot be told, and its speaker facts stay free-standing.
The owner's node is found by its exact
name in the declared scope, or created there (`person`, the configured aliases)
the first time a fact about the owner is written — a decision of the operator,
not of a token, so no visibility is widened; the facts themselves belong to the
scopes of their source, as always.

**Assistants.** The operator may also name, per scope, the assistant whose
answers a turn carries: `assistants = {"user:alice": "NOVA"}`, inherited and
resolved like the owners (the nearest declaration, and a source split between
two has none); a declaration on `*` names it everywhere. Without one the assistant is `Assistant`, as before. The name
labels the assistant part of a source ("NOVA: …"), the extraction prompt
says that this label is the assistant, not the person speaking, and a fact
from that part says `"NOVA"` in `from` (P6, §8.5). It changes no rule: the
part's provenance is still `assistant`. A declared name is plain text and
never one of the service's own labels (`User`, `Tool`, `Web`, `Document`,
`External`) nor the owner's name or alias in that scope, so a label always
tells whose words a quote is.

**Selves.** The operator may declare, per scope, that the memory is also the
agent's own (the agent perspective, §4.2):
`selves = {"user:alice": {name = "NOVA", aliases = ["Nova"]}}`, inherited and
resolved like the owners (the nearest declaration; a source split between two
has none). A declaration on `*` is refused, like an owner's. A self is the
assistant of its scope: its name labels the assistant part as an assistant
declaration does, an assistant declared on the same scope under another name
is refused, and the self's name and aliases obey the assistant's rules (plain
text, never a service label, never the owner's name or alias in a related
scope). The self's node is found by its exact name in the declared scope, or
created there (kind `other`, the configured aliases) the first time a fact
about it is written, like the owner's.

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
   names that may write **one subject in two forms** (below), and, at
   tier 3, names close by embedding (§6.1; never a merge of two existing
   nodes);
4. LLM adjudication for the candidates (tier 2+), batched. What it attaches
   stays attached for the rest of the source, whatever new nodes the answer
   proposes afterwards.
Steps 1 and 2 attach a node memory already holds only when the source or its
session names it (below).
An unresolved mention creates a new node: so does a candidate without a chat
model, or when the adjudication's answer is truncated. Merges are possible
later (`merge_nodes`) and keep both histories.

**A known subject the source does not name.** The extraction prompt shows the
model the names memory already holds, so a model may write one of them as the
subject of something the source says about another object: "Omar's report"
written under the name of the report memory knows ("Nadia's annual report"), or
under a word of it ("report"). Steps 1 and 2 would attach the fact to the known
node, and nothing would show it. So a mention matched by step 1 or 2 to a node
memory already holds is a **candidate only** when neither the source nor an
earlier source of the same session (the host's `session_id`) writes one of that
node's names or aliases: whole words, the same spelling in another script, or,
for a name in a script written without spaces, its letters. A word of the name
is not enough: "Karim" attaches to Karim Benali on its own in a source or a
session that wrote the full name, and goes to the adjudication in another one.
The adjudication sees the sentence the fact quotes (or the start of the source,
for an edge) and the known node. It attaches, or the mention stays unattached:
its facts become **fact proposals on the known node** (§4.4b, counted
`facts.unnamed`), its edges are not written, the aliases the answer gives that
node are alias proposals, and no node is created under a name memory already
holds. The owner is named by the label of their part and is never judged here,
nor is the activity the source or its session set from the owner's words
(§4.2, agent perspective); a node the same answer creates is not a known one. The cost is one adjudication
for a source that names a known subject only by a pronoun, a translation or a
word of its name, outside the session that named it (`adjudication_calls`).
Limits: a session is taken as context as a whole, so a session that named one
report and then speaks of another by the same words is not caught; the
adjudication can still be wrong, and then attaches as any adjudication does.

For entities the asymmetry is the reverse of values' (§5.4, where a false
"same" only keeps a conflict open). A **wrong attachment** silently corrupts
another person's memory: Ivana's facts written on Ivan's node are recalled as
his, close his own, and nothing shows it. A **split node** is visible, loses
nothing, and a dream can propose the merge (§5.6). So a resemblance of
letters attaches nothing by itself: if it did, whenever a single known name
was alike, which of Ivan and Ivana got the other's facts would depend on who
was named first.

**One subject in two forms.** A file or a page is often named once
by its file name or slug and once by a title: "the-north-garden.pdf" and
"Plan north garden", "the north garden" and "Plan: the north garden".
Normalisation (step 1) already reads any separator (`-`, `_`, `.`) as a space
and folds case, so "the-north-garden" and "The north garden" are one name.
For the rest, two names are **in two forms of one subject** when their
words — split on any separator, folded, without a file extension (a final
dot right after a letter or digit, then one to five letters or digits, the
first a letter: "North garden.pdf") — have at least two words in
common, more words in common than words apart, and at most one word of each
that the other does not hold ("the" and "plan" above). The rule reads only how names are
written, never a list of words of one language. It makes a **candidate**,
never a match: an article or a kind word may be what tells two subjects
apart ("Plan north garden" and "Photo north garden" are candidates of
each other), so the adjudication decides, and without a chat model the
mention becomes a new node. "Project garden blue" and "Project garden green" (two
words in common, two apart) are not even candidates. When the mention
becomes a new node anyway (the adjudication says "new", its answer is
truncated, or there is no chat model) and one of the two names is
**written as a slug** — a file name; two words or more joined without
spaces by an underscore, or by any separator without a capital
("the-north-garden"; a hyphenated proper name, "Jean-Pierre" or
"Saint-Étienne", is a name, not a slug); or cased letters without a capital
where the other name has capitals — the service records a **`merge`
proposal** (§4.4b, counted `duplicates.merge_proposed`) between the new node
and the known one, when both nodes have compatible kinds and the same scope
set (dream rule 12) — also when one source names both forms and both nodes
are new. The split is visible in the review queue and the dream adjudicates
it on names and facts (only for nodes of the same kind: a pair whose kind one
side left open waits for a reviewer). Nothing is merged at write time. A
merge proposal makes the two nodes **known duplicates** (§5.4). An alias is
stored folded (no case, no dots), so only display names are read for this.

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
adjudication could see it: a Qwen3.8 27B variant gave Иван the
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
wrong alias merges for good. A description the source does **not** hold is
the model's own words, not a name anyone used: it is dropped, never proposed
(still counted `aliases.descriptive`) — "nom de plume pas encore choisi" written
for a name the source says is not known yet would otherwise sit in the review
queue as a candidate alias. The check reads only the shape and the source, in
any language.

**A name is plain text.** A model often answers in Markdown, and the source
it reads may be a reply written in Markdown, so every node name, alias and
alias proposal — from an extraction, `remember`, an accepted proposal, a
dream's theme or merge — goes through one rule, applied by the store itself
so that no path can bypass it, and by each entry point before it so that a
refusal is counted or answered there: Markdown is stripped (bold, italics,
strike-through, code ticks, link syntax keeping its text, heading, quote and
bullet markers; a `*` or `_` inside a word or without a partner stays,
`john_doe`, `*NSYNC`, and so does a
numbered item, "1. FC Köln"), then a name whose brackets or double quotes do
not balance — `()`, `[]`, `{}`, the CJK and full-width brackets, guillemets
and curly quotes counted in either direction (»x« and «x», „x“ and “x”) — is
refused, never repaired: "Name A (Name B" may stand for "Name A
(Name B)" or for "Name A", and a guess would make one of them an
alias for good. An apostrophe is not a quote (O'Brien, d'Artagnan). The one
completion made is the source's own: the span of the source an alias is
stored as ends on a word, so "Lyon (bureau)" reads "Lyon (bureau" until the
bracket the source holds right next to it closes it. A refused node name, fact
subject or edge end refuses its item (counted in `rejected`); a refused alias
is dropped, never proposed (`aliases.malformed`); `remember` answers 422; a
stored alias proposal that is no name cannot be accepted (422), only
rejected. A stored alias is never rewritten by the rule; it applies on write.

**A name not given is no name.** When the source says a name exists but does
not give it ("her pen name is not chosen yet"), the extraction answer
says so structurally: `null` in `aliases`. It is dropped and counted
(`aliases.unknown`), never an alias, never a proposal; the information, if it
matters, is a fact.

**Neighbouring subjects.** A turn often describes two subjects side by side
("backup.zip holds three folders. report.pdf has twelve pages."), and a
model may attach one's sentence to the other, or give one a value that
mixes both ("twelve pages and three folders" as the archive's content). When the source has
two subjects or more — the nodes the answer proposes, the nodes its facts
and edges resolve to, the known nodes the text names (tier 0), each with its
names, aliases and the mentions that resolved to it — a fact is **not
attached on its own** and becomes a fact proposal (§4.4b, counted
`facts.neighbour`) when:
- its **quote names another subject** — one of its names or aliases, whole
  words — **and not its own**: none of its names, aliases or the mentions
  that resolved to it is written there in full (or in another script with
  the same spelling, Иван for Ivan) once the other subjects' names are set
  aside. Never a near or an inflected form ("Karima" is not Karim; "Ивана"
  costs a proposal), never a form made only of other subjects' words (an
  alias "PDF" beside report.pdf), and a part of a longer name is that name's
  ("Karim" inside "Karim Benali", when they are two nodes). Another subject
  that the fact's statement or value also names, or that an edge of the
  answer links to the fact's subject, is its object, not a neighbour ("Karim
  leads the solar project", quoted "leads the solar project"). A fact about
  the owner or the speaker is not judged by its quote: the person speaking is
  named by the label of their part, and pronouns are not read;
- its **value names both** its own subject and another one: it holds a word
  of each one's names that the other subjects' names do not share, of at
  least three letters and not digits only ("zip" of backup.zip, "pdf" of
  report.pdf; in backup.zip beside photos.zip, "zip" names neither) — unless
  the value names that other subject in full, as its object ("a zip archive
  holding report.pdf").
The asymmetry of entities again: a proposal loses nothing and a reviewer
decides it; a wrong attachment is silent. The rule reads names only, in every
language, with no word list. Its limits: a subject that the answer does not
propose and memory does not know is no subject here; a quote that names no
subject at all is not judged; scripts written without spaces are not split
into words; a value that describes another subject without any word of its
names is not seen; a model that also writes the neighbour's name into the
fact's statement makes it an object, the price of reading verb-first quotes
of relations as they are.

### 5.2 Link proposal
Edges are **proposed** from: co-occurrence in the same **sentence** of a
source (not merely the same turn: a turn about two unrelated people must not
link them; a node is in a sentence when its name, an alias, the mention that
resolved to it or the quote of one of its facts or edges lies there), explicit
relation phrases found by extraction, and (tier 3) semantic proximity between
node summaries (`similar_to`, §6.1). Proposed edges are usable in recall at reduced weight.
An edge becomes **confirmed** when: stated explicitly, re-observed in N
independent sources (configurable), or confirmed by a person (§5.7). An
explicit link between two nodes that already share a live edge of that type
(the same direction; either direction for `co_occurs`) visible to the
writer confirms that edge rather than adding a twin.

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
  duplicate and a `same` value is kept or links, as below.

**Observations** (§4.2). A new `observed` fact and a group of the same key
whose facts are all `observed` are two readings of one state: past the guards
above (an explicit fact, a free speaker fact, an external fact against one
that is not), a different value from a **later source** — every fact of the
group from another source, and its `observed_at` not earlier than any of
theirs — closes the group without the replacement or the restatement
question, whatever the key ("the shelf holds 12 jars" then "the shelf
holds the spices", both under `content`: the shelf's current content is the
newer one). An older observation arriving late is kept beside it with the
`conflict` flag; two values observed in one source both stay live, neither
closed nor in conflict. The same value is a duplicate or a restatement, as
below, and is seen again (§4.2). An observation against lasting facts, or a
lasting fact against observed ones, follows the ordinary rules.

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
  `different`; another unit or wording `unknown`. Unit symbols of one meaning
  are one unit: `o` (octet) and `B` (byte), bare or with a decimal prefix
  (`Ko` / `ko` / `kB` / `KB`, `Mo` / `MB`, `Go` / `GB`, `To`, `Po`), so
  "120 o" and "120 B" are `same`. Case-sensitive on purpose (`Mb` is a
  megabit, `b` a bit, a lower-case `mo` a month: "100 MB" against "100 Mb" is
  `unknown`; any other unit is compared folded), and nothing is converted:
  "1 Ko" against "1000 o" is `unknown`;
- **names** (places, employers…) are transliterated **for comparison only,
  never stored**: Cyrillic and Greek to plain Latin, kana to Hepburn, and for
  Arabic and Hebrew, which leave vowels out, the consonant skeletons are
  compared. An equal form (or one letter apart on a long name) is `same`. A
  different form is **never** `different`: exonyms (Londres / London, Pékin /
  北京) look nothing alike. Chinese characters have no table here: `unknown`.

Within one language, values that differ as strings supersede, unless
the comparison says `same` or the restatement question answers `same`
(**Restatements**, below). With the **replacement question** (`[extraction]
replacement_question`, on by default, see below), `role`, `team`
and `preference.*` are the exception, since they may hold
several values at once ("je suis aussi photographe" after "je suis comptable"
must not close the accountant): within one language too, a different value
of these keys goes to the replacement question ("replaces, or both true?"),
and without an answer — no chat model, a truncated or invalid one, `unsure` —
both stay live with a visible conflict rather than an automatic close; a
comparison saying `different` (another date or number) does not settle them
either: a different value may be an addition. **Free-form keys** (outside the
vocabulary, below) holding one value (`cardinality` `one`, not a set) take
the same question, within one language and across languages: a key is a slug
the model chose, and a generic one (`content`, `info`, `details`) may carry
two unrelated statements about one subject ("the shelf holds 12 jars" and
"spare jars are kept on this shelf"), so a shared key alone does not
prove that a new value replaces the old one. Nothing reads the key's words to
tell a vague key from a precise one (`content` and `size` have the same
shape); the question reads the two statements and their quotes. One
difference with role, team and preferences: a comparison saying `different`
(the same value with another number or date, "12 jars" then "13 jars")
still supersedes a later value without a question, since the rest of the
value is the same aspect. Across languages:

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
  `employer`, `status.*`, `date.*`, `commitment.*`; never `role` or `team`, which may hold
  several values) and only when the new fact is later: "do these two values
  name the same thing? same / different / unsure". `different` supersedes,
  `same` links, and `unsure`, an invalid answer or anything else leaves both
  facts live with the `conflict` flag. The answer goes through a
  strict schema (P8). The bench measures false supersessions with `--traps`;
  if they rise, the adjudication is switched off.
  With the replacement question switched on, for `role`, `team` and
  `preference.*`, which may hold several values at once, and for free-form
  keys of one value, which may hold two unrelated statements, the question is
  another, in a call of its own: "does the new value
  replace the old one, or can both be true at once? same / replaces / both /
  unsure" ("now works as an accountant" replaces; "is also a photographer"
  adds; two statements about unrelated matters under one generic key are
  both true). `replaces` supersedes, `same` links, `both` keeps both facts live
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
  model call per extraction that changes a role, team, preference or
  free-form value, under P11 like every background call. Off
  (`replacement_question = false`), these keys close within one language and
  stay in conflict across languages (free-form keys: unless the comparison
  says `different`).

**Restatements.** A value told again within one language, in other words,
another order or another unit symbol ("120 o", then "120 B", then "120 o"; a
list of the same items told differently), is the value already held: the held
fact is **kept** as it is. Nothing is written and nothing is closed; the job
report counts it (`facts.kept_same`). The new fact is written and linked
(`same_as`) instead, as across languages, when keeping would lose something:
a value more precise than the held one (a day refining a month), so that
recall can prefer it; a value the comparison proves different from another
member of the group (a new day beside a month and its linked day); a fact
that says when its value ended (`valid_to`); the first fact not from
`external` content in a group that only has such facts (P6: later external
content must not close what the owner confirmed); free `speaker` facts
(several people may speak in one scope). A value is kept when the
deterministic comparison says `same`, or when the **restatement question**
(`[extraction] restatement_question`, on by default) answers `same`: "is b the
value of a told again, in other words or units? same / different / unsure",
in one batched call of its own before the write transaction, through a strict
schema (P8; stored values are data). It is asked only for a change that would
otherwise supersede within one language and does not say when its value
ended (so never against an explicit fact,
never when P6 keeps the conflict, nor for role, team, preferences or
free-form keys while their own replacement question is on), whose values the comparison does not
prove different, and which share at least half of the shorter value's words (folded;
no word list: "120 o" and "120 octets" share their number, a list told again
its items; "Lyon" and "Paris" share nothing and close without a call); the
pair shows the member in the new statement's language that shares those
words. Only
`same` changes anything: `different`, `unsure`, an invalid or truncated
answer leave the supersession that happens without a model — within one
language, closing a value by its own restatement keeps the current answer
right (see **Undetermined languages**), so the question stops the churn of
closures, not a wrong answer. A `same` **never closes nor rewrites** a fact:
the kept fact stays as it is, and another group of the key is closed only when
the new value proves it outdated on its own, by the kept fact, exactly as for a
duplicate of an equal value. A false `same` drops the new value and leaves
the held one current: it never closes a value. The replacement question's `same` keeps the held fact the
same way within one language; across languages, `same` links (above). The
verdicts are counted as `values.restatement`, beside `values.adjudicated` and
`values.replacement`. Off (`restatement_question = false`), a change within
one language supersedes unless the comparison says `same`. Cost: one batched
call per extraction holding such a change, under P11.

**A comparison form of the value** (`[extraction] value_norm`, off by
default). Exonyms and translated words
("Londres" / "London", "comptable" / "accountant") are never proved by
the comparison above, so across languages they cost a model call and stay
in conflict without a clear answer. Switched on, the extraction prompt asks
for each fact with a value a `value_norm` too: the same value in one
canonical form (a date in ISO, a number with an English unit, a place,
organisation or job by its usual English name, anything else in short
lowercase English), through the same closed schema (at most 200
characters; a longer form is dropped, never its fact). P12 is untouched: the statement and the value stay in their
source's language; the form is stored beside the fact **for comparison
only** — never shown in a response (recall, expand, a node, the timeline,
`at`, lookup, proposals, a job report), never indexed for search, never
sent to a model (extraction, adjudications, summaries, dreams). `forget`
deletes it with the fact (P5).

A form is a model's claim, so it may only ever turn a pair into `same`,
never into `different` (a false `same` only keeps a fact live; a false
`different` closes one still true). Exactly:

- it is read only **across languages**, after the P6 rules (an explicit
  fact, a free speaker fact, external content against what is not) have
  kept their conflict — never against an explicit fact, never to let
  external content join what the owner said nor the owner's words join
  external content; within one language nothing changes;
- it is read only when the deterministic comparison is `unknown`: a
  value proved `same` is linked anyway, one proved `different` (another
  date or number) is not overruled;
- when both facts carry a form and the two forms are equal (case and
  spaces aside, no other tolerance: "painter" is not "printer"), the new
  fact is **linked** (`same_as`, no conflict),
  without asking the chat model — for every key, earlier or later, role,
  team and preferences included. Forms that differ prove nothing: the
  pair goes on as without them (the adjudication, else the conflict).

The job report counts these links (`values.norm`). The bench measures the
switch with `--value-norm` (§11.1).

Superseding a fact closes its whole equivalence group (the "Лион" linked to
"Lyon" states the same, now-outdated value). A supersession closes, never
erases (P7): a wrong one stays recoverable. An `explicit` fact is still never
closed by an extracted one (P6): `same` links it, anything else flags the
conflict. `forget` of a group's first fact re-roots the group on the next one
(P5: no reference to a forgotten fact remains). A value's own language is not
used: values are too short to detect; the statements' languages are.

**A new value reaches a known duplicate.** Two live nodes are **known
duplicates** when a pending `merge` proposal pairs them (§4.4b; an accepted
one has already merged them, a rejected one says they are two). A
correction that names only one of them would otherwise change the value
there and leave the outdated one current on the other. So when extraction
writes a value (or restates one it already holds), every visible live
equivalence group of a known duplicate with the same key that the new fact
would close or leave in conflict on one node (by the rules above) is
**flagged `conflict`** against it (counted `duplicates.flagged`) — never
closed: until a merge is decided we cannot tell that the two nodes are one
subject, and a flag only stops the stale value from being silently current
(recall reports the conflict). This holds whether or not the corrected node
held that key. A group holding the new value, or the same value in another
form (`same`), is left alone; so is a fact already in conflict. The dream's
own operations (`supersede_fact`) do not reach a known duplicate.

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
   leave") and `date.<event>` (`date.birth`, `date.deadline`); the agent
   perspective adds a fourth, `commitment.<topic>` (§4.2), named only in
   its own prompt line, never in the shared vocabulary. There is no
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
decides, in one language as across languages. Nor does it hold for a
generic free-form key, which may carry two unrelated statements: free-form
keys take the same question. A key aligned
wrongly still closes a fact that is still true.
The benchmark therefore counts **false supersessions** (a closed fact whose
replacement is not a later value of the same person's same attribute), with
traps built for it (`--traps`: where someone works next to where they live,
and values **added** rather than replaced — a second role, a second team, a
second favourite drink; a value closed by an addition is counted apart as a
**false replaces**);
a key or a synonym that produces them is removed from the vocabulary rather
than tuned around.

**Sets: aspects that hold several elements.** Some aspects are a set, told one
element at a time or all at once: the members of a club, the languages
someone speaks, the ingredients of a recipe. Superseding there
loses the set one element at a time, each new one closing the previous. The
closed extraction schema (P8) therefore carries, on each fact, a
**`cardinality`** — `one` (the aspect holds one current value), `many` (the
fact is one element of a set) or `unsure` — and an optional **`removes`**: the
value of an element that the fact says was removed or corrected. The model
proposes; the service decides, and for **free-form keys only**: the vocabulary
keys keep the rules above, whatever the answer says (`home` and `employer`
hold one value, role, team and preferences go through the replacement
question). For a free-form key of a node (or of the free speaker facts, which
never close anyway):

- the key is a **set** when a live fact of that key is already an element
  (flagged `many`) — what the store knows outweighs a `one` from the model,
  since when we cannot tell, we do not supersede — or when the new fact says
  `many` and the key holds no live value yet. A `many` on a key that already
  holds a value that is not an element does not turn it into a set silently
  (a model's slip, or a page asking for it): it is treated as `unsure`. A new
  element is written flagged `many`; it closes nothing and conflicts with
  nothing. An element equal to a live one is a duplicate, and the same
  spelling in another script (`same_spelling`, §5.6 rule 14: no one-letter
  tolerance, Daniela is not Daniel) links;
- `one`, on a key that is not a set: the rules above (with the replacement
  question on, a change goes to it, unless the comparison proves the value
  different; off, a change within one language supersedes);
- `unsure`, no cardinality, or a word the schema does not know (the fact is
  kept, the word is not): **an unsure cardinality adds, it does not close** —
  the new value stays live beside the old one, flagged `conflict`;
- **removing or correcting one element**: on a set, `removes` closes only the
  one element its text names — equal after normalisation, the same spelling
  in another script, or, ending with "…" as the prompt shows a long value,
  the start of one element; a text naming several closes none — and only
  where the guards above allow a close: never an `explicit` element, never a
  fact not from external content by an `external` part, never a free speaker
  fact, never by an earlier fact; each of these leaves the `conflict` flag
  instead (on the element already held, when a correction restates one). A
  correction (`removes` and a `value`) adds its value as a new element; a
  removal without a value is written without its key: it states a change, it
  is not an element. A `removes` that names none of the set's live elements
  closes nothing, and its fact is flagged `conflict` with the set;
- **a remark about a set is not an element**: the prompt asks that a remark on
  an aspect ("the recipe is too salty", "the recipe has ten ingredients") take
  another key. When a model still gives it the set's key, it only adds: a set
  loses an element through `removes` only.

The prompt's line of known keys marks the sets, with up to 8 of each one's
live elements (each cut at 80 characters, with "…"), quoted as JSON strings
(stored text is data, P8), so that a removal can name an element as memory
holds it. A fact proposal (§4.4b) keeps the `many` flag, so that an accepted
element joins its set; like `remember`, accepting it closes nothing. A dream
never supersedes an element, and a consolidation of elements stays one
(§5.6). Recall shows each element as its own fact, its handle flagged `many`:
the packer fits as many elements as the budget holds, the most relevant first,
and the node's expansion lists them. The job report counts `elements`
(`added`, `removed`, `unmatched`, `unsure`).

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
small budgets). What the budget keeps then favours the subject asked about.

When the question names that subject **exactly** — the whole name or a
whole alias as consecutive whole words of the question, case, Latin
diacritics and elision folded ("Kiki", "d'Émilie"), never a piece of a word
nor an inflected ending — the subject's live, visible facts also **join**
the candidates, through one indexed query on the node (pinned first, then
most recent, at most as many as the recall's candidate bound): "Où habite
Kiki ?", Kiki an alias of Karim, finds "Karim a déménagé à Osaka", which
shares no word with the question. A joined fact the search also found keeps
its place; the others follow the subject's searched facts, in that order,
ahead of the other candidates. They go through the same equivalence groups,
exclusions and packer as any candidate, so the budget holds whatever the
subject's size. The view filters them like any read (P9): a node the view
cannot see is never named, a fact of a visible node in a scope the view
cannot see never joins. An inflected form anchors but adds nothing
("Карима" reorders Карим's facts; "Karima" never brings Karim's facts), nor
does a name in a script written without spaces (whole words cannot be told
there): a wrong join injects another person's facts, a missed one only
leaves the order as search found it. Measured with the oracle extractor on
the synthetic corpus (tier 2, 600 tokens): correct current answers 17 → 27
of 27, 1.40 → 1.90 per 1k injected tokens, raw recall 0.63 → 1.00; no
question answered before is lost at 150, 300 or 600 tokens, on that corpus
or with `--traps`, `--name-traps`, `--rephrase` or `--speaker`
(`python bench/run.py`, each flag apart).

A question naming two subjects or none, or two homonyms visible in the
view, is ranked without anchoring and nothing joins; nothing joins either
when another visible node bears the very name or alias the question names
(a homonym the bounded search did not return). Measured case (synthetic
corpus, tier 2, a French store): "Où habite Karim maintenant ?" matched
other people's facts on "habite" and Karim's other facts on "maintenant"
ahead of "Karim a déménagé à Osaka", which only the subject's name matched. A store's write-side defects still cost answers
it cannot recover: an answer on another node of the same person (a script
split awaiting its merge), on a fact wrongly filed as the speaker's, or
behind a stale value left live is pushed back when the named node's facts
move ahead.

**A node with no fact of its own ranks behind the facts that match the
same words.** A model often proposes a node named after another subject's value — a
place "bureau de Osaka" beside "Karim travaille au bureau de Osaka.", whose
workplace is the value — and that node holds no live fact. Its lexical
document is its name alone, so on common words of the question ("bureau",
"de") its short name outscores the longer fact that holds the value. So a
node candidate with no live fact in the view, which the question matched
only on words that a fact candidate also matched (every query word the node
matched, matched by the fact too), is moved to just after the last such
fact; everything else keeps its order, nothing is added or removed, and it
runs before subject anchoring. The rule reads matched words only, never a
word list, so it holds in every language; the node is still written (the
model proposed it, and a place often gains facts later), and a node with
facts of its own, or one reached only by a vector neighbour or an edge path,
is ranked as before. The test is on matched words, not on meaning: a
fact that tells the value in other words (an elision, "d'Osaka" beside a
node "bureau de Osaka"; a paraphrase; another language) does not hold the
node back, and any fact matching the same words does, whoever it is about.
Measured with the fake model on a French store holding two workplaces and
places "bureau de Osaka" and "bureau de Montréal" beside them: ranked
without anchoring, as for a question naming two subjects, both place
handles (117 tokens each) outscore Karim's workplace fact on "Où se trouve
le bureau de Karim ?" by length normalisation alone; with this rule both
follow it.

**A request for an activity brings that activity's rules first.** A rule
of method ("label each bed with its sowing date") never resembles the
request that needs it ("make me a map of the balcony"), so ranking rules by
the words they share with a request leaves them behind facts that merely
share words. What a later request repeats is the activity (§4.2), and its
name keeps the noun of what is made. So a request **matches** an activity —
a visible, live node of kind `task` the recall found — when a word of its
name or of one of its aliases is written whole in the request, by the
activity's own rule (§4.2: three letters or more, not a function word; two
letters in a row in a script written without spaces); at tier 3 (§6.1)
also when the node is a vector neighbour of the request at least as close
as `embeddings.activity_threshold`, which is unset by default: a request
compared with a short name is not the name-against-name pair the model's
thresholds were calibrated on, so vectors match no activity until a
threshold is measured and set (measured with bge-m3 on the `--self`
corpus, `vector_probes`: requests in other words at 0.71-0.75 from their
activity, unrelated ones at 0.58 at most, 0.65 between them; a request
sharing no word nor domain with its activity matched none safely). Every
matched activity, at most
`MAX_ACTIVITIES` (3), the activity with the most matched words first (then
the search's order), brings a **block** ahead of every other candidate but
the facts of the subject the question names (subject anchoring, above):
one common word ("notes", "map") may match an activity, and the subject
asked about keeps the first places. The blocks take at most two thirds
of the budget; what does not fit in that share is left out of them
(measured with the oracle on the `--self` map case at 600 tokens: half
the budget kept 1 of its 3 rules, two thirds 2, no bound 3).

An activity is read with its **family**: a model names each occurrence of
one activity with its object ("painting a portrait of the baker", "painting a
portrait of the mayor"), and a dream may group them under one generic activity
(`group_activities`, §5.6) by `instance_of` edges. A matched activity that
is an instance of a live, visible generic one, or a generic one with
instances, stands for the generic one, its instances and nothing else (one
hop, live and visible edges and nodes only); two matched activities of one
family count once. A block holds:
1. the family's rules: the live, visible `method` facts linked to any of
   its activities (§4.2), or to a node merged into one, most recent first,
   at most 10;
2. the rules of what they `uses` (§4.2): the live, visible `method` facts
   of the nodes a live, visible `uses` edge from one of its activities
   reaches, one hop, at most 3 nodes and 5 rules each;
3. its **timeline**: the episodes linked to its activities (§5.8), the most
   recent first — "the last time" before the earlier ones —, at most 3.
A fact the search also found moves into its block; each block's `why` says
which activity brought it (`rule of the activity: <name>`, `rule of <node>,
used for <name>`, `episode of the activity: <name>`, the name of the
activity the fact is linked to; for a family, the generic one's). Blocks go through the
same equivalence groups, exclusions and packer as any candidate, so the
budget holds; a wrong match costs tokens, never a fact, and nothing is
written. Nothing of this needs a model: the links were set at write time,
and the hot path only reads them (P3). A store without activities (the user
perspective) ranks exactly as before.

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
| `group_activities` | group the occurrences of one activity named with their objects under one generic `task` node, linked by `instance_of` edges (§4.2, §5.5) | medium |
| `consolidate_facts` | replace N overlapping `extracted` facts on one node by one consolidated fact; the originals are closed (`valid_to`) and point at the new one (`consolidated_into`), never deleted | medium |
| `merge_nodes` | merge two nodes judged identical — proposed only by the **adjudication** below, never by a unit's answer; both histories kept, aliases unioned | high |
| `supersede_fact` | close an `extracted` fact contradicted by a later source | high |
| `flag_conflict` | mark a conflict for a human or the agent to settle — the ONLY thing a dream may do about an `explicit` fact | low |

`decay_edges` and `reembed` need no model: the service runs them itself at
the start of a run, journalled like the others (`reembed` only queues work on
derived data, so it has nothing to undo). The model proposes the other eight:
seven in a unit's answer, `merge_nodes` only through the adjudication of a
pair. A unit whose node is a `task` also shows the other live `task` nodes of
its scope set that share a word of its name (§4.2's word rule, at most 10):
the only ones `group_activities` may name besides it.

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
    that language. A statement the detector cannot judge is labelled with
    the originals' language when that is the required one, and otherwise
    keeps an empty `language` (§5.4: it never borrows a language it was not
    shown in). A dream never consolidates facts of a `same_as` group
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
cancelled, exactly as at a budget limit. A dream that runs inside a background
run that is cancelled (§8.4) stops there too, as at that run's budget limit: it
finishes with `stopped_by: run_cancelled`, never as a cancelled dream. A run
interrupted by a crash goes back to the queue and resumes where it was.

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
| `group_activities` | `name`, `members` (2–20) | live `task` member nodes of one scope set; a name of at most 200 characters, no word mixing scripts, not the speaker word, with at least two words of §4.2's rule (or one word of at least four letters in a script written without spaces: `one_word` — a single word is the verb a model kept, "writing", not an activity; measured with a Qwen3.8 27B variant on the `--self` corpus, it grouped release notes and a recipe card under "writing"), every such word written whole in every member's name or an alias (`not_shared`: the generic name is what the occurrences have in common, never a new idea); no member already `instance_of` another node (`grouped`); a node of that name in view that is not a `task` (`name_taken`) | the live `task` node of that name in view, or a new one in the members' scope set, and a confirmed `instance_of` edge from each member that is not that node; a member already linked to it is left as is, and nothing left to link is a `duplicate` |
| `consolidate_facts` | `node`, `facts` (2–20), `statement` | live `extracted` facts of that node, not pinned, not flagged `conflict`, in no `same_as` group, one language (rule 13); one key and equivalent values (rule 14, `other_value`); a complete statement (§4.2) naming the node, no word mixing scripts, in the required language | a new `extracted` fact — the originals' key, the first original's value, the lowest confidence, the earliest `valid_from`, `observed` as of the latest day when every original is (§4.2), the activities every original shares (§4.2); the originals closed with `consolidated_into` |
| `merge_nodes` | `keep`, `absorb` | two distinct live nodes of the same kind, identical scope sets, every name of `absorb` passing the name rule (§5.7, `invalid_name`) | `absorb` archived with `merged_into`; its name and aliases join `keep`'s; its facts and edges move to `keep` (an edge left joining `keep` to itself is closed); a pending merge proposal between the two is accepted |
| `supersede_fact` | `fact`, `by` | the write path's own rules (§5.4) checked again: one node, both live, `fact` extracted, the same single-valued key and neither fact an element of a set (§5.4), `by` later, P6's provenance rule, and the values not the same within one language or proven different across two — any of them failing **downgrades** it to `flag_conflict` (`downgraded:<cause>`) | `fact` closed at `by`'s `valid_from`, `superseded_by` = `by` |
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

### 5.7 Corrections by a person

The dream proposes; a person also corrects memory directly: a name left with
debris in it, a wrong alias, a kind the extraction got wrong, two nodes the
person knows are one. A correction goes through the service, so it keeps its
provenance and can be undone; a host that rewrote the record itself would
lose both.

*Correcting a node.* `PATCH /v1/nodes/{id}` (`write`, by a token that
touches the node, §4.6) renames the node (`name`), adds and removes aliases
(`add_aliases`, `remove_aliases`) and changes its `kind`, in one
transaction. A rename **keeps the old name as an alias**, so that a later
mention under that name still finds the node (§5.1), unless `keep_old_name`
is false — the way to drop a name with debris in it. An alias spelled like
the new name is dropped: the name already says it. Removing an alias the
node does not have is refused (422). The new name and every added alias pass
the store's **one name rule**, the same for every write path: not empty once
its whitespace is collapsed, at most 300 characters. An alias spelled like
the name the patch leaves alone stays. The node must be live (409,
`not_live`: restore it, or undo its merge, first). The node keeps its
scopes, facts and edges; its name vector is recomputed (§6.1). A patch that
changes nothing writes nothing.

*Merging two nodes.* Only a token holding **`review`** merges, and only by an
explicit decision: accepting a pending `merge` proposal in the review queue
(`POST /v1/review/{id}`, optionally naming the node to `keep`), or naming two
nodes the person knows are one (`POST /v1/nodes/merge`, `keep` and
`absorb`), which records a `merge` proposal between them (or takes the
pending one) and accepts it through the same path: one guard, one journal.
The two must be distinct live nodes with **identical scope
sets** (§5.6 rule 12: a merge never widens visibility; else 409,
`scope_mismatch`), both touched by the token, and every name and alias of
`absorb` must pass the name rule, since they join `keep`'s (else 409,
`invalid_name`: correct the name first). Their kinds may differ: a kind the
extraction got wrong is no reason to refuse what a person decided, and
`keep` keeps its own kind (`absorb` is archived with its own, so an undo
restores it). Which kind survives is then the reviewer's choice: accepting a
`merge` proposal between two kinds names `keep` (else 422, `keep_required`),
where two nodes of one kind may leave it to the default. This guard is one function, the dream's as well (§5.6,
`merge_nodes`), where the kinds must be the same (409, `kind_mismatch`):
only a person merges two kinds. The merge is the dream's
`merge_nodes`, the same code (§5.6): `absorb` archived with `merged_into`,
its name and aliases joined to `keep`'s, its facts (an explicit one
unchanged) and edges moved to `keep`, an edge left joining `keep` to itself
closed, the proposal accepted. Without `keep`, the node with more live facts
is kept, else the older. No model reaches this path: a dream's merges wait in
the review queue as `merge_nodes` (§5.6 rule 5), no MCP tool corrects or
merges a node (§9), and accepting a `merge` proposal through
`POST /v1/proposals/{id}` — the `write` route an agent holds — is refused
(409). A queued `merge_nodes` of a pair a person merged meanwhile is dropped
`stale` when accepted (§5.6 rule 8).

*Optimistic concurrency.* A correction may carry the revisions (§4.4) its
author read: `rev` on a patch, `revs` (node id → rev) on a merge. A record
that moved since refuses it (409, `stale`, with the current revisions): two
people correcting one node never overwrite each other silently.
`GET /v1/nodes/{id}` shows the node's `rev`.

*Deciding a proposed edge.* `POST /v1/edges/{id}` (`write`, by a token that
touches the edge) with `decision`: `confirm` makes a `proposed` edge
`confirmed` (it no longer decays, §5.3); `reject` archives it (P5: reversible,
and it leaves recall, expansion and the node view). Only a live `proposed`
edge is decided (409, `not_proposed` for a confirmed one, `not_live` for a
closed or archived one); an optional `rev` is the edge's decision revision
(§4.4) the author read, as the node view shows it (409, `stale`, with the
current one): a reinforcement or a decay that leaves the edge open since does
not make it stale, a change of status, life, ends or type does. `POST /v1/link` on a pair that already has the live edge
(§5.2) confirms it the same way when it is proposed, and answers with that
edge (200, `created: false`) rather than adding a twin. Both are journalled
like a node's correction (`confirm_edge` | `reject_edge`) and undone the same
way (`write`): the edge is `proposed` again, or live again, its weight and use
counters left as they are. Such an undo is refused (`depends`) only when the
edge's decision revision moved since; its entry still records the full `rev`
in `after_revs` (§4.4), so a merge undone after it finds the state it left.

*The corrections journal.* Every correction that changed something is an
entry: its id, the operation (`update_node` | `merge_nodes` | `confirm_edge`
| `reject_edge`), the **name of
the token** that made it and when, what was asked, the records it touches,
the state it changed and the revisions right after, and its status
(`applied` | `undone`, with when and by which token). A correction's answer
carries its id; `GET /v1/corrections/{id}` (`read`, a token that touches
every record it names) shows it. `POST /v1/corrections/{id}/undo` reverts it
exactly — the old name, kind and aliases; for a merge, the moved facts and
edges back on the absorbed node, its closed edges reopened, the added aliases
removed, the absorbed node live again without `merged_into`, the proposal
pending again (the one a direct merge recorded is deleted) — under the rule
of a dream's undo (§5.6 rule 4): refused (409, `depends`, naming the
records) when a record it changed has moved since, so later corrections are
undone first, latest first (undoing one puts back the state the one before
it left, which can then be undone in turn). Undoing a merge needs `review`,
like making it; undoing a patch or an edge's decision, `write`. `forget` wipes the content of
every entry that mentions the forgotten record and its id (P5); such an
entry can no longer be undone (`forgotten`). Logs carry counts and ids only.

### 5.8 Session state and its close

The extraction reads one source at a time, yet how a turn is meant often
depends on the turns before it, and a work session changes the state of one
thing turn after turn. So the service keeps, per session, a **session state**:
the frame an earlier turn opened (§4.2) and whether its episode is written, in
the agent perspective and a session the host named the **activities** its
turns named (§4.2), in the order they were last named: the last is the
**current activity**, until a later turn names another (an activity named
again becomes current again; when another turn of the session changed the
state meanwhile, the activity this source names wins, else the one that
turn set stays), and
the **session facts** — the facts whose duration is `task` or `unsure`, and,
in a session the host named, a note of each fact read as `observed` (§4.2),
with their source, span and provenance — and, in a session the host named, the
**turn trace**: for every turn read as sincere, the person's own words (the
owner parts only, P6; at most 400 characters), so that the close can say
what the session was about even when the extraction kept few notes. A turn
whose frame is not sincere leaves no trace (its content is never memory,
§4.2), nor does a turn of a scope set's default session. A session is keyed by the host's session id
**and** the exact scope set of its turns (P9: the same id sent into two scope
sets is two sessions); a turn sent without a session id belongs to its scope
set's default session. The state is shown to the next extraction calls of the
session — the open frame and the newest session facts (20), observation
notes left out since they are memory already — as data between
its own markers (`<<<SESSION>>>`, P8: stored text never closes the block), and
nowhere else: it is never recalled, expanded, exported or dreamed over. It is
written in the extraction's write transaction; when another turn of the
session changed it meanwhile, a frame that turn opened stays open unless this
source ended it (the cautious reading). The turn trace is never shown to
the extraction. A session left with no frame, no activity, no fact and no
trace is deleted. Forgetting a source deletes its session facts and its trace;
forgetting a node deletes the session facts whose subject is its name or one
of its aliases, and the traces that name it (whole words, P5), and leaves no
session with it among its activities.

**Closing a session.** When the host calls session end (§8.1), or when a
session has been idle for `extraction.session_idle_minutes` (120; 0: never) by
the store's clock — the worker queues a `session_close` job for it before it
claims its next job, once; a close that finds the session seen again since it
was queued leaves it open, and a close job that failed for good does not hold
the session — its state is **consolidated**, in one chat call
(P11, under the job's budget) — only when the session holds a session fact:
a session that kept no work in progress expires, trace included, with no
call. The call reads the newest session facts (at most 40 and 8000
characters; observation notes only fill the room the work notes leave), each
with its duration, and the newest turns of the trace (at
most 40 and 6000 characters), each block between its own data markers, and
the target language (P12: the scope's forced one, else the majority
language of the notes and turns; an episode in another language is asked
again once). The answer is a
closed schema, `{"episode": "<one sentence>" | null, "keep": [indices]}`:
an unknown key, an episode that is not a short string, an index that is not
one of the `unsure` notes — a `task` note is never kept — is rejected and
counted, never applied; an answer that is not JSON is asked again once. The
model proposes, the service decides (P8): the episode is written about the
person speaking, under `activity`, flagged `episode`, dated by the session's
start, through the ordinary write path of the source of the person's own
words in the session — P6 applies, so a session whose work came only from
the assistant or from outside leaves the episode as a fact proposal. An
episode that names a known subject - a visible node's name or alias, whole
words (§5.1's rule), other than the person speaking - that no source of the
session names is a fact proposal too: the model fills a generic word ("the
post") with the one known subject that fits, and a wrong object written into
the owner's history is silent (§5.1, the asymmetry of entities). Each
kept note is written through the write path of its own source (resolution,
contradictions, provenance: a note whose value only an external part of its
source holds was kept with that part's provenance, so about the owner it is
only proposed). Everything the call read is then deleted: the
rest **expires**.

**In the agent perspective** (§4.2) the episode is about the self, in one
sentence that names the person it worked with ("NOVA planned the spring
garden with Alice; the tomatoes go by the south wall"): one episode, on the
self's node, never a second one on the owner's. It is self-side, so the
rule of §4.2 decides it from its anchor: written when it leans on the
owner's own words, or on an assistant note whose source held no external
part that was not vouched (`external`, §4.4; flagged `vouched` when it held
one); otherwise a fact proposal. A note quoted in a vouched part is only
proposed at the close: which part was vouched is not kept. The check for a known
subject no source names leaves out the self's and the owner's names, and
the session's activities (§4.2, set from the owner's words). The episode
is linked to every activity the session named (a garden plan, its map and
the map's labels: one episode, three links); a kept rule of method is
linked to none (which activity was current at its turn is not kept). A kept
self-side note follows the same rule from its own source. Without a chat endpoint at the close, everything expires.
The close reports counts only: notes read, notes older than the window
(unread, they expire), turns of the trace read, calls, rejected parts, kept,
written, proposed, expired.

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
a Qwen3.8 27B variant on the synthetic corpus: two auto-applied `refresh_summary`, a
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
(remember, link, confirm or reject a proposed edge, archive, restore, correct
a node), `forget` (hard
delete, P5), `dream` (start, follow, cancel and undo a dream, §8.6; change a
probe set, §8.7), `review` (list and decide the review queue — fact, alias
and merge proposals and queued dream operations — §4.4b, §8.6; merge two
nodes and undo that merge, §5.7), `admin` (background
mode, export/import, stats). A token handed to an agent normally holds
`read` + `write` only. `review` is never implied by another permission.

**Lists page by cursor.** Every route that returns a list of records
(`GET /v1/nodes`, `POST /v1/timeline`, `GET /v1/proposals`, `GET /v1/archive`,
`GET /v1/review`, `GET /v1/dreams`, `GET /v1/dream/{id}/plan`) takes `cursor` and `limit`
(default 100, at most 1000; the timeline's default is 200) beside
`budget_tokens`, and answers with `next_cursor`. A page is the longest run of
items, in the list's order, that starts just after the cursor, holds at most
`limit` items and fits under `budget_tokens` (P1) — whichever bound comes first
ends it. An item that does not fit is never skipped: it opens the next page,
and `omitted` counts the items of the `limit` window the budget left for it.
`next_cursor` resumes exactly after the last item delivered; it is `null` when
nothing follows, and when nothing was delivered (`reason: budget_too_small`:
the same cursor with a larger budget gets the page). Pagination is keyset on a
unique sort key (the item's own order key, then its id), never an offset, so a
write between two pages never repeats nor skips an item that was there before
it: in a newest-first list a new item lands at the head (outside the walk), in
an oldest-first one at the end (delivered once). In the review queue, a merge
proposal that comes back because its queued `merge_nodes` was decided keeps
its place in the queue's order and shows on the next walk. A cursor is opaque: a
position signed (HMAC-SHA256) with a key kept in the encrypted store, bound to
the route, the view, the sort and every filter of the request that made it. A
cursor altered, made by another service, or replayed with another view, route,
sort or filter is refused with 400 before anything is read (P9). `POST
/v1/expand` keeps its own `cursor`: a position in the caller's list of ids.

### 8.1 Lifecycle (Door 2)
| Endpoint | Purpose | Hot path? |
|---|---|---|
| `POST /v1/session/start` | Returns the **profile block**: pinned facts, then in the agent perspective (§4.2) the self's live rules (`commitment.*` and `preference.*`, oldest first), then the most central nodes for the scope, under budget. **Deterministic and stable** across calls when nothing changed, so it can sit in a cacheable prompt prefix. | yes |
| `POST /v1/recall` | `query` (usually the user's message, optionally recent turns) → handles under budget; optional `query_vector` (`model`, `values`) at tier 3 | yes |
| `POST /v1/turns` | Record a turn: `user` (the owner's words), `assistant`, and `external` parts (`kind`: `tool` \| `web` \| `document` \| `other`, `text`, optional `ref`, of which only the short form is kept, P6) for content from outside the conversation, each with optional `vouched` (the host's statement that the owner gave that content or asked for it: the agent perspective may learn from it, §4.2); `tool_summaries` is still accepted and counts as `external` parts of kind `tool` (P6). `record: false` is the host's statement that the turn is not to be remembered (a test started inside a real session): nothing of it is written, not even queued — the host's statement, beside iifa's own reading of the intent (§4.2), neither replacing the other. Enqueue only. | yes (enqueue) |
| `POST /v1/precompact` | The harness is about to compact: receive the messages about to be dropped, enqueue extraction, return a **short** block of what must survive (under budget) | yes |
| `POST /v1/session/end` | Close the session: its session state consolidated (§5.8), then the summaries of the nodes it touched refreshed and edge decay applied. A host that never calls it relies on the idle timeout (§5.8) | no |

### 8.2 Reading
| Endpoint | Purpose |
|---|---|
| `POST /v1/expand` | ids → content at a chosen level (1 = facts, 2 = facts + source excerpts), under budget; paginates with a cursor |
| `GET /v1/nodes` | the nodes the view sees, paged: id, name, kind, aliases, live facts, pending proposals (those about the node, as `GET /v1/nodes/{id}` shows them) and `changed_at`; `sort` = `name` (normalised name, then id) or `changed` (newest first); `prefix` matches the start of the normalised name or of an alias (case and Latin diacritics folded, as names are matched, §5.1; never a substring); archived and merged nodes only with `include_archived`. A walk lists the nodes that existed at its first page, and `changed_at` — the latest of the node's creation and the recording of a fact of it in the view's scopes, archived and closed facts included — counts only facts recorded by then, so no new record moves a sort key; a node archived or forgotten meanwhile leaves the walk, and one whose facts are forgotten, re-scoped or moved to another node (a merge) during the walk may move in a `changed` walk. One query per page, never one per node |
| `GET /v1/nodes/{id}` | a node (with its `rev`, §4.4) with its live facts, top edges (each with its decision `rev`, §4.4) and pending proposals (§4.4b) visible in the view, under budget |
| `GET /v1/corrections/{id}` | one entry of the corrections journal (§5.7): operation, token name, when, what was asked, records, status |
| `POST /v1/timeline` | node or scope + date range → fact changes (`recorded`, `closed`), newest first, paged |
| `POST /v1/at` | "what was known at date D" for a node |
| `POST /v1/path` | how two nodes are related (bounded path search) |
| `GET /v1/lookup` | `external_id` → the visible records carrying it (§4.6); or `name` → the visible nodes with that name or alias (what `memory_about` uses) |

### 8.3 Writing
`POST /v1/remember` (explicit fact, P6; with `about_self: true`, a fact
about the self of the target scopes, below), `POST /v1/link` (201 with a new
edge; 200 with `created: false` when it confirms the live edge already there,
§5.2), `POST /v1/edges/{id}` (`decision`: `confirm` | `reject` a proposed
edge, optional `rev`, §5.7; answers with the edge's status, its decision
`rev` and `correction_id`; `/link` answers with the edge's `rev` too), `POST /v1/archive` and
`POST /v1/restore` (node, fact, edge — reversible, P5), `GET /v1/archive`
(archived items, most recently archived first, paged; each item says what a
person needs to choose between `restore` and `forget`: a node's `name` and
`node_kind` (and `merged_into`), a fact's `statement`, `node_id` and
`node_name`, an edge's `type`, `status`, `src`, `dst`, `src_name` and
`dst_name` — a name only when that node is visible in the view, else null,
P9), `POST /v1/forget` (node, fact, edge or
source — hard delete, `forget` permission, P5; returns what was removed, by
count), `PATCH /v1/facts/{id}` to
attach a free-standing fact to a node (`node_id`), detach it, pin it, or
replace its scope set (`scopes`, §4.6). With `scopes`, the fact's node — the
current one, or the `node_id` given — must be visible from the new set (409
for the current one: detach it or name another).
A person's corrections (§5.7): `PATCH /v1/nodes/{id}` (`write`: `name`,
`keep_old_name`, `kind`, `add_aliases`, `remove_aliases`, optional `rev`),
`POST /v1/nodes/merge` (**`review`**: `keep`, `absorb`, optional `revs`),
both answering with the id of their journal entry (`correction_id`, null when
nothing changed), and `POST /v1/corrections/{id}/undo` (the permission
that made the correction: `write` for a patch, `review` for a merge).
`POST /v1/proposals/{id}` with `decision: accept | reject` (§4.4b; `write`,
by a token bound to one of the proposal's scopes; a decided proposal is not
decided again; a `fact` proposal also needs `review`, and its acceptance
answers with the id of the explicit fact written). `GET /v1/proposals` (view,
optional `kind`; `read`): the pending proposals visible in the view, oldest
first, paged — the only place a fact proposal without a node shows.

**The agent remembering about itself.** In the agent perspective (§4.2),
`remember` with `about_self: true` writes on the self's node (found, or
created in the declared scope, like at extraction): origin `extracted`
(nothing it writes is beyond correction, P6), confidence 1.0, provenance
`assistant` with the self's name in `from`. It is refused (422) when the
target scopes have no self, and with `node` or `node_id`. `remember` never
calls a model (P3), so only what can be decided without one applies: a live
fact of the self that states the same thing (the same statement, or the
same value of the same key) is an echo: nothing is written, and the answer
returns that fact with `echo: true`; another value of a key the self
already holds is written flagged `conflict` with the held fact, never
closing it.

`precompact` messages take their provenance from their `role`: `user` is the
owner's, `assistant` the assistant's, anything else (`tool`, `system`…)
external (P6).

### 8.4 Operations
`GET/PUT /v1/background` (mode `auto` | `paused` | `on_demand`, concurrency
per endpoint — P11), `POST /v1/background/run` (process now, with an optional
budget; answers with a run id and what it processed; with `"wait": false` it
answers 202 with the run id at once and the run goes on in its own thread),
`GET /v1/background/run/{run_id}` (a run's report, live while it runs; the
current run and the last 20 are kept in memory: a restart interrupts a run and
forgets them), `POST /v1/background/run/{run_id}/cancel` (stop a run early,
as at a budget limit: no new job starts, a job in flight is put back untouched
at its next model call or finishes if it makes none (a call already waiting at
a yielded gate is still made), and the run ends with `stopped_by: cancelled`;
the mode is unchanged; 404 for an unknown run, 409 for a finished one; admin), `POST /v1/background/yield` and
`POST /v1/background/resume` (a foreground turn starts / ends). At most one run
at a time; a second is refused with the current run's id. A run's report and
its `/v1/health` summary say whether a cancel was asked (`cancel_requested`,
then `jobs_left` is 0), and the report names the token that asked
(`cancelled_by`).
`GET /v1/health` (tier, backends reachable, queue depth, stale vectors, and the
run in progress as counts and ids — its id, the name of the token that started
it, when, its budget, jobs done and left, whether the background is yielded — so
every client sees a run it did not start),
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

**Where a fact came from** (P6). A fact that is not the owner's own words
carries `from`: the assistant's declared name (`"NOVA"`, §4.6; `"assistant"`
when none is declared), or `"<kind> <ref> <day read>"` for external
content (`"web example.com 2026-03-14"`; the kind is `external` when the
harness gave none, the ref is left out when there is none). The field is
left out for the owner's words and for a fact without provenance
(`remember`, an import), so a recall of the owner's facts costs exactly what
it costs without it, and the few tokens it adds elsewhere are counted like
the rest of the handle (P1). The same `from` rides on every fact a read
route returns: recall and precompact handles, the profile's pinned facts,
`expand` (with `provenance`, `part_kind`, `part_ref` and `read_at` beside it)
and the facts a node's expansion lists, the node view, the timeline and
`/v1/at`. A node handle carries the flag `external_only` when every live fact
of the node visible in the view is `external`: known only from what was read.

**When it was observed** (§4.2). An observed fact carries the flag
`observed` on every read route, and its handle's `age_days` counts from
`observed_at` rather than from `valid_from`: « observed 7 days ago » is the
flag and the age, a word more than a lasting fact's handle. `expand` and the
node view give `observed_at` itself.

**The activities it was learnt during** (§4.2). `expand` of a fact linked to
activities gives `activities` (`[{"id", "name"}]`), each read through
`merged_into`, and only those the reader may see (P9): an activity the
reader cannot see is never named. The handle's `expand_tokens` counts them.

**A newer observation in review** (§4.4b). A live observed fact whose node
has a pending fact proposal, visible in the view (P9), that is itself an
observation (flagged `observed`) of the same attribute with another value,
observed on a later day than the fact's `observed_at`, carries the flag
`newer_pending` in recall handles and in `expand` (a fact, and the facts a
node's expansion lists): memory did not close it
(when we cannot tell, we do not supersede), yet it may no longer be current,
and the agent must not read it as today's state. The flag is computed when
the fact is read, never stored: deciding the proposal removes it, with
nothing to undo, and a proposal the reader cannot see changes nothing.

**A newer observation of the same thing** (§4.2). A model does not always
key the observations of one thing alike (its status one month, its content
the next), and nothing links two keys, so neither closes the other. A live
observed fact whose node holds another live observed fact, visible in the
view, under another attribute and observed on a later day, carries the flag
`newer_observed` wherever `newer_pending` rides: two keys may be two
aspects that are both current, so nothing is closed, but the agent learns
that the thing was seen since. Computed at read, like `newer_pending`.

**A conflict is seen from both sides** (§5.4). A fact written in conflict
carries `conflict` and names the fact it contradicts in `conflicts_with`;
the contradicted fact stays as it was stored. So when it is read, a live
fact that a live fact, visible in the view and still flagged `conflict`,
names in `conflicts_with` carries `conflict` too, wherever `newer_pending`
rides: the agent never reads one side of an open conflict as settled. Only
a contradiction with at least the same authority shows (P6): an `explicit`
fact only by another `explicit` one, and a fact that is not external
content never by external content - otherwise a page could cast doubt on
any fact of the owner's without changing it. Any fact, not only an
observation; computed at read, never stored, so the flag goes when the
conflict is resolved.

### 8.6 Dream (§5.6)
| Endpoint | Purpose | Permission |
|---|---|---|
| `POST /v1/dream` | Start a run: `scope`, `operations` (subset of the catalogue, default all), `budget` (calls, tokens, seconds, operations), `dry_run`, `review_policy` (`low` / `medium` / `high` → `auto` \| `review`, only stricter than the configured policy, else 422). Returns a run id at once (202) — the work is asynchronous; 409 while a run is pending or running on the scope, an ancestor or a descendant | `dream` |
| `GET /v1/dreams` | The runs on `scope` and below it, newest first, paged; optional `status`; each with its id, scope, status, `dry_run`, `started_by` (the name of the token that started it, null when unknown) and its times | `dream` |
| `GET /v1/dream/{id}` | Status and, once finished, the dream report | `dream` |
| `GET /v1/dream/{id}/plan` | The operations proposed / applied / sent to review / dropped, each with its reason, in order, paged (`after`, a plain `seq`, is the same position as `cursor`; `next` beside `next_cursor`) | `dream` |
| `GET /v1/review` | The review queue of a view: pending `alias`, `fact` and `merge` proposals and queued dream operations, oldest first, paged; a `merge` proposal and a queued `merge_nodes` carry `adjudication` (`verdict`, `reason`, `run_id`, `at`; null when the pair was never adjudicated); an item about nodes carries `nodes` (each one the view sees: `id`, `name`, `kind`, `aliases`, its first three `facts`; a merge's two sides in order, `node_id` then `value`, `keep` then `absorb`; a queued operation's nodes in the order its payload names them); a queued operation that names facts carries `facts` (each one the view sees: `id`, `statement`); a `fact` proposal whose accept would close live facts carries `replaces` (each one the view sees: `id`, `statement`) | `review` |
| `POST /v1/review/{id}` | `accept` or `reject` one item (a proposal or a queued operation); an operation whose inputs moved since its snapshot is dropped (409, `stale`); accepting a `merge` proposal merges the two nodes as a person's decision (§5.7; optional `keep`), journalled as a correction whose id the answer carries | `review` |
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
`memory_about` (node lookup by name/alias); `memory_remember` takes
`about_self` (§8.3). **`memory_forget` is not exposed
by default** (P5): enabling it is a server option, and even then it requires a
token with the `forget` permission.

- **Tool descriptions carry the doctrine**, briefly: recall returns handles —
  expand only what you need; remember explicit, durable facts, not chatter;
  archive what is wrong or obsolete — it can be restored.
- Every tool result obeys the budget (default from config, overridable per call).
- Results are wrapped as recalled data (P8).
- Transports: stdio and streamable HTTP.
- **No dream tool** — on purpose (§5.6 rule 11) — and no tool that corrects
  or merges a node: those are a person's decisions (§5.7).

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
- **neighbouring subjects** (`--neighbours`): turns, with no event and no
  question, that describe two of the owner's projects side by side — a web
  service and a command-line tool, each in a sentence of its own, lasting
  facts ("Atlas-1 is a Go web service with 4 contributors. Ember-1 is a Rust
  command-line tool with 3 plugins."), in English or French.
  Each fact written from them is judged by the sentence its excerpt lies in:
  **right** on that sentence's file, **wrong** on the other one (must be 0),
  unjudged when its excerpt lies in no sentence of one file alone; a file
  without a right fact is **missing**; the fact proposals of those turns are
  counted (§5.1). Each such turn is followed by an **aside**: the web service
  told something unrelated to its description ("Atlas-1 is deployed on the
  shared server."), whose gold answer keys it `description` too and says it
  holds one value — the slip of a model on a generic key. Both are true. Each
  aside is compared with what the turn before it wrote on that project: no
  key in common is **other key** (nothing to replace); with one, the aside is
  **kept** (the description still live), **closed** (superseded by the
  aside: must be 0) or in **conflict** (§5.4); either side writing nothing
  on the project is **missing**. The keys of both sides are listed;
- **name adjudications**, every corpus: each mention sent to the adjudication
  with its candidates (name, score, whether the candidate came from name
  vectors) and the answer (the candidate chosen, or a new node), as each
  turn's extraction report gives them, with the number of adjudication calls;
  a pair of two different people among them is a **near-name pair**, and its
  answer is reported (attached is a wrong attachment).
- **observed states** (`--observed`, §4.2): in `user:alice`, a few things
  whose state changes without anyone telling the memory — a shelf, a
  service, a queue — each observed by the owner two or three times in
  different months ("I checked: Pantry holds 3 jars.", the
  turn dated that month), the state under one generic key (`content`,
  `state`), and told once, under the same key, something lasting about it
  ("Pantry is where spare jars go."). Counted after
  ingestion, per thing: the newest observation **current** (live, flagged
  `observed`, as of its turn's day), the older ones **stale** (still live:
  must be 0), the lasting fact **kept** (still live: all) or **closed** (must
  be 0); and a recall per thing ("What is the state of Pantry?") at each
  budget: the newest observation shown with the flag, and no stale one (of
  the stale ones shown, those flagged `newer_pending`, `newer_observed` or
  `conflict`, §8.5, are counted);
`--resolve-threshold` and `--merge-threshold` replace the embedding model's
profile for a run, so that one corpus is measured at two profiles.

**Intent** (`--intent`, §4.2): for every person and each of their languages
among English, French and Russian, a turn that names one of their homes
without stating it — a hypothesis, a question, a guess, a possibility, an
example — told just before the sincere turn that states that home (its
**control**); and in `user:alice`, role-play personas in each of those languages:
a frame over three turns ("for this exercise, I am a pilot named…", a turn
inside it, "end of the exercise") and a persona the task sets ("answer the
recruiter as…"), each with the one episode an ideal model leaves. The oracle's
gold holds the content as an ideal model reads it (labelled `hypothesis`,
`pretend`…), so the oracle run proves the mechanics. Counted, like chatter:
facts written and nodes created from the turns not meant (target zero), the
proposals they made, the episodes written against those expected, per kind;
the **controls** written, duplicated, proposed or lost (over-filtering is the
opposite failure, and it counts as one); `dropped_by_intent` per frame and
modality, and the facts kept as session state. `--no-read-intent` turns the
reading off for a run against the same corpus. **Work session**
(`--worksession`, §5.8): the last month of `user:alice` becomes one session of
iterations on a slide deck (sizes, colours, a count, a bug, a request about
the task), that month's lasting events told in between, then the host closes
the session. Counted after the close: the task facts live in memory (must be
0), the session facts left (0), the episodes written (1), and the lasting
events written (all).

**The agent's own memory** (`--self`, §4.2): a scope of its own,
`user:alice/agent:nova`, declares its owner and its self, so the main corpus
is read as without it. Each of its turns carries the assistant's reply and,
for some, a page or a document, and is a case: what must become the agent's
memory (a lesson learnt on one customer's backup, a view, commitments, the
relation with the owner, rules of method read for the task in a guide the
owner attached, which the host vouches for, with limits beside them) and
what must not (the same kind of guide found by the agent on its own, not
vouched, whose rules are only proposed; the agent restating a page in the same words, in other words, a fake label with its
name inside a page; its own fact said again; a role the owner set, a
translation, a draft for the owner; politeness). The gold of the hostile
cases is what a fooled model answers. Counted per category, against what
each case must leave: facts written, proposed and echoed; `hostile_written`
and `unmeant_written` (must be 0); the job report's `facts.self`,
`facts.method`, `facts.self_proposed`, `facts.self_echo`; the assistant
turns read as not sincere; the self's commitments the profile carries. Then
two recalls per budget: the lesson for another customer's backup
(`transfer_shown`), and a request that names the rules' task in other words
(`rules_shown`, `limits_shown`, `rules_first`: a rule ranks before every
limit), and a request for another map (`activity_rules_shown`,
`activity_first`: the map's rules ahead of everything, §5.5); the report
lists every activity the model named with the facts and rules linked to it.
At tier 3, `vector_probes` measures matching an activity by vectors
(§5.5): requests in other words for three activities, sharing no word with
any name a model is likely to give them, and requests no activity must
answer; for each, the cosine of every task node the vector search returns
and whether a word already matches it; per threshold (0.45 to 0.75, and
words only), the positives whose rules come in a block and the negatives
that bring one, at 600 tokens.

`--rephrase`, `--speaker`, `--name-traps`, `--neighbours`, `--intent`,
`--worksession`, `--observed` and `--self` each make another corpus,
compared only with runs that used it; the default corpus is unchanged.

**The comparison form** (`--value-norm`, §5.4): the extraction prompt asks
for each value's `value_norm` and the write path links the values whose
forms agree across languages. Against the same command without it, on the
default, `--rephrase` and `--traps` corpora, it answers whether the switch
keeps the headline metric, saves value adjudications and adds no false
supersession. The oracle answers with the corpus's own value ids as forms
(an ideal model); the numbers that decide the default come from a real
model.

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
- [x] A list cursor never crosses a view: an altered cursor, one made by another service, or one replayed with another scope, route, sort or filter is refused and returns nothing.
  - `tests/api/test_paging.py::test_p9_a_cursor_never_crosses_a_view_a_route_or_a_filter`
  - `tests/api/test_paging.py::test_a_cursor_signed_by_another_service_is_refused`
- [x] The archive list never names a node the view does not see (an archived fact's node, an archived edge's ends), and the dream list never shows a run outside the token's scopes.
  - `tests/security/test_management_reads.py::test_p9_the_archive_never_names_an_invisible_node`
  - `tests/security/test_management_reads.py::test_p9_the_dream_list_never_shows_a_run_outside_the_tokens_scopes`
- [x] A re-scope never grants nor takes away a scope the token does not own, never leaves the fact under a node of a scope it left, and a re-scoped fact leaves no statistic behind in the scopes it left.
  - `tests/security/test_p9_scope_isolation.py::test_p9_a_rescope_never_grants_or_takes_away_a_scope_the_token_does_not_own`
  - `tests/security/test_p9_scope_isolation.py::test_p9_a_rescope_never_leaves_a_fact_under_a_node_of_the_scope_it_left`
  - `tests/security/test_p9_scope_isolation.py::test_p9_a_rescoped_fact_moves_its_statistics_with_it`
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
- [x] Two nodes are merged by a person only through a token holding `review` — never a `write` token, a `dream` token or a model's path — never across scope sets; a person's merge undoes exactly (facts, aliases, edges, archived state, the proposal), and `forget` wipes the corrections journal (raw bytes).
  - `tests/security/test_node_merges.py::test_only_a_review_token_merges_two_nodes`
  - `tests/security/test_node_merges.py::test_no_model_path_reaches_a_persons_merge`
  - `tests/security/test_node_merges.py::test_a_merge_across_scope_sets_is_refused`
  - `tests/security/test_node_merges.py::test_a_persons_merge_moves_everything_and_undo_restores_exactly`
  - `tests/security/test_node_merges.py::test_p5_forget_wipes_the_corrections_journal`
- [x] A proposed edge is confirmed or rejected only by a `write` token that touches it, journalled with the token's name and undone exactly; `forget` of the edge wipes that entry (raw bytes).
  - `tests/security/test_management_reads.py::test_an_edge_is_decided_only_by_a_write_token_that_touches_it`
  - `tests/security/test_management_reads.py::test_p5_forget_wipes_an_edge_decision_from_the_journal`
- [x] A probe set cannot name a record its scope does not see; `forget` deletes every probe that expected the record, question included (raw bytes).
  - `tests/api/test_probes.py::test_p9_an_invisible_or_unknown_expected_id_is_refused_alike`
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_forget_deletes_every_probe_that_expected_the_record`
- [x] A value's comparison form (§5.4) is never shown by a route, never searched, never sent to a model, and only ever links: it closes no fact, never overrules a value proved different, never acts within one language, against an explicit fact or for external content, and `forget` deletes it.
  - `tests/security/test_value_norm.py::test_the_comparison_form_is_never_shown_by_any_route`
  - `tests/security/test_value_norm.py::test_the_comparison_form_is_never_searched`
  - `tests/security/test_value_norm.py::test_the_comparison_form_is_never_sent_to_a_model`
  - `tests/security/test_value_norm.py::test_the_comparison_form_is_never_sent_to_the_dream_model`
  - `tests/security/test_value_norm.py::test_the_comparison_form_never_closes_a_fact`
  - `tests/security/test_value_norm.py::test_external_content_never_joins_the_owner_by_its_form`
  - `tests/security/test_value_norm.py::test_the_owner_never_joins_external_content_by_its_form`
  - `tests/security/test_p5_encryption_archive_forget.py::test_p5_forget_leaves_no_trace_in_the_file`
- [x] Memory poisoning (P6): a hostile web page quoted in a turn (a fake "User:" line, "the user lives in Moscow", instructions) yields no fact about the owner — at most fact proposals — and closes no fact the owner stated; a token without `review` (an agent's) can neither accept nor reject a fact proposal.
  - `tests/security/test_p6_poisoning.py::test_p6_a_hostile_page_yields_no_fact_about_the_owner_only_proposals`
  - `tests/security/test_p6_poisoning.py::test_p6_a_page_never_closes_what_the_owner_said_about_someone`
  - `tests/security/test_p6_poisoning.py::test_p6_an_agent_token_can_neither_accept_nor_reject_a_fact_proposal`
- [x] The agent's own memory (P6, §4.2): a page in a turn gives the agent no belief and no rule of method, whether the reply restates it word for word or in other words, nor does a fake label with the agent's name written in the page, nor an answer that cites the owner's words as asking to keep it; only a reviewer accepts it, unless the host vouched for every external part of the turn (then flagged `vouched`), which never covers the owner's facts — for what the model reads as self-side (about the agent, or keyed `method`, §4.2); an echo of the agent's own fact writes nothing and closes nothing; the agent never closes what the owner said about it; neither the reply nor a page sets the activity a rule is linked to, nor names it in a way no node may be named.
  - `tests/security/test_p6_self.py::test_p6_a_page_restated_by_the_agent_gives_it_no_belief_verbatim_or_paraphrased`
  - `tests/security/test_p6_self.py::test_p6_a_page_gives_no_rule_of_method_on_any_subject`
  - `tests/security/test_p6_self.py::test_p6_a_fake_self_label_inside_a_page_is_still_the_page`
  - `tests/security/test_p6_self.py::test_p6_no_claim_in_the_answer_lets_a_restated_rule_through`
  - `tests/security/test_p6_self.py::test_p6_one_part_not_vouched_keeps_every_rule_of_the_turn_for_review`
  - `tests/security/test_p6_self.py::test_p6_vouching_never_gives_a_fact_about_the_owner`
  - `tests/security/test_p6_self.py::test_an_echo_of_the_agents_own_fact_writes_and_closes_nothing`
  - `tests/security/test_p6_self.py::test_the_agent_never_closes_what_the_owner_said_about_it_nor_the_reverse`
  - `tests/security/test_p6_self.py::test_p6_a_kept_note_from_a_turn_with_a_page_is_only_proposed_at_the_close`
  - `tests/security/test_p6_self.py::test_p6_a_session_episode_leaning_on_a_turn_with_a_page_is_only_proposed`
  - `tests/security/test_p6_self.py::test_p6_neither_the_reply_nor_a_page_sets_the_activity`
  - `tests/security/test_p6_self.py::test_an_activity_named_like_no_node_may_be_is_refused`
- [x] Intent before content (§4.2), the mechanics: a non-sincere frame or a modality other than `fact` writes no fact from the content (at most one episode, never carrying its values); a missing or unsure frame or modality makes a proposal, never a fact; a frame carries over its session and never into another scope set; `record: false` writes nothing.
  - `tests/security/test_intent.py::test_a_non_sincere_frame_writes_nothing_of_its_content_and_one_episode`
  - `tests/security/test_intent.py::test_every_non_sincere_frame_writes_no_fact_from_its_content`
  - `tests/security/test_intent.py::test_an_episode_that_carries_a_value_of_the_content_is_only_proposed`
  - `tests/security/test_intent.py::test_a_modality_other_than_fact_writes_no_fact`
  - `tests/security/test_intent.py::test_a_missing_or_unsure_frame_or_modality_makes_a_proposal_never_a_fact`
  - `tests/security/test_intent.py::test_a_frame_holds_over_its_session_until_a_turn_ends_it`
  - `tests/security/test_intent.py::test_p9_a_frame_is_never_carried_into_another_scope_set`
  - `tests/security/test_intent.py::test_a_turn_sent_with_record_false_writes_nothing_at_all`
- [x] Session state (§5.8): a `task` fact never becomes memory on its own; the close's answer goes through a closed schema; stored session text never closes its prompt block; forgetting a source or a node takes its session facts along (raw bytes).
  - `tests/model/test_work_sessions.py::test_task_facts_are_session_state_shown_to_the_next_turn_and_never_memory`
  - `tests/model/test_work_sessions.py::test_the_close_leaves_one_episode_keeps_only_the_unsure_it_names_and_expires_the_rest`
  - `tests/model/test_work_sessions.py::test_the_consolidation_schema_is_closed`
  - `tests/model/test_work_sessions.py::test_p8_a_session_fact_never_closes_the_block_it_is_shown_in`
  - `tests/model/test_work_sessions.py::test_p5_forgetting_a_source_or_a_node_takes_its_session_facts_along`

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

A second workflow runs when a version tag (`v0.1.0a3`) is pushed: it builds
the wheel from the tagged tree and publishes a GitHub release named after
the tag, with that version's section of `CHANGELOG.md` as its notes and the
wheel attached; a version with `a`, `b` or `rc` is marked a pre-release. It
refuses a tag whose version is not the one in `pyproject.toml`, or one
without a changelog section. It can also be started by hand for a tag that
already exists.

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
  .github/workflows/release.yml  on a version tag: the GitHub release (changelog notes, wheel)
```
