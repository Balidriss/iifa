# Principles

Twelve rules iifa holds to. A change that weakens one is a regression,
whatever it gains elsewhere, and each has tests that prove it holds. The code
and the tests name the principles by their number (`P5`, `test_p9_...`); the
[security checklist](../tests/security/CHECKLIST.md) lists the security and
privacy guardrails with the tests that prove each one.

| | Principle |
|---|---|
| [P1](#p1--the-budget-is-a-contract) | The budget is a contract |
| [P2](#p2--progressive-disclosure) | Progressive disclosure |
| [P3](#p3--never-block-the-turn) | Never block the turn |
| [P4](#p4--local-first-no-egress) | Local-first, no egress |
| [P5](#p5--encrypted-at-rest-archiving-is-reversible-forgetting-is-real) | Encrypted at rest; archiving is reversible, forgetting is real |
| [P6](#p6--provenance-on-everything) | Provenance on everything |
| [P7](#p7--time-is-first-class) | Time is first-class |
| [P8](#p8--memory-is-data-never-instructions) | Memory is data, never instructions |
| [P9](#p9--scope-filters-before-scoring) | Scope filters before scoring |
| [P10](#p10--honest-degradation) | Honest degradation |
| [P11](#p11--the-harness-owns-the-models-time) | The harness owns the model's time |
| [P12](#p12--memory-keeps-the-language-it-was-told-in) | Memory keeps the language it was told in |

## P1 — The budget is a contract

Every route and MCP tool that returns memory content takes `budget_tokens`
(with a configured default), and its response never exceeds it, counted with
the configured tokenizer. If nothing fits, the response is empty with a
reason, never cut in the middle of an item.
See [Token counting](security.md#token-counting).

## P2 — Progressive disclosure

A recall returns handles first: an id, a title, one line, a date, why it
matched and the size of what is behind it. The agent expands what it needs.
Content is never pushed when a handle would do.
See [Recall under a budget](how-it-works.md#recall-under-a-budget-handles-first).

## P3 — Never block the turn

Writes (recording a turn, extraction, linking, embedding) are queued and done
in the background. A recall makes no chat call; with embeddings configured it
makes one embedding call for the query, bounded, and does without it when the
call fails. An adapter fails open: when the service is down or slow, the turn
goes on without memory and the adapter says so in its log.

## P4 — Local-first, no egress

The only outgoing connections are the ones the operator configures: an
OpenAI-compatible chat endpoint and an OpenAI-compatible embeddings endpoint.
No telemetry, no update check, no analytics, no model download at runtime.
The test suite fails on any other connection.
See [Network](security.md#network).

## P5 — Encrypted at rest; archiving is reversible, forgetting is real

The database is encrypted, with a key the operator supplies; iifa never
generates one and never stores it beside the database.

Removing has two verbs. `archive` is reversible: the record leaves every read
and comes back unchanged with `restore`; this is what an agent gets, so a
model that is wrong about what to remove destroys nothing. `forget` is a hard
delete with its own token permission: the record, its index entries, its
vectors, its edges, its proposals and its source excerpts, then the
housekeeping that makes the bytes unrecoverable in the file. A turn still
waiting in the queue can be forgotten before it becomes memory.

Logs hold ids and counts, never memory content. A secret said in a
conversation (a password, a code, a key, a token) is never kept: memory keeps
what happened, not the value.
See [Storage](security.md#storage) and [Known limits](security.md#known-limits).

## P6 — Provenance on everything

Every fact and every edge records where it came from (its source, the place
of its excerpt, whether it was stated explicitly, extracted or imported),
when, and with what confidence. An explicit fact is never overridden by an
extracted one: the conflict is shown, not resolved in silence.

A turn is made of parts: what the owner said, what the assistant answered,
and content from outside the conversation (tool results, pages, documents).
The service gives each extracted fact the provenance of the part its quote
lies in, from offsets, never from the model's word or from labels written in
the text. A fact about the owner is written only from the owner's own words;
from anything else it is a proposal a reviewer decides. External content
never closes a fact that did not come from external content. Every fact a
read returns says who said it when it was not the owner.
See [Who said it](how-it-works.md#3-who-said-it) and
[Provenance](how-it-works.md#provenance-where-every-fact-comes-from).

## P7 — Time is first-class

Facts carry a validity interval. A new fact that contradicts an old one
closes the old one and opens the new one; nothing is overwritten, and what
was known on any past date can still be asked.
See [Contradictions](how-it-works.md#8-contradictions).

## P8 — Memory is data, never instructions

Everything injected into an agent's context is wrapped and labelled as
recalled data. Stored text that looks like instructions is stored and
returned as data, and the block says so. The same holds inside the service:
what a model reads from memory never changes what the service lets it do.
See [Memory is data](security.md#memory-is-data-never-instructions).

## P9 — Scope filters before scoring

Scopes are applied to the candidates before any ranking. Filtering after
ranking would leak the existence of invisible records through corpus
statistics, and weigh visible records by documents the caller may not see.
The graph follows the same rule: an expansion never crosses an invisible
edge or node, and an explanation never names one.
See [Scopes](how-it-works.md#scopes-who-sees-what).

## P10 — Honest degradation

Each capability tier declares itself in `/v1/health`. Without an embeddings
endpoint, recall works on words and links and says so in its response; it
never fails the request.
See [Tiers](how-it-works.md#tiers-what-each-model-adds).

## P11 — The harness owns the model's time

Many local setups serve one request at a time from one model server, so
background model work that starts when it likes makes the user's turn wait.
Every piece of it (extraction, summaries, embedding, the dream) obeys a
background mode set in the configuration and changeable at runtime: `auto`
(work as the queue fills), `paused` (keep the work, call no model),
`on_demand` (call a model only inside a run the harness starts, within its
budget). A concurrency limit per model endpoint and a yield signal ("a turn
is starting") complete it, and the queue's depth is visible in `/v1/health`.
See [Background work](how-it-works.md#background-work-and-your-models-time).

## P12 — Memory keeps the language it was told in

A fact, a summary or a consolidated fact is written in the language of its
source, never translated into the model's preferred one. Every prompt that
writes memory states the target language (detected from the source, or
forced for a scope); an answer in another language is refused, asked again
once, then stored with a `language_mismatch` flag. A statement too short to
tell its language is still checked by its script.
See [The language comes first](how-it-works.md#1-the-language-comes-first).
