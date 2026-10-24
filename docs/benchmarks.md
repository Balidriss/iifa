# Benchmarks

Every number here was measured, with the command that reproduces it. Results
depend on the machine and the model; each section says which.

- [The headline metric](#the-headline-metric)
- [Quickstart timing](#quickstart-timing)
- [Recall and write latency](#recall-and-write-latency)
- [Turn recording](#turn-recording)
- [The benchmark harness](#the-benchmark-harness)
- [Results against other approaches](#results-against-other-approaches)
- [The replacement question](#the-replacement-question)
- [Choosing a storage language](#choosing-a-storage-language)
- [Embedding models](#embedding-models)
- [Extraction against a real model](#extraction-against-a-real-model)
- [Test suite](#test-suite)
- [Public datasets](#public-datasets)

## The headline metric

**Correct current answers per 1,000 injected tokens.** A question counts as
answered correctly when the current value was injected into the context
**and no outdated one**. Raw recall (the current value was injected) is shown
beside it, never alone: raw recall rewards injecting every version of a
value, which a naive vector search does.

## Quickstart timing

Timed from an empty virtualenv and an empty Docker cache on a Windows laptop,
every quickstart command run to the end:

| Path | Machine time | Of which |
|---|---|---|
| pip | 27 s | 23 s of `pip install` |
| Docker | 23 s | 18 s of build, base image download included |

## Recall and write latency

Tier 0 (no model) at 10,000 facts: 50 nodes, 25 edges, one scope, then 200
recall queries in three scripts at a budget of 600 tokens. Measured on a
shared cloud container, Python 3.11, Linux x86_64.

```bash
python bench/recall_latency.py --facts 10000 --queries 200
```

| Metric | Value |
|---|---|
| recall p50 | 13.9 ms |
| recall p95 | 23.3 ms |
| recall p99 | 31.5 ms |
| recall max | 39.9 ms |
| write path (`store.add_fact`), 10,000 calls, total | 5.93 s |
| write p50 / p95 / p99 | 0.48 / 0.81 / 1.06 ms |
| write max | 106 ms |
| writes above 10× the median | 12, together 0.78 s (13 % of the total) |

The design target is a recall p95 under 50 ms on a laptop CPU.

**Why the whole write distribution is shown.** A p95 once hid a problem: on
Windows the same 10,000 writes took 144 s at a p95 of 1.2 ms. The time went
into WAL checkpoints, which SQLite ran on the writing connection every ~65
facts, with two fsyncs each. Checkpoints now run on a background thread
(`[store] checkpoint_interval`, 0.5 s) and a write only appends to the WAL.
On the machine above the total dropped from 7.91 s to 5.93 s and the p99
from 7.56 ms to 1.06 ms. The remaining tail is the writer's backstop
checkpoint under a continuous stream (12 in 10,000 writes); an agent that
remembers a few facts per turn never reaches it. Windows has not been
measured again since.

## Turn recording

`POST /v1/turns` only queues; the target is under 10 ms. Measured in the suite
(`tests/perf/test_turn_enqueue.py`: 200 inserts into the queue, then 100 calls
through the route in-process, with the worker paused so the enqueue is
measured alone):

| Metric | Value |
|---|---|
| queue insert p50 / p95 | 0.18 / 0.28 ms |
| `POST /v1/turns` in-process p50 / p95 | 2.5 / 3.6 ms |

## The benchmark harness

`bench/run.py` runs iifa and three baselines on the same corpus and questions,
with whole units under the same budget, counted with the same tokenizer:

- **no memory**;
- **truncated history**: the most recent turns that fit;
- **vector top-k**: the raw turns closest to the question.

Every number and the exact command go to one JSON file in `bench/results/`
(not committed). `--keep-store PATH` keeps the encrypted store so you can read
what a model wrote (debugging only; delete it after).

**The corpus** (seed 7): 188 turns over 12 months in English, French, Russian,
Arabic, Chinese and Japanese. 47 facts about nine people who move, change
roles and change their minds, each name spelt in the turn's script, and 141
chatter turns. Three scopes, one holding another person named Karim. 27
questions, 16 of them about an attribute that changed.

Without `--chat-url`, an **oracle** answers every extraction call with the
correct answer, through the real parser, checks, resolution and contradiction
rules. That measures the retrieval layer, not a model.

Every run also counts what chatter turns produced (facts, nodes, and nodes
named by a speaker word, which must be 0) and, per system, **duplicates**
(questions where two injected units support one event).

Other corpora, each compared only with runs that used it:

| Option | What it adds |
|---|---|
| `--traps` | each person works next to where they live: false supersessions |
| `--rephrase` | a value told again the same month in another language: links, conflicts, and a value closed by its own restatement (counted `restated`) |
| `--speaker` | Alice, the declared owner of `user:alice`, speaks in the first person and is asked about in the first person |
| `--name-traps` | Karima beside Karim, Ivana beside Ivan, Nadine beside Nadia: wrong attachments |

`--dream` adds a dry and a real dream run to a benchmark run, measured
against the known correct answers.

## Results against other approaches

At 150 and 600 tokens, on an RTX 4060 (8 GB) with Ollama in WSL, bge-m3 for
every embedding:

| System | Correct / 1k tokens @150 | Correct @600 | Correct / 1k tokens @600 | Raw recall @600 |
|---|---|---|---|---|
| truncated history | 0.76 | 0.41 | 0.69 | 0.48 |
| vector top-k (raw turns) | 2.86 | 0.41 | 0.69 | **1.00** |
| iifa tier 2, oracle | 3.15 | 0.59 | 1.22 | 0.63 |
| iifa tier 3, oracle | **4.11** | **0.67** | 1.15 | 0.78 |
| iifa tier 2, qwen2.5:7b | 1.84 | 0.33 | 0.63 | 0.44 |
| iifa tier 3, qwen2.5:7b | 2.66 | 0.33 | 0.58 | 0.63 |
| iifa tier 2, qwen3.5:9b | 2.61 | 0.48 | 1.00 | 0.59 |
| iifa tier 3, qwen3.5:9b | 3.06 | 0.44 | 0.79 | 0.70 |

```bash
python bench/run.py                                                          # oracle, tier 2
python bench/run.py --embeddings-url http://127.0.0.1:11434/v1 --embeddings-model bge-m3
python bench/run.py --chat-url http://127.0.0.1:11434/v1 --model qwen2.5:7b \
    --embeddings-url http://127.0.0.1:11434/v1 --embeddings-model bge-m3
python bench/run.py --chat-url http://127.0.0.1:11434/v1 --model qwen3.5:9b \
    --no-json-mode --extra reasoning_effort=none \
    --embeddings-url http://127.0.0.1:11434/v1 --embeddings-model bge-m3
```

What it says:

- **Naive vector top-k finds everything and decides nothing.** It injects the
  current value and every earlier one: raw recall 1.00, but it is correct only
  on the 11 attributes that never changed (0.41), whatever the budget. Its
  contradiction accuracy is 0.
- **iifa with a perfect extractor** gives the most correct answers per token
  at small budgets (4.11 per 1k tokens at 150 with tier 3), and the most
  correct answers at 600 (0.67). Tier 3 adds about 140 ms per recall (the
  question's embedding; p95 ≈ 220 ms).
- **A real model needs a vocabulary of attribute names.** Supersession needs
  a later fact with the *same attribute name* as the earlier one. Left to pick
  names freely (read from the stores with `--keep-store`), qwen2.5:7b leaves
  96 of its 108 facts without an attribute and uses `preference` / `prefers`,
  `works_as` / `job_title`: **0 supersessions, 0 conflicts**. qwen3.5:9b
  leaves 86 of 143 without one and uses `profession`, `occupation` and `role`
  for one thing, `lives_in` and `location` for another (even `valid_from` as
  a key): **6 supersessions**, each where the same key happens to recur, 0
  conflicts.
- **With the vocabulary** (one name per aspect — `home`, `workplace`,
  `role`, … — plus the subject's existing keys in the prompt; only
  unambiguous variants mapped), qwen3.5:9b on the default corpus: correct
  current answers at 600 tokens 0.44 → 0.59 (tier 3), per 1k tokens at 150
  3.06 → 4.28, 11 supersessions, none false. On the trap corpus (`--traps`):
  1 false supersession of 11, a model error (a move written as an office).
- **Aligning synonym attribute names by embeddings stays off for bge-m3**: its
  cosines put traps (`answers` ~ `questions` 0.79) above true synonyms
  (`marital` ~ `relationship` 0.65).

  ```bash
  python bench/key_names.py --embeddings-url http://127.0.0.1:11434/v1 --model bge-m3
  ```
- Without the speaker and complete-statement rules, both models turn chatter
  into facts about a "User" node (qwen2.5:7b: 108 facts for 47 told; 37
  duplicates, 26 fragments, 11 cross-script drops), and short English
  statements translated from another language escape the language check when
  their language cannot be detected.
- A change told in another language than the previous value is iifa's weak
  spot even with the oracle: both facts stay, flagged `conflict`.
- 27 questions make each one worth 0.04: read smaller differences as noise.
  No system injected a record from a scope the question's view cannot see
  (leaks: 0).
- Ingestion: enqueue p95 ≤ 0.3 ms; 188 turns in 18 s with the oracle and
  bge-m3, 856 s with qwen2.5:7b, 795 s with qwen3.5:9b.

## The replacement question

A new role, team or preference does not always replace the old one: "I am
also a photographer" keeps the accountant. So iifa asks the model, for each
such pair: does the new value replace the old one, or can both be true? (See
[contradictions](how-it-works.md#8-contradictions).) These runs decided to
ask it by default, and to show the model the source's own words.

qwen3.5:9b and bge-m3, correct current answers at 600 tokens, tier 2 / tier 3:

| Corpus | Question off | Question, statements only | Question with the source's quotes (default) |
|---|---|---|---|
| default | 0.59 / 0.67 | 0.52 / 0.59 | 0.59 / 0.70 |
| `--rephrase` | 0.81 / 0.85 | 0.70 / 0.78 | 0.81 / 0.89 |
| `--traps` | 0.58 / 0.71 | – | 0.60 / 0.69 |

- **Statements alone lose answers.** The model drops the word that decides
  when it writes a statement ("Émilie travaille *maintenant* comme
  photographe." is stored without "maintenant"; Russian "также", "also", is
  dropped too), so a change reads as an addition. Shown the quotes, it
  answers from them.
- **False replaces** (an added value closing one still true) on the trap
  corpus: 15 with the question off, 1 with statements only, 0 with quotes.
  With quotes, the 14 added values were all kept with "both", and no change
  was kept as "both".
- Verdicts with quotes (replaces / both / same / unsure): default 7 / 2 / 1 /
  0, `--rephrase` 11 / 1 / 7 / 0, `--traps` 7 / 15 / 1 / 1. On `--rephrase`,
  conflicts fell from 8 (off) to 0. False supersessions: 0 in every run with
  quotes.
- With the oracle (tier 2, 600 tokens), on the default corpus: conflicts 5 →
  0, correct supersessions 15 → 20, correct current answers 0.59 → 0.63 (per
  1k tokens 1.22 → 1.40). The oracle always answers "replaces" for a change,
  so a model's runs are the real test.
- The trap corpus at tier 3 is one question lower of 45 with the question on:
  read as noise.
- Cost: one batched model call per extraction that changes a role, team or
  preference, inside the background budget.

```bash
python bench/run.py --chat-url http://127.0.0.1:11434/v1 --model qwen3.5:9b     --no-json-mode --extra reasoning_effort=none     --embeddings-url http://127.0.0.1:11434/v1 --embeddings-model bge-m3 [--rephrase | --traps]
# add --no-replacement-question for the "off" column
```

## Choosing a storage language

Correct current answers out of 27, default corpus, qwen3.5:9b and bge-m3,
tier 3, at 150 / 300 / 600 tokens:

| Storage language | @150 | @300 | @600 |
|---|---|---|---|
| each fact in its source's language (default) | 13 | 18 | 19 |
| forced French | 17 | 20 | 23 |
| forced English | 14 | 23 | 22 |

The gain comes from **one** language, not from English: forced French and
forced English are about equal, and both beat the source's language. Hence
the advice: a single-owner scope forces its owner's language; a shared scope
keeps each source's.

Caveat at tier 2 (no embeddings): a French store answers French questions no
better than the source's language (17 of 27 at 600 tokens), because common
words of the question ("habite", "maintenant") pull other subjects' facts in
ahead of the one asked about.

## Embedding models

`bench/recall_quality.py`: 16 facts in six languages and 12 questions whose
answer is in another language or script. RTX 4060 (8 GB), Ollama in WSL, one
embedding model loaded at a time:

| | tier 0 | bge-m3 | qwen3-embedding:0.6b |
|---|---|---|---|
| recall@1 / @3 / @5 | 0.25 / 0.33 / 0.33 | **0.92 / 1.00 / 1.00** | 0.83 / 1.00 / 1.00 |
| MRR | 0.29 | **0.96** | 0.92 |
| recall p50 / p95 (question embedding included) | 0.1 / 0.5 ms | 153 / 162 ms | 118 / 123 ms |
| cosine "Karim" vs Карим, كريم, 卡里姆, カリム | – | 0.64–0.69 | 0.85–0.92 |
| cosine "Karim" vs other people | – | 0.25–0.38 | 0.53–0.79 |
| loaded in VRAM | – | 0.66 GB | 2.4 GB |

```bash
python bench/recall_quality.py --embeddings-url http://127.0.0.1:11434/v1 --model bge-m3 --repeat 5 --json bench-bge-m3.json
python bench/recall_quality.py --embeddings-url http://127.0.0.1:11434/v1 --model qwen3-embedding:0.6b --repeat 5 --json bench-qwen3-embedding.json
```

**bge-m3 is the recommended embedding model**: the widest gap between one
person's name across scripts and other people, the best recall@1, and a
quarter of the memory once loaded. The same command calibrates any other
model: read its name table, then set the thresholds.

**Built-in thresholds** (cosines; see
[Embedding thresholds](configuration.md#embedding-thresholds)):

| Model | resolve | merge | link |
|---|---|---|---|
| bge-m3 | 0.53 | 0.84 | 0.70 (provisional) |
| qwen3-embedding:0.6b | 0.84 | 0.98 | 0.70 (provisional) |

`bench/thresholds.py` calibrates `resolve` and `merge` on a list of names the
bench corpus does not hold, then verifies them on the corpus. Near names
(Bogdan / Bogdana) score as high as one name's adapted forms, so `merge` sits
above every such trap and makes few proposals, while `resolve` only sends a
candidate to the model. With bge-m3, `resolve` catches 85 % of one person's
forms and `merge` 5 % (traps reach 0.84); with qwen3-embedding:0.6b, 52 % and
1 % (traps reach 0.97). A name cosine alone cannot tell Bogdan from Bogdana,
which is why a merge is also judged on facts, by a model, then by a person.
No model has a measured `key_threshold` yet.

## Extraction against a real model

A manual run, on purpose: the test suite always fakes the model. It feeds the
multilingual synthetic corpus (English, French, Russian, Arabic, Chinese,
Japanese, a mixed sample) to any OpenAI-compatible endpoint through the real
worker, one job at a time. It prints each job's result as it finishes, then
what it wrote, every fact's language check, the proposals, the summaries,
calls and tokens. Ctrl+C stops it with a report of what was written so far.

```bash
python bench/extraction_manual.py --chat-url http://127.0.0.1:11434/v1 --model llama3.2
python bench/extraction_manual.py --chat-url http://127.0.0.1:11434/v1 --model qwen3.5:9b \
    --no-json-mode --extra reasoning_effort=none      # a thinking model behind Ollama
```

**One person across three scripts** (qwen2.5:7b + bge-m3, turns in Russian,
Arabic, then French about the same person):

```bash
python bench/extraction_manual.py --chat-url http://127.0.0.1:11434/v1 --model qwen2.5:7b \
    --embeddings-url http://127.0.0.1:11434/v1 --embeddings-model bge-m3 --samples ru,ar,fr --corpus 0
```

"Карим" is created from the Russian turn. The Arabic "كريم" is near it (0.63)
and goes to the deciding call, which answers "new". The embed job then
proposes a merge of the two (never applies it) and a `similar_to` edge. The
French "Karim" is attached to "Карим" by the deciding call. 15 model calls,
30 s. In the other order (French first) the model reuses the known name
"Karim" for the Russian and Arabic turns and gives the source's spelling as
`subject_as_written`: the three facts land on one node, which gains the
aliases "карим" and "كريم" (9 model calls, 51 s).

## Test suite

No network, no model, no GPU:

```bash
python -m pytest -n auto
```

1842 passed, 8 skipped (perf tests apart) on a Windows laptop with 16
workers: 1 min 40 s with pytest's temporary directory on an SSD, 4 min 14 s on
a spinning disk (`--basetemp` picks the directory; the stores each test
creates make the difference). The skipped tests compare the default token
counter with `tiktoken`; they run in CI on Linux, where the encoding cache is
warmed in a setup step, and skip wherever it is cold, since the suite never
downloads.

Coverage: 96 % of lines on `src/iifa/`, 100 % of branches on the budget
packer and the permission checks.

## Public datasets

iifa ships no third-party data. Two public long-term-memory benchmarks are
downloaded by whoever runs them, into `bench/data/` (ignored by git), and keep
their own licences. Neither runs through `bench/run.py` yet. Both are in
English, so the multilingual properties stay measured on the synthetic
corpus.

**LongMemEval** (Wu et al., ICLR 2025,
[repository](https://github.com/xiaowu0162/LongMemEval)): 500 questions over
chat histories. The repository and the Hugging Face dataset card
(`xiaowu0162/longmemeval-cleaned`, the 2025-09 cleaned release) declare the
MIT licence.

```bash
mkdir -p bench/data/longmemeval && cd bench/data/longmemeval
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json      # 15.4 MB
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json   # 277 MB
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json   # 2.74 GB, optional
```

**LoCoMo** (Maharana et al., 2024,
[repository](https://github.com/snap-research/locomo)): ten very long
two-person conversations with annotated questions. Licence **CC BY-NC 4.0**:
non-commercial use only, with attribution. Use it to measure; never
redistribute it with anything built on iifa.

```bash
mkdir -p bench/data/locomo
curl -L -o bench/data/locomo/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

Cite them when you publish numbers measured on them (the BibTeX entries are
in each repository's README).
