# Dependencies

Every runtime dependency, with why the standard library or an existing
dependency is not enough (SPEC §7.5, §16). Keep this list short.

## Runtime

| Package | Why |
|---|---|
| `fastapi` | The HTTP API (§8): request validation, OpenAPI document (the contract the snapshot test guards), dependency injection for auth. The standard library has no equivalent. |
| `uvicorn` | ASGI server for FastAPI. The standard library's `http.server` is not an ASGI server. |
| `pydantic` | Request/response schemas with validation; FastAPI is built on it, so it is a direct dependency we rely on rather than a transitive one. |
| `sqlcipher3-wheels` | SQLCipher binding (§7.1) with prebuilt wheels for Linux, macOS **and Windows** on Python 3.11–3.14, which is what makes the Windows CI matrix possible. The stdlib `sqlite3` module cannot open an encrypted file. |
| `httpx` | The Python SDK (`iifa.client`, §10.1): one HTTP client with a blocking and an asyncio API, connection pooling, timeouts, and an in-process ASGI transport that lets the MCP endpoint mounted on the service call the API without a socket. `urllib` has no async API and no ASGI transport. Also the model client (`iifa.llm`, §7.5): the same pooled client with per-call timeouts toward the configured chat and embeddings endpoints, `trust_env=False` so no proxy variable can redirect a call (P4). |
| `mcp` | The official MCP Python SDK (§7.5, §9): protocol types, the stdio and streamable HTTP transports, tool schema generation from signatures. Reimplementing the protocol would be the opposite of "thin". Pinned to the 2.x line, whose API differs from 1.x. It pulls `httpx2`, `jsonschema`, `pyjwt[crypto]`, `sse-starlette`, `python-multipart` and `opentelemetry-api`; the last one is an API package with no exporter, so nothing leaves the process (P4 is enforced by the socket guard in tests either way). |

Everything else is the standard library: `tomllib` for the config file,
`hashlib`/`hmac` for token digests and constant-time comparison,
`unicodedata` for the query tokenizer, `os.urandom` for ULIDs, `logging`,
`threading` (the background worker, the checkpoint thread and the P11 gate
are a thread, a `Condition` and a lock), `argparse`, `difflib` for fuzzy
entity resolution (SPEC §5.1 step 2), and `re`/`unicodedata` again for
**language detection** (P12, `iifa.language`): scripts by Unicode range plus
function-word lists for the Latin-script languages, which is enough to tell a
fact's language from its source's and costs no dependency, no model file and no
download at runtime (P4). A language-identification package was considered
and not added: the ones that work offline either ship a model file of tens of
megabytes or need compiled extensions without Windows wheels, for a job the
standard library does well enough at sentence length, and the detector answers
"undetermined" rather than guessing when a text holds no evidence. The Claude Code hook script (`adapters/claude-code`)
uses only the standard library on purpose, so a broken or missing `iifa`
install can never break a Claude Code session.
The Hermes Agent provider (`adapters/hermes`) is standard library only for
the same reason, and because Hermes' environment is not ours to fill. The
opencode plugin (`adapters/opencode`, TypeScript) has no dependency: it uses
opencode's `tool` helper, which opencode installs itself, and its tests run
with Node's built-in `node --test` (Node 22.18+ runs TypeScript directly).

## Optional extras

| Extra | Package | Why |
|---|---|---|
| `exact-tokens` | `tiktoken` | An exact token counter (§7.3). Its encoding file must be pre-cached by the operator (`TIKTOKEN_CACHE_DIR`): iifa never downloads at runtime (P4). |
| `vectors` | `numpy` | Tier 3 (§6.1): a brute-force cosine scan of 10 000 vectors of dimension 1024 takes milliseconds instead of ~124 ms with `math.sumprod` on Python 3.12 (~370 ms on 3.11), measured on a 16-core laptop. Optional on purpose: without it tier 3 still works on a pure-Python path, identical results, slower past the size `/v1/health` reports; installs without embeddings stay light. `sqlite-vec` would fit behind the same interface. |
| `openai-agents` | `openai-agents` (>=0.24, <1) | Only for the OpenAI Agents SDK example (`adapters/openai-agents`, SPEC 10.5): it builds `Agent` tools and dynamic instructions, so it needs the SDK itself; the standard library cannot stand in for it. Never imported by `src/iifa` (a test checks). CI installs it so the example's tests run; it pulls `openai` and `websockets`, and changes none of the core's pins. |

## Dev

| Package | Why |
|---|---|
| `pytest`, `pytest-xdist` | The suite, run in parallel (§13). |
| `pytest-cov` | Coverage floors: 90 % lines on `src/iifa/`, 100 % branches on the packer and the permission checks. |
| `hypothesis` | Property-based tests for the packer and scope filtering (§13.1). |
| `ruff` | Formatter and linter (§13.3). |
| `mypy` | Strict type checking on `src/` (§13.3). |
| `numpy` | The same as the `vectors` extra, so that the parity test (numpy path = pure-Python path) always runs. |

## Build

| Package | Why |
|---|---|
| `hatchling` | PEP 517 build backend; not installed at runtime. |

## Container

| Image | Why |
|---|---|
| `python:3.12-slim-bookworm` | The Docker image's base (`Dockerfile`, both stages): the official Python image, Debian slim, on the Python the suite is checked with. `sqlcipher3-wheels` installs from its manylinux wheel, so no compiler and no system SQLCipher package are needed. Overridable with `--build-arg PYTHON_IMAGE=...`. |
