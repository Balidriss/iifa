# Development

## Set up

```bash
git clone https://github.com/Balidriss/iifa && cd iifa
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,exact-tokens,openai-agents]"   # the last extra only for the Agents SDK example's tests
```

## Checks

```bash
ruff check . && ruff format --check . && mypy
python -m pytest -n auto --cov
node --test adapters/opencode/test/plugin.test.ts   # the opencode plugin (Node 22.18+, no npm install)
docker build -t iifa:e2e . && python tests/e2e/docker_smoke.py --image iifa:e2e   # the image, end to end
```

Coverage floors: 90 % of lines on `src/iifa/`, 100 % of branches on the
budget packer and the permission checks. The suite never touches the network
and always fakes the model. Its timing: [Benchmarks](benchmarks.md#test-suite).

## After certain changes

```bash
IIFA_UPDATE_SNAPSHOT=1 python -m pytest tests/api/test_openapi_snapshot.py   # after an intended API change
python -m tests.fixtures.make_fixture_db                                       # after a new schema version
```

## Against a real model

```bash
python bench/extraction_manual.py --chat-url http://127.0.0.1:11434/v1 --model llama3.2   # extraction
python bench/run.py                                                                        # the benchmark (oracle extraction)
```

More options and the results: [Benchmarks](benchmarks.md).

## Platforms

Linux and Windows, tested; macOS expected, untested. Python 3.11 to 3.13.
The SQLCipher binding (`sqlcipher3-wheels`) ships wheels for all three; CI
runs the suite on Linux and Windows only. The Docker image is built and checked for `linux/amd64`.

## Rules of the repository

- The design is in [SPEC.md](../SPEC.md). A change of behaviour changes the
  SPEC first.
- Every runtime dependency is justified in
  [DEPENDENCIES.md](../DEPENDENCIES.md).
- User-facing changes go to [CHANGELOG.md](../CHANGELOG.md), one short line
  each, under `Unreleased`.
