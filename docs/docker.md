# Docker

The image runs the service as an unprivileged user and keeps the two secrets
of a memory apart: the database on a volume, the key in a read-only file
mounted somewhere else.

- [Run it](#run-it)
- [Where things live](#where-things-live)
- [The key file](#the-key-file)
- [Publishing the port safely](#publishing-the-port-safely)
- [Connecting a model on the host](#connecting-a-model-on-the-host)
- [Build options](#build-options)
- [What CI checks](#what-ci-checks)

## Run it

```bash
git clone https://github.com/Balidriss/iifa && cd iifa
docker build -t iifa .

mkdir -p -m 0700 ~/.config/iifa && chmod 0700 ~/.config/iifa
openssl rand -base64 32 > ~/.config/iifa/db.key
openssl rand -hex 24 > ~/.config/iifa/token
chmod 0644 ~/.config/iifa/db.key

docker volume create iifa-data
docker run -d --name iifa \
  -e IIFA_HOST=0.0.0.0 -e IIFA_TOKEN="$(cat ~/.config/iifa/token)" \
  -p 127.0.0.1:8765:8765 \
  --mount type=volume,source=iifa-data,target=/data \
  --mount type=bind,source="$HOME/.config/iifa/db.key",target=/run/secrets/iifa_db_key,readonly \
  iifa
```

**Git Bash on Windows:** put `MSYS_NO_PATHCONV=1` in front of `docker run`, or
Git Bash rewrites `/data` into a Windows path.

## Where things live

| What | Where | How |
|---|---|---|
| The database | `/data/iifa.db`, on a volume | `--mount type=volume,source=iifa-data,target=/data`. Only `iifa.db` and its `-wal` / `-shm` files live there, encrypted |
| The key | `/run/secrets/iifa_db_key`, a read-only file | a bind mount or a Docker secret. Never in the image, never on the data volume |
| A token | `IIFA_TOKEN`, or `[[tokens]]` in a config file | beyond a first try, mount a config file as a secret and set `IIFA_CONFIG=/run/secrets/iifa.toml`: an environment variable shows in `docker inspect` |

The image is Python 3.12 slim and runs as uid 10001.

## The key file

The service refuses to start without its key (`database key file not found`,
exit code 2).

On Linux, a bind-mounted file keeps its owner and mode from the host, so the
container's user must be able to read it. The setup above needs no `sudo`:
the file is `0644` inside a `0700` directory. The container reads it; no
other local user can enter the directory.

On a machine where the directory alone should not be trusted, give the file
to the container's user instead:

```bash
sudo chown 10001 ~/.config/iifa/db.key && sudo chmod 0400 ~/.config/iifa/db.key
```

Docker Desktop (macOS, Windows) needs neither.

## Publishing the port safely

Inside the container the service binds `127.0.0.1` unless `IIFA_HOST` says
otherwise, so a container started without it answers nobody.
`IIFA_HOST=0.0.0.0` opens it to Docker's network only. Then publish it on the
host's loopback:

- `-p 127.0.0.1:8765:8765`: this machine only. Use this.
- Never `-p 8765:8765`: Docker publishes on every interface of the host, and
  its rules go around host firewalls such as ufw.
- To reach it from another machine, put a TLS reverse proxy in front (on the
  host or in the same Docker network), and give each client its own
  scope-bound token without `forget` or `admin`. iifa itself speaks plain
  HTTP.

## Connecting a model on the host

A model server running on the host (Ollama, llama.cpp, vLLM) is reached from
the container at `host.docker.internal`:

```bash
docker run ... \
  -e IIFA_CHAT_URL=http://host.docker.internal:11434/v1 -e IIFA_CHAT_MODEL=qwen3.5:9b \
  iifa
```

On Linux, add `--add-host=host.docker.internal:host-gateway`. All the model
settings are in [Configuration](configuration.md#environment-variables).

## Build options

```bash
docker build --build-arg IIFA_EXTRAS=vectors,exact-tokens -t iifa .
```

`vectors` adds numpy for fast embeddings search; `exact-tokens` adds
tiktoken (see [Token counting](security.md#token-counting)).

The image has a healthcheck (a connection from inside: `/v1/health` itself
needs a token). A clean `docker stop` exits 0 within a second.

## What CI checks

CI builds the image on every push and runs `tests/e2e/docker_smoke.py`
against it: health, remember, recall (and a bad scope's 422), archive, a stop
and a restart on the same volume. It also checks the promises above: a
non-root user, no key in the image or on the data volume, and a database file
that plain SQLite cannot read.

The image is built and checked for `linux/amd64` (CI, and Docker Desktop on
Windows); other architectures are untested.
