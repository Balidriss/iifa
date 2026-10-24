"""End-to-end check of the Docker image (SPEC 13.1): start the container, health,
remember, recall, archive, stop - plus what the image promises (SPEC 12, P5): a
non-root user, the key neither in the image nor on the data volume, the database
encrypted on that volume, memory kept across a restart, the default bind on
127.0.0.1 inside the container.

Needs Docker and a built image; the standard library only, so it runs before
anything is installed. CI runs it (``docker-e2e`` job); by hand:

    docker build -t iifa:e2e .
    python tests/e2e/docker_smoke.py --image iifa:e2e
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

SCOPE = "user:alice/mode:assistant"
CANARY = "Karim leads the solar planning " + secrets.token_hex(4)
READY_WITHIN = 60.0  # seconds for the service to answer /v1/health
UID = "10001"


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check, encoding="utf-8"
    )


def say(step: str) -> None:
    print(f"docker-e2e: {step}", flush=True)


class Api:
    def __init__(self, base: str, token: str) -> None:
        self.base = base
        self.token = token

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        status, payload = self.raw(method, path, body)
        if status >= 400:
            raise AssertionError(f"{method} {path} answered {status}: {payload}")
        return payload

    def raw(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    def wait_ready(self) -> dict[str, Any]:
        until = time.monotonic() + READY_WITHIN
        while True:
            try:
                health: dict[str, Any] = self.call("GET", "/v1/health")
                return health
            except (OSError, AssertionError) as exc:
                if time.monotonic() > until:
                    raise AssertionError(f"not ready within {READY_WITHIN:.0f} s: {exc}") from exc
                time.sleep(0.5)


def run_service(image: str, name: str, volume: str, key_file: Path, token: str) -> tuple[str, Api]:
    docker(
        "run", "--detach", "--name", name,
        "--env", "IIFA_HOST=0.0.0.0",  # inside the container; published on loopback only
        "--env", f"IIFA_TOKEN={token}",
        "--publish", "127.0.0.1::8765",
        "--mount", f"type=volume,source={volume},target=/data",
        "--mount", f"type=bind,source={key_file},target=/run/secrets/iifa_db_key,readonly",
        image,
    )  # fmt: skip
    port = docker("port", name, "8765/tcp").stdout.strip().splitlines()[0].rsplit(":", 1)[1]
    return name, Api(f"http://127.0.0.1:{port}", token)


def logs(name: str) -> str:
    result = docker("logs", name, check=False)
    return result.stdout + result.stderr


def check_image(image: str) -> None:
    config = json.loads(docker("image", "inspect", image).stdout)[0]["Config"]
    assert config["User"] == f"{UID}:{UID}", f"image user is {config['User']!r}"
    env = dict(item.split("=", 1) for item in config["Env"])
    assert "IIFA_DB_KEY" not in env, "a key is baked into the image"
    assert env["IIFA_HOST"] == "127.0.0.1", "the default bind inside is not loopback"
    key_dir = PurePosixPath(env["IIFA_DB_KEY_FILE"]).parent
    assert key_dir != PurePosixPath(env["IIFA_DB_PATH"]).parent, "the key sits next to the database"
    say("image: non-root user, no key in its environment, loopback by default")
    probe = docker(
        "run", "--rm", "--entrypoint", "python", image, "-c",
        "import os, sys; sys.exit(os.path.exists(os.environ['IIFA_DB_KEY_FILE']))",
        check=False,
    )  # fmt: skip
    assert probe.returncode == 0, "the image carries a key file"
    refused = docker("run", "--rm", "--env", "IIFA_TOKEN=x", image, check=False)
    assert refused.returncode == 2 and "database key file not found" in refused.stderr, (
        refused.returncode,
        refused.stderr,
    )
    say("image: holds no key file, and refuses to start without one")


def check_service(api: Api, name: str) -> str:
    health = api.wait_ready()
    assert health["status"] == "ok" and health["tier"] == 0, health
    say(f"health: ok, version {health['version']}, tier {health['tier']}")
    assert docker("exec", name, "id", "-u").stdout.strip() == UID
    remembered = api.call(
        "POST",
        "/v1/remember",
        {"scope": SCOPE, "statement": CANARY, "node": {"name": "Karim", "kind": "person"}},
    )
    fact_id: str = remembered["fact"]["id"]
    say(f"remember: {fact_id}")
    recalled = api.call(
        "POST", "/v1/recall", {"scope": SCOPE, "query": "who leads the solar planning?"}
    )
    assert fact_id in [item["id"] for item in recalled["items"]], recalled
    assert recalled["used_tokens"] <= recalled["budget_tokens"], recalled
    say(f"recall: found, {recalled['used_tokens']}/{recalled['budget_tokens']} tokens")
    status, refused = api.raw(
        "POST", "/v1/recall", {"scope": "user:alice/assistant", "query": "solar"}
    )
    assert status == 422 and "needs kind:name, like mode:assistant" in json.dumps(refused)
    say("recall with a bad scope: 422 naming the segment")
    api.call("POST", "/v1/archive", {"id": fact_id})
    recalled = api.call(
        "POST", "/v1/recall", {"scope": SCOPE, "query": "who leads the solar planning?"}
    )
    assert fact_id not in [item["id"] for item in recalled["items"]], recalled
    say("archive: the fact no longer comes back")
    listing = docker("exec", name, "ls", "-A", "/data").stdout.split()
    assert listing and all(entry.startswith("iifa.db") for entry in listing), listing
    raw = docker(
        "exec", name, "python", "-c",
        "import sys; data = open('/data/iifa.db', 'rb').read(); "
        f"sys.exit(data.startswith(b'SQLite format 3') or {CANARY.encode()!r} in data)",
        check=False,
    )  # fmt: skip
    assert raw.returncode == 0, "the database file is readable without the key"
    say(f"data volume: {', '.join(listing)} only; the file is encrypted")
    return fact_id


def stop(name: str) -> None:
    started = time.monotonic()
    docker("stop", "--time", "20", name)
    code = json.loads(docker("inspect", name).stdout)[0]["State"]["ExitCode"]
    assert code == 0, f"exit code {code} on stop\n{logs(name)}"
    say(f"stop: clean exit in {time.monotonic() - started:.1f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--image", default="iifa:e2e")
    args = parser.parse_args(argv)
    tag = secrets.token_hex(4)
    volume, names = f"iifa-e2e-data-{tag}", []
    token = secrets.token_urlsafe(24)
    with tempfile.TemporaryDirectory() as tmp:
        # the README's setup: a 0644 key inside a 0700 directory. Docker mounts
        # the file as root, so uid 10001 reads it; no other host user can.
        secrets_dir = Path(tmp) / "iifa"
        secrets_dir.mkdir(mode=0o700)
        secrets_dir.chmod(0o700)
        key_file = secrets_dir / "db.key"
        key_file.write_text(secrets.token_urlsafe(32), encoding="utf-8")
        key_file.chmod(0o644)
        docker("volume", "create", volume)
        try:
            check_image(args.image)
            name, api = run_service(args.image, f"iifa-e2e-{tag}", volume, key_file, token)
            names.append(name)
            fact_id = check_service(api, name)
            stop(name)
            name, api = run_service(args.image, f"iifa-e2e-{tag}-again", volume, key_file, token)
            names.append(name)
            api.wait_ready()
            archived = api.call("GET", f"/v1/archive?scope={SCOPE}")
            assert fact_id in json.dumps(archived), archived
            say("restart on the same volume: the archived fact is still there")
            stop(name)
        except Exception:
            for name in names:
                print(logs(name), file=sys.stderr)
            raise
        finally:
            for name in names:
                docker("rm", "--force", name, check=False)
            docker("volume", "rm", "--force", volume, check=False)
    say("passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
