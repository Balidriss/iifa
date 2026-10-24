"""The real service on a loopback port, for tests that need a socket: the SDK's
blocking client, the MCP server over stdio (a subprocess), the Claude Code hooks
(a subprocess) and the reference loop. Same pattern as the fake model server:
uvicorn in a thread, a random port, no network beyond 127.0.0.1."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI

from iifa.api.app import create_app
from iifa.config import Settings
from tests.helpers import TOKENS, make_settings


class LiveService:
    """``with LiveService(tmp_path) as service:`` then ``service.url``."""

    def __init__(self, tmp_path: Path, **settings_overrides: Any) -> None:
        self.settings: Settings = make_settings(tmp_path, **settings_overrides)
        self.app: FastAPI = create_app(self.settings)
        self.host = "127.0.0.1"
        self.port = 0
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def token(name: str = "all") -> str:
        return TOKENS[name][0]

    def start(self) -> LiveService:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, 0))
        sock.listen(64)
        self.port = sock.getsockname()[1]
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="error", lifespan="on"
        )
        self._server = uvicorn.Server(config)
        server = self._server

        def run() -> None:
            asyncio.run(server.serve(sockets=[sock]))

        self._thread = threading.Thread(target=run, name="live-iifa-service", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 15
        while not server.started:
            if time.monotonic() > deadline:  # pragma: no cover - only on a broken machine
                raise RuntimeError("the service did not start")
            time.sleep(0.005)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=15)
        self._server = None
        self._thread = None

    def __enter__(self) -> LiveService:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
