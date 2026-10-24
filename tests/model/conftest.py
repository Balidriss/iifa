"""Model-dependent tests (SPEC 13.1): the app at tier 2 against the scripted fake
model server, the worker in ``on_demand`` mode (see ``tests.tier2``)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from iifa.api.app import create_app
from iifa.config import Settings
from tests.conftest import FakeClock
from tests.fakes.openai_server import FakeOpenAIServer
from tests.helpers import auth
from tests.tier2 import tier2_settings


@pytest.fixture
def settings(tmp_path: Path, fake_openai: FakeOpenAIServer) -> Settings:
    return tier2_settings(tmp_path, fake_openai)


@pytest.fixture
def app(settings: Settings, clock: FakeClock) -> Iterator[FastAPI]:
    application = create_app(settings, clock=clock)
    yield application
    application.state.store.close()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, headers=auth("all")) as test_client:
        yield test_client
