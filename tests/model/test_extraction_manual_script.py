"""The manual extraction run (``bench/extraction_manual.py``) works end to end
against the fake model server, so the owner's local run against a real model
starts from a script that is known to run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.fakes.openai_server import Fail, FakeOpenAIServer, Reply, chat_json

SCRIPT = Path(__file__).resolve().parents[2] / "bench" / "extraction_manual.py"


def load_script() -> object:
    spec = importlib.util.spec_from_file_location("extraction_manual", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["extraction_manual"] = module
    spec.loader.exec_module(module)
    return module


def test_manual_run_prints_what_it_wrote_and_its_language_checks(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_openai.set_default(
        chat=chat_json(
            {
                "nodes": [{"name": "Karim", "kind": "person"}],
                "facts": [
                    {
                        "subject": "Karim",
                        "statement": "Karim dirige la planification du projet solaire.",
                        "quote": "dirige la planification",
                        "attribute": "rôle",
                        "value": "planification",
                    }
                ],
                "edges": [],
            }
        )
    )
    fake_openai.script(chat=[])
    module = load_script()
    code = module.main(  # type: ignore[attr-defined]
        [
            "--chat-url",
            fake_openai.base_url,
            "--model",
            "fake-chat",
            "--samples",
            "fr,ru",
            "--corpus",
            "0",
            "--verbose",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "queued sample:fr" in out and "queued sample:ru" in out
    assert "model calls" in out and "language checks" in out
    assert "Karim dirige la planification du projet solaire." in out
    assert (
        "summary:" in out
    )  # the session end wrote summaries (the fake answers JSON, kept as text)
    assert (
        len(fake_openai.chat_calls) >= 3
    )  # two turns (one retried on the Russian source) + summaries
    assert all(c.model == "fake-chat" for c in fake_openai.chat_calls)


def test_manual_run_refuses_an_unknown_sample_and_help_works(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_script()
    with pytest.raises(SystemExit):
        module.main(["--chat-url", fake_openai.base_url, "--model", "m", "--samples", "klingon"])  # type: ignore[attr-defined]
    with pytest.raises(SystemExit) as stopped:
        module.main(["--help"])  # type: ignore[attr-defined]
    assert stopped.value.code == 0
    assert "multilingual synthetic corpus" in capsys.readouterr().out
    fake_openai.script(chat=[Reply("not json")])


KARIM = chat_json(
    {
        "nodes": [{"name": "Karim", "kind": "person"}],
        "facts": [
            {
                "subject": "Karim",
                "statement": "Karim dirige la planification du projet solaire.",
                "quote": "dirige la planification",
            }
        ],
        "edges": [],
    }
)


def test_manual_run_reports_each_job_as_it_goes(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_openai.set_default(chat=KARIM)
    module = load_script()
    code = module.main(  # type: ignore[attr-defined]
        [
            *("--chat-url", fake_openai.base_url, "--model", "fake-chat"),
            *("--samples", "fr,en", "--corpus", "0", "--no-session-end"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    lines = out.splitlines()
    progress = [line for line in lines if line.startswith("[")]
    assert [line.split()[0] for line in progress] == ["[1/2]", "[2/2]"]
    assert all(" turn " in line and "done" in line and " s" in line for line in progress)
    # progress comes before the final report
    assert lines.index(progress[-1]) < lines.index("nodes:")


def test_manual_run_survives_an_interruption_with_a_partial_report(
    fake_openai: FakeOpenAIServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_openai.set_default(chat=KARIM)
    module = load_script()
    real_run = module.Worker.run  # type: ignore[attr-defined]
    calls = {"n": 0}

    def interrupted(self: object, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(module.Worker, "run", interrupted)  # type: ignore[attr-defined]
    code = module.main(  # type: ignore[attr-defined]
        ["--chat-url", fake_openai.base_url, "--model", "fake-chat", "--samples", "fr,en,ru"]
    )
    out = capsys.readouterr().out
    assert code == 130
    assert "interrupted after 1 of" in out
    assert "Karim dirige la planification du projet solaire." in out  # what was written so far
    assert "totals:" in out


def test_manual_run_passes_json_mode_extras_and_the_cap(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_openai.set_default(chat=KARIM)
    module = load_script()
    code = module.main(  # type: ignore[attr-defined]
        [
            *("--chat-url", fake_openai.base_url, "--model", "fake-chat", "--samples", "fr"),
            *("--corpus", "0", "--no-session-end", "--no-json-mode"),
            *("--extra", "reasoning_effort=none", "--extra", 'options={"num_ctx": 4096}'),
            *("--max-tokens", "900"),
        ]
    )
    assert code == 0
    body = fake_openai.chat_calls[0].body
    assert "response_format" not in body
    assert body["reasoning_effort"] == "none" and body["options"] == {"num_ctx": 4096}
    assert body["max_tokens"] <= 900
    with pytest.raises(SystemExit):
        module.main(  # type: ignore[attr-defined]
            ["--chat-url", fake_openai.base_url, "--model", "m", "--extra", "max_tokens=5"]
        )
    capsys.readouterr()


def test_a_retried_job_does_not_advance_the_progress_count(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str]
) -> None:
    """A job retried after a transport failure is counted once, never "[12/11]":
    only a final outcome advances the count."""
    fake_openai.set_default(chat=KARIM)
    fake_openai.script(chat=[Fail(500)])
    module = load_script()
    module.main(  # type: ignore[attr-defined]
        [
            *("--chat-url", fake_openai.base_url, "--model", "fake-chat"),
            *("--samples", "fr,en", "--corpus", "0", "--no-session-end"),
        ]
    )
    progress = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    assert [line.split()[0] for line in progress] == ["[retry]", "[1/2]"]
    assert " retry " in progress[0] and " done " in progress[1]


def test_manual_run_at_tier_3_embeds_after_every_turn(
    fake_openai: FakeOpenAIServer, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_openai.set_default(chat=KARIM)
    module = load_script()
    code = module.main(  # type: ignore[attr-defined]
        [
            *("--chat-url", fake_openai.base_url, "--model", "fake-chat"),
            *("--embeddings-url", fake_openai.base_url, "--embeddings-model", "fake-embed"),
            *("--samples", "fr,ru", "--corpus", "0", "--no-session-end"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "calibration uncalibrated" in out  # fake-embed has no profile
    progress = [line.split()[0] for line in out.splitlines() if line.startswith("[")]
    assert progress == ["[1/2]", "[embed]", "[2/2]", "[embed]"]
    assert fake_openai.embedding_calls and all(
        c.model == "fake-embed" for c in fake_openai.embedding_calls
    )
    with pytest.raises(SystemExit):
        module.main(  # type: ignore[attr-defined]
            ["--chat-url", fake_openai.base_url, "--model", "m", "--embeddings-url", "http://x"]
        )
    capsys.readouterr()
