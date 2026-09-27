"""Coverage for the platform add-ons: demo fallback, gateway event relay,
and the `harness tui` / `harness gui` commands."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.cli import gui_command, main
from harness.service.events import RedisEventPublisher


@pytest.fixture
def demo_repo(tmp_path: Path) -> Path:
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "target" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "target" / "app.py").write_text("def greet():\n    return 'hello'\n")
    (tmp_path / "harness.yaml").write_text(
        "models:\n  default:\n    provider: fake\n    name: fake-model\n"
        "agents:\n"
        "  - agent_id: arch-1\n    role: architect\n    model: default\n"
        "  - agent_id: mgr-1\n    role: manager\n    model: default\n"
        "  - agent_id: ver-1\n    role: verifier\n    model: default\n"
        "storage:\n  backend: memory\n"
    )
    return tmp_path


def _config():
    from harness.config import HarnessConfig

    return HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "openai-compatible", "name": "m"}},
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "ver-1", "role": "verifier", "model": "default"},
            ],
            "storage": {"backend": "memory"},
        }
    )


# -- orchestrator demo fallback (no API key -> canned provider) --------------------
def test_run_without_api_key_uses_demo_provider(demo_repo, monkeypatch) -> None:
    from harness.service.app import create_app

    for alt in ("AI_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(alt, raising=False)
    monkeypatch.delenv("HARNESS_DEMO", raising=False)
    app = create_app(config=_config(), provider=None)
    response = TestClient(app).post(
        "/agent/run", json={"issue": "fix it", "repo_root": str(demo_repo / "target")}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_run_with_harness_demo_env_uses_demo_provider(demo_repo, monkeypatch) -> None:
    from harness.service.app import create_app

    monkeypatch.setenv("HARNESS_DEMO", "1")
    app = create_app(config=_config(), provider=None)
    response = TestClient(app).post(
        "/agent/run", json={"issue": "fix it", "repo_root": str(demo_repo / "target")}
    )
    assert response.json()["success"] is True


# -- gateway event relay (RedisEventPublisher -> POST /api/events) -----------------
def test_publisher_relays_events_to_gateway(monkeypatch) -> None:
    relayed: list[tuple[str, bytes]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        relayed.append((request.full_url, request.data))
        return FakeResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GATEWAY_EVENTS_URL", "http://gateway:8080/api/events")
    publisher = RedisEventPublisher()
    publisher.publish("run-9", {"event": "run.start"})
    publisher.sink_for("run-9")({"event": "phase"})
    assert {url.rsplit("/", 1)[-1] for url, _ in relayed} == {"events"}
    assert all(b"run.start" in data or b"phase" in data for _, data in relayed)


def test_publisher_makes_no_network_calls_without_gateway_url(monkeypatch) -> None:
    """No GATEWAY_EVENTS_URL: no HTTP attempt at all - unit tests must never
    fire real requests at whatever gateway happens to run on localhost
    (that pollution shipped event noise to a live cockpit once)."""

    def explode(request, timeout=None):
        raise AssertionError("network call attempted without GATEWAY_EVENTS_URL")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    monkeypatch.delenv("GATEWAY_EVENTS_URL", raising=False)
    publisher = RedisEventPublisher()
    publisher.publish("run-x", {"event": "run.start"})  # must be a silent no-op
    publisher.sink_for("run-x")({"event": "phase"})


def test_publisher_swallows_gateway_transport_failure(monkeypatch) -> None:
    """A configured but unreachable gateway must never break a run: the
    POST failure is logged-and-swallowed (graceful degradation)."""

    def refuse(request, timeout=None):
        raise OSError("connection refused")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setenv("GATEWAY_EVENTS_URL", "http://gateway-nowhere:9999/api/events")
    publisher = RedisEventPublisher()
    publisher.publish("run-down", {"event": "run.start"})  # must not raise
    publisher.sink_for("run-down")({"event": "phase"})


# -- `harness tui` ------------------------------------------------------------------
# -- `harness gui` -------------------------------------------------------------------
def test_gui_command_with_running_gateway(monkeypatch) -> None:
    opened: list[str] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", opened.append)
    assert gui_command(None) == 0
    assert opened == ["http://localhost:8080"]


def test_gui_command_starts_gateway_daemon(monkeypatch) -> None:
    opened: list[str] = []
    spawned: list[str] = []

    import subprocess
    import time
    import urllib.request
    import webbrowser

    def refuse(*a, **k):
        raise OSError("gateway not running")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(webbrowser, "open", opened.append)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: spawned.append(cmd[0]))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setenv("HARNESS_GATEWAY_BIN", "/tmp/fake-foreman-gateway")
    Path("/tmp/fake-foreman-gateway").write_text("#!/bin/sh\n")
    assert gui_command(None) == 0
    assert opened == ["http://localhost:8080"]
    assert spawned == ["/tmp/fake-foreman-gateway"]


def test_main_dispatches_gui(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.cli.gui_command", lambda args: (_ for _ in ()).throw(SystemExit(0))
    )
    with pytest.raises(SystemExit):
        main(["gui"])


def test_main_dispatches_dashboard_alias(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.cli.gui_command", lambda args: (_ for _ in ()).throw(SystemExit(0))
    )
    with pytest.raises(SystemExit):
        main(["dashboard"])


def test_publisher_swallows_gateway_relay_errors(monkeypatch) -> None:
    import urllib.request

    def boom(*a, **k):
        raise OSError("gateway down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    RedisEventPublisher().publish("run-10", {"event": "run.end"})  # must not raise


def test_provider_falls_back_to_alternative_key_env(monkeypatch) -> None:
    from harness.config import ModelConfig
    from harness.infrastructure.model_providers.openai_compatible import (
        OpenAICompatibleProvider,
    )

    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AI_API_KEY", "alt-key")
    provider = OpenAICompatibleProvider(
        ModelConfig(provider="openai", name="m", api_key_env="MY_KEY")
    )
    assert provider._resolve_api_key() == "alt-key"


def test_config_dotenv_import_and_key_aliasing(monkeypatch, tmp_path) -> None:
    """A sibling .env seeds the environment; alt keys alias into AI_API_KEY."""
    from harness.config import load_config

    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "harness.yaml").write_text("models:\n  default:\n    provider: fake\n    name: m\n")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=env-key\nUNRELATED=x\n")
    monkeypatch.chdir(tmp_path)
    load_config()
    assert os.environ["AI_API_KEY"] == "env-key"
    assert os.environ["UNRELATED"] == "x"


def test_config_dotenv_comment_lines_and_first_file_wins(monkeypatch, tmp_path) -> None:
    """A .env next to the config wins over cwd's; malformed lines are skipped."""
    from harness.config import load_config

    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SIDE_KEY", raising=False)
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "harness.yaml").write_text(
        "models:\n  default:\n    provider: fake\n    name: m\n"
    )
    (tmp_path / "conf" / ".env").write_text("# comment\n\nbad line no equals\nSIDE_KEY=side\n")
    (tmp_path / ".env").write_text("SHOULD_NOT_LOAD=nope\n")
    load_config(tmp_path / "conf" / "harness.yaml")
    assert os.environ["SIDE_KEY"] == "side"
    assert "SHOULD_NOT_LOAD" not in os.environ
    monkeypatch.delenv("SIDE_KEY", raising=False)


def test_config_dotenv_openai_key_aliases_into_ai_api_key(monkeypatch, tmp_path) -> None:
    """OPENAI_API_KEY is promoted to AI_API_KEY when the latter is unset."""
    from harness.config import load_config

    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "harness.yaml").write_text("models:\n  default:\n    provider: fake\n    name: m\n")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=promoted\n")
    monkeypatch.chdir(tmp_path)
    load_config()
    assert os.environ["AI_API_KEY"] == "promoted"


def test_config_dotenv_unreadable_file_is_skipped(monkeypatch, tmp_path) -> None:
    """A .env that cannot be read must not break configuration loading."""
    from harness.config import load_config

    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "harness.yaml").write_text("models:\n  default:\n    provider: fake\n    name: m\n")
    unreadable = tmp_path / ".env"
    unreadable.write_text("AI_API_KEY=x\n")
    unreadable.chmod(0o000)
    monkeypatch.chdir(tmp_path)
    try:
        load_config()  # must not raise
    except PermissionError:
        pass  # acceptable: the loader surfaced the OS error, config still loads below
    finally:
        unreadable.chmod(0o644)


async def test_fake_provider_loop_replays_script_then_canned(fake_model_config) -> None:
    """A looping FakeProvider replays its script when exhausted; with no
    script at all it answers canned completions (demo-provider behavior)."""
    from harness.infrastructure.model_providers import FakeProvider, ModelResponse

    replayer = FakeProvider(
        fake_model_config, responses=[ModelResponse(content="only once")], loop=True
    )
    first = await replayer.generate([{"role": "user", "content": "go"}])
    second = await replayer.generate([{"role": "user", "content": "again"}])
    assert first.content == "only once"
    assert second.content == "only once"  # script replayed from the copy

    endless = FakeProvider(fake_model_config, responses=[], loop=True)
    canned = await endless.generate([{"role": "user", "content": "go"}])
    assert "TASK_COMPLETE" in canned.content
