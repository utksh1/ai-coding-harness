"""Sandbox environment tests (review finding #12).

Child processes (test runs, code execution) never see harness credentials;
network egress is argument-checked and proxy-poisoned.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.security.sandbox import network_egress_violation, sandbox_env
from harness.tools.execution import CodeExecutionTool


def test_sandbox_env_strips_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every credential-shaped variable the harness holds stays OUT of the
    child environment (the review's AI_API_KEY leak class)."""
    for name in (
        "AI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CODEX_API_KEY",
        "GEMINI_API_KEY",
        "LUNA_API_KEY",
        "GATEWAY_EVENTS_URL",
        "GATEWAY_EVENTS_LOG",
        "REDIS_URL",
        "ORCHESTRATOR_URL",
        "SOME_RANDOM_TOKEN",
        "DB_PASSWORD",
        "CLIENT_SECRET",
    ):
        monkeypatch.setenv(name, "leak-me")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/user")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")

    env = sandbox_env()
    for name in (
        "AI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "LUNA_API_KEY",
        "GATEWAY_EVENTS_URL",
        "GATEWAY_EVENTS_LOG",
        "REDIS_URL",
        "ORCHESTRATOR_URL",
        "SOME_RANDOM_TOKEN",
        "DB_PASSWORD",
        "CLIENT_SECRET",
    ):
        assert name not in env, f"credential leaked into child env: {name}"
    # The safe set survives.
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == "/home/user"
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_sandbox_env_poisons_proxies_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = sandbox_env(allow_network=False)
    assert env["http_proxy"] == "http://127.0.0.1:9"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert env["no_proxy"] == ""


def test_sandbox_env_defaults_without_path_or_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    env = sandbox_env()
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"]  # tempdir fallback


def test_network_egress_violation_matrix() -> None:
    assert network_egress_violation(["curl", "http://x"]) == "network executable blocked: curl"
    assert network_egress_violation(["git", "status"]) is None
    assert network_egress_violation(["git", "clone", "https://github.com/x"]) is not None
    assert network_egress_violation(["pip", "install", "requests"]) is not None
    assert network_egress_violation(["pip", "list"]) is None
    assert network_egress_violation(["python", "-c", "print(1)"]) is None
    # Fetcher hidden deep in argv is caught too.
    assert network_egress_violation(["make", "test", "curl"]) is not None
    # Network allowed: everything passes.
    assert network_egress_violation(["git", "clone", "https://x"], allow_network=True) is None


def test_code_execution_rejects_egress_arguments(tmp_path: Path) -> None:
    tool = CodeExecutionTool(tmp_path)
    errors = tool.validate_input({"command": ["git", "clone", "https://github.com/x/y"]})
    assert any("network" in e for e in errors)


def test_run_tests_child_env_has_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration: the spawned test runner's environment is the sandboxed
    one - a test that dumps os.environ cannot see the harness's key."""
    monkeypatch.setenv("AI_API_KEY", "super-secret-value")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin:/bin"))
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "test_env.py").write_text(
        "import os\n\n\ndef test_env_clean():\n    assert 'AI_API_KEY' not in os.environ\n"
    )
    from harness.tools.execution import RunTestsTool

    result = RunTestsTool(tmp_path).execute()
    assert result.success, result.error or result.output
