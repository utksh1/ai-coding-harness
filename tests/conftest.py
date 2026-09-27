"""Shared fixtures: isolated stores, scripted providers, mock GitHub transports.

These are the "mock utilities for model API / GitHub API responses" required
by foundation issue 1.8; use them instead of hitting any network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from harness.agents.task import Task
from harness.config import HarnessConfig, ModelConfig
from harness.infrastructure.context_store import (
    ContextStore,
    MemoryContextStore,
    SQLiteContextStore,
)
from harness.infrastructure.git_local import GitService
from harness.infrastructure.github import GitHubService
from harness.infrastructure.model_providers.fake import FakeProvider


@pytest.fixture
def memory_store() -> MemoryContextStore:
    return MemoryContextStore()


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteContextStore:
    return SQLiteContextStore(tmp_path / "context.db")


@pytest.fixture(params=["memory", "sqlite"], ids=["memory-store", "sqlite-store"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> ContextStore:
    if request.param == "memory":
        return MemoryContextStore()
    return SQLiteContextStore(tmp_path / "context.db")


@pytest.fixture
def sample_config() -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {
                "default": {"provider": "fake", "name": "fake-model", "api_key_env": "AI_API_KEY"}
            },
            "agents": [
                {"agent_id": "architect-1", "role": "architect", "model": "default"},
                {"agent_id": "implementer-1", "role": "implementer", "model": "default"},
            ],
        }
    )


@pytest.fixture
def fake_model_config() -> ModelConfig:
    return ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY")


@pytest.fixture
def make_fake_provider(fake_model_config: ModelConfig) -> Callable[..., FakeProvider]:
    """Factory: `make_fake_provider(*script)` builds a scripted offline provider."""

    def _make(*script: Any) -> FakeProvider:
        return FakeProvider(fake_model_config, responses=list(script))

    return _make


@pytest.fixture
def sample_task() -> Task:
    return Task(
        id="task-1",
        title="Fix parser crash on empty input",
        description="parser.py crashes with IndexError when the issue body is empty.",
        acceptance_criteria=["empty input no longer raises", "existing tests still pass"],
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> GitService:
    """A scratch repository directory (call `init` or use `git_ready`)."""
    repo = GitService(tmp_path / "repo")
    repo.repo_path.mkdir(parents=True, exist_ok=True)
    return repo


@pytest.fixture
async def git_ready(git_repo) -> GitService:
    """An initialized scratch repository with repo-local identity.

    CI runners and eval environments have no global git identity; the
    repository must carry its own for commits to work anywhere.
    """
    await git_repo.init()
    await git_repo.set_identity("Harness Test", "harness@test.local")
    return git_repo


@pytest.fixture
def github_offline() -> GitHubService:
    """Force-unavailable service: proves the graceful-degradation path."""
    return GitHubService(token="", client=httpx.AsyncClient())


@pytest.fixture
def mock_github() -> Callable[[dict[str, Any]], GitHubService]:
    """Factory: `mock_github({path_suffix: response_json})` returns a service
    whose REST calls are answered from the map (recorded into `service.sent`)."""
    captured: list[httpx.Request] = []

    def _make(responses: dict[str, Any]) -> GitHubService:
        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            for suffix, payload in responses.items():
                if request.url.path.endswith(suffix):
                    return httpx.Response(200, json=payload)
            return httpx.Response(200, json={})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://api.github.com"
        )
        return GitHubService(token="test-token", client=client)

    _make.captured = captured  # type: ignore[attr-defined]
    return _make


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Run a test inside an empty cwd with no API key configured."""
    monkeypatch.chdir(tmp_path)
    for alt in ("AI_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(alt, raising=False)
    yield tmp_path


@pytest.fixture(autouse=True)
def _isolated_run_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Keep the orchestrator's persisted run->repo registry OUT of the repo.

    `harness.service.app` writes `.harness/run-roots.json` relative to the
    process cwd; without this, every service test that hits /agent/run
    pollutes the repository tree (and the next real process then loads
    pytest tmp-dir entries as "known runs").
    """
    try:
        from harness.service import app as service_app
    except ImportError:  # pragma: no cover - platform extra not installed
        yield tmp_path
        return
    monkeypatch.setattr(service_app, "RUN_ROOTS_FILE", tmp_path / "run-roots.json")
    yield tmp_path
