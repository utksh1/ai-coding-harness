"""Architect-stage resilience: transport failures at t=0 must not kill runs.

Live finding (run 99013099): a flapping provider pool (HTTP 503 for the
first minutes) crashed the whole run at the architect's first structured
call - the recovery ladder only covers specialist tasks, which do not
exist yet at that point. The stage now retries with backoff; each retry
starts from a clean planning window and emits an `architect.retry` event
for the cockpits.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from harness.config import HarnessConfig
from harness.engine.pipeline import HarnessPipeline
from harness.infrastructure.context_store import SQLiteContextStore
from harness.infrastructure.model_providers import FakeProvider, ModelResponse

PROFILE_JSON = (
    '{"languages": ["Python"], "frameworks": ["pytest"], '
    '"test_framework": "pytest", "build_system": "pyproject.toml", '
    '"conventions": ["typed"], "notes": "tiny demo repo"}'
)
PLAN_JSON = (
    '{"issue_summary": "greeting missing", "complexity": 3, "subtasks": [{'
    '"id": "st-1", "title": "add greeting", '
    '"description": "app.greet() should return hello", '
    '"specialty": "verification", "complexity": 2, "files": ["app.py"], '
    '"acceptance_criteria": ["greet returns hello"], "depends_on": []}], '
    '"risks": [], "needs_collaboration": false}'
)
VERDICT_JSON = '{"approved": true, "issues": [], "summary": "greet works"}'


@pytest.fixture
def demo_repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "test_greet.py").write_text(
        "from app import greet\n\n\ndef test_greet():\n    assert greet() == 'hello'\n"
    )
    (tmp_path / "app.py").write_text("def greet():\n    return 'hello'\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    return tmp_path


@pytest.fixture
def config(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {
                "default": {"provider": "fake", "name": "fake-model", "api_key_env": "AI_API_KEY"}
            },
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "ver-1", "role": "verifier", "model": "default"},
            ],
            "storage": {"backend": "sqlite", "sqlite_path": ".harness/pipeline.db"},
            "run": {"results_dir": "results"},
        }
    )


def _build(demo_repo: Path, config: HarnessConfig, provider: FakeProvider):
    from harness.security.audit import AuditLog

    store = SQLiteContextStore(demo_repo / ".harness" / "pipeline.db")
    pipeline = HarnessPipeline(
        demo_repo, config, provider, store, audit=AuditLog(demo_repo / ".harness" / "audit.jsonl")
    )
    return pipeline, store


async def test_transient_outage_at_t0_is_retried(
    demo_repo: Path, config: HarnessConfig, fake_model_config, monkeypatch
) -> None:
    # First analyze call dies like a provider pool outage; the stage retries.
    monkeypatch.setattr(HarnessPipeline, "ARCHITECT_STAGE_BACKOFF_SECONDS", 0.0)
    provider = FakeProvider(
        fake_model_config,
        responses=[
            RuntimeError("model request failed after 7 attempts: HTTP 503"),
            ModelResponse(content=PROFILE_JSON),  # analyze, take 2
            ModelResponse(content=PLAN_JSON),  # decompose
            ModelResponse(content="TASK_COMPLETE: done"),
            ModelResponse(content=VERDICT_JSON),
        ],
    )
    pipeline, store = _build(demo_repo, config, provider)
    outcome = await pipeline.run("app.greet() should return 'hello' when called")
    assert outcome.success, outcome.outcome_line
    events = [
        json.loads(line)
        for line in (outcome.evidence_path / "trace.jsonl").read_text().splitlines()
    ]
    retries = [e for e in events if e.get("event") == "architect.retry"]
    assert len(retries) == 1
    assert "HTTP 503" in retries[0]["error"]
    store.close()


async def test_persistent_outage_fails_honestly_after_stage_retries(
    demo_repo: Path, config: HarnessConfig, fake_model_config, monkeypatch
) -> None:
    monkeypatch.setattr(HarnessPipeline, "ARCHITECT_STAGE_BACKOFF_SECONDS", 0.0)
    provider = FakeProvider(
        fake_model_config,
        responses=[RuntimeError("model request failed after 7 attempts: HTTP 503")] * 3,
    )
    pipeline, store = _build(demo_repo, config, provider)
    with pytest.raises(RuntimeError, match="model request failed"):
        await pipeline.run("fix the greeting")
    store.close()


async def test_auth_error_is_never_stage_retried(
    demo_repo: Path, config: HarnessConfig, fake_model_config, monkeypatch
) -> None:
    from harness.infrastructure.model_providers import ModelAuthError

    monkeypatch.setattr(HarnessPipeline, "ARCHITECT_STAGE_BACKOFF_SECONDS", 0.0)
    provider = FakeProvider(
        fake_model_config,
        responses=[ModelAuthError("provider rejected credentials (HTTP 401)")],
    )
    pipeline, store = _build(demo_repo, config, provider)
    with pytest.raises(ModelAuthError):
        await pipeline.run("fix the greeting")
    # Exactly one model call: no retries burned on a credentials failure.
    assert len(provider.calls) == 1
    store.close()
