"""Wall-clock rail: `run.wall_clock_seconds` must actually stop the run.

Live finding (run ec64a196): the config knob existed but nothing enforced
it - a rate-limited provider can starve a run for hours without spending
the token budget, so the run just kept going. The governor now carries a
deadline, every model call and stage boundary checks it, and the pipeline
finalizes honestly: `budget.exhausted` event for the cockpits, run.end with
success=false and the stop reason, verification skipped instead of limping.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from harness.config import HarnessConfig
from harness.engine.budget import BudgetGovernor, RunDeadlineExceeded
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


async def test_deadline_mid_run_finalizes_honestly(
    demo_repo: Path, config: HarnessConfig, fake_model_config, monkeypatch
) -> None:
    """Deadline trips during the specialist phase -> honest, complete finalize."""
    provider = FakeProvider(
        fake_model_config,
        responses=[ModelResponse(content=PROFILE_JSON), ModelResponse(content=PLAN_JSON)],
    )
    pipeline, store = _build(demo_repo, config, provider)

    expired: dict[str, bool] = {"flag": False}
    real_check = BudgetGovernor.check

    def flip_then_check(self: BudgetGovernor) -> None:
        if expired["flag"]:
            raise RunDeadlineExceeded("wall clock exceeded: 999s >= 1800s limit")
        real_check(self)

    monkeypatch.setattr(BudgetGovernor, "check", flip_then_check)

    async def run_batch_that_expires(self: Any, *args: Any, **kwargs: Any) -> list:
        expired["flag"] = True
        raise RunDeadlineExceeded("wall clock exceeded: 999s >= 1800s limit")

    monkeypatch.setattr(HarnessPipeline, "_run_batch", run_batch_that_expires)

    outcome = await pipeline.run("fix the greeting")

    assert outcome.success is False
    assert "stopped by budget governor" in outcome.outcome_line
    assert "wall clock exceeded" in outcome.outcome_line
    # Planning only: the deadline fired before any specialist/verification call.
    assert len(provider.calls) == 2

    events = [
        json.loads(line)
        for line in (outcome.evidence_path / "trace.jsonl").read_text().splitlines()
    ]
    stops = [e for e in events if e.get("event") == "budget.exhausted"]
    assert stops, "budget.exhausted event must reach the cockpit"
    assert "wall clock exceeded" in stops[0]["reason"]
    run_end = [e for e in events if e.get("event") == "run.end"][-1]
    assert run_end["success"] is False
    assert "wall clock exceeded" in run_end["stop_reason"]
    store.close()


async def test_deadline_during_planning_is_not_stage_retried(
    demo_repo: Path, config: HarnessConfig, fake_model_config, monkeypatch
) -> None:
    """A rail stop never heals with retries - propagate on the first attempt.

    BudgetExhausted subclasses RuntimeError, so without the explicit
    exclusion the architect stage would burn all 3 transport retries on a
    wall-clock stop (and sleep between them) before re-raising.
    """
    monkeypatch.setattr(HarnessPipeline, "ARCHITECT_STAGE_BACKOFF_SECONDS", 0.0)
    provider = FakeProvider(fake_model_config, responses=[ModelResponse(content=PROFILE_JSON)] * 5)

    def deadline_already_passed(self: BudgetGovernor) -> None:
        raise RunDeadlineExceeded("wall clock exceeded: 1s >= 1s limit")

    monkeypatch.setattr(BudgetGovernor, "check", deadline_already_passed)

    pipeline, store = _build(demo_repo, config, provider)
    architect = pipeline._agents["arch-1"]
    analyze_entries = {"count": 0}
    real_analyze = architect.analyze_repository

    async def counting_analyze(*args: Any, **kwargs: Any) -> Any:
        analyze_entries["count"] += 1
        return await real_analyze(*args, **kwargs)

    monkeypatch.setattr(architect, "analyze_repository", counting_analyze)

    with pytest.raises(RunDeadlineExceeded, match="wall clock exceeded"):
        await pipeline.run("fix the greeting")

    assert analyze_entries["count"] == 1, "rail stops must not burn stage retries"
    assert len(provider.calls) == 0, "no model call should be dispatched past the deadline"
    store.close()
