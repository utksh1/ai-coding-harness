"""Milestone 4.4 pipeline-level repair flows: genuine re-route and the
'add collaborators' spawn, exercised through HarnessPipeline offline.

Audit §8: the Manager's L2 decisions must alter execution. With a KeyError
failure the escalation categorizes as skill_gap -> 'reassign' swaps to the
other specialist; with repeated unknown failures it categorizes as
complex_task -> 'add collaborators' spawns a bounded collaborator.
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
    '"conventions": [], "notes": "repair-flow repo"}'
)
PLAN_JSON = (
    '{"issue_summary": "greeting missing", "complexity": 3, "subtasks": [{'
    '"id": "st-1", "title": "add greeting", '
    '"description": "app.greet() should return hello", '
    '"specialty": "verification", "complexity": 2, "files": ["app.py"], '
    '"acceptance_criteria": ["greet returns hello"], "depends_on": []}], '
    '"risks": [], "needs_collaboration": false}'
)
VERDICT_JSON = '{"approved": true, "issues": [], "summary": "greet works", "criteria_dispositions": [{"criterion": "greet returns hello", "satisfied": true, "evidence": "diff shows the fix"}]}'


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


def _config(extra_specialist: bool) -> HarnessConfig:
    agents = [
        {"agent_id": "arch-1", "role": "architect", "model": "default"},
        {"agent_id": "mgr-1", "role": "manager", "model": "default"},
        {"agent_id": "ver-1", "role": "verifier", "model": "default"},
    ]
    if extra_specialist:
        agents.append({"agent_id": "impl-2", "role": "implementer", "model": "default"})
    return HarnessConfig.model_validate(
        {
            "models": {
                "default": {"provider": "fake", "name": "fake-model", "api_key_env": "AI_API_KEY"}
            },
            "agents": agents,
            "storage": {"backend": "sqlite", "sqlite_path": ".harness/pipeline.db"},
            "run": {"results_dir": "results"},
        }
    )


def _trace_events(outcome) -> list[dict]:
    return [
        json.loads(line)
        for line in (outcome.evidence_path / "trace.jsonl").read_text().splitlines()
    ]


async def test_repeated_failures_spawn_collaborator(demo_repo: Path, fake_model_config) -> None:
    """complex_task L2 -> 'add collaborators' spawns a real extra specialist."""
    boom = RuntimeError("boom")
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=PLAN_JSON),
            boom,  # L1 attempt 1
            boom,  # L1 attempt 2
            boom,  # L1 attempt 3 -> attempt>=2 categorizes complex_task
            ModelResponse(content="TASK_COMPLETE: collaborator fixed it"),
            ModelResponse(content=VERDICT_JSON),
        ],
    )
    store = SQLiteContextStore(demo_repo / ".harness" / "pipeline.db")
    pipeline = HarnessPipeline(demo_repo, _config(extra_specialist=False), provider, store)
    outcome = await pipeline.run("app.greet() should return 'hello'")

    assert outcome.success, outcome.outcome_line
    events = _trace_events(outcome)
    kinds = [event["event"] for event in events]
    assert "specialist.collaborator_added" in kinds
    assert "recovery.l2_reroute" in kinds
    assert any(event["event"] == "recovery.l1_retry" for event in events)
    store.close()


async def test_skill_gap_failure_reassigns_to_other_specialist(
    demo_repo: Path, fake_model_config
) -> None:
    """skill_gap L2 -> 'reassign' genuinely swaps to a different specialist."""
    missing = KeyError("spec")
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=PLAN_JSON),
            missing,  # ver-1 attempt 1 (KeyError -> skill_gap at L2)
            missing,
            missing,
            ModelResponse(content="TASK_COMPLETE: reassigned specialist fixed it"),
            ModelResponse(content=VERDICT_JSON),
        ],
    )
    store = SQLiteContextStore(demo_repo / ".harness" / "pipeline2.db")
    pipeline = HarnessPipeline(demo_repo, _config(extra_specialist=True), provider, store)
    outcome = await pipeline.run("app.greet() should return 'hello'")

    assert outcome.success, outcome.outcome_line
    kinds = [event["event"] for event in _trace_events(outcome)]
    assert "recovery.l2_reroute" in kinds
    assert "specialist.collaborator_added" not in kinds
    store.close()


async def test_specialist_success_first_try_has_no_recovery_events(
    demo_repo: Path, fake_model_config
) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=PLAN_JSON),
            ModelResponse(content="TASK_COMPLETE: done"),
            ModelResponse(content=VERDICT_JSON),
        ],
    )
    store = SQLiteContextStore(demo_repo / ".harness" / "pipeline3.db")
    pipeline = HarnessPipeline(demo_repo, _config(extra_specialist=False), provider, store)
    outcome = await pipeline.run("greet works")
    assert outcome.success
    kinds = [event["event"] for event in _trace_events(outcome)]
    assert not any(kind.startswith("recovery.") for kind in kinds)
    store.close()


async def test_collaborator_cap_bounds_spawning(demo_repo: Path, fake_model_config) -> None:
    """At most 2 collaborators per pipeline; a capped re-route degrades to
    guidance-only retries and the run still ends honestly (audit §8)."""
    from harness.config import BudgetConfig
    from harness.engine.budget import BudgetGovernor
    from harness.engine.evidence import EvidencePack

    reframe_json = '{"title": "st-1", "description": "simpler retry", "acceptance_criteria": []}'
    boom = RuntimeError("boom")
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=PLAN_JSON),
            boom,
            boom,
            boom,  # L1 exhausted -> complex_task
            boom,  # L2 attempt 1 (collaborator cap already reached -> guidance only)
            boom,  # L2 attempt 2
            ModelResponse(content=reframe_json),  # L3 reframe
            boom,  # L3 attempt
            ModelResponse(content=VERDICT_JSON),
        ],
    )
    store = SQLiteContextStore(demo_repo / ".harness" / "pipeline4.db")
    pipeline = HarnessPipeline(demo_repo, _config(extra_specialist=False), provider, store)

    # Pre-arrange: the collaborator cap is already spent.
    governor = BudgetGovernor(store, BudgetConfig(), "pre-run")
    pack = EvidencePack(demo_repo / "results", "pre-run")
    assert pipeline._add_collaborator("ver-1", governor, pack, "pre-run") is not None
    assert pipeline._add_collaborator("ver-1", governor, pack, "pre-run") is not None
    assert pipeline._add_collaborator("ver-1", governor, pack, "pre-run") is None

    outcome = await pipeline.run("greet works")
    assert not outcome.success
    assert pipeline._collaborators_added == 2
    kinds = [event["event"] for event in _trace_events(outcome)]
    assert "specialist.collaborator_added" not in kinds
    store.close()


async def test_tool_limitation_guidance_falls_through_reroute(
    demo_repo: Path, fake_model_config
) -> None:
    """'tool guidance' L2 decisions keep the same specialist (reroute -> None)
    but the guidance still reaches the task (covered by 4.3's prompt wiring)."""
    bad_tool = RuntimeError("unknown tool 'zap'")
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=PLAN_JSON),
            bad_tool,
            bad_tool,
            bad_tool,  # L1 exhausted; message contains "unknown tool" -> tool_limitation
            ModelResponse(content="TASK_COMPLETE: fixed with the suggested tool"),
            ModelResponse(content=VERDICT_JSON),
        ],
    )
    store = SQLiteContextStore(demo_repo / ".harness" / "pipeline5.db")
    pipeline = HarnessPipeline(demo_repo, _config(extra_specialist=False), provider, store)
    outcome = await pipeline.run("greet works")

    assert outcome.success, outcome.outcome_line
    kinds = [event["event"] for event in _trace_events(outcome)]
    assert "recovery.l2_guidance" in kinds
    assert "recovery.l2_reroute" not in kinds
    store.close()
