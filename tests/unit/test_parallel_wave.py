"""Parallel wave execution + planned collaboration tests (findings #9/#11).

File-disjoint batch members run CONCURRENTLY on distinct agents with
state-independent providers; injected shared providers (scripted tests)
degrade to serial honestly. High-complexity top-3 routing triggers a real
runner-up review-and-fix pass.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from harness.config import HarnessConfig
from harness.engine.pipeline import HarnessPipeline
from harness.infrastructure.context_store import MemoryContextStore
from harness.infrastructure.model_providers import FakeProvider, ModelResponse
from harness.infrastructure.model_providers.base import ModelConfig

PROFILE_JSON = (
    '{"languages": ["Python"], "frameworks": ["pytest"], "test_framework": "pytest", '
    '"build_system": "pyproject.toml", "conventions": [], "notes": ""}'
)

fake_model_config = ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY")


@pytest.fixture
def work_repo(tmp_path: Path) -> Path:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (repo / "app.py").write_text("def a():\n    return 1\n")
    (repo / "other.py").write_text("def b():\n    return 2\n")
    (repo / "test_all.py").write_text("def test_ok():\n    assert True\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "b"],
        check=True,
    )
    return repo


def _parallel_plan() -> str:
    """Two INDEPENDENT file-disjoint subtasks (no depends_on): one batch."""
    return json.dumps(
        {
            "issue_summary": "two parallel edits",
            "complexity": 2,
            "subtasks": [
                {
                    "id": "st-1",
                    "title": "edit app",
                    "description": "d1",
                    "specialty": "refactoring",
                    "complexity": 1,
                    "files": ["app.py"],
                    "acceptance_criteria": ["c1"],
                    "depends_on": [],
                },
                {
                    "id": "st-2",
                    "title": "edit other",
                    "description": "d2",
                    "specialty": "refactoring",
                    "complexity": 1,
                    "files": ["other.py"],
                    "acceptance_criteria": ["c2"],
                    "depends_on": [],
                },
            ],
            "risks": [],
            "needs_collaboration": False,
        }
    )


def _verdict(criteria: list[str]) -> str:
    return json.dumps(
        {
            "approved": True,
            "issues": [],
            "summary": "ok",
            "criteria_dispositions": [
                {"criterion": c, "satisfied": True, "evidence": "x"} for c in criteria
            ],
        }
    )


def _config() -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "fake", "name": "fake-model"}},
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "impl-1", "role": "implementer", "model": "default"},
                {"agent_id": "impl-2", "role": "implementer", "model": "default"},
            ],
            "storage": {"backend": "memory"},
            "run": {"results_dir": "results"},
        }
    )


class _SlowProvider(FakeProvider):
    """Scripted provider that PARKS briefly per call so parallel execution
    observably overlaps (wall-clock proof, not just event ordering)."""

    def __init__(self, cfg: Any, responses: list[Any], delay: float) -> None:
        super().__init__(cfg, responses=responses, loop=True)
        self._delay = delay

    async def generate(self, messages: Any, tools: Any = None, **overrides: Any) -> Any:
        await asyncio.sleep(self._delay)
        return await super().generate(messages, tools, **overrides)


async def test_parallel_wave_overlaps_execution(work_repo: Path) -> None:
    """Two file-disjoint subtasks on distinct agents with per-agent provider
    instances: the wave runs them CONCURRENTLY (wall-clock < sum of parts)."""

    def factory(model_cfg: Any, role: str | None = None, agent_id: str | None = None) -> Any:
        return _SlowProvider(
            model_cfg,
            responses=[ModelResponse(content="TASK_COMPLETE: edited")],
            delay=0.4,
        )

    class _FixedArchitect:
        """Bypasses the LLM architect: fixed plan, scripted verdict."""

        agent_id = "arch-1"

        async def review(self, diff: str, plan: Any, evidence: str = "") -> Any:
            from harness.agents.architect import ReviewVerdict

            return ReviewVerdict(
                approved=True,
                issues=[],
                summary="ok",
                criteria_dispositions=[
                    __import__(
                        "harness.agents.architect", fromlist=["CriterionDisposition"]
                    ).CriterionDisposition(criterion=c, satisfied=True, evidence="x")
                    for c in ("c1", "c2")
                ],
            )

    pipeline = HarnessPipeline(
        work_repo, _config(), store=MemoryContextStore(), provider_factory=factory
    )
    # Fix the plan and architect directly: this test is about the WAVE, not
    # the architect stage.
    from harness.agents.architect import Plan

    fixed_plan = Plan.model_validate(json.loads(_parallel_plan()))
    pipeline._architect = _FixedArchitect()

    pack_trace: list[dict[str, Any]] = []

    def tracer(event: dict[str, Any]) -> None:
        pack_trace.append(event)

    from harness.engine.evidence import EvidencePack

    pack = EvidencePack(work_repo / "results", "wave-1", event_sink=tracer)

    # Directly drive _run_batch with the fixed plan: wave of 2.
    batch = fixed_plan.subtasks
    started = time.monotonic()
    from harness.config import BudgetConfig
    from harness.engine.budget import BudgetGovernor

    governor = BudgetGovernor(pipeline._store, BudgetConfig(total_tokens=1_000_000), "wave-1")
    results = await pipeline._run_batch(
        batch,
        governor,
        _StubMetrics(),
        pack,
        "wave-1",
        _FixedArchitect(),  # type: ignore[arg-type]
    )
    elapsed = time.monotonic() - started
    assert len(results) == 2
    # Two 0.4s+ specialist loops: serial >= 0.8s, parallel < 0.75s (margin).
    assert elapsed < 0.75, f"wave ran serially ({elapsed:.2f}s)"
    # Both specialists got their own agent (distinct agents in the wave).
    assigned = [e["agent"] for e in pack_trace if e.get("event") == "specialist.assigned"]
    assert len(set(assigned)) == 2


class _StubMetrics:
    def record_result(self, result: Any, agent_id: str = "") -> None:
        return None

    def report(self) -> dict[str, Any]:
        return {}

    def stage_started(self, name: str) -> None:
        return None

    def stage_finished(self, name: str) -> None:
        return None


async def test_shared_injected_provider_degrades_to_serial(work_repo: Path) -> None:
    """A shared injected provider (scripted test mode) never races: the wave
    falls back to serial execution (honest degradation, finding #11)."""
    provider = FakeProvider(
        fake_model_config,
        responses=[ModelResponse(content="TASK_COMPLETE: edited")],
        loop=True,
    )
    pipeline = HarnessPipeline(work_repo, _config(), provider, MemoryContextStore())
    from harness.agents.architect import Plan
    from harness.engine.evidence import EvidencePack

    fixed_plan = Plan.model_validate(json.loads(_parallel_plan()))

    class _FixedArchitect:
        agent_id = "arch-1"

    # The wave contains 2 members on distinct agents, but the shared provider
    # makes it unsafe -> the SERIAL path executes (honest degradation).
    wave, _overflow = pipeline._plan_wave(fixed_plan.subtasks)
    assert len(wave) == 2
    assert pipeline._wave_parallel_safe(wave) is False

    from harness.config import BudgetConfig
    from harness.engine.budget import BudgetGovernor

    pack = EvidencePack(work_repo / "results", "serial-1")
    governor = BudgetGovernor(
        MemoryContextStore(), BudgetConfig(total_tokens=1_000_000), "serial-1"
    )
    results = await pipeline._run_batch(
        fixed_plan.subtasks,
        governor,
        _StubMetrics(),
        pack,
        "serial-1",
        _FixedArchitect(),  # type: ignore[arg-type]
    )
    assert len(results) == 2  # serial wave of 2 completes
    assert all(r.success for r in results)


async def test_planned_collaboration_runs_runner_up_review(work_repo: Path) -> None:
    """High-complexity top-3 routing is REAL: runner-up agents execute a
    review-and-fix pass over the primary's work (advisory, evented)."""

    def factory(model_cfg: Any, role: str | None = None, agent_id: str | None = None) -> Any:
        return FakeProvider(
            model_cfg, responses=[ModelResponse(content="TASK_COMPLETE: x")], loop=True
        )

    # Custom plan: complexity 9 (above threshold), single subtask.
    plan_json = json.dumps(
        {
            "issue_summary": "hard task",
            "complexity": 9,
            "subtasks": [
                {
                    "id": "st-hard",
                    "title": "hard",
                    "description": "do the hard thing",
                    "specialty": "refactoring",
                    "complexity": 9,
                    "files": ["app.py"],
                    "acceptance_criteria": ["ch"],
                    "depends_on": [],
                }
            ],
            "risks": [],
            "needs_collaboration": False,
        }
    )
    from harness.agents.architect import Plan
    from harness.engine.evidence import EvidencePack

    pipeline = HarnessPipeline(
        work_repo, _config(), store=MemoryContextStore(), provider_factory=factory
    )
    fixed_plan = Plan.model_validate(json.loads(plan_json))
    events: list[dict[str, Any]] = []
    pack = EvidencePack(work_repo / "results", "collab-1", event_sink=events.append)

    class _FixedArchitect:
        agent_id = "arch-1"

    from harness.config import BudgetConfig
    from harness.engine.budget import BudgetGovernor

    governor = BudgetGovernor(pipeline._store, BudgetConfig(total_tokens=1_000_000), "collab-1")
    results = await pipeline._run_batch(
        fixed_plan.subtasks,
        governor,
        _StubMetrics(),
        pack,
        "collab-1",
        _FixedArchitect(),  # type: ignore[arg-type]
    )
    assert results and results[0].success
    collab_events = [e for e in events if e.get("event") == "specialist.collaborator_added"]
    assert collab_events, "planned collaboration never fired"
    assert collab_events[0]["reason"] == "planned"
    advisory = [e for e in events if e.get("event") == "specialist.result" and e.get("advisory")]
    assert advisory, "collaborator's review pass left no event"


def test_plan_wave_claims_distinct_agents(work_repo: Path) -> None:
    """Wave planning never gives two batch members the same agent."""
    pipeline = HarnessPipeline(work_repo, _config(), object(), MemoryContextStore())
    from harness.agents.architect import Plan

    fixed_plan = Plan.model_validate(json.loads(_parallel_plan()))
    wave, overflow = pipeline._plan_wave(fixed_plan.subtasks)
    agents = [agent_id for _, agent_id in wave]
    assert len(agents) == 2
    assert len(set(agents)) == 2
    assert overflow == []


async def test_gather_wave_propagates_first_real_exception(work_repo: Path) -> None:
    """A wave member blowing the budget mid-flight propagates the exception
    after the wave completes - the caller's BudgetExhausted handling stays
    exact (never swallowed by gather)."""
    from harness.config import BudgetConfig
    from harness.engine.budget import BudgetExhausted, BudgetGovernor
    from harness.engine.evidence import EvidencePack

    class _ExplodingProvider(FakeProvider):
        def __init__(self, cfg: Any, explode: bool) -> None:
            super().__init__(cfg, responses=[ModelResponse(content="TASK_COMPLETE: x")], loop=True)
            self._explode = explode

        async def generate(self, messages: Any, tools: Any = None, **overrides: Any) -> Any:
            if self._explode:
                raise BudgetExhausted("token budget exhausted (test)")
            return await super().generate(messages, tools, **overrides)

    def factory(model_cfg: Any, role: str | None = None, agent_id: str | None = None) -> Any:
        return FakeProvider(
            model_cfg, responses=[ModelResponse(content="TASK_COMPLETE: x")], loop=True
        )

    pipeline = HarnessPipeline(
        work_repo, _config(), store=MemoryContextStore(), provider_factory=factory
    )
    # Inject an infrastructure-level raiser OUTSIDE the agent loop: the
    # manager's completion hook raising BudgetExhausted (the ladder absorbs
    # task-level exceptions by design; the wave guard covers raisers between
    # the executor and the ledger - store/IO class failures).
    real_manager = pipeline._manager
    assert real_manager is not None

    class ExplodingLedgerManager:
        register_specialist = real_manager.register_specialist
        assign_task = real_manager.assign_task
        handle_escalation = real_manager.handle_escalation

        async def acknowledge_completion(self, *args: Any, **kwargs: Any) -> None:
            raise BudgetExhausted("budget died during ledger update (test)")

    pipeline._manager = ExplodingLedgerManager()
    from harness.agents.architect import Plan

    fixed_plan = Plan.model_validate(json.loads(_parallel_plan()))
    pack = EvidencePack(work_repo / "results", "boom-1")
    governor = BudgetGovernor(MemoryContextStore(), BudgetConfig(total_tokens=1_000_000), "boom-1")

    class _FixedArchitect:
        agent_id = "arch-1"

    with pytest.raises(BudgetExhausted):
        await pipeline._run_batch(
            fixed_plan.subtasks,
            governor,
            _StubMetrics(),
            pack,
            "boom-1",
            _FixedArchitect(),  # type: ignore[arg-type]
        )


def test_wave_parallel_safe_rejects_unknown_agent(work_repo: Path) -> None:
    pipeline = HarnessPipeline(work_repo, _config(), object(), MemoryContextStore())
    assert pipeline._wave_parallel_safe([(_subtask(), "ghost-agent")]) is False


def test_planned_collaboration_skips_unknown_runner_up(work_repo: Path) -> None:
    """A runner-up id with no agent is skipped honestly (no crash)."""
    from harness.engine.evidence import EvidencePack

    pipeline = HarnessPipeline(work_repo, _config(), object(), MemoryContextStore())
    pack = EvidencePack(work_repo / "results", "skip-1")
    task = _subtask().to_task()

    async def go() -> None:
        await pipeline._planned_collaboration(
            task,
            ["ghost-1", "ghost-2"],
            _Primary(),
            _NullGovernor(),
            pack,
            "skip-1",
            _StubMetrics(),
        )

    import asyncio

    asyncio.run(go())


class _Primary:
    agent_id = "impl-1"


class _NullGovernor:
    def check(self) -> None:
        return None


def _subtask() -> Any:
    from harness.agents.architect import SubTask

    return SubTask(
        id="st-1",
        title="t",
        description="d",
        specialty="refactoring",
        complexity=1,
        files=["app.py"],
    )
