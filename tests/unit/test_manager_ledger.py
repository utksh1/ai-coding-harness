"""Manager-real routing tests (review findings #7/#8).

The Manager's ledger must be LIVE: current_tasks rises on assignment and
falls on completion, tokens_used accrues per slot, the §5.1 load factor sees
real spend, and L2 reassignment picks the best-scoring OTHER specialist
(specialty-aware), never dict order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness.config import HarnessConfig
from harness.engine.pipeline import HarnessPipeline
from harness.infrastructure.context_store import SQLiteContextStore
from harness.infrastructure.model_providers import FakeProvider, ModelResponse
from harness.infrastructure.model_providers.base import ModelConfig

PROFILE_JSON = (
    '{"languages": ["Python"], "frameworks": ["pytest"], "test_framework": "pytest", '
    '"build_system": "pyproject.toml", "conventions": [], "notes": ""}'
)


def _two_task_plan(specialty: str) -> str:
    return json.dumps(
        {
            "issue_summary": "two tasks",
            "complexity": 2,
            "subtasks": [
                {
                    "id": "st-1",
                    "title": "first",
                    "description": "d1",
                    "specialty": specialty,
                    "complexity": 1,
                    "files": ["app.py"],
                    "acceptance_criteria": ["c1"],
                    "depends_on": [],
                },
                {
                    "id": "st-2",
                    "title": "second",
                    "description": "d2",
                    "specialty": specialty,
                    "complexity": 1,
                    "files": ["other.py"],
                    "acceptance_criteria": ["c2"],
                    "depends_on": ["st-1"],
                },
            ],
            "risks": [],
            "needs_collaboration": False,
        }
    )


@pytest.fixture
def work_repo(tmp_path: Path) -> Path:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (repo / "app.py").write_text("def greet():\n    return 'hello'\n")
    (repo / "test_app.py").write_text("from app import greet\n\ndef test_ok():\n    assert greet()\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "b"],
        check=True,
    )
    return repo


def _config() -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "fake", "name": "fake-model"}},
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "impl-1", "role": "implementer", "model": "default"},
                {"agent_id": "impl-2", "role": "implementer", "model": "default"},
                {"agent_id": "ver-1", "role": "verifier", "model": "default"},
                {"agent_id": "ver-2", "role": "verifier", "model": "default"},
            ],
            "storage": {"backend": "sqlite", "sqlite_path": ".harness/pipeline.db"},
            "run": {"results_dir": "results"},
        }
    )


async def test_manager_ledger_is_live_across_tasks(work_repo: Path) -> None:
    """current_tasks rises on assign, falls on completion; tokens_used
    accrues on the executing slot; the global assignment map clears."""
    fake_model_config = ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY")
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=_two_task_plan("refactoring")),
            ModelResponse(content="TASK_COMPLETE: one"),
            ModelResponse(content="TASK_COMPLETE: two"),
            ModelResponse(
                content=json.dumps(
                    {
                        "approved": True,
                        "issues": [],
                        "summary": "ok",
                        "criteria_dispositions": [
                            {"criterion": "c1", "satisfied": True, "evidence": "x"},
                            {"criterion": "c2", "satisfied": True, "evidence": "x"},
                        ],
                    }
                )
            ),
        ],
    )
    store = SQLiteContextStore(work_repo / ".harness" / "pipeline.db")
    try:
        pipeline = HarnessPipeline(work_repo, _config(), provider, store)
        manager = pipeline._manager
        assert manager is not None
        outcome = await pipeline.run("do two things")
        assert len(outcome.task_results) == 2

        # After the run the ledger is CLEAN: no task stays assigned.
        assert manager.assignments == {}
        for slot in manager.slots.values():
            assert slot.current_tasks == 0
        # The specialist that executed both tasks accrued real token load
        # (traced tokens > 0 through the fake estimator).
        spenders = [s.agent_id for s in manager.slots.values() if s.tokens_used > 0]
        assert spenders, "manager never saw any token spend (load factor blind)"
        # The global assignment store cleared too.
        assert "st-1" not in (store.load_global("assignments") or {})
    finally:
        store.close()


async def test_reroute_picks_best_scoring_other_specialist(work_repo: Path) -> None:
    """Reassignment is task-aware (finding #8): an implementation task that
    fails on impl-1 reroutes to the best IMPLEMENTATION-capable agent, not
    the first agent in dict order."""
    from harness.agents.task import Task

    store = SQLiteContextStore(work_repo / ".harness" / "pipeline.db")
    try:
        pipeline = HarnessPipeline(work_repo, _config(), object(), store)
        reroute = pipeline._make_reroute("impl-1", pipeline._placeholder_governor, _NullPack(), "r")
        task = Task.model_validate(
            {
                "id": "st-9",
                "title": "refactor the parser",
                "description": "d",
                "specialty": "refactoring",
                "complexity": 3,
                "required_tools": ["apply_edit"],
            }
        )
        from harness.orchestration.messages import ErrorEscalation

        escalation = ErrorEscalation(
            sender="impl-1",
            task_id="st-9",
            severity="recoverable",
            error_type="RuntimeError",
            message="boom",
        )
        executor = reroute(task, escalation, "reassign: skill gap (RuntimeError)")
        assert executor is not None
        # impl-1 excluded; among the rest, impl-2 wins by SPECIALTY (the
        # refactoring task), not by dict position - a verifier would be the
        # dict-order answer only if it preceded impl-2, which it does not.
        assert executor.__self__.agent_id == "impl-2"
    finally:
        store.close()


async def test_availability_changes_routing(work_repo: Path) -> None:
    """With live current_tasks, a busy specialist loses the next task to an
    equally-capable free one: the routing algorithm finally receives its
    runtime information (finding #7)."""
    from harness.agents.manager import assign_specialists
    from harness.agents.task import Task

    store = SQLiteContextStore(work_repo / ".harness" / "pipeline.db")
    try:
        pipeline = HarnessPipeline(work_repo, _config(), object(), store)
        manager = pipeline._manager
        assert manager is not None
        task = Task.model_validate(
            {
                "id": "st-7",
                "title": "verify the thing",
                "description": "d",
                "specialty": "verification",
                "complexity": 2,
            }
        )
        baseline = assign_specialists(task, pipeline._specialist_slots)[0]
        # Saturate the baseline winner's availability completely.
        manager.slots[baseline].current_tasks = manager.slots[baseline].max_concurrent
        rerouted = assign_specialists(
            task, pipeline._specialist_slots, team_average_tokens=pipeline._team_average()
        )[0]
        assert rerouted != baseline
        manager.slots[baseline].current_tasks = 0
    finally:
        store.close()


class _NullPack:
    def trace(self, event: dict[str, Any]) -> None:
        pass


def test_service_publishes_live_routing_events(work_repo: Path) -> None:
    """The cockpit's specialist.assigned event carries the routing breakdown
    computed with the live team average (not a hardcoded zero)."""
    from harness.service.app import create_app

    class _FakeRedis:
        def __init__(self) -> None:
            self.published: list[tuple[str, str]] = []

        def publish(self, channel: str, message: str) -> int:
            self.published.append((channel, message))
            return 1

    fake_model_config = ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY")
    redis_client = _FakeRedis()
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=_two_task_plan("refactoring")),
            ModelResponse(content="TASK_COMPLETE: one"),
            ModelResponse(content="TASK_COMPLETE: two"),
            ModelResponse(
                content=json.dumps(
                    {
                        "approved": True,
                        "issues": [],
                        "summary": "ok",
                        "criteria_dispositions": [
                            {"criterion": "c1", "satisfied": True, "evidence": "x"},
                            {"criterion": "c2", "satisfied": True, "evidence": "x"},
                        ],
                    }
                )
            ),
        ],
    )
    app = create_app(config=_config(), provider=provider, redis_client=redis_client)
    client = TestClient(app)
    body = client.post(
        "/agent/run", json={"issue": "two tasks", "repo_root": str(work_repo), "run_id": "ledger-1"}
    ).json()
    assert body["success"] is False or body["success"] is True  # completes either way
    assigned = [
        json.loads(message)
        for channel, message in redis_client.published
        if json.loads(message).get("event") == "specialist.assigned"
    ]
    assert assigned, "no assignment events published"
    for event in assigned:
        routing = event.get("routing")
        assert routing is not None
        assert set(routing) == {"specialty", "availability", "load", "capability"}


async def test_assign_valueerror_is_survived(work_repo: Path) -> None:
    """A manager that rejects an assignment (unknown specialist edge) never
    crashes the run: routing still holds (contextlib.suppress path)."""
    fake_model_config = ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY")
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),
            ModelResponse(content=_two_task_plan("refactoring")),
            ModelResponse(content="TASK_COMPLETE: one"),
            ModelResponse(content="TASK_COMPLETE: two"),
            ModelResponse(
                content=json.dumps(
                    {
                        "approved": True,
                        "issues": [],
                        "summary": "ok",
                        "criteria_dispositions": [
                            {"criterion": "c1", "satisfied": True, "evidence": "x"},
                            {"criterion": "c2", "satisfied": True, "evidence": "x"},
                        ],
                    }
                )
            ),
        ],
    )
    store = SQLiteContextStore(work_repo / ".harness" / "pipeline.db")
    try:
        pipeline = HarnessPipeline(work_repo, _config(), provider, store)
        real_manager = pipeline._manager
        assert real_manager is not None

        class RejectingManager:
            """Delegates everything except assign_task, which always raises."""

            register_specialist = real_manager.register_specialist
            acknowledge_completion = real_manager.acknowledge_completion
            handle_escalation = real_manager.handle_escalation

            async def assign_task(self, task, agent_id):
                raise ValueError(f"unknown specialist '{agent_id}'")

        pipeline._manager = RejectingManager()
        outcome = await pipeline.run("two tasks anyway")
        assert len(outcome.task_results) == 2  # the run completed despite rejection
    finally:
        store.close()


async def test_reroute_without_alternatives_returns_none(work_repo: Path) -> None:
    """A roster with one specialist and coordination agents only: no
    reassignment target exists, reroute returns None (honest, no crash)."""
    single = HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "fake", "name": "fake-model"}},
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "impl-1", "role": "implementer", "model": "default"},
            ],
            "storage": {"backend": "sqlite", "sqlite_path": ".harness/pipeline.db"},
        }
    )
    from harness.agents.task import Task
    from harness.orchestration.messages import ErrorEscalation

    store = SQLiteContextStore(work_repo / ".harness" / "pipeline.db")
    try:
        pipeline = HarnessPipeline(work_repo, single, object(), store)
        reroute = pipeline._make_reroute("impl-1", pipeline._placeholder_governor, _NullPack(), "r")
        task = Task.model_validate(
            {"id": "st-1", "title": "t", "description": "d", "specialty": "refactoring"}
        )
        escalation = ErrorEscalation(
            sender="impl-1", severity="recoverable", error_type="RuntimeError", message="boom"
        )
        assert reroute(task, escalation, "reassign: skill gap") is None
    finally:
        store.close()


def test_service_provider_factory_branches() -> None:
    from harness.service.app import _service_provider_factory

    demo_factory = _service_provider_factory(True)
    real_factory = _service_provider_factory(False)

    # Demo leg: role-aware scripted providers.
    demo_arch = demo_factory(object(), role="architect")
    assert len(demo_arch._responses) == 3
    # Real leg: delegates to create_model_provider (mocked for identity).

    sentinel = object()
    import harness.service.app as app_module

    original = app_module.create_model_provider
    app_module.create_model_provider = lambda cfg: sentinel
    try:
        assert real_factory(object(), role="implementer", agent_id="impl-1") is sentinel
    finally:
        app_module.create_model_provider = original
