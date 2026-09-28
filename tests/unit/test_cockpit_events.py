"""Cockpit event contract tests (docs/cockpit-events.md, platform P4).

Two layers:

- unit: LLMAgent emission points (agent.step / agent.tool / agent.usage),
  the digest renderers, tracer isolation, routing_breakdown factor math,
  RecoveryLadder run_id/agent injection + executor attribution;
- integration: the pipeline emits the enriched v2 shapes (plan subtask
  objects, routing breakdown, batch numbers, result attribution with
  steps/tokens, collaborator role) and /api/agents carries the roster
  hierarchy fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

from harness.agents.llm_agent import (
    DEFAULT_KEEP_RECENT,
    LLMAgent,
    StoreWindow,
    _args_digest,
    _result_digest,
)
from harness.agents.manager import (
    WEIGHTS,
    SpecialistSlot,
    assignment_score,
    routing_breakdown,
)
from harness.agents.task import Task
from harness.config import BudgetConfig, HarnessConfig
from harness.engine.budget import BudgetGovernor
from harness.engine.recovery import RecoveryLadder
from harness.infrastructure.context_store import MemoryContextStore
from harness.infrastructure.model_providers import FakeProvider, ModelResponse, ToolCall
from harness.tools.base import Tool, ToolResult, ToolTier


class EchoTool(Tool):
    name, tier, description = "echo_tool", ToolTier.BASIC, "echo content back"
    parameters: ClassVar[dict] = {"type": "object", "properties": {"content": {"type": "string"}}}

    def validate_input(self, arguments: dict) -> list[str]:
        return [] if "content" in arguments else ["missing 'content'"]

    def check_permissions(self, context: dict) -> bool:
        return True

    def execute(self, content: str = "") -> ToolResult:
        return ToolResult(success=True, output=f"echo: {content}")


class BlockedTool(EchoTool):
    """Tier-2 tool invoked by a tier-1 agent: the fast-rejection path."""

    name, tier = "blocked_tool", ToolTier.DEVELOPMENT


@pytest.fixture
def store() -> MemoryContextStore:
    return MemoryContextStore()


def _agent(
    store, provider, *, role: str = "implementer", tools: list[Tool] | None = None
) -> LLMAgent:
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=100_000), "corr-events")
    agent = LLMAgent(
        agent_id="impl-1",
        model_config={"provider": "fake"},
        tools=tools or [EchoTool()],
        context_window=StoreWindow(store, "impl-1", "t-1"),
        provider=provider,
        store=store,
        governor=governor,
        role=role,
        keep_recent=DEFAULT_KEEP_RECENT,
    )
    return agent


def _task() -> Task:
    return Task(id="t-1", title="echo", description="echo the content")


# -- unit: LLMAgent emissions -------------------------------------------------


async def test_loop_emits_step_tool_and_usage_events(store, fake_model_config) -> None:
    events: list[dict] = []
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(
                content="",
                tool_calls=[ToolCall(name="echo_tool", arguments={"content": "hi"})],
                prompt_tokens=100,
                completion_tokens=10,
            ),
            ModelResponse(content="TASK_COMPLETE: done", prompt_tokens=120, completion_tokens=8),
        ],
    )
    agent = _agent(store, provider)
    agent.attach_tracer(events.append, "run-evt")
    result = await agent.execute_task(_task())

    assert result.success
    kinds = [(e["event"], e.get("phase")) for e in events]
    assert ("agent.step", "thinking") in kinds
    assert ("agent.step", "responding") in kinds

    tool_events = [e for e in events if e["event"] == "agent.tool"]
    assert len(tool_events) == 1
    tool_event = tool_events[0]
    assert tool_event["tool"] == "echo_tool"
    assert tool_event["args_digest"] == "hi"
    assert tool_event["ok"] is True
    assert tool_event["result_digest"] == "echo: hi"
    assert tool_event["task"] == "t-1"
    assert tool_event["run_id"] == "run-evt"
    assert tool_event["agent"] == "impl-1"
    assert tool_event["role"] == "implementer"
    assert isinstance(tool_event["duration_ms"], int)
    assert tool_event["step"] >= 1

    usage_events = [e for e in events if e["event"] == "agent.usage"]
    assert len(usage_events) == 2
    assert usage_events[0]["total_tokens_agent"] == 110
    assert usage_events[1]["total_tokens_agent"] == 238  # cumulative per agent
    assert usage_events[0]["task"] == "t-1"

    # step numbering matches the loop position, not a global counter
    steps = [e for e in events if e["event"] == "agent.step"]
    assert steps[0]["step"] == 1 and steps[-1]["step"] == 2
    assert steps[0]["max_steps"] == agent.max_steps

    # steps_used surfaces for the pipeline's specialist.result event
    assert agent.steps_used == 2


async def test_tracer_failure_never_breaks_the_run(store, fake_model_config) -> None:
    def exploding_tracer(event: dict) -> None:
        raise RuntimeError("cockpit down")

    provider = FakeProvider(
        fake_model_config,
        responses=[ModelResponse(content="TASK_COMPLETE: fine")],
    )
    agent = _agent(store, provider)
    agent.attach_tracer(exploding_tracer, "run-boom")
    result = await agent.execute_task(_task())
    assert result.success  # the run survived a broken cockpit


async def test_tier_rejected_tool_call_is_traced(store, fake_model_config) -> None:
    """The fast rejections (tier cap) still reach the cockpit log."""
    events: list[dict] = []
    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(
                content="",
                tool_calls=[ToolCall(name="blocked_tool", arguments={"content": "x"})],
            ),
            ModelResponse(content="TASK_COMPLETE: gave up on that tool"),
        ],
    )
    locator = _agent(store, provider, role="locator", tools=[BlockedTool()])
    locator.attach_tracer(events.append, "run-tier")
    await locator.execute_task(_task())

    tool_events = [e for e in events if e["event"] == "agent.tool"]
    assert any(e["ok"] is False and "tier" in e["result_digest"] for e in tool_events)


async def test_structured_call_emits_steps(store, fake_model_config) -> None:
    events: list[dict] = []
    provider = FakeProvider(
        fake_model_config,
        responses=[ModelResponse(content='{"description": "clearer text"}')],
    )
    agent = _agent(store, provider, role="manager")
    agent.attach_tracer(events.append, "run-struct")
    parsed = await agent.structured_call("Reply with JSON", "clarify", '{"description": str}')

    assert parsed == {"description": "clearer text"}
    thinking = [e for e in events if e["event"] == "agent.step" and e["phase"] == "thinking"]
    responding = [e for e in events if e["event"] == "agent.step" and e["phase"] == "responding"]
    assert len(thinking) == 1 and len(responding) == 1
    assert thinking[0]["step"] == 1
    # task label falls back to the window's task id
    assert thinking[0]["task"] == "t-1"


def test_attach_tracer_resets_counters_and_detaches(store, fake_model_config) -> None:
    agent = _agent(store, FakeProvider(fake_model_config, responses=[]))
    agent.attach_tracer(lambda e: None, "r1")
    agent.traced_tokens = 500
    agent.attach_tracer(None, None)
    assert agent.tracer is None
    agent.attach_tracer(lambda e: None, "r2")
    assert agent.traced_tokens == 0


# -- unit: digests ---------------------------------------------------------------


def test_args_digest_paths_commands_patterns_nodes() -> None:
    assert _args_digest("read", {"path": "src/app.py"}) == "src/app.py"
    assert _args_digest("run_cmd", {"command": ["pytest", "-q"]}) == "pytest -q"
    assert _args_digest("run_cmd", {"command": "pytest -q"}) == "pytest -q"
    assert _args_digest("search", {"pattern": "def greet", "path": "src/"}) == (
        "pattern 'def greet' in src/"
    )
    assert _args_digest("search", {"pattern": "def greet"}) == "pattern 'def greet'"
    assert _args_digest("run_tests", {"node_id": "tests/test_x.py::test_a"}) == (
        "tests/test_x.py::test_a"
    )
    assert _args_digest("odd", {"whatever": "first-value"}) == "first-value"
    assert _args_digest("odd", {}) == ""
    long_path = "x" * 500
    assert len(_args_digest("read", {"path": long_path})) == 120


def test_args_digest_content_never_rides_the_event() -> None:
    secret = "AWS_KEY=super-secret-value"
    digest = _args_digest("write_file", {"path": "cfg.py", "content": secret})
    assert secret not in digest
    assert digest == "cfg.py"


def test_result_digest_truncates_and_flattens() -> None:
    assert _result_digest(ToolResult(success=True, output="ok")) == "ok"
    assert _result_digest(ToolResult(success=False, error="boom\nat line 3")) == "boom at line 3"
    multiline = "line1\nline2\nline3\n"
    assert "\n" not in _result_digest(ToolResult(success=True, output=multiline))
    assert len(_result_digest(ToolResult(success=True, output="x" * 500))) == 120


# -- unit: routing breakdown ------------------------------------------------------


def test_routing_breakdown_sums_to_assignment_score() -> None:
    task = Task(id="t", title="t", description="d", specialty="bugfix", required_tools={"echo"})
    slot = SpecialistSlot(
        agent_id="impl-1",
        specialties={"frontend"},
        current_tasks=1,
        max_concurrent=2,
        tokens_used=0,
        available_tools={"echo"},
        model_tier=3,
        role="implementer",
    )
    breakdown = routing_breakdown(task, slot)
    assert set(breakdown) == {"specialty", "availability", "load", "capability"}
    # role fallback (bugfix -> implementer) drives the specialty factor
    assert breakdown["specialty"] == pytest.approx(1.0 * WEIGHTS["specialty"])
    assert breakdown["availability"] == pytest.approx(0.5 * WEIGHTS["availability"])
    assert sum(breakdown.values()) == pytest.approx(
        assignment_score(task, slot, team_average_tokens=0)
    )


# -- unit: recovery ladder --------------------------------------------------------


async def test_recovery_events_carry_run_and_agent() -> None:
    events: list[dict] = []

    async def failing_executor(task: Task) -> object:
        from harness.agents.task import TaskResult

        return TaskResult(task_id=task.id, success=False, error="ValueError: nope")

    class StubArchitect:
        async def reframe(self, task: Task, why: str) -> Task:
            return task

    class StubManager:
        async def handle_escalation(self, escalation) -> object:
            from harness.orchestration.messages import AgentStatus, StatusUpdate

            return StatusUpdate(
                sender="mgr-1", status=AgentStatus.WORKING, detail="add collaborators"
            )

    async def classify_async(task: Task, result: object) -> object:
        return _escalation_for(result)

    ladder = RecoveryLadder(
        StubManager(),
        StubArchitect(),
        store,
        on_event=events.append,
        run_id="run-rec",
        agent_id="impl-1",
    )
    result = await ladder.run(
        _task(),
        failing_executor,
        classify=classify_async,
    )
    assert result.success is False
    recovery = [e for e in events if str(e.get("event", "")).startswith("recovery.")]
    assert recovery
    assert all(e["run_id"] == "run-rec" for e in recovery)
    l1 = [e for e in recovery if e["event"] == "recovery.l1_retry"]
    assert l1 and all(e.get("agent") == "impl-1" for e in l1)
    l2 = [e for e in recovery if e["event"].startswith("recovery.l2")]
    assert l2 and all(e.get("agent") == "impl-1" for e in l2)


async def test_ladder_records_executing_agent() -> None:
    from harness.agents.task import TaskResult

    class Agent:
        agent_id = "impl-1-collab-1"

        async def execute_task(self, task: Task) -> TaskResult:
            return TaskResult(task_id=task.id, success=True, summary="done")

    store2 = MemoryContextStore()
    ladder = RecoveryLadder(manager=None, architect=None, store=store2)

    async def no_classify(task: Task, result: object) -> object:
        return _escalation_for(result)

    result = await ladder.run(_task(), Agent().execute_task, classify=no_classify)
    assert result.success
    assert ladder.last_executor_agent is not None
    assert ladder.last_executor_agent.agent_id == "impl-1-collab-1"


def _escalation_for(result: object) -> object:
    from harness.orchestration.messages import ErrorEscalation, Severity

    return ErrorEscalation(
        sender="impl-1",
        task_id="t-1",
        severity=Severity.TRANSIENT,
        error_type="ValueError",
        message=str(getattr(result, "error", "")),
        attempt=1,
    )


# -- integration fixtures ---------------------------------------------------------


class EventSinkCapture:
    """Event sink recording every streamed event (the gateway's view)."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        self.events.append(event)


@pytest.fixture
def event_sink_capture() -> EventSinkCapture:
    return EventSinkCapture()


PROFILE_JSON = (
    '{"languages": ["Python"], "frameworks": ["pytest"], "test_framework": "pytest", '
    '"build_system": "pyproject.toml", "conventions": ["typed"], "notes": "demo"}'
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
    import subprocess

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
def pipeline_config() -> HarnessConfig:
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


@pytest.fixture
def scripted_demo_provider(fake_model_config) -> FakeProvider:
    return FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content=PROFILE_JSON),  # architect.analyze
            ModelResponse(content=PLAN_JSON),  # architect.decompose
            ModelResponse(content="TASK_COMPLETE: verified greet() returns hello"),
            ModelResponse(content=VERDICT_JSON),  # architect.review
        ],
    )


@pytest.fixture
def service_app_client(fake_model_config) -> TestClient:
    from fastapi.testclient import TestClient

    from harness.service.app import create_app

    config = HarnessConfig.model_validate(
        {
            "models": {"default": {"provider": "fake", "name": "fake-model"}},
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "ver-1", "role": "verifier", "model": "default"},
            ],
            "storage": {"backend": "memory"},
        }
    )
    app = create_app(config=config, provider=FakeProvider(fake_model_config, responses=[]))
    return TestClient(app)


# -- integration: pipeline enriched events ---------------------------------------


async def test_pipeline_emits_v2_contract_events(
    demo_repo: Path, pipeline_config, scripted_demo_provider, event_sink_capture
) -> None:
    from harness.engine.pipeline import HarnessPipeline
    from harness.infrastructure.context_store import SQLiteContextStore
    from harness.security.audit import AuditLog

    store = SQLiteContextStore(demo_repo / ".harness" / "cockpit.db")
    pipeline = HarnessPipeline(
        demo_repo,
        pipeline_config,
        scripted_demo_provider,
        store,
        audit=AuditLog(demo_repo / ".harness" / "audit.jsonl"),
        event_sink=event_sink_capture,
    )
    outcome = await pipeline.run("app.greet() should return 'hello'", run_id="run-v2")

    events = event_sink_capture.events
    kinds = [e["event"] for e in events]

    # plan carries subtask OBJECTS, not bare ids
    plan_event = next(e for e in events if e["event"] == "architect.plan")
    assert plan_event["run_id"] == "run-v2"
    subtask = plan_event["subtasks"][0]
    assert isinstance(subtask, dict)
    assert subtask["id"] == "st-1"
    assert subtask["title"] == "add greeting"
    assert subtask["specialty"] == "verification"
    assert subtask["complexity"] == 2
    assert subtask["files"] == ["app.py"]
    assert subtask["depends_on"] == []
    assert "greet returns hello" in subtask["acceptance"]

    # baseline carries the reproduction test label
    baseline_event = next(e for e in events if e["event"] == "baseline.captured")
    assert "reproduction_test" in baseline_event

    # assignment carries role, batch, and the routing factor breakdown
    assigned = next(e for e in events if e["event"] == "specialist.assigned")
    assert assigned["agent"] == "ver-1"
    assert assigned["role"] == "verifier"
    assert assigned["batch"] == 1
    routing = assigned["routing"]
    assert set(routing) == {"specialty", "availability", "load", "capability"}
    assert routing["specialty"] == pytest.approx(0.4)  # verification is ver-1's preset

    # result carries attribution + steps/tokens
    result_event = next(e for e in events if e["event"] == "specialist.result")
    assert result_event["role"] == "verifier"
    assert result_event["steps"] >= 1
    assert result_event["tokens"] >= 0

    # per-agent activity events flowed from the agent loop
    assert "agent.step" in kinds
    assert "agent.usage" in kinds
    usage = [e for e in events if e["event"] == "agent.usage"]
    assert all(e["run_id"] == "run-v2" for e in usage)
    step = next(e for e in events if e["event"] == "agent.step")
    assert step["agent"] and step["max_steps"] >= 1

    # the trace.jsonl spine agrees with the streamed events
    trace = [
        json.loads(line)
        for line in (outcome.evidence_path / "trace.jsonl").read_text().splitlines()
    ]
    streamed = {(e.get("event"), e.get("run_id")) for e in events}
    traced = {(e.get("event"), e.get("run_id")) for e in trace}
    assert streamed <= traced


# -- integration: /api/agents hierarchy fields -----------------------------------


def test_agents_endpoint_carries_hierarchy_fields(service_app_client) -> None:
    response = service_app_client.get("/api/agents")
    assert response.status_code == 200
    agents = {a["agent_id"]: a for a in response.json()["agents"]}
    assert agents["arch-1"]["level"] == 1
    assert agents["mgr-1"]["level"] == 2
    assert agents["ver-1"]["level"] == 3
    assert "verification" in agents["ver-1"]["specialties"]
    assert agents["ver-1"]["tool_tier"] == 2  # DEVELOPMENT
    assert agents["arch-1"]["tool_tier"] == 2
    assert agents["mgr-1"]["tool_tier"] == 1  # BASIC
