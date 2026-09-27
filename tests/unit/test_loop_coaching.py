"""Loop coaching: the harness teaches a flailing model back on course.

Live-run findings (run e54878f8, gpt-5.6-luna): 30 of 121 tool calls failed
with repeated identical argument mistakes (six 'empty path' in a row), edits
were never followed by run_tests, and the step limit arrived silently. These
tests pin the three coaching hooks - usage coaching after repeated identical
failures, step-budget warnings, verify-your-edits nudges - plus the failure
dedup cache and the self-recovering apply_edit error.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from harness.agents.llm_agent import DEFAULT_KEEP_RECENT, LLMAgent, StoreWindow
from harness.agents.task import Task
from harness.config import BudgetConfig
from harness.engine.budget import BudgetGovernor
from harness.infrastructure.context_store import MemoryContextStore
from harness.infrastructure.model_providers import FakeProvider, ModelResponse, ToolCall
from harness.tools.base import Tool, ToolResult, ToolTier
from harness.tools.editing import ApplyEditTool


class FailingSearchTool(Tool):
    """Deterministic failure source with a stable error signature."""

    name, tier, description = "search_text", ToolTier.BASIC, "regex search"
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"],
    }

    def validate_input(self, arguments: dict) -> list[str]:
        return [] if arguments.get("pattern") else ["'pattern' is required"]

    def check_permissions(self, context: dict) -> bool:
        return True

    def execute(self, pattern: str = "", path: str = "", **_: object) -> ToolResult:
        return ToolResult(success=False, error="empty path")


class SucceedingWriterTool(Tool):
    """Mutating tool that succeeds - drives the unverified-edit tracking."""

    name, tier, description = "apply_edit", ToolTier.DEVELOPMENT, "edit a file"
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "search": {"type": "string"},
            "replace": {"type": "string"},
        },
        "required": ["path", "search", "replace"],
    }

    def validate_input(self, arguments: dict) -> list[str]:
        return []

    def check_permissions(self, context: dict) -> bool:
        return True

    def execute(self, **_: object) -> ToolResult:
        return ToolResult(success=True, output="diff applied")


def _text(text: str) -> ModelResponse:
    return ModelResponse(content=text)


def _call(name: str, **arguments: str) -> ModelResponse:
    return ModelResponse(content="", tool_calls=[ToolCall(name=name, arguments=arguments)])


@pytest.fixture
def store() -> MemoryContextStore:
    return MemoryContextStore()


def _agent(
    store,
    provider,
    *,
    tools: list[Tool],
    max_steps: int = 12,
) -> LLMAgent:
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=1_000_000), "coach")
    return LLMAgent(
        agent_id="impl-1",
        model_config={"provider": "fake"},
        tools=tools,
        context_window=StoreWindow(store, "impl-1", "t-1"),
        provider=provider,
        store=store,
        governor=governor,
        role="implementer",
        model_tier=3,
        max_steps=max_steps,
        keep_recent=DEFAULT_KEEP_RECENT,
    )


TASK = Task(id="t-1", title="do a thing", description="the thing")


async def test_repeated_failures_trigger_usage_coaching(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("search_text", pattern="def parse", path=""),
            _call("search_text", pattern="def parse", path=""),
            _text("TASK_COMPLETE: done"),
        ],
    )
    agent = _agent(store, provider, tools=[FailingSearchTool()])
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    coaching = [t for t in context.recent if "failed the same way" in t.content]
    assert coaching, "no coaching nudge after repeated identical failures"
    assert "search_text" in coaching[0].content
    assert "Correct usage" in coaching[0].content
    assert agent._coach_nudges == 1


async def test_coaching_is_bounded_not_spammed(store, fake_model_config) -> None:
    # Six identical failures, but only MAX_COACH_NUDGES coaching messages.
    provider = FakeProvider(
        fake_model_config,
        responses=[_call("search_text", pattern="x", path="")] * 6 + [_text("TASK_COMPLETE: done")],
    )
    agent = _agent(store, provider, tools=[FailingSearchTool()], max_steps=10)
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    coaching = [t for t in context.recent if "failed the same way" in t.content]
    assert len(coaching) <= 4


async def test_step_budget_warning_injected(store, fake_model_config) -> None:
    # 10 steps of tool calls; warning fires at 70% (step 7) and 90% (step 9).
    provider = FakeProvider(
        fake_model_config,
        responses=[_call("search_text", pattern="x", path="ok")] * 9
        + [_text("TASK_COMPLETE: done")],
    )
    agent = _agent(store, provider, tools=[FailingSearchTool()], max_steps=10)
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    warnings = [t for t in context.recent if "STEP BUDGET" in t.content]
    assert len(warnings) == 2
    assert "7 of 10" in warnings[0].content
    assert "9 of 10" in warnings[1].content


async def test_unverified_edit_nudge_after_grace(store, fake_model_config) -> None:
    # A successful edit at step 1, then four non-test rounds: the nudge fires.
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("apply_edit", path="a.py", search="x", replace="y"),
            _call("search_text", pattern="z", path="."),
            _call("search_text", pattern="z2", path="."),
            _call("search_text", pattern="z3", path="."),
            _call("search_text", pattern="z4", path="."),
            _text("TASK_COMPLETE: done"),
        ],
    )

    class OkSearch(FailingSearchTool):
        def execute(self, pattern: str = "", path: str = "", **_: object) -> ToolResult:
            return ToolResult(success=True, output="no matches")

    agent = _agent(store, provider, tools=[SucceedingWriterTool(), OkSearch()])
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    nudges = [t for t in context.recent if "have not run the tests" in t.content]
    assert nudges, "edit was never verified but no nudge fired"


async def test_failure_dedup_returns_cached_note(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("search_text", pattern="dup", path=""),
            _call("search_text", pattern="dup", path=""),
            _text("TASK_COMPLETE: done"),
        ],
    )
    agent = _agent(store, provider, tools=[FailingSearchTool()])
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    dedup_notes = [t for t in context.recent if "already failed with this exact error" in t.content]
    assert dedup_notes, "identical failing call re-executed instead of deduped"


def test_apply_edit_failure_returns_closest_region(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    )
    tool = ApplyEditTool(tmp_path)
    result = tool.execute(path="calc.py", search="    return a + c", replace="    return a - c")
    assert not result.success
    assert "closest actual region" in (result.error or "")
    assert "return a + b" in (result.error or "")
    assert "1|" in (result.error or "")  # line-numbered like filesystem_read


def test_apply_edit_noop_guides_already_applied(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("x = 1\n")
    tool = ApplyEditTool(tmp_path)
    result = tool.execute(path="calc.py", search="x = 1", replace="x = 1")
    assert not result.success
    assert "already applied" in (result.error or "")
    assert "run_tests" in (result.error or "")


def test_apply_edit_truly_missing_stays_plain(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("x = 1\n")
    tool = ApplyEditTool(tmp_path)
    result = tool.execute(path="calc.py", search="zzz_qqq_12345", replace="y")
    assert not result.success
    assert "closest actual region" not in (result.error or "")
    assert "read the file again" in (result.error or "")


class FlakySearchTool(FailingSearchTool):
    """Fails the first call, succeeds the second - exercises streak clearing."""

    calls: ClassVar[int] = 0

    def execute(self, pattern: str = "", path: str = "", **_: object) -> ToolResult:
        FlakySearchTool.calls += 1
        if FlakySearchTool.calls == 1:
            return ToolResult(success=False, error="empty path")
        return ToolResult(success=True, output="match found")


class RunTestsTool(Tool):
    """Successful test run - completes the edit -> test -> no-nudge cycle."""

    name, tier, description = "run_tests", ToolTier.DEVELOPMENT, "run the test suite"
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def validate_input(self, arguments: dict) -> list[str]:
        return []

    def check_permissions(self, context: dict) -> bool:
        return True

    def execute(self, **_: object) -> ToolResult:
        return ToolResult(success=True, output="3 passed")


async def test_run_tests_after_edit_clears_pending_nudge(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("apply_edit", path="a.py", search="x", replace="y"),
            _call("run_tests", path="."),
            _text("TASK_COMPLETE: done"),
        ],
    )
    agent = _agent(
        store, provider, tools=[SucceedingWriterTool(), RunTestsTool(), FlakySearchTool()]
    )
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    assert not [t for t in context.recent if "have not run the tests" in t.content]


async def test_success_after_failure_clears_streak(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("search_text", pattern="flaky", path=""),
            _call("search_text", pattern="flaky", path="now-ok"),
            _text("TASK_COMPLETE: done"),
        ],
    )
    agent = _agent(store, provider, tools=[FlakySearchTool()])
    await agent.execute_task(TASK)
    assert agent._failure_streaks == {}


async def test_usage_hint_falls_back_to_schema(store, fake_model_config) -> None:
    class OddTool(FailingSearchTool):
        name, description = "odd_tool", "an odd tool"

    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("odd_tool", pattern="x", path=""),
            _call("odd_tool", pattern="x", path=""),
            _text("TASK_COMPLETE: done"),
        ],
    )
    agent = _agent(store, provider, tools=[OddTool()])
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    coaching = [t for t in context.recent if "failed the same way" in t.content]
    assert coaching and "odd_tool" in coaching[0].content
    assert "an odd tool" in coaching[0].content


def test_apply_edit_empty_file_no_fuzzy_context(tmp_path: Path) -> None:
    (tmp_path / "empty.py").write_text("")
    result = ApplyEditTool(tmp_path).execute(path="empty.py", search="abc", replace="x")
    assert not result.success
    assert "read the file again" in (result.error or "")


def test_search_text_nonexistent_dir_errors_cleanly(tmp_path: Path) -> None:
    from harness.tools.editing import SearchTextTool

    result = SearchTextTool(tmp_path).execute(pattern="x", path="ghost_dir")
    assert not result.success and "not a directory" in (result.error or "")


async def test_unknown_tool_coaching_uses_generic_hint(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("nonexistent_tool", path="x"),
            _call("nonexistent_tool", path="x"),
            _text("TASK_COMPLETE: done"),
        ],
    )
    agent = _agent(store, provider, tools=[FailingSearchTool()])
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    coaching = [t for t in context.recent if "failed the same way" in t.content]
    assert coaching
    assert "Check the tool's parameter schema" in coaching[0].content
    assert "nonexistent_tool" in coaching[0].content
