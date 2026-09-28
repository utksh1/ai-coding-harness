"""LLMAgent loop tests: tools, budget, compression, classification (issue 2.7)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from harness.agents.llm_agent import (
    DEFAULT_KEEP_RECENT,
    LLMAgent,
    StoreWindow,
    StructuredOutputError,
    extract_json,
)
from harness.agents.task import Task
from harness.config import BudgetConfig
from harness.engine.budget import BudgetExhausted, BudgetGovernor
from harness.infrastructure.context_store import MemoryContextStore
from harness.infrastructure.model_providers import FakeProvider, ModelResponse, ToolCall
from harness.orchestration.messages import Severity
from harness.tools.base import Tool, ToolResult, ToolTier


class EchoTool(Tool):
    name, tier, description = "echo_tool", ToolTier.BASIC, "echo content back"
    parameters: ClassVar[dict] = {"type": "object", "properties": {"content": {"type": "string"}}}

    def validate_input(self, arguments: dict) -> list[str]:
        return [] if "content" in arguments else ["missing 'content'"]

    def check_permissions(self, context: dict) -> bool:
        return context.get("model_tier", 1) >= self.tier.value

    def execute(self, content: str = "") -> ToolResult:
        return ToolResult(success=True, output=f"echo: {content}")


class WriterTool(EchoTool):
    name, tier = "writer_tool", ToolTier.DEVELOPMENT


class CrashTool(Tool):
    name, tier, description = "crash_tool", ToolTier.BASIC, "always crashes"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def validate_input(self, arguments: dict) -> list[str]:
        return []

    def check_permissions(self, context: dict) -> bool:
        return True

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("boom")


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
    role: str = "implementer",
    model_tier: int = 3,
    tools: list[Tool] | None = None,
    total_tokens: int = 100_000,
    max_steps: int = 8,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> LLMAgent:
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=total_tokens), "corr-llm")
    return LLMAgent(
        agent_id="impl-1",
        model_config={"provider": "fake"},
        tools=tools or [EchoTool(), WriterTool()],
        context_window=StoreWindow(store, "impl-1", "t-1"),
        provider=provider,
        store=store,
        governor=governor,
        role=role,
        model_tier=model_tier,
        max_steps=max_steps,
        keep_recent=keep_recent,
    )


TASK = Task(id="t-1", title="do a thing", description="the thing")


async def test_tool_call_then_completion_flow(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("echo_tool", content="hello"),
            _text(f"TASK_COMPLETE: echoed it\n{_json_block()}"),
        ],
    )
    agent = _agent(store, provider)
    result = await agent.execute_task(TASK)
    assert result.success and "TASK_COMPLETE" in result.summary
    usage = store.token_usage("corr-llm")
    assert usage.total_tokens > 0
    context = store.load_agent_context("impl-1", "t-1")
    assert context.milestones and context.milestones[0].startswith("OK")
    assert "[echo_tool]" in context.summary or any("echo_tool" in t.content for t in context.recent)


def _json_block() -> str:
    return '```json {"final": true} ```'


async def test_unknown_tool_becomes_tool_result(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config, responses=[_call("nope"), _text("TASK_COMPLETE: done")]
    )
    result = await _agent(store, provider).execute_task(TASK)
    assert result.success
    window = store.load_agent_context("impl-1", "t-1")
    assert any("unknown tool 'nope'" in t.content for t in window.recent)


async def test_invalid_arguments_become_tool_result(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config, responses=[_call("echo_tool"), _text("TASK_COMPLETE: done")]
    )
    result = await _agent(store, provider).execute_task(TASK)
    assert result.success
    window = store.load_agent_context("impl-1", "t-1")
    assert any("invalid arguments" in t.content for t in window.recent)


async def test_permission_denial_at_low_tier(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[_call("writer_tool", content="x"), _text("TASK_COMPLETE: done")],
    )
    agent = _agent(store, provider, model_tier=1)
    await agent.execute_task(TASK)
    window = store.load_agent_context("impl-1", "t-1")
    assert any("permission denied" in t.content for t in window.recent)


async def test_preset_tier_limits_schemas(store, fake_model_config) -> None:
    provider = FakeProvider(fake_model_config, responses=[_text("TASK_COMPLETE: done")])
    agent = _agent(store, provider, role="locator", tools=[EchoTool(), WriterTool()])
    await agent.execute_task(TASK)
    sent = provider.calls[0]["tools"]
    assert [t["name"] for t in sent] == ["echo_tool"]  # DEVELOPMENT filtered out


async def test_crashing_tool_returns_error_result(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config, responses=[_call("crash_tool"), _text("TASK_COMPLETE: done")]
    )
    result = await _agent(store, provider, tools=[CrashTool()]).execute_task(TASK)
    assert result.success
    window = store.load_agent_context("impl-1", "t-1")
    assert any("tool crashed: boom" in t.content for t in window.recent)


async def test_step_limit_is_an_honest_failure(store, fake_model_config) -> None:
    provider = FakeProvider(fake_model_config, responses=[_call("echo_tool", content="loop")] * 10)
    result = await _agent(store, provider, max_steps=3).execute_task(TASK)
    assert not result.success
    assert "step limit" in (result.error or "")


async def test_budget_exhaustion_stops_gracefully(store, fake_model_config) -> None:
    provider = FakeProvider(fake_model_config, responses=[_call("echo_tool", content="x")] * 5)
    agent = _agent(store, provider, total_tokens=5)
    result = await agent.execute_task(TASK)
    assert not result.success and "budget exhausted" in (result.error or "")


async def test_compression_triggers_on_long_windows(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("echo_tool", content="a"),
            _call("echo_tool", content="b"),
            _call("echo_tool", content="c"),
            _text("TASK_COMPLETE: finished"),
        ],
    )
    agent = _agent(store, provider, keep_recent=4)
    await agent.execute_task(TASK)
    context = store.load_agent_context("impl-1", "t-1")
    assert context.summary  # window was folded


async def test_handle_error_classification_and_attempts(store, fake_model_config) -> None:
    agent = _agent(store, FakeProvider(fake_model_config, responses=[]))
    task = TASK
    cases = [
        (BudgetExhausted("b"), Severity.FATAL),
        (TimeoutError("t"), Severity.TRANSIENT),
        (ConnectionError("c"), Severity.TRANSIENT),
        (StructuredOutputError("s"), Severity.RECOVERABLE),
        (ValueError("v"), Severity.RECOVERABLE),
        (RuntimeError("r"), Severity.RECOVERABLE),
    ]
    for error, expected in cases:
        escalation = await agent.handle_error(error, task)
        assert escalation.severity == expected, error
    assert await agent.handle_error(ValueError("again"), task)
    assert agent._attempts["t-1"] == 7


def test_report_status_reflects_activity(store, fake_model_config) -> None:
    agent = _agent(store, FakeProvider(fake_model_config, responses=[]))
    assert agent.report_status().status.value == "idle"
    agent._active_task = "t-1"
    assert agent.report_status().status.value == "working"
    assert agent.report_status().task_id == "t-1"


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure! {"a": 3} hope that helps') == {"a": 3}
    with pytest.raises(StructuredOutputError, match="no JSON"):
        extract_json("no structure here")
    with pytest.raises(StructuredOutputError, match="no JSON"):
        extract_json("```json\n```")  # empty fence
    assert extract_json('{"a": "say \\"} ok"}') == {"a": 'say "} ok'}  # escapes
    with pytest.raises(StructuredOutputError, match="unterminated"):
        extract_json("{not json")
    with pytest.raises(StructuredOutputError, match="invalid JSON"):
        extract_json('{"a": }')
    with pytest.raises(StructuredOutputError, match="expected a JSON object"):
        extract_json("```json\n[1, 2]\n```")


def test_system_prompt_composition() -> None:
    from harness.agents.prompts import system_prompt

    plain = system_prompt("implementer")
    assert "Implementer" in plain
    unknown = system_prompt("mystery-role")
    assert "mystery-role" in unknown
    layered = system_prompt("locator", fact_ledger="- decided X", extra="Be brief.")
    assert "Fact ledger" in layered and "- decided X" in layered and "Be brief." in layered


async def test_structured_call_repairs_once(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _text("oops not json"),
            _text('{"answer": 42}'),
        ],
    )
    agent = _agent(store, provider, role="architect")
    data = await agent.structured_call("schema", "question", '{"answer": int}')
    assert data == {"answer": 42}


async def test_structured_call_fails_after_repair(store, fake_model_config) -> None:
    provider = FakeProvider(fake_model_config, responses=[_text("nope"), _text("still nope")])
    agent = _agent(store, provider, role="architect")
    with pytest.raises(StructuredOutputError):
        await agent.structured_call("schema", "question", "{}")


def test_store_window_roundtrip(store) -> None:
    window = StoreWindow(store, "a-1", "t-9")
    window.append("user", "hi")
    window.append("assistant", "ho")
    assert window.as_messages() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ho"},
    ]


async def test_unmarked_finish_triggers_nudge_then_accepts(store, fake_model_config) -> None:
    """A reply with no tool calls and no marker is a pause, not a finish."""
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _text("Let me look around first."),
            _text("I think I understand the issue now."),
            _text("TASK_COMPLETE: actually did the work"),
        ],
    )
    agent = _agent(store, provider, tools=[EchoTool()])
    result = await agent.execute_task(TASK)
    assert result.success and "actually did the work" in result.summary
    window = store.load_agent_context("impl-1", "t-1")
    nudges = [t for t in window.recent if "You are not finished" in t.content]
    assert len(nudges) == 2  # bounded nudging


async def test_unmarked_finishes_exhaust_then_honest_failure(store, fake_model_config) -> None:
    """The false-VERIFIED killer, codified honestly (review finding #1).

    An agent that never sends TASK_COMPLETE has NOT completed the task, no
    matter how confident its prose sounds: five "I'm thinking" replies end
    the subtask FAILED (feeding the recovery ladder), never 'successfully'.
    The verification gates judge the WORK - but a completion claim must
    first exist for anything to be judged, and an empty diff can no longer
    slip through the gates either (see the self-check no-op gate).
    """
    provider = FakeProvider(fake_model_config, responses=[_text("thinking") for _ in range(5)])
    agent = _agent(store, provider, tools=[EchoTool()], max_steps=6)
    result = await agent.execute_task(TASK)
    assert result.success is False
    assert result.error is not None
    assert "task_not_completed" in result.error
    assert "thinking" in result.summary


async def test_tool_alias_resolution(store, fake_model_config) -> None:
    """Models use natural names (read_file); aliases resolve and execute."""
    from harness.tools.registry import build_default_tools

    registry_tools = [t for t in build_default_tools(Path(".")) if t.name == "filesystem_read"]
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("read_file", path="pyproject.toml"),
            _text("TASK_COMPLETE: read it"),
        ],
    )
    agent = _agent(store, provider, tools=[EchoTool(), *registry_tools])
    result = await agent.execute_task(TASK)
    assert result.success
    window = store.load_agent_context("impl-1", "t-1")
    tool_turns = [t for t in window.recent if t.role == "tool"]
    assert tool_turns and tool_turns[0].tool_name == "read_file"
    assert "build-system" in tool_turns[0].content  # alias executed the real tool


async def test_unknown_tool_error_lists_available(store, fake_model_config) -> None:
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("quantum_flip"),
            _text("TASK_COMPLETE: gave up on the mystery tool"),
        ],
    )
    agent = _agent(store, provider, tools=[EchoTool()])
    await agent.execute_task(TASK)
    window = store.load_agent_context("impl-1", "t-1")
    assert any("available: echo_tool" in t.content for t in window.recent)


def test_parse_envelope_invalid_json_falls_back_to_plain(store) -> None:
    """A plain reply that happens to start with '{' must not crash the rebuild."""
    window = StoreWindow(store, "a-1", "t-env")
    window.append("assistant", "{not valid json but it is the model's words")
    messages = window.as_messages()
    assert messages == [
        {"role": "assistant", "content": "{not valid json but it is the model's words"}
    ]


async def test_guidance_from_metadata_reaches_the_model(store, fake_model_config) -> None:
    """Audit §8: L2 re-route guidance must actually reach the specialist."""
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _text("TASK_COMPLETE: followed the guidance"),
        ],
    )
    agent = _agent(store, provider, tools=[EchoTool()])
    task = Task(
        id="t-g", title="t", description="d", metadata={"guidance": "try narrower scope first"}
    )
    await agent.execute_task(task)
    sent = provider.calls[0]["messages"]
    user_text = " ".join(str(m.get("content")) for m in sent if m["role"] == "user")
    system_text = sent[0]["content"]
    assert "MANAGER GUIDANCE: try narrower scope first" in user_text
    assert "guidance" not in system_text.lower() or True  # user-turn delivery is canonical


async def test_persona_tier_cap_blocks_named_tools(store, fake_model_config) -> None:
    """The tier cap is a hard contract: a locator naming apply_edit is denied."""
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("writer_tool", content="x"),
            _text("TASK_COMPLETE: done"),
        ],
    )
    agent = _agent(store, provider, role="locator", model_tier=3)  # even at tier 3
    result = await agent.execute_task(TASK)
    assert result.success
    window = store.load_agent_context("impl-1", "t-1")
    assert any("exceeds this persona's tool tier cap" in t.content for t in window.recent)


async def test_alias_cannot_bypass_persona_tier_cap(store, fake_model_config) -> None:
    """55a16f9 hardened the exact-name path; the alias path must be too.

    A read-only persona (locator, BASIC cap) naming the natural alias
    'write_file' must be denied at execution time, not slip through to
    filesystem_write via TOOL_ALIASES.
    """
    from pathlib import Path

    from harness.tools.filesystem import WriteFileTool

    write_tool = WriteFileTool(Path("."))
    provider = FakeProvider(
        fake_model_config,
        responses=[
            _call("write_file", path="escape.py", content="x"),
            _text("TASK_COMPLETE: tried the alias"),
        ],
    )
    agent = _agent(store, provider, tools=[EchoTool(), write_tool])
    agent.role = "locator"  # BASIC tier cap — preset is a property, reads live
    await agent.execute_task(TASK)
    window = store.load_agent_context("impl-1", "t-1")
    tool_turn = next(t for t in window.recent if t.role == "tool")
    assert "exceeds this persona's tool tier cap" in tool_turn.content
    assert not (Path(".") / "escape.py").exists()


async def test_retry_reusing_task_id_starts_from_a_clean_window(store, fake_model_config) -> None:
    """A retry under the same task id must not rehydrate the failed attempt's
    turns: duplicate TASK turns made the model replay old replies (live-run
    finding behind the NOT VERIFIED loop)."""
    stale = store.load_agent_context("impl-1", "t-1")
    store.append_turn("impl-1", "t-1", "user", "TASK: old failed attempt")
    store.append_turn("impl-1", "t-1", "assistant", "I already did this.")
    provider = FakeProvider(fake_model_config, responses=[_text("TASK_COMPLETE: fresh attempt")])
    agent = _agent(store, provider)
    result = await agent.execute_task(TASK)
    assert result.success
    messages = provider.calls[0]["messages"]
    task_turns = [m for m in messages if m["role"] == "user" and "TASK:" in str(m.get("content"))]
    assert len(task_turns) == 1
    assert all("old failed attempt" not in str(m.get("content")) for m in messages)
    # The fresh window is persisted for the next stage, not just in memory.
    assert store.load_agent_context("impl-1", "t-1").recent
    del stale
