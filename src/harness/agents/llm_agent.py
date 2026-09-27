"""Concrete LLM-backed agent loop (milestone 2, issue 2.7 - specialist base).

`LLMAgent` implements the `BaseAgent` contract over a model provider and the
context store:

- every step checks the budget governor and records usage to the ledger;
- tool calls go through validation + permission gating, and failures become
  tool results (never exceptions) so the recovery ladder can reason over them;
- the conversation lives in the store's three-window context, compressed
  automatically when the recent window grows past `keep_recent`;
- bare replies end the loop (optionally marked with `TASK_COMPLETE:`), and a
  structured-JSON helper with one repair round supports the Architect/Manager
  contracts.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

from harness.agents.base import BaseAgent, ContextWindow
from harness.agents.prompts import FINAL_MARKER, ROLE_PRESETS, RolePreset, system_prompt
from harness.agents.task import Task, TaskResult
from harness.engine.budget import BudgetExhausted, BudgetGovernor, GovernorMode
from harness.infrastructure.context_store import ContextStore
from harness.infrastructure.logging import get_logger
from harness.infrastructure.model_providers import (
    ModelProvider,
    ModelResponse,
)
from harness.infrastructure.model_providers.capability import (
    CapabilityCache,
    ModelCapabilities,
    parse_tool_call_blocks,
    render_tool_manual,
    strip_think_blocks,
)
from harness.knowledge.registry import knowledge_section
from harness.orchestration.messages import AgentStatus, ErrorEscalation, Severity, StatusUpdate
from harness.tools.base import AsyncExecutableTool, Tool, ToolResult, ToolTier

logger = get_logger(__name__)

DEFAULT_KEEP_RECENT = 24
DEFAULT_MAX_STEPS = 16
DEFAULT_STALE_TOOL_RESULTS = 6
"""Tool results outside the newest N keep full text; older ones are stubbed.

Tool outputs dominate window bytes and replay verbatim on every step (M5
issue #61). The store stays lossless; only message assembly degrades stale
results to one-line stubs.
"""
MAX_UNMARKED_NUDGES = 2
_CAPABILITY_CACHE = CapabilityCache()
REPEAT_STRIKE_THRESHOLD = 2
DEDUP_TOOLS = frozenset({"filesystem_read", "filesystem_list", "search_text"})
"""Deterministic tools whose identical repeat calls return a cached result."""
MUTATING_TOOLS = frozenset({"apply_edit", "code_execution"})
"""Successful calls invalidate the dedup cache (files may have changed)."""
_NUDGE = (
    "You are not finished: use the available tools to complete the task now. "
    f"Reply with '{FINAL_MARKER}: <summary>' ONLY when done."
)
_REPEAT_NUDGE = (
    "Loop detected: you repeated identical tool call(s) and received cached "
    "results. Change approach - read different files, apply an edit, or "
    f"finish with {FINAL_MARKER}."
)

_STEP_LIMIT_ERROR = "step limit reached before the agent finished"


class StructuredOutputError(Exception):
    """The model's reply could not be parsed as the required JSON structure."""


def compose_task_prompt(task: Task) -> str:
    """Full task brief for a specialist (audit §9).

    Planning output must reach execution: the description is joined by the
    Architect's acceptance criteria, the expected files, the required tools,
    and any Manager guidance attached to the task. Long sections are capped
    (observation compression, improvements §2.2).
    """
    sections = [f"TASK: {task.title}", task.description.strip()]
    if task.acceptance_criteria:
        criteria = "\n".join(f"- {c[:200]}" for c in task.acceptance_criteria[:10])
        sections.append(f"ACCEPTANCE CRITERIA:\n{criteria}")
    if task.files:
        sections.append("EXPECTED FILES: " + ", ".join(task.files[:12]))
    if task.required_tools:
        sections.append("REQUIRED TOOLS: " + ", ".join(task.required_tools[:12]))
    guidance = task.metadata.get("guidance")
    if guidance:
        sections.append(f"MANAGER GUIDANCE: {str(guidance)[:600]}")
    return "\n\n".join(sections)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Accepts bare JSON, ```json fenced blocks, or JSON embedded in prose.
    Raises `StructuredOutputError` when no object is present.
    """
    fenced_start = text.find("```json")
    if fenced_start >= 0:
        candidate = text[fenced_start + 7 :]
        fenced_end = candidate.find("```")
        if fenced_end >= 0:
            candidate = candidate[:fenced_end]
    else:
        brace_start = text.find("{")
        if brace_start < 0:
            msg = f"no JSON object found in reply: {text[:120]!r}"
            raise StructuredOutputError(msg)
        candidate = _balanced_object(text, brace_start)
    if not candidate.strip():
        msg = f"no JSON object found in reply: {text[:120]!r}"
        raise StructuredOutputError(msg)
    try:
        parsed = json.loads(candidate.strip())
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON in model reply: {exc}"
        raise StructuredOutputError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"expected a JSON object, got {type(parsed).__name__}"
        raise StructuredOutputError(msg)
    return parsed


def _balanced_object(text: str, start: int) -> str:
    """Slice the first balanced {...} object starting at `start` (quote-aware)."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    msg = "unterminated JSON object in reply"
    raise StructuredOutputError(msg)


def _stale_tool_stub(turn: Any) -> str:
    """One-line replacement for a tool result outside the newest N (issue #61)."""
    first = next((ln.strip() for ln in turn.content.splitlines() if ln.strip()), "(empty)")
    name = turn.tool_name or "tool"
    return f"[elided stale {name} result ({len(turn.content)} chars): {first[:100]}]"


class StoreWindow:
    """ContextWindow implementation backed by the context store."""

    def __init__(
        self,
        store: ContextStore,
        agent_id: str,
        task_id: str,
        stale_tool_results: int = DEFAULT_STALE_TOOL_RESULTS,
    ) -> None:
        self._store = store
        self._agent_id = agent_id
        self._task_id = task_id
        self._stale_tool_results = stale_tool_results

    @property
    def task_id(self) -> str:
        """The window's task namespace (structured calls compress against it)."""
        return self._task_id

    def append(
        self,
        role: str,
        content: str,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str = "",
        tool_name: str = "",
    ) -> None:
        self._store.append_turn(
            self._agent_id,
            self._task_id,
            role,
            content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    def as_messages(self) -> list[dict[str, Any]]:
        """Provider-neutral projection of the recent window.

        Assistant turns carrying `tool_calls` emit them flat
        (`{id, name, arguments}`); tool turns emit their `tool_call_id` and
        `tool_name`. Each provider projects this onto its own wire format in
        its `_payload`, so the native tool-calling protocol round-trips.

        Tool results outside the newest `stale_tool_results` are assembled as
        one-line stubs (#61): the store keeps them losslessly, but replaying
        every old output at full size on every step is the largest single
        prompt-token cost in multi-round tasks.
        """
        context = self._store.load_agent_context(self._agent_id, self._task_id)
        tool_positions = [i for i, turn in enumerate(context.recent) if turn.role == "tool"]
        cutoff = len(tool_positions) - self._stale_tool_results
        stale_positions = set(tool_positions[: max(0, cutoff)])
        messages: list[dict[str, Any]] = []
        for position, turn in enumerate(context.recent):
            message: dict[str, Any] = {"role": turn.role, "content": turn.content}
            if turn.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.get("id", ""),
                        "name": call.get("name", ""),
                        "arguments": call.get("arguments") or {},
                    }
                    for call in turn.tool_calls
                ]
            if turn.tool_call_id:
                message["tool_call_id"] = turn.tool_call_id
                message["tool_name"] = turn.tool_name
                if position in stale_positions:
                    message["content"] = _stale_tool_stub(turn)
            messages.append(message)
        return messages


Tracer = Callable[[dict[str, Any]], None]
"""Cockpit event sink: one JSON-serializable event dict per call.

Attached per run by the pipeline (``attach_tracer``); emission failures are
swallowed so a cockpit can never break an eval run."""


class LLMAgent(BaseAgent):
    """Agent loop over a model provider with tools, budgeting, and context."""

    def __init__(
        self,
        agent_id: str,
        model_config: dict[str, Any],
        tools: list[Tool],
        context_window: ContextWindow,
        provider: ModelProvider,
        store: ContextStore,
        governor: BudgetGovernor,
        role: str = "implementer",
        model_tier: int = 3,
        max_steps: int = DEFAULT_MAX_STEPS,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        stale_tool_results: int = DEFAULT_STALE_TOOL_RESULTS,
        knowledge_enabled: bool = True,
        knowledge_max_chars: int = 1500,
    ) -> None:
        super().__init__(agent_id, model_config, tools, context_window)
        self.provider = provider
        self.store = store
        # The pipeline replaces this placeholder with the per-run governor
        # before execute_task; constructing without one is a wiring error.
        self.governor = governor
        self.role = role
        self.model_tier = model_tier
        self.max_steps = max_steps
        self.keep_recent = keep_recent
        self.stale_tool_results = stale_tool_results
        self.knowledge_enabled = knowledge_enabled
        self.knowledge_max_chars = knowledge_max_chars
        self._active_task: str | None = None
        self._attempts: dict[str, int] = {}
        self._capabilities: ModelCapabilities | None = None
        self.capability_cache: CapabilityCache = _CAPABILITY_CACHE
        # Cockpit tracing (contract: docs/cockpit-events.md). None until the
        # pipeline attaches one per run; counters reset with it.
        self.tracer: Tracer | None = None
        self.run_id: str | None = None
        self.traced_tokens: int = 0
        self._structured_steps: int = 0
        self._last_steps: int = 0
        self._reset_dedup_state()

    def attach_tracer(self, tracer: Tracer | None, run_id: str | None) -> None:
        """Attach the cockpit event tracer for one run (or detach with None)."""
        self.tracer = tracer
        self.run_id = run_id
        self.traced_tokens = 0
        self._structured_steps = 0
        self._last_steps = 0

    @property
    def steps_used(self) -> int:
        """Steps consumed by the most recent ``execute_task`` (cockpit reads)."""
        return self._last_steps

    def _emit(self, event: dict[str, Any]) -> None:
        """Emit one cockpit event; tracing must never break a run."""
        if self.tracer is None:
            return
        payload = {
            **event,
            "run_id": self.run_id,
            "agent": self.agent_id,
            "role": self.role,
        }
        # Cockpit isolation by contract: a broken event sink must never
        # break the eval run (mirrors EvidencePack's event_sink guard).
        with contextlib.suppress(Exception):
            self.tracer(payload)

    def _emit_step(self, phase: str, step: int, task: str | None) -> None:
        self._emit(
            {
                "event": "agent.step",
                "task": task or "ad-hoc",
                "step": step,
                "max_steps": self.max_steps,
                "phase": phase,
            }
        )

    def _emit_usage(self, response: ModelResponse, task: str | None) -> None:
        self.traced_tokens += response.prompt_tokens + response.completion_tokens
        self._emit(
            {
                "event": "agent.usage",
                "task": task or "ad-hoc",
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.prompt_tokens + response.completion_tokens,
                "total_tokens_agent": self.traced_tokens,
            }
        )

    def _emit_tool(
        self, name: str, arguments: dict[str, Any], result: ToolResult, duration_ms: float
    ) -> None:
        self._emit(
            {
                "event": "agent.tool",
                "task": self._active_task
                or getattr(self.context_window, "task_id", None)
                or "ad-hoc",
                "step": self._last_steps or self._structured_steps,
                "tool": name,
                "args_digest": _args_digest(name, arguments),
                "ok": result.success,
                "duration_ms": round(duration_ms),
                "result_digest": _result_digest(result),
            }
        )

    def _trace_task(self) -> str | None:
        """Best-effort task label for an in-flight model call."""
        return self._active_task or getattr(self.context_window, "task_id", None)

    def _reset_dedup_state(self) -> None:
        """Fresh per-task dedup cache (#62): files may change between tasks."""
        self._dedup_cache: dict[tuple[str, str], str] = {}
        self._dedup_strikes: dict[tuple[str, str], int] = {}
        self._round_dedup_strikes: list[int] = []
        self._repeat_nudges = 0

    # -- BaseAgent contract ---------------------------------------------------
    @property
    def preset(self) -> RolePreset | None:
        return ROLE_PRESETS.get(self.role)

    async def native_tool_calls(self) -> bool:
        """Whether this agent's model takes native tool calls (§3.1).

        `auto` probes once per model and caches; explicit config wins.
        Probe failure is optimistic (native), matching pre-probe behavior.
        """
        mode = self.provider._config.tool_call_mode  # same package boundary
        if mode == "native":
            return True
        if mode == "text":
            return False
        if self._capabilities is None:
            self._capabilities = await self.capability_cache.get_or_probe(self.provider)
            logger.info(
                "model capabilities probed",
                model=self.provider.model,
                native_tool_calls=self._capabilities.native_tool_calls,
                detail=self._capabilities.detail,
            )
        return self._capabilities.native_tool_calls

    async def execute_task(self, task: Task) -> TaskResult:
        """Run the tool loop until the model stops calling tools or limits hit."""
        self._active_task = task.id
        # Rebind the window to *this* task: agents are reusable, and a stale
        # window would split the conversation across task ids or hide the
        # task text from the first model call (audit §10).
        self._reset_dedup_state()
        # Recovery-ladder retries reuse the task id, so drop the failed
        # attempt's persisted turns before reopening the window: replaying
        # them appends duplicate TASK turns and the model re-sends its old
        # replies instead of acting (live-run finding).
        self.store.clear_window(self.agent_id, task.id)
        self.context_window = StoreWindow(
            self.store, self.agent_id, task.id, stale_tool_results=self.stale_tool_results
        )
        self.context_window.append("user", compose_task_prompt(task))
        try:
            summary, success, error = await self._loop(task)
        except BudgetExhausted:
            self._record_milestone(task, success=False, note="stopped: token budget")
            return self.governor.exhausted_result(task.id)
        finally:
            self._active_task = None
        if success:
            self._record_milestone(task, success=True, note=summary[:200])
        return TaskResult(task_id=task.id, success=success, summary=summary, error=error)

    async def structured_call(
        self, instruction: str, user_prompt: str, schema_hint: str
    ) -> dict[str, Any]:
        """One JSON-structured model call with a single repair round.

        The Architect/Manager contracts need parsed structures, not prose:
        first ask, and on a malformed reply ask once more with the schema
        restated before giving up (feeds the recovery ladder as evidence).
        Tools are withheld on structured calls: models with tool access
        answer "I need to read the files first" with a tool call instead of
        the required JSON (live-run finding, M4).
        """
        self.context_window.append("user", user_prompt)
        task = self._trace_task()
        self._structured_steps += 1
        self._emit_step("thinking", self._structured_steps, task)
        reply = await self._generate(instruction, use_tools=False)
        self._emit_step("responding", self._structured_steps, task)
        try:
            parsed = extract_json(reply.content)
        except StructuredOutputError:
            repair = (
                f"Your last reply was not valid JSON for the required schema.\n"
                f"Schema: {schema_hint}\nReply with ONLY the JSON object."
            )
            self.context_window.append("user", repair)
            self._structured_steps += 1
            self._emit_step("thinking", self._structured_steps, task)
            reply = await self._generate(instruction, use_tools=False)
            self._emit_step("responding", self._structured_steps, task)
            parsed = extract_json(reply.content)
            self.context_window.append("assistant", "recovered with valid JSON")
        self._compress_window_after_structured()
        return parsed

    def _compress_window_after_structured(self) -> None:
        """Bound a structured-call window's growth across recovery rounds (#63).

        The Architect's planning window survives across recovery rounds (the
        task id is fixed), so without folding, every re-analysis re-sends the
        whole accumulated history. Windows without a task namespace (test
        fakes) are skipped.
        """
        window_task = getattr(self.context_window, "task_id", None)
        if window_task is not None:
            self._maybe_compress(window_task)

    async def _generate(self, instruction: str, use_tools: bool = True) -> ModelResponse:
        self.governor.check()
        self.governor.reserve(_estimate_tokens(self.context_window.as_messages()))
        response = await self.provider.generate(
            [
                {
                    "role": "system",
                    "content": system_prompt(
                        self.role,
                        knowledge=self._persona_knowledge(),
                        extra=instruction,
                    ),
                },
                *self.context_window.as_messages(),
            ],
            self._tool_schemas() if use_tools else None,
        )
        self.governor.record(
            self.agent_id, self.provider.model, response.prompt_tokens, response.completion_tokens
        )
        self._emit_usage(response, self._trace_task())
        self.context_window.append("assistant", response.content or "")
        return response

    async def handle_error(self, error: Exception, task: Task) -> ErrorEscalation:
        """Classify a failure for the recovery ladder (issue 2.11 contract)."""
        self._attempts[task.id] = self._attempts.get(task.id, 0) + 1
        severity = _classify_error(error)
        return ErrorEscalation(
            sender=self.agent_id,
            task_id=task.id,
            severity=severity,
            error_type=type(error).__name__,
            message=str(error)[:500],
            attempt=self._attempts[task.id],
        )

    def report_status(self) -> StatusUpdate:
        status = AgentStatus.WORKING if self._active_task else AgentStatus.IDLE
        return StatusUpdate(sender=self.agent_id, status=status, task_id=self._active_task)

    # -- loop internals ---------------------------------------------------------
    async def _loop(self, task: Task) -> tuple[str, bool, str | None]:
        context = self.store.load_agent_context(self.agent_id, task.id)
        ledger = context.summary if context.summary else ""
        unmarked_finishes = 0
        use_native = await self.native_tool_calls()
        for _step in range(self.max_steps):
            self.governor.check()
            self.governor.reserve(_estimate_tokens(self._messages(task, ledger, use_native)))
            self._last_steps = _step + 1
            self._emit_step("thinking", _step + 1, task.id)
            response = await self.provider.generate(
                self._messages(task, ledger, use_native),
                self._tool_schemas() if use_native else None,
            )
            self.governor.record(
                self.agent_id,
                self.provider.model,
                response.prompt_tokens,
                response.completion_tokens,
            )
            self._emit_step("responding", _step + 1, task.id)
            self._emit_usage(response, task.id)
            content = strip_think_blocks(response.content or "")
            if response.tool_calls:
                calls: list[dict[str, Any]] = [
                    {
                        "id": call.id or f"call_{index + 1}",
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for index, call in enumerate(response.tool_calls)
                ]
                self.context_window.append("assistant", content, tool_calls=calls)
                for call in calls:
                    result = await self._invoke_tool(call["name"], call["arguments"])
                    self.context_window.append(
                        "tool",
                        f"[{call['name']}] {result.output or result.error}",
                        tool_call_id=call["id"],
                        tool_name=call["name"],
                    )
                round_strikes = self._round_dedup_strikes
                self._round_dedup_strikes = []
                if (
                    round_strikes
                    and max(round_strikes) >= REPEAT_STRIKE_THRESHOLD
                    and self._repeat_nudges < MAX_UNMARKED_NUDGES
                ):
                    self._repeat_nudges += 1
                    self.context_window.append("user", _REPEAT_NUDGE)
                continue
            if not use_native and (calls := parse_tool_call_blocks(content)):
                # Text protocol: results go back as user turns so the wire
                # stays valid for models without native tool-role semantics.
                self.context_window.append("assistant", content, tool_calls=calls)
                for call in calls:
                    result = await self._invoke_tool(call["name"], call["arguments"])
                    self.context_window.append(
                        "user",
                        f"TOOL_RESULT ({call['name']}): {result.output or result.error}",
                        tool_call_id=call["id"],
                        tool_name=call["name"],
                    )
                continue
            if FINAL_MARKER in content:
                self._maybe_compress(task.id)
                return content, True, None
            # A reply with neither tool calls nor the marker is the model
            # pausing, not finishing (audit §6): nudge it back to work a
            # bounded number of times, then accept its last word - the
            # verification gates, not the model's word, judge the truth.
            if unmarked_finishes < MAX_UNMARKED_NUDGES:
                unmarked_finishes += 1
                self.context_window.append("user", _NUDGE)
                continue
            self._maybe_compress(task.id)
            return content, True, None
        self._maybe_compress(task.id)
        return _STEP_LIMIT_ERROR, False, _STEP_LIMIT_ERROR

    def _persona_knowledge(self) -> str:
        """Bounded skill card for this role (#78); empty when disabled."""
        if not self.knowledge_enabled:
            return ""
        return knowledge_section(self.role, max_chars=self.knowledge_max_chars)

    def _messages(self, task: Task, ledger: str, use_native: bool = True) -> list[dict[str, Any]]:
        mode_directive = _MODE_DIRECTIVES.get(self.governor.mode(), "")
        tool_manual = "" if use_native else render_tool_manual(self._tool_schemas())
        return [
            {
                "role": "system",
                "content": system_prompt(
                    self.role,
                    fact_ledger=ledger,
                    knowledge=self._persona_knowledge(),
                    extra=(
                        f"Working repo task id: {task.id}. "
                        f"End with {FINAL_MARKER} when done.{mode_directive}"
                        + (f"\n\n{tool_manual}" if tool_manual else "")
                    ),
                ),
            },
            *self.context_window.as_messages(),
        ]

    def _tool_schemas(self) -> list[dict[str, Any]]:
        permitted = self.permitted_tools({"agent_id": self.agent_id, "model_tier": self.model_tier})
        allowed_tier = self.preset.max_tool_tier if self.preset else ToolTier.ADVANCED
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in permitted
            if tool.tier <= allowed_tier
        ]

    async def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        # Cockpit tracing wraps every call (including the fast rejections:
        # a blocked tier-cap attempt is exactly what the operator wants to
        # see in the agent's activity log). Tool crashes are converted to
        # failure ToolResults inside, so `result` always exists.
        started = time.monotonic()
        result = await self._invoke_tool_untraced(name, arguments)
        self._emit_tool(name, arguments, result, (time.monotonic() - started) * 1000)
        return result

    async def _invoke_tool_untraced(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        from harness.tools.registry import TOOL_ALIASES

        tool = next((t for t in self.tools if t.name == name), None)
        # The persona's tier cap is a hard contract, not a schema hint: a
        # read-only locator must not execute apply_edit by merely naming it
        # (live-run finding — the locator "fixed" the bug itself).
        allowed_tier = self.preset.max_tool_tier if self.preset else ToolTier.ADVANCED
        if tool is not None and tool.tier > allowed_tier:
            return ToolResult(
                success=False,
                error=f"tool '{name}' exceeds this persona's tool tier cap "
                f"({allowed_tier.name}); use the tools your role provides",
            )
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None and name in TOOL_ALIASES:
            # Models call tools by natural names (read_file, grep, ...);
            # resolve the registry's canonical instance (live-run finding).
            canonical = TOOL_ALIASES[name]
            tool = next((t for t in self.tools if t.name == canonical), None)
        if tool is None:
            available = ", ".join(t.name for t in self.tools)
            return ToolResult(success=False, error=f"unknown tool '{name}'; available: {available}")
        # The persona's tier cap is a hard contract enforced at EXECUTION time
        # and AFTER alias resolution — otherwise a read-only persona reaches a
        # tier-2 tool through a natural-name alias (55a16f9's fix hardened the
        # exact-name path; this closes the alias path).
        allowed_tier = self.preset.max_tool_tier if self.preset else ToolTier.ADVANCED
        if tool.tier > allowed_tier:
            return ToolResult(
                success=False,
                error=f"tool '{tool.name}' exceeds this persona's tool tier cap "
                f"({allowed_tier.name}); use the tools your role provides",
            )
        if errors := tool.validate_input(arguments):
            return ToolResult(success=False, error=f"invalid arguments: {'; '.join(errors)}")
        context = {"agent_id": self.agent_id, "model_tier": self.model_tier}
        if not tool.check_permissions(context):
            return ToolResult(
                success=False,
                error=f"permission denied for tool '{name}' at tier {self.model_tier}",
            )
        result = await self._dedup_or_execute(tool, arguments)
        self.governor.record(self.agent_id, f"tool:{name}", 0, 0)
        return result

    async def _dedup_or_execute(self, tool: Tool, arguments: dict[str, Any]) -> ToolResult:
        """Cache deterministic tools' identical repeat calls (#62).

        A cache hit returns the stored output with a marker instead of
        re-executing; strikes per key drive the loop nudge in `_loop`. Any
        successful mutating tool invalidates the whole cache - finer-grained
        path tracking is not worth the correctness risk.
        """
        key = (tool.name, json.dumps(arguments, sort_keys=True, default=str))
        if tool.name in DEDUP_TOOLS and key in self._dedup_cache:
            strike = self._dedup_strikes.get(key, 0) + 1
            self._dedup_strikes[key] = strike
            self._round_dedup_strikes.append(strike)
            return ToolResult(
                success=True,
                data={"dedup_hit": True},
                output=(
                    f"{self._dedup_cache[key]}\n"
                    "[dedup: identical call already executed; cached result above - "
                    "change arguments or move on]"
                ),
            )
        result = await _call_tool(tool, arguments)
        if result.success:
            if tool.name in DEDUP_TOOLS:
                self._dedup_cache[key] = result.output
                self._dedup_strikes.pop(key, None)
            elif tool.name in MUTATING_TOOLS:
                self._dedup_cache.clear()
                self._dedup_strikes.clear()
        return result

    def _maybe_compress(self, task_id: str) -> None:
        context = self.store.load_agent_context(self.agent_id, task_id)
        if len(context.recent) >= self.keep_recent:
            folded = self.store.compress(self.agent_id, task_id, keep_recent=self.keep_recent // 2)
            if folded:
                logger.info("context compressed", agent=self.agent_id, folded=folded)

    def _record_milestone(self, task: Task, *, success: bool, note: str) -> None:
        context = self.store.load_agent_context(self.agent_id, task.id)
        context.milestones.append(("OK" if success else "FAIL") + f": {note}")
        self.store.save_agent_context(context)


async def _call_tool(tool: Tool, arguments: dict[str, Any]) -> ToolResult:
    try:
        if isinstance(tool, AsyncExecutableTool):
            return await tool.execute_async(**arguments)
        return tool.execute(**arguments)
    except Exception as exc:
        return ToolResult(success=False, error=f"tool crashed: {exc}")


_DIGEST_PATH_KEYS = ("path", "file", "file_path", "repo_path", "directory")
_DIGEST_COMMAND_KEYS = ("command", "cmd")
_DIGEST_PATTERN_KEYS = ("pattern", "query", "regex")
_DIGEST_NODE_KEYS = ("node_id", "test", "node", "name")


def _args_digest(name: str, arguments: dict[str, Any]) -> str:
    """Short human rendering of one tool call's arguments (never contents).

    Cockpit events are broadcast, so digests carry identifiers (paths,
    patterns, commands) only - a read's payload must never ride an event.
    Search-style tools carry pattern + scope: the pattern wins over the
    scope path (the interesting argument), rendered together.
    """
    for key in _DIGEST_PATTERN_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            scope = arguments.get("path") or arguments.get("directory") or ""
            prefix = f"'{value[:60]}'"
            return (
                f"pattern {prefix} in {str(scope)[:40]}".rstrip() if scope else f"pattern {prefix}"
            )
    for key in _DIGEST_PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value[:120]
    for key in _DIGEST_COMMAND_KEYS:
        value = arguments.get(key)
        if isinstance(value, list) and value:
            return " ".join(str(part) for part in value)[:120]
        if isinstance(value, str) and value:
            return value[:120]
    for key in _DIGEST_NODE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value[:120]
    if arguments:
        first = next(iter(arguments.values()))
        return str(first)[:120]
    return ""


def _result_digest(result: ToolResult) -> str:
    """Truncated outcome line for one tool call (output or error)."""
    text = (result.output or result.error or "").strip().replace("\n", " ")
    return text[:120]


_MODE_DIRECTIVES: dict[GovernorMode, str] = {
    GovernorMode.NORMAL: "",
    GovernorMode.SURGICAL: (
        " BUDGET MODE surgical: no re-planning, minimal exploration; "
        "work from the localization you already have."
    ),
    GovernorMode.FINALIZE: (
        " BUDGET MODE finalize-only: run verification and repair "
        "verified-failing tests only, then finish."
    ),
}


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap chars/4 estimate used for pre-dispatch budget reservation."""
    return sum(len(str(message.get("content") or "")) for message in messages) // 4


def _classify_error(error: Exception) -> Severity:
    """Error classification per DESIGN_SPEC §7.1."""
    if isinstance(error, BudgetExhausted):
        return Severity.FATAL
    if isinstance(error, (TimeoutError, ConnectionError)):
        return Severity.TRANSIENT
    if isinstance(error, (StructuredOutputError, json.JSONDecodeError, ValueError)):
        return Severity.RECOVERABLE
    return Severity.RECOVERABLE
