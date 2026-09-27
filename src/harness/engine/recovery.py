"""Recovery ladder (milestone 2, issue 2.11 - error recovery & escalation).

Implements the unattended variant of the DESIGN_SPEC §7 hierarchy:

- L1 self-repair: retry the same specialist with its own error evidence
  (`self_repair` attempts),
- L2 manager intervention: the Manager categorizes and adds routing guidance
  (`re_route` attempts),
- L3 architect re-plan: the Architect reframes the task smaller and clearer
  (single attempt),
- L4 graceful failure: an honest `TaskResult(success=False)` - never a hang.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from harness.agents.task import Task, TaskResult
from harness.engine.budget import BudgetExhausted, GovernorMode
from harness.infrastructure.context_store import ContextStore
from harness.infrastructure.logging import get_logger
from harness.orchestration.messages import ErrorEscalation, Severity

logger = get_logger(__name__)

Executor = Callable[[Task], Awaitable[TaskResult]]
Classifier = Callable[[Task, TaskResult], Awaitable[ErrorEscalation]]
Rerouter = Callable[[Task, ErrorEscalation, str], Executor | None]
"""Given (task, escalation, manager guidance detail), return a replacement
executor for a genuine re-route, or None to retry with guidance only."""
Tracer = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class AttemptPolicy:
    self_repair: int = 3
    re_route: int = 2
    re_plan: int = 1


class RecoveryLadder:
    """Drives one task through the graduated recovery levels.

    Evidence reaches execution: every retry carries the previous failure in
    its task metadata (the specialist prompt includes it), and L2 can
    genuinely re-route to a different executor via the `reroute` callback
    instead of merely relabeling the same retry (audit §8).
    """

    def __init__(
        self,
        manager: Any,
        architect: Any,
        store: ContextStore,
        policy: AttemptPolicy | None = None,
        *,
        reroute: Rerouter | None = None,
        on_event: Tracer | None = None,
        governor: Any | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        self._manager = manager
        self._architect = architect
        self._store = store
        self._policy = policy or AttemptPolicy()
        self._reroute = reroute
        self._on_event = on_event
        self._governor = governor
        self._run_id = run_id
        self._agent_id = agent_id
        #: The agent object behind the LAST executor invoked (bound methods
        #: carry their owner): the pipeline reads it to attribute the final
        #: ``specialist.result`` to the agent that actually finished the work
        #: (a collaborator may have taken over at L2).
        self.last_executor_agent: Any | None = None

    def _trace(self, event: dict[str, Any]) -> None:
        if self._on_event is not None:
            payload = {**event, "run_id": self._run_id}
            # L1/L2 belong to the working specialist; L3 is the Architect's.
            if self._agent_id and str(payload.get("event", "")).startswith(
                ("recovery.l1", "recovery.l2")
            ):
                payload.setdefault("agent", self._agent_id)
            self._on_event(payload)

    async def run(self, task: Task, executor: Executor, classify: Classifier) -> TaskResult:
        """Execute `task`, escalating through L1-L3 before giving up gracefully."""
        last_result: TaskResult | None = None
        last_escalation: ErrorEscalation | None = None

        # L1: self-repair - the same specialist, but with its own failure
        # evidence injected so the retry is informed, not blind.
        for attempt in range(self._policy.self_repair):
            last_result = await self._attempt(task, executor)
            if last_result.success:
                return last_result
            last_escalation = await classify(task, last_result)
            if last_escalation.severity == Severity.FATAL:
                return self._give_up(task, last_result, "fatal failure")
            self._trace(
                {
                    "event": "recovery.l1_retry",
                    "task": task.id,
                    "attempt": attempt + 1,
                    "error_type": last_escalation.error_type,
                }
            )
            task = self._with_failure_evidence(task, last_escalation, attempt + 1)

        # L2: manager intervention - guidance reaches the task, and the
        # reroute callback may genuinely swap the executor.
        current_executor = executor
        for _ in range(self._policy.re_route):
            if last_escalation is None:  # no L1 attempt ran (empty policy)
                break
            guidance = await self._manager.handle_escalation(last_escalation)
            if "escalate to architect" in guidance.detail:
                break
            task = task.model_copy(
                update={
                    "metadata": {**task.metadata, "guidance": guidance.detail},
                }
            )
            replacement = (
                self._reroute(task, last_escalation, guidance.detail) if self._reroute else None
            )
            if replacement is not None:
                current_executor = replacement
                self._trace(
                    {
                        "event": "recovery.l2_reroute",
                        "task": task.id,
                        "guidance": guidance.detail[:200],
                    }
                )
            else:
                self._trace(
                    {
                        "event": "recovery.l2_guidance",
                        "task": task.id,
                        "guidance": guidance.detail[:200],
                    }
                )
            last_result = await self._attempt(task, current_executor)
            if last_result.success:
                return last_result
            last_escalation = await classify(task, last_result)

        # L3: architect reframes the task once - unless the budget governor
        # has left NORMAL (surgical mode forbids re-plans, audit §13).
        if (
            self._governor is not None
            and self._governor.mode() is not GovernorMode.NORMAL
            and self._policy.re_plan
        ):
            self._trace({"event": "recovery.l3_skipped_budget", "task": task.id})
            return self._give_up(task, last_result, "re-planning suppressed by budget governor")
        for _ in range(self._policy.re_plan):
            if last_escalation is not None:
                task = await self._architect.reframe(
                    task, f"{last_escalation.error_type}: {last_escalation.message}"
                )
            self._trace({"event": "recovery.l3_replan", "task": task.id})
            last_result = await self._attempt(task, current_executor)
            if last_result.success:
                return last_result
            last_escalation = await classify(task, last_result)

        return self._give_up(task, last_result, "all recovery levels exhausted")

    @staticmethod
    def _with_failure_evidence(task: Task, escalation: ErrorEscalation, attempt: int) -> Task:
        """Attach the failure to the task so the next attempt sees it."""
        evidence = (
            f"Previous attempt {attempt} failed ({escalation.error_type}): "
            f"{escalation.message[:300]}. Address that specific failure."
        )
        recovery = {
            "attempt": attempt,
            "error_type": escalation.error_type,
            "message": escalation.message[:400],
        }
        return task.model_copy(
            update={
                "metadata": {
                    **task.metadata,
                    "guidance": evidence,
                    "recovery": {**task.metadata.get("recovery", {}), **recovery},
                }
            }
        )

    async def _attempt(self, task: Task, executor: Executor) -> TaskResult:
        self.last_executor_agent = getattr(executor, "__self__", None)
        try:
            return await executor(task)
        except BudgetExhausted:
            raise
        except Exception as exc:
            # Prefix the original type: the pipeline classifier recovers it so
            # the Manager's categorization sees KeyError/AttributeError etc.
            logger.warning("executor raised", task=task.id, error=str(exc)[:200])
            return TaskResult(
                task_id=task.id, success=False, error=f"{type(exc).__name__}: {exc}"[:500]
            )

    def _give_up(self, task: Task, result: TaskResult | None, reason: str) -> TaskResult:
        summary = result.summary if result else ""
        error = result.error if result else "unknown failure"
        logger.warning("task failed after recovery", task=task.id, reason=reason)
        return TaskResult(
            task_id=task.id,
            success=False,
            summary=summary or f"gave up: {reason}",
            error=f"{reason}: {error}",
        )
