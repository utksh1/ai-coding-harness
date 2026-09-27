"""Token budget governor (milestone 2, issue 2.13 - resource management).

Meters every model call through the context store's usage ledger and exposes
the operating mode the rest of the harness must respect:

- below `warn_fraction`      -> NORMAL
- at/above `warn_fraction`   -> SURGICAL (no re-plans, minimal exploration)
- at/above `surgical_fraction` -> FINALIZE (verify + fix failing tests only)
- at/above `total_tokens`    -> `BudgetExhausted` (stop honestly)
"""

from __future__ import annotations

import time
from enum import StrEnum

from harness.agents.task import TaskResult
from harness.config import BudgetConfig
from harness.infrastructure.context_store import ContextStore


class BudgetExhausted(RuntimeError):  # noqa: N818 - domain state name, not an error suffix case
    """Raised when the run's token budget is spent; callers finalize gracefully."""


class RunDeadlineExceeded(BudgetExhausted):
    """Wall-clock deadline exceeded (`run.wall_clock_seconds`).

    Subclasses `BudgetExhausted` so every existing graceful-stop catch site
    (agent loop, recovery ladder, pipeline finalize) applies unchanged; the
    message distinguishes the two stop reasons.
    """


class GovernorMode(StrEnum):
    NORMAL = "normal"
    SURGICAL = "surgical"
    FINALIZE = "finalize"


class BudgetGovernor:
    """Per-run token meter and wall-clock rail backed by the context-store usage ledger."""

    def __init__(
        self,
        store: ContextStore,
        budget: BudgetConfig,
        correlation_id: str,
        wall_clock_seconds: float | None = None,
    ) -> None:
        self._store = store
        self._budget = budget
        self._correlation_id = correlation_id
        self._wall_clock_seconds = wall_clock_seconds
        self._started_at = time.monotonic()
        self._deadline = (
            self._started_at + wall_clock_seconds if wall_clock_seconds is not None else None
        )
        self._stop_reason = ""

    @property
    def correlation_id(self) -> str:
        return self._correlation_id

    @property
    def stop_reason(self) -> str:
        """Human-readable reason set by whichever rail tripped first, if any."""
        return self._stop_reason

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def remaining_seconds(self) -> float | None:
        return None if self._deadline is None else max(0.0, self._deadline - time.monotonic())

    def used_tokens(self) -> int:
        return self._store.token_usage(self._correlation_id).total_tokens

    def mode(self) -> GovernorMode:
        """Current operating mode derived from spend vs configured thresholds."""
        warn, surgical, _ = self._budget.thresholds()
        used = self.used_tokens()
        if used >= surgical:
            return GovernorMode.FINALIZE
        if used >= warn:
            return GovernorMode.SURGICAL
        return GovernorMode.NORMAL

    def check(self) -> None:
        """Raise when the run must stop. Cheap; call per step.

        Two independent rails: the wall clock (`run.wall_clock_seconds`,
        enforced since a throttled provider can starve the run without ever
        spending tokens) and the token cap. Whichever trips first sets
        `stop_reason` so the honest-failure path can report it.
        """
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self._stop_reason = (
                f"wall clock exceeded: {self.elapsed_seconds():.0f}s >= "
                f"{self._wall_clock_seconds:.0f}s limit"
            )
            raise RunDeadlineExceeded(self._stop_reason)
        if self.used_tokens() >= self._budget.total_tokens:
            self._stop_reason = (
                f"token budget exhausted: {self.used_tokens()} >= {self._budget.total_tokens}"
            )
            raise BudgetExhausted(self._stop_reason)

    def reserve(self, prompt_estimate: int, completion_reserve: int = 1024) -> None:
        """Refuse a dispatch whose predicted cost would overshoot the cap.

        `check()` alone lets the final call burn past the limit because usage
        is recorded only after completion; reserving the estimated prompt
        plus a completion floor before dispatch closes that overshoot
        (audit §13).
        """
        projected = self.used_tokens() + max(0, prompt_estimate) + completion_reserve
        if projected > self._budget.total_tokens:
            msg = f"token budget exhausted: projected {projected} >= {self._budget.total_tokens}"
            raise BudgetExhausted(msg)

    def record(self, agent_id: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Append one model call to the ledger."""
        self._store.record_token_usage(
            self._correlation_id, agent_id, model, prompt_tokens, completion_tokens
        )

    def exhausted_result(self, task_id: str, summary: str = "") -> TaskResult:
        """Honest failure result for a task stopped by the budget governor."""
        reason = self._stop_reason or f"budget exhausted after {self.used_tokens()} tokens"
        fallback_summary = (
            f"stopped by budget governor: {reason}"
            if self._stop_reason
            else "stopped by token budget governor"
        )
        return TaskResult(
            task_id=task_id,
            success=False,
            summary=summary or fallback_summary,
            error=reason,
        )
