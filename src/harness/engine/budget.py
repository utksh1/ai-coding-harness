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
from collections.abc import Callable
from enum import StrEnum

from harness.agents.task import TaskResult
from harness.config import BudgetConfig
from harness.infrastructure.context_store import ContextStore


class BudgetExhausted(RuntimeError):  # noqa: N818 - domain state name, not an error suffix case
    """Raised when the run's token budget is spent; callers finalize gracefully."""


class RunDeadlineExceeded(BudgetExhausted):
    """Wall-clock rail tripped (`run.wall_clock_seconds` / `run.max_duration_seconds`).

    The wall clock is a STALL window: `wall_clock_seconds` is the maximum
    time without recorded model progress (live finding: a throttled provider
    can starve a run for hours without spending tokens - waiting in retry
    backoff is exactly the "productive"-looking hang this rail exists to
    kill). `max_duration_seconds` (default 4x the stall window) is the
    absolute runaway cap from run start, so a run that keeps making real
    progress still ends within a bounded multiple of the operator's dial.

    Subclasses `BudgetExhausted` so every existing graceful-stop catch site
    (agent loop, recovery ladder, pipeline finalize) applies unchanged; the
    message distinguishes which of the two wall clocks tripped.
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
        max_duration_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._budget = budget
        self._correlation_id = correlation_id
        self._clock = clock
        self._wall_clock_seconds = wall_clock_seconds
        self._started_at = self._clock()
        self._last_progress_at = self._started_at
        # Absolute runaway cap; when unset it derives from the stall window
        # so a single dial still bounds total run length (4x headroom: a
        # genuinely productive but throttled run may legitimately outlast
        # the stall window many times over - measured live at ~3 model
        # calls/minute under upstream 429 throttling).
        if max_duration_seconds is None and wall_clock_seconds is not None:
            max_duration_seconds = 4.0 * wall_clock_seconds
        self._max_duration_seconds = max_duration_seconds
        self._deadline = (
            self._started_at + max_duration_seconds if max_duration_seconds is not None else None
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
        return self._clock() - self._started_at

    def remaining_seconds(self) -> float | None:
        """Seconds left on the ABSOLUTE (duration) rail, if one is armed."""
        return None if self._deadline is None else max(0.0, self._deadline - self._clock())

    def seconds_since_progress(self) -> float:
        """Seconds since the last recorded model call (stall-rail distance)."""
        return self._clock() - self._last_progress_at

    def note_progress(self) -> None:
        """Refresh the stall window (called on every recorded model call)."""
        self._last_progress_at = self._clock()

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

        Three rails, checked cheapest-deadliest first: the absolute duration
        cap, the stall window (no model progress for `wall_clock_seconds`
        - enforced since a throttled provider can starve the run in retry
        backoff without ever spending tokens), and the token cap. Whichever
        trips first sets `stop_reason` so the honest-failure path can report
        it. Both wall-clock trips raise `RunDeadlineExceeded` with a
        "wall clock exceeded:" prefix that says WHICH clock tripped.
        """
        now = self._clock()
        if self._deadline is not None and now >= self._deadline:
            self._stop_reason = (
                f"wall clock exceeded: total run duration {self.elapsed_seconds():.0f}s "
                f">= {self._max_duration_seconds:.0f}s absolute limit"
            )
            raise RunDeadlineExceeded(self._stop_reason)
        if (
            self._wall_clock_seconds is not None
            and now - self._last_progress_at >= self._wall_clock_seconds
        ):
            self._stop_reason = (
                f"wall clock exceeded: no model progress for "
                f"{now - self._last_progress_at:.0f}s "
                f"(stall limit {self._wall_clock_seconds:.0f}s)"
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
        """Append one model call to the ledger (and refresh the stall window)."""
        self._store.record_token_usage(
            self._correlation_id, agent_id, model, prompt_tokens, completion_tokens
        )
        self._last_progress_at = self._clock()

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
