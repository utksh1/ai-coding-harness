"""Budget governor tests (issue 2.13)."""

from __future__ import annotations

import time

import pytest

from harness.config import BudgetConfig
from harness.engine.budget import (
    BudgetExhausted,
    BudgetGovernor,
    GovernorMode,
    RunDeadlineExceeded,
)
from harness.infrastructure.context_store import MemoryContextStore


@pytest.fixture
def store() -> MemoryContextStore:
    return MemoryContextStore()


def _governor(store: MemoryContextStore, **budget_kwargs: object) -> BudgetGovernor:
    defaults: dict[str, object] = {
        "total_tokens": 1000,
        "warn_fraction": 0.5,
        "surgical_fraction": 0.9,
    }
    defaults.update(budget_kwargs)
    return BudgetGovernor(store, BudgetConfig(**defaults), "corr-1")  # type: ignore[arg-type]


def test_modes_across_thresholds(store) -> None:
    governor = _governor(store)  # warn at 500, finalize at 900, stop at 1000
    governor.record("a-1", "m", 499, 0)
    assert governor.mode() == GovernorMode.NORMAL
    governor.record("a-1", "m", 10, 0)  # 509
    assert governor.mode() == GovernorMode.SURGICAL
    governor.record("a-1", "m", 391, 0)  # 900
    assert governor.mode() == GovernorMode.FINALIZE
    assert governor.used_tokens() == 900


def test_check_raises_at_total(store) -> None:
    governor = _governor(store)
    governor.record("a-1", "m", 999, 1)
    with pytest.raises(BudgetExhausted, match="budget exhausted"):
        governor.check()


def test_check_passes_under_budget(store) -> None:
    _governor(store).check()  # no usage yet


def test_exhausted_result_is_honest(store) -> None:
    governor = _governor(store)
    governor.record("a-1", "m", 100, 1)
    result = governor.exhausted_result("task-1")
    assert result.success is False
    assert "budget exhausted after 101 tokens" in result.error
    assert result.summary == "stopped by token budget governor"
    custom = governor.exhausted_result("task-1", summary="partial work kept")
    assert custom.summary == "partial work kept"


def test_correlation_id_and_isolation(store) -> None:
    governor = _governor(store)
    assert governor.correlation_id == "corr-1"
    governor.record("a-1", "m", 10, 5)
    other = BudgetGovernor(store, BudgetConfig(total_tokens=1000), "corr-2")
    assert other.used_tokens() == 0
    assert governor.used_tokens() == 15


def test_wall_clock_stall_raises_run_deadline(store) -> None:
    """No model progress for wall_clock_seconds -> stall trip (fake clock)."""
    now = {"t": 0.0}
    governor = BudgetGovernor(
        store,
        BudgetConfig(total_tokens=1000),
        "corr-wc",
        wall_clock_seconds=0.01,
        clock=lambda: now["t"],
    )
    assert governor.remaining_seconds() is not None  # absolute rail armed (4x)
    now["t"] = 0.02  # no progress since start -> stall window exceeded
    with pytest.raises(RunDeadlineExceeded, match="wall clock exceeded"):
        governor.check()
    assert "stall limit" in governor.stop_reason
    assert governor.remaining_seconds() == pytest.approx(0.02)  # 0.04 absolute - 0.02
    assert governor.elapsed_seconds() == pytest.approx(0.02)
    assert governor.seconds_since_progress() == pytest.approx(0.02)


def test_progress_refreshes_stall_window(store) -> None:
    """The rail kills starvation, not slowness: recorded model calls keep a
    throttled-but-productive run alive past the stall window (live finding:
    the duration-cap reading killed a real run mid-implementation at ~3
    productive calls/minute under rolling 429 walls)."""
    now = {"t": 0.0}
    governor = BudgetGovernor(
        store,
        BudgetConfig(total_tokens=100_000),
        "corr-live",
        wall_clock_seconds=100.0,
        clock=lambda: now["t"],
    )
    for t, record in [(50.0, True), (140.0, True), (230.0, True)]:
        now["t"] = t
        if record:
            governor.record("a-1", "m", 10, 0)  # refreshes the stall window
        governor.check()  # each gap is 90s < 100s stall -> never trips
    assert governor.stop_reason == ""


def test_duration_cap_trips_despite_progress(store) -> None:
    """Absolute runaway cap: even a fully productive run ends within
    max_duration_seconds (default 4x the stall window)."""
    now = {"t": 0.0}
    governor = BudgetGovernor(
        store,
        BudgetConfig(total_tokens=100_000),
        "corr-dur",
        wall_clock_seconds=100.0,
        clock=lambda: now["t"],
    )
    for t in (100.0, 200.0, 300.0, 390.0):
        now["t"] = t
        governor.record("a-1", "m", 10, 0)  # stall window always fresh
        governor.check()
    now["t"] = 400.0  # 4x stall window since start
    with pytest.raises(RunDeadlineExceeded, match="absolute limit"):
        governor.check()
    assert "total run duration" in governor.stop_reason
    assert governor.remaining_seconds() == 0.0


def test_explicit_duration_override_beats_derivation(store) -> None:
    now = {"t": 0.0}
    governor = BudgetGovernor(
        store,
        BudgetConfig(total_tokens=100_000),
        "corr-ovr",
        wall_clock_seconds=100.0,
        max_duration_seconds=150.0,
        clock=lambda: now["t"],
    )
    now["t"] = 120.0
    governor.record("a-1", "m", 10, 0)  # progress 30s ago: stall is fine
    now["t"] = 150.0
    with pytest.raises(RunDeadlineExceeded, match="150s absolute limit"):
        governor.check()


def test_note_progress_extends_stall_window_without_tokens(store) -> None:
    """Public heartbeat for non-model progress (tool-side activity): keeps
    the run off the stall rail without touching the token ledger."""
    now = {"t": 0.0}
    governor = BudgetGovernor(
        store,
        BudgetConfig(total_tokens=100_000),
        "corr-np",
        wall_clock_seconds=100.0,
        clock=lambda: now["t"],
    )
    now["t"] = 90.0
    governor.note_progress()
    now["t"] = 150.0  # 60s since the heartbeat, still under the 100s stall
    governor.check()
    assert governor.used_tokens() == 0
    assert governor.seconds_since_progress() == pytest.approx(60.0)


def test_run_deadline_is_a_budget_exhausted() -> None:
    # Every existing graceful-stop catch site must keep applying.
    assert issubclass(RunDeadlineExceeded, BudgetExhausted)


def test_no_wall_clock_never_deadlines(store) -> None:
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=1000), "corr-nc")
    assert governor.remaining_seconds() is None
    time.sleep(0.01)
    governor.check()  # token rail only


def test_deadline_checked_before_token_rail(store) -> None:
    now = {"t": 0.0}
    governor = BudgetGovernor(
        store,
        BudgetConfig(total_tokens=1),
        "corr-both",
        wall_clock_seconds=0.01,
        clock=lambda: now["t"],
    )
    governor.record("a-1", "m", 50, 50)  # tokens over cap too; t stays 0.0
    now["t"] = 0.02  # 0.02s without progress -> stall (checked before tokens)
    with pytest.raises(RunDeadlineExceeded):
        governor.check()


def test_exhausted_result_reports_wall_clock_reason(store) -> None:
    now = {"t": 0.0}
    governor = BudgetGovernor(
        store,
        BudgetConfig(total_tokens=1000),
        "corr-res",
        wall_clock_seconds=0.01,
        clock=lambda: now["t"],
    )
    now["t"] = 0.02
    with pytest.raises(RunDeadlineExceeded):
        governor.check()
    result = governor.exhausted_result("task-9")
    assert result.success is False
    assert "wall clock exceeded" in result.error
    assert "stopped by budget governor" in result.summary
    custom = governor.exhausted_result("task-9", summary="kept partial edit")
    assert custom.summary == "kept partial edit"
