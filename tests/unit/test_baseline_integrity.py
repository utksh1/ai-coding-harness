"""Milestone 4.10: reproduction-first baseline + test-integrity guard.

The fail->pass proof: a declared reproduction test must fail before the
patch and pass after; pre-existing failures are exempt from regression
blame; test-file edits without plan allowance block the run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.agents.architect import Plan, SubTask
from harness.tools.execution import RunTestsTool
from harness.verification.baseline import Baseline, capture_baseline, parse_failed_tests
from harness.verification.integrity import changed_files, classify_diff, is_test_file
from harness.verification.pipeline import VerificationPipeline

PLAN = Plan.model_validate(
    {
        "issue_summary": "greet",
        "complexity": 2,
        "subtasks": [
            SubTask(
                id="st-1",
                title="greet",
                description="",
                files=["app.py"],
                acceptance_criteria=["greet returns hello"],
            )
        ],
    }
)


@pytest.fixture
def repro_repo(tmp_path: Path) -> Path:
    """A repo whose suite has one pre-existing failure and one target bug."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "app.py").write_text("def greet():\n    return ''\n")  # bug: not 'hello'
    (tmp_path / "test_greet.py").write_text(
        "from app import greet\n\n\ndef test_greet():\n    assert greet() == 'hello'\n"
    )
    (tmp_path / "test_legacy.py").write_text(
        "def test_legacy():\n    assert False\n"
    )  # pre-existing
    return tmp_path


def test_parse_failed_tests_extracts_node_ids() -> None:
    output = (
        "FAILED tests/test_a.py::test_one - assert False\n"
        "FAILED tests/test_b.py::test_two::nested - AssertionError: boom\n"
        "1 failed, 2 passed"
    )
    assert parse_failed_tests(output) == {
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two::nested",
    }


def test_parse_failed_tests_empty() -> None:
    assert parse_failed_tests("3 passed in 0.01s") == set()


def test_is_test_file_patterns() -> None:
    assert is_test_file("tests/test_app.py")
    assert is_test_file("src/test_app.py")
    assert is_test_file("pkg/app_test.py")
    assert is_test_file("tests/conftest.py")
    assert is_test_file("ui/button.test.js")
    assert is_test_file("api/user.spec.ts")
    assert not is_test_file("src/app.py")
    assert not is_test_file("testing_utils.py")


def test_changed_files_from_diff() -> None:
    diff = "diff --git a/app.py\n--- a/app.py\n+++ b/app.py\n+x\n--- /dev/null\n+++ /dev/null\n"
    assert changed_files(diff) == ["app.py"]


def test_classify_diff_flags_test_edits_and_debug() -> None:
    diff = (
        "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n"
        "+print('debug')\n"
        "diff --git a/tests/test_app.py b/tests/test_app.py\n--- a/tests/test_app.py\n"
        "+++ b/tests/test_app.py\n+assert True\n"
    )
    report = classify_diff(diff)
    assert report.violations == ["test file modified without plan allowance: tests/test_app.py"]
    assert any("debug output added" in warning for warning in report.warnings)


def test_classify_diff_allows_test_edits_when_planned() -> None:
    diff = "diff --git a/tests/test_app.py\n+++ b/tests/test_app.py\n+assert True\n"
    report = classify_diff(diff, allow_test_edits=True)
    assert report.violations == []


def test_classify_diff_clean() -> None:
    diff = "diff --git a/src/app.py\n+++ b/src/app.py\n+return 1\n"
    report = classify_diff(diff)
    assert report.violations == [] and report.warnings == []


async def test_capture_baseline_records_pre_existing_failures(repro_repo: Path) -> None:
    tool = RunTestsTool(repro_repo)
    baseline = await capture_baseline(repro_repo, tool, reproduction_test="test_greet.py")
    assert baseline.runnable
    assert baseline.framework == "pytest"
    assert baseline.failed == {"test_legacy.py::test_legacy", "test_greet.py::test_greet"}
    assert baseline.reproduction_failing_before is True
    report = baseline.to_report()
    assert report["pre_existing_failures"] == [
        "test_greet.py::test_greet",
        "test_legacy.py::test_legacy",
    ]
    assert report["reproduction_failing_before"] is True


async def test_capture_baseline_not_runnable(tmp_path: Path) -> None:
    baseline = await capture_baseline(tmp_path, RunTestsTool(tmp_path))
    assert not baseline.runnable
    assert baseline.to_report()["framework"] == "none"


async def test_baseline_rejects_regression_and_requires_repro_flip(repro_repo: Path) -> None:
    """The core fail->pass contract, judged against the real baseline."""
    tool = RunTestsTool(repro_repo)
    baseline = await capture_baseline(repro_repo, tool, reproduction_test="test_greet.py")

    # Patch NOT applied: repro still failing -> verification fails honestly.
    verification = VerificationPipeline(repro_repo, baseline)
    diff = ""  # unchanged working tree
    results = await verification.run(diff, PLAN, architect=None)
    by_name = {r.name: r for r in results}
    assert not by_name["3-local-tests"].passed
    assert "reproduction test still failing" in by_name["3-local-tests"].detail


async def test_baseline_passes_when_bug_fixed(repro_repo: Path) -> None:
    """After the fix: repro flips to pass, pre-existing failure is exempt."""
    (repro_repo / "app.py").write_text("def greet():\n    return 'hello'\n")
    tool = RunTestsTool(repro_repo)
    baseline = await capture_baseline(repro_repo, tool, reproduction_test="test_greet.py")

    # Recapture after fix would flip repro_failing_before; simulate instead:
    # verification must judge the *current* run against the *pre-patch* baseline.
    verification = VerificationPipeline(repro_repo, baseline)
    diff = "diff --git a/app.py\n+++ b/app.py\n+hello\n"
    results = await verification.run(diff, PLAN, architect=None)
    by_name = {r.name: r for r in results}
    tests = by_name["3-local-tests"]
    # test_legacy still fails (pre-existing) but is exempt; repro passes now.
    assert tests.evidence["reproduction_passes_after"] is True
    assert tests.evidence["regressions"] == []
    assert tests.passed
    assert "no regressions vs baseline" in tests.detail


async def test_capture_baseline_normalizes_command_style_repro(repro_repo: Path) -> None:
    """Regression (live Claude run 2026-09-27): the architect emitted a full
    command 'pytest test_greet.py -v'; the raw remainder 'test_greet.py -v'
    used to reach pytest as ONE argv element -> 'file or directory not
    found' -> the gate failed a correct patch. The baseline must store the
    normalized node id and probe the real test."""
    tool = RunTestsTool(repro_repo)
    baseline = await capture_baseline(repro_repo, tool, reproduction_test="pytest test_greet.py -v")
    assert baseline.reproduction_test == "test_greet.py"
    assert baseline.reproduction_failing_before is True


async def test_baseline_passes_when_bug_fixed_with_command_style_repro(
    repro_repo: Path,
) -> None:
    """The live-run regression: correct fix + command-style repro command
    must verify PASS (repro file exists, flips to green, no regressions)."""
    (repro_repo / "app.py").write_text("def greet():\n    return 'hello'\n")
    tool = RunTestsTool(repro_repo)
    baseline = await capture_baseline(repro_repo, tool, reproduction_test="pytest test_greet.py -v")
    verification = VerificationPipeline(repro_repo, baseline)
    diff = "diff --git a/app.py\n+++ b/app.py\n+hello\n"
    results = await verification.run(diff, PLAN, architect=None)
    tests = {r.name: r for r in results}["3-local-tests"]
    assert tests.evidence["reproduction_passes_after"] is True
    assert "file or directory not found" not in tests.evidence["reproduction_output_tail"]
    assert tests.passed
    assert "no regressions vs baseline" in tests.detail


async def test_baseline_flags_new_regression(repro_repo: Path) -> None:
    """A newly broken test is a regression even with pre-existing failures."""
    tool = RunTestsTool(repro_repo)
    baseline = await capture_baseline(repro_repo, tool)
    # Simulate the patch breaking a previously-green test.
    (repro_repo / "test_fresh.py").write_text("def test_fresh():\n    assert False\n")
    verification = VerificationPipeline(repro_repo, baseline)
    results = await verification.run(
        "diff --git a/app.py\n+++ b/app.py\n+1\n", PLAN, architect=None
    )
    by_name = {r.name: r for r in results}
    tests = by_name["3-local-tests"]
    assert not tests.passed
    assert "test_fresh.py::test_fresh" in tests.detail


async def test_integrity_stage_blocks_test_edits(repro_repo: Path) -> None:
    """Modifying a *tracked* test file is the tamper case: blocked."""
    verification = VerificationPipeline(repro_repo)
    diff = (
        "diff --git a/tests/test_greet.py b/tests/test_greet.py\n"
        "--- a/tests/test_greet.py\n+++ b/tests/test_greet.py\n"
        "-assert greet() == 'hello'\n+assert True\n"
    )
    results = await verification.run(diff, PLAN, architect=None)
    assert results[0].name == "1-integrity"
    assert not results[0].passed
    assert "test file modified" in results[0].detail
    # blocking failure stopped the run
    assert [r.name for r in results][-1] == "1-integrity"


async def test_integrity_stage_allows_new_test_creation(repro_repo: Path) -> None:
    """Greenfield authoring (new test file from /dev/null) is allowed with a
    warning — creation is the job, tampering is the crime."""
    verification = VerificationPipeline(repro_repo)
    diff = (
        "diff --git a/tests/test_new.py b/tests/test_new.py\n"
        "--- /dev/null\n+++ b/tests/test_new.py\n"
        "+def test_new():\n+    assert True\n"
    )
    results = await verification.run(diff, PLAN, architect=None)
    integrity = results[0]
    assert integrity.passed
    assert any("new test file(s) authored" in w for w in integrity.evidence["warnings"])


async def test_integrity_stage_allows_planned_test_edits(repro_repo: Path) -> None:
    plan = PLAN.model_copy(update={"allow_test_edits": True})
    verification = VerificationPipeline(repro_repo)
    diff = "diff --git a/tests/test_greet.py\n+++ b/tests/test_greet.py\n+assert True\n"
    results = await verification.run(diff, plan, architect=None)
    assert results[0].name == "1-integrity" and results[0].passed


async def test_degraded_baseline_reports_honestly(tmp_path: Path) -> None:
    """No runnable suite: verification degrades to exit-code judgment."""
    (tmp_path / "loose_file.txt").write_text("not a repo")
    baseline = Baseline(runnable=False, framework="none")
    verification = VerificationPipeline(tmp_path, baseline)
    results = await verification.run("", PLAN, architect=None)
    tests = next(r for r in results if r.name == "3-local-tests")
    assert "baseline not runnable" in tests.detail


async def test_integrity_stage_warns_on_debug_prints(repro_repo: Path) -> None:
    """Non-blocking: debug additions become warnings, gate still passes."""
    verification = VerificationPipeline(repro_repo)
    diff = "diff --git a/app.py\n+++ b/app.py\n+print('debugging the greet bug')\n"
    results = await verification.run(diff, PLAN, architect=None)
    integrity = results[0]
    assert integrity.passed
    assert "diff-minimality warnings" in integrity.detail
    assert integrity.evidence["warnings"]
