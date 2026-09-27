"""Reproduction-first baseline (Milestone 4.10; improvements §1.1).

Before any edit, the target repo's suite runs once and the result is
recorded: which tests already fail on unmodified code. Verification then
diffs current results against the baseline - a pre-existing failure can
never be blamed on our patch - and a declared reproduction test must flip
from failing (before) to passing (after): the strongest "evidence over
claims" artifact the evidence pack can carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from harness.tools.execution import RunTestsTool, detect_test_runner


@dataclass
class Baseline:
    """Pre-patch test-suite state of the target repository."""

    runnable: bool
    framework: str = "none"
    all_green: bool = False
    failed: set[str] = field(default_factory=set)
    reproduction_test: str = ""
    reproduction_failing_before: bool | None = None
    output_tail: str = ""

    def to_report(self) -> dict[str, object]:
        """JSON-serializable snapshot for the evidence pack."""
        return {
            "runnable": self.runnable,
            "framework": self.framework,
            "all_green": self.all_green,
            "pre_existing_failures": sorted(self.failed),
            "reproduction_test": self.reproduction_test,
            "reproduction_failing_before": self.reproduction_failing_before,
        }


def parse_failed_tests(output: str) -> set[str]:
    """Extract node ids from pytest's `-rf` short summary lines."""
    failed: set[str] = set()
    for line in output.splitlines():
        if line.startswith("FAILED "):
            node = line.split()[1]
            node = node.split(" - ", 1)[0]
            failed.add(node)
    return failed


async def capture_baseline(
    repo_root: Path, run_tests: RunTestsTool, reproduction_test: str = ""
) -> Baseline:
    """Run the suite once on unmodified code; optionally probe the
    reproduction test (expected to fail before the patch).

    The stored `reproduction_test` is normalized to a bare pytest node id
    (path or path::test): architects routinely emit full command lines such
    as 'pytest test.py -v', and a raw command string would later be passed
    to the runner as ONE argv element ("file or directory not found")."""
    from harness.verification.pipeline import _reproduction_node

    framework, _ = detect_test_runner(repo_root)
    if framework == "none":
        return Baseline(runnable=False, framework="none")

    normalized_repro = _reproduction_node(reproduction_test)
    extra = ["-rf", "--tb=no"] if framework == "pytest" else []
    result = await run_tests.execute_async(extra_args=extra)
    failed = parse_failed_tests(result.output or "") if framework == "pytest" else set()

    reproduction_failing_before: bool | None = None
    if normalized_repro and framework == "pytest":
        repro = await run_tests.execute_async(path=normalized_repro, extra_args=["--tb=no"])
        reproduction_failing_before = not repro.success

    return Baseline(
        runnable=True,
        framework=framework,
        all_green=result.success,
        failed=failed,
        reproduction_test=normalized_repro,
        reproduction_failing_before=reproduction_failing_before,
        output_tail=(result.output or "")[-2000:],
    )
