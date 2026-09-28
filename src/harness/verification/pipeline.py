"""Verification pipeline (milestones 3-4, issues 3.4-3.6 + 4.10).

Six stages per DESIGN_SPEC §9, adapted to the unattended evaluation
environment: Stage 2's "CI/CD" is the local full test suite (no CI runners
exist at eval time; our own repo's CI covers the published-harness case).
Stage 1 is the diff-integrity gate (test-file protection + diff minimality,
M4 #53); when a baseline exists, Stage 3 judges regressions against the
pre-patch suite run and requires a declared reproduction test to pass.
Stages 1/2/3/5 are deterministic; Stage 4 is the AST smell pass
(non-blocking advisories); Stage 6 is the Architect's LLM review verdict.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.agents.architect import ArchitectAgent, Plan, ReviewVerdict
from harness.engine.budget import BudgetExhausted
from harness.security.secret_scanner import block_reason, scan_diff
from harness.tools.editing import SyntaxCheckTool
from harness.tools.execution import RunTestsTool
from harness.verification.baseline import Baseline, parse_failed_tests
from harness.verification.code_review import review_paths
from harness.verification.integrity import classify_diff

Tracer = Callable[[dict[str, Any]], None]
"""Optional event sink (the evidence pack's trace) for live stage events."""


NON_CODE_SPECIALTIES = frozenset(
    {
        "verification",
        "code-review",
        "localization",
        "code-navigation",
        "coordination",
        "architecture",
    }
)
"""Specialties whose subtasks plausibly produce analysis, not edits. Every
other specialty (backend-api, database, frontend, refactoring, security,
devops, documentation, testing, ...) implies a working-tree change; an empty
diff after such a plan is a no-op run, not a pass."""


def plan_requires_changes(plan: Plan | None) -> bool:
    """Does this plan expect the working tree to change at all?

    Mechanical no-op gate: a subtask that declares target files, or carries
    a code-implying specialty, makes an empty diff a FAILURE. Only plans that
    are entirely analysis-flavored (locate/review/verify) accept an empty
    diff as legitimate."""
    if plan is None:
        return False
    return any(
        bool(subtask.files) or subtask.specialty not in NON_CODE_SPECIALTIES
        for subtask in plan.subtasks
    )


def _planned_file_coverage(plan: Plan | None, diff_files: list[str]) -> dict[str, list[str]]:
    """Which subtasks' declared files the diff actually touched.

    Advisory evidence for the final review: the architect compares the plan's
    predicted file set against what really changed instead of trusting the
    implementer's summary."""
    if plan is None:
        return {}
    touched = set(diff_files)
    covered: dict[str, list[str]] = {"addressed": [], "untouched_declared_files": []}
    for subtask in plan.subtasks:
        if subtask.files and not (set(subtask.files) & touched):
            covered["untouched_declared_files"].append(
                f"{subtask.id}: {', '.join(subtask.files[:3])}"
            )
        elif subtask.files:
            covered["addressed"].append(subtask.id)
    return covered


def _git_measurable(root: Path) -> bool:
    """Can `git diff` produce a meaningful answer in this directory?

    Subdirectories of a git repo count (fixtures checked out inside a parent
    repo): `rev-parse --is-inside-work-tree` resolves through parents.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - git missing
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


@dataclass
class StageResult:
    name: str
    passed: bool
    detail: str = ""
    duration_seconds: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    blocking: bool = True


def _reproduction_node(raw: str) -> str:
    """Bare pytest node id: architects emit 'pytest node::id' but also full
    command lines with flags ('pytest test.py -v', 'python -m pytest -k x
    test.py') (#81). Tokenize the command and keep the first path-like
    argument; flags and their values are never paths."""

    def strip_runner(tokens: list[str]) -> list[str]:
        # 'python -m pytest ...' / 'python3 -m pytest ...' / 'pytest ...'
        is_module_run = (
            len(tokens) >= 3
            and tokens[0] in {"python", "python3"}
            and tokens[1] == "-m"
            and tokens[2] == "pytest"
        )
        if is_module_run:
            return tokens[3:]
        if tokens and tokens[0] == "pytest":
            return tokens[1:]
        return tokens

    value_flags = {"-k", "-m", "--tb", "--maxfail", "--junitxml", "--junit-xml", "-p", "--rootdir"}
    tokens = strip_runner([t for t in raw.strip().split() if t])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in value_flags:
            i += 2  # flag plus its value
            continue
        if tok.startswith("-"):
            i += 1  # boolean flag (possibly --opt=value)
            continue
        return tok
    return ""


class VerificationPipeline:
    """Runs the quality gates over one change set and reports a verdict."""

    def __init__(self, repo_root: Path, baseline: Baseline | None = None) -> None:
        self._root = repo_root
        self._baseline = baseline
        self.results: list[StageResult] = []
        self.cancelled = False

    def changed_files(self, diff: str) -> list[str]:
        """Repo-relative files touched by a unified diff."""
        files: list[str] = []
        for line in diff.splitlines():
            if line.startswith("+++ b/"):
                files.append(line[6:])
        return files

    async def run(
        self,
        diff: str,
        plan: Plan | None,
        architect: ArchitectAgent | None = None,
        baseline: Baseline | None = None,
        run_id: str | None = None,
        tracer: Tracer | None = None,
    ) -> list[StageResult]:
        """Execute stages 1-6 in order; stops at the first blocking failure.

        With `run_id` + `tracer`, every stage outcome is also emitted as a
        `verification.stage` event (platform cockpit streaming) and lands in
        the run's trace.jsonl — the evidence pack gains the verification
        timeline, and the dashboard's stage panel becomes live.
        """
        self.results = []
        stages: tuple[tuple[str, Callable[..., Awaitable[StageResult]]], ...] = (
            ("1-integrity", self._stage_integrity),
            ("2-self-check", self._stage_self_check),
            ("3-local-tests", self._stage_local_tests),
            ("4-code-review", self._stage_code_review),
            ("5-security", self._stage_security),
            ("6-final-review", self._stage_final_review),
        )
        for name, stage in stages:
            if self.cancelled:
                result = StageResult(name="cancelled", passed=False, detail="pipeline cancelled")
                self.results.append(result)
                self._emit_stage(tracer, run_id, result)
                break
            try:
                result = await _timed(stage, diff, plan, architect)
            except BudgetExhausted as exc:
                # The budget governor is a hard stop, not a quality verdict:
                # keep every stage that already ran, record the exhaustion as
                # a blocking (unreviewed) outcome, and let the caller finalize
                # the evidence pack. Live-run finding (parse repo): an
                # exhaustion mid-stage-6 used to propagate up, crashing the
                # CLI with a traceback and skipping evidence finalization.
                result = StageResult(
                    name,
                    passed=False,
                    detail=f"skipped: token budget exhausted ({exc})",
                    blocking=True,
                )
            self.results.append(result)
            self._emit_stage(tracer, run_id, result)
            if not result.passed and result.blocking:
                break
        return self.results

    def _emit_stage(self, tracer: Tracer | None, run_id: str | None, result: StageResult) -> None:
        """Trace one stage outcome when the platform wiring is present."""
        if tracer is None or run_id is None:
            return
        tracer(
            {
                "event": "verification.stage",
                "run_id": run_id,
                "stage": result.name,
                "passed": result.passed,
                "blocking": result.blocking,
                "detail": result.detail[:300],
                "duration_seconds": result.duration_seconds,
            }
        )

    # -- stages ---------------------------------------------------------------
    async def _stage_integrity(
        self, diff: str, plan: Plan | None, architect: ArchitectAgent | None
    ) -> StageResult:
        """Diff-integrity gate: test-file protection + diff minimality."""
        allow = bool(plan.allow_test_edits) if plan else False
        report = classify_diff(diff, allow_test_edits=allow)
        detail = "diff clean"
        if report.violations:
            detail = "; ".join(report.violations[:5])
        elif report.warnings:
            detail = f"{len(report.warnings)} diff-minimality warnings"
        return StageResult(
            "1-integrity",
            not report.violations,
            detail,
            evidence={"violations": report.violations, "warnings": report.warnings},
        )

    async def _stage_self_check(
        self, diff: str, plan: Plan | None, architect: ArchitectAgent | None
    ) -> StageResult:
        files = self.changed_files(diff)
        if not files:
            # No-op gate (false-VERIFIED killer): a plan that expects code
            # changes and produces an empty diff has demonstrated nothing,
            # no matter what the agent's summary claims. Analysis-only plans
            # (locate/review/verify) legitimately change nothing, and a
            # non-git target has no measurable diff at all (stated honestly,
            # never silently passed off as "clean").
            if not _git_measurable(self._root):
                return StageResult(
                    "2-self-check", True, "diff not measurable (target is not a git repo)"
                )
            if plan_requires_changes(plan):
                return StageResult(
                    "2-self-check",
                    False,
                    "no changed files: implementation subtasks produced no diff "
                    "(requested change not demonstrated)",
                    evidence={"no_diff": True},
                )
            return StageResult("2-self-check", True, "no changed files (analysis-only plan)")
        tool = SyntaxCheckTool(self._root)
        result = tool.execute(paths=files)
        coverage = _planned_file_coverage(plan, files)
        evidence: dict[str, Any] = {"files": files}
        evidence.update(coverage)
        detail = result.error or result.output
        if coverage.get("untouched_declared_files"):
            detail = (
                (detail + " | " if detail else "")
                + f"{len(coverage['untouched_declared_files'])} subtask(s) with untouched declared files"
            )
        return StageResult("2-self-check", result.success, detail, evidence=evidence)

    async def _stage_local_tests(
        self, diff: str, plan: Plan | None, architect: ArchitectAgent | None
    ) -> StageResult:
        tool = RunTestsTool(self._root)
        result = await tool.execute_async(extra_args=["-rf", "--tb=no"])
        baseline = self._baseline
        detail = result.error or "test suite green"
        evidence: dict[str, Any] = {"output": (result.output or "")[-4000:]}
        passed = result.success

        if baseline is not None:
            if not baseline.runnable:
                detail = (
                    "baseline not runnable; verification degraded to exit-code check "
                    "(honest degradation, improvements §1.5)"
                )
            else:
                failed_now = parse_failed_tests(result.output or "")
                regressions = sorted(
                    failed_now - baseline.failed - {baseline.reproduction_test}
                    if baseline.reproduction_test
                    else failed_now - baseline.failed
                )
                evidence["regressions"] = regressions
                evidence["pre_existing_failures"] = sorted(baseline.failed)
                repro_ok: bool | None = None
                if baseline.reproduction_test:
                    repro = await tool.execute_async(
                        path=_reproduction_node(baseline.reproduction_test),
                        extra_args=["--tb=no"],
                    )
                    repro_ok = repro.success
                    evidence["reproduction_passes_after"] = repro_ok
                    evidence["reproduction_failing_before"] = baseline.reproduction_failing_before
                    evidence["reproduction_output_tail"] = (repro.output or repro.error or "")[
                        -800:
                    ]
                    evidence["reproduction_command_cwd"] = str(tool._root)
                if regressions:
                    passed = False
                    detail = f"baseline regressions: {', '.join(regressions[:5])}"
                elif baseline.reproduction_test and baseline.reproduction_failing_before is False:
                    # Reproduction invariant: a test that already passed at
                    # baseline proves nothing about the patch. VERIFIED
                    # requires FAIL-before -> PASS-after, not PASS -> PASS.
                    passed = False
                    detail = (
                        "reproduction test was already passing at baseline "
                        f"(invalid reproduction evidence): {baseline.reproduction_test}"
                    )
                elif baseline.reproduction_test and not repro_ok:
                    passed = False
                    detail = f"reproduction test still failing: {baseline.reproduction_test}"
                else:
                    passed = True
                    detail = "no regressions vs baseline" + (
                        "; reproduction test passes" if baseline.reproduction_test else ""
                    )
        return StageResult(
            "3-local-tests",
            passed,
            detail,
            evidence=evidence,
        )

    async def _stage_code_review(
        self, diff: str, plan: Plan | None, architect: ArchitectAgent | None
    ) -> StageResult:
        files = [self._root / rel for rel in self.changed_files(diff)]
        findings = review_paths(files, self._root)
        serialized = [
            {"path": f.path, "line": f.line, "kind": f.kind, "detail": f.detail} for f in findings
        ]
        return StageResult(
            "4-code-review",
            True,  # non-blocking advisories
            f"{len(findings)} advisory findings",
            blocking=False,
            evidence={"findings": serialized},
        )

    async def _stage_security(
        self, diff: str, plan: Plan | None, architect: ArchitectAgent | None
    ) -> StageResult:
        findings = scan_diff(diff)
        reason = block_reason(findings)
        return StageResult(
            "5-security",
            reason is None,
            reason or "no secrets in added lines",
            evidence={"finding_count": len(findings)},
        )

    async def _stage_final_review(
        self, diff: str, plan: Plan | None, architect: ArchitectAgent | None
    ) -> StageResult:
        if architect is None or plan is None:
            return StageResult(
                "6-final-review", True, "skipped (no architect/plan supplied)", blocking=False
            )
        verdict: ReviewVerdict = await architect.review(
            diff, plan, evidence=stage_report(self.results)
        )
        evidence: dict[str, Any] = {
            "issues": verdict.issues,
            "criteria_dispositions": [
                {"criterion": d.criterion, "satisfied": d.satisfied, "evidence": d.evidence}
                for d in verdict.criteria_dispositions
            ],
        }
        # Machine-checked acceptance coverage (review finding #4): the LLM's
        # `approved` is necessary but NOT sufficient - every plan criterion
        # must carry an explicit disposition, and every disposition must be
        # satisfied. A verdict that skips criteria fails here even when it
        # says approved.
        expected = {c for st in plan.subtasks for c in st.acceptance_criteria}
        judged = {d.criterion for d in verdict.criteria_dispositions}
        missing = sorted(expected - judged)
        unsatisfied = sorted(d.criterion for d in verdict.criteria_dispositions if not d.satisfied)
        if missing:
            evidence["missing_criteria"] = missing
            return StageResult(
                "6-final-review",
                False,
                "acceptance criteria not individually addressed: " + "; ".join(missing[:5]),
                evidence=evidence,
            )
        if unsatisfied:
            evidence["unsatisfied_criteria"] = unsatisfied
            return StageResult(
                "6-final-review",
                False,
                "acceptance criteria not satisfied: " + "; ".join(unsatisfied[:5]),
                evidence=evidence,
            )
        return StageResult(
            "6-final-review",
            verdict.approved,
            verdict.summary or ("approved" if verdict.approved else "; ".join(verdict.issues)),
            evidence=evidence,
        )


async def _timed(
    stage: Callable[..., Awaitable[StageResult]],
    diff: str,
    plan: Plan | None,
    architect: ArchitectAgent | None,
) -> StageResult:
    started = time.monotonic()
    result = await stage(diff, plan, architect)
    result.duration_seconds = round(time.monotonic() - started, 3)
    return result


def _sanitize_timing(text: str) -> str:
    """Normalize wall-clock noise in test output for review evidence.

    The final-review prompt embeds stage evidence; pytest's 'in 0.03s'
    summary and float durations make those bytes timing-dependent, which
    both flaks the A/B token bench (+/-1 token) and lets irrelevant timing
    digits influence the verdict. Content survives; wall clock does not."""
    import re

    return re.sub(r"\bin \d+(?:\.\d+)?s\b", "in <t>s", text)


def stage_report(results: list[StageResult]) -> str:
    """Markdown summary of a pipeline run (evidence-pack input)."""
    lines = ["| Stage | Result | Detail | Duration |", "|---|---|---|---|"]
    for result in results:
        detail = _sanitize_timing(result.detail)
        lines.append(
            f"| {result.name} | {'PASS' if result.passed else 'FAIL'} "
            f"| {detail[:160]} | {result.duration_seconds:.3f}s |"
        )
    for result in results:
        if result.evidence:
            lines.append("")
            lines.append(f"### {result.name} evidence")
            lines.append("```json")
            lines.append(
                _sanitize_timing(json.dumps(result.evidence, indent=2, sort_keys=True, default=str))
            )
            lines.append("```")
    overall = all(r.passed for r in results if r.blocking)
    lines.append("")
    lines.append(f"**Overall: {'VERIFIED' if overall else 'NOT VERIFIED'}**")
    return "\n".join(lines)
