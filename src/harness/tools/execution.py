"""Execution tools (milestone 3, issues 3.2-3.3): test runner + sandboxed exec.

Both tools run subprocesses with no shell interpolation, hard timeouts, and
capped output. `CodeExecutionTool` additionally applies resource limits
(CPU seconds, address space) via `resource.setrlimit` where the platform
supports it, and runs with a minimal environment - the eval-mode stand-in
for the spec's Docker sandbox (no containers in the evaluation environment).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from harness.security.input_guard import sanitize_path, validate_command
from harness.security.sandbox import network_egress_violation, sandbox_env
from harness.tools.base import AsyncExecutableTool, Tool, ToolResult, ToolTier

MAX_OUTPUT_BYTES = 20_000
DEFAULT_TIMEOUT = 60.0
SANDBOX_TIMEOUT = 30.0  # spec §13.2: 30 second CPU limit


def _trunc(output: str) -> str:
    if len(output) <= MAX_OUTPUT_BYTES:
        return output
    return output[:MAX_OUTPUT_BYTES] + f"\n... [truncated at {MAX_OUTPUT_BYTES} bytes]"


def _limits_preexec() -> None:  # pragma: no cover - runs in child process
    """Child-side resource caps; failure to apply a cap must not abort the run."""
    import resource

    for limit, value in (
        (resource.RLIMIT_CPU, int(SANDBOX_TIMEOUT)),
        (resource.RLIMIT_AS, 512 * 1024 * 1024),
    ):
        with contextlib.suppress(Exception):
            resource.setrlimit(limit, (value, value))


def detect_test_runner(repo_root: Path) -> tuple[str, list[str]]:
    """Detect (name, command) for the repository's test runner."""
    has_pytest_config = (
        (repo_root / "pyproject.toml").exists()
        or (repo_root / "pytest.ini").exists()
        or (repo_root / "setup.cfg").exists()
    )
    has_bare_tests = any(
        path.name.startswith("test_") or path.name.endswith("_test.py")
        for path in repo_root.rglob("*.py")
        if ".venv" not in path.parts and "__pycache__" not in path.parts
    )
    if has_pytest_config or has_bare_tests:
        return "pytest", [sys.executable, "-m", "pytest", "-q", "--no-header"]
    if (repo_root / "package.json").exists():
        return "npm", ["npm", "test", "--silent"]
    if (repo_root / "Makefile").exists():
        return "make", ["make", "test"]
    return "none", []


class RunTestsTool(AsyncExecutableTool):
    """test_runner: run the detected test suite (or an explicit path subset)."""

    name, tier = "run_tests", ToolTier.DEVELOPMENT
    description = (
        "Run the repository's test suite (auto-detected: pytest/npm/make). "
        "Optional 'path' limits to one test file or directory."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def __init__(self, repo_root: Path, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._root = repo_root
        self._timeout = timeout
        # Parallel-wave safety (review finding #11): concurrent specialists
        # on one working tree must not race test runs (pyc conflicts, fixture
        # clashes, port collisions). One lock per tool instance serializes
        # suite runs; file edits stay parallel (batches are file-disjoint).
        self._run_lock = asyncio.Lock()

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        return []

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return context.get("model_tier", 1) >= self.tier.value

    def execute(self, path: str | None = None, **_: Any) -> ToolResult:
        name, command = detect_test_runner(self._root)
        if name == "none":
            return ToolResult(
                success=False, error="no test runner detected (looked for pytest/npm/make)"
            )
        argv = list(command)
        if path and name == "pytest":
            argv.append(path)
        try:
            proc = subprocess.run(
                argv,
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
                env=sandbox_env(),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"test run exceeded {self._timeout}s timeout",
                data={"framework": name},
            )
        output = _trunc(f"{proc.stdout}\n{proc.stderr}".strip())
        passed = proc.returncode == 0
        return ToolResult(
            success=passed,
            output=output,
            error=None if passed else f"tests failed (exit {proc.returncode})",
            data={"framework": name, "exit_code": proc.returncode},
        )

    async def execute_async(self, path: str | None = None, **_: Any) -> ToolResult:
        """Awaitable variant: subprocess without blocking the event loop."""
        name, command = detect_test_runner(self._root)
        if name == "none":
            return ToolResult(
                success=False, error="no test runner detected (looked for pytest/npm/make)"
            )
        argv = list(command)
        if path and name == "pytest":
            argv.append(path)
        async with self._run_lock:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=self._root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=sandbox_env(),
                )
            except OSError as exc:
                return ToolResult(success=False, error=f"cannot spawn test runner: {exc}")
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), self._timeout)
            except TimeoutError:
                proc.kill()
                return ToolResult(
                    success=False,
                    error=f"test run exceeded {self._timeout}s timeout",
                    data={"framework": name},
                )
        output = _trunc(f"{stdout.decode()}\n{stderr.decode()}".strip())
        passed = proc.returncode == 0
        return ToolResult(
            success=passed,
            output=output,
            error=None if passed else f"tests failed (exit {proc.returncode})",
            data={"framework": name, "exit_code": proc.returncode},
        )


class CodeExecutionTool(AsyncExecutableTool):
    """code_execution: sandboxed command execution (Tier 3).

    No shell, whitelisted argv[0] PLUS network-egress argument checks,
    project-dir confinement, 30s CPU limit, 512 MB memory cap, output
    truncation, and an allowlist child environment with dead proxies.

    HONEST BOUNDARY (review finding #12): this is process-level defense in
    depth, NOT a container. A determined payload can escape it (crafting
    env, spawning processes); the eval host's isolation is the real
    boundary. What this stops is the COMMON case: credentials leaking into
    untrusted test output and accidental network egress.
    """

    name, tier = "code_execution", ToolTier.ADVANCED
    description = (
        "Execute an allowlisted command (argv list) in the repo under "
        "sandbox limits. Args: command (array of strings)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"command": {"type": "array", "items": {"type": "string"}}},
        "required": ["command"],
    }

    ALLOWED_COMMANDS = (
        "python",
        "python3",
        "pytest",
        "node",
        "npm",
        "make",
        "git",
        "pip",
        "grep",
        "ls",
        "cat",
        "ruff",
    )

    def __init__(self, repo_root: Path, timeout: float = SANDBOX_TIMEOUT) -> None:
        self._root = repo_root
        self._timeout = timeout

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        command = arguments.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(arg, str) for arg in command)
        ):
            return ["'command' must be a non-empty array of strings"]
        errors = validate_command(self.ALLOWED_COMMANDS, command)
        # Arg-level egress check (review finding #12): the first-executable
        # allowlist alone misses 'git clone', 'pip install requests', or
        # fetchers hidden deeper in argv.
        violation = network_egress_violation(command, allow_network=False)
        if violation:
            errors = [*errors, violation]
        return errors

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return context.get("model_tier", 1) >= self.tier.value

    def execute(self, command: list[str], **_: Any) -> ToolResult:
        argv = list(command)
        if argv[0] in ("python", "python3"):
            # Pin the harness's own interpreter: PATH lookup would silently
            # pick a different Python (or none) in locked-down environments.
            argv[0] = sys.executable
        try:
            proc = subprocess.run(
                argv,
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
                preexec_fn=_limits_preexec if os.name == "posix" else None,
                env=sandbox_env(allow_network=False, extra={"LANG": "C"}),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, error=f"command exceeded {self._timeout}s sandbox timeout"
            )
        output = _trunc(f"{proc.stdout}\n{proc.stderr}".strip())
        return ToolResult(
            success=proc.returncode == 0,
            output=output,
            error=None if proc.returncode == 0 else f"exit {proc.returncode}",
            data={"exit_code": proc.returncode, "confined_to": str(self._root)},
        )

    async def execute_async(self, command: list[str], **_: Any) -> ToolResult:
        """Awaitable variant: sandboxed subprocess without blocking the loop."""
        argv = list(command)
        if argv and argv[0] in ("python", "python3"):
            argv[0] = sys.executable
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_limits_preexec if os.name == "posix" else None,
                env=sandbox_env(allow_network=False, extra={"LANG": "C"}),
            )
        except OSError as exc:
            return ToolResult(success=False, error=f"cannot spawn command: {exc}")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), self._timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return ToolResult(
                success=False, error=f"command exceeded {self._timeout}s sandbox timeout"
            )
        output = _trunc(f"{stdout.decode()}\n{stderr.decode()}".strip())
        return ToolResult(
            success=proc.returncode == 0,
            output=output,
            error=None if proc.returncode == 0 else f"exit {proc.returncode}",
            data={"exit_code": proc.returncode, "confined_to": str(self._root)},
        )


class SecurityScanTool(Tool):
    """security_scan: secret + vulnerability pattern scan (Tier 3)."""

    name, tier = "security_scan", ToolTier.ADVANCED
    description = (
        "Scan repo files (or one path) for secrets and dangerous patterns. Args: path (optional)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root
        from harness.security.secret_scanner import scan_path

        self._scan_path = scan_path

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        return []

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return context.get("model_tier", 1) >= self.tier.value

    def execute(self, path: str = ".", **_: Any) -> ToolResult:
        try:
            root = sanitize_path(self._root, path)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        if not root.exists():
            return ToolResult(success=False, error=f"path does not exist: {path}")
        findings = self._scan_path(root)
        if findings:
            report = "\n".join(f"{f.path}:{f.line}: {f.kind}" for f in findings[:40])
            return ToolResult(
                success=False,
                output=report,
                error=f"{len(findings)} security findings",
                data={"findings": [f.__dict__ for f in findings[:40]]},
            )
        return ToolResult(
            success=True, output="no secrets or dangerous patterns found", data={"findings": 0}
        )
