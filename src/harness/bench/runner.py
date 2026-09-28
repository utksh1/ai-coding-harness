"""One offline measurement: build the fixture, run the pipeline, read the spend.

This module is executed as a standalone script (`python runner.py`) with
`PYTHONPATH` pointed at whichever code version is being measured - current
tree or a baseline worktree - so both sides run through their own real
pipeline. The fixture path is fixed (same name under the system temp dir) so
the deterministic token estimator sees byte-identical prompts across runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

FIXTURE_DIR_NAME = "harness-bench-fixture"
FIXTURE_ISSUE = "app.greet() should return 'hello' when called"
PROFILE_JSON = (
    '{"languages": ["Python"], "frameworks": ["pytest"], '
    '"test_framework": "pytest", "build_system": "pyproject.toml", '
    '"conventions": ["typed"], "notes": "tiny demo repo"}'
)
PLAN_JSON = (
    '{"issue_summary": "greeting missing", "complexity": 3, "subtasks": [{'
    '"id": "st-1", "title": "fix the greeting", '
    '"description": "app.greet() should return hello", '
    '"specialty": "refactoring", "complexity": 2, "files": ["app.py"], '
    '"acceptance_criteria": ["greet returns hello"], "depends_on": []}], '
    '"risks": [], "needs_collaboration": false}'
)
VERDICT_JSON = (
    '{"approved": true, "issues": [], "summary": "greet works", '
    '"criteria_dispositions": [{"criterion": "greet returns hello", '
    '"satisfied": true, "evidence": "diff changes greet to return hello"}]}'
)


def fixture_root() -> Path:
    """Stable per-machine fixture location (fixed name => stable prompt bytes)."""
    return Path(tempfile.gettempdir()) / FIXTURE_DIR_NAME


def build_fixture(root: Path | None = None) -> Path:
    """Create the deterministic offline target: a green pytest repo under git."""
    root = root or fixture_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (root / "app.py").write_text("def greet():\n    return ''\n", encoding="utf-8")  # the bug
    (root / "test_greet.py").write_text(
        "from app import greet\n\n\ndef test_greet():\n    assert greet() == 'hello'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=harness@localhost",
            "-c",
            "user.name=Harness Bench",
            "commit",
            "-qm",
            "bench fixture",
        ],
        check=True,
    )
    return root


def build_bench_config(results_dir: Path, db_path: Path) -> Any:
    """HarnessConfig for the offline bench: fake provider, eval-trio agents."""
    from harness.config import HarnessConfig

    return HarnessConfig.model_validate(
        {
            "models": {
                "default": {"provider": "fake", "name": "bench-model", "api_key_env": "AI_API_KEY"}
            },
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": "default"},
                {"agent_id": "mgr-1", "role": "manager", "model": "default"},
                {"agent_id": "ver-1", "role": "verifier", "model": "default"},
            ],
            "storage": {"backend": "sqlite", "sqlite_path": str(db_path)},
            "run": {"results_dir": str(results_dir)},
        }
    )


async def run_once(fixture: Path, results_dir: Path, issue: str) -> dict[str, Any]:
    """Run the full pipeline offline and return the evidence pack's token report."""
    from harness.engine.pipeline import HarnessPipeline
    from harness.infrastructure.context_store import SQLiteContextStore
    from harness.infrastructure.model_providers import FakeProvider, ModelResponse
    from harness.infrastructure.model_providers.base import ToolCall
    from harness.security.audit import AuditLog

    config = build_bench_config(
        results_dir=results_dir, db_path=results_dir.parent / ".harness" / "bench.db"
    )
    provider = FakeProvider(
        config.models["default"],
        responses=[
            ModelResponse(content=PROFILE_JSON),  # architect.analyze
            ModelResponse(content=PLAN_JSON),  # architect.decompose
            ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="bench-fix",
                        name="apply_edit",
                        arguments={
                            "path": "app.py",
                            "search": "return ''",
                            "replace": "return 'hello'",
                        },
                    )
                ],
            ),
            ModelResponse(content="TASK_COMPLETE: greet() now returns hello"),
            ModelResponse(content=VERDICT_JSON),  # architect.review
        ],
    )
    store = SQLiteContextStore(results_dir.parent / ".harness" / "bench-context.db")
    try:
        pipeline = HarnessPipeline(
            fixture,
            config,
            provider,
            store,
            audit=AuditLog(results_dir.parent / ".harness" / "bench-audit.jsonl"),
        )
        outcome = await pipeline.run(issue)
    finally:
        store.close()
    report_path = (outcome.evidence_path or results_dir) / "token-report.json"
    token_report: dict[str, Any] = {}
    if report_path.exists():
        token_report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "success": outcome.success,
        "outcome_line": outcome.outcome_line,
        "run_id": outcome.run_id,
        "token_report": token_report,
    }


def main(argv: list[str] | None = None) -> int:
    """Script entry: `PYTHONPATH=<measured-src> python runner.py --issue ...`."""
    parser = argparse.ArgumentParser(description="Run one offline bench measurement")
    parser.add_argument("--issue", default=FIXTURE_ISSUE)
    parser.add_argument("--fixture-dir", default=None, help="override the fixture location")
    parser.add_argument("--work-dir", default=None, help="where stores/results live")
    args = parser.parse_args(argv)

    fixture = build_fixture(Path(args.fixture_dir) if args.fixture_dir else None)
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="harness-bench-"))
    work.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(run_once(fixture, work / "results", args.issue))
    print(json.dumps(result))
    return 0 if result["success"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    sys.exit(main())
