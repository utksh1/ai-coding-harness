"""Code-review smells + verification pipeline (issues 3.4-3.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.agents.architect import Plan, SubTask
from harness.config import BudgetConfig
from harness.engine.budget import BudgetGovernor
from harness.infrastructure.context_store import MemoryContextStore
from harness.infrastructure.model_providers import FakeProvider, ModelResponse
from harness.verification.code_review import review_paths
from harness.verification.pipeline import VerificationPipeline, stage_report


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "smelly.py").write_text(
        "def undocumented_public(a):\n"
        "    if a:\n"
        "        for i in range(3):\n"
        "            if i:\n"
        "                while True:\n"
        "                    if a:\n"
        "                        if i:\n"
        "                            if a:\n"
        "                                return a\n"
        "    return None\n"
        "\n"
        "try:\n"
        "    pass\n"
        "except:\n"
        "    pass\n"
    )
    (tmp_path / "src" / "clean.py").write_text(
        '"""Module."""\n\n\ndef documented():\n    """Hi."""\n    return 1\n'
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    return tmp_path


def test_review_paths_detects_smells(repo) -> None:
    findings = review_paths([repo / "src" / "smelly.py", repo / "src" / "clean.py"], repo)
    kinds = {f.kind for f in findings}
    assert {"deep-nesting", "bare-except", "missing-docstring"} <= kinds, kinds
    assert not [f for f in findings if f.path.endswith("clean.py")]


def test_review_paths_skips_non_python(repo) -> None:
    assert review_paths([repo / "pyproject.toml", repo / "ghost.py"], repo) == []


def _pipeline(repo: Path) -> VerificationPipeline:
    return VerificationPipeline(repo)


async def test_pipeline_all_pass(repo, fake_model_config) -> None:
    store = MemoryContextStore()
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=100_000), "corr-v")
    from harness.agents.architect import ArchitectAgent
    from harness.agents.llm_agent import StoreWindow

    provider = FakeProvider(
        fake_model_config,
        responses=[
            ModelResponse(content='{"approved": true, "issues": [], "summary": "solid"}'),
        ],
    )
    architect = ArchitectAgent(
        agent_id="arch",
        model_config={},
        tools=[],
        context_window=StoreWindow(store, "arch", "planning"),
        provider=provider,
        store=store,
        governor=governor,
    )
    plan = Plan(
        subtasks=[SubTask(id="st-1", title="t", description="d", acceptance_criteria=["works"])]
    )
    diff = ""  # no changes: self-check trivially passes, review skipped files
    results = await _pipeline(repo).run(diff, plan, architect)
    by_name = {r.name: r for r in results}
    assert by_name["2-self-check"].passed
    assert by_name["3-local-tests"].passed
    assert by_name["4-code-review"].passed and not by_name["4-code-review"].blocking
    assert by_name["5-security"].passed
    assert by_name["6-final-review"].passed
    assert "VERIFIED" in stage_report(results)


async def test_pipeline_blocks_on_failed_tests(repo) -> None:
    (repo / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    results = await _pipeline(repo).run(diff="", plan=None, architect=None)
    names = [r.name for r in results]
    assert names == [
        "1-integrity",
        "2-self-check",
        "3-local-tests",
    ]  # blocking failure stops the run
    assert not results[-1].passed


async def test_pipeline_security_stage_blocks(repo) -> None:
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n")
    import subprocess

    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    (repo / "leak.py").write_text('api_key = "sk-1234567890abcdef1234"\n')
    subprocess.run(["git", "-C", str(repo), "add", "leak.py"], check=True)
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "HEAD"], capture_output=True, text=True, check=True
    ).stdout
    results = await _pipeline(repo).run(diff, plan=None, architect=None)
    security = {r.name: r for r in results}["5-security"]
    assert not security.passed and "secret detected" in security.detail


async def test_pipeline_cancellation(repo) -> None:
    pipeline = _pipeline(repo)
    pipeline.cancelled = True
    results = await pipeline.run(diff="", plan=None, architect=None)
    assert results[0].name == "cancelled"


def test_changed_files_parses_diff() -> None:
    diff = "\n".join(["--- a/x.py", "+++ b/x.py", "+code"])
    assert VerificationPipeline(Path(".")).changed_files(diff) == ["x.py"]


async def test_pipeline_budget_exhaustion_degrades_gracefully(repo, fake_model_config) -> None:
    """Live-run regression (parse repo): BudgetExhausted raised inside a
    stage (the Architect's final review) must degrade to a recorded stage
    result — never propagate. The CLI must print an outcome and finalize the
    evidence pack instead of crashing with a traceback."""
    store = MemoryContextStore()
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=1000), "corr-budget")
    governor.record("arch", "fake-model", 5000, 0)  # already over budget

    from harness.agents.architect import ArchitectAgent
    from harness.agents.llm_agent import StoreWindow

    provider = FakeProvider(fake_model_config, responses=[ModelResponse(content="{}")])
    architect = ArchitectAgent(
        agent_id="arch",
        model_config={},
        tools=[],
        context_window=StoreWindow(store, "arch", "planning"),
        provider=provider,
        store=store,
        governor=governor,
    )
    plan = Plan(
        subtasks=[SubTask(id="st-1", title="t", description="d", acceptance_criteria=["x"])]
    )
    events: list[dict] = []
    (repo / "x.py").write_text("code = 1\n")  # self-check parses changed files
    diff = "\n".join(["--- a/x.py", "+++ b/x.py", "@@ -1 +1 @@", "+code = 2"])
    results = await _pipeline(repo).run(
        diff, plan, architect, run_id="r-budget", tracer=events.append
    )
    by_name = {r.name: r for r in results}
    # deterministic stages still ran and are recorded
    assert by_name["1-integrity"].passed
    assert by_name["3-local-tests"].passed
    # the exhaustion is the stage-6 verdict: blocking, honest, no exception
    assert not by_name["6-final-review"].passed
    assert by_name["6-final-review"].blocking
    assert "token budget exhausted" in by_name["6-final-review"].detail
    # the cockpit streamed the stop too
    stage_events = [e for e in events if e.get("event") == "verification.stage"]
    assert any(e.get("stage") == "6-final-review" for e in stage_events)
