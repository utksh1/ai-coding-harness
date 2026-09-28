"""Milestone 4.8: chunked, evidence-fed architect final review (audit §20).

The final gate never silently truncates: the diff is split into per-file
chunks, each reviewed via its own structured call, and files beyond the
chunk budget are named in the verdict (failing the gate honestly).
Verification evidence accompanies the review prompt.
"""

from __future__ import annotations

import json

from harness.agents.architect import ArchitectAgent, Plan, split_diff_chunks
from harness.agents.llm_agent import StoreWindow
from harness.config import BudgetConfig, ModelConfig
from harness.engine.budget import BudgetGovernor
from harness.infrastructure.context_store import MemoryContextStore
from harness.infrastructure.model_providers.base import ModelResponse
from harness.infrastructure.model_providers.fake import FakeProvider

PLAN = Plan.model_validate(
    {
        "issue_summary": "greeting",
        "complexity": 2,
        "subtasks": [
            {
                "id": "st-1",
                "title": "greet",
                "description": "",
                "specialty": "implementer",
                "complexity": 2,
                "files": ["app.py"],
                "acceptance_criteria": ["greet returns hello"],
                "depends_on": [],
            }
        ],
    }
)


def _architect(script: list[ModelResponse]) -> tuple[ArchitectAgent, FakeProvider]:
    config = ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY")
    provider = FakeProvider(config, responses=script)
    store = MemoryContextStore()
    agent = ArchitectAgent(
        agent_id="arch-1",
        model_config={},
        tools=[],
        context_window=MemoryContextStore(),
        provider=provider,
        store=store,
        governor=BudgetGovernor(store, BudgetConfig(), "run-1"),
    )
    from harness.agents.llm_agent import StoreWindow

    agent.context_window = StoreWindow(store, "arch-1", "planning")
    return agent, provider


def _verdict(
    approved: bool, issues: list[str] | None = None, dispositions: list[dict] | None = None
) -> ModelResponse:
    payload: dict = {"approved": approved, "issues": issues or [], "summary": "chunk"}
    if dispositions is not None:
        payload["criteria_dispositions"] = dispositions
    return ModelResponse(content=json.dumps(payload))


def test_split_diff_chunks_single_small_diff() -> None:
    diff = "diff --git a/app.py\n+++ b/app.py\n+code\n"
    chunks, unreviewed = split_diff_chunks(diff)
    assert len(chunks) == 1
    assert unreviewed == []


def test_split_diff_chunks_groups_files_under_budget() -> None:
    file_a = "diff --git a/a.py\n+++ b/a.py\n" + "+" + "x" * 900 + "\n"
    file_b = "diff --git a/b.py\n+++ b/b.py\n" + "+" + "y" * 900 + "\n"
    chunks, unreviewed = split_diff_chunks(file_a + file_b, max_chars=2000)
    assert len(chunks) == 1  # both fit in one 2000-char chunk
    assert "a.py" in chunks[0] and "b.py" in chunks[0]
    assert unreviewed == []


def test_split_diff_chunks_flags_overflow_honestly() -> None:
    files = "".join(
        f"diff --git a/f{i}.py\n+++ b/f{i}.py\n" + "+" + "z" * 600 + "\n" for i in range(16)
    )
    chunks, unreviewed = split_diff_chunks(files, max_chars=2000)
    assert len(chunks) == 5  # capped at _MAX_REVIEW_CHUNKS
    assert unreviewed == ["f15.py"]  # the 16th file lands beyond the budget


def test_split_diff_chunks_empty() -> None:
    assert split_diff_chunks("") == ([], [])


async def test_review_single_chunk_includes_evidence() -> None:
    agent, provider = _architect([_verdict(True)])
    diff = "diff --git a/app.py\n+++ b/app.py\n+hello\n"
    verdict = await agent.review(diff, PLAN, evidence="stage2: 12/12 tests pass")
    assert verdict.approved
    prompt = provider.calls[0]["messages"][1]["content"]
    assert "VERIFICATION EVIDENCE:" in prompt
    assert "12/12 tests pass" in prompt
    assert "chunk 1/1" in prompt


async def test_review_merges_per_chunk_verdicts() -> None:
    diff = (
        "diff --git a/a.py\n+++ b/a.py\n+" + "x" * 11_000 + "\n"
        "diff --git a/b.py\n+++ b/b.py\n+" + "y" * 11_000 + "\n"
    )
    agent, provider = _architect([_verdict(True), _verdict(False, ["n+1 query"])])
    verdict = await agent.review(diff, PLAN)
    assert not verdict.approved
    assert "n+1 query" in verdict.issues
    assert len(provider.calls) == 2


async def test_review_overflow_fails_gate_honestly() -> None:
    files = "".join(
        f"diff --git a/f{i}.py\n+++ b/f{i}.py\n" + "+" + "z" * 4_500 + "\n" for i in range(12)
    )
    agent, provider = _architect([_verdict(True) for _ in range(5)])
    verdict = await agent.review(files, PLAN)
    assert not verdict.approved
    assert any("not fully reviewed" in issue for issue in verdict.issues)
    assert len(provider.calls) == 5  # budget respected


async def test_review_empty_diff_judges_criteria_from_evidence() -> None:
    """Empty diff is no longer an auto-approve (review finding #4).

    Criteria are judged from the verification evidence via exactly ONE
    structured call - never zero: a synthetic verdict would let acceptance
    criteria pass unjudged on empty-diff runs."""
    agent, provider = _architect(
        [
            _verdict(
                approved=True,
                dispositions=[{"criterion": "greet returns hello", "satisfied": True}],
            )
        ]
    )
    verdict = await agent.review("", PLAN)
    assert verdict.approved
    assert len(provider.calls) == 1
    assert "judge each acceptance criterion" in provider.calls[0]["messages"][-1]["content"]
    assert [d.criterion for d in verdict.criteria_dispositions] == ["greet returns hello"]


def test_merge_dispositions_across_chunks_single_unsatisfied_wins() -> None:
    """A criterion judged satisfied in one chunk and not in another FAILS:
    acceptance is proven only where every judgment agrees."""
    from harness.agents.architect import CriterionDisposition, ReviewVerdict, _merge_verdicts

    v1 = ReviewVerdict(
        approved=True,
        criteria_dispositions=[
            CriterionDisposition(criterion="c1", satisfied=True, evidence="a"),
            CriterionDisposition(criterion="c2", satisfied=True, evidence="a"),
        ],
    )
    v2 = ReviewVerdict(
        approved=True,
        criteria_dispositions=[
            CriterionDisposition(criterion="c2", satisfied=False, evidence="b"),
            CriterionDisposition(criterion="c3", satisfied=True, evidence="b"),
        ],
    )
    merged = _merge_verdicts([v1, v2], [])
    by_criterion = {d.criterion: d.satisfied for d in merged.criteria_dispositions}
    assert by_criterion == {"c1": True, "c2": False, "c3": True}


async def test_stage6_requires_disposition_for_every_criterion(tmp_path) -> None:
    """The machine check: approved=True is NOT enough when a criterion is
    unjudged (or judged unsatisfied) - stage 6 fails honestly."""
    from harness.agents.architect import Plan
    from harness.verification.pipeline import VerificationPipeline

    plan = Plan.model_validate(
        {
            "issue_summary": "s",
            "complexity": 1,
            "subtasks": [
                {
                    "id": "st-1",
                    "title": "t",
                    "description": "d",
                    "specialty": "localization",
                    "files": [],
                    "acceptance_criteria": ["c1", "c2"],
                    "depends_on": [],
                }
            ],
        }
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    store = MemoryContextStore()
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=100_000), "corr-cov")
    provider = FakeProvider(
        ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY"),
        responses=[
            ModelResponse(
                content=json.dumps(
                    {
                        "approved": True,
                        "issues": [],
                        "summary": "looks good",
                        "criteria_dispositions": [
                            {"criterion": "c1", "satisfied": True, "evidence": "x"}
                        ],
                    }
                )
            ),
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
    results = await VerificationPipeline(tmp_path).run("", plan, architect)
    stage6 = next(r for r in results if r.name == "6-final-review")
    assert not stage6.passed
    assert "not individually addressed: c2" in stage6.detail
    assert stage6.evidence["missing_criteria"] == ["c2"]


async def test_stage6_fails_unsatisfied_criterion(tmp_path) -> None:
    from harness.agents.architect import Plan
    from harness.verification.pipeline import VerificationPipeline

    plan = Plan.model_validate(
        {
            "issue_summary": "s",
            "complexity": 1,
            "subtasks": [
                {
                    "id": "st-1",
                    "title": "t",
                    "description": "d",
                    "specialty": "localization",
                    "files": [],
                    "acceptance_criteria": ["c1"],
                    "depends_on": [],
                }
            ],
        }
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    store = MemoryContextStore()
    governor = BudgetGovernor(store, BudgetConfig(total_tokens=100_000), "corr-unsat")
    provider = FakeProvider(
        ModelConfig(provider="fake", name="fake-model", api_key_env="AI_API_KEY"),
        responses=[
            ModelResponse(
                content=json.dumps(
                    {
                        "approved": True,
                        "issues": [],
                        "summary": "seems done",
                        "criteria_dispositions": [
                            {"criterion": "c1", "satisfied": False, "evidence": "no covering test"}
                        ],
                    }
                )
            ),
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
    results = await VerificationPipeline(tmp_path).run("", plan, architect)
    stage6 = next(r for r in results if r.name == "6-final-review")
    assert not stage6.passed
    assert "not satisfied: c1" in stage6.detail
