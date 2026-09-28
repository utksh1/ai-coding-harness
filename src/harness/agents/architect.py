"""Architect agent (milestone 2, issues 2.1-2.3): analyze, decompose, review.

Per DESIGN_SPEC §2.1 adapted to the evaluation contract: repository analysis
consumes a deterministic repo summary (the filesystem indexer lands in
milestone 3 with the tool runtime), decomposition emits an in-process Plan
instead of GitHub issues, and final review judges the aggregate diff against
the plan's acceptance criteria locally.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from harness.agents.llm_agent import LLMAgent, StoreWindow
from harness.agents.task import Task
from harness.engine.budget import BudgetGovernor
from harness.infrastructure.context_store import ContextStore
from harness.infrastructure.model_providers import ModelProvider
from harness.tools.base import Tool
from harness.tools.filesystem import compact_repo_summary

PROFILE_SCHEMA = (
    '{"languages": [str], "frameworks": [str], "test_framework": str, '
    '"build_system": str, "conventions": [str], "notes": str}'
)
PLAN_SCHEMA = (
    '{"issue_summary": str, "complexity": int(1-10), '
    '"subtasks": [{"id": str, "title": str, "description": str, '
    '"specialty": str, "complexity": int, "files": [str], '
    '"acceptance_criteria": [str], "depends_on": [str]}], '
    '"risks": [str], "needs_collaboration": bool, '
    '"reproduction_test": str, "allow_test_edits": bool}'
)
VERDICT_SCHEMA = (
    '{"approved": bool, "issues": [str], "summary": str, '
    '"criteria_dispositions": [{"criterion": str, "satisfied": bool, "evidence": str}]}'
)
"""The verdict must judge EVERY acceptance criterion it can see: one
`criteria_dispositions` entry per criterion (satisfied + evidence). The
verification pipeline mechanically checks coverage - a verdict that skips
criteria fails stage 6 even when `approved` is true (review finding #4:
acceptance must be proven, not asserted)."""


class RepositoryProfile(BaseModel):
    """Structured understanding of the target repository."""

    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    test_framework: str = ""
    build_system: str = ""
    conventions: list[str] = Field(default_factory=list)
    notes: str = ""


class SubTask(BaseModel):
    """One atomic unit of the plan; `files` drives the Manager's overlap gate."""

    id: str
    title: str
    description: str
    specialty: str = "implementer"
    complexity: int = Field(default=5, ge=1, le=10)
    files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    def to_task(self) -> Task:
        return Task(
            id=self.id,
            title=self.title,
            description=self.description,
            acceptance_criteria=self.acceptance_criteria,
            specialty=self.specialty,
            complexity=self.complexity,
            files=self.files,
            metadata={"depends_on": self.depends_on},
        )


class Plan(BaseModel):
    """The Architect's decomposition of one issue."""

    issue_summary: str = ""
    complexity: int = Field(default=5, ge=1, le=10)
    subtasks: list[SubTask] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    needs_collaboration: bool = False
    reproduction_test: str = Field(
        default="",
        description="Bare pytest node id (file::test) that fails before the patch and must pass after. NOT a command - never include the 'pytest' prefix.",
    )
    allow_test_edits: bool = Field(
        default=False,
        description="Explicit plan-level allowance for modifying test files.",
    )


class CriterionDisposition(BaseModel):
    """One acceptance criterion judged by the final review."""

    criterion: str
    satisfied: bool
    evidence: str = ""


class ReviewVerdict(BaseModel):
    """Final review outcome with evidence-backed findings."""

    approved: bool
    issues: list[str] = Field(default_factory=list)
    summary: str = ""
    criteria_dispositions: list[CriterionDisposition] = Field(default_factory=list)


class ArchitectAgent(LLMAgent):
    """Top-tier agent: analysis, decomposition, final review, global rules."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("role", "architect")
        super().__init__(**kwargs)

    async def analyze_repository(self, repo_summary: dict[str, Any]) -> RepositoryProfile:
        """Turn a deterministic repo summary into a structured profile.

        The profile is also saved to the global context so every downstream
        agent shares one understanding of the codebase.
        """
        prompt = (
            "Analyze this repository summary and reply with JSON only.\n"
            f"Summary: {compact_repo_summary(repo_summary)}"
        )
        data = await self.structured_call(
            "Reply with JSON matching: " + PROFILE_SCHEMA, prompt, PROFILE_SCHEMA
        )
        profile = RepositoryProfile.model_validate(data)
        self.store.save_global("repo_profile", profile.model_dump())
        return profile

    async def decompose(self, issue_text: str, profile: RepositoryProfile | None = None) -> Plan:
        """Decompose one issue into subtasks with acceptance criteria."""
        profile_section = f"Repository profile: {profile.model_dump()}\n" if profile else ""
        prompt = (
            f"{profile_section}Decompose this issue into ATOMIC subtasks. Rules: "
            "every subtask MUST list the concrete files it will touch (from the "
            "repository summary); keep the number of subtasks minimal; each "
            f"acceptance criterion must be machine-checkable.\nISSUE:\n{issue_text}"
        )
        data = await self.structured_call(
            "Reply with JSON matching: " + PLAN_SCHEMA, prompt, PLAN_SCHEMA
        )
        plan = Plan.model_validate(data)
        self.store.save_global("plan", plan.model_dump())
        return plan

    async def review(self, diff: str, plan: Plan, evidence: str = "") -> ReviewVerdict:
        """Judge the diff against acceptance criteria + verification evidence.

        Never silently truncates (audit §20): the diff is split into per-file
        chunks; each gets a structured call, capped at _MAX_REVIEW_CHUNKS, and
        files beyond the budget are named in the verdict so the gate fails
        honestly rather than reviewing a blind spot.
        """
        context = _ReviewContext(plan, evidence)
        chunks, unreviewed = split_diff_chunks(diff)
        verdicts: list[ReviewVerdict] = []
        if chunks:
            for index, chunk in enumerate(chunks):
                data = await self.structured_call(
                    "Reply with JSON matching: " + VERDICT_SCHEMA,
                    context.chunk_prompt(chunk, index, len(chunks)),
                    VERDICT_SCHEMA,
                )
                verdicts.append(ReviewVerdict.model_validate(data))
        else:
            # No diff to review: acceptance criteria are still judged - from
            # the verification evidence and task outcomes. A synthetic
            # auto-approve here would let criteria pass unjudged (review
            # finding #4); the criteria-coverage gate in stage 6 requires a
            # real disposition for every criterion either way.
            data = await self.structured_call(
                "Reply with JSON matching: " + VERDICT_SCHEMA,
                context.chunk_prompt(
                    "(no diff: judge each acceptance criterion strictly from the "
                    "verification evidence and task outcomes above)",
                    0,
                    0,
                ),
                VERDICT_SCHEMA,
            )
            verdicts.append(ReviewVerdict.model_validate(data))
        return _merge_verdicts(verdicts, unreviewed)

    async def reframe(self, task: Task, escalation_message: str) -> Task:
        """Level-3 recovery: restate a task the team could not complete."""
        prompt = (
            f"This task repeatedly failed. Reframe it smaller and clearer.\n"
            f"TASK: {task.title}\n{task.description}\nFAILURE: {escalation_message}"
        )
        data = await self.structured_call(
            'Reply with JSON: {"title": str, "description": str, "acceptance_criteria": [str]}',
            prompt,
            '{"title": str, "description": str, "acceptance_criteria": [str]}',
        )
        return task.model_copy(
            update={
                "title": data.get("title", task.title),
                "description": data.get("description", task.description),
                "acceptance_criteria": data.get("acceptance_criteria", task.acceptance_criteria),
                "metadata": {**task.metadata, "reframed": True},
            }
        )


_REVIEW_CHUNK_CHARS = 12_000
_MAX_REVIEW_CHUNKS = 5


def split_diff_chunks(
    diff: str, max_chars: int = _REVIEW_CHUNK_CHARS
) -> tuple[list[str], list[str]]:
    """Split a unified diff into per-file chunks under `max_chars`.

    Returns (chunks, unreviewed_files). Never silently drops content
    (audit §20): files beyond the caller's chunk budget are returned in
    `unreviewed_files` so the final gate can fail honestly instead of
    reviewing a truncated artifact.
    """
    if not diff.strip():
        return [], []
    files: list[str] = []
    current: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            files.append("".join(current))
            current = []
        current.append(line)
    if current:
        files.append("".join(current))

    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for file_diff in files:
        if size + len(file_diff) > max_chars and buffer:
            chunks.append("".join(buffer))
            buffer, size = [], 0
        buffer.append(file_diff)
        size += len(file_diff)
    if buffer:
        chunks.append("".join(buffer))

    if len(chunks) <= _MAX_REVIEW_CHUNKS:
        return chunks, []
    dropped = "".join(chunks[_MAX_REVIEW_CHUNKS:])
    unreviewed = [
        line[6:].strip()
        for line in dropped.splitlines()
        if line.startswith("+++ b/") and line[6:].strip() != "/dev/null"
    ]
    return chunks[:_MAX_REVIEW_CHUNKS], [name for name in unreviewed if name]


class _ReviewContext:
    """Shared prompt preamble for one review pass (criteria + evidence)."""

    def __init__(self, plan: Plan, evidence: str) -> None:
        criteria = "\n".join(
            f"- {st.id}: {'; '.join(st.acceptance_criteria) or st.title}" for st in plan.subtasks
        )
        self.preamble = f"CRITERIA:\n{criteria or '- (none)'}\n\nVERIFICATION EVIDENCE:\n{evidence or '(none collected)'}"

    def chunk_prompt(self, chunk: str, index: int, total: int) -> str:
        return (
            f"Review chunk {index + 1}/{total} of this diff against the "
            "acceptance criteria and evidence. Reply with JSON only.\n"
            "For EVERY criterion you can judge from this chunk, emit one "
            "criteria_dispositions entry {criterion, satisfied, evidence}; "
            "criteria not judgeable from this chunk may be omitted (they are "
            "judged from other chunks, and the pipeline fails any criterion "
            "left unjudged overall).\n"
            f"{self.preamble}\n\nDIFF CHUNK:\n{chunk}"
        )


def _merge_verdicts(verdicts: list[ReviewVerdict], unreviewed_files: list[str]) -> ReviewVerdict:
    issues: list[str] = []
    for verdict in verdicts:
        issues.extend(verdict.issues)
    approved = all(v.approved for v in verdicts) and not unreviewed_files
    if unreviewed_files:
        issues.append("diff not fully reviewed (over chunk budget): " + ", ".join(unreviewed_files))
    summary = "; ".join(v.summary for v in verdicts if v.summary)
    # Criterion dispositions merge across chunks: a criterion judged in ANY
    # chunk counts as judged, and a single unsatisfied verdict wins (a
    # criterion is satisfied only where EVERY judgment of it is).
    by_criterion: dict[str, CriterionDisposition] = {}
    for verdict in verdicts:
        for disposition in verdict.criteria_dispositions:
            previous = by_criterion.get(disposition.criterion)
            if previous is None or previous.satisfied:
                by_criterion[disposition.criterion] = disposition
    return ReviewVerdict(
        approved=approved,
        issues=issues,
        summary=summary,
        criteria_dispositions=list(by_criterion.values()),
    )


def build_architect(
    agent_id: str,
    model_config: dict[str, Any],
    provider: ModelProvider,
    store: ContextStore,
    governor: BudgetGovernor,
    tools: list[Tool] | None = None,
) -> ArchitectAgent:
    """Convenience factory wiring the architect role defaults."""
    return ArchitectAgent(
        agent_id=agent_id,
        model_config=model_config,
        tools=tools or [],
        context_window=StoreWindow(store, agent_id, "planning"),
        provider=provider,
        store=store,
        governor=governor,
    )
