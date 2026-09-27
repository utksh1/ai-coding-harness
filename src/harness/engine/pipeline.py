"""Pipeline orchestration (milestone 3, issue 3.6): the whole harness, end to end.

One issue in, one verified evidence pack out:

1. input security scan (issue 3.7) - flagged, never silently ignored
2. Architect: analyze repository -> decompose issue (issues 2.1-2.2)
3. Manager: route subtasks; disjoint file-sets run concurrently (2.4-2.6)
4. Specialists execute under the recovery ladder (2.7-2.11) with the
   budget governor metering every call (2.13)
5. Verification pipeline: self-check, tests, smells, secrets, final review
6. Evidence pack + metrics + audit trail

Spec sections intentionally absent: GitHub PR merges (no credentials at
eval time), Docker sandboxing (subprocess limits instead), web dashboard
(Textual cockpit reads the same evidence).
"""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.agents.architect import ArchitectAgent, Plan, RepositoryProfile, SubTask
from harness.agents.llm_agent import StoreWindow
from harness.agents.manager import (
    ManagerAgent,
    SpecialistSlot,
    assign_specialists,
    execution_batches,
    routing_breakdown,
)
from harness.agents.specialists import build_agent
from harness.agents.task import Task, TaskResult
from harness.config import HarnessConfig
from harness.engine.budget import BudgetGovernor
from harness.engine.evidence import EvidencePack, build_summary
from harness.engine.recovery import Executor, RecoveryLadder, Rerouter
from harness.infrastructure.model_providers import ModelAuthError
from harness.monitoring.metrics import MetricsCollector
from harness.security.audit import AuditLog
from harness.security.input_guard import detect_prompt_injection
from harness.tools.execution import RunTestsTool, detect_test_runner
from harness.tools.filesystem import summarize_repository
from harness.tools.registry import build_default_tools
from harness.verification.baseline import Baseline, capture_baseline
from harness.verification.pipeline import VerificationPipeline, stage_report

_ROUTABLE_ERROR_TYPES = {
    "KeyError",
    "AttributeError",
    "TimeoutError",
    "ConnectionError",
    "ValueError",
    "TypeError",
}


@dataclass
class PipelineOutcome:
    run_id: str
    success: bool
    outcome_line: str
    plan: Plan | None = None
    stage_results: list[Any] = field(default_factory=list)
    task_results: list[TaskResult] = field(default_factory=list)
    evidence_path: Path | None = None
    flags: list[str] = field(default_factory=list)


class HarnessPipeline:
    """Wires every milestone-2 component into one eval-mode run."""

    def __init__(
        self,
        repo_root: Path,
        config: HarnessConfig,
        provider: Any,
        store: Any,
        audit: AuditLog | None = None,
        event_sink: Any = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()  # canonical: macOS /tmp symlinks
        self._config = config
        self._store = store
        self._audit = audit or AuditLog(self._repo_root / ".harness" / "audit.jsonl")
        self._event_sink = event_sink
        self._provider = provider
        self._tools = build_default_tools(self._repo_root)
        self._agents: dict[str, Any] = {}
        self._coordination_ids: set[str] = set()
        self._manager: ManagerAgent | None = None
        self._architect: ArchitectAgent | None = None
        self._collaborators_added = 0
        # Placeholder governor: replaced per-run in `run()` before any call.
        self._placeholder_governor = BudgetGovernor(store, config.budget, "unassigned")
        self._build_agents()

    def _build_agents(self) -> None:
        slots: list[SpecialistSlot] = []
        for agent_config in self._config.agents:
            if not agent_config.enabled:
                continue
            model_config = {"provider": self._config.models["default"].provider}
            if agent_config.role == "architect":
                self._architect = ArchitectAgent(
                    agent_id=agent_config.agent_id,
                    model_config=model_config,
                    tools=self._tools,
                    context_window=StoreWindow(
                        self._store,
                        agent_config.agent_id,
                        "planning",
                        stale_tool_results=agent_config.stale_tool_results,
                    ),
                    provider=self._provider,
                    store=self._store,
                    governor=self._placeholder_governor,
                )
                self._agents[agent_config.agent_id] = self._architect
                self._coordination_ids.add(agent_config.agent_id)
                continue
            if agent_config.role == "manager":
                self._manager = ManagerAgent(
                    agent_id=agent_config.agent_id,
                    model_config=model_config,
                    tools=self._tools,
                    context_window=StoreWindow(
                        self._store,
                        agent_config.agent_id,
                        "coordination",
                        stale_tool_results=agent_config.stale_tool_results,
                    ),
                    provider=self._provider,
                    store=self._store,
                    governor=self._placeholder_governor,
                )
                self._agents[agent_config.agent_id] = self._manager
                self._coordination_ids.add(agent_config.agent_id)
                continue
            agent = build_agent(
                agent_id=agent_config.agent_id,
                role=agent_config.role,
                model_config=model_config,
                provider=self._provider,
                store=self._store,
                governor=self._placeholder_governor,  # replaced per-run
                tools=self._tools,
                model_tier=agent_config.model_tier,
                stale_tool_results=agent_config.stale_tool_results,
                knowledge_enabled=agent_config.knowledge,
                knowledge_max_chars=agent_config.knowledge_max_chars,
                max_steps=self._config.run.max_steps,
            )
            self._agents[agent.agent_id] = agent
            from harness.agents.prompts import ROLE_PRESETS

            preset = ROLE_PRESETS.get(agent_config.role)
            specialties = set(preset.specialties) if preset else {agent_config.role}
            slots.append(
                SpecialistSlot(
                    agent_id=agent.agent_id,
                    specialties=weak_specialties(set(specialties)),
                    available_tools={tool.name for tool in self._tools},
                    model_tier=agent_config.model_tier,
                    role=agent_config.role,
                )
            )
        if self._manager is not None:
            for slot in slots:
                self._manager.register_specialist(slot)
        self._specialist_slots = slots

    async def run(
        self,
        issue_text: str,
        demo_mode: bool = False,
        event_sink: Any = None,
        run_id: str | None = None,
    ) -> PipelineOutcome:
        """Execute one full pipeline run.

        `run_id` (optional) lets the platform layer correlate the gateway's
        task id, the streamed events, and the evidence directory under ONE
        id. Left unset, the pipeline mints its own (eval-mode CLI behavior).
        """
        run_id = run_id or uuid.uuid4().hex[:12]
        governor = BudgetGovernor(self._store, self._config.budget, run_id)
        metrics = MetricsCollector(self._store, governor)
        pack = EvidencePack(
            self._repo_root / self._config.run.results_dir,
            run_id,
            event_sink=event_sink or self._event_sink,
        )
        for agent in self._agents.values():
            agent.governor = governor
            # Cockpit event contract (docs/cockpit-events.md): per-agent
            # step/tool/usage events flow through the run's evidence trace.
            agent.attach_tracer(pack.trace, run_id)
        flags = [
            f"prompt-injection pattern: {pattern}"
            for pattern in detect_prompt_injection(issue_text)
        ]
        if demo_mode:
            flags.append("DEMO MODE: scripted model responses (illustrative only)")
        pack.trace(
            {"event": "run.start", "run_id": run_id, "flags": flags, "issue": issue_text[:2000]}
        )
        self._audit.append("pipeline", "run.start", run_id, {"flags": len(flags)})

        architect = self._architect
        if architect is None:
            return self._no_architect_outcome(run_id, pack)
        metrics.stage_started("architect")
        profile, plan = await self._architect_stage_with_retry(architect, issue_text, run_id, pack)
        metrics.stage_finished("architect")
        pack.trace(
            {"event": "architect.profile", "run_id": run_id, "profile": profile.model_dump()}
        )
        pack.trace(
            {
                "event": "architect.plan",
                "run_id": run_id,
                "reproduction_test": plan.reproduction_test,
                "risks": [risk[:200] for risk in plan.risks][:5],
                # v2 contract: subtask OBJECTS (title/specialty/complexity/
                # files/dependencies), not bare ids - the cockpit plan board
                # renders from this event alone.
                "subtasks": [
                    {
                        "id": s.id,
                        "title": s.title,
                        "description": s.description[:300],
                        "specialty": s.specialty,
                        "complexity": s.complexity,
                        "files": list(s.files),
                        "depends_on": list(s.depends_on),
                        "acceptance": "; ".join(s.acceptance_criteria)[:300],
                    }
                    for s in plan.subtasks
                ],
            }
        )

        # Reproduction-first baseline (improvements §1.1): run the target
        # suite BEFORE any specialist touches a file.
        baseline: Baseline | None = None
        if detect_test_runner(self._repo_root)[0] != "none":
            run_tool = RunTestsTool(self._repo_root)
            baseline = await capture_baseline(self._repo_root, run_tool, plan.reproduction_test)
            pack.trace(
                {
                    "event": "baseline.captured",
                    "run_id": run_id,
                    "runnable": baseline.runnable,
                    "pre_existing_failures": len(baseline.failed),
                    "reproduction_test": plan.reproduction_test,
                }
            )

        task_results: list[TaskResult] = []
        metrics.stage_started("specialists")
        if self._manager is not None and plan.subtasks:
            for batch_no, batch in enumerate(execution_batches(plan.subtasks), start=1):
                outcomes = await self._run_batch(
                    batch, governor, metrics, pack, run_id, architect, batch_no
                )
                task_results.extend(outcomes)
        metrics.stage_finished("specialists")
        # Live token meter for the cockpit: cumulative usage after the
        # specialist phase (the UI sets, never accumulates, this value).
        pack.trace(self._tokens_event(run_id, governor, "specialists"))

        metrics.stage_started("verification")
        diff = self._working_diff()
        verification = VerificationPipeline(self._repo_root, baseline)
        stage_results = await verification.run(
            diff, plan, architect, run_id=run_id, tracer=pack.trace
        )
        metrics.stage_finished("verification")
        pack.trace(self._tokens_event(run_id, governor, "verification"))
        pack.patch(diff)
        pack.test_report(stage_report(stage_results))
        pack.token_report(metrics.report())
        if baseline is not None:
            pack.baseline_report(baseline.to_report())
        overall = all(r.passed for r in stage_results if r.blocking) and all(
            r.success for r in task_results
        )
        outcome_line = (
            "VERIFIED: all tasks completed and gates passed"
            if overall
            else "NOT VERIFIED: see test-report.md and task failures"
        )
        plan_markdown = (
            "\n".join(
                f"- **{s.id}** ({s.specialty}, complexity {s.complexity}): {s.title}"
                for s in plan.subtasks
            )
            or "_no subtasks_"
        )
        pack.summary(
            build_summary(
                run_id, issue_text, plan_markdown, stage_report(stage_results), outcome_line, flags
            )
        )
        pack.trace({"event": "run.end", "run_id": run_id, "success": overall})
        self._audit.append("pipeline", "run.end", run_id, {"success": overall, "flags": len(flags)})
        return PipelineOutcome(
            run_id=run_id,
            success=overall,
            outcome_line=outcome_line,
            plan=plan,
            stage_results=stage_results,
            task_results=task_results,
            evidence_path=pack.path,
            flags=flags,
        )

    async def _run_batch(
        self,
        batch: list[SubTask],
        governor: BudgetGovernor,
        metrics: MetricsCollector,
        pack: EvidencePack,
        run_id: str,
        architect: ArchitectAgent,
        batch_no: int = 1,
    ) -> list[TaskResult]:
        """Execute one file-disjoint batch, strictly sequentially.

        The batch structure exists so a future worktree fan-out can run its
        members in parallel; until worktrees land, concurrent specialists
        share ONE working tree (edits interleave, test runs race), so the
        audit's §11 finding is honored by serializing inside the batch.
        """
        results: list[TaskResult] = []

        from harness.orchestration.messages import ErrorEscalation

        def make_reroute(agent_id: str) -> Rerouter:
            """Genuine L2 re-route (audit §8): 'reassign' swaps to another
            specialist; 'add collaborators' spawns one (bounded per run)."""

            def reroute(task: Task, escalation: ErrorEscalation, guidance: str) -> Executor | None:
                if "reassign" in guidance:
                    alternative = next(
                        (
                            a
                            for aid, a in self._agents.items()
                            if aid != agent_id and aid not in self._coordination_ids
                        ),
                        None,
                    )
                    return alternative.execute_task if alternative else None
                if "add collaborators" in guidance:
                    collaborator = self._add_collaborator(agent_id, governor, pack, run_id)
                    return collaborator.execute_task if collaborator else None
                return None

            return reroute

        async def run_one(subtask: SubTask) -> TaskResult:
            task = subtask.to_task()
            chosen = assign_specialists(task, self._specialist_slots, team_average_tokens=0)
            agent_id = chosen[0] if chosen else next(iter(self._agents))
            agent = self._agents[agent_id]
            slot = next(
                (s for s in self._specialist_slots if s.agent_id == agent_id),
                None,
            )
            pack.trace(
                {
                    "event": "specialist.assigned",
                    "run_id": run_id,
                    "task": task.id,
                    "agent": agent_id,
                    "role": getattr(agent, "role", ""),
                    "batch": batch_no,
                    # The Manager's delegation moment: the §5.1 factor
                    # contributions that put the task on this agent.
                    "routing": routing_breakdown(task, slot) if slot else None,
                }
            )
            ladder = RecoveryLadder(
                self._manager,
                architect,
                self._store,
                reroute=make_reroute(agent_id),
                on_event=pack.trace,
                governor=governor,
                run_id=run_id,
                agent_id=agent_id,
            )

            async def classify(t: Task, r: TaskResult) -> ErrorEscalation:
                escalation = await agent.handle_error(RuntimeError(r.error or "task failed"), t)
                # _attempt prefixes the original exception type; keep it so the
                # Manager's categorization sees KeyError/AttributeError etc.
                type_name = (r.error or "").split(":", 1)[0]
                if type_name in _ROUTABLE_ERROR_TYPES:
                    escalation = escalation.model_copy(update={"error_type": type_name})
                return escalation

            result = await ladder.run(task, agent.execute_task, classify)
            metrics.record_result(result, agent_id=agent_id)
            # Attribute the result to the agent that ACTUALLY finished the
            # work (an L2 reroute may have handed it to a collaborator).
            executor = getattr(ladder, "last_executor_agent", None) or agent
            pack.trace(
                {
                    "event": "specialist.result",
                    "run_id": run_id,
                    "task": task.id,
                    "agent": getattr(executor, "agent_id", agent_id),
                    "role": getattr(executor, "role", ""),
                    "success": result.success,
                    "summary": result.summary[:400],
                    "steps": getattr(executor, "steps_used", 0),
                    "tokens": getattr(executor, "traced_tokens", 0),
                }
            )
            return result

        for subtask in batch:
            results.append(await run_one(subtask))
        return results

    def _add_collaborator(
        self, primary_agent_id: str, governor: BudgetGovernor, pack: EvidencePack, run_id: str
    ) -> Any | None:
        """Spawn an extra specialist for a complex task (bounded, per run)."""
        if self._collaborators_added >= 2:
            return None
        self._collaborators_added += 1
        agent_id = f"{primary_agent_id}-collab-{self._collaborators_added}"
        agent = build_agent(
            agent_id=agent_id,
            role="implementer",
            model_config={"provider": self._config.models["default"].provider},
            provider=self._provider,
            store=self._store,
            governor=self._placeholder_governor,
            tools=self._tools,
            max_steps=self._config.run.max_steps,
        )
        agent.governor = governor
        agent.attach_tracer(pack.trace, run_id)
        self._agents[agent_id] = agent
        self._specialist_slots.append(
            SpecialistSlot(
                agent_id=agent_id,
                specialties=weak_specialties({"implementer"}),
                available_tools={tool.name for tool in self._tools},
                role="implementer",
            )
        )
        pack.trace(
            {
                "event": "specialist.collaborator_added",
                "run_id": run_id,
                "agent": agent_id,
                "for": primary_agent_id,
                "role": "implementer",
            }
        )
        return agent

    def _tokens_event(self, run_id: str, governor: BudgetGovernor, phase: str) -> dict[str, Any]:
        """Cumulative token-usage event for the live cockpit meters."""
        usage = self._store.token_usage(run_id)
        return {
            "event": "tokens.usage",
            "run_id": run_id,
            "phase": phase,
            "governor_mode": governor.mode().value,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

    def _working_diff(self) -> str:
        try:
            # Intent-to-add first: untracked files (greenfield creation) must
            # appear in the patch, or the evidence pack hides new work.
            subprocess.run(
                ["git", "add", "--intent-to-add", "-A"],
                cwd=self._repo_root,
                capture_output=True,
                timeout=60,
                check=False,
            )
            proc = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=self._repo_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return proc.stdout
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover - git is probed
            return ""

    ARCHITECT_STAGE_ATTEMPTS = 3
    ARCHITECT_STAGE_BACKOFF_SECONDS = 20.0

    async def _architect_stage_with_retry(
        self, architect: ArchitectAgent, issue_text: str, run_id: str, pack: EvidencePack
    ) -> tuple[RepositoryProfile, Plan]:
        """Analyze + decompose with bounded stage-level retries.

        The recovery ladder covers specialist TASKS, but the architect's two
        structured calls happen before any task exists - a flapping provider
        pool (live finding: HTTP 503 for the first ~3 minutes of a run) used
        to kill the whole run at t=0. Transport-class failures now retry the
        stage with backoff (each attempt is a full provider retry ladder);
        auth errors propagate immediately - credentials never heal.
        """
        window = architect.context_window
        planning_task = window.task_id if isinstance(window, StoreWindow) else "planning"
        last_error: RuntimeError | TimeoutError | ConnectionError | None = None
        for attempt in range(1, self.ARCHITECT_STAGE_ATTEMPTS + 1):
            try:
                profile = await architect.analyze_repository(summarize_repository(self._repo_root))
                plan = await architect.decompose(issue_text, profile)
                return profile, plan
            except ModelAuthError:
                raise
            except (RuntimeError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                if attempt >= self.ARCHITECT_STAGE_ATTEMPTS:
                    break
                pack.trace(
                    {
                        "event": "architect.retry",
                        "run_id": run_id,
                        "attempt": attempt,
                        "error": str(exc)[:200],
                    }
                )
                # Drop the half-built planning window so the retry starts
                # clean instead of replaying a failed turn.
                self._store.clear_window(architect.agent_id, planning_task)
                architect.context_window = StoreWindow(
                    self._store,
                    architect.agent_id,
                    planning_task,
                    stale_tool_results=architect.stale_tool_results,
                )
                await asyncio.sleep(self.ARCHITECT_STAGE_BACKOFF_SECONDS * attempt)
        assert last_error is not None  # the loop only exits via failure
        raise last_error

    def _no_architect_outcome(self, run_id: str, pack: EvidencePack) -> PipelineOutcome:
        pack.summary(
            build_summary(
                run_id,
                "(no issue)",
                "_no architect configured_",
                "no verification ran",
                "FAILED: no architect agent in configuration",
                [],
            )
        )
        return PipelineOutcome(
            run_id=run_id,
            success=False,
            outcome_line="FAILED: no architect agent in configuration",
            evidence_path=pack.path,
        )


def weak_specialties(specialties: set[str]) -> set[str]:
    """A slot covers its preset specialties plus the generic fallback."""
    return specialties | {"implementer"}
