"""Manager agent (milestone 2, issues 2.4-2.6): routing, monitoring, conflicts.

The assignment algorithm is DESIGN_SPEC §5.1 verbatim (specialty 40%,
availability 20%, load balance 20%, capability 20%) implemented as pure
functions so it is testable without any model. LLM judgment is reserved for
the few places the spec demands it (reframing unclear requirements).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.agents.architect import SubTask
from harness.agents.llm_agent import LLMAgent
from harness.agents.task import Task
from harness.orchestration.messages import (
    AgentStatus,
    CoordinationKind,
    CoordinationMessage,
    ErrorEscalation,
    StatusUpdate,
)

WEIGHTS = {"specialty": 0.4, "availability": 0.2, "load": 0.2, "capability": 0.2}
COLLABORATION_THRESHOLD = 7  # complexity above which top-3 specialists collaborate


@dataclass
class SpecialistSlot:
    """Manager's view of one specialist (its capacity and track record)."""

    agent_id: str
    specialties: set[str] = field(default_factory=set)
    current_tasks: int = 0
    max_concurrent: int = 2
    tokens_used: int = 0
    available_tools: set[str] = field(default_factory=set)
    model_tier: int = 3
    performance: float = 0.8  # historical success rate in specialty
    role: str = ""  # the agent's preset role (role-aware routing, audit §19)


def specialty_match(required: str | None, specialties: set[str], performance: float) -> float:
    """40% factor: exact specialty hit, scaled by historical success."""
    if not required:
        return 0.5  # unrouted tasks are neutral, not disqualified
    return 1.0 if required in specialties else 0.0


def role_matches_specialty(role: str, required: str | None) -> bool:
    """Whether an agent's *role* can take a task of this specialty.

    The Architect emits free-form specialties ("bugfix", "testing");
    ``SPECIALTY_ROLES`` (specialists module) maps them onto roles that can do
    the work. Live-run finding (M4 #58): without this, a "bugfix" task tied
    at 0.5 across every slot and landed on the read-only Locator.
    """
    if not required:
        return False
    from harness.agents.specialists import roles_for_specialty

    # One fallback, one source of truth: unknown specialties resolve via
    # ``roles_for_specialty`` to an editing-capable default ("implementer").
    # The old inline ``SPECIALTY_ROLES.get(required, (required,))`` treated the
    # specialty itself as a role name, so architect coinages like
    # "core-logic" matched nobody, every slot scored 0 on the 40% specialty
    # factor, and the availability/load tie-break routed implementation work
    # to the read-only Locator (live-run finding, parse repo).
    return role in roles_for_specialty(required)


def availability(current_tasks: int, max_concurrent: int) -> float:
    """20% factor: 1.0 when idle, 0.0 when saturated."""
    if max_concurrent <= 0:
        return 0.0
    return max(0.0, 1.0 - current_tasks / max_concurrent)


def load_balance(tokens_used: int, team_average: int) -> float:
    """20% factor: 1.0 at/below team average, decaying as overage grows."""
    if team_average <= 0:
        return 1.0
    if tokens_used <= team_average:
        return 1.0
    return max(0.0, 1.0 - (tokens_used - team_average) / team_average)


def capability(required_tools: set[str], available_tools: set[str], model_tier: int) -> float:
    """20% factor: tool coverage weighted by capability tier."""
    if not required_tools:
        coverage = 1.0
    else:
        coverage = len(required_tools & available_tools) / len(required_tools)
    return coverage * min(1.0, model_tier / 4)


def _specialty_hit(task: Task, slot: SpecialistSlot) -> float:
    """The 40% factor's raw hit: preset specialty, else role fallback."""
    hit = specialty_match(task.specialty, slot.specialties, slot.performance)
    if hit:
        return hit
    return 1.0 if role_matches_specialty(slot.role, task.specialty) else 0.0


def assignment_score(task: Task, slot: SpecialistSlot, team_average_tokens: int) -> float:
    """Weighted multi-factor score from DESIGN_SPEC §5.1."""
    return (
        _specialty_hit(task, slot) * WEIGHTS["specialty"]
        + availability(slot.current_tasks, slot.max_concurrent) * WEIGHTS["availability"]
        + load_balance(slot.tokens_used, team_average_tokens) * WEIGHTS["load"]
        + capability(set(task.required_tools), slot.available_tools, slot.model_tier)
        * WEIGHTS["capability"]
    )


def routing_breakdown(
    task: Task, slot: SpecialistSlot, team_average_tokens: int = 0
) -> dict[str, float]:
    """Per-factor contributions of one routing decision (cockpit events).

    Keys are the DESIGN_SPEC §5.1 factors; values are the *weighted*
    contributions, so their sum equals the slot's assignment score. The
    Manager's delegation moment - "why did the work go here" - made legible
    for the cockpit's routing bars (docs/cockpit-events.md §3).
    """
    return {
        "specialty": round(_specialty_hit(task, slot) * WEIGHTS["specialty"], 4),
        "availability": round(
            availability(slot.current_tasks, slot.max_concurrent) * WEIGHTS["availability"], 4
        ),
        "load": round(load_balance(slot.tokens_used, team_average_tokens) * WEIGHTS["load"], 4),
        "capability": round(
            capability(set(task.required_tools), slot.available_tools, slot.model_tier)
            * WEIGHTS["capability"],
            4,
        ),
    }


def rank_specialists(
    task: Task, slots: list[SpecialistSlot], team_average_tokens: int = 0
) -> list[tuple[SpecialistSlot, float]]:
    """All slots scored, highest first."""
    scored = [(slot, assignment_score(task, slot, team_average_tokens)) for slot in slots]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def assign_specialists(
    task: Task, slots: list[SpecialistSlot], team_average_tokens: int = 0
) -> list[str]:
    """Route per §5.1: top-3 collaborators above the complexity threshold, else top-1."""
    ranked = rank_specialists(task, slots, team_average_tokens)
    if not ranked:
        return []
    if task.complexity > COLLABORATION_THRESHOLD:
        return [slot.agent_id for slot, _ in ranked[:3]]
    return [ranked[0][0].agent_id]


def file_overlap(subtasks: list[SubTask]) -> dict[tuple[str, str], set[str]]:
    """Pairwise file-set intersections (issue 2.6 - conflict detection)."""
    overlaps: dict[tuple[str, str], set[str]] = {}
    for i, first in enumerate(subtasks):
        for second in subtasks[i + 1 :]:
            shared = set(first.files) & set(second.files)
            if shared:
                overlaps[(first.id, second.id)] = shared
    return overlaps


def execution_batches(subtasks: list[SubTask]) -> list[list[SubTask]]:
    """Group subtasks into sequential batches (issue 2.6 + M4 dependency safety).

    Within a batch file-sets are disjoint, so a batch can run in parallel
    worktrees. Safety rules (audit §11):

    - a subtask never enters a batch before every known `depends_on` id is
      placed in an earlier batch,
    - an empty/unknown file-set means sequential: it runs as a solo batch,
      never parallel with anything,
    - a dependency cycle degrades to strictly sequential solo batches.
    """
    remaining = list(subtasks)
    known_ids = {subtask.id for subtask in subtasks}
    batches: list[list[SubTask]] = []
    placed: set[str] = set()

    while remaining:
        ready = [
            subtask
            for subtask in remaining
            if all(dep in placed for dep in subtask.depends_on if dep in known_ids)
        ]
        if not ready:
            for subtask in remaining:
                batches.append([subtask])
            break
        batch: list[SubTask] = []
        used_files: set[str] = set()
        for subtask in ready:
            files = set(subtask.files)
            if not files or (used_files & files):
                continue  # unknown footprint -> solo batch, never parallel
            batch.append(subtask)
            used_files |= files
        if batch:
            for subtask in batch:
                remaining.remove(subtask)
                placed.add(subtask.id)
            batches.append(batch)
            continue
        # Nothing parallelizable this round: place the first ready solo.
        first = ready[0]
        remaining.remove(first)
        placed.add(first.id)
        batches.append([first])
    return batches


class ManagerAgent(LLMAgent):
    """Coordination-tier agent implementing the BaseManager contract."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("role", "manager")
        super().__init__(**kwargs)
        self.slots: dict[str, SpecialistSlot] = {}
        self.assignments: dict[str, str] = {}  # task_id -> agent_id

    def register_specialist(self, slot: SpecialistSlot) -> None:
        self.slots[slot.agent_id] = slot

    # -- BaseManager contract -------------------------------------------------
    async def assign_task(self, task: Task, agent_id: str) -> None:
        """Route `task` to the named specialist and log the decision."""
        if agent_id not in self.slots:
            msg = f"cannot assign to unknown specialist '{agent_id}'"
            raise ValueError(msg)
        self.assignments[task.id] = agent_id
        self.slots[agent_id].current_tasks += 1
        self.store.save_global(
            "assignments",
            {**(self.store.load_global("assignments") or {}), task.id: agent_id},
        )
        self.context_window.append(
            "assistant",
            f"assigned {task.id} -> {agent_id}",
        )

    async def monitor_progress(self) -> list[StatusUpdate]:
        """Poll specialists; active slots report load, idle slots report idle."""
        return [
            StatusUpdate(
                sender=agent_id,
                status=AgentStatus.WORKING if slot.current_tasks > 0 else AgentStatus.IDLE,
                detail=(
                    f"active tasks={slot.current_tasks} tokens={slot.tokens_used}"
                    if slot.current_tasks > 0
                    else ""
                ),
            )
            for agent_id, slot in self.slots.items()
        ]

    async def handle_escalation(self, escalation: ErrorEscalation) -> StatusUpdate:
        """Categorize and absorb an escalation (DESIGN_SPEC §2.2.4)."""
        category = categorize_escalation(escalation)
        if category == "skill_gap":
            return StatusUpdate(
                sender=self.agent_id,
                status=AgentStatus.BLOCKED,
                task_id=escalation.task_id,
                detail=f"reassign: skill gap ({escalation.error_type})",
            )
        if category == "tool_limitation":
            return StatusUpdate(
                sender=self.agent_id,
                status=AgentStatus.WORKING,
                task_id=escalation.task_id,
                detail=f"tool guidance: {escalation.message[:120]}",
            )
        if category == "complex_task":
            return StatusUpdate(
                sender=self.agent_id,
                status=AgentStatus.WORKING,
                task_id=escalation.task_id,
                detail="add collaborators",
            )
        if category == "unclear_requirements":
            return StatusUpdate(
                sender=self.agent_id,
                status=AgentStatus.WORKING,
                task_id=escalation.task_id,
                detail="reframe requirements",
            )
        return StatusUpdate(
            sender=self.agent_id,
            status=AgentStatus.BLOCKED,
            task_id=escalation.task_id,
            detail="escalate to architect",
        )

    async def reframe_requirements(self, task: Task, confusion: str) -> Task:
        """LLM-assisted clarification for 'unclear requirements' escalations."""
        data = await self.structured_call(
            'Reply with JSON: {"description": str}',
            f"Task '{task.title}' confused a specialist: {confusion}\n"
            f"Rewrite the description to be unambiguous.",
            '{"description": str}',
        )
        return task.model_copy(
            update={
                "description": data.get("description", task.description),
                "metadata": {**task.metadata, "reframed_by": self.agent_id},
            }
        )

    def coordination_log(
        self, kind: CoordinationKind, recipient: str, payload: dict[str, Any]
    ) -> CoordinationMessage:
        return CoordinationMessage(
            sender=self.agent_id, recipient=recipient, kind=kind, payload=payload
        )


def categorize_escalation(escalation: ErrorEscalation) -> str:
    """Map an escalation to a §2.2.4 category from its type and message."""
    error_type = escalation.error_type.lower()
    message = escalation.message.lower()
    if "unknown tool" in message or "permission" in message:
        return "tool_limitation"
    if "syntax" in error_type or "structure" in error_type:
        return "unclear_requirements"
    if escalation.severity.value == "transient":
        return "transient"
    if "complexity" in message:
        return "complex_task"
    if escalation.error_type in {"KeyError", "AttributeError"}:
        return "skill_gap"
    if escalation.attempt >= 2:
        return "complex_task"
    return "unknown"
