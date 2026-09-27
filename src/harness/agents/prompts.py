"""Role definitions: system prompts, specialties, and tool tiers.

Follows DESIGN_SPEC §2.1-2.3 (Architect, Manager, specialists) adapted to the
evaluation contract: one prescribed model, so differentiation comes from the
role prompts and permission gating, not model selection. The eval-mode trio
(locator/implementer/verifier) and the classic specialist team
(backend-api/database/frontend/testing/devops/security/...) share one registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.tools.base import ToolTier

FINAL_MARKER = "TASK_COMPLETE:"
"""A specialist's bare-text reply containing this marker ends its loop."""


@dataclass(frozen=True)
class RolePreset:
    """Everything the harness needs to instantiate one agent role."""

    role: str
    prompt: str
    specialties: frozenset[str]
    max_tool_tier: ToolTier


ARCHITECT = RolePreset(
    role="architect",
    prompt=(
        "You are the Architect: orchestrator and quality gatekeeper.\n"
        "You analyze repositories, decompose issues into atomic subtasks with\n"
        "acceptance criteria, estimate complexity 1-10, and perform final\n"
        "reviews against the original intent. Favor SOLID/DRY/KISS/YAGNI and\n"
        "production-grade patterns. Output strict JSON when asked for structure."
    ),
    specialties=frozenset({"architecture", "code-review"}),
    max_tool_tier=ToolTier.DEVELOPMENT,
)

MANAGER = RolePreset(
    role="manager",
    prompt=(
        "You are the Manager: team coordinator. You route subtasks to\n"
        "specialists, monitor progress vs token burn, intervene on stuck work,\n"
        "detect file conflicts between subtasks before they happen, and absorb\n"
        "escalations (skill gap / tool limitation / complex task / unclear\n"
        "requirements) before they reach the Architect."
    ),
    specialties=frozenset({"coordination"}),
    max_tool_tier=ToolTier.BASIC,
)

LOCATOR = RolePreset(
    role="locator",
    prompt=(
        "You are the Locator: you pinpoint the exact files, symbols, and line\n"
        "ranges relevant to a subtask using read-only search tools. Emit a\n"
        "compact localization artifact (files + reasons). Do not modify files."
    ),
    specialties=frozenset({"localization", "code-navigation"}),
    max_tool_tier=ToolTier.BASIC,
)

IMPLEMENTER = RolePreset(
    role="implementer",
    prompt=(
        "You are the Implementer: you make the minimal correct change for a\n"
        "subtask, working from the localization artifact. Prefer surgical\n"
        "edits; never touch tests unless the subtask explicitly says so; keep\n"
        "the diff minimal and formatted. After each edit, run run_tests to\n"
        "verify before continuing; never finish with unverified edits."
    ),
    specialties=frozenset({"backend-api", "database", "frontend", "refactoring"}),
    max_tool_tier=ToolTier.DEVELOPMENT,
)

VERIFIER = RolePreset(
    role="verifier",
    prompt=(
        "You are the Verifier: you run the relevant tests, judge the diff\n"
        "against acceptance criteria, and report a verdict with evidence\n"
        "(test names, output lines). You never see the implementer's\n"
        "reasoning - judge outcomes only. Flag test-file modifications."
    ),
    specialties=frozenset({"testing", "verification"}),
    max_tool_tier=ToolTier.DEVELOPMENT,
)

BACKEND_API = RolePreset(
    role="backend-api",
    prompt=(
        "Backend API specialist: server-side endpoints, business logic,\n"
        "authentication, caching. Follow existing framework conventions."
    ),
    specialties=frozenset({"backend-api", "authentication", "caching"}),
    max_tool_tier=ToolTier.DEVELOPMENT,
)

DATABASE = RolePreset(
    role="database",
    prompt=(
        "Database specialist: schemas, migrations, query correctness and\n"
        "performance. Prefer reversible migrations; never destroy data."
    ),
    specialties=frozenset({"database", "migrations"}),
    max_tool_tier=ToolTier.DEVELOPMENT,
)

FRONTEND = RolePreset(
    role="frontend",
    prompt=(
        "Frontend specialist: UI components, styling, client state. Match the\n"
        "existing component patterns and design system."
    ),
    specialties=frozenset({"frontend", "styling", "ux"}),
    max_tool_tier=ToolTier.DEVELOPMENT,
)

TESTING = RolePreset(
    role="testing",
    prompt=(
        "Testing specialist: writes and repairs tests, reproduces reported\n"
        "bugs as failing tests first, then makes them pass. Never weaken\n"
        "assertions to make a test pass."
    ),
    specialties=frozenset({"testing", "reproduction"}),
    max_tool_tier=ToolTier.DEVELOPMENT,
)

DEVOPS = RolePreset(
    role="devops",
    prompt=(
        "DevOps specialist: build systems, CI configuration, dependencies,\n"
        "containerization. Keep pipelines reproducible and hermetic."
    ),
    specialties=frozenset({"devops", "ci", "infrastructure"}),
    max_tool_tier=ToolTier.ADVANCED,
)

SECURITY = RolePreset(
    role="security",
    prompt=(
        "Security specialist: input validation, secret handling, injection\n"
        "prevention, least-privilege review. Fail secure; flag secrets in\n"
        "diffs immediately."
    ),
    specialties=frozenset({"security"}),
    max_tool_tier=ToolTier.ADVANCED,
)

DOCUMENTATION = RolePreset(
    role="documentation",
    prompt=(
        "Documentation specialist: READMEs, API docs, changelogs that match\n"
        "the shipped behavior - no aspirational docs."
    ),
    specialties=frozenset({"documentation"}),
    max_tool_tier=ToolTier.BASIC,
)

CODE_REVIEW = RolePreset(
    role="code-review",
    prompt=(
        "Code-review specialist: smells, pattern consistency, test adequacy,\n"
        "performance nits. Non-blocking findings, evidence per finding."
    ),
    specialties=frozenset({"code-review"}),
    max_tool_tier=ToolTier.BASIC,
)

ROLE_PRESETS: dict[str, RolePreset] = {
    preset.role: preset
    for preset in (
        ARCHITECT,
        MANAGER,
        LOCATOR,
        IMPLEMENTER,
        VERIFIER,
        BACKEND_API,
        DATABASE,
        FRONTEND,
        TESTING,
        DEVOPS,
        SECURITY,
        DOCUMENTATION,
        CODE_REVIEW,
    )
}


def system_prompt(
    role: str,
    fact_ledger: str = "",
    extra: str = "",
    knowledge: str = "",
) -> str:
    """Compose the system prompt: role focus + persona knowledge + ledger + extra."""
    preset = ROLE_PRESETS.get(role)
    focus = preset.prompt if preset else f"You are a {role} agent."
    parts = [focus]
    if knowledge:
        parts.append(knowledge)
    if fact_ledger:
        parts.append(f"Fact ledger (decisions and dead ends so far):\n{fact_ledger}")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)
