"""Python Agent Orchestrator service (platform P1, issue #70).

FastAPI wrapper around the proven engine: exposes the
TECHNICAL_IMPLEMENTATION.md §2.2 endpoints (architect analyze/decompose,
manager assign, specialist execute, status) plus `/agent/run` (full
pipeline) and `/health`. The graded core never imports this module - it
is an optional `harness[platform]` add-on.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from harness import __version__
from harness.agents.prompts import ROLE_PRESETS
from harness.config import HarnessConfig
from harness.engine.evidence import EvidencePack
from harness.engine.pipeline import HarnessPipeline
from harness.infrastructure.context_store import create_context_store
from harness.infrastructure.logging import get_logger
from harness.infrastructure.model_providers import create_model_provider
from harness.service.events import RedisEventPublisher
from harness.service.models import MAX_PROFILES, ModelStore
from harness.service.projects import ProjectStore, run_history

logger = get_logger(__name__)


class AnalyzeRequest(BaseModel):
    repo_root: str = Field(min_length=1)


class DecomposeRequest(BaseModel):
    issue: str = Field(min_length=1)
    repo_root: str = Field(min_length=1)


class AssignRequest(BaseModel):
    task: dict[str, Any]
    agent_id: str = Field(min_length=1)


class ExecuteRequest(BaseModel):
    task: dict[str, Any]


class ProjectRequest(BaseModel):
    path: str = Field(min_length=1)
    name: str | None = None


class RunRequest(BaseModel):
    issue: str = Field(min_length=1)
    repo_root: str = "."
    demo_mode: bool = False
    run_id: str | None = None
    """Gateway-supplied id: threaded through the pipeline so the task, the
    streamed events, and the evidence directory share ONE id."""
    model_profile: str = "default"
    """Which `models:` entry powers this run (default/gemini/luna/...).
    Unknown names fall back to `default` so a stale cockpit profile never
    400s a run."""
    followup_of: str | None = None
    """Prior run id in the same repo: its summary + patch stats are prepended
    to the issue so the architect continues the session instead of starting
    cold (chat-style continuation, Codex-like)."""


class CancelRequest(BaseModel):
    run_id: str = Field(min_length=1)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ModelUpsertRequest(BaseModel):
    """Add or fully replace one model profile.

    Fields are ModelConfig's (provider/name/api_key_env/base_url/temperature/
    max_tokens/request_timeout_seconds/max_retries/tool_call_mode/extra);
    unknown keys are ignored, values are validated against the real schema.
    A KEY VALUE is never accepted - only the env var NAME holding it.
    `profile` is validated by the store (soft-fail body, house style) so the
    settings UI gets one error shape for every rejection.
    """

    profile: str = Field(min_length=1, max_length=40)
    fields: dict[str, Any] = Field(default_factory=dict)


class AgentModelRequest(BaseModel):
    """Re-bind one agent to a profile ("default" = follow the run picker)."""

    profile: str = Field(min_length=1, max_length=40)


EVIDENCE_FILES = (
    "patch.diff",
    "summary.md",
    "test-report.md",
    "token-report.json",
    "baseline.json",
)
"""Whitelist of evidence-pack files exposed over HTTP (no traversal risk)."""


FOLLOWUP_CONTEXT_MAX_CHARS = 1800
"""Prior-run context budget prepended to a follow-up issue: enough to carry
the verdict, the summary, and what changed; not enough to smuggle a whole
patch into the prompt."""


def _followup_context(run_roots: dict[str, str], followup_of: str, results_dir: str) -> str | None:
    """Build the continuation block for a follow-up run.

    Reads the prior run's evidence pack (summary.md + patch.diff + the
    run.end verdict from its event trace). Returns None when the prior run
    is unknown or left no pack - a follow-up to a vanished run degrades to
    a plain fresh issue rather than an error."""
    from harness.service.projects import _run_verdict

    if not _safe_run_id(followup_of):
        return None
    repo_root = run_roots.get(followup_of)
    if repo_root is None:
        return None
    pack_dir = Path(repo_root) / results_dir / followup_of
    if not pack_dir.is_dir():
        return None
    parts = [f"CONTINUATION of run {followup_of} in this repository."]
    parts.append(f"Previous verdict: {_run_verdict(pack_dir)}.")
    evidence_found = False
    try:
        summary = (pack_dir / "summary.md").read_text(encoding="utf-8").strip()
        if summary:
            evidence_found = True
            parts.append(
                "Previous run summary (what was done, what was left):\n"
                + summary[: FOLLOWUP_CONTEXT_MAX_CHARS // 2]
            )
    except OSError:
        pass
    try:
        diff = (pack_dir / "patch.diff").read_text(encoding="utf-8")
        files = sum(1 for line in diff.splitlines() if line.startswith("diff --git "))
        additions = sum(
            1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
        )
        if files:
            evidence_found = True
            parts.append(
                f"Working tree already carries the previous patch: {files} file(s), "
                f"+{additions}/-{deletions} lines. Do NOT redo finished work."
            )
    except OSError:
        pass
    # Only a verdict and nothing else is not continuation evidence: a pack
    # with no summary and no patch degrades to a plain fresh issue.
    return "\n\n".join(parts) if evidence_found else None


def _safe_run_id(run_id: str) -> bool:
    """Run ids are hex-ish tokens; anything path-like is refused."""
    return bool(run_id) and "/" not in run_id and "\\" not in run_id and ".." not in run_id


RUN_ROOT_REGISTRY_MAX = 256
"""Cap on remembered run->repo mappings (evicts oldest first)."""

RUN_ROOTS_FILE = Path(".harness") / "run-roots.json"
"""Where the run->repo registry persists: evidence links must survive an
orchestrator restart (the cockpit's Diff tab fetches patch.diff long after
the run finished). Best-effort by design - the graded eval path never
reads it."""


def _load_run_roots() -> dict[str, str]:
    """Read the persisted registry; any damage yields an empty registry."""
    try:
        data = json.loads(RUN_ROOTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_run_roots(registry: dict[str, str]) -> None:
    """Persist the registry; persistence is a convenience, never a gate."""
    try:
        RUN_ROOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RUN_ROOTS_FILE.write_text(json.dumps(registry, sort_keys=True, indent=0), encoding="utf-8")
    except OSError:
        pass


def _remember_run_root(
    registry: dict[str, str], run_id: str, repo_root: str, cap: int = RUN_ROOT_REGISTRY_MAX
) -> None:
    """Record where a run executed so evidence lookups need no repo_root.

    The gateway's evidence proxy does not know the target repo; the
    orchestrator does (it executed the run there). The registry wins over
    query params and falls back to them for unknown runs."""
    registry.pop(run_id, None)
    registry[run_id] = repo_root
    while len(registry) > cap:
        registry.pop(next(iter(registry)))


def _service_provider_factory(effective_demo: bool) -> Any:
    """Provider factory for the service path: one provider instance per
    agent (per-agent model binding - architect on one profile, specialists
    on another - is resolved by the pipeline, not collapsed here). The demo
    leg is role-aware: each agent's scripted part matches its calls."""

    if effective_demo:
        from harness.infrastructure.model_providers.fake import build_demo_provider

        def _demo_factory(
            model_cfg: Any, role: str | None = None, agent_id: str | None = None
        ) -> Any:
            return build_demo_provider(model_cfg, role=role)

        return _demo_factory

    def _real_factory(model_cfg: Any, role: str | None = None, agent_id: str | None = None) -> Any:
        return create_model_provider(model_cfg)

    return _real_factory


def _architect_of(pipeline: HarnessPipeline) -> Any:
    """The pipeline's architect, or a planning-phase fallback one."""
    if pipeline._architect is not None:
        return pipeline._architect
    from harness.agents.architect import build_architect

    provider = pipeline._provider or pipeline._provider_for(pipeline._model_profile)
    return build_architect(
        "orchestrator-architect",
        {"provider": "service"},
        provider,
        pipeline._store,
        governor=_service_governor(pipeline),
    )


def _service_governor(pipeline: HarnessPipeline) -> Any:
    from harness.engine.budget import BudgetGovernor

    return BudgetGovernor(pipeline._store, pipeline._config.budget, "service")


def create_app(
    config: HarnessConfig | None = None,
    provider: Any = None,
    redis_client: Any = None,
) -> FastAPI:
    """Build the orchestrator app.

    `provider`/`redis_client` are injection seams for tests; production
    builds the configured model provider and reads REDIS_URL.
    """
    app = FastAPI(title="Foreman Agent Orchestrator", version=__version__)
    resolved_config = config
    publisher = RedisEventPublisher(client=redis_client)

    def _config() -> HarnessConfig:
        nonlocal resolved_config
        if resolved_config is None:
            from harness.config import load_config

            resolved_config = load_config()
        return resolved_config

    def _effective_config() -> HarnessConfig:
        """harness.yaml overlaid with the runtime model settings
        (ModelStore: profile adds/edits/removals + per-agent bindings).

        Identity when no overrides exist; a frozen model_copy otherwise, so
        every consumer of `_config()` that must reflect settings changes
        (model resolution, pipeline construction, the agents roster) goes
        through this instead.
        """
        return model_store.apply(_config())

    def _drop_pipeline_cache() -> int:
        """Model settings changed: cached pipelines hold providers built from
        the OLD profiles. Clearing the cache means the NEXT run (or agent
        request) builds fresh from the effective config. An in-flight run
        keeps its own pipeline object and finishes on the models it started
        with - a settings change never re-routes a live run underneath it."""
        dropped = len(pipelines)
        pipelines.clear()
        return dropped

    pipelines: dict[str, HarnessPipeline] = {}
    run_roots: dict[str, str] = _load_run_roots()
    projects = ProjectStore()
    model_store = ModelStore()
    active_run_tasks: dict[str, asyncio.Task[Any]] = {}
    # One pipeline run per repo at a time: the gateway retries /agent/run
    # after transport hiccups, and a retried POST landing while the original
    # run still lives would spawn a SECOND pipeline on the SAME working tree
    # (interleaved edits, no-op apply_edit errors, racing test runs - live
    # finding from the ec64a196/97ab7e14 collision). In-process only: an
    # orchestrator restart clears it with the dying run, which is exactly
    # when retries SHOULD be allowed.
    active_repo_runs: dict[str, str] = {}

    def _resolve_model(profile_name: str) -> Any:
        """Model config for a named profile; unknown names fall back to
        `default` (a stale cockpit picker must never 400 a run)."""
        cfg = _effective_config()
        return cfg.models.get(profile_name) or cfg.models["default"]

    def _pipeline(
        repo_root: str,
        event_sink: Any = None,
        demo: bool | None = None,
        model_profile: str = "default",
    ) -> HarnessPipeline:
        """One shared pipeline per (repo root, profile, mode): agent/manager
        state must persist across requests (assign -> status -> execute).

        `demo` forces the scripted provider for THIS cache entry - the
        per-request checkbox must not silently hit the real model because a
        real provider was cached first (integration finding). Model profiles
        participate in the key: two profiles on one repo are different
        agents with different providers.
        """
        key = f"{repo_root}:{model_profile}:demo" if demo else f"{repo_root}:{model_profile}"
        if key not in pipelines:
            cfg = _effective_config()
            store = create_context_store(cfg.storage)
            model_cfg = _resolve_model(model_profile)
            key_env = model_cfg.api_key_env
            has_key = bool(
                os.environ.get(key_env)
                or os.environ.get("AI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("CODEX_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
            )
            effective_demo = (
                demo
                or (os.environ.get("HARNESS_DEMO") == "1")
                or (not has_key and model_cfg.provider != "fake")
            )
            resolved: Any = None
            factory: Any = None
            if provider is not None:
                # Test injection: one provider overrides every binding; no
                # factory (the pipeline must not build around the injection).
                resolved = provider
            else:
                factory = _service_provider_factory(effective_demo)

            pipelines[key] = HarnessPipeline(
                repo_root=Path(repo_root),
                config=cfg,
                provider=resolved,
                provider_factory=factory,
                store=store,
                event_sink=event_sink,
                model_profile=model_profile,
            )
        return pipelines[key]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "orchestrator", "version": __version__}

    @app.get("/api/agents")
    async def agents() -> dict[str, Any]:
        """The configured agent roster (gateway proxies this to the cockpit).

        v2 contract (docs/cockpit-events.md): specialties, tool tier, and the
        hierarchy level so the cockpits can render the org tree without a
        run in flight. Reflects the EFFECTIVE config: a per-agent model
        binding made in settings shows up here immediately.
        """
        cfg = _effective_config()
        levels = {"architect": 1, "manager": 2}
        return {
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "role": agent.role,
                    "model": agent.model,
                    "specialties": sorted(
                        ROLE_PRESETS[agent.role].specialties
                        if agent.role in ROLE_PRESETS
                        else {agent.role}
                    ),
                    "tool_tier": ROLE_PRESETS[agent.role].max_tool_tier.value
                    if agent.role in ROLE_PRESETS
                    else 3,
                    "level": levels.get(agent.role, 3),
                }
                for agent in cfg.agents
                if agent.enabled
            ]
        }

    @app.get("/api/models")
    async def models() -> dict[str, Any]:
        """EFFECTIVE model profiles + per-agent bindings (never key values).

        The Models settings view renders from this: every profile the
        harness can run on (yaml-seeded plus runtime-added, minus removed),
        which are runtime-managed, and what each agent is currently bound
        to. `api_key_env` is the NAME of the env var holding the key - the
        key itself never crosses this boundary.
        """
        base = _config()
        effective = _effective_config()
        bindings = model_store.agent_bindings()
        overrides = model_store.overrides_summary()
        return {
            "default_profile": "default",
            "profiles": [
                {
                    "profile": name,
                    "provider": m.provider,
                    "model": m.name,
                    "api_key_env": m.api_key_env,
                    "base_url": m.base_url,
                    "managed": name in overrides["profiles"],
                    "from_yaml": name in base.models,
                }
                for name, m in sorted(effective.models.items())
            ],
            "agents": {
                agent.agent_id: {
                    "model": agent.model,
                    "role": agent.role,
                    "overridden": agent.agent_id in bindings,
                }
                for agent in effective.agents
                if agent.enabled
            },
            "overrides": overrides,
            "limits": {"max_profiles": MAX_PROFILES},
        }

    @app.post("/api/models")
    async def upsert_model(request: ModelUpsertRequest) -> dict[str, Any]:
        """Add or fully replace one profile (settings-based model management).

        Takes effect on the NEXT run: the pipeline cache is dropped so new
        pipelines build from the new profile. An in-flight run finishes on
        the models it started with.
        """
        ok, detail = model_store.upsert_profile(request.profile, request.fields, _config().models)
        if not ok:
            return {"saved": False, "error": detail}
        dropped = _drop_pipeline_cache()
        return {"saved": True, "profile": request.profile, "pipelines_rebuilt": dropped}

    @app.patch("/api/models/{profile}")
    async def patch_model(profile: str, request: ModelUpsertRequest) -> dict[str, Any]:
        """Edit an existing EFFECTIVE profile: current values + patch -> save.

        Editing a yaml-seeded profile stores a full override (the yaml file
        is never rewritten by the API).
        """
        effective = _effective_config()
        current = effective.models.get(profile)
        if current is None:
            return {"saved": False, "error": f"unknown profile '{profile}'"}
        if request.profile != profile:
            return {"saved": False, "error": "profile name cannot be renamed in place"}
        merged = {**current.model_dump(), **request.fields}
        ok, detail = model_store.upsert_profile(profile, merged, _config().models)
        if not ok:
            return {"saved": False, "error": detail}
        dropped = _drop_pipeline_cache()
        return {"saved": True, "profile": profile, "pipelines_rebuilt": dropped}

    @app.delete("/api/models/{profile}")
    async def delete_model(profile: str) -> dict[str, Any]:
        ok, detail = model_store.remove_profile(profile, _config().models)
        if not ok:
            return {"saved": False, "error": detail}
        dropped = _drop_pipeline_cache()
        return {"saved": True, "removed": profile, "pipelines_rebuilt": dropped}

    @app.post("/api/models/reset")
    async def reset_models() -> dict[str, Any]:
        """Drop every runtime override: back to harness.yaml as written."""
        model_store.reset()
        dropped = _drop_pipeline_cache()
        return {"saved": True, "reset": True, "pipelines_rebuilt": dropped}

    @app.put("/api/agents/{agent_id}/model")
    async def set_agent_model(agent_id: str, request: AgentModelRequest) -> dict[str, Any]:
        """Re-bind one agent to any effective profile, individually.

        'default' un-binds: the agent follows the run-level profile picker.
        """
        ok, detail = model_store.set_agent_model(
            agent_id, request.profile, _config().models, _config().agents
        )
        if not ok:
            return {"saved": False, "error": detail}
        dropped = _drop_pipeline_cache()
        return {
            "saved": True,
            "agent": agent_id,
            "profile": request.profile,
            "pipelines_rebuilt": dropped,
        }

    @app.get("/api/fs")
    async def browse_fs(path: str = ".") -> dict[str, Any]:
        """Directory listing for the cockpit's folder picker.

        Read-only, one level at a time: {path, parent, entries[{name, is_dir,
        size}]} with directories first. The picker is the whole point of
        "open this agent in a folder" - the server it runs on is the machine
        whose folders you open, so this lists the orchestrator host's
        filesystem (the gateway stays the only exposed surface). Errors
        degrade to {found: false, error} - never a 500.
        """
        try:
            target = Path(path).expanduser().resolve()
        except (OSError, ValueError) as exc:
            return {"found": False, "error": f"invalid path: {exc}"}
        try:
            if not target.exists():
                return {"found": False, "error": f"not found: {target}"}
            if not target.is_dir():
                return {"found": False, "error": f"not a directory: {target}"}
            entries = []
            for child in sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
                try:
                    is_dir = child.is_dir()
                    entries.append(
                        {
                            "name": child.name,
                            "is_dir": is_dir,
                            "size": 0 if is_dir else child.stat().st_size,
                        }
                    )
                except OSError:
                    continue  # unreadable entry: skip, not fail
            return {
                "found": True,
                "path": str(target),
                "parent": str(target.parent) if str(target.parent) != str(target) else None,
                "entries": entries[:500],
                "truncated": len(entries) > 500,
            }
        except PermissionError:
            return {"found": False, "error": f"permission denied: {target}"}
        except OSError as exc:
            return {"found": False, "error": str(exc)}

    @app.post("/api/projects")
    async def register_project(request: ProjectRequest) -> dict[str, Any]:
        try:
            project = projects.register(request.path, request.name)
        except ValueError as exc:
            return {"registered": False, "error": str(exc)}
        return {"registered": True, "project": project}

    @app.get("/api/projects")
    async def list_projects() -> dict[str, Any]:
        return {
            "projects": [
                {**p, "active_run": active_repo_runs.get(p["path"])}
                for p in projects.list_projects()
            ]
        }

    @app.get("/api/projects/{project_id}")
    async def project_detail(project_id: str) -> dict[str, Any]:
        project = projects.detail(project_id)
        if project is None:
            return {"found": False, "error": f"unknown project {project_id}"}
        return {
            "found": True,
            "project": {**project, "active_run": active_repo_runs.get(project["path"])},
            "runs": run_history(project["path"], _config().run.results_dir),
        }

    @app.delete("/api/projects/{project_id}")
    async def unregister_project(project_id: str) -> dict[str, Any]:
        return {"removed": projects.unregister(project_id)}

    @app.get("/api/projects/{project_id}/resolve")
    async def resolve_project(project_id: str) -> dict[str, Any]:
        """Project id (or raw path) -> absolute repo root: the cockpit's
        project picker talks ids, /agent/run wants a path."""
        resolved = projects.resolve(project_id)
        if resolved is None:
            return {"found": False, "error": f"unknown project {project_id}"}
        return {"found": True, "path": resolved}

    @app.post("/agent/architect/analyze")
    async def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        from harness.tools.filesystem import summarize_repository

        pipeline = _pipeline(request.repo_root)
        architect = _architect_of(pipeline)
        profile = await architect.analyze_repository(summarize_repository(pipeline._repo_root))
        return {"profile": profile.model_dump()}

    @app.post("/agent/architect/decompose")
    async def decompose(request: DecomposeRequest) -> dict[str, Any]:
        from harness.tools.filesystem import summarize_repository

        pipeline = _pipeline(request.repo_root)
        architect = _architect_of(pipeline)
        plan = await architect.decompose(
            request.issue,
            await architect.analyze_repository(summarize_repository(pipeline._repo_root)),
        )
        return {"plan": plan.model_dump()}

    @app.post("/agent/manager/assign")
    async def assign(request: AssignRequest) -> dict[str, Any]:
        from harness.agents.task import Task

        pipeline = _pipeline(".")
        manager = pipeline._manager
        if manager is None:
            return {"assigned": False, "detail": "no manager configured"}
        task = Task.model_validate(request.task)
        await manager.assign_task(task, request.agent_id)
        return {"assigned": True, "task": task.id, "agent": request.agent_id}

    @app.post("/agent/specialist/execute")
    async def execute(request: ExecuteRequest) -> dict[str, Any]:
        from harness.agents.task import Task

        task = Task.model_validate(request.task)
        pipeline = _pipeline(".")
        agent_id, agent = next(iter(pipeline._agents.items()))
        result = await agent.execute_task(task)
        return {"result": result.model_dump(), "agent": agent_id}

    @app.get("/agent/status/{agent_id}")
    async def status(agent_id: str) -> dict[str, Any]:
        pipeline = _pipeline(".")
        agent = pipeline._agents.get(agent_id)
        if agent is None:
            return {"agent_id": agent_id, "known": False}
        update = agent.report_status()
        return {
            "agent_id": agent_id,
            "known": True,
            "status": update.status.value,
            "task_id": update.task_id,
        }

    @app.post("/agent/run")
    async def run(request: RunRequest) -> dict[str, Any]:
        import uuid

        run_id = request.run_id or uuid.uuid4().hex[:12]
        if not _safe_run_id(run_id):
            return {"success": False, "error": "invalid run_id", "outcome": "FAILED"}
        busy_run = active_repo_runs.get(request.repo_root)
        if busy_run is not None:
            # Synchronous check-and-set (no await between): concurrent POSTs
            # serialize on the event loop, so the first one wins the repo.
            reason = (
                f"run {busy_run} already active in this repo"
                if busy_run == run_id
                else f"repo busy: run {busy_run} active"
            )
            logger.warning("run rejected: repo busy", run_id=run_id, busy_run=busy_run)
            publisher.publish(
                run_id,
                {
                    "event": "run.failed",
                    "run_id": run_id,
                    "error": f"REJECTED: {reason}",
                    "stage": "admission",
                },
            )
            publisher.publish(run_id, {"event": "run.end", "run_id": run_id, "success": False})
            return {
                "run_id": run_id,
                "success": False,
                "outcome": f"REJECTED: {reason}",
                "evidence_path": "",
                "flags": ["concurrent run rejected"],
            }
        active_repo_runs[request.repo_root] = run_id
        _remember_run_root(run_roots, run_id, request.repo_root)
        _save_run_roots(run_roots)
        model_cfg = _resolve_model(request.model_profile)
        pipeline = _pipeline(
            request.repo_root,
            event_sink=publisher.sink_for(run_id),
            demo=request.demo_mode or None,
            model_profile=request.model_profile,
        )
        key_env = model_cfg.api_key_env
        has_key = bool(
            os.environ.get(key_env)
            or os.environ.get("AI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("CODEX_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        demo = (
            request.demo_mode
            or (os.environ.get("HARNESS_DEMO") == "1")
            or (not has_key and model_cfg.provider != "fake")
        )
        # Chat-style continuation: a follow-up carries the prior run's
        # verdict + summary + patch footprint so the architect continues the
        # session instead of re-planning blind.
        issue_text = request.issue
        context = _followup_context(run_roots, request.followup_of or "", _config().run.results_dir)
        if context:
            issue_text = f"{context}\n\n---\n\nFOLLOW-UP REQUEST: {request.issue}"
            logger.info(
                "followup context attached",
                run_id=run_id,
                followup_of=request.followup_of,
                context_chars=len(context),
            )
        # Cancellation surface: the pipeline runs as a tracked asyncio task
        # so POST /agent/cancel can stop it mid-flight (chat stop button).
        run_task: asyncio.Task[Any] | None = None
        try:
            run_task = asyncio.ensure_future(
                pipeline.run(
                    issue_text,
                    demo_mode=demo,
                    run_id=run_id,
                    event_sink=publisher.sink_for(run_id),
                )
            )
            active_run_tasks[run_id] = run_task
            outcome = await run_task
        except asyncio.CancelledError:
            if run_task is not None and run_task.cancelled():
                # Cancelled via /agent/cancel: finalize honestly, never hang.
                publisher.publish(
                    run_id,
                    {
                        "event": "run.failed",
                        "run_id": run_id,
                        "error": "CANCELLED by user",
                        "stage": "cancel",
                    },
                )
                publisher.publish(
                    run_id,
                    {
                        "event": "run.end",
                        "run_id": run_id,
                        "success": False,
                        "stop_reason": "cancelled",
                    },
                )
                return {
                    "run_id": run_id,
                    "success": False,
                    "outcome": "CANCELLED: run stopped by user",
                    "evidence_path": "",
                    "flags": ["cancelled"],
                }
            raise  # pragma: no cover - handler-task cancel (client drop)
        except Exception as exc:  # transport death mid-run: fail honestly, never hang
            logger.error("agent run crashed", run_id=run_id, error=str(exc)[:300])
            publisher.publish(
                run_id, {"event": "run.failed", "run_id": run_id, "error": str(exc)[:300]}
            )
            publisher.publish(run_id, {"event": "run.end", "run_id": run_id, "success": False})
            return {
                "run_id": run_id,
                "success": False,
                "outcome": f"FAILED: model transport error ({type(exc).__name__})",
                "evidence_path": "",
                "flags": [f"transport error: {type(exc).__name__}"],
            }
        finally:
            active_run_tasks.pop(run_id, None)
            if active_repo_runs.get(request.repo_root) == run_id:
                active_repo_runs.pop(request.repo_root, None)
        return {
            "run_id": outcome.run_id,
            "success": outcome.success,
            "outcome": outcome.outcome_line,
            "evidence_path": str(outcome.evidence_path),
            "flags": outcome.flags,
        }

    @app.post("/agent/cancel")
    async def cancel_run(request: CancelRequest) -> dict[str, Any]:
        """Stop a live run: the chat 'stop' button.

        Cancels the pipeline's asyncio task; the /agent/run handler observes
        the cancellation, emits run.failed/run.end, and answers its (patient,
        detached) caller honestly. Unknown or already-finished runs are
        reported as such - cancellation is idempotent."""
        run_id = request.run_id
        if not _safe_run_id(run_id):
            return {"cancelled": False, "error": "invalid run_id"}
        task = active_run_tasks.get(run_id)
        if task is None or task.done():
            return {"cancelled": False, "detail": f"run {run_id} is not active"}
        task.cancel()
        return {"cancelled": True, "run_id": run_id}

    @app.get("/evidence/latest")
    async def evidence_latest(repo_root: str = ".") -> dict[str, Any]:
        results = _config().run.results_dir
        from pathlib import Path

        root = Path(repo_root) / results
        if not root.is_dir():
            return {"found": False}
        runs = sorted(
            (d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True
        )
        pack = EvidencePack(root, runs[0].name) if runs else None
        return {
            "found": pack is not None,
            "run_id": pack.run_id if pack else None,
            "path": str(pack.path) if pack else None,
        }

    @app.get("/api/evidence/{run_id}/files")
    async def evidence_files(run_id: str, repo_root: str = ".") -> dict[str, Any]:
        """List one run's evidence-pack files (cockpit diff/report loading).

        `repo_root` is resolved from the run registry first: the gateway's
        proxy calls without a repo_root, and the orchestrator already knows
        where each run executed."""
        if not _safe_run_id(run_id):
            return {"found": False, "error": "invalid run_id"}
        root = Path(run_roots.get(run_id) or repo_root) / _config().run.results_dir
        pack_dir = root / run_id
        if not pack_dir.is_dir():
            return {"found": False, "run_id": run_id}
        files = sorted(item.name for item in pack_dir.iterdir() if item.is_file())
        return {"found": True, "run_id": run_id, "files": files}

    @app.get("/api/evidence/{run_id}/file/{name}")
    async def evidence_file(run_id: str, name: str, repo_root: str = ".") -> dict[str, Any]:
        """Serve one whitelisted evidence file's text content."""
        if not _safe_run_id(run_id) or name not in EVIDENCE_FILES:
            return {"found": False, "error": "file not exposed"}
        root = run_roots.get(run_id) or repo_root
        path = Path(root) / _config().run.results_dir / run_id / name
        if not path.is_file():
            return {"found": False, "run_id": run_id, "name": name}
        return {
            "found": True,
            "run_id": run_id,
            "name": name,
            "content": path.read_text(encoding="utf-8"),
        }

    return app
