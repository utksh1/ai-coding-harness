"""Python Agent Orchestrator service (platform P1, issue #70).

FastAPI wrapper around the proven engine: exposes the
TECHNICAL_IMPLEMENTATION.md §2.2 endpoints (architect analyze/decompose,
manager assign, specialist execute, status) plus `/agent/run` (full
pipeline) and `/health`. The graded core never imports this module - it
is an optional `harness[platform]` add-on.
"""

from __future__ import annotations

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


class RunRequest(BaseModel):
    issue: str = Field(min_length=1)
    repo_root: str = "."
    demo_mode: bool = False
    run_id: str | None = None
    """Gateway-supplied id: threaded through the pipeline so the task, the
    streamed events, and the evidence directory share ONE id."""


EVIDENCE_FILES = (
    "patch.diff",
    "summary.md",
    "test-report.md",
    "token-report.json",
    "baseline.json",
)
"""Whitelist of evidence-pack files exposed over HTTP (no traversal risk)."""


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


def _architect_of(pipeline: HarnessPipeline) -> Any:
    """The pipeline's architect, or a planning-phase fallback one."""
    if pipeline._architect is not None:
        return pipeline._architect
    from harness.agents.architect import build_architect

    return build_architect(
        "orchestrator-architect",
        {"provider": "service"},
        pipeline._provider,
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

    pipelines: dict[str, HarnessPipeline] = {}
    run_roots: dict[str, str] = _load_run_roots()
    # One pipeline run per repo at a time: the gateway retries /agent/run
    # after transport hiccups, and a retried POST landing while the original
    # run still lives would spawn a SECOND pipeline on the SAME working tree
    # (interleaved edits, no-op apply_edit errors, racing test runs - live
    # finding from the ec64a196/97ab7e14 collision). In-process only: an
    # orchestrator restart clears it with the dying run, which is exactly
    # when retries SHOULD be allowed.
    active_repo_runs: dict[str, str] = {}

    def _pipeline(
        repo_root: str, event_sink: Any = None, demo: bool | None = None
    ) -> HarnessPipeline:
        """One shared pipeline per (repo root, mode): agent/manager state must
        persist across requests (assign -> status -> execute).

        `demo` forces the scripted provider for THIS cache entry - the
        per-request checkbox must not silently hit the real model because a
        real provider was cached first (integration finding).
        """
        key = f"{repo_root}:demo" if demo else repo_root
        if key not in pipelines:
            cfg = _config()
            store = create_context_store(cfg.storage)
            key_env = cfg.models["default"].api_key_env
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
                or (not has_key and cfg.models["default"].provider != "fake")
            )
            resolved: Any
            if effective_demo and provider is None:
                from harness.infrastructure.model_providers.fake import build_demo_provider

                resolved = build_demo_provider(cfg.models["default"])
            else:
                resolved = provider or create_model_provider(cfg.models["default"])
            pipelines[key] = HarnessPipeline(
                repo_root=Path(repo_root),
                config=cfg,
                provider=resolved,
                store=store,
                event_sink=event_sink,
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
        run in flight.
        """
        cfg = _config()
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
        pipeline = _pipeline(
            request.repo_root, event_sink=publisher.sink_for(run_id), demo=request.demo_mode or None
        )
        key_env = _config().models["default"].api_key_env
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
            or (not has_key and _config().models["default"].provider != "fake")
        )
        try:
            outcome = await pipeline.run(
                request.issue,
                demo_mode=demo,
                run_id=run_id,
                event_sink=publisher.sink_for(run_id),
            )
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
            if active_repo_runs.get(request.repo_root) == run_id:
                active_repo_runs.pop(request.repo_root, None)
        return {
            "run_id": outcome.run_id,
            "success": outcome.success,
            "outcome": outcome.outcome_line,
            "evidence_path": str(outcome.evidence_path),
            "flags": outcome.flags,
        }

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
