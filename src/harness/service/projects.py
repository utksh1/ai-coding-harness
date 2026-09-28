"""Project registry for multi-project management (platform P4).

The orchestrator tracks the repositories it works on so the cockpits and the
`foreman` CLI can offer a project list instead of raw paths. The registry is
a best-effort JSON sidecar (`.harness/projects.json` next to the service's
cwd): registering a project never gates a run - /agent/run still accepts any
repo_root - it only makes the cockpit's project picker and the per-project
run history possible.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from harness.infrastructure.logging import get_logger

logger = get_logger(__name__)

PROJECTS_FILE = Path(".harness") / "projects.json"
"""Registry location (relative to the orchestrator's working directory)."""

PROJECTS_MAX = 64
"""Cap on registered projects: a cockpit picker, not a warehouse."""

GIT_TIMEOUT_SECONDS = 4.0


def _slugify(name: str) -> str:
    """Project id: lowercase alphanumerics/dashes, stable and path-safe."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


class ProjectStore:
    """Persistent, best-effort project registry.

    Every mutation re-reads and rewrites the file: the orchestrator is a
    single-process service, so last-write-wins is the concurrency model and
    any I/O damage degrades to an empty registry instead of an error.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else PROJECTS_FILE

    # -- persistence ------------------------------------------------------
    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): value
            for key, value in data.items()
            if isinstance(value, dict) and value.get("path")
        }

    def _save(self, registry: dict[str, dict[str, Any]]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(registry, sort_keys=True, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("project registry save failed", error=str(exc)[:200])

    # -- API surface ------------------------------------------------------
    def register(self, raw_path: str, name: str | None = None) -> dict[str, Any]:
        """Register a repository path; idempotent re-registration refreshes it.

        Raises ValueError for a path that does not exist or is not a directory.
        """
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"not a directory: {raw_path}")
        project_name = (name or path.name).strip() or path.name
        project_id = _slugify(project_name)
        registry = self._load()
        existing = registry.get(project_id)
        if existing is not None and existing.get("path") != str(path) and name is None:
            # Auto-named collision (two different repos both called "api"):
            # disambiguate with a path hash so registering never silently
            # overwrites. Explicit names are trusted (caller's choice).
            suffix = re.sub(r"[^a-z0-9]", "", str(path).lower())[-6:]
            project_id = f"{project_id}-{suffix}" if suffix else project_id
        registry.pop(project_id, None)
        registry[project_id] = {
            "id": project_id,
            "name": project_name,
            "path": str(path),
            "registered_at": _now_iso(),
        }
        while len(registry) > PROJECTS_MAX:
            registry.pop(next(iter(registry)))
        self._save(registry)
        return self.detail(project_id) or {"id": project_id, "name": project_name, "path": str(path)}

    def unregister(self, project_id: str) -> bool:
        registry = self._load()
        if project_id not in registry:
            return False
        del registry[project_id]
        self._save(registry)
        return True

    def get(self, project_id: str) -> dict[str, Any] | None:
        return self._load().get(project_id)

    def list_projects(self) -> list[dict[str, Any]]:
        """All registered projects, enriched with git + last-run info."""
        return sorted(
            (self._enrich(entry) for entry in self._load().values()),
            key=lambda p: p.get("name", ""),
        )

    def detail(self, project_id: str) -> dict[str, Any] | None:
        """One project enriched with git branch, HEAD, dirty state, and stats."""
        entry = self._load().get(project_id)
        return self._enrich(entry) if entry is not None else None

    @staticmethod
    def _enrich(entry: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(entry)
        enriched.update(_git_state(entry["path"]))
        enriched.update(_repo_stats(entry["path"]))
        return enriched

    def resolve(self, project_id_or_path: str) -> str | None:
        """Project id or raw path -> repo root path (project picker support)."""
        entry = self._load().get(project_id_or_path)
        if entry is not None:
            return entry["path"]
        if Path(project_id_or_path).is_dir():
            return str(Path(project_id_or_path).resolve())
        return None


def _now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _git_state(repo_path: str) -> dict[str, Any]:
    """Best-effort git metadata: never raises, never blocks the request long."""
    git_dir = Path(repo_path) / ".git"
    if not git_dir.exists():
        return {"git": False}
    try:
        branch = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.splitlines()
        return {
            "git": True,
            "branch": branch or "(detached)",
            "head": head or "?",
            "dirty": bool(status),
            "changed_files": len(status),
        }
    except (OSError, subprocess.SubprocessError):
        return {"git": True, "branch": "?", "head": "?", "dirty": False, "changed_files": 0}


def _repo_stats(repo_path: str) -> dict[str, Any]:
    """Repo-size + language signals for the project list (cheap heuristics)."""
    root = Path(repo_path)
    py_files = 0
    total_files = 0
    try:
        for path in root.rglob("*"):
            if not path.is_file() or any(
                part in {".git", ".harness", "node_modules", "__pycache__", ".venv"}
                for part in path.parts[len(root.parts) :]
            ):
                continue
            total_files += 1
            if path.suffix == ".py":
                py_files += 1
            if total_files >= 5000:  # bounded walk: a picker, not an indexer
                break
    except OSError:  # pragma: no cover - dir deleted mid-walk
        pass
    return {"files": total_files, "python_files": py_files}


def run_history(repo_path: str, results_dir: str, limit: int = 20) -> list[dict[str, Any]]:
    """Evidence-pack runs executed in one repo, newest first.

    Verdicts are read from each pack's events.jsonl (run.end event) - a
    best-effort read: a pack mid-write or missing the trace yields an
    'unknown' verdict rather than an error.
    """
    root = Path(repo_path) / results_dir
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            (d for d in root.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )[:limit]
    except OSError:  # pragma: no cover - stat raced with pack cleanup
        return []
    for pack_dir in candidates:
        runs.append(
            {
                "run_id": pack_dir.name,
                "verdict": _run_verdict(pack_dir),
                "finished_at": pack_dir.stat().st_mtime,
            }
        )
    return runs


def _run_verdict(pack_dir: Path) -> str:
    """VERIFIED / FAILED / INTERRUPTED from the run's own event trace."""
    try:
        lines = (pack_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unknown"
    verdict = "unknown"
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = event.get("event")
        if name == "run.end":
            verdict = "VERIFIED" if event.get("success") else "FAILED"
        elif name == "budget.exhausted" and verdict == "unknown":
            verdict = "STOPPED"
    return verdict
