"""Tier 1 filesystem tools (milestone 3, issue 3.1): read and list operations.

All paths are confined to the repository root via `sanitize_path`; reads are
line-numbered and size-capped so tool output stays budget-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness.security.input_guard import sanitize_path
from harness.tools.base import Tool, ToolResult, ToolTier

MAX_READ_BYTES = 60_000
MAX_LIST_ENTRIES = 400
SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".harness",
    "node_modules",
    ".harness-bak",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".db", ".sqlite"}


def _summary_error(message: str) -> ToolResult:
    return ToolResult(success=False, error=message)


class ReadFileTool(Tool):
    """filesystem_read: line-numbered file contents with optional range."""

    name, tier = "filesystem_read", ToolTier.BASIC
    description = "Read a file's contents (line-numbered). Optional 1-based start/end lines."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["path"],
    }

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not arguments.get("path"):
            errors.append("'path' is required")
        for key in ("start_line", "end_line"):
            if key in arguments and (not isinstance(arguments[key], int) or arguments[key] < 1):
                errors.append(f"'{key}' must be a positive integer")
        return errors

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return True

    def execute(
        self, path: str, start_line: int | None = None, end_line: int | None = None, **_: Any
    ) -> ToolResult:
        # Models pass "" for the path when they mean the obvious thing;
        # erroring burns a whole round-trip. Default like filesystem_list.
        path = path or "."
        try:
            target = sanitize_path(self._root, path)
        except ValueError as exc:
            return _summary_error(str(exc))
        if not target.is_file():
            return _summary_error(f"not a file: {path}")
        try:
            raw = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _summary_error(f"cannot read {path}: {exc}")
        if len(raw) > MAX_READ_BYTES:
            raw = raw[:MAX_READ_BYTES] + f"\n... [truncated at {MAX_READ_BYTES} bytes]"
        lines = raw.splitlines()
        if start_line is not None or end_line is not None:
            start = (start_line or 1) - 1
            end = end_line if end_line is not None else len(lines)
            lines = lines[start:end]
            numbered = [f"{start + i + 1:5d}| {line}" for i, line in enumerate(lines)]
        else:
            numbered = [f"{i + 1:5d}| {line}" for i, line in enumerate(lines)]
        output = "\n".join(numbered) if numbered else "[empty file]"
        return ToolResult(success=True, output=output, data={"path": path, "lines": len(lines)})


class ListDirTool(Tool):
    """filesystem_list: gitignore-aware directory listing."""

    name, tier = "filesystem_list", ToolTier.BASIC
    description = "List a directory's entries (files with sizes, subdirectories marked)."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "defaults to repo root"}},
    }

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        return []

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return True

    def execute(self, path: str = ".", **_: Any) -> ToolResult:
        path = path or "."
        try:
            target = sanitize_path(self._root, path)
        except ValueError as exc:
            return _summary_error(str(exc))
        if not target.is_dir():
            return _summary_error(f"not a directory: {path}")
        entries: list[str] = []
        try:
            children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError as exc:
            return _summary_error(f"cannot list {path}: {exc}")
        for child in children:
            if child.name in SKIP_DIRS or child.suffix in SKIP_SUFFIXES:
                continue
            if child.is_dir():
                entries.append(f"{child.name}/")
            else:
                entries.append(f"{child.name} ({child.stat().st_size} bytes)")
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"... [capped at {MAX_LIST_ENTRIES} entries]")
                break
        header = f"{path} ({len(entries)} entries)"
        return ToolResult(
            success=True, output="\n".join([header, *entries]), data={"count": len(entries)}
        )


COMPACT_SUMMARY_MAX_CHARS = 4000


def compact_repo_summary(
    summary: dict[str, Any], max_chars: int = COMPACT_SUMMARY_MAX_CHARS
) -> str:
    """Compact, bounded serialization of a repo summary for prompts (#63).

    Empty fields are dropped (they carry no planning signal) and the result
    is hard-capped so a large repository can never balloon the Architect's
    analysis prompt. Truncation is explicit, never silent.
    """
    lean = {k: v for k, v in summary.items() if v}
    text = json.dumps(lean, sort_keys=True)
    if len(text) > max_chars:
        text = text[:max_chars] + f"... [repo summary truncated at {max_chars} chars]"
    return text


def summarize_repository(repo_root: Path) -> dict[str, Any]:
    """Deterministic repo summary for the Architect's analysis stage.

    Deliberately zero-LLM: extension counts, build/dependency files, and
    top-level layout - the cheap context the harness always starts from.
    """
    extensions: dict[str, int] = {}
    build_files: list[str] = []
    markers = {
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "package.json",
        "Makefile",
        "go.mod",
        "Cargo.toml",
        "setup.cfg",
        "pytest.ini",
        "tox.ini",
        ".github",
    }
    total_files = 0
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        total_files += 1
        suffix = path.suffix or "[none]"
        extensions[suffix] = extensions.get(suffix, 0) + 1
        if path.name in markers or any(part in markers for part in path.parts[1:]):
            build_files.append(str(path.relative_to(repo_root)))
    top_level = sorted(
        p.name
        for p in repo_root.iterdir()
        if not p.name.startswith(".") and p.name not in SKIP_DIRS
    )
    return {
        "root": str(repo_root),
        "total_files": total_files,
        "extensions": dict(sorted(extensions.items(), key=lambda kv: -kv[1])[:12]),
        "build_files": build_files,
        "top_level": top_level,
    }


class WriteFileTool(Tool):
    """filesystem_write: create, overwrite, or append to a repo file (tier 2).

    The creation hand the harness was missing (M5 review): `apply_edit` only
    modifies existing files. Overwrites keep a backup in `.harness/backups/`
    like `apply_edit`; `create` refuses to clobber an existing file.
    """

    name, tier = "filesystem_write", ToolTier.DEVELOPMENT
    description = (
        "Create or write a repo file. Args: path, content, "
        "mode ('create' [default] | 'overwrite' | 'append')."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["create", "overwrite", "append"]},
        },
        "required": ["path", "content"],
    }

    MAX_WRITE_BYTES = 512_000

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not isinstance(arguments.get("path"), str) or not arguments.get("path"):
            errors.append("'path' must be a non-empty string")
        if not isinstance(arguments.get("content"), str):
            errors.append("'content' must be a string")
        if arguments.get("mode", "create") not in {"create", "overwrite", "append"}:
            errors.append("'mode' must be create|overwrite|append")
        return errors

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return context.get("model_tier", 1) >= self.tier.value

    def execute(self, path: str, content: str = "", mode: str = "create", **_: Any) -> ToolResult:
        try:
            target = sanitize_path(self._root, path)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        if len(content) > self.MAX_WRITE_BYTES:
            return ToolResult(
                success=False,
                error=f"content exceeds {self.MAX_WRITE_BYTES} byte write cap",
            )
        existed = target.exists()
        if existed and mode == "create":
            return ToolResult(
                success=False,
                error=f"{path} already exists; use mode='overwrite' to replace it",
                data={"path": path},
            )
        if not existed and mode in {"overwrite", "append"}:
            return ToolResult(
                success=False,
                error=f"{path} does not exist; use mode='create'",
                data={"path": path},
            )
        if existed and mode == "overwrite":
            self._backup(target, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("a" if mode == "append" else "w", encoding="utf-8") as handle:
                handle.write(content)
        except OSError as exc:
            return ToolResult(success=False, error=f"cannot write {path}: {exc}")
        return ToolResult(
            success=True,
            output=f"{mode}: {path} ({len(content)} bytes)",
            data={"path": path, "mode": mode, "existed_before": existed},
        )

    def _backup(self, target: Path, path: str) -> None:
        backup_dir = self._root / ".harness" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / f"{target.name}.bak").write_text(
            target.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
        )


class GlobTool(Tool):
    """glob_files: pattern-matched file listing across the repo (tier 1)."""

    name, tier = "glob_files", ToolTier.BASIC
    description = (
        "List repo files matching a glob pattern (e.g. '**/*.py', 'src/*.js'). "
        "Args: pattern, path (optional root subdir)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"],
    }

    MAX_MATCHES = 200

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        return [] if arguments.get("pattern") else ["'pattern' is required"]

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return True

    def execute(self, pattern: str, path: str = ".", **_: Any) -> ToolResult:
        try:
            root = sanitize_path(self._root, path)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        if not root.is_dir():
            return ToolResult(success=False, error=f"not a directory: {path}")
        matches: list[str] = []
        for candidate in sorted(root.rglob(pattern)):
            if not candidate.is_file() or candidate.suffix in {".pyc", ".db", ".sqlite"}:
                continue
            if any(part in SKIP_DIRS for part in candidate.parts):
                continue
            rel = candidate.relative_to(self._root).as_posix()
            matches.append(rel)
            if len(matches) >= self.MAX_MATCHES:
                matches.append(f"... [capped at {self.MAX_MATCHES} files]")
                break
        if not matches:
            return ToolResult(
                success=True, output=f"no files match {pattern!r}", data={"matches": 0}
            )
        return ToolResult(success=True, output="\n".join(matches), data={"matches": len(matches)})
