"""Search and editing tools (milestone 3, issues 3.1-3.2): grep + apply_edit.

`apply_edit` uses the search/replace format (improvement roadmap 1.2): the
model supplies exact existing text and its replacement. A whitespace-tolerant
fallback catches near-misses, and every successful edit returns the unified
diff so the trace shows precisely what changed.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from harness.security.input_guard import sanitize_path
from harness.tools.base import Tool, ToolResult, ToolTier

MAX_SEARCH_RESULTS = 60
MAX_SEARCH_BYTES_PER_FILE = 200_000


class SearchTextTool(Tool):
    """grep_search: regex text search across the repository."""

    name, tier = "search_text", ToolTier.BASIC
    description = (
        "Regex search across repo files. Args: pattern, path (optional "
        "subdirectory), glob (optional filename filter, e.g. '*.py')."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
        },
        "required": ["pattern"],
    }

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        if not arguments.get("pattern"):
            errors.append("'pattern' is required")
        try:
            re.compile(arguments.get("pattern", ""))
        except re.error as exc:
            errors.append(f"invalid regex: {exc}")
        return errors

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return True

    def execute(
        self, pattern: str, path: str = ".", glob: str | None = None, **_: Any
    ) -> ToolResult:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:  # defensive: validation should catch this
            return ToolResult(success=False, error=f"invalid regex: {exc}")
        # Empty/omitted path means repo root - models pass "" constantly
        # (live-run finding: six identical 'empty path' failures in one task).
        path = path or "."
        try:
            root = sanitize_path(self._root, path)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        if root.is_file():
            # A file path is a legitimate grep target (the model wants the
            # matches inside that one file); erroring forced read -> search
            # -> fail loops in live runs.
            return self._search_files([root], compiled, pattern)
        if not root.is_dir():
            return ToolResult(success=False, error=f"not a directory: {path}")
        candidates = root.rglob(glob) if glob else root.rglob("*")
        return self._search_files(candidates, compiled, pattern)

    def _search_files(self, candidates: Any, compiled: re.Pattern[str], pattern: str) -> ToolResult:
        matches: list[str] = []
        files_scanned = 0
        for file_path in candidates:
            if not file_path.is_file() or file_path.suffix in {".pyc", ".db"}:
                continue
            if any(
                part in {".git", ".venv", "__pycache__", ".harness"} for part in file_path.parts
            ):
                continue
            try:
                if file_path.stat().st_size > MAX_SEARCH_BYTES_PER_FILE:
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            files_scanned += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    rel = file_path.relative_to(self._root)
                    matches.append(f"{rel}:{line_number}: {line.strip()[:160]}")
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        matches.append(f"... [capped at {MAX_SEARCH_RESULTS} matches]")
                        return ToolResult(
                            success=True,
                            output="\n".join(matches),
                            data={
                                "matches": MAX_SEARCH_RESULTS,
                                "files_scanned": files_scanned,
                                "capped": True,
                            },
                        )
        if not matches:
            return ToolResult(
                success=True,
                output=f"no matches for {pattern!r}",
                data={"matches": 0, "files_scanned": files_scanned},
            )
        return ToolResult(
            success=True,
            output="\n".join(matches),
            data={"matches": len(matches), "files_scanned": files_scanned},
        )


def _whitespace_tolerant(lines: list[str], search_lines: list[str]) -> int | None:
    """Find `search_lines` in `lines` ignoring leading-whitespace differences."""

    def normalize(sequence: list[str]) -> list[str]:
        return [line.strip() for line in sequence]

    target = normalize(search_lines)
    if not target:
        return None
    for start in range(len(lines) - len(target) + 1):
        if normalize(lines[start : start + len(target)]) == target:
            return start
    return None


FUZZY_CONTEXT_LINES = 4
FUZZY_CONTEXT_MAX_LINES = 40
FUZZY_MIN_SIMILARITY = 0.4


def _closest_context(original: str, search: str) -> str | None:
    """Line-numbered region of `original` most similar to `search`.

    The single biggest live-run failure mode is a failed `apply_edit`
    followed by a full file re-read (40+ reads in one task, step budget
    gone). Returning the closest actual region lets the model correct its
    `search` text in ONE round-trip instead of read -> guess -> fail again.
    Returns ``None`` when nothing is reasonably similar (truly not found).
    """
    lines = original.splitlines()
    query = next((q.strip() for q in search.splitlines() if q.strip()), None)
    if not lines or not query:
        return None
    best_ratio, best_idx = 0.0, 0
    for index, line in enumerate(lines):
        ratio = difflib.SequenceMatcher(None, query, line.strip()).ratio()
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, index
    if best_ratio < FUZZY_MIN_SIMILARITY:
        return None
    span = min(len(search.splitlines()) + FUZZY_CONTEXT_LINES, FUZZY_CONTEXT_MAX_LINES)
    start = max(0, best_idx - FUZZY_CONTEXT_LINES)
    end = min(len(lines), best_idx + span)
    snippet = "\n".join(f"{n + 1:5d}| {lines[n]}" for n in range(start, end))
    return (
        f"closest actual region in the file (line {best_idx + 1}, "
        f"similarity {best_ratio:.2f}):\n{snippet}"
    )


class ApplyEditTool(Tool):
    """apply_edit: search/replace edit with fuzzy fallback and diff output."""

    name, tier = "apply_edit", ToolTier.DEVELOPMENT
    description = (
        "Edit a file: provide exact existing text ('search') and its "
        "replacement ('replace'). Whitespace differences are tolerated."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "search": {"type": "string"},
            "replace": {"type": "string"},
        },
        "required": ["path", "search", "replace"],
    }

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        for key in ("path", "search", "replace"):
            if not isinstance(arguments.get(key), str):
                errors.append(f"'{key}' must be a string")
        return errors

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return context.get("model_tier", 1) >= self.tier.value

    def execute(self, path: str, search: str, replace: str, **_: Any) -> ToolResult:
        try:
            target = sanitize_path(self._root, path)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))
        if not target.is_file():
            return ToolResult(success=False, error=f"not a file: {path}")
        try:
            with target.open(encoding="utf-8", newline="") as handle:
                original = handle.read()  # newline="" preserves CRLF exactly
        except OSError as exc:
            return ToolResult(success=False, error=f"cannot read {path}: {exc}")

        updated = self._apply(original, search, replace)
        if updated is None:
            context = _closest_context(original, search)
            if context:
                message = (
                    "search text not found in "
                    + path
                    + ". The 'search' argument must be the EXACT current file text. "
                    "Copy it from the actual region below and retry apply_edit:\n" + context
                )
            else:
                message = (
                    "search text not found (even whitespace-tolerant); "
                    f"read the file again and copy the exact text: {path}"
                )
            return ToolResult(success=False, error=message, data={"path": path})
        if updated == original:
            return ToolResult(
                success=False,
                error=(
                    "edit would not change the file: 'search' and 'replace' produce "
                    "identical content. If this change is already applied, verify "
                    "with run_tests and move on."
                ),
                data={"path": path},
            )
        self._backup(target, original)
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
        diff = "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        return ToolResult(success=True, output=diff, data={"path": path})

    def _apply(self, original: str, search: str, replace: str) -> str | None:
        if search in original:
            return original.replace(search, replace, 1)
        lines = original.splitlines(keepends=True)
        search_lines = search.splitlines(keepends=True)
        replace_lines = replace.splitlines(keepends=True)
        start = _whitespace_tolerant(
            [line.rstrip("\n") for line in lines],
            [line.rstrip("\n") for line in search_lines],
        )
        if start is None:
            return None
        # preserve the original block's line endings in the replacement
        eol = "\r\n" if lines[start].endswith("\r\n") else "\n"
        rebuilt = replace_lines
        if rebuilt and not rebuilt[-1].endswith("\n"):
            rebuilt[-1] = rebuilt[-1] + eol
        return "".join([*lines[:start], *rebuilt, *lines[start + len(search_lines) :]])

    def _backup(self, target: Path, original: str) -> None:
        backup_dir = self._root / ".harness" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        safe_name = target.relative_to(self._root.resolve()).name
        (backup_dir / f"{safe_name}.bak").write_text(original, encoding="utf-8")


class SyntaxCheckTool(Tool):
    """syntax_check: compile/parse validation for Python and JSON files."""

    name, tier = "syntax_check", ToolTier.DEVELOPMENT
    description = (
        "Validate syntax of edited files (Python via compile, JSON via "
        "parse). Pass one or more repo-relative paths."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
        "required": ["paths"],
    }

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root

    def validate_input(self, arguments: dict[str, Any]) -> list[str]:
        paths = arguments.get("paths")
        if not isinstance(paths, list) or not paths:
            return ["'paths' must be a non-empty array"]
        return []

    def check_permissions(self, context: dict[str, Any]) -> bool:
        return context.get("model_tier", 1) >= self.tier.value

    def execute(self, paths: list[str], **_: Any) -> ToolResult:
        findings: list[str] = []
        for rel in paths:
            try:
                target = sanitize_path(self._root, rel)
            except ValueError as exc:
                findings.append(f"{rel}: {exc}")
                continue
            if not target.is_file():
                findings.append(f"{rel}: not a file")
                continue
            text = target.read_text(encoding="utf-8", errors="replace")
            if target.suffix == ".py":
                try:
                    compile(text, str(target), "exec")
                except SyntaxError as exc:
                    findings.append(f"{rel}: line {exc.lineno}: {exc.msg}")
            elif target.suffix == ".json":
                try:
                    import json

                    json.loads(text)
                except json.JSONDecodeError as exc:
                    findings.append(f"{rel}: {exc}")
        if findings:
            return ToolResult(
                success=False,
                output="syntax errors found",
                error="\n".join(findings),
                data={"errors": findings},
            )
        return ToolResult(
            success=True, output="all files parse cleanly", data={"checked": len(paths)}
        )
