"""Search, edit, and syntax tools (issues 3.1-3.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.editing import ApplyEditTool, SearchTextTool, SyntaxCheckTool


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    )
    return tmp_path


def test_search_text_matches(repo) -> None:
    result = SearchTextTool(repo).execute(pattern=r"def \w+\(a, b\)")
    assert result.success and result.data["matches"] == 2
    assert "src/calc.py:1" in result.output


def test_search_text_no_match_and_validation(repo) -> None:
    tool = SearchTextTool(repo)
    assert tool.validate_input({}) != []
    assert tool.validate_input({"pattern": "("}) != []  # bad regex
    assert tool.validate_input({"pattern": "def"}) == []
    result = tool.execute(pattern="quantum_loop")
    assert result.success and result.data["matches"] == 0


def test_search_text_glob_and_errors(repo) -> None:
    (repo / "notes.md").write_text("def not code\n")
    result = SearchTextTool(repo).execute(pattern="def", glob="*.py")
    assert "notes.md" not in result.output
    # A file path is a legitimate single-file grep target (live-run finding:
    # erroring here forced read -> search -> fail loops).
    single = SearchTextTool(repo).execute(pattern=r"def \w+", path="src/calc.py")
    assert single.success and "src/calc.py:1" in single.output
    assert not SearchTextTool(repo).execute(pattern="x", path="../outside").success
    # Empty path means repo root, not an error (six identical 'empty path'
    # failures burned a real run's budget).
    empty = SearchTextTool(repo).execute(pattern="def \\w+", path="")
    assert empty.success and "src/calc.py:1" in empty.output


def test_apply_edit_exact(repo) -> None:
    result = ApplyEditTool(repo).execute(
        path="src/calc.py", search="    return a + b", replace="    return a + b  # summed"
    )
    assert result.success and "-    return a + b" in result.output
    assert "summed" in (repo / "src" / "calc.py").read_text()
    assert (repo / ".harness" / "backups" / "calc.py.bak").exists()


def test_apply_edit_whitespace_tolerant(repo) -> None:
    result = ApplyEditTool(repo).execute(
        path="src/calc.py",
        search="def add(a, b):\n  return a + b",
        replace="def add(a, b):\n    return float(a + b)",
    )
    assert result.success
    assert "float(a + b)" in (repo / "src" / "calc.py").read_text()


def test_apply_edit_failures(repo) -> None:
    tool = ApplyEditTool(repo)
    assert tool.validate_input({"path": "x", "search": "a"}) != []  # replace missing
    missing = tool.execute(path="src/calc.py", search="NOT PRESENT", replace="x")
    assert not missing.success and "not found" in missing.error
    unchanged = tool.execute(path="src/calc.py", search="def add(a, b):", replace="def add(a, b):")
    assert not unchanged.success and "would not change" in unchanged.error
    assert not tool.execute(path="src/ghost.py", search="a", replace="b").success
    assert not tool.execute(path="../evil.py", search="a", replace="b").success
    assert tool.check_permissions({"model_tier": 1}) is False
    assert tool.check_permissions({"model_tier": 2}) is True


def test_apply_edit_preserves_line_endings(repo) -> None:
    (repo / "crlf.txt").write_bytes(b"a = 1\r\nb = 2\r\n")
    result = ApplyEditTool(repo).execute(path="crlf.txt", search="b = 2", replace="b = 3")
    assert result.success
    assert b"\r\n" in (repo / "crlf.txt").read_bytes()


def test_syntax_check(repo) -> None:
    tool = SyntaxCheckTool(repo)
    assert tool.validate_input({}) != []
    ok = tool.execute(paths=["src/calc.py"])
    assert ok.success and ok.data["checked"] == 1
    (repo / "broken.py").write_text("def f(:\n")
    (repo / "bad.json").write_text("{nope}")
    bad = tool.execute(paths=["broken.py", "bad.json", "src/calc.py", "ghost.py"])
    assert not bad.success
    assert any("broken.py" in e for e in bad.data["errors"])
    assert any("bad.json" in e for e in bad.data["errors"])
    assert tool.check_permissions({"model_tier": 1}) is False


def test_apply_edit_survives_symlinked_repo_root(tmp_path: Path) -> None:
    """macOS /tmp is a symlink (/tmp -> /private/tmp): resolved targets vs
    unresolved roots must not break relative_to (live gateway-run finding)."""
    real = tmp_path / "real-repo"
    real.mkdir()
    (real / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    link = tmp_path / "link-repo"
    link.symlink_to(real)

    tool = ApplyEditTool(link)  # root passed through the symlink path
    result = tool.execute(path="calc.py", search="return a - b", replace="return a + b")
    assert result.success, result.error
    assert "return a + b" in (real / "calc.py").read_text()
