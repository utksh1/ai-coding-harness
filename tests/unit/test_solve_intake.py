"""Issue intake + solve command + demo mode (eval-day hardening)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harness.cli import _read_issue, main, solve_command


@pytest.fixture
def demo_repo(tmp_path: Path) -> Path:
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "target" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "target" / "app.py").write_text("def greet():\n    return 'hello'\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path / "target")], check=True)
    subprocess.run(["git", "-C", str(tmp_path / "target"), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path / "target"),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    (tmp_path / "harness.yaml").write_text(
        "models:\n  default:\n    provider: fake\n    name: fake-model\n"
        "agents:\n"
        "  - agent_id: arch-1\n    role: architect\n    model: default\n"
        "  - agent_id: mgr-1\n    role: manager\n    model: default\n"
        "  - agent_id: ver-1\n    role: verifier\n    model: default\n"
        "storage:\n  backend: memory\n"
    )
    return tmp_path


class AlwaysTTY:
    def isatty(self) -> bool:
        return True


# -- intake priority ------------------------------------------------------------
def test_intake_inline_beats_everything(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_ISSUE", "env issue")
    issue, error = _read_issue(
        __import__("argparse").Namespace(issue="inline issue", issue_file=None)
    )
    assert (issue, error) == ("inline issue", None)


def test_intake_file_over_env(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "issue.md").write_text("file issue\n")
    monkeypatch.setenv("HARNESS_ISSUE", "env issue")
    monkeypatch.setenv("HARNESS_ISSUE_FILE", str(tmp_path / "issue.md"))
    issue, _ = _read_issue(__import__("argparse").Namespace(issue=None, issue_file=None))
    assert issue == "file issue\n"


def test_intake_env_issue(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_ISSUE", "from env")
    issue, error = _read_issue(__import__("argparse").Namespace(issue=None, issue_file=None))
    assert (issue, error) == ("from env", None)


def test_intake_missing_file_is_error(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_ISSUE_FILE", "/nowhere/issue.md")
    issue, error = _read_issue(__import__("argparse").Namespace(issue=None, issue_file=None))
    assert issue is None and "not found" in error


def test_intake_tty_no_issue(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", AlwaysTTY())
    issue, error = _read_issue(__import__("argparse").Namespace(issue=None, issue_file=None))
    assert issue is None and error is None


# -- solve command ---------------------------------------------------------------
def test_solve_demo_end_to_end(demo_repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(demo_repo)
    monkeypatch.setenv("HARNESS_DEMO", "1")
    monkeypatch.setenv("HARNESS_TARGET_REPO", str(demo_repo / "target"))
    monkeypatch.setattr("sys.stdin", AlwaysTTY())  # no accidental stdin reads

    exit_code = solve_command(
        __import__("argparse").Namespace(issue="greet should say hello", issue_file=None, repo=None)
    )
    out = capsys.readouterr().out
    assert exit_code == 0, out
    assert "DEMO MODE" in out and "VERIFIED" in out
    assert "evidence:" in out
    # evidence pack landed under the TARGET repo's results/
    runs = list((demo_repo / "target" / "results").iterdir())
    assert runs and (runs[0] / "summary.md").exists()
    summary = (runs[0] / "summary.md").read_text()
    assert "DEMO MODE" in summary


def test_solve_inline_issue_flag(demo_repo: Path, monkeypatch) -> None:
    monkeypatch.chdir(demo_repo)
    monkeypatch.setenv("HARNESS_DEMO", "1")
    monkeypatch.setattr("sys.stdin", AlwaysTTY())
    exit_code = solve_command(
        __import__("argparse").Namespace(
            issue="demo issue", issue_file=None, repo=str(demo_repo / "target")
        )
    )
    assert exit_code == 0


def test_solve_requires_issue(demo_repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(demo_repo)
    monkeypatch.setattr("sys.stdin", AlwaysTTY())
    assert (
        solve_command(__import__("argparse").Namespace(issue=None, issue_file=None, repo=None)) == 2
    )
    assert "no issue supplied" in capsys.readouterr().out


def test_solve_missing_key_fails_fast(demo_repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(demo_repo)
    monkeypatch.delenv("HARNESS_DEMO", raising=False)
    for alt in ("AI_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(alt, raising=False)
    (demo_repo / "harness.yaml").write_text(
        "models:\n  default:\n    provider: openai-compatible\n    name: m\n"
    )
    monkeypatch.setattr("sys.stdin", AlwaysTTY())
    exit_code = solve_command(
        __import__("argparse").Namespace(issue="x", issue_file=None, repo=str(demo_repo / "target"))
    )
    assert exit_code == 3
    assert "AI_API_KEY is not set" in capsys.readouterr().out


def test_solve_bad_config(demo_repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(demo_repo)
    (demo_repo / "harness.yaml").write_text(
        "agents:\n  - agent_id: a\n    role: verifier\n    model: nope\n"
    )
    monkeypatch.setattr("sys.stdin", AlwaysTTY())
    assert (
        solve_command(__import__("argparse").Namespace(issue="x", issue_file=None, repo=None)) == 1
    )


def test_solve_via_make_run_piped_issue(demo_repo: Path, monkeypatch, capsys) -> None:
    """The eval protocol: `make run` with the issue piped in on stdin."""
    monkeypatch.chdir(demo_repo)
    monkeypatch.setenv("HARNESS_DEMO", "1")
    monkeypatch.setenv("HARNESS_TARGET_REPO", str(demo_repo / "target"))

    class PipedStdin:
        def isatty(self) -> bool:
            return False

        def read(self) -> str:
            return "piped issue: greet should say hello\n"

    monkeypatch.setattr("sys.stdin", PipedStdin())
    assert main(["run"]) == 0
    out = capsys.readouterr().out
    assert "VERIFIED" in out and "DEMO MODE" in out


def test_main_solve_dispatch(demo_repo: Path, monkeypatch) -> None:
    monkeypatch.chdir(demo_repo)
    monkeypatch.setenv("HARNESS_DEMO", "1")
    monkeypatch.setattr("sys.stdin", AlwaysTTY())
    assert main(["solve", "--issue", "demo issue", "--repo", str(demo_repo / "target")]) == 0


def test_demo_provider_is_labeled(fake_model_config) -> None:
    from harness.infrastructure.model_providers.fake import build_demo_provider

    provider = build_demo_provider(fake_model_config)
    assert len(provider._responses) == 5  # profile, plan, write tool call, specialist, verdict


async def test_demo_provider_tails_the_verdict_on_extra_calls(fake_model_config) -> None:
    """A dirty-tree demo run exceeds the scripted call count (the final
    review makes one call per diff chunk). Extra calls must keep receiving
    the VERDICT, never wrap to the profile (live finding: ReviewVerdict
    crashed on {'languages': ...})."""
    import json

    from harness.infrastructure.model_providers.fake import build_demo_provider

    provider = build_demo_provider(fake_model_config)
    # consume the whole script: profile, plan, specialist, verdict
    for _ in range(4):
        await provider.generate([{"role": "user", "content": "go"}])
    # extra chunks: verdict, verdict, verdict...
    for _ in range(3):
        extra = await provider.generate([{"role": "user", "content": "chunk"}])
        parsed = json.loads(extra.content)
        assert parsed["approved"] is True
        assert "languages" not in parsed


def test_solve_bad_issue_file_is_error(demo_repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(demo_repo)
    monkeypatch.setattr("sys.stdin", AlwaysTTY())
    exit_code = solve_command(
        __import__("argparse").Namespace(issue=None, issue_file="/nowhere/issue.md", repo=None)
    )
    assert exit_code == 2
    assert "issue file not found" in capsys.readouterr().out


def test_run_command_intake_error(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_ISSUE_FILE", "/nowhere/issue.md")
    assert main(["run"]) == 2
    assert "issue file not found" in capsys.readouterr().out
