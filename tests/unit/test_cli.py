"""CLI tests: doctor exit codes (issue 1.8's "example tests" for the shell surface)."""

from __future__ import annotations

from harness.cli import main


def test_doctor_ok_without_key(isolated_env, capsys) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "environment ready" in out
    assert "AI_API_KEY" in out  # warn-only, offline mode announced


def test_doctor_validates_config(isolated_env) -> None:
    (isolated_env / "harness.yaml").write_text(
        "agents:\n  - agent_id: a\n    role: verifier\n    model: nope\n"
    )
    assert main(["doctor"]) == 1


def test_version_flag(capsys) -> None:
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "harness" in capsys.readouterr().out


def test_solve_mid_run_auth_rejection_exits_3(isolated_env, monkeypatch, capsys) -> None:
    """Documented exit-code contract: credentials rejected mid-run (401/403/
    402) is exit 3 with a clean message — not a raw traceback (live-run
    finding: the proxy exhausted its quota and the CLI crashed on httpx)."""
    import harness.engine.pipeline as pipeline_module
    from harness.infrastructure.model_providers.base import ModelAuthError

    (isolated_env / "harness.yaml").write_text(
        "models:\n"
        "  default:\n"
        "    provider: openai-compatible\n"
        "    name: m\n"
        "    api_key_env: AI_API_KEY\n"
    )
    monkeypatch.setenv("AI_API_KEY", "sk-test")

    class _RaisingPipeline:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        async def run(self, *args: object, **kwargs: object) -> object:
            raise ModelAuthError("proxy rejected credentials (HTTP 402)")

    monkeypatch.setattr(pipeline_module, "HarnessPipeline", _RaisingPipeline)
    rc = main(["solve", "--issue", "do the thing", "--repo", str(isolated_env)])
    assert rc == 3
    assert "credentials rejected" in capsys.readouterr().out


def test_solve_builds_providers_via_factory(isolated_env, monkeypatch, capsys) -> None:
    """The CLI's provider factory path (non-demo): create_model_provider is
    invoked per models-profile - per-agent model bindings are honored, not
    collapsed into one prebuilt provider."""
    import json as _json
    import subprocess as _subprocess

    from harness.infrastructure.model_providers import FakeProvider, ModelResponse

    (isolated_env / "harness.yaml").write_text(
        "models:\n  default:\n    provider: fake\n    name: fake-model\n"
        "    api_key_env: AI_API_KEY\n"
    )
    monkeypatch.setenv("AI_API_KEY", "sk-test")
    # A git repo so the no-op diff gate is measurable (the scripted specialist
    # makes no edits: honest NOT VERIFIED, never a crash).
    repo = isolated_env
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (repo / "app.py").write_text("def greet():\n    return 'hello'\n")
    (repo / "test_greet.py").write_text(
        "from app import greet\n\ndef test_ok():\n    assert greet()\n"
    )
    _subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    _subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    _subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "b"],
        check=True,
    )

    calls: list[str] = []

    def _fake_builder(model_cfg):
        calls.append(model_cfg.name)
        return FakeProvider(
            model_cfg,
            responses=[
                ModelResponse(
                    content='{"languages": ["Python"], "test_framework": "pytest", '
                    '"build_system": "pyproject.toml", "frameworks": [], "conventions": [], "notes": ""}'
                ),
                ModelResponse(
                    content=_json.dumps(
                        {
                            "issue_summary": "s",
                            "complexity": 2,
                            "subtasks": [
                                {
                                    "id": "st-1",
                                    "title": "t",
                                    "description": "d",
                                    "specialty": "refactoring",
                                    "complexity": 1,
                                    "files": ["app.py"],
                                    "acceptance_criteria": ["c"],
                                    "depends_on": [],
                                }
                            ],
                            "risks": [],
                            "needs_collaboration": False,
                        }
                    )
                ),
                ModelResponse(content="TASK_COMPLETE: nothing edited (honest fail)"),
                ModelResponse(
                    content=_json.dumps(
                        {
                            "approved": True,
                            "issues": [],
                            "summary": "ok",
                            "criteria_dispositions": [
                                {"criterion": "c", "satisfied": True, "evidence": "x"}
                            ],
                        }
                    )
                ),
            ],
        )

    import harness.infrastructure.model_providers as providers_module

    monkeypatch.setattr(providers_module, "create_model_provider", _fake_builder)
    rc = main(["solve", "--issue", "do the thing", "--repo", str(repo)])
    assert rc in (0, 1)  # completed honestly either way
    assert calls, "provider factory never invoked by the CLI"
    out = capsys.readouterr().out
    assert "outcome:" in out
