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
