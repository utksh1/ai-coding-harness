"""Command-line entry point.

- `harness doctor [--probe-model]` - validate the runtime environment
  (`make setup` target; the model probe is opt-in because it costs tokens)
- `harness run` - health summary + evidence pointer (headless by design)
- `harness replay [RUN_ID]` - replay a recorded evidence trace offline
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from harness import __version__
from harness.config import ConfigError, ConfigLoader

API_KEY_ENV = "AI_API_KEY"


def doctor(probe_model: bool = False) -> int:
    """Check the runtime environment. Warn-only about a missing API key so
    `make setup` never fails on machines that only run offline tests."""
    from harness.monitoring.health import run_health_checks

    print(f"harness {__version__}")
    loader = ConfigLoader()
    path = loader.resolve_path()
    if path is None:
        print("[warn] no configuration file found; using built-in defaults")
        if not os.environ.get(API_KEY_ENV):
            print(
                f"[warn] environment variable {API_KEY_ENV} is not set; "
                "the harness will run in offline/test mode only"
            )
        else:
            print(
                f"[ok] environment variable {API_KEY_ENV} is set; "
                "copy config.example.yaml to harness.yaml to use it"
            )
        print("[ok] environment ready")
        return 0

    try:
        config = loader.load()
    except ConfigError as exc:
        print(f"[error] configuration invalid:\n{exc}")
        return 1

    print(f"[ok] configuration valid: {path}")
    default_model = config.models.get("default")
    key_env = default_model.api_key_env if default_model else API_KEY_ENV
    has_key = bool(
        os.environ.get(key_env)
        or os.environ.get("AI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("CODEX_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not has_key:
        print(
            f"[warn] environment variable {key_env} is not set; "
            "the harness will run in offline/test mode only"
        )
    report = run_health_checks(Path.cwd(), config_path=path, model_probe=probe_model)
    for check in report.checks:
        if check["name"] == "model-api":
            mark = "ok" if check["ok"] else "warn"
            print(f"[{mark}] model probe: {check['detail']}")
    print("[ok] environment ready")
    return 0


def _read_issue(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Issue intake, priority per design §7: --issue > --issue-file >
    HARNESS_ISSUE > HARNESS_ISSUE_FILE > piped stdin."""
    if getattr(args, "issue", None):
        return args.issue, None
    issue_file = getattr(args, "issue_file", None) or os.environ.get("HARNESS_ISSUE_FILE")
    if issue_file:
        path = Path(issue_file)
        if not path.is_file():
            return None, f"issue file not found: {issue_file}"
        return path.read_text(encoding="utf-8"), None
    if os.environ.get("HARNESS_ISSUE"):
        return os.environ["HARNESS_ISSUE"], None
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped, None
    return None, None


def solve_command(args: argparse.Namespace) -> int:
    """Run the full pipeline on one issue; print the outcome + evidence path."""
    import asyncio

    from harness.engine.pipeline import HarnessPipeline
    from harness.infrastructure.context_store import create_context_store
    from harness.infrastructure.model_providers import create_model_provider

    issue, error = _read_issue(args)
    if error:
        print(f"[error] {error}")
        return 2
    if not issue:
        print(
            "[error] no issue supplied: use --issue, --issue-file, "
            "HARNESS_ISSUE(_FILE), or pipe the issue text on stdin"
        )
        return 2

    repo_root = Path(args.repo or os.environ.get("HARNESS_TARGET_REPO") or ".").resolve()
    try:
        config = ConfigLoader().load()
    except ConfigError as exc:
        print(f"[error] configuration invalid:\n{exc}")
        return 1

    provider: Any
    demo_mode = os.environ.get("HARNESS_DEMO") == "1"
    if demo_mode:
        from harness.infrastructure.model_providers.fake import build_demo_provider

        provider = build_demo_provider(config.models["default"])
        print("[warn] DEMO MODE: scripted model responses; evidence is illustrative only")
    else:
        provider = create_model_provider(config.models["default"])
        key_env = config.models["default"].api_key_env
        has_key = bool(
            os.environ.get(key_env)
            or os.environ.get("AI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("CODEX_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or key_env.startswith("sk-")
            or len(key_env) > 25
        )
        if config.models["default"].provider != "fake" and not has_key:
            print(
                f"[error] environment variable {key_env} is not set; "
                "cannot authenticate the configured model"
            )
            return 3

    store = create_context_store(config.storage)
    pipeline = HarnessPipeline(repo_root=repo_root, config=config, provider=provider, store=store)
    from harness.infrastructure.model_providers.base import ModelAuthError

    try:
        outcome = asyncio.run(pipeline.run(issue, demo_mode=demo_mode))
    except ModelAuthError as exc:
        # Credentials rejected mid-run (401/403/402): the documented
        # exit-code contract maps this to 3, not a traceback.
        print(f"[error] model credentials rejected: {exc}")
        store.close()
        return 3
    print(f"outcome: {outcome.outcome_line}")
    print(f"run: {outcome.run_id}  evidence: {outcome.evidence_path}")
    for flag in outcome.flags:
        print(f"flag: {flag}")
    store.close()
    return 0 if outcome.success else 1


def run_command() -> int:
    """`make run`: solve a supplied issue (headless), else the cockpit/summary."""
    issue, error = _read_issue(argparse.Namespace(issue=None, issue_file=None))
    if error:
        print(f"[error] {error}")
        return 2
    if issue:
        return solve_command(
            argparse.Namespace(
                issue=issue,
                issue_file=None,
                repo=os.environ.get("HARNESS_TARGET_REPO") or ".",
            )
        )

    from harness.monitoring.health import run_health_checks

    report = run_health_checks(Path.cwd())
    print(report.summary())
    return 0 if report.ready else 1


def _adhoc_pack() -> Any:  # EvidencePack; late import keeps CLI startup light
    from harness.engine.evidence import EvidencePack

    return EvidencePack(Path.cwd() / "results", "adhoc")


def replay_command(run_id: str | None) -> int:
    """Replay a recorded evidence trace (offline demo / post-run audit)."""
    from harness.engine.evidence import find_evidence, format_event

    pack = find_evidence(Path.cwd() / "results", run_id)
    if pack is None:
        print("[error] no evidence pack found under results/; run the pipeline first")
        return 1
    events = pack.read_trace()
    for event in events:
        print(format_event(event))
    print(f"[ok] replayed {pack.run_id} ({len(events)} events) - headless mode")
    return 0


def bench_command(args: argparse.Namespace) -> int:
    """`harness bench`: offline A/B token benchmark (milestone 5, issue 5.5)."""
    from harness.bench.tokens import main as bench_main

    argv = []
    if args.issue:
        argv += ["--issue", args.issue]
    if args.ref:
        argv += ["--ref", args.ref]
    if args.save_baseline:
        argv.append("--save-baseline")
    if args.baseline_path:
        argv += ["--baseline-path", args.baseline_path]
    if args.json_out:
        argv += ["--json-out", args.json_out]
    return bench_main(argv)


def gui_command(args: argparse.Namespace | None = None) -> int:
    """`harness gui`: launch and open the web dashboard in your browser."""
    import subprocess
    import time
    import urllib.request
    import webbrowser

    gateway_url = "http://localhost:8080"
    is_running = False
    try:
        with urllib.request.urlopen(gateway_url + "/api/health", timeout=0.5) as resp:
            if resp.status == 200:
                is_running = True
    except Exception:
        pass

    if not is_running:
        gateway_bin = os.environ.get("HARNESS_GATEWAY_BIN") or (
            Path(__file__).resolve().parent.parent.parent / "bin" / "foreman-gateway"
        )
        if Path(gateway_bin).is_file():
            print("[info] starting Foreman Gateway daemon on port 8080...")
            subprocess.Popen(
                [str(gateway_bin)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(0.5)

    print(f"[ok] opening Foreman Web Dashboard at {gateway_url}...")
    webbrowser.open(gateway_url)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    parser.add_argument("--version", action="version", version=f"harness {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="validate the runtime environment")
    doctor_parser.add_argument(
        "--probe-model",
        action="store_true",
        help="make one tiny live model call (costs tokens; offline CI omits this)",
    )
    subparsers.add_parser(
        "run", help="launch the harness (TUI on a TTY, headless summary otherwise)"
    )
    subparsers.add_parser("gui", help="launch and open the web dashboard in your browser")
    subparsers.add_parser("dashboard", help="alias for 'harness gui'")
    replay_parser = subparsers.add_parser(
        "replay", help="replay a recorded evidence trace (default: most recent)"
    )
    replay_parser.add_argument("run_id", nargs="?", default=None)
    solve_parser = subparsers.add_parser(
        "solve", help="run the full pipeline on one issue and write the evidence pack"
    )
    solve_parser.add_argument("--issue", help="issue text inline")
    solve_parser.add_argument("--issue-file", help="path to a file holding the issue text")
    solve_parser.add_argument("--repo", help="target repository root (default: cwd)")
    bench_parser = subparsers.add_parser(
        "bench", help="offline A/B token benchmark on the fixture repo"
    )
    bench_parser.add_argument("--issue", help="override the fixture issue text")
    bench_parser.add_argument("--ref", help="git ref measured as the baseline side")
    bench_parser.add_argument(
        "--save-baseline", action="store_true", help="record current spend as the baseline"
    )
    bench_parser.add_argument("--baseline-path", help="baseline JSON path override")
    bench_parser.add_argument("--json-out", help="write full delta JSON to this path")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(probe_model=args.probe_model)
    if args.command in ("gui", "dashboard"):
        return gui_command(args)
    if args.command == "replay":
        return replay_command(args.run_id)
    if args.command == "solve":
        return solve_command(args)
    if args.command == "bench":
        return bench_command(args)
    return run_command()


if __name__ == "__main__":
    sys.exit(main())
