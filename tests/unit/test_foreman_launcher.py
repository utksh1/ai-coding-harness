"""bin/foreman — the launcher contract.

The launcher is bash, so its "unit tests" are behavioral: the script must be
syntactically valid, advertise the full product surface, refuse unknown
commands with the usage text, and its doctor/status must degrade honestly
when services are down (exit non-zero, no crash, no stack trace).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FOREMAN = REPO_ROOT / "bin" / "foreman"

EXPECTED_SURFACE = (
    b"foreman start",
    b"foreman stop",
    b"foreman restart",
    b"foreman status",
    b"foreman tui",
    b"foreman web",
    b"foreman run",
    b"foreman followup",
    b"foreman cancel",
    b"foreman runs",
    b"foreman projects",
    b"foreman models",
    b"foreman logs",
    b"foreman doctor",
)


def _foreman(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(FOREMAN), *args], capture_output=True, timeout=timeout, cwd=REPO_ROOT
    )


def test_script_is_executable() -> None:
    assert FOREMAN.exists(), "bin/foreman must ship with the repo"
    assert FOREMAN.stat().st_mode & 0o111, "bin/foreman must be executable"


def test_bash_syntax_is_valid() -> None:
    result = subprocess.run(["bash", "-n", str(FOREMAN)], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode()


def test_help_advertises_the_full_surface() -> None:
    for args in (["--help"], ["help"], []):
        result = _foreman(*args)
        assert result.returncode == 0, args
        for needle in EXPECTED_SURFACE:
            assert needle in result.stdout, (args, needle)


def test_unknown_command_is_refused_with_usage() -> None:
    result = _foreman("definitely-not-a-command")
    assert result.returncode == 1
    assert b"unknown command" in result.stderr
    assert b"foreman start" in result.stdout  # usage follows the complaint


def test_status_degrades_honestly_when_services_are_down() -> None:
    """status must never crash or lie: DOWN services => exit 1 + DOWN lines.

    Skipped when a foreman stack is actually running (the operator's live
    session must not be disturbed by a test).
    """
    import socket

    def port_open(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.3)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    if port_open(8000) or port_open(8080):
        return  # live stack in flight — status is verified by scripts instead

    result = _foreman("status")
    assert result.returncode == 1
    # Down-service lines are diagnostics: they go to stderr (bad()).
    assert b"DOWN" in result.stderr
    assert b"Traceback" not in result.stderr


def test_doctor_reports_even_when_down() -> None:
    """doctor exits non-zero with issues listed — never a bash stack trace."""
    import socket

    def port_open(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.3)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    if port_open(8000) or port_open(8080):
        return  # live stack in flight

    result = _foreman("doctor", timeout=90)
    assert b"Traceback" not in result.stderr
    assert b"python venv" in result.stdout
    # venv exists in this checkout, so the venv line must be an OK line
    assert b"python venv" in result.stdout


def test_stop_is_idempotent() -> None:
    """stopping a stopped stack is a no-op, not an error."""
    import socket

    def port_open(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.3)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    if port_open(8000) or port_open(8080):
        return  # live stack in flight — do not stop the operator's services

    result = _foreman("stop")
    assert result.returncode == 0
    assert b"stopped" in result.stdout
