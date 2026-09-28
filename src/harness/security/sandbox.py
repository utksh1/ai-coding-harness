"""Process-level sandbox environment for child tools (review finding #12).

The harness runs commands and test suites from ARBITRARY repositories: those
child processes must never see the harness's own credentials (AI_API_KEY,
GEMINI_API_KEY, gateway tokens, ...) and, when network commands are disabled,
must not reach the network by accident.

This is defense-in-depth at the process level - an ALLOWLIST environment
plus poisoned proxies. It is NOT a container: a determined payload can still
spawn its own processes with crafted env; the eval host's isolation (the
documented boundary) is what actually contains that class. The allowlist
exists so the COMMON case (curious test printing os.environ) leaks nothing.
"""

from __future__ import annotations

import os
import tempfile

ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "TERM",
        "TMPDIR",
        "SHELL",
        "USER",
        "LOGNAME",
        "PYTHONDONTWRITEBYTECODE",
        "VIRTUAL_ENV",
    }
)
"""Variables passed through to child processes verbatim.

Deliberately minimal: anything credential-shaped (*KEY*, *TOKEN*, *SECRET*,
*PASSWORD*, *CREDENTIAL*, *AUTH*, gateway/redis/orchestrator wiring) is absent
by construction - a denylist would always be one env var behind."""

ENV_ALLOWLIST_PREFIXES = ("LC_",)
"""Locale families pass through as a group (LC_ALL, LC_CTYPE, ...)."""

DEAD_PROXY = "http://127.0.0.1:9"
"""A closed local port: proxy-aware clients fail fast instead of reaching out."""

PROXY_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)

NETWORK_EXECUTABLES = frozenset(
    {
        "curl",
        "wget",
        "nc",
        "ncat",
        "netcat",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "telnet",
        "ftp",
        "tftp",
        "socat",
    }
)
"""Fetch/remote executables: blocked ANYWHERE in argv (not just argv[0] -
'git ssh://' style indirections hide them in arguments)."""

NETWORK_SUBCOMMANDS = {
    "git": frozenset({"clone", "fetch", "push", "pull", "remote", "submodule", "ls-remote"}),
    "pip": frozenset({"install", "download"}),
    "npm": frozenset({"install", "add", "publish", "update"}),
    "pip3": frozenset({"install", "download"}),
}
"""Allowlisted executables whose SUBCOMMANDS reach the network: blocked when
network commands are disabled ('git status' stays fine, 'git clone' does not)."""


def sandbox_env(allow_network: bool = False, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build the child-process environment: allowlist + poisoned proxies.

    `extra` merges LAST (caller overrides - e.g. PYTHONDONTWRITEBYTECODE) but
    must never be used to smuggle credentials past the allowlist by callers
    inside this package: it exists for path/locale fixes, not secrets."""
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in ENV_ALLOWLIST or name.startswith(ENV_ALLOWLIST_PREFIXES):
            env[name] = value
    env.setdefault("PATH", "/usr/bin:/bin")
    env.setdefault("HOME", tempfile.gettempdir())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if not allow_network:
        for var in PROXY_VARS:
            env[var] = DEAD_PROXY
        env["no_proxy"] = ""
        env["NO_PROXY"] = ""
    if extra:
        env.update(extra)
    return env


def network_egress_violation(argv: list[str], allow_network: bool = False) -> str | None:
    """Name the first network-egress violation in argv, or None.

    Checks fetch executables anywhere in the list and network subcommands of
    allowlisted runners ('git clone ...', 'pip install requests')."""
    if allow_network:
        return None
    for arg in argv:
        if arg in NETWORK_EXECUTABLES:
            return f"network executable blocked: {arg}"
    if argv:
        head = os.path.basename(argv[0])
        subcommands = NETWORK_SUBCOMMANDS.get(head)
        if subcommands:
            for arg in argv[1:]:
                if os.path.basename(arg) in subcommands:
                    return f"{head} {arg} requires network access (disabled)"
    return None
