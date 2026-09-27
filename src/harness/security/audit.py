"""Tamper-evident audit log (milestone 3, issue 3.10).

Append-only JSONL where each entry carries `entry_hash` and `prev_hash`
(SHA-256 chain): any mutation of a past line breaks `verify()`. The chain is
the eval-mode stand-in for the spec's signature service - cryptographic
tamper evidence with zero infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(entry: dict[str, Any], prev_hash: str) -> str:
    payload = dict(entry, prev_hash=prev_hash)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class AuditLog:
    """Append-only hash-chained event log."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev_hash = self._load_last_hash()
        self.count = 0

    def _load_last_hash(self) -> str:
        if not self._path.exists():
            return GENESIS
        last = GENESIS
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        last = json.loads(line)["entry_hash"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # a corrupted tail must not stop the chain
        return last

    def append(
        self, actor: str, action: str, target: str = "", detail: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Record one event; returns the stored entry (with its hash)."""
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "actor": actor,
            "action": action,
            "target": target,
            "detail": detail or {},
        }
        with self._lock:
            # The log directory may have been removed after construction
            # (long-lived orchestrator, target repo cleaned between runs):
            # recreate it and start a fresh chain - an append must never
            # crash the run (Errno 2 on open("a") without the parent dir).
            if not self._path.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._prev_hash = GENESIS
            entry["prev_hash"] = self._prev_hash
            entry["entry_hash"] = _digest(
                {k: v for k, v in entry.items() if k not in {"prev_hash", "entry_hash"}},
                self._prev_hash,
            )
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
            self._prev_hash = entry["entry_hash"]
            self.count += 1
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        parsed: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a corrupted tail must not break reads
        return parsed

    def verify(self) -> tuple[bool, str | None]:
        """Re-walk the chain; returns (ok, first_bad_reason)."""
        expected_prev = GENESIS
        for index, entry in enumerate(self.entries()):
            claimed_hash = entry.get("entry_hash")
            claimed_prev = entry.get("prev_hash")
            if claimed_prev != expected_prev:
                return False, f"entry {index}: broken prev_hash link"
            recomputed = _digest(
                {k: v for k, v in entry.items() if k not in {"prev_hash", "entry_hash"}},
                claimed_prev,
            )
            if recomputed != claimed_hash:
                return False, f"entry {index}: hash mismatch (tampered)"
            expected_prev = claimed_hash
        return True, None

    def export(self, destination: Path) -> Path:
        """Copy the raw log for compliance export."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            destination.write_text(self._path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            destination.write_text("", encoding="utf-8")
        return destination
