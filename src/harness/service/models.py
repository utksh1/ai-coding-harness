"""Runtime model registry (settings-based model management, platform P6).

harness.yaml stays the bootstrap: its `models:` and `agents:` seed the
registry. From the cockpit (or `foreman models`) operators add, edit, and
remove profiles and re-bind any agent to any profile; those settings persist
in `.harness/models.json` and OVERLAY the yaml:

- ``profiles``  adds/overrides a named profile (full ModelConfig dict)
- ``removed``   tombstones yaml-native profiles (gone until un-removed)
- ``agents``    per-agent profile bindings overriding ``agents[].model``

The effective config is what pipelines are built from, so a settings change
takes effect on the NEXT run without an orchestrator restart (the service
drops its pipeline cache on every mutation - see app.py).

Security invariants:
- Profiles carry ``api_key_env`` NAMES, never key values. The API layer
  neither accepts nor returns a key; the orchestrator resolves keys from its
  OWN environment at provider-build time.
- ``default`` cannot be removed and the last effective profile cannot be
  removed: ``_resolve_model`` falls back to ``models["default"]`` and the
  engine's fallback ladder depends on that anchor existing.
- A profile bound by any agent cannot be removed (rebind first): a silent
  rebind would change what an agent runs on without the operator seeing it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from harness.config import AgentConfig, HarnessConfig, ModelConfig
from harness.infrastructure.logging import get_logger

logger = get_logger(__name__)

MODELS_FILE = Path(".harness") / "models.json"
"""Registry location (relative to the orchestrator's working directory).

``HARNESS_MODELS_FILE`` overrides it - the deployer's knob for a shared
volume, AND the test suite's hermetic isolation (same pattern as the
project registry: pytest shares the repo-root CWD with a running
``foreman start``).
"""

PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
"""Profile ids: lowercase slug, 1-40 chars - they land in cache keys and UI."""

MAX_PROFILES = 32
"""Cap on effective profiles: a picker, not a warehouse."""


class ModelStore:
    """Persistent, best-effort model-settings overlay.

    Concurrency model is last-write-wins on a single-process service (same
    contract as the project registry). File damage degrades to an empty
    overlay, i.e. harness.yaml as written - never an error path for a run.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            override = os.environ.get("HARNESS_MODELS_FILE")
            path = Path(override) if override else MODELS_FILE
        self._path = path

    # -- persistence ------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"profiles": {}, "removed": [], "agents": {}}
        if not isinstance(data, dict):
            return {"profiles": {}, "removed": [], "agents": {}}
        return {
            "profiles": data.get("profiles") if isinstance(data.get("profiles"), dict) else {},
            "removed": data.get("removed") if isinstance(data.get("removed"), list) else [],
            "agents": data.get("agents") if isinstance(data.get("agents"), dict) else {},
        }

    def _save(self, state: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    # -- effective views --------------------------------------------------
    def effective_models(self, base: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        """Merged model table: yaml profiles minus tombstones plus overrides."""
        state = self._load()
        merged = {k: v for k, v in base.items() if k not in set(state["removed"])}
        for name, raw in state["profiles"].items():
            try:
                merged[name] = ModelConfig.model_validate(raw)
            except Exception as exc:
                logger.warning("skipping invalid stored profile", profile=name, error=str(exc))
        if not merged:
            # A fully-tombstoned registry would break the fallback ladder;
            # keep the yaml defaults alive instead.
            merged = dict(base)
        return merged

    def effective_agents(self, base: list[AgentConfig]) -> list[AgentConfig]:
        """Agent roster with per-agent model bindings applied."""
        state = self._load()
        bindings = state["agents"]
        if not bindings:
            return list(base)
        return [
            agent.model_copy(update={"model": bindings[agent.agent_id]})
            if agent.agent_id in bindings
            else agent
            for agent in base
        ]

    def apply(self, config: HarnessConfig) -> HarnessConfig:
        """Effective HarnessConfig (identity when nothing is overridden)."""
        state = self._load()
        if not (state["profiles"] or state["removed"] or state["agents"]):
            return config
        return config.model_copy(
            update={
                "models": self.effective_models(config.models),
                "agents": self.effective_agents(config.agents),
            }
        )

    def agent_bindings(self) -> dict[str, str]:
        return dict(self._load()["agents"])

    def overrides_summary(self) -> dict[str, Any]:
        state = self._load()
        return {
            "profiles": sorted(state["profiles"]),
            "removed": sorted(state["removed"]),
            "agents": dict(state["agents"]),
        }

    # -- mutations --------------------------------------------------------
    def upsert_profile(
        self,
        name: str,
        fields: dict[str, Any],
        base: dict[str, ModelConfig],
        agent_bindings: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Add or fully replace a profile from raw UI fields.

        Validation is the real ModelConfig schema, so an invalid temperature
        or unknown provider is rejected here, not at run time.
        """
        if not PROFILE_NAME.match(name or ""):
            return False, (
                f"invalid profile name '{name}': lowercase letters, digits, dashes (1-40 chars)"
            )
        state = self._load()
        merged = self.effective_models(base)
        if name not in merged and len(merged) >= MAX_PROFILES:
            return False, f"profile limit reached ({MAX_PROFILES})"
        known_fields = set(ModelConfig.model_fields)
        clean = {k: v for k, v in fields.items() if k in known_fields}
        try:
            cfg = ModelConfig.model_validate(clean)
        except Exception as exc:
            return False, f"invalid model config: {exc}"
        state["profiles"][name] = cfg.model_dump()
        if name in state["removed"]:
            state["removed"].remove(name)
        self._save(state)
        logger.info("model profile upserted", profile=name, provider=cfg.provider)
        return True, "saved"

    def remove_profile(
        self,
        name: str,
        base: dict[str, ModelConfig],
        agent_bindings: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        state = self._load()
        merged = self.effective_models(base)
        if name not in merged:
            return False, f"unknown profile '{name}'"
        if name == "default":
            return False, "the 'default' profile is the fallback anchor and cannot be removed"
        if len(merged) <= 1:
            return False, "cannot remove the last profile"
        bindings = agent_bindings if agent_bindings is not None else self.agent_bindings()
        bound = [aid for aid, prof in bindings.items() if prof == name]
        if bound:
            return False, f"profile bound by agents: {', '.join(sorted(bound))} (rebind first)"
        if name in state["profiles"]:
            del state["profiles"][name]
        if name in base:  # yaml-native: tombstone so it stays gone
            state["removed"].append(name)
        self._save(state)
        logger.info("model profile removed", profile=name)
        return True, "removed"

    def set_agent_model(
        self,
        agent_id: str,
        profile: str,
        base: dict[str, ModelConfig],
        base_agents: list[AgentConfig],
    ) -> tuple[bool, str]:
        """Bind one agent to a profile ('default' = follow the run picker)."""
        if not any(a.agent_id == agent_id for a in base_agents):
            return False, f"unknown agent '{agent_id}'"
        merged = self.effective_models(base)
        if profile != "default" and profile not in merged:
            return False, f"unknown profile '{profile}'"
        state = self._load()
        state["agents"][agent_id] = profile
        self._save(state)
        logger.info("agent model rebound", agent=agent_id, profile=profile)
        return True, "saved"

    def reset(self) -> tuple[bool, str]:
        """Drop every override: back to harness.yaml as written."""
        self._save({"profiles": {}, "removed": [], "agents": {}})
        logger.info("model settings reset to harness.yaml")
        return True, "reset"
