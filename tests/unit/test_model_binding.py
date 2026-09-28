"""Per-agent model binding tests (review finding #6).

`agents[].model` names a `models:` profile and each agent gets the provider
built FROM that profile - the runtime must actually be multi-model, not one
provider with aspirational configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.config import HarnessConfig
from harness.engine.pipeline import HarnessPipeline
from harness.infrastructure.context_store import MemoryContextStore


def _config(agent_models: dict[str, str]) -> HarnessConfig:
    return HarnessConfig.model_validate(
        {
            "models": {
                "default": {"provider": "fake", "name": "fake-main"},
                "cheap": {"provider": "fake", "name": "fake-cheap"},
            },
            "agents": [
                {"agent_id": "arch-1", "role": "architect", "model": agent_models.get("arch-1", "default")},
                {"agent_id": "mgr-1", "role": "manager", "model": agent_models.get("mgr-1", "default")},
                {"agent_id": "impl-1", "role": "implementer", "model": agent_models.get("impl-1", "default")},
                {"agent_id": "ver-1", "role": "verifier", "model": agent_models.get("ver-1", "default")},
            ],
            "storage": {"backend": "memory"},
        }
    )


class _Recorder:
    """Provider stand-in with identity: one per factory call."""

    def __init__(self, profile: str) -> None:
        self.model = f"model-for-{profile}"
        self.profile = profile


def _factory(calls: list[str]) -> Any:
    def build(model_cfg: Any) -> _Recorder:
        calls.append(model_cfg.name)
        return _Recorder(model_cfg.name)

    return build


def test_agents_get_providers_from_their_own_profiles(tmp_path: Path) -> None:
    """Architect on `cheap`, specialists on `default`: two distinct
    providers; agents sharing a profile share ONE instance."""
    calls: list[str] = []
    pipeline = HarnessPipeline(
        tmp_path,
        _config({"arch-1": "cheap"}),
        store=MemoryContextStore(),
        provider_factory=_factory(calls),
    )
    architect_provider = pipeline._architect.provider
    impl_provider = pipeline._agents["impl-1"].provider
    ver_provider = pipeline._agents["ver-1"].provider
    assert architect_provider is not impl_provider
    assert architect_provider.model == "model-for-fake-cheap"
    assert impl_provider.model == "model-for-fake-main"
    assert ver_provider is impl_provider  # same profile -> shared instance
    assert calls.count("fake-cheap") == 1
    assert calls.count("fake-main") == 1


def test_run_level_profile_overrides_default_bound_agents(tmp_path: Path) -> None:
    """model_profile='cheap' moves every 'default'-bound agent to cheap;
    an agent explicitly bound to a real profile keeps its binding."""
    calls: list[str] = []
    pipeline = HarnessPipeline(
        tmp_path,
        _config({"arch-1": "default", "impl-1": "cheap"}),
        store=MemoryContextStore(),
        provider_factory=_factory(calls),
        model_profile="cheap",
    )
    # impl-1 explicitly cheap; arch/mgr/ver follow the run profile (cheap)
    assert pipeline._architect.provider.model == "model-for-fake-cheap"
    assert pipeline._agents["impl-1"].provider is pipeline._architect.provider
    assert "fake-main" not in calls  # default profile never built


def test_unknown_agent_model_falls_back_to_run_profile(tmp_path: Path) -> None:
    calls: list[str] = []
    pipeline = HarnessPipeline(
        tmp_path,
        _config({"ver-1": "nonexistent-profile"}),
        store=MemoryContextStore(),
        provider_factory=_factory(calls),
        model_profile="cheap",
    )
    assert pipeline._agents["ver-1"].provider.model == "model-for-fake-cheap"
    assert pipeline._architect.provider is pipeline._agents["ver-1"].provider


def test_injected_provider_overrides_all_bindings(tmp_path: Path) -> None:
    """Tests/demo: one provider for every agent, regardless of bindings."""
    sentinel = _Recorder("injected")
    pipeline = HarnessPipeline(
        tmp_path,
        _config({"arch-1": "cheap"}),
        provider=sentinel,
        store=MemoryContextStore(),
    )
    assert pipeline._architect.provider is sentinel
    assert pipeline._agents["impl-1"].provider is sentinel


def test_model_config_carries_real_model_identity(tmp_path: Path) -> None:
    pipeline = HarnessPipeline(
        tmp_path,
        _config({"arch-1": "cheap"}),
        store=MemoryContextStore(),
        provider_factory=_factory([]),
    )
    assert pipeline._architect.model_config == {
        "provider": "fake",
        "model": "fake-cheap",
        "profile": "cheap",
    }
    assert pipeline._agents["impl-1"].model_config["model"] == "fake-main"


def test_collaborator_inherits_primary_profile(tmp_path: Path) -> None:
    """A collaborator spawned for a cheap-bound primary runs cheap too."""
    calls: list[str] = []
    pipeline = HarnessPipeline(
        tmp_path,
        _config({"impl-1": "cheap"}),
        store=MemoryContextStore(),
        provider_factory=_factory(calls),
    )
    collaborator = pipeline._add_collaborator(
        "impl-1", pipeline._placeholder_governor, _NullPack(), "run-1"
    )
    assert collaborator is not None
    assert collaborator.provider is pipeline._agents["impl-1"].provider
    assert collaborator.model_config["profile"] == "cheap"


class _NullPack:
    """Evidence-pack stand-in: only trace() is needed for spawn events."""

    def trace(self, event: dict[str, Any]) -> None:
        pass
