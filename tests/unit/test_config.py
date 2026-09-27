"""Unit tests for the configuration system (foundation issue 1.3)."""

from __future__ import annotations

import pytest

from harness.config import ConfigError, ConfigLoader, load_config


def test_defaults_when_no_file(isolated_env) -> None:
    config = load_config()
    assert config.version == 1
    assert config.storage.backend == "sqlite"
    assert config.models["default"].api_key_env == "AI_API_KEY"


def test_budget_thresholds() -> None:
    config = load_config()
    warn, surgical, stop = config.budget.thresholds()
    # The 70/90/100 split is the contract; the total itself is the deployer's
    # dial (harness.yaml) and must not be pinned by this test.
    assert warn == int(config.budget.total_tokens * 0.7)
    assert surgical == int(config.budget.total_tokens * 0.9)
    assert stop == config.budget.total_tokens


def test_example_config_is_valid() -> None:
    config = (
        ConfigLoader("config.example.yaml").load()
        if ConfigLoader("config.example.yaml").resolve_path()
        else load_config()
    )
    assert isinstance(config.agents, list)


def test_env_interpolation(isolated_env, monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODEL_NAME", "test-model")
    (isolated_env / "harness.yaml").write_text(
        "models:\n  default:\n    name: ${TEST_MODEL_NAME}\n"
    )
    assert load_config().models["default"].name == "test-model"


def test_missing_env_reference_is_a_hard_error(isolated_env) -> None:
    (isolated_env / "harness.yaml").write_text(
        "models:\n  default:\n    name: ${DEFINITELY_NOT_SET}\n"
    )
    with pytest.raises(ConfigError, match="DEFINITELY_NOT_SET"):
        load_config()


def test_unknown_model_reference_rejected(isolated_env) -> None:
    (isolated_env / "harness.yaml").write_text(
        "agents:\n  - agent_id: a\n    role: verifier\n    model: nope\n"
    )
    with pytest.raises(ConfigError, match="unknown model"):
        load_config()


def test_invalid_schema_reports_field_paths(isolated_env) -> None:
    (isolated_env / "harness.yaml").write_text("budget:\n  total_tokens: not-a-number\n")
    with pytest.raises(ConfigError, match="total_tokens"):
        load_config()


def test_non_mapping_yaml_rejected(isolated_env) -> None:
    (isolated_env / "harness.yaml").write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config()


def test_explicit_path_wins(isolated_env) -> None:
    custom = isolated_env / "custom.yaml"
    custom.write_text("run:\n  max_steps: 7\n")
    assert load_config(custom).run.max_steps == 7
