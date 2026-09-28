"""Configuration system (foundation issue 1.3).

A typed YAML schema validated by pydantic. API credentials are never stored in
configuration files: models reference the *name* of an environment variable
(`api_key_env`, default `AI_API_KEY`) and the value is read at runtime. String
values may embed `${VAR}` references that are interpolated from the environment
at load time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.infrastructure.logging import configure_logging, get_logger

logger = get_logger(__name__)

ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_CONFIG_PATHS = ("harness.yaml", "harness.example.yaml")


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or validated; message is user-facing."""


def _interpolate(value: Any) -> Any:
    """Recursively replace ${VAR} references with environment values.

    A reference to an unset variable is an error, not a silent empty string:
    silently missing credentials are the worst failure mode at evaluation time.
    """
    if isinstance(value, dict):
        return {key: _interpolate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate(item) for item in value]
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            var = match.group(1)
            if var not in os.environ:
                msg = f"environment variable '{var}' referenced in config is not set"
                raise ConfigError(msg)
            return os.environ[var]

        return ENV_REF.sub(_sub, value)
    return value


class ModelConfig(BaseModel):
    """One named model. The eval contract supplies credentials via `api_key_env`."""

    model_config = ConfigDict(frozen=True)

    provider: Literal["openai", "anthropic", "google", "openai-compatible", "fake"] = (
        "openai-compatible"
    )
    name: str = "gpt-4o-mini"
    api_key_env: str = Field(
        default="AI_API_KEY",
        description="Environment variable HOLDING the key; the key itself is never in config.",
    )
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    tool_call_mode: Literal["auto", "native", "text"] = Field(
        default="auto",
        description=(
            "auto: probe the model once (native tool calls vs text protocol); "
            "native/text force the convention (improvements §3.1)."
        ),
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific kwargs passed through to the API client.",
    )


class BudgetConfig(BaseModel):
    """Token budget governor thresholds (fractions of `total_tokens`)."""

    model_config = ConfigDict(frozen=True)

    total_tokens: int = Field(default=2_000_000, ge=1)
    warn_fraction: float = Field(default=0.7, gt=0, lt=1)
    surgical_fraction: float = Field(default=0.9, gt=0, le=1)

    def thresholds(self) -> tuple[int, int, int]:
        """Return (warn, surgical, stop) token counts."""
        return (
            int(self.total_tokens * self.warn_fraction),
            int(self.total_tokens * self.surgical_fraction),
            self.total_tokens,
        )


class AgentConfig(BaseModel):
    """One agent in the hierarchy; `role` selects its behavior in milestone 2."""

    model_config = ConfigDict(frozen=True)

    agent_id: str = Field(min_length=1)
    role: Literal[
        "architect",
        "manager",
        "locator",
        "implementer",
        "verifier",
        "backend-api",
        "database",
        "frontend",
        "testing",
        "devops",
        "security",
        "documentation",
        "code-review",
    ] = "implementer"
    model: str = Field(description="Key into the top-level `models` mapping.")
    knowledge: bool = Field(
        default=True, description="Inject the persona skill card into the system prompt (#78)."
    )
    knowledge_max_chars: int = Field(
        default=1500, ge=200, le=8000, description="Skill-card cap in characters."
    )
    model_tier: int = Field(
        default=3, ge=1, le=4, description="Capability tier used by tool permission gating."
    )
    stale_tool_results: int = Field(
        default=6,
        ge=0,
        description=(
            "Tool results outside the newest N are assembled as one-line stubs "
            "(lossless in the store; M5 issue #61)."
        ),
    )
    enabled: bool = True


class ToolsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: list[str] = Field(default_factory=list)
    command_timeout_seconds: float = Field(default=60.0, gt=0)
    allow_network_commands: bool = Field(default=False)


class StorageConfig(BaseModel):
    """Context-store backend. `sqlite` needs nothing at eval time.

    `postgres` remains declared for the post-eval deployment vision but is
    rejected at validation until implemented (audit §15 - an advertised
    option that raises NotImplementedError mid-run is a trap).
    """

    model_config = ConfigDict(frozen=True)

    backend: Literal["memory", "sqlite", "postgres"] = "sqlite"
    sqlite_path: str = ".harness/context.db"
    postgres_dsn_env: str = Field(
        default="HARNESS_POSTGRES_DSN",
        description="Env var holding the PostgreSQL DSN (never the DSN itself).",
    )


class LoggingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_format: bool = True
    file_path: str | None = ".harness/harness.log"
    max_bytes: int = Field(default=10_000_000, ge=1)
    backup_count: int = Field(default=5, ge=0)


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    wall_clock_seconds: float = Field(default=1800.0, gt=0)
    max_duration_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Absolute runaway cap from run start. Defaults to 4x "
            "wall_clock_seconds. wall_clock_seconds itself is the STALL "
            "window: max time without recorded model progress (a throttled "
            "provider waiting in retry backoff is the hang this kills)."
        ),
    )
    max_steps: int = Field(default=200, ge=1)
    results_dir: str = "results"


class HarnessConfig(BaseModel):
    """Root configuration schema (`harness.yaml`)."""

    model_config = ConfigDict(frozen=True)

    version: int = Field(default=1, ge=1)
    models: dict[str, ModelConfig] = Field(
        default_factory=lambda: {"default": ModelConfig()},
        description="Named model configs; agents reference them by key.",
    )
    agents: list[AgentConfig] = Field(default_factory=list)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    def validate_references(self) -> list[str]:
        """Cross-field checks beyond pydantic's per-field validation."""
        errors: list[str] = []
        for agent in self.agents:
            if agent.model not in self.models:
                errors.append(f"agent '{agent.agent_id}' references unknown model '{agent.model}'")
        if self.storage.backend == "postgres":
            # Honest config surface (audit §15): the postgres backend is an
            # unimplemented scaffold; selecting it must fail fast, not at
            # the first store call mid-run.
            errors.append(
                "storage.backend 'postgres' is not supported in eval mode; "
                "use 'sqlite' (default) or 'memory'"
            )
        return errors


def _load_dotenv(path: Path | None = None) -> None:
    """Load environment variables from .env if present and set AI_API_KEY fallback."""
    candidates = [Path.cwd() / ".env"]
    if path and path.parent:
        candidates.insert(0, path.parent / ".env")
    for cand in candidates:
        if cand.is_file():
            try:
                for line in cand.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass
            break

    if "AI_API_KEY" not in os.environ or not os.environ["AI_API_KEY"]:
        for alt in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
            if os.environ.get(alt):
                os.environ["AI_API_KEY"] = os.environ[alt]
                break


class ConfigLoader:
    """Load, interpolate, and validate `harness.yaml` into a `HarnessConfig`."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None

    def resolve_path(self) -> Path | None:
        """Explicit path, else the first default that exists."""
        if self._path is not None:
            return self._path if self._path.exists() else None
        for candidate in DEFAULT_CONFIG_PATHS:
            if (path := Path(candidate)).exists():
                return path
        return None

    def load(self) -> HarnessConfig:
        """Load configuration; missing file yields defaults, invalid file is fatal."""
        path = self.resolve_path()
        _load_dotenv(path)
        if path is None:
            logger.info("no configuration file found; using built-in defaults")
            return HarnessConfig()

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            msg = f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
            raise ConfigError(msg)

        try:
            config = HarnessConfig.model_validate(_interpolate(raw))
        except ValidationError as exc:
            details = "\n".join(
                f"  - {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            msg = f"{path}: invalid configuration\n{details}"
            raise ConfigError(msg) from exc

        if refs := config.validate_references():
            msg = f"{path}: invalid configuration\n" + "\n".join(f"  - {e}" for e in refs)
            raise ConfigError(msg)

        log_cfg = config.logging
        configure_logging(
            level=log_cfg.level,
            json_format=log_cfg.json_format,
            file_path=log_cfg.file_path,
            max_bytes=log_cfg.max_bytes,
            backup_count=log_cfg.backup_count,
        )
        logger.info("configuration loaded", path=str(path))
        return config


def load_config(path: str | Path | None = None) -> HarnessConfig:
    """Convenience wrapper around `ConfigLoader`."""
    return ConfigLoader(path).load()
