"""Model provider contracts and shared transport (foundation issue 1.5).

Providers translate a provider-agnostic message/tool format into each API's
wire format and back. All providers:

- read the API key from the env var named by `ModelConfig.api_key_env`
  (never from config files),
- retry transient failures (timeouts, 429/5xx) with exponential backoff and
  `Retry-After` awareness; auth failures (401/403) fail immediately,
- return usage counts so the budget governor can meter every call.
"""

from __future__ import annotations

import asyncio
import os
import random
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

from harness.config import ModelConfig

logger = structlog.get_logger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ModelAuthError(Exception):
    """API key missing or rejected; not retried."""


class ToolCall(BaseModel):
    """One tool invocation requested by the model."""

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Unified response across providers."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    stop_reason: str = ""
    provider: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelProvider(ABC):
    """Abstract provider: one configured model behind a single `generate` call.

    `client` is an optional pre-built httpx.AsyncClient (test seam: inject an
    httpx.MockTransport to exercise retry/rate-limit handling offline).
    """

    def __init__(self, config: ModelConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._api_key: str | None = None
        self._client = client

    @property
    def name(self) -> str:
        """Provider key (e.g. 'openai-compatible')."""
        return self._config.provider

    @property
    def model(self) -> str:
        return self._config.name

    def _resolve_api_key(self) -> str:
        if self._api_key is None:
            env = self._config.api_key_env
            key = os.environ.get(env)
            if not key:
                for alt in ("AI_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
                    if os.environ.get(alt):
                        key = os.environ[alt]
                        break
            if not key:
                msg = (
                    f"environment variable '{env}' is not set; cannot authenticate "
                    f"model '{self._config.name}' ({self._config.provider})"
                )
                raise ModelAuthError(msg)
            self._api_key = key
        return self._api_key

    @abstractmethod
    def _endpoint(self) -> str:
        """Full URL for the generation endpoint."""

    @abstractmethod
    def _headers(self, api_key: str) -> dict[str, str]:
        """Auth + content headers for the request."""

    @abstractmethod
    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Build the provider-specific request body."""

    @abstractmethod
    def _parse_response(
        self, data: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ModelResponse:
        """Translate a provider response body into a `ModelResponse`."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> ModelResponse:
        """Send one chat-completion request with retries; never returns partials."""
        api_key = self._resolve_api_key()
        payload = self._payload(messages, tools)
        payload.update(overrides)

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                if self._client is not None:
                    response = await self._client.post(
                        self._endpoint(), headers=self._headers(api_key), json=payload
                    )
                else:
                    async with httpx.AsyncClient(
                        timeout=self._config.request_timeout_seconds
                    ) as client:
                        response = await client.post(
                            self._endpoint(),
                            headers=self._headers(api_key),
                            json=payload,
                        )
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("model request timed out", attempt=attempt + 1)
                await self._backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("model request failed", attempt=attempt + 1, error=str(exc))
                await self._backoff(attempt)
                continue

            if response.status_code in {401, 403, 402}:
                # 402 = payment/quota exhausted: a credentials-class failure
                # (live-run finding: a mid-run 402 crashed the CLI with a raw
                # httpx traceback instead of the documented exit code 3).
                raise ModelAuthError(
                    f"{self._config.provider} rejected credentials "
                    f"(HTTP {response.status_code}): {response.text[:300]}"
                )
            if response.status_code in RETRYABLE_STATUS:
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                logger.warning(
                    "rate limited or server error",
                    status=response.status_code,
                    attempt=attempt + 1,
                    retry_after=retry_after,
                )
                await self._backoff(attempt, delay)
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
                continue

            response.raise_for_status()
            parsed = self._parse_response(response.json(), messages)
            logger.info(
                "model call complete",
                provider=self._config.provider,
                model=self._config.name,
                prompt_tokens=parsed.prompt_tokens,
                completion_tokens=parsed.completion_tokens,
                tool_calls=len(parsed.tool_calls),
            )
            return parsed

        msg = f"model request failed after {self._config.max_retries + 1} attempts: {last_error}"
        raise RuntimeError(msg)

    async def _backoff(self, attempt: int, delay: float | None = None) -> None:
        """Exponential backoff with jitter; explicit `delay` (Retry-After) wins."""
        base = self._config.extra.get("backoff_base_seconds", 0.5)
        wait = delay if delay is not None else base * (2**attempt) + random.uniform(0, 0.25)
        await asyncio.sleep(wait)

    async def aclose(self) -> None:  # noqa: B027 - intentional no-op default; not all
        """Release provider resources. Providers with sockets override this."""


def _approx_tokens(text: str) -> int:
    """Rough token estimate (chars/4) when a provider omits usage counts."""
    return max(1, len(text) // 4)
