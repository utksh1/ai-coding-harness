"""Model provider tests: payloads, parsing, fake flow, auth, retry (issue 1.5/1.8)."""

from __future__ import annotations

import httpx
import pytest

from harness.config import ModelConfig
from harness.infrastructure.model_providers import (
    FakeProvider,
    ModelAuthError,
    ModelResponse,
    OpenAICompatibleProvider,
    ToolCall,
    create_model_provider,
)
from harness.infrastructure.model_providers.base import RETRYABLE_STATUS

MSGS = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "fix the bug in parser.py"},
]
TOOLS = [
    {"name": "read_file", "description": "read", "parameters": {"type": "object", "properties": {}}}
]


def test_openai_payload_and_endpoint() -> None:
    provider = OpenAICompatibleProvider(
        ModelConfig(provider="openai-compatible", name="m1", base_url="http://x/v1/")
    )
    payload = provider._payload(MSGS, TOOLS)
    assert payload["model"] == "m1"
    assert payload["tools"][0] == {"type": "function", "function": TOOLS[0]}
    assert provider._endpoint() == "http://x/v1/chat/completions"


def test_openai_tool_call_parsing() -> None:
    provider = OpenAICompatibleProvider(ModelConfig(provider="openai-compatible", name="m1"))
    data = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "read_file", "arguments": '{"path": "parser.py"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
    }
    response = provider._parse_response(data, MSGS)
    assert response.tool_calls == [
        ToolCall(id="c1", name="read_file", arguments={"path": "parser.py"})
    ]
    assert (response.prompt_tokens, response.completion_tokens) == (100, 5)


def test_openai_malformed_tool_arguments_do_not_raise() -> None:
    provider = OpenAICompatibleProvider(ModelConfig(provider="openai-compatible", name="m1"))
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "", "function": {"name": "x", "arguments": "{not json"}}]
                }
            }
        ]
    }
    response = provider._parse_response(data, MSGS)
    assert response.tool_calls[0].arguments == {"_raw": "{not json"}


def test_factory_dispatch(fake_model_config) -> None:
    provider = create_model_provider(fake_model_config)
    assert provider.name == "fake"


async def test_fake_provider_script_and_exhaustion(make_fake_provider) -> None:
    provider = make_fake_provider(
        ModelResponse(content="thinking", tool_calls=[ToolCall(name="read_file", arguments={})]),
        ModelResponse(content="done"),
    )
    first = await provider.generate(MSGS, TOOLS)
    assert first.tool_calls[0].name == "read_file"
    assert first.prompt_tokens > 0  # estimated when provider omits usage
    assert (await provider.generate(MSGS)).content == "done"
    with pytest.raises(RuntimeError, match="exhausted"):
        await provider.generate(MSGS)


async def test_missing_api_key_fails_fast(monkeypatch) -> None:
    monkeypatch.delenv("MY_KEY", raising=False)
    for alt in ("AI_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(alt, raising=False)
    provider = OpenAICompatibleProvider(
        ModelConfig(provider="openai", name="m", api_key_env="MY_KEY")
    )
    with pytest.raises(ModelAuthError, match="MY_KEY"):
        await provider.generate(MSGS)


def _retry_transport(statuses: list[int]) -> httpx.AsyncClient:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses[min(calls["n"], len(statuses) - 1)]
        calls["n"] += 1
        return httpx.Response(status, json={"choices": [{"message": {"content": "ok"}}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
async def test_retryable_statuses_eventually_succeed(monkeypatch, status: int) -> None:
    monkeypatch.setenv("K", "v")
    config = ModelConfig(
        provider="openai",
        name="m",
        api_key_env="K",
        max_retries=3,
        extra={"backoff_base_seconds": 0.001},
    )
    client = _retry_transport([status, 200])
    provider = OpenAICompatibleProvider(config, client=client)
    response = await provider.generate(MSGS)
    assert response.content == "ok"


async def test_auth_rejection_is_not_retried(monkeypatch) -> None:
    monkeypatch.setenv("K", "v")
    client = _retry_transport([401])
    provider = OpenAICompatibleProvider(
        ModelConfig(provider="openai", name="m", api_key_env="K"), client=client
    )
    with pytest.raises(ModelAuthError):
        await provider.generate(MSGS)


async def test_payment_required_is_credentials_class_not_retried(monkeypatch) -> None:
    """Live-run finding: a 402 (quota/payment exhausted) mid-run must raise
    ModelAuthError so the CLI can exit 3 per the documented contract — not
    escape as a raw httpx traceback."""
    monkeypatch.setenv("K", "v")
    client = _retry_transport([402])
    provider = OpenAICompatibleProvider(
        ModelConfig(
            provider="openai",
            name="m",
            api_key_env="K",
            max_retries=3,
            extra={"backoff_base_seconds": 0.001},
        ),
        client=client,
    )
    with pytest.raises(ModelAuthError, match="402"):
        await provider.generate(MSGS)


async def test_retries_exhaustion_raises(monkeypatch) -> None:
    monkeypatch.setenv("K", "v")
    client = _retry_transport([503])
    provider = OpenAICompatibleProvider(
        ModelConfig(
            provider="openai",
            name="m",
            api_key_env="K",
            max_retries=2,
            extra={"backoff_base_seconds": 0.001},
        ),
        client=client,
    )
    with pytest.raises(RuntimeError, match="attempts"):
        await provider.generate(MSGS)


# --- transport branches and remaining seams (100% coverage pass) --------------


def _handler_raising(exc: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


async def test_non_injected_client_path_succeeds(monkeypatch) -> None:
    """The no-injection branch (base builds its own AsyncClient) is covered."""
    monkeypatch.setenv("K", "v")
    transport_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
    )
    monkeypatch.setattr(
        "harness.infrastructure.model_providers.base.httpx.AsyncClient",
        lambda **kwargs: transport_client,
    )
    provider = OpenAICompatibleProvider(ModelConfig(provider="openai", name="m", api_key_env="K"))
    response = await provider.generate(MSGS)
    assert response.content == "ok"
    assert response.total_tokens > 0
    await transport_client.aclose()


async def test_timeout_is_retried_then_exhausted(monkeypatch) -> None:
    monkeypatch.setenv("K", "v")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            _handler_raising(
                httpx.ConnectTimeout("slow", request=httpx.Request("POST", "http://x"))
            )
        )
    )
    provider = OpenAICompatibleProvider(
        ModelConfig(
            provider="openai",
            name="m",
            api_key_env="K",
            max_retries=1,
            extra={"backoff_base_seconds": 0.001},
        ),
        client=client,
    )
    with pytest.raises(RuntimeError, match="attempts"):
        await provider.generate(MSGS)
    await client.aclose()


async def test_transport_error_is_retried_then_exhausted(monkeypatch) -> None:
    monkeypatch.setenv("K", "v")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            _handler_raising(
                httpx.ConnectError("refused", request=httpx.Request("POST", "http://x"))
            )
        )
    )
    provider = OpenAICompatibleProvider(
        ModelConfig(
            provider="openai",
            name="m",
            api_key_env="K",
            max_retries=1,
            extra={"backoff_base_seconds": 0.001},
        ),
        client=client,
    )
    with pytest.raises(RuntimeError, match="attempts"):
        await provider.generate(MSGS)
    await client.aclose()


async def test_retry_after_header_overrides_backoff(monkeypatch) -> None:
    monkeypatch.setenv("K", "v")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0"}, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        ModelConfig(provider="openai", name="m", api_key_env="K", max_retries=1),
        client=client,
    )
    with pytest.raises(RuntimeError, match="attempts"):
        await provider.generate(MSGS)
    await client.aclose()


async def test_fake_provider_pass_through_seams(fake_model_config) -> None:
    provider = FakeProvider(fake_model_config, responses=[ModelResponse(content="x")])
    assert provider._resolve_api_key() == "fake-key"
    assert provider._endpoint() == "fake://generate"
    assert provider._headers("k") == {}
    assert provider._payload(MSGS, TOOLS) == {"messages": MSGS, "tools": TOOLS}
    with pytest.raises(NotImplementedError):
        provider._parse_response({}, MSGS)
    await provider.aclose()


async def test_factory_dispatch_and_unknown_provider(fake_model_config) -> None:
    from types import SimpleNamespace

    from harness.infrastructure.model_providers import (
        OpenAICompatibleProvider,
        create_model_provider,
    )

    assert isinstance(create_model_provider(fake_model_config), FakeProvider)
    assert isinstance(
        create_model_provider(ModelConfig(provider="openai-compatible", name="m")),
        OpenAICompatibleProvider,
    )
    bogus = SimpleNamespace(provider="does-not-exist")
    with pytest.raises(ValueError, match="unknown model provider"):
        create_model_provider(bogus)  # type: ignore[arg-type]


async def test_fake_provider_scripted_exception(make_fake_provider) -> None:
    provider = make_fake_provider(ValueError("model blew up"))
    with pytest.raises(ValueError, match="model blew up"):
        await provider.generate(MSGS)


def test_safe_json_accepts_dict_arguments() -> None:
    from harness.infrastructure.model_providers.openai_compatible import _safe_json

    assert _safe_json({"already": "parsed"}) == {"already": "parsed"}
