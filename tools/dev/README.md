# Dev tools

Sandbox-local bridges and utilities that are not part of the harness product
but make development and validation possible inside restricted environments.

## zai_openai_shim.mjs

A loopback-only OpenAI-compatible chat-completions server bridging the
z-ai-web-dev-sdk (GLM family) — used when external providers are
region-blocked or out of quota. The upstream backend already speaks the
OpenAI wire format, including native `tools` / `tool_calls`, so the shim is a
transparent passthrough that adds what the SDK lacks for server use:

- one shared SDK instance (single auth handshake)
- throttling (min interval + bounded concurrency) — the backend 429s bursts
- retry with exponential backoff for 429 / 5xx / network errors
- per-request deadline returning a 504 JSON error (feeds the harness
  provider's own retry ladder)
- `/healthz` stats and `/v1/models`

### Run

```bash
bun tools/dev/zai_openai_shim.mjs
# SHIM_PORT=8788 ZAI_MODEL=glm-4-plus SHIM_MIN_INTERVAL_MS=2000 ...
```

Then point the harness at it:

```yaml
models:
  default:
    provider: openai-compatible
    name: glm-4-plus
    api_key_env: AI_API_KEY      # any non-empty value; the shim ignores it
    base_url: http://127.0.0.1:8788/v1
```

Requires the `z-ai-web-dev-sdk` package on disk (`ZAI_SDK_PATH` overrides the
default install location). The shim binds to loopback only and performs no
authentication — it is a dev convenience, never expose it.
