#!/usr/bin/env bun
// zai_openai_shim.mjs — local OpenAI-compatible chat-completions bridge over
// z-ai-web-dev-sdk (GLM family).
//
// Purpose: gives the Foreman harness a reachable LLM backend in dev
// sandboxes where external providers are blocked, region-restricted, or out
// of quota. It speaks the exact wire contract consumed by
// src/harness/infrastructure/model_providers/openai_compatible.py:
//
//   POST /v1/chat/completions  {model, messages, tools?, temperature, max_tokens}
//   GET  /v1/models
//   GET  /healthz
//
// The upstream backend supports native OpenAI function calling (tools param
// in, tool_calls out) and accepts assistant/tool role replay, so the shim is
// a transparent passthrough on the message body. What it adds:
//
//   - one shared SDK instance (auth handshake once, not per request)
//   - upstream throttling: min interval between calls + bounded concurrency
//     (the backend 429s bursts; observed with 2 requests ~1s apart)
//   - in-shim retry with exponential backoff for 429/5xx/network errors, so
//     the harness's own provider ladder stays a second line of defense
//   - per-request deadline -> 504 JSON (engages the provider ladder)
//   - OpenAI-style JSON error envelopes and a /healthz stats page
//
// Run:      bun tools/dev/zai_openai_shim.mjs
// Env:      SHIM_PORT=8788  SHIM_HOST=127.0.0.1
//           ZAI_SDK_PATH=<path to z-ai-web-dev-sdk dist/index.js>
//           ZAI_MODEL=glm-4-plus            (default model when unset)
//           SHIM_MAX_CONCURRENCY=2          (parallel upstream calls)
//           SHIM_MIN_INTERVAL_MS=2000       (min gap between upstream starts)
//           SHIM_RETRIES=4                  (429/5xx/network attempts)
//           SHIM_BACKOFF_BASE_MS=3000       (x2 per retry, +jitter)
//           SHIM_DEADLINE_MS=220000         (per-request wall clock)
//
// The shim only listens on loopback and ignores Authorization headers (local
// trust boundary). It is dev tooling: point harness.yaml's default model at
// http://127.0.0.1:8788/v1 with any non-empty AI_API_KEY value.

import http from 'node:http';

const PORT = Number(process.env.SHIM_PORT || 8788);
const HOST = process.env.SHIM_HOST || '127.0.0.1';
const SDK_PATH =
  process.env.ZAI_SDK_PATH ||
  '/home/z/.bun/install/global/node_modules/z-ai-web-dev-sdk/dist/index.js';
const DEFAULT_MODEL = process.env.ZAI_MODEL || 'glm-4-plus';
const MAX_CONCURRENCY = Number(process.env.SHIM_MAX_CONCURRENCY || 2);
const MIN_INTERVAL_MS = Number(process.env.SHIM_MIN_INTERVAL_MS || 2000);
const RETRIES = Number(process.env.SHIM_RETRIES || 4);
const BACKOFF_BASE_MS = Number(process.env.SHIM_BACKOFF_BASE_MS || 3000);
const DEADLINE_MS = Number(process.env.SHIM_DEADLINE_MS || 220000);
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const stats = {
  started_at: Date.now(),
  requests: 0,
  upstream_calls: 0,
  upstream_retries: 0,
  upstream_429: 0,
  errors: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  last_error: '',
  last_request_at: 0,
};

const log = (level, msg, extra = {}) => {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, msg, ...extra });
  console.log(line);
};

// --- upstream throttle: gap between call starts + bounded concurrency ------
let inFlight = 0;
let lastStart = 0;
const waiters = [];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function acquireSlot(deadline) {
  while (true) {
    if (Date.now() > deadline) throw new Error('deadline exceeded while queued');
    if (inFlight < MAX_CONCURRENCY) {
      const gap = lastStart + MIN_INTERVAL_MS - Date.now();
      if (gap <= 0) {
        inFlight += 1;
        lastStart = Date.now();
        return;
      }
      await sleep(Math.min(gap, 1000));
      continue;
    }
    const notified = new Promise((resolve) => waiters.push(resolve));
    const winner = await Promise.race([notified, sleep(1000)]);
    if (winner) {
      const gap = lastStart + MIN_INTERVAL_MS - Date.now();
      if (gap > 0) await sleep(Math.min(gap, 5000));
      inFlight += 1;
      lastStart = Date.now();
      return;
    }
  }
}

function releaseSlot() {
  inFlight -= 1;
  const notify = waiters.shift();
  if (notify) notify(true);
}

// --- upstream call with retry/backoff ---------------------------------------
function classifyError(err) {
  const text = String(err && err.message ? err.message : err);
  const statusMatch = text.match(/status (\d{3})/);
  const status = statusMatch ? Number(statusMatch[1]) : 0;
  if (status === 429 || /too many requests/i.test(text)) return { retryable: true, status: 429 };
  if (status >= 500) return { retryable: true, status };
  if (status >= 400 && status < 500) return { retryable: false, status };
  if (/fetch failed|network|ECONN|timeout|socket/i.test(text)) return { retryable: true, status: 503 };
  return { retryable: false, status: 500 };
}

async function callUpstream(zai, body, deadline) {
  let attempt = 0;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    attempt += 1;
    await acquireSlot(deadline);
    const upstreamStart = Date.now();
    try {
      stats.upstream_calls += 1;
      const payload = {
        messages: body.messages,
        stream: false,
        thinking: { type: 'disabled' },
      };
      const model = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : DEFAULT_MODEL;
      if (model) payload.model = model;
      if (Number.isFinite(body.temperature)) payload.temperature = body.temperature;
      if (Number.isFinite(body.max_tokens) && body.max_tokens > 0) payload.max_tokens = body.max_tokens;
      if (Array.isArray(body.tools) && body.tools.length) {
        payload.tools = body.tools;
        if (body.tool_choice) payload.tool_choice = body.tool_choice;
      }
      const completion = await zai.chat.completions.create(payload);
      const elapsed = Date.now() - upstreamStart;
      const usage = completion && completion.usage ? completion.usage : {};
      stats.prompt_tokens += Number(usage.prompt_tokens || 0);
      stats.completion_tokens += Number(usage.completion_tokens || 0);
      log('info', 'upstream ok', {
        model: payload.model,
        messages: body.messages.length,
        tools: (body.tools || []).length,
        attempt,
        upstream_ms: elapsed,
        prompt_tokens: usage.prompt_tokens || 0,
        completion_tokens: usage.completion_tokens || 0,
      });
      return { completion, model: payload.model, attempts: attempt };
    } catch (err) {
      const { retryable, status } = classifyError(err);
      if (status === 429) stats.upstream_429 += 1;
      stats.last_error = String(err && err.message ? err.message : err).slice(0, 300);
      const canRetry = retryable && attempt <= RETRIES && Date.now() < deadline;
      log(canRetry ? 'warn' : 'error', 'upstream error', {
        attempt,
        status,
        retryable,
        error: stats.last_error,
      });
      if (!canRetry) {
        const httpStatus = status >= 400 ? status : 502;
        throw Object.assign(new Error(stats.last_error), { httpStatus });
      }
      stats.upstream_retries += 1;
      const backoff = BACKOFF_BASE_MS * 2 ** (attempt - 1) + Math.random() * 500;
      await sleep(Math.min(backoff, Math.max(deadline - Date.now(), 0)));
    } finally {
      releaseSlot();
    }
  }
}

// --- response shaping --------------------------------------------------------
function shapeCompletion(completion, model) {
  const shaped = { ...(completion || {}) };
  shaped.object = shaped.object || 'chat.completion';
  shaped.model = shaped.model || model;
  if (!shaped.created) shaped.created = Math.floor(Date.now() / 1000);
  shaped.choices = Array.isArray(shaped.choices) && shaped.choices.length ? shaped.choices : [
    { index: 0, message: { role: 'assistant', content: '' }, finish_reason: 'stop' },
  ];
  const choice = shaped.choices[0];
  if (!choice.message) choice.message = { role: 'assistant', content: '' };
  if (!choice.finish_reason) choice.finish_reason = 'stop';
  if (!shaped.usage) {
    shaped.usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  }
  return shaped;
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function sendError(res, status, message, type = 'shim_error') {
  stats.errors += 1;
  sendJson(res, status, { error: { message, type, code: status } });
}

function readBody(req, deadline) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(Object.assign(new Error('request body too large'), { httpStatus: 413 }));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// --- server ------------------------------------------------------------------
async function main() {
  const { default: ZAI } = await import(SDK_PATH);
  const zai = await ZAI.create();
  log('info', 'z-ai shim ready', { sdk: SDK_PATH, model: DEFAULT_MODEL });

  const server = http.createServer(async (req, res) => {
    const started = Date.now();
    const url = (req.url || '').split('?')[0];
    try {
      if (req.method === 'GET' && (url === '/healthz' || url === '/health')) {
        sendJson(res, 200, {
          ok: true,
          uptime_s: Math.floor((Date.now() - stats.started_at) / 1000),
          ...stats,
          started_at: new Date(stats.started_at).toISOString(),
        });
        return;
      }
      if (req.method === 'GET' && (url === '/v1/models' || url === '/models')) {
        sendJson(res, 200, {
          object: 'list',
          data: [{ id: DEFAULT_MODEL, object: 'model', owned_by: 'zai-shim' }],
        });
        return;
      }
      const isChat =
        req.method === 'POST' && (url === '/v1/chat/completions' || url === '/chat/completions');
      if (!isChat) {
        sendError(res, 404, `no such route: ${req.method} ${url}`, 'not_found');
        return;
      }

      stats.requests += 1;
      stats.last_request_at = Date.now();
      const deadline = started + DEADLINE_MS;
      const raw = await readBody(req, deadline);
      let body;
      try {
        body = JSON.parse(raw || '{}');
      } catch {
        sendError(res, 400, 'request body is not valid JSON', 'invalid_json');
        return;
      }
      if (!Array.isArray(body.messages) || body.messages.length === 0) {
        sendError(res, 400, 'messages must be a non-empty array', 'invalid_request');
        return;
      }
      if (body.stream) {
        sendError(res, 400, 'streaming is not supported by this shim', 'stream_unsupported');
        return;
      }

      const { completion, model, attempts } = await callUpstream(zai, body, deadline);
      const shaped = shapeCompletion(completion, model);
      log('info', 'request ok', { attempts, total_ms: Date.now() - started });
      sendJson(res, 200, shaped);
    } catch (err) {
      const httpStatus = err.httpStatus || (Date.now() > started + DEADLINE_MS ? 504 : 502);
      sendError(res, httpStatus, String(err && err.message ? err.message : err).slice(0, 400));
    }
  });

  server.requestTimeout = 0; // long model calls; deadline handled in-shim
  server.headersTimeout = 60_000;
  server.listen(PORT, HOST, () => {
    log('info', 'listening', {
      host: HOST,
      port: PORT,
      max_concurrency: MAX_CONCURRENCY,
      min_interval_ms: MIN_INTERVAL_MS,
      retries: RETRIES,
      deadline_ms: DEADLINE_MS,
    });
  });

  const shutdown = () => {
    log('info', 'shutting down');
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 3000).unref();
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

main().catch((err) => {
  log('error', 'fatal startup failure', { error: String(err).slice(0, 400) });
  process.exit(1);
});
