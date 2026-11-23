# Plan: Ollama-compatible proxy in front of vLLM

A small Python service that listens on `http://127.0.0.1:11434` (Ollama's default
address) and forwards everything to a local vLLM server. Clients that expect
Ollama — the `ollama` CLI, `ollama-python`/`ollama-js`, Open WebUI, editor
plugins, anything using the OpenAI SDK against `:11434/v1` — keep working
unchanged, while inference is actually served by vLLM.

```
Ollama client ──► proxy 127.0.0.1:11434 ──► vLLM 127.0.0.1:8000
                  /api/*  (translated)       /v1/*  (OpenAI API, Bearer key)
                  /v1/*   (pass-through)
```

## 1. Target backend (given)

```bash
vllm serve /workspace/qwen3.8-27b-fp8 \
  --served-model-name dev-test \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --api-key "$VLLM_API_KEY"
```

What this implies for the proxy:

| Backend fact | Consequence |
|---|---|
| One model, served as `dev-test` | Proxy advertises one model; Ollama-style names (`dev-test:latest`, aliases) must map onto it |
| `--api-key` | Proxy injects `Authorization: Bearer $VLLM_API_KEY`; clients send no key (or a dummy one) |
| `--max-model-len 65536` | Reported as context length; `num_ctx` from clients cannot change it |
| `--enable-auto-tool-choice` + `qwen3_coder` parser | Tool calls arrive in OpenAI format and must be converted to Ollama format |
| `--reasoning-parser qwen3` | Reasoning arrives in a separate field and maps to Ollama's `thinking` |
| `--enable-prefix-caching` | Proxy must keep message/prompt prefixes byte-stable (no reordering, no injected timestamps) to preserve cache hits |
| No embedding model | `/api/embed` and `/v1/embeddings` are unsupported unless a second backend is configured |

## 2. Goals and non-goals

**Goals**

- Drop-in replacement for the Ollama HTTP server on `127.0.0.1:11434`.
- Native Ollama API under `/api`, OpenAI-compatible API under `/v1`.
- Full streaming support in both dialects, including tool calls and thinking.
- Stateless, single process, minimal dependencies.

**Non-goals**

- Model management (pull, push, create, copy, delete). vLLM serves a fixed
  model; these endpoints are stubbed so clients do not break.
- Loading/unloading models, `keep_alive` semantics, multi-model scheduling.
- Reproducing Ollama's prompt templating (`template`, `raw` Modelfile logic).

## 3. Stack

- Python 3.11+, managed with `uv` (matches the vLLM setup).
- **FastAPI** (Starlette) for routing, **uvicorn** as server.
- **httpx** `AsyncClient` for upstream calls (one shared client, HTTP keep-alive,
  connect timeout ~5 s, no read timeout for generations).
- **pydantic v2** models for Ollama request/response shapes, with unknown fields
  allowed so newer clients are not rejected.
- Tests: `pytest`, `pytest-asyncio`, `respx` (mock vLLM).

## 4. Configuration

Environment variables (optionally a `.env` file); no secrets in the repo.

| Variable | Default | Purpose |
|---|---|---|
| `PROXY_HOST` | `127.0.0.1` | Listen address |
| `PROXY_PORT` | `11434` | Listen port |
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | Upstream |
| `VLLM_API_KEY` | — (required) | Bearer token sent upstream; never logged |
| `DEFAULT_MODEL` | first id from upstream `/v1/models` | Model used when a name does not resolve |
| `MODEL_ALIASES` | empty | `alias=served-name,...`, e.g. `qwen3:latest=dev-test` |
| `STRICT_MODELS` | `false` | `true`: unknown model → 404 (Ollama behaviour). `false`: fall back to `DEFAULT_MODEL` |
| `CONTEXT_LENGTH` | `65536` | Reported by `/api/show`; should equal `--max-model-len` |
| `OLLAMA_VERSION` | `0.12.0` | Value returned by `/api/version` (clients gate features on it) |
| `ALLOWED_ORIGINS` | localhost origins | CORS, mirroring `OLLAMA_ORIGINS` |
| `LOG_LEVEL` | `info` | Logging; bodies only logged at `debug` |

Model name resolution, applied to every request: strip a trailing `:latest`,
apply `MODEL_ALIASES`, compare with upstream model ids, then apply the
`STRICT_MODELS` rule. Responses echo the name the client sent.

## 5. Endpoints

### 5.1 `/v1` — OpenAI-compatible (pass-through)

vLLM already speaks this dialect, so `/v1/*` is a streaming reverse proxy with a
few targeted rewrites.

| Route | Handling |
|---|---|
| `POST /v1/chat/completions` | Forward; rewrite `model`; relay SSE bytes unbuffered |
| `POST /v1/completions` | Same |
| `POST /v1/responses`, `POST /v1/messages` | Forward as-is if the installed vLLM exposes them; otherwise upstream 404 is returned |
| `GET /v1/models`, `GET /v1/models/{id}` | Forward; add configured aliases to the list |
| `POST /v1/embeddings` | Forward if an embeddings backend is configured, else 501 in OpenAI error format |
| anything else under `/v1` | Generic catch-all forward |

Rewrites:

- Replace the client's `Authorization` header with the vLLM key; drop hop-by-hop
  headers; never forward the vLLM key back.
- Resolve `model` (section 4) in the request; restore the client's name in the
  response (`model` field of JSON bodies and SSE chunks).
- Reasoning field: Ollama's `/v1` exposes `message.reasoning`; vLLM uses
  `reasoning_content` or `reasoning` depending on version. Emit `reasoning`,
  keeping the original field too.
- Errors stay in OpenAI format: `{"error": {"message", "type", "code"}}`.

### 5.2 `/api` — Ollama native (translated)

| Route | Handling |
|---|---|
| `GET /`, `HEAD /` | `200`, body `Ollama is running` (clients use this as a liveness probe) |
| `GET /api/version` | `{"version": OLLAMA_VERSION}` |
| `GET /api/tags` | Built from upstream `/v1/models` (cached ~10 s) in Ollama's `models[]` shape |
| `GET /api/ps` | Same model list, reported as loaded, `expires_at` far in the future |
| `POST /api/show` | Static `details`, `model_info` (context length), `capabilities: ["completion","tools","thinking"]` |
| `POST /api/chat` | → `POST /v1/chat/completions` (section 6) |
| `POST /api/generate` | → `POST /v1/chat/completions`; `raw: true` → `POST /v1/completions` (section 6.5) |
| `POST /api/embed`, `POST /api/embeddings` | → `/v1/embeddings` if configured, else `501 {"error": ...}` |
| `POST /api/pull` | Known model: success status stream (`pulling manifest` → `success`). Unknown: error |
| `POST /api/push`, `/api/create`, `/api/copy`, `DELETE /api/delete`, `/api/blobs/*` | `501 {"error": "not supported by vLLM proxy"}` |

`/api/tags`, `/api/ps` and `/api/show` need placeholder values for fields vLLM
cannot provide (`digest`, `size`, `modified_at`, `details.family`,
`parameter_size`, `quantization_level`); these are derived deterministically
from the model id plus config so they are stable across calls.

## 6. Translation: Ollama ⇄ OpenAI

### 6.1 Request (`/api/chat`)

| Ollama | OpenAI / vLLM |
|---|---|
| `model` | resolved served name |
| `messages[].role/content` | same |
| `messages[].images` (base64) | content parts `{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}` |
| `messages[].thinking` (assistant history) | dropped by default (keeps prefix cache stable and matches Qwen template practice) |
| `messages[].tool_calls[].function.arguments` (object) | `arguments` as JSON string, plus synthesized `id` and `type:"function"` |
| `role:"tool"` + `tool_name` | `role:"tool"` + `tool_call_id` matched to the preceding assistant tool call (by id if present, else by name/order) |
| `tools` | `tools` (same schema), `tool_choice:"auto"` |
| `stream` (default **true**) | `stream`, plus `stream_options:{"include_usage":true}` |
| `format:"json"` | `response_format:{"type":"json_object"}` |
| `format:{schema}` | `response_format:{"type":"json_schema","json_schema":{...}}` |
| `think: true/false` | `chat_template_kwargs:{"enable_thinking": bool}` |
| `think: "low"/"medium"/"high"` | treated as `true` |
| `options.num_predict` | `max_tokens` (`-1`/`-2` → omitted) |
| `options.temperature`, `top_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty` | same names |
| `options.top_k`, `min_p` | vLLM extra sampling params of the same name |
| `options.repeat_penalty` | `repetition_penalty` |
| `options.num_ctx`, `num_gpu`, `num_thread`, `mirostat*`, … | ignored (logged at debug) |
| `keep_alive` | ignored |

Tool-call ids are synthesized deterministically (`call_<index>_<hash>`), so the
same conversation history always serializes to the same upstream prompt.

### 6.2 Response, non-streaming

```json
{
  "model": "<name the client sent>",
  "created_at": "<RFC3339 UTC>",
  "message": {
    "role": "assistant",
    "content": "...",
    "thinking": "...",
    "tool_calls": [{"function": {"name": "...", "arguments": {}}}]
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 0, "load_duration": 0,
  "prompt_eval_count": 0, "prompt_eval_duration": 0,
  "eval_count": 0, "eval_duration": 0
}
```

- `thinking` ← `reasoning_content` or `reasoning`, whichever is present.
- `tool_calls[].function.arguments` ← parsed JSON object (invalid JSON is kept as
  `{"_raw": "..."}` rather than failing the request).
- `done_reason` ← `finish_reason` (`stop`/`tool_calls` → `stop`, `length` → `length`).
- Counts ← `usage.prompt_tokens` / `usage.completion_tokens`.
- Durations are in nanoseconds, measured by the proxy: `total_duration` is wall
  time, `eval_duration` is first-token to last-token, `prompt_eval_duration` is
  request to first token, `load_duration` is `0`.

### 6.3 Response, streaming

Upstream is SSE (`data: {...}\n\n`, terminated by `data: [DONE]`); downstream is
NDJSON (`application/x-ndjson`, one JSON object per line).

- Each content delta → `{"model","created_at","message":{"role":"assistant","content":"<delta>"},"done":false}`.
- Each reasoning delta → the same with `"thinking":"<delta>"` and empty `content`.
- Tool-call deltas arrive fragmented (name, then argument string pieces). They
  are accumulated per `index` and emitted as one complete `tool_calls` chunk
  when the call is complete (next index starts, or `finish_reason` arrives).
- The final usage chunk produces the closing line with `"done":true`,
  `done_reason` and all counters; `[DONE]` is consumed, not forwarded.
- Lines are flushed immediately; no response buffering or compression.

### 6.4 Lifecycle and errors

- Client disconnect cancels the upstream request, so vLLM aborts generation and
  frees the GPU slot.
- Upstream errors before the first byte → same status code with
  `{"error": "<message>"}`. Context overflow (vLLM 400) → `400` with the
  upstream message.
- Errors after streaming has started → a final NDJSON line `{"error": "..."}`
  (what Ollama clients expect mid-stream).
- vLLM unreachable → `502`; vLLM still loading (`/health` not ready) → `503`.
  The proxy starts and stays up regardless of vLLM state.

### 6.5 `/api/generate`

- `prompt` (+ optional `system`, `images`) is wrapped into chat messages and sent
  to `/v1/chat/completions`, so the model's chat template, the reasoning parser
  and `think` all apply. Response field is `response` instead of `message`,
  plus `thinking`.
- `raw: true` → `/v1/completions` with the prompt untouched.
- `suffix` → `/v1/completions` with `suffix` if upstream accepts it, else 400.
- `template` → ignored with a debug log; `context` → accepted and ignored, the
  response returns an empty `context` array.
- Empty `prompt` → immediate `{"done":true,"done_reason":"load"}` (clients use
  this to preload); with `keep_alive: 0` → `"done_reason":"unload"`. No upstream
  call in either case.

## 7. Cross-cutting concerns

- **Security**: bind to loopback only by default; the vLLM key lives only in the
  environment, is attached only to upstream requests, and is redacted from logs
  and error messages. Binding to a non-loopback address logs a warning, since
  the proxy would then expose the model without authentication.
- **CORS**: same defaults as Ollama (localhost, `127.0.0.1`, `app://`,
  `vscode-webview://` etc.), configurable.
- **Concurrency**: fully async; no proxy-side queue — vLLM's continuous batching
  handles parallel requests. Shared httpx pool sized generously (e.g. 100).
- **Logging**: one line per request (method, path, model, status, token counts,
  duration, stream yes/no). Prompt content only at `debug`.
- **Health**: `GET /healthz` reports proxy status and upstream `/health`.
- **Port conflict**: a real Ollama service on the host must be stopped/disabled
  (`systemctl disable --now ollama`), otherwise `11434` is taken; the proxy exits
  with a clear message if the bind fails.

## 8. Project layout

```
pyproject.toml              # uv project, entry point `ollama-vllm-proxy`
README.md
src/ollama_vllm_proxy/
  __main__.py               # uvicorn launcher
  app.py                    # FastAPI app, lifespan (httpx client), CORS, error handlers
  config.py                 # settings from env
  upstream.py               # httpx client, auth header, error mapping, model list cache
  models.py                 # name resolution, aliases, Ollama model metadata
  schemas.py                # pydantic models for Ollama requests/responses
  routes/
    root.py                 # /, /api/version, /healthz
    openai.py               # /v1/* pass-through
    chat.py                 # /api/chat
    generate.py             # /api/generate
    embed.py                # /api/embed, /api/embeddings
    registry.py             # /api/tags, /api/ps, /api/show, pull/push/create/copy/delete stubs
  translate/
    request.py              # Ollama → OpenAI (messages, tools, options, format, think)
    response.py             # OpenAI → Ollama (non-streaming)
    stream.py               # SSE parser, tool-call accumulator, NDJSON emitter, timings
tests/
  unit/                     # translation tables, tool-call accumulation, name resolution
  integration/              # app + respx-mocked vLLM, streaming and error paths
  e2e/                      # opt-in, against a live vLLM
deploy/ollama-vllm-proxy.service   # systemd unit (After=vllm.service, EnvironmentFile for the key)
```

## 9. Milestones

1. **Skeleton** — project setup, config, lifespan-managed httpx client, `/`,
   `/api/version`, `/healthz`.
2. **`/v1` pass-through** — streaming reverse proxy, auth injection, model
   rewrite. *Check*: OpenAI SDK with `base_url=http://127.0.0.1:11434/v1`
   streams a chat completion.
3. **Model discovery** — `/api/tags`, `/api/ps`, `/api/show`, name resolution.
   *Check*: `ollama list` and `ollama show dev-test` work.
4. **`/api/chat`** — non-streaming, then streaming, options mapping.
   *Check*: `ollama run dev-test` gives an interactive chat.
5. **Thinking + tools + structured output** — `think`, `thinking` deltas,
   tool-call accumulation, tool result round-trip, `format`.
   *Check*: `ollama-python` tool-calling example completes a full round trip.
6. **`/api/generate`** — chat-wrapped and raw modes, load/unload shortcuts.
7. **Stubs and errors** — pull/push/create/copy/delete, embeddings, upstream
   failure mapping, mid-stream errors, client-disconnect cancellation.
8. **Packaging** — README, systemd unit, CI (ruff, pytest).

## 10. Testing

- **Unit**: every row of the mapping tables in section 6; fragmented tool-call
  streams (single, multiple, interleaved with content and reasoning); invalid
  argument JSON; name resolution with and without `STRICT_MODELS`.
- **Integration** (mocked vLLM): golden request/response pairs per endpoint;
  streaming framing (each line valid JSON, exactly one `done:true`, last);
  upstream 400/401/500/timeouts; disconnect propagates cancellation; the vLLM
  key never appears in any response or log line.
- **Client compatibility** (live vLLM): `ollama` CLI (`list`, `show`, `run`,
  `ps`), `ollama-python` (chat, generate, stream, tools, think), OpenAI SDK
  (chat, stream, tools), and Open WebUI pointed at the proxy.
- **Prefix cache check**: send the same multi-turn conversation twice and
  confirm from vLLM metrics that the second request hits the prefix cache.

## 11. Open points to verify during implementation

1. **Reasoning field name** in the installed vLLM (`reasoning_content` vs
   `reasoning`) — the proxy reads both, but tests should pin the actual one.
2. **Thinking toggle** — confirm that `chat_template_kwargs.enable_thinking` is
   what this Qwen chat template honours, and what the model does when `think`
   is omitted (Ollama clients on recent versions expect thinking to be on for
   thinking-capable models).
3. **Structured output parameter** — `response_format` with `json_schema` vs the
   vLLM-specific structured-output field for the installed version.
4. **Vision** — whether the served checkpoint accepts images; if not,
   `messages[].images` should return a clear 400 and `vision` stays out of
   `capabilities`.
5. **`/v1/responses` and `/v1/messages`** availability upstream.
6. **Embeddings** — whether a second vLLM instance with an embedding model is
   wanted; if so add `EMBED_BASE_URL` / `EMBED_MODEL`.
7. **`OLLAMA_VERSION` value** — pick one high enough that target clients enable
   tools and thinking.
