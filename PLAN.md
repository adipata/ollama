# Plan: Ollama-compatible proxy in front of vLLM

A small Python service that listens on `http://127.0.0.1:11434` (Ollama's default
address) and forwards everything to vLLM running on a Runpod pod, reached through
an SSH tunnel that the application manages itself. Clients that expect
Ollama — `ollama-python`/`ollama-js`, Open WebUI, editor plugins, anything
using the OpenAI SDK against `:11434/v1` — keep working once they are given a
proxy API key, while inference is actually served by vLLM.

```
                  local machine (Linux/macOS)        │        Runpod pod
                                                     │
Ollama client ──► proxy 127.0.0.1:11434 ══ SSH ══════╪══► vLLM 127.0.0.1:8000
 (proxy API key)  /api/*   (translated)   (managed   │      /v1/*, /metrics, /health
                  /v1/*    (pass-through)  by the    ├══► GPU helper 127.0.0.1:9400
                  /admin/* (keys, deploy,  proxy)    │      /metrics, /inventory
                            monitoring)              └══► remote commands, SFTP, log tail
                      │
                      └─► SQLite (API keys, pod and profiles, request metadata,
                                  engine and GPU samples)
```

Callers must present an API key issued by the proxy. Keys are created in a
management web interface, which also hosts a live monitoring page and a
downloadable session report modelled on the one in
`adipata/confidential-inference-test` (`proxy/perfproxy`).

The same interface deploys the backend: after you start a plain Ubuntu pod
(`runpod/base:2.1.0-cuda1320-ubuntu2404`) in the Runpod console and enter its
SSH address, a **Deploy** button installs vLLM, downloads the model, installs
the monitoring helper, starts vLLM and shows the progress and the vLLM log.

## 1. Target backend (given)

vLLM runs in a Runpod pod started by hand from the plain Ubuntu 24.04 template
`runpod/base:2.1.0-cuda1320-ubuntu2404`, which contains no vLLM. The proxy
installs and starts it there (section 12) with the command of the selected
deployment profile; the default profile is:

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

At deploy time the key is passed through the `VLLM_API_KEY` environment
variable rather than `--api-key`, so it stays off the command line.

What this implies for the proxy:

| Backend fact | Consequence |
|---|---|
| One model, served as `dev-test` | Proxy advertises one model; Ollama-style names (`dev-test:latest`, aliases) must map onto it |
| `--api-key` | Proxy injects `Authorization: Bearer $VLLM_API_KEY`; clients never see this key and use proxy-issued keys instead |
| `--max-model-len 65536` | Reported as context length; `num_ctx` from clients cannot change it |
| `--enable-auto-tool-choice` + `qwen3_coder` parser | Tool calls arrive in OpenAI format and must be converted to Ollama format |
| `--reasoning-parser qwen3` | Reasoning arrives in a separate field and maps to Ollama's `thinking` |
| `--enable-prefix-caching` | Proxy must keep message/prompt prefixes byte-stable (no reordering, no injected timestamps) to preserve cache hits |
| No embedding model | `/api/embed` and `/v1/embeddings` are unsupported unless a second backend is configured |
| vLLM binds `127.0.0.1` inside the pod | Nothing is exposed on Runpod's public ports except SSH; all traffic goes through the tunnel |
| Pod IP, SSH port and host key change with every rental | Connection details are entered in the management interface, not in config files |
| No network volume; everything is on the pod's container disk | Nothing survives the pod. Every new pod needs the full install and download, so Deploy must do all of it unattended |

## 2. Goals and non-goals

**Goals**

- Drop-in replacement for the Ollama HTTP server on `127.0.0.1:11434`.
- Native Ollama API under `/api`, OpenAI-compatible API under `/v1`.
- Full streaming support in both dialects, including tool calls and thinking.
- Mandatory API keys for every `/api` and `/v1` call, managed through a web
  interface.
- Monitoring: a live page plus a self-contained downloadable report covering
  requests, the vLLM engine and the GPUs.
- One-click deployment of vLLM onto a running Runpod pod, with a managed SSH
  tunnel, live progress and log view.
- Single process, one SQLite file, minimal dependencies. Runs natively on
  Linux and macOS.

**Non-goals**

- Model management (pull, push, create, copy, delete). vLLM serves a fixed
  model; these endpoints are stubbed so clients do not break.
- Loading/unloading models, `keep_alive` semantics, multi-model scheduling.
- Reproducing Ollama's prompt templating (`template`, `raw` Modelfile logic).
- Creating, stopping or terminating Runpod pods. You do that in the Runpod
  console; the application never talks to the Runpod API and holds no Runpod
  credentials.
- Windows and Docker packaging.
- Per-key expiry, rate limits, quotas or usage billing. A key has a name and
  can be disabled or revoked; nothing more.
- Admin user accounts. One admin token protects the management interface.
- Storing prompts, completions, reasoning text or tool arguments. Only timing
  and token metadata is recorded.

**Known consequence of mandatory keys**: clients that cannot send an
`Authorization` header stop working. That includes the stock `ollama` CLI
against a local server. The OpenAI SDK (`api_key=`), `ollama-python` /
`ollama-js` (`headers=`), Open WebUI and most editor plugins can send one.

## 3. Stack

- Python 3.11+, managed with `uv` (matches the vLLM setup).
- **FastAPI** (Starlette) for routing, **uvicorn** as server.
- **httpx** `AsyncClient` for upstream calls (one shared client, HTTP keep-alive,
  connect timeout ~5 s, no read timeout for generations).
- **pydantic v2** models for Ollama request/response shapes, with unknown fields
  allowed so newer clients are not rejected.
- **sqlite3** from the standard library (no ORM), WAL mode.
- **asyncssh** for the pod connection: remote commands, SFTP upload, local port
  forwards and log tailing over one multiplexed connection, in-process, with
  explicit host-key handling.
- Management and monitoring pages: static HTML with inline CSS/JS and inline
  SVG charts, served by the proxy. No build step, no external assets.
- Tests: `pytest`, `pytest-asyncio`, `respx` (mock vLLM).

## 4. Configuration

Environment variables (optionally a `.env` file); no secrets in the repo.

| Variable | Default | Purpose |
|---|---|---|
| `PROXY_HOST` | `127.0.0.1` | Listen address |
| `PROXY_PORT` | `11434` | Listen port |
| `VLLM_BASE_URL` | empty | Set only for *external mode*: use an already running vLLM at this URL and disable pod management |
| `VLLM_API_KEY` | empty | External mode only. In managed mode the key is generated at deploy time (section 12) |
| `SSH_KEY_PATH` | `~/.ssh/id_ed25519` | Default private key offered in the pod form; the key file is read, never copied |
| `DEFAULT_MODEL` | served name of the active profile | Model used when a name does not resolve |
| `MODEL_ALIASES` | empty | `alias=served-name,...`, e.g. `qwen3:latest=dev-test` |
| `STRICT_MODELS` | `false` | `true`: unknown model → 404 (Ollama behaviour). `false`: fall back to `DEFAULT_MODEL` |
| `CONTEXT_LENGTH` | from the active profile | Reported by `/api/show`; override only in external mode |
| `OLLAMA_VERSION` | `0.12.0` | Value returned by `/api/version` (clients gate features on it) |
| `ALLOWED_ORIGINS` | localhost origins | CORS, mirroring `OLLAMA_ORIGINS` |
| `LOG_LEVEL` | `info` | Logging; bodies only logged at `debug` |
| `DATA_DIR` | `./data` | Holds `proxy.sqlite3` and `admin_token.txt` |
| `ADMIN_TOKEN` | generated at first start | Overrides the token file; protects `/admin/*` |
| `SAMPLE_INTERVAL_S` | `1.5` | Interval for scraping vLLM `/metrics` and the GPUs |
| `GPU_EXPORTER_URL` | empty | External mode only: URL of a GPU helper, or `off`. In managed mode the helper is installed on the pod and reached through the tunnel |
| `RETENTION_DAYS` | `30` | Samples and request rows older than this are pruned; `0` keeps everything |

Pod address, deployment profiles and the generated vLLM key are not
environment settings; they are entered in the management interface and stored
in SQLite.

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

- Validate the client's proxy key (section 9), then replace the `Authorization`
  header with the vLLM key; drop hop-by-hop headers; never forward the vLLM key
  back.
- Resolve `model` (section 4) in the request; restore the client's name in the
  response (`model` field of JSON bodies and SSE chunks).
- Reasoning field: Ollama's `/v1` exposes `message.reasoning`; vLLM uses
  `reasoning_content` or `reasoning` depending on version. Emit `reasoning`,
  keeping the original field too.
- Errors stay in OpenAI format: `{"error": {"message", "type", "code"}}`.

### 5.2 `/api` — Ollama native (translated)

| Route | Handling |
|---|---|
| `GET /`, `HEAD /` | `200`, body `Ollama is running` (liveness probe; the only route besides `/healthz` that needs no key, and it returns no data) |
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
- No pod connected, tunnel down or nothing deployed → `503` with a message that
  names the state (for example `backend not deployed`), so a client-side error
  points at the management page rather than looking like a model failure.

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

- **Security**: bind to loopback only by default; the vLLM key is generated at
  deploy time, stored only in the local database and in a `0600` file on the
  pod, attached only to upstream requests, and redacted from logs and error
  messages. Proxy keys and the admin token are never logged either.
- **CORS**: same defaults as Ollama (localhost, `127.0.0.1`, `app://`,
  `vscode-webview://` etc.), configurable.
- **Concurrency**: fully async; no proxy-side queue — vLLM's continuous batching
  handles parallel requests. Shared httpx pool sized generously (e.g. 100).
- **Logging**: one line per request (method, path, model, status, token counts,
  duration, stream yes/no). Prompt content only at `debug`.
- **Health**: `GET /healthz` reports proxy status and upstream `/health` (no key
  needed, no data beyond up/down).
- **Port conflict**: a real Ollama service on the host must be stopped/disabled
  (`systemctl disable --now ollama`), otherwise `11434` is taken; the proxy exits
  with a clear message if the bind fails.

## 8. SQLite database

One file, `DATA_DIR/proxy.sqlite3`, opened in WAL mode with
`synchronous=NORMAL` and a busy timeout.

- **Writes** go through a single writer thread fed by a bounded queue, so the
  request path never waits on disk. Rows are committed in small batches. If the
  queue overflows, rows are dropped and counted (shown on the monitoring page)
  rather than slowing inference. Key management writes are synchronous.
- **Reads** (key lookup, monitoring, export) use separate read connections; key
  lookups are served from an in-memory cache that is refreshed on every key
  change.
- **Schema versioning** with `PRAGMA user_version` and ordered migration
  functions run at startup.
- **Retention**: a periodic task prunes `requests`, `engine_samples` and
  `inventory` rows older than `RETENTION_DAYS`.

| Table | Content |
|---|---|
| `api_keys` | `id`, `name` (unique), `key_prefix` (first characters, for display), `key_sha256`, `enabled`, `created_at`, `revoked_at` |
| `requests` | One row per proxied inference request (below) |
| `engine_samples` | `ts`, `name`, `labels`, `value` — `vllm:*` series from `/metrics`, `gpu:*` series, `proxy:open_requests` |
| `inventory` | `ts`, `kind`, `data` (JSON) — GPU identities and limits, driver/CUDA, vLLM version and cache config, vLLM startup lines; a row is written only when the content changes |
| `pod` | The current pod connection: SSH host, port, user, key path, pinned host-key fingerprint, last connection state |
| `profiles` | Deployment profiles (section 12.3) |
| `deployments` | One row per Deploy click: profile snapshot, start/end time, final state, and per-step status with timestamps |
| `secrets` | The generated vLLM API key and optional Hugging Face token (see below) |
| `state` | Key/value settings such as the admin token hash and counters for rejected requests |

The vLLM key and the Hugging Face token must be usable in clear text, so they
cannot be hashed like API keys. The database file and `DATA_DIR` are created
with mode `0600`/`0700`, these two values are never returned by any endpoint
and never logged, and the SSH private key is not stored at all — only its path.

`requests` columns:

- **Identity**: `key_id`, `api` (`ollama` or `openai`), `endpoint`, `model` (as
  sent by the client), `stream`, optional `session_id` from an `X-Session-Id`
  header (validated against `^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$`, else null).
- **Outcome**: `ok`, `client_cancelled`, `upstream_error` or `proxy_error`; HTTP
  status; `finish_reason`; message count.
- **Timings** in ms from arrival on the proxy clock: first byte, first content,
  first visible text, first reasoning, first tool-call fragment, end, longest
  gap between content events; requests open at the proxy on arrival.
- **Tokens**: prompt, completion, cached, reasoning (from the final usage chunk).
- **Derived**: decode tok/s, delivered tok/s, context fraction
  (`prompt + completion` over `CONTEXT_LENGTH`), cached fraction.

Never stored: prompts, completions, reasoning text, tool arguments, raw keys.
Requests rejected for a missing or invalid key do not create rows; they only
increment a counter, so a misconfigured client cannot fill the database.

## 9. API keys

- **Format**: `sk-ovp-` followed by 32 random base62 characters from
  `secrets`. The full key is shown once, at creation. The database holds only
  its SHA-256 hash and a short prefix for recognising it in lists.
- **Where clients send it**: `Authorization: Bearer <key>`. `x-api-key: <key>`
  is accepted as well, for `/v1/messages`-style clients.
- **Enforcement**: one middleware in front of every `/api/*` and `/v1/*` route,
  including discovery endpoints such as `/api/tags` and `/v1/models`. Only
  `GET|HEAD /`, `/healthz` and the static admin pages (which contain no data)
  are reachable without a credential.
- **Failure responses**: `401` with `{"error": "..."}` under `/api` and with the
  OpenAI error object under `/v1`, plus `WWW-Authenticate: Bearer`. Missing,
  unknown, disabled and revoked keys give the same message.
- **Effect of changes**: disabling or revoking a key applies to new requests
  immediately; in-flight streams are allowed to finish.
- **No keys yet**: every inference call is rejected until the first key is
  created; the startup log says so and points at the management page.
- The key identifies the caller everywhere else: it is the grouping dimension
  ("actor") in the monitoring page and the report.

## 10. Management interface

Served by the proxy itself under `/admin`, on the same loopback listener.

**Admin authentication**: a single admin token. At first start the proxy
generates one, writes it to `DATA_DIR/admin_token.txt` with mode `0600` and
logs the file path (not the token). `ADMIN_TOKEN` overrides it. The pages open
as `http://127.0.0.1:11434/admin#<token>` or by pasting the token; a token in
the URL fragment never reaches the server, is removed from the address bar and
is kept only in that tab's `sessionStorage`. The JSON endpoints require it as
`Authorization: Bearer <admin token>`. The admin token is not valid as an API
key and API keys are not valid on `/admin/*`.

**Pages**

| Page | Purpose |
|---|---|
| `GET /admin` | API keys: list (name, prefix, created, status), create, rename, enable/disable, revoke. After creation the key is displayed once with a copy button and a ready-to-paste client snippet |
| `GET /admin/pod` | Pod connection, deployment profiles, Deploy / Stop / Restart, progress and logs (section 12) |
| `GET /admin/monitor` | Live monitoring page (section 11) |

**Admin API**

| Endpoint | Purpose |
|---|---|
| `GET /admin/api/keys` | List keys (never returns the key or its hash) |
| `POST /admin/api/keys` `{"name": "..."}` | Create; the response is the only place the full key appears |
| `PATCH /admin/api/keys/{id}` `{"name"?, "enabled"?}` | Rename, enable or disable |
| `DELETE /admin/api/keys/{id}` | Revoke permanently (row is kept so historical requests keep their key name) |
| `GET /admin/api/status` | Upstream health, model, writer queue depth, dropped rows, rejected-request counters |
| `GET /admin/api/live?window=` | Snapshot behind the monitoring page |
| `GET /admin/api/export?from=&to=&key=&session=&format=html\|json&tz_offset=` | Report download (section 11.2) |

**CLI** (works directly on the SQLite file, for bootstrap and recovery):
`ollama-vllm-proxy keys create|list|disable|enable|revoke` and
`ollama-vllm-proxy admin-token show|rotate`.

**Page hardening**: strict CSP (`default-src 'none'`, scripts and styles from
the page only, `connect-src 'self'`), `Referrer-Policy: no-referrer`,
`X-Frame-Options: DENY`, no cookies (so no CSRF surface), and all values
rendered as text, never as HTML.

## 11. Monitoring

Modelled on the dashboard and session report of
`confidential-inference-test` (`perfproxy/live.py`, `export.py`,
`static/dashboard.html`), with the test-harness concepts removed: no runs,
slots, worker events or marks. The "actor" is the API key, and a report is
scoped by time range instead of by run.

### 11.1 Data collection

- **Request observer**: wraps every proxied inference call on both `/v1` and
  `/api`, watches the upstream SSE stream to timestamp first byte, first
  content, first text, first reasoning, first tool fragment and end, reads the
  final usage chunk and writes one `requests` row. `stream_options.include_usage`
  is forced upstream; if a `/v1` client did not ask for usage, the usage-only
  chunk is dropped so the client sees the stream it expected.
- **Engine sampler**: every `SAMPLE_INTERVAL_S` scrapes vLLM `/metrics` and
  keeps the `vllm:*` series (running and waiting requests, KV-cache usage,
  prompt and generation token counters, prefix-cache hits and queries,
  preemptions) plus `proxy:open_requests`. Histogram `_bucket` series are
  dropped, and the `model_name` label is stripped because it contains the
  weights path.
- **GPU helper on the pod**: `gpu_exporter.py`, vendored from
  `confidential-inference-test/pod/` (standard library only), is uploaded and
  started by Deploy (section 12) and listens on `127.0.0.1:9400` inside the
  pod. It runs `nvidia-smi --query-gpu ... -lms` and `nvidia-smi dmon -s t` as
  long-lived child processes and serves `/metrics` (per-GPU utilization,
  memory used/total, power and power limit, temperature, SM clock, PCIe RX/TX)
  and `/inventory`. The proxy scrapes it through the tunnel on every sample and
  stores the values as `gpu:*` series. If the helper is not running, the GPU
  sections are omitted and everything else keeps working.
- **Inventory**: from the helper — GPU names, memory, ECC, power limit, PCIe
  max, max SM clock, bus ID, VBIOS, driver and CUDA versions,
  `nvidia-smi topo -m`, host CPU/RAM and the vLLM startup lines parsed from
  `/workspace/vllm.log` with key-like arguments redacted; from vLLM —
  `/version` and `vllm:cache_config_info`. Fetched every 30 s.
- A small in-memory ring (about 25 minutes of samples, last 2,000 requests)
  serves the live page cheaply; longer windows are read from SQLite.

### 11.2 Live page and report

The live page (`/admin/monitor`) polls `/admin/api/live` once per sample
interval, with a window selector from 2 minutes up to the retention period.
The report is the same content for a chosen time range, optionally filtered by
key or session, saved as one self-contained file.

| Section | Live page | Report |
|---|---|---|
| Summary tiles: open requests, engine running/waiting, KV cache % and preemptions, output and prompt tok/s, completed requests, errors and writer drops | yes | yes (totals for the range) |
| In-flight requests with current phase (queued/prefill, reasoning, answering, tool call) | yes | — |
| Hardware and vLLM: GPU table, driver, vLLM version and cache config, startup lines, `nvidia-smi topo -m` | yes | yes |
| Engine charts: concurrency, KV cache used, output tok/s, prompt tok/s, prefix cache hit rate | yes | yes |
| GPU charts: utilization, memory used, power, temperature, SM clock, PCIe traffic, plus a per-GPU summary table (util mean/p90, memory peak, power mean/max, temp max, SM clock min, PCIe mean/max) | yes | yes |
| Time to first content per request (log-scale scatter, hollow dots for requests that did not complete OK) | yes | yes |
| Per key: requests, OK, cancelled, errors, first-content p50/p90, text p50, tool p50, end p50/p90, gap p90, decode and delivered tok/s p50, cached share, output tokens, context max | yes | yes |
| Requests table: time, key, session, API, outcome, finish, messages, prompt/cached/output/reasoning tokens, open at start, first content/text/tool, end, max gap, decode and delivered tok/s, context | last 30 | all in range |

- **Charts** are inline SVG generated without any charting library. On the live
  page they have a crosshair tooltip, keyboard navigation and a "Show data"
  table. Light and dark themes follow the system setting.
- **Report file**: no scripts, no external requests, opens offline and prints
  cleanly; the full data set is embedded as JSON (`#report-data`), and
  `format=json` returns only that data. Times are rendered in the viewer's time
  zone via `tz_offset`. A **Download report** button on the live page opens a
  picker for time range, key and session.
- **CLI**: `ollama-vllm-proxy export --from ... --to ... [--key NAME] [--session ID] -o report.html`.
- Engine and GPU charts are engine-wide: a report filtered to one key still
  shows the load caused by all keys.

## 12. Pod deployment (Runpod)

### 12.1 What you do, and what the application does

You:

1. Start a pod in the Runpod console with these settings:
   - Template **Ubuntu 24.04**, image `runpod/base:2.1.0-cuda1320-ubuntu2404`.
   - **Container disk**: model size + 25 GB (about 9 GB for the vLLM
     environment, the rest for caches, logs and headroom). For the default
     profile use **80 GB**; the Pod page shows the figure for each profile.
     No network volume.
   - **Expose only TCP port 22.** Remove the HTTP ports: the image also runs
     JupyterLab, nginx and a web terminal, which are not needed and should not
     be published.
   - Your public SSH key on the Runpod account.
2. Copy the address from **Connect → SSH over exposed TCP** into the **Pod**
   page: host, port, user (`root`) and the private key path.
3. Click **Connect**, confirm the host-key fingerprint, pick a profile and
   click **Deploy**.
4. Terminate the pod in the Runpod console when you are done. The application
   cannot do this, and the page says so whenever a pod is connected: stopping
   vLLM does not stop the billing.

The application does everything else: SSH connection, tunnel, vLLM
installation, model download, helper installation, starting vLLM, readiness
detection and log display.

### 12.2 Pod baseline (measured on a pod from this template)

| Finding | Consequence for Deploy |
|---|---|
| Ubuntu 24.04.5, user `root`, no systemd (PID 1 is `docker-init` running `/start.sh`) | Processes are started detached with `setsid nohup` and tracked by PID files; no service units |
| `sshd` running on port 22, `authorized_keys` provisioned by Runpod | Key-based SSH works without any setup |
| `uv`, Python 3.12.3 (also 3.11, 3.13), `venv`, `pip`, `git`, `curl`, `gcc` present | No `apt` step needed; the install uses the `uv` already in the image |
| `vllm`, `hf` not installed | Deploy installs both into a dedicated virtual environment |
| Measured install: `uv venv --python 3.12` + `uv pip install vllm huggingface_hub` takes 1 min 41 s and gives vLLM 0.34.0 with PyTorch 2.14.0+cu130; `torch.cuda.is_available()` is true | The install step is short next to the download; no custom PyTorch index or CUDA setup is needed |
| The environment is 8.9 GB, but the disk grew by 18 GB because `uv` keeps a second copy in its cache | The install runs with `--no-cache`, so the environment costs about 9 GB, not 18 |
| vLLM 0.34.0 lists `qwen3_coder` among its tool-call parsers | The default profile's tool parser is valid for the pinned version |
| Driver 595.112.07, CUDA 13.2 libraries and cuDNN 9 in the image | New enough for current PyTorch wheels, which bundle their own CUDA runtime |
| `/workspace` is a plain directory on the 30 GB container overlay, not a volume | Disk size is whatever you set as container disk; nothing persists. Paths stay under `/workspace` for consistency |
| Container memory limit is 31 GB although `free` reports 125 GB | Preflight reads the cgroup limit (`/sys/fs/cgroup/memory.max`), not `free` |
| `/dev/shm` is 14 GB | Enough for vLLM; preflight warns below 1 GB |
| Open-file limit is 1024 | The vLLM launcher raises it (`ulimit -n 65535`) before starting |
| Ports 8000 and 9400 are free; 22, 3001, 7270, 7861, 8001, 8081, 8888, 9091, 19123 are taken by image services | vLLM and the helper keep their planned ports, bound to `127.0.0.1` |
| `huggingface.co`, `pypi.org`, `astral.sh` reachable | Install and download need no mirror |
| Environment contains `RUNPOD_API_KEY` and `JUPYTER_PASSWORD` | Everything Deploy starts runs with a clean, whitelisted environment, so these never reach vLLM, its log or the helper |
| Environment sets `HF_HUB_ENABLE_HF_TRANSFER` and `HF_XET_HIGH_PERFORMANCE` | The download step installs the matching accelerator packages and sets its own Hugging Face variables explicitly instead of inheriting them |
| `RUNPOD_POD_ID`, `RUNPOD_DC_ID`, `RUNPOD_GPU_NAME`, `RUNPOD_GPU_COUNT`, `RUNPOD_MEM_GB`, `RUNPOD_CPU_COUNT` available | Recorded (values of these six only) in the inventory and shown in the report header |
| All `nvidia-smi --query-gpu` fields used by the helper and `nvidia-smi dmon -s t` work | The GPU helper from confidential-inference-test runs unchanged |
| Target hardware is one RTX PRO 6000; the first probe ran on an RTX PRO 4000 Blackwell with 24 GB | Profiles are written for a single GPU (no `--tensor-parallel-size`). Preflight still compares GPU memory with the model size and stops before downloading, so a pod rented with the wrong GPU fails fast |

### 12.3 Deployment profiles

Stored in SQLite and edited in the UI. A profile is structured, not a free-form
command, so the proxy always knows the model name and context length.

| Field | Default profile (`qwen3.8-27b-fp8`) |
|---|---|
| Hugging Face repo, optional revision | `Qwen/Qwen3.8-27B-FP8` |
| Local directory | `/workspace/qwen3.8-27b-fp8` |
| Served model name | `dev-test` |
| Max model length | `65536` |
| Extra vLLM arguments (ordered flag/value list) | `--kv-cache-dtype fp8`, `--enable-prefix-caching`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3` |
| vLLM version | `0.34.0` (the version verified on this image; editable per profile) |
| Python version for the environment | `3.12` |
| Extra environment variables | none |
| Needs Hugging Face token | no |

`--host 127.0.0.1`, `--port 8000`, `--served-model-name`, `--max-model-len` and
the API key are always set by the application and cannot be overridden through
the extra arguments, so a profile can never expose vLLM on a public interface
or run it without a key. Profiles can be created, duplicated, edited and
deleted; the page shows the exact command a profile will run before you deploy
it. Each deployment stores a snapshot of the profile it used.

### 12.4 SSH connection and tunnel

- One asyncssh connection per pod, key-based only, with keepalives
  (15 s interval, 3 misses) and automatic reconnect with backoff. A
  passphrase-protected key is used through `ssh-agent`; the application never
  asks for or stores a passphrase.
- **Host key**: on first connect the page shows the pod's host-key fingerprint
  and requires confirmation; it is then pinned for that host and port. A changed
  key blocks the connection with an explicit warning until re-confirmed. Host
  keys are never accepted silently.
- **Port forwards** inside the same connection: a local ephemeral loopback
  port → pod `127.0.0.1:8000` (vLLM) and another → pod `127.0.0.1:9400` (GPU
  helper). The proxy's upstream URL is derived from these, so nothing listens
  on local port 8000 and your manual `ssh -N -L` command is no longer needed.
- A dropped connection does not affect what runs on the pod. The proxy returns
  `503` until the tunnel is back, then resumes without a redeploy.

### 12.5 What Deploy does

Deploy is a sequence of steps. Each has a status (pending, running, done,
skipped, failed), timestamps and its own output, all visible on the page and
recorded in `deployments`.

| # | Step | Detail |
|---|---|---|
| 1 | Connect | Open or reuse the SSH connection, verify the pinned host key |
| 2 | Preflight | GPUs and total GPU memory against the model size; free container disk against model size + environment; container memory limit; `/dev/shm`; `uv` and the profile's Python version present; ports 8000 and 9400 free or held by our own processes; `huggingface.co` and `pypi.org` reachable. Problems stop the deployment here with a specific message, before anything is downloaded |
| 3 | Install helper | Upload `gpu_exporter.py` and the launcher scripts by SFTP to `/workspace/ovp/`, start the helper on `127.0.0.1:9400` with the system `python3`, check its `/healthz` through the tunnel. Done first so GPU and disk activity are visible during the rest |
| 4 | Install vLLM | `uv venv /workspace/venv --python <version>`, then `uv pip install --no-cache` of `vllm==<profile version>`, `huggingface_hub` and the download accelerators into it (about 1 min 40 s, 9 GB). Verified with `vllm --version`; the installed versions of vLLM, PyTorch and transformers go to the inventory. Skipped when the environment already has the requested version |
| 5 | Download model | `hf download <repo> --local-dir <dir>` from the environment; resumable, and near-instant when the files are already present. Progress is the downloaded size against the repo size |
| 6 | Write secrets | Generate a fresh vLLM API key locally, write it (and the Hugging Face token if the profile needs one) to `/workspace/ovp/vllm.env`, mode `0600` |
| 7 | Start vLLM | `vllm serve <dir> ...` from the profile, output to `/workspace/vllm.log` |
| 8 | Wait until ready | Poll `/health` through the tunnel while streaming the log; fail at once, showing the last log lines, if the process exits |
| 9 | Activate | Read `/v1/models`, check the served name and max model length against the profile, switch the proxy to this backend |

Steps 4 and 5 run in parallel, since both are network-bound and independent;
step 6 waits for both. On a fresh pod the whole sequence is dominated by the
model download and the model load.

Implementation rules:

- **Long-running remote processes are detached** (`setsid nohup`, PID file and
  log file under `/workspace/ovp/`), so the download and vLLM survive an SSH
  drop or a proxy restart. On reconnect or restart the application reads the
  PID files and logs and resumes the deployment from the step it was in rather
  than starting over.
- **No secret on a command line**: the vLLM key reaches the process through the
  environment file (`VLLM_API_KEY`), not `--api-key`, so it does not appear in
  `ps` output or in the log's argument dump.
- **Clean environment**: the launchers start every process with `env -i` and
  an explicit list (`PATH`, `HOME`, `LANG`, `LD_LIBRARY_PATH`, the `NVIDIA_*`
  and `CUDA_*` variables, `HF_HOME`, plus the profile's own variables), so the
  pod's `RUNPOD_API_KEY` and `JUPYTER_PASSWORD` are not inherited.
- **No shell injection**: remote commands are built from argument lists with
  `shlex.quote`; profile values are never interpolated into shell text. Repo,
  path and model-name fields are validated against strict patterns.
- **Idempotent**: clicking Deploy again with the same profile skips finished
  work (helper running, environment installed, model present) and restarts vLLM only if it is not
  healthy. Deploying a different profile stops the running vLLM first, after a
  confirmation that states in-flight requests will be cut.
- **One deployment at a time**; the button is disabled while one is running,
  and a **Cancel** button stops the current step and kills the remote process
  it started.

Other actions on the page: **Stop vLLM**, **Restart vLLM** (same profile, no
download), **Disconnect** (closes SSH, leaves the pod running) and
**Delete model files** (frees `/workspace` space; asks for confirmation).

### 12.6 Progress and logs

- A state line for the backend: `no pod` → `connected` → `deploying (step)` →
  `ready`, or `failed (step)`, `tunnel down`, `stopped`. The same state drives
  the `503` messages of the inference endpoints and a status tile on the
  monitoring page.
- The step list with live status, and a progress bar for the download.
- A log panel with a source switch: **vLLM** (`/workspace/vllm.log`),
  **install**, **download**, **GPU helper**. It loads the last 500 lines and then follows the
  file (`tail -F` over an SSH channel, relayed to the browser as server-sent
  events), with pause, auto-scroll and a download-full-log link. Key-like
  values are redacted before display; logs are relayed, not stored in SQLite.

**Admin API additions** (admin token required)

| Endpoint | Purpose |
|---|---|
| `GET/PUT /admin/api/pod` | Read or set SSH host, port, user, key path |
| `POST /admin/api/pod/connect` `{"accept_fingerprint"?}` | Connect; returns the fingerprint to confirm on first use |
| `POST /admin/api/pod/disconnect` | Close the SSH connection |
| `GET/POST /admin/api/profiles`, `PUT/DELETE /admin/api/profiles/{id}` | Profile CRUD |
| `POST /admin/api/deploy` `{"profile_id"}` | Start a deployment |
| `POST /admin/api/deploy/cancel` | Cancel the running deployment |
| `POST /admin/api/vllm/stop`, `POST /admin/api/vllm/restart` | Stop or restart vLLM |
| `GET /admin/api/deploy/status` | Backend state, current deployment and step statuses |
| `GET /admin/api/deploy/events` (SSE) | Live state and step updates |
| `GET /admin/api/logs/stream?source=vllm\|install\|download\|helper` (SSE) | Live log tail |

## 13. Project layout

```
pyproject.toml              # uv project, entry point `ollama-vllm-proxy`
README.md
src/ollama_vllm_proxy/
  __main__.py               # uvicorn launcher and CLI (serve, keys, admin-token, export)
  app.py                    # FastAPI app, lifespan (httpx client, store, samplers), CORS, error handlers
  store.py                  # SQLite: schema, migrations, writer thread, read helpers, retention
  auth.py                   # API-key middleware, key cache, admin-token check
  keys.py                   # key generation, hashing, CRUD
  observe.py                # per-request record: timings, usage, outcome
  sampler.py                # vLLM /metrics scraper, inventory collection
  pod/
    ssh.py                  # asyncssh connection, host-key pinning, reconnect, port forwards
    profiles.py             # profile model, validation, command builder
    deploy.py               # deployment state machine, steps, resume
    logs.py                 # remote tail → SSE, redaction
    remote/                 # files uploaded to the pod
      gpu_exporter.py       # vendored from confidential-inference-test/pod
      start_vllm.sh         # detached launcher: env file, PID file, log file
      install_vllm.sh       # uv venv + uv pip install, version check
      start_download.sh
  live.py                   # in-memory ring and live snapshot
  export.py                 # report data set, SVG charts, self-contained HTML
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
    admin.py                # /admin pages and /admin/api/* (keys, status, live, export)
    pod.py                  # /admin/api/pod, profiles, deploy, vllm, logs
  static/
    admin.html              # key management page
    pod.html                # pod connection, profiles, deploy, progress, logs
    monitor.html            # live monitoring page
  translate/
    request.py              # Ollama → OpenAI (messages, tools, options, format, think)
    response.py             # OpenAI → Ollama (non-streaming)
    stream.py               # SSE parser, tool-call accumulator, NDJSON emitter, timings
tests/
  unit/                     # translation tables, tool-call accumulation, name resolution,
                            # key hashing, metrics and nvidia-smi parsers, percentile maths,
                            # profile validation and command building
  integration/              # app + respx-mocked vLLM, streaming and error paths
  e2e/                      # opt-in, against a live vLLM
deploy/ollama-vllm-proxy.service   # systemd user unit for Linux; macOS runs it with `uv run`
```

## 14. Milestones

1. **Skeleton** — project setup, config, lifespan-managed httpx client, `/`,
   `/api/version`, `/healthz`.
2. **SQLite store** — schema, migrations, writer thread, retention task.
3. **API keys** — key CRUD, CLI, auth middleware on `/api` and `/v1`, admin
   token bootstrap. *Check*: a call without a key gets 401; a key created with
   the CLI is accepted; a revoked key is rejected.
4. **Pod connection** — pod form, asyncssh connection with host-key pinning,
   reconnect, port forwards, backend state. *Check*: after Connect, vLLM started
   by hand on the pod is reachable through the proxy's own tunnel.
5. **Profiles and Deploy** — profile CRUD, command builder, deployment steps
   with detached remote processes and resume, vLLM and helper installation,
   Stop and Restart. *Check*: on a fresh pod, one click ends in state `ready`; killing
   the proxy mid-download and restarting it resumes the same deployment.
6. **Pod page** — step list, download progress, live log panel.
   *Check*: the vLLM log scrolls live in the browser during model load.
7. **`/v1` pass-through** — streaming reverse proxy, auth swap, model rewrite,
   `503` states when no backend is ready.
   *Check*: OpenAI SDK with `base_url=http://127.0.0.1:11434/v1` and a proxy key
   streams a chat completion.
8. **Model discovery** — `/api/tags`, `/api/ps`, `/api/show`, name resolution.
   *Check*: `ollama-python` `list()` and `show()` work with the key in `headers`.
9. **`/api/chat`** — non-streaming, then streaming, options mapping.
   *Check*: `ollama-python` streams a chat.
10. **Thinking + tools + structured output** — `think`, `thinking` deltas,
   tool-call accumulation, tool result round-trip, `format`.
   *Check*: `ollama-python` tool-calling example completes a full round trip.
11. **`/api/generate`** — chat-wrapped and raw modes, load/unload shortcuts.
12. **Stubs and errors** — pull/push/create/copy/delete, embeddings, upstream
   failure mapping, mid-stream errors, client-disconnect cancellation.
13. **Management page** — `/admin` key management UI on top of the admin API.
    *Check*: create a key in the browser, use it, revoke it, see it rejected.
14. **Request observer and samplers** — `requests` rows for both dialects,
    vLLM `/metrics` scraping, GPU helper scraping through the tunnel, inventory.
    *Check*: rows and samples appear in SQLite with no prompt content.
15. **Live monitoring page** — `/admin/api/live` and `/admin/monitor`.
16. **Report export** — HTML and JSON export, CLI, download picker.
    *Check*: the file opens offline with all sections of the reference report
    that apply.
17. **Packaging** — README, systemd unit, CI (ruff, pytest).

## 15. Testing

- **Unit**: every row of the mapping tables in section 6; fragmented tool-call
  streams (single, multiple, interleaved with content and reasoning); invalid
  argument JSON; name resolution with and without `STRICT_MODELS`.
- **Integration** (mocked vLLM): golden request/response pairs per endpoint;
  streaming framing (each line valid JSON, exactly one `done:true`, last);
  upstream 400/401/500/timeouts; disconnect propagates cancellation; the vLLM
  key never appears in any response or log line.
- **Auth**: every `/api` and `/v1` route rejects missing, unknown, disabled and
  revoked keys with the dialect's error shape; the admin token does not work as
  an API key and vice versa; the full key is returned only by the create call
  and is absent from the database, logs and list responses.
- **Store**: migrations from an empty file, writer-queue overflow counting,
  retention pruning, concurrent reads during writes.
- **Monitoring**: observer timings against a scripted fake upstream (known
  delays for first byte, reasoning, text, tool call); cancelled and failed
  requests recorded with the right outcome; `/metrics` and `nvidia-smi` parser
  fixtures; export of a seeded database compared with a golden data set; the
  HTML report contains no `<script>` and no external URL; no prompt text
  anywhere in SQLite after a full test run.
- **Pod deployment**: integration tests against a local throw-away SSH server
  (an sshd container with a stub `uv`, `vllm`, `hf` and `nvidia-smi` on the path):
  host-key confirmation and mismatch, every step's success and failure path,
  resume after a dropped connection and after a proxy restart, cancel, Stop and
  Restart, redeploy idempotence, profile switch. Assertions that the vLLM key
  never appears in a remote command line, a log relayed to the browser, an API
  response or a local log, and that hostile profile values (quotes, `;`, `$()`)
  are rejected or passed as inert arguments.
- **Admin pages**: Playwright smoke test — create, use, disable and revoke a key,
  open the monitoring page, download a report, run a deployment against the
  stub SSH server and watch the log panel.
- **Client compatibility** (live vLLM): `ollama-python` (chat, generate, stream,
  tools, think) and the OpenAI SDK (chat, stream, tools), both with a proxy
  key, and Open WebUI pointed at the proxy.
- **Real pod** (manual, before release): one full deployment of the default
  profile on a Runpod pod, from a fresh pod to a streamed chat completion.
- **Prefix cache check**: send the same multi-turn conversation twice and
  confirm from vLLM metrics that the second request hits the prefix cache.

## 16. Open points to verify during implementation

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
8. **vLLM `/metrics` names** — metric and label names differ between vLLM
   versions; pin the series used by the charts against the installed build.
9. **`/metrics` and `/version` authentication** — whether the installed vLLM
   requires the API key on these routes (the proxy sends it either way).
10. **Reasoning parser name** — vLLM 0.34.0 does not list the accepted values
    of `--reasoning-parser` in its help, so `qwen3` is still unconfirmed for
    this version. It is checked the first time vLLM starts; an invalid name
    makes vLLM exit at once, which Deploy reports with the log lines.
11. **Model size and disk figure** — the 80 GB container-disk recommendation
    assumes about 30 GB of weights for `Qwen/Qwen3.8-27B-FP8`; replace it with
    the size reported by the Hugging Face API, which preflight uses anyway.
12. **First start on the RTX PRO 6000** — confirm that the default profile
    loads and serves with vLLM 0.34.0 / PyTorch 2.14.0+cu130 on driver 595,
    and note the load time for the readiness timeout.
13. **Download accelerator** — which of `hf_transfer` / `hf_xet` the installed
    `huggingface_hub` actually uses, and whether a missing package makes
    `hf download` fail when the image's variables are set.
14. **Host key stability** — whether the pod's SSH host key changes when a pod
    is stopped and resumed, which decides how often the fingerprint prompt
    appears.
15. **`VLLM_API_KEY` from the environment** — confirm the installed vLLM reads
    the key from the environment variable; otherwise fall back to a config
    file passed with `--config`, still keeping it off the command line.
16. **Hugging Face token** — whether any profile you plan to use needs a gated
    repo; the default one does not.
