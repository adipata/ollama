# Ollama-compatible proxy for vLLM

A local service on `http://127.0.0.1:11434` that looks like Ollama to clients and
forwards inference to vLLM running on a Runpod pod. It exposes Ollama's native
API under `/api` and the OpenAI-compatible API under `/v1`, requires an API key
on every call, deploys vLLM to the pod from its management pages, and records
request and GPU metrics for a live monitoring page and a downloadable report.

The design is in [PLAN.md](PLAN.md).

## Install and run

Linux or macOS, Python 3.11+, [uv](https://docs.astral.sh/uv/).

```bash
uv venv && uv pip install -e .
.venv/bin/ollama-vllm-proxy            # serves on 127.0.0.1:11434
```

Stop a local Ollama first if one is running (`systemctl disable --now ollama`);
the proxy exits with a clear message when the port is taken.

On first start the proxy writes an admin token to `data/admin_token.txt`. Open
the management pages with it:

```
http://127.0.0.1:11434/admin#<contents of data/admin_token.txt>
```

The token in the URL fragment never reaches the server; the page removes it
from the address bar and keeps it only in that browser tab.

| Page | Purpose |
|---|---|
| `/admin` | Create, rename, disable and revoke API keys |
| `/admin/pod` | Connect to the pod, choose a deployment profile, deploy vLLM, watch the log |
| `/admin/monitor` | Live monitoring and report download |

## Deploying vLLM on a Runpod pod

1. In the Runpod console start a pod from the **Ubuntu 24.04** template
   (`runpod/base:2.1.0-cuda1320-ubuntu2404`), with a container disk of the
   model size + 25 GB (100 GB covers both variants of the default profile),
   no network volume, and only TCP port 22 exposed.
2. On `/admin/pod`, paste the command Runpod shows under
   **Connect → SSH over exposed TCP**, for example
   `ssh root@213.173.99.7 -p 10909 -i ~/.ssh/id_ed25519`, click **Connect** and
   confirm the host-key fingerprint.
3. Pick a profile. The page shows which model variant, GPU split and context
   length fit the pod's GPUs. Click **Deploy** and follow the steps and the log.
4. Terminate the pod in the Runpod console when you are done. Stopping vLLM
   does not stop the billing, and the proxy cannot terminate the pod.

The proxy keeps its own SSH tunnel to the pod, so no `ssh -L` command is
needed. Each deployment generates a fresh vLLM API key that only the proxy
knows. A Hugging Face token for gated models can be saved on the same page.

## Using it from clients

Every `/api` and `/v1` call needs `Authorization: Bearer <key>` (or
`x-api-key: <key>`). Create a key on `/admin`, or from the command line:

```bash
.venv/bin/ollama-vllm-proxy keys create my-laptop
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="sk-ovp-...")

from ollama import Client
client = Client(host="http://127.0.0.1:11434", headers={"Authorization": "Bearer sk-ovp-..."})
```

The stock `ollama` command-line tool cannot send a key to a local server, so it
does not work against the proxy.

What is supported:

| Route | Behaviour |
|---|---|
| `POST /api/chat`, `POST /api/generate` | Translated to vLLM, streaming (NDJSON) or not; tools, `think`, `format`, images, the common `options` |
| `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `GET /api/version` | Built from the model vLLM serves |
| `POST /api/pull` | Reports success for the served model; nothing is downloaded |
| `POST /api/embed`, push, create, copy, delete | `501` — not applicable to a fixed vLLM server |
| `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` | Passed through, recorded for monitoring |
| any other `/v1/*` | Passed through unchanged |

Model names: `:latest` is stripped and `MODEL_ALIASES` are applied. A name that
still does not match falls back to the served model unless `STRICT_MODELS=true`.

Send an `X-Session-Id` header to group requests into a session for reports.

## Configuration

Environment variables, or a `.env` file in the working directory.

| Variable | Default | Purpose |
|---|---|---|
| `PROXY_HOST` / `PROXY_PORT` | `127.0.0.1` / `11434` | Listen address |
| `DATA_DIR` | `./data` | SQLite database and `admin_token.txt` |
| `ADMIN_TOKEN` | generated | Overrides the token file |
| `SSH_KEY_PATH` | `~/.ssh/id_ed25519` | Key used when the pasted pod command has no `-i` |
| `MODEL_ALIASES` | empty | `alias=served-name,...` |
| `STRICT_MODELS` | `false` | Reject unknown model names with 404 |
| `OLLAMA_VERSION` | `0.12.0` | Value returned by `/api/version` |
| `ALLOWED_ORIGINS` | localhost origins | Extra CORS origins, comma separated |
| `SAMPLE_INTERVAL_S` | `1.5` | Metrics sampling interval |
| `RETENTION_DAYS` | `30` | Age at which request rows and samples are pruned (`0` keeps all) |
| `LOG_LEVEL` | `info` | |
| `VLLM_BASE_URL`, `VLLM_API_KEY` | empty | *External mode*: use an already running vLLM and disable pod management |
| `GPU_EXPORTER_URL` | empty | External mode only: URL of a GPU helper |
| `POD_REMOTE_ROOT` | `/workspace` | Directory on the pod for the environment, models and logs |
| `POD_VLLM_PORT` / `POD_HELPER_PORT` | `8000` / `9400` | Ports used inside the pod (loopback only) |
| `POD_READY_TIMEOUT_S` | `1800` | How long Deploy waits for vLLM to load the model |
| `POD_HF_ENDPOINT` | `https://huggingface.co` | Hugging Face API used for model sizes |

The database holds the vLLM API key and the Hugging Face token in clear text
(they have to be usable), so `data/` is created with owner-only permissions.
Prompts, completions and tool arguments are never stored.

## Command line

```bash
ollama-vllm-proxy serve                              # default
ollama-vllm-proxy keys create|list|disable|enable|revoke [name]
ollama-vllm-proxy admin-token show|rotate
ollama-vllm-proxy export --last 2h -o report.html    # session report
```

## Development

```bash
uv pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python tests/fake_vllm.py --port 8000      # a stand-in for vLLM
VLLM_BASE_URL=http://127.0.0.1:8000 VLLM_API_KEY=test-key .venv/bin/ollama-vllm-proxy
```

A systemd user unit is in `deploy/ollama-vllm-proxy.service`.
