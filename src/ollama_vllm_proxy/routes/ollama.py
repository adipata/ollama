"""/api - Ollama's native API, translated to vLLM's OpenAI API."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from ..backend import BackendUnavailable
from ..forward import ProxyCall
from ..translate.request import BadRequest, chat_payload, generate_payload
from ..translate.response import OllamaAdapter, load_response
from .common import get_ctx, json_body, new_record, served_model

CAPABILITIES = ["completion", "tools", "thinking"]


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


async def root(request: Request) -> Response:
    return PlainTextResponse("Ollama is running")


async def healthz(request: Request) -> Response:
    ctx = get_ctx(request)
    ok = ctx.backend.ready
    return JSONResponse({"proxy": "ok", "backend": "ready" if ok else "not_ready"}, status_code=200)


async def version(request: Request) -> Response:
    return JSONResponse({"version": get_ctx(request).settings.ollama_version})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _is_unload(body: dict[str, Any]) -> bool:
    keep_alive = body.get("keep_alive")
    return keep_alive in (0, "0", "0s", "0m") and not isinstance(keep_alive, bool)


async def chat(request: Request) -> Response:
    ctx = get_ctx(request)
    body = await json_body(request)
    client_model = body.get("model") if isinstance(body.get("model"), str) else ""
    if not client_model:
        raise BadRequest("model is required")
    served = await served_model(ctx, client_model)
    if not body.get("messages"):
        # Ollama clients send an empty chat to preload (or unload) a model.
        return JSONResponse(load_response("chat", client_model, "unload" if _is_unload(body) else "load"))
    stream = body.get("stream") is not False
    payload = chat_payload(body, served, stream)
    rec = new_record(request, ctx, api="ollama", endpoint="chat", body=body, stream=stream)
    rec.requested_max_tokens = payload.get("max_tokens")
    return ProxyCall(
        ctx, rec, path="/v1/chat/completions", body=payload, stream=stream,
        adapter=OllamaAdapter("chat", client_model),
    )


async def generate(request: Request) -> Response:
    ctx = get_ctx(request)
    body = await json_body(request)
    client_model = body.get("model") if isinstance(body.get("model"), str) else ""
    if not client_model:
        raise BadRequest("model is required")
    served = await served_model(ctx, client_model)
    if not body.get("prompt") and not body.get("images"):
        return JSONResponse(load_response("generate", client_model, "unload" if _is_unload(body) else "load"))
    stream = body.get("stream") is not False
    path, payload = generate_payload(body, served, stream)
    rec = new_record(request, ctx, api="ollama", endpoint="generate", body=body, stream=stream)
    rec.requested_max_tokens = payload.get("max_tokens")
    rec.messages_count = len(payload["messages"]) if "messages" in payload else None
    return ProxyCall(ctx, rec, path=path, body=payload, stream=stream, adapter=OllamaAdapter("generate", client_model))


async def embed(request: Request) -> Response:
    return JSONResponse({"error": "embeddings are not supported by this backend"}, status_code=501)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _iso(ts: float | None) -> str:
    dt = datetime.fromtimestamp(ts or 0, tz=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _family(model_id: str) -> str:
    match = re.match(r"[A-Za-z]+", model_id.rsplit("/", 1)[-1])
    return match.group(0).lower() if match else "vllm"


def _details(model_id: str) -> dict[str, Any]:
    family = _family(model_id)
    return {
        "parent_model": "",
        "format": "safetensors",
        "family": family,
        "families": [family],
        "parameter_size": "",
        "quantization_level": "",
    }


def _entry(ctx, name: str, model_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "modified_at": _iso(ctx.backend.ready_since or ctx.started_at),
        "size": 0,
        "digest": hashlib.sha256(model_id.encode()).hexdigest(),
        "details": _details(model_id),
    }


async def _model_names(ctx) -> list[tuple[str, str]]:
    """(advertised name, served id) pairs; empty while the backend is not ready."""
    try:
        models = await ctx.backend.models(ctx.client)
    except BackendUnavailable:
        return []
    ids = [m["id"] for m in models]
    pairs = [(i, i) for i in ids]
    pairs += [(alias, target) for alias, target in ctx.settings.model_aliases.items() if target in ids]
    return pairs


async def tags(request: Request) -> Response:
    ctx = get_ctx(request)
    return JSONResponse({"models": [_entry(ctx, name, mid) for name, mid in await _model_names(ctx)]})


async def ps(request: Request) -> Response:
    ctx = get_ctx(request)
    models = []
    for name, mid in await _model_names(ctx):
        entry = _entry(ctx, name, mid)
        entry.pop("modified_at")
        entry.update({"expires_at": "2999-12-31T23:59:59Z", "size_vram": 0})
        if ctx.backend.context_length:
            entry["context_length"] = ctx.backend.context_length
        models.append(entry)
    return JSONResponse({"models": models})


async def show(request: Request) -> Response:
    ctx = get_ctx(request)
    body = await json_body(request)
    name = body.get("model") or body.get("name")
    if not isinstance(name, str) or not name:
        raise BadRequest("model is required")
    served = await served_model(ctx, name)
    family = _family(served)
    info: dict[str, Any] = {"general.architecture": family, "general.basename": served}
    if ctx.backend.context_length:
        info[f"{family}.context_length"] = ctx.backend.context_length
    return JSONResponse(
        {
            "modelfile": f"# served by vLLM through ollama-vllm-proxy\nFROM {served}\n",
            "parameters": "",
            "template": "",
            "details": _details(served),
            "model_info": info,
            "capabilities": CAPABILITIES,
            "modified_at": _iso(ctx.backend.ready_since or ctx.started_at),
        }
    )


# ---------------------------------------------------------------------------
# Model management (not applicable to a fixed vLLM server)
# ---------------------------------------------------------------------------


async def pull(request: Request) -> Response:
    ctx = get_ctx(request)
    body = await json_body(request)
    name = body.get("model") or body.get("name")
    names = {n for n, _ in await _model_names(ctx)}
    candidates = {name, name[: -len(":latest")] if isinstance(name, str) and name.endswith(":latest") else name}
    if not isinstance(name, str) or not (candidates & names):
        return JSONResponse({"error": f"pull is not supported: model '{name}' is not served by this backend"}, status_code=404)
    if body.get("stream") is False:
        return JSONResponse({"status": "success"})
    lines = b"".join(json.dumps({"status": s}).encode() + b"\n" for s in ("pulling manifest", "success"))
    return Response(lines, media_type="application/x-ndjson")


async def unsupported(request: Request) -> Response:
    return JSONResponse({"error": "not supported by the vLLM proxy"}, status_code=501)


routes = [
    Route("/", root, methods=["GET", "HEAD"]),
    Route("/healthz", healthz, methods=["GET"]),
    Route("/api/version", version, methods=["GET"]),
    Route("/api/chat", chat, methods=["POST"]),
    Route("/api/generate", generate, methods=["POST"]),
    Route("/api/embed", embed, methods=["POST"]),
    Route("/api/embeddings", embed, methods=["POST"]),
    Route("/api/tags", tags, methods=["GET"]),
    Route("/api/ps", ps, methods=["GET"]),
    Route("/api/show", show, methods=["POST"]),
    Route("/api/pull", pull, methods=["POST"]),
    Route("/api/push", unsupported, methods=["POST"]),
    Route("/api/create", unsupported, methods=["POST"]),
    Route("/api/copy", unsupported, methods=["POST"]),
    Route("/api/delete", unsupported, methods=["DELETE"]),
    Route("/api/blobs/{digest:path}", unsupported, methods=["GET", "HEAD", "POST"]),
]
