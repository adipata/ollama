"""/v1 - OpenAI-compatible pass-through to vLLM."""

from __future__ import annotations

import json
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..backend import BackendUnavailable
from ..forward import OpenAIAdapter, ProxyCall
from .common import get_ctx, json_body, new_record, served_model

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length", "content-encoding", "host", "authorization",
    "x-api-key", "accept-encoding", "cookie", "origin", "referer",
}


async def _inference(request: Request, endpoint: str, path: str) -> Response:
    ctx = get_ctx(request)
    body = await json_body(request)
    client_model = body.get("model") if isinstance(body.get("model"), str) else None
    served = await served_model(ctx, client_model)
    stream = bool(body.get("stream"))
    rec = new_record(request, ctx, api="openai", endpoint=endpoint, body=body, stream=stream)
    body["model"] = served
    wants_usage = True
    if stream:
        options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        wants_usage = bool(options.get("include_usage"))
        if not wants_usage:
            # Usage is needed for the request record; the extra chunk is hidden from the client.
            body["stream_options"] = {**options, "include_usage": True}
            rec.include_usage_forced = True
    adapter = OpenAIAdapter(client_model, served, client_wants_usage=wants_usage)
    return ProxyCall(ctx, rec, path=path, body=body, stream=stream, adapter=adapter)


async def chat_completions(request: Request) -> Response:
    return await _inference(request, "chat.completions", "/v1/chat/completions")


async def completions(request: Request) -> Response:
    return await _inference(request, "completions", "/v1/completions")


async def models(request: Request) -> Response:
    ctx = get_ctx(request)
    upstream = await ctx.backend.models(ctx.client)
    data = list(upstream)
    known = {m["id"] for m in data}
    for alias, target in ctx.settings.model_aliases.items():
        base = next((m for m in data if m["id"] == target), None)
        if base is not None and alias not in known:
            data.append({**base, "id": alias})
    model_id = request.path_params.get("model_id")
    if model_id is not None:
        found = next((m for m in data if m["id"] == model_id), None)
        if found is None:
            raise BackendUnavailable(f"model '{model_id}' not found", 404)
        return JSONResponse(found)
    return JSONResponse({"object": "list", "data": data})


async def embeddings(request: Request) -> Response:
    return JSONResponse(
        {"error": {"message": "embeddings are not supported by this backend", "type": "not_implemented",
                   "param": None, "code": 501}},
        status_code=501,
    )


async def passthrough(request: Request) -> Response:
    """Any other /v1 route: forwarded as-is (streamed), without a request record."""
    ctx = get_ctx(request)
    backend = ctx.backend
    backend.require_ready()
    raw = await request.body()
    client_model: Any = None
    if raw and "json" in request.headers.get("content-type", ""):
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("model"), str):
            client_model = body["model"]
            body["model"] = await served_model(ctx, client_model)
            raw = json.dumps(body).encode()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    headers.update(backend.auth_headers())
    headers["accept-encoding"] = "identity"
    url = backend.base_url + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    upstream = ctx.client.build_request(request.method, url, content=raw, headers=headers)
    try:
        resp = await ctx.client.send(upstream, stream=True)
    except httpx.HTTPError as exc:
        raise BackendUnavailable(f"upstream unavailable ({type(exc).__name__})", 502) from None

    async def relay():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            pass
        finally:
            await resp.aclose()

    out_headers = {
        k: v for k, v in resp.headers.items() if k.lower() in ("content-type", "cache-control", "retry-after")
    }
    return StreamingResponse(relay(), status_code=resp.status_code, headers=out_headers)


_ANY = ["GET", "POST", "PUT", "PATCH", "DELETE"]

routes = [
    Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    Route("/v1/completions", completions, methods=["POST"]),
    Route("/v1/models", models, methods=["GET"]),
    Route("/v1/models/{model_id:path}", models, methods=["GET"]),
    Route("/v1/embeddings", embeddings, methods=["POST"]),
    Route("/v1/{rest:path}", passthrough, methods=_ANY),
]
