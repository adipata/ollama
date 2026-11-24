"""/admin/api - pod connection, profiles, deployment, logs (PLAN section 12).

Authentication is done by the middleware for everything under /admin/api.
In external mode (VLLM_BASE_URL set) the pod is not managed by the proxy:
reading still works, every action answers 409.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..pod import logs as podlogs
from ..pod import profiles as prof
from ..pod.deploy import SECRET_HF_TOKEN, PodError, PodManager
from ..pod.ssh import SSHError
from .common import get_ctx, json_body

_SSE_HEADERS = {"cache-control": "no-store", "x-accel-buffering": "no"}


def get_manager(ctx: Any) -> PodManager:
    """The one PodManager of this application (created on first use, so tests need no lifespan)."""
    manager = ctx.extras.get("pod")
    if manager is None:
        manager = ctx.extras["pod"] = PodManager(ctx)
    return manager


async def startup(ctx: Any) -> None:
    await get_manager(ctx).startup()


async def shutdown(ctx: Any) -> None:
    manager = ctx.extras.get("pod")
    if manager is not None:
        await manager.shutdown()


def _mgr(request: Request) -> PodManager:
    return get_manager(get_ctx(request))


def _error(exc: PodError) -> Response:
    return JSONResponse({"error": exc.message}, status_code=exc.status)


def _guard(handler):
    """Turn PodError into its JSON error response."""

    async def wrapped(request: Request) -> Response:
        try:
            return await handler(request)
        except PodError as exc:
            return _error(exc)

    wrapped.__name__ = handler.__name__
    return wrapped


# -- connection ---------------------------------------------------------------------


@_guard
async def pod_get(request: Request) -> Response:
    return JSONResponse(_mgr(request).describe())


@_guard
async def pod_put(request: Request) -> Response:
    body = await json_body(request)
    manager = _mgr(request)
    parsed = manager.set_connection(body.get("ssh_command"))
    return JSONResponse({"parsed": parsed, **manager.describe()})


@_guard
async def pod_connect(request: Request) -> Response:
    body = await json_body(request, allow_empty=True)
    accept = body.get("accept_fingerprint")
    if accept is not None and not isinstance(accept, str):
        raise PodError("accept_fingerprint must be the fingerprint shown for this pod")
    manager = _mgr(request)
    result = await manager.connect(accept)
    return JSONResponse({**result, "status": manager.describe()})


@_guard
async def pod_disconnect(request: Request) -> Response:
    manager = _mgr(request)
    await manager.disconnect()
    return JSONResponse(manager.describe())


@_guard
async def pod_delete_models(request: Request) -> Response:
    body = await json_body(request, allow_empty=True)
    return JSONResponse(await _mgr(request).delete_models(body.get("dirs")))


# -- Hugging Face token (write-only) ------------------------------------------------


@_guard
async def hf_token_get(request: Request) -> Response:
    return JSONResponse({"set": bool(get_ctx(request).store.get_secret(SECRET_HF_TOKEN))})


@_guard
async def hf_token_put(request: Request) -> Response:
    manager = _mgr(request)
    manager.require_enabled()
    body = await json_body(request)
    token = body.get("token")
    if not isinstance(token, str) or not 8 <= len(token.strip()) <= 256 or any(c.isspace() for c in token.strip()):
        raise PodError("token must be a Hugging Face access token (no spaces)")
    manager.ctx.store.set_secret(SECRET_HF_TOKEN, token.strip())
    manager.hf.forget()
    manager.publish()
    return JSONResponse({"set": True})


@_guard
async def hf_token_delete(request: Request) -> Response:
    manager = _mgr(request)
    manager.require_enabled()
    manager.ctx.store.delete_secret(SECRET_HF_TOKEN)
    manager.hf.forget()
    manager.publish()
    return JSONResponse({"set": False})


# -- profiles -----------------------------------------------------------------------


def _profile_error(exc: prof.ProfileError) -> Response:
    return JSONResponse({"error": str(exc)}, status_code=404 if str(exc) == "no such profile" else 400)


@_guard
async def profiles_list(request: Request) -> Response:
    manager = _mgr(request)
    return JSONResponse({"profiles": manager.profiles.list(), "remote_root": manager.cfg.remote_root,
                         "template": prof.default_profile()})


@_guard
async def profiles_create(request: Request) -> Response:
    manager = _mgr(request)
    manager.require_enabled()
    body = await json_body(request)
    try:
        return JSONResponse(manager.profiles.create(body.get("name"), body.get("data")), status_code=201)
    except prof.ProfileError as exc:
        return _profile_error(exc)


@_guard
async def profiles_update(request: Request) -> Response:
    manager = _mgr(request)
    manager.require_enabled()
    body = await json_body(request)
    try:
        return JSONResponse(manager.profiles.update(request.path_params["profile_id"], body.get("name"),
                                                    body.get("data")))
    except prof.ProfileError as exc:
        return _profile_error(exc)


@_guard
async def profiles_delete(request: Request) -> Response:
    manager = _mgr(request)
    manager.require_enabled()
    if not manager.profiles.delete(request.path_params["profile_id"]):
        return JSONResponse({"error": "no such profile"}, status_code=404)
    return JSONResponse({"deleted": True})


@_guard
async def profile_plan(request: Request) -> Response:
    gpu = request.query_params.get("gpu") or None
    count_raw = request.query_params.get("count")
    count = None
    if count_raw:
        if not count_raw.isdigit():
            raise PodError("count must be a number")
        count = int(count_raw)
    return JSONResponse(await _mgr(request).plan_for(request.path_params["profile_id"], gpu, count))


@_guard
async def profile_sizing(request: Request) -> Response:
    return JSONResponse(await _mgr(request).sizing(request.path_params["profile_id"]))


@_guard
async def catalog_get(request: Request) -> Response:
    return JSONResponse({"catalog": _mgr(request).catalog()})


@_guard
async def catalog_put(request: Request) -> Response:
    manager = _mgr(request)
    manager.require_enabled()
    body = await json_body(request)
    return JSONResponse({"catalog": manager.set_catalog(body.get("catalog"))})


# -- deployment ---------------------------------------------------------------------


@_guard
async def deploy_start(request: Request) -> Response:
    body = await json_body(request)
    return JSONResponse(await _mgr(request).deploy(body), status_code=202)


@_guard
async def deploy_cancel(request: Request) -> Response:
    return JSONResponse(await _mgr(request).cancel())


@_guard
async def vllm_stop(request: Request) -> Response:
    return JSONResponse(await _mgr(request).stop_vllm())


@_guard
async def vllm_restart(request: Request) -> Response:
    body = await json_body(request, allow_empty=True)
    return JSONResponse(await _mgr(request).restart_vllm(body.get("context")), status_code=202)


@_guard
async def deploy_status(request: Request) -> Response:
    return JSONResponse(_mgr(request).describe())


async def status_events(manager: PodManager, heartbeat: float = 15.0) -> AsyncIterator[bytes]:
    """SSE body: the full status now, then again whenever something changes."""
    queue = manager.subscribe()
    try:
        last = ""
        while True:
            text = json.dumps(manager.describe(), separators=(",", ":"))
            if text != last:
                last = text
                yield b"data: " + text.encode() + b"\n\n"
            try:
                await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                yield b": keep-alive\n\n"
            await asyncio.sleep(0.1)  # coalesce bursts of updates
            while not queue.empty():
                queue.get_nowait()
    finally:
        manager.unsubscribe(queue)


async def deploy_events(request: Request) -> Response:
    return StreamingResponse(status_events(_mgr(request)), media_type="text/event-stream", headers=_SSE_HEADERS)


# -- logs ---------------------------------------------------------------------------


def _source(request: Request) -> str:
    source = request.query_params.get("source", "vllm")
    if source not in podlogs.SOURCES:
        raise PodError(f"source must be one of: {', '.join(podlogs.SOURCES)}")
    return source


@_guard
async def logs_stream(request: Request) -> Response:
    source = _source(request)
    return StreamingResponse(podlogs.stream(_mgr(request), source), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@_guard
async def logs_download(request: Request) -> Response:
    source = _source(request)
    manager = _mgr(request)
    if manager.ssh is None or not manager.ssh.up:
        raise PodError("no pod connected", 409)
    try:
        text = await podlogs.full_text(manager.ssh, podlogs.log_path(manager.cfg.remote_root, source),
                                       manager.secret_values())
    except SSHError as exc:
        raise PodError(str(exc), 502) from None
    return Response(text, media_type="text/plain; charset=utf-8", headers={
        "content-disposition": f'attachment; filename="{source}.log"', "cache-control": "no-store"})


routes: list = [
    Route("/admin/api/pod", pod_get, methods=["GET"]),
    Route("/admin/api/pod", pod_put, methods=["PUT"]),
    Route("/admin/api/pod/connect", pod_connect, methods=["POST"]),
    Route("/admin/api/pod/disconnect", pod_disconnect, methods=["POST"]),
    Route("/admin/api/pod/delete-models", pod_delete_models, methods=["POST"]),
    Route("/admin/api/settings/hf-token", hf_token_get, methods=["GET"]),
    Route("/admin/api/settings/hf-token", hf_token_put, methods=["PUT"]),
    Route("/admin/api/settings/hf-token", hf_token_delete, methods=["DELETE"]),
    Route("/admin/api/profiles", profiles_list, methods=["GET"]),
    Route("/admin/api/profiles", profiles_create, methods=["POST"]),
    Route("/admin/api/profiles/{profile_id:int}", profiles_update, methods=["PUT"]),
    Route("/admin/api/profiles/{profile_id:int}", profiles_delete, methods=["DELETE"]),
    Route("/admin/api/profiles/{profile_id:int}/plan", profile_plan, methods=["GET"]),
    Route("/admin/api/profiles/{profile_id:int}/sizing", profile_sizing, methods=["GET"]),
    Route("/admin/api/gpu-catalog", catalog_get, methods=["GET"]),
    Route("/admin/api/gpu-catalog", catalog_put, methods=["PUT"]),
    Route("/admin/api/deploy", deploy_start, methods=["POST"]),
    Route("/admin/api/deploy/cancel", deploy_cancel, methods=["POST"]),
    Route("/admin/api/deploy/status", deploy_status, methods=["GET"]),
    Route("/admin/api/deploy/events", deploy_events, methods=["GET"]),
    Route("/admin/api/vllm/stop", vllm_stop, methods=["POST"]),
    Route("/admin/api/vllm/restart", vllm_restart, methods=["POST"]),
    Route("/admin/api/logs/stream", logs_stream, methods=["GET"]),
    Route("/admin/api/logs/download", logs_download, methods=["GET"]),
]
