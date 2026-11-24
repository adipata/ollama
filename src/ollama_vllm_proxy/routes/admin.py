"""/admin - static management pages and the key-management API.

The pages contain no data: everything is fetched from /admin/api/* with the
admin token, which the page keeps in sessionStorage.
"""

from __future__ import annotations

from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ..keys import KeyError_
from .common import get_ctx, json_body

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
_PAGES = {"/admin": "admin.html", "/admin/": "admin.html", "/admin/pod": "pod.html", "/admin/monitor": "monitor.html"}


def _static(name: str) -> Response:
    path = STATIC_DIR / name
    if "/" in name or "\\" in name or path.suffix not in _TYPES or not path.is_file():
        return Response("not found", status_code=404, media_type="text/plain")
    return Response(
        path.read_bytes(),
        media_type=_TYPES[path.suffix],
        headers={
            "content-security-policy": _CSP,
            "referrer-policy": "no-referrer",
            "x-frame-options": "DENY",
            "x-content-type-options": "nosniff",
            "cache-control": "no-store",
        },
    )


async def page(request: Request) -> Response:
    return _static(_PAGES[request.url.path])


async def static_file(request: Request) -> Response:
    return _static(request.path_params["name"])


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _key_error(exc: KeyError_) -> Response:
    status = 404 if str(exc) == "no such key" else 400
    return JSONResponse({"error": str(exc)}, status_code=status)


async def keys_list(request: Request) -> Response:
    return JSONResponse({"keys": get_ctx(request).keys.list()})


async def keys_create(request: Request) -> Response:
    body = await json_body(request)
    try:
        return JSONResponse(get_ctx(request).keys.create(str(body.get("name") or "")), status_code=201)
    except KeyError_ as exc:
        return _key_error(exc)


async def keys_update(request: Request) -> Response:
    body = await json_body(request)
    name = body.get("name") if isinstance(body.get("name"), str) else None
    enabled = body.get("enabled") if isinstance(body.get("enabled"), bool) else None
    try:
        return JSONResponse(get_ctx(request).keys.update(request.path_params["key_id"], name=name, enabled=enabled))
    except KeyError_ as exc:
        return _key_error(exc)


async def keys_revoke(request: Request) -> Response:
    try:
        return JSONResponse(get_ctx(request).keys.revoke(request.path_params["key_id"]))
    except KeyError_ as exc:
        return _key_error(exc)


async def status(request: Request) -> Response:
    ctx = get_ctx(request)
    return JSONResponse(
        {
            "backend": ctx.backend.describe(),
            "open_requests": ctx.open_requests,
            "rejected_requests": ctx.rejected,
            "active_keys": ctx.keys.count_active(),
            "writer": {"queued": ctx.store.queued, "dropped": ctx.store.dropped, "failed": ctx.store.failed},
            "started_at": ctx.started_at,
            "listen": f"{ctx.settings.host}:{ctx.settings.port}",
        }
    )


routes = [
    Route("/admin", page, methods=["GET"]),
    Route("/admin/", page, methods=["GET"]),
    Route("/admin/pod", page, methods=["GET"]),
    Route("/admin/monitor", page, methods=["GET"]),
    Route("/admin/static/{name}", static_file, methods=["GET"]),
    Route("/admin/api/keys", keys_list, methods=["GET"]),
    Route("/admin/api/keys", keys_create, methods=["POST"]),
    Route("/admin/api/keys/{key_id:int}", keys_update, methods=["PATCH"]),
    Route("/admin/api/keys/{key_id:int}", keys_revoke, methods=["DELETE"]),
    Route("/admin/api/status", status, methods=["GET"]),
]
