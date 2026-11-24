"""Monitoring: the engine sampler task, the live snapshot and the report download.

``/admin/api/*`` is already behind the admin token (AuthMiddleware); the
handlers here add none of their own.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import anyio
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .. import export
from ..live import MAX_POINTS
from ..sampler import Sampler
from .common import get_ctx

if TYPE_CHECKING:
    from ..app import Context

MIN_WINDOW_S = 30.0
DEFAULT_WINDOW_S = 300.0
RING_WINDOW_S = 900.0  # windows up to this are served from memory when the ring covers them


class Monitor:
    """Per-application monitoring state, kept in ``ctx.extras["monitor"]``."""

    def __init__(self, ctx: "Context"):
        self.ctx = ctx
        self.sampler = Sampler(ctx)
        self._history: dict[float, tuple[float, dict[str, Any]]] = {}  # window -> (expires, history)

    def max_window(self) -> float:
        days = self.ctx.settings.retention_days
        return (days if days > 0 else 365) * 86400.0

    def cache_ttl(self, window_s: float) -> float:
        """Database windows change slowly: one chart point spans window/MAX_POINTS seconds."""
        return min(60.0, window_s / MAX_POINTS)

    async def snapshot(self, window_s: float) -> dict[str, Any]:
        ctx = self.ctx
        now = time.time()
        if window_s <= RING_WINDOW_S and ctx.live.covers(now - window_s):
            snap = ctx.live.snapshot(window_s)
            snap["refresh_s"] = ctx.live.interval
            return snap
        cached = self._history.get(window_s)
        if cached is None or cached[0] <= time.monotonic():
            data = await anyio.to_thread.run_sync(export.history, ctx.store.read, now - window_s, now)
            if len(self._history) > 16:
                self._history.clear()
            cached = (time.monotonic() + self.cache_ttl(window_s), data)
            self._history[window_s] = cached
        snap = ctx.live.snapshot(window_s, history=cached[1])
        snap["refresh_s"] = max(ctx.live.interval, self.cache_ttl(window_s))
        return snap


def monitor_of(ctx: "Context") -> Monitor:
    mon = ctx.extras.get("monitor")
    if mon is None:
        mon = ctx.extras["monitor"] = Monitor(ctx)
    return mon


def _bad(message: str) -> Response:
    return JSONResponse({"error": message}, status_code=400)


def _float(request: Request, name: str) -> float | None:
    raw = request.query_params.get(name, "").strip()
    if not raw:
        return None
    value = float(raw)  # ValueError is turned into a 400 by the callers
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(name)
    return value


async def live(request: Request) -> Response:
    ctx = get_ctx(request)
    mon = monitor_of(ctx)
    try:
        window = _float(request, "window") or DEFAULT_WINDOW_S
    except ValueError:
        return _bad("window must be a number of seconds")
    window = min(max(window, MIN_WINDOW_S), mon.max_window())
    return JSONResponse(await mon.snapshot(window), headers={"cache-control": "no-store"})


async def export_options(request: Request) -> Response:
    ctx = get_ctx(request)
    try:
        t0, t1 = _float(request, "from"), _float(request, "to")
    except ValueError:
        return _bad("from and to must be epoch seconds")
    data = await anyio.to_thread.run_sync(export.options, ctx.store.read, t0, t1)
    data["now"] = time.time()
    data["retention_days"] = ctx.settings.retention_days
    return JSONResponse(data, headers={"cache-control": "no-store"})


async def export_report(request: Request) -> Response:
    ctx = get_ctx(request)
    q = request.query_params
    fmt = q.get("format", "html")
    if fmt not in ("html", "json"):
        return _bad("format must be html or json")
    try:
        t1 = _float(request, "to") or time.time()
        t0 = _float(request, "from")
        tz = int(float(q.get("tz_offset", "0") or 0))
    except ValueError:
        return _bad("from, to and tz_offset must be numbers")
    if t0 is None:
        t0 = t1 - 3600.0
    key, session = q.get("key", "").strip()[:128], q.get("session", "").strip()[:128]
    backend = ctx.backend

    def build() -> tuple[str, str]:
        ctx.store.flush()  # include the rows still waiting in the writer queue
        report = export.build_report(
            ctx.store.read, t0, t1, key or None, session or None,
            served_model=backend.served_model or None,
            context_length=backend.context_length,
            proxy={"dropped": ctx.store.dropped, "failed": ctx.store.failed, "rejected": ctx.rejected},
        )
        if fmt == "json":
            return export.filename(report, "json"), export.to_json(report)
        return export.filename(report, "html"), export.render_html(report, max(-900, min(900, tz)))

    try:
        name, text = await anyio.to_thread.run_sync(build)
    except export.ExportError as exc:
        return _bad(str(exc))
    return Response(
        text,
        media_type="application/json" if fmt == "json" else "text/html; charset=utf-8",
        headers={
            "content-disposition": f'attachment; filename="{name}"',
            "content-security-policy": "default-src 'none'; style-src 'unsafe-inline'",
            "x-content-type-options": "nosniff",
            "cache-control": "no-store",
        },
    )


routes: list = [
    Route("/admin/api/live", live, methods=["GET"]),
    Route("/admin/api/export/options", export_options, methods=["GET"]),
    Route("/admin/api/export", export_report, methods=["GET"]),
]


async def startup(ctx: Any) -> None:
    # httpx logs every request at INFO; with a scrape every 1.5 s that would bury everything else.
    httpx_log = logging.getLogger("httpx")
    if httpx_log.level == logging.NOTSET and logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        httpx_log.setLevel(logging.WARNING)
    monitor_of(ctx).sampler.start()


async def shutdown(ctx: Any) -> None:
    mon = ctx.extras.get("monitor")
    if mon is not None:
        await mon.sampler.stop()
