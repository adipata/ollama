"""Application assembly: context, middleware, lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .auth import AuthMiddleware, error_body
from .backend import Backend, BackendUnavailable
from .config import Settings
from .keys import KeyStore, load_admin_token
from .live import Live
from .routes import admin, monitor, ollama, openai, pod
from .store import Store
from .translate.request import BadRequest

log = logging.getLogger("ovp.app")


class Context:
    """Everything a request handler needs; one instance per application."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.store = Store(settings.db_path)
        self.keys = KeyStore(self.store)
        self.admin_token = load_admin_token(settings.admin_token_path, settings.admin_token)
        self.backend = Backend(settings)
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            trust_env=False,  # the upstream is local or behind our own tunnel; never use a system proxy
        )
        self.live = Live(self)
        self.open_requests = 0
        self.rejected = 0  # requests refused for a missing or invalid API key
        self.started_at = time.time()
        self.extras: dict[str, Any] = {}  # owned by the pod and monitoring modules

    async def aclose(self) -> None:
        await self.client.aclose()
        self.store.stop()


async def _retention_loop(ctx: Context) -> None:
    while True:
        ctx.store.prune(ctx.settings.retention_days)
        await asyncio.sleep(3600)


def _error(request: Request, status: int, message: str, etype: str) -> Response:
    return Response(
        error_body(request.url.path, status, message, etype), status_code=status, media_type="application/json"
    )


async def _backend_unavailable(request: Request, exc: BackendUnavailable) -> Response:
    etype = "not_found_error" if exc.status == 404 else "service_unavailable"
    return _error(request, exc.status, exc.message, etype)


async def _bad_request(request: Request, exc: BadRequest) -> Response:
    return _error(request, 400, str(exc), "invalid_request_error")


def create_app(settings: Settings | None = None, ctx: Context | None = None) -> Starlette:
    settings = settings or (ctx.settings if ctx else Settings.from_env())
    ctx = ctx or Context(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        ctx.store.start()
        tasks = [asyncio.create_task(_retention_loop(ctx))]
        await pod.startup(ctx)
        await monitor.startup(ctx)
        if not ctx.keys.count_active():
            log.warning("no API keys yet: every /api and /v1 call is rejected until one is created at /admin")
        log.info("admin token file: %s", settings.admin_token_path)
        try:
            yield
        finally:
            await monitor.shutdown(ctx)
            await pod.shutdown(ctx)
            for task in tasks:
                task.cancel()
            await ctx.aclose()

    origin_regex = settings.origin_regex()
    cors: dict[str, Any] = {"allow_methods": ["*"], "allow_headers": ["*"]}
    if origin_regex is None:
        cors["allow_origins"] = ["*"]
    else:
        cors["allow_origin_regex"] = origin_regex
    app = Starlette(
        routes=[*ollama.routes, *admin.routes, *pod.routes, *monitor.routes, *openai.routes],
        middleware=[Middleware(CORSMiddleware, **cors), Middleware(AuthMiddleware, ctx=ctx)],
        exception_handlers={BackendUnavailable: _backend_unavailable, BadRequest: _bad_request},
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    return app
