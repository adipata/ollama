"""Helpers shared by the inference routes."""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from starlette.requests import Request

from ..backend import resolve_model
from ..observe import RequestRecord, nonneg_int, opaque
from ..translate.request import BadRequest

if TYPE_CHECKING:
    from ..app import Context


def get_ctx(request: Request) -> "Context":
    return request.app.state.ctx


async def json_body(request: Request, allow_empty: bool = False) -> dict[str, Any]:
    raw = await request.body()
    if not raw.strip():
        if allow_empty:
            return {}
        raise BadRequest("request body must be a JSON object")
    try:
        body = json.loads(raw)
    except ValueError:
        raise BadRequest("request body is not valid JSON") from None
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    return body


async def served_model(ctx: "Context", requested: Any) -> str:
    """Check the backend is usable and map the client's model name to the served one."""
    models = await ctx.backend.models(ctx.client)
    return resolve_model(ctx.backend, ctx.settings, requested, [m["id"] for m in models])


def new_record(request: Request, ctx: "Context", *, api: str, endpoint: str, body: dict[str, Any], stream: bool) -> RequestRecord:
    state = request.scope.get("state") or {}
    messages = body.get("messages")
    tools = body.get("tools")
    model = body.get("model")
    return RequestRecord(
        request_id=uuid.uuid4().hex,
        deployment_id=ctx.backend.label,
        key_id=state.get("key_id"),
        actor_id=state.get("key_name"),
        api=api,
        session_id=opaque(request.headers.get("x-session-id")),
        endpoint=endpoint,
        model=model[:128] if isinstance(model, str) else None,
        stream=stream,
        requested_max_tokens=nonneg_int(body.get("max_tokens") or body.get("max_completion_tokens")),
        messages_count=len(messages) if isinstance(messages, list) else None,
        tools_count=len(tools) if isinstance(tools, list) else None,
        t_arrival_wall=time.time(),
        t0=time.monotonic(),
    )
