"""Forwarding of one inference request with disconnect-aware cancellation.

The upstream call runs inside the ASGI response so that one task can watch for
the client's ``http.disconnect`` the whole time (including a long prefill
before the first byte) and cancel the upstream request, which makes vLLM abort
the sequence.

An *adapter* decides what the client sees: the OpenAI adapter relays the
upstream bytes, the Ollama adapters translate them to NDJSON.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import anyio
import httpx
from starlette.responses import Response

from .observe import (
    RequestRecord,
    SSESplitter,
    apply_error,
    finalize,
    inspect_completion,
    is_usage_only,
    observe_stream_object,
)

if TYPE_CHECKING:
    from .app import Context

log = logging.getLogger("ovp.forward")


class OpenAIAdapter:
    """Pass-through for /v1: upstream bytes, with the client's model name restored."""

    stream_content_type = b"text/event-stream; charset=utf-8"

    def __init__(self, client_model: str | None, served_model: str, client_wants_usage: bool = True):
        self.client_model = client_model
        self.rewrite = bool(client_model) and client_model != served_model
        self.client_wants_usage = client_wants_usage

    def _fix(self, obj: dict) -> bool:
        """Apply response rewrites in place; returns True when something changed."""
        changed = False
        if self.rewrite and "model" in obj:
            obj["model"] = self.client_model
            changed = True
        for choice in obj.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            for key in ("delta", "message"):
                msg = choice.get(key)
                # Ollama's /v1 calls the field "reasoning"; keep vLLM's name as well.
                if isinstance(msg, dict) and msg.get("reasoning_content") and "reasoning" not in msg:
                    msg["reasoning"] = msg["reasoning_content"]
                    changed = True
        return changed

    # streaming
    def comment(self, raw: bytes) -> bytes:
        return raw + b"\n\n"

    def event(self, obj: dict, raw: bytes, rec: RequestRecord) -> bytes:
        if not self.client_wants_usage and is_usage_only(obj):
            return b""  # the proxy forced include_usage; the client did not ask for it
        if self._fix(obj):
            return b"data: " + json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode() + b"\n\n"
        return raw + b"\n\n"

    def malformed(self, raw: bytes) -> bytes:
        return raw + b"\n\n"

    def done(self, rec: RequestRecord) -> bytes:
        return b"data: [DONE]\n\n"

    def end(self, rec: RequestRecord) -> bytes:
        return b""

    def interrupted(self, rec: RequestRecord) -> bytes:
        return (
            b'data: {"error": {"message": "upstream stream interrupted", '
            b'"type": "proxy_upstream_interrupted"}}\n\n'
        )

    # buffered
    def body(self, status: int, obj: Any, raw: bytes, ctype: str, rec: RequestRecord) -> tuple[int, str, bytes]:
        if status == 200 and isinstance(obj, dict) and self._fix(obj):
            return status, "application/json", json.dumps(obj, ensure_ascii=False).encode()
        return status, ctype or "application/json", raw

    def error(self, status: int, message: str, etype: str) -> tuple[str, bytes]:
        body = {"error": {"message": message, "type": etype, "param": None, "code": status}}
        return "application/json", json.dumps(body).encode()


class ProxyCall(Response):
    def __init__(
        self,
        ctx: "Context",
        rec: RequestRecord,
        *,
        path: str,
        body: dict[str, Any],
        stream: bool,
        adapter: Any,
    ):
        self.ctx = ctx
        self.rec = rec
        self.path = path
        self.body = body
        self.stream = stream
        self.adapter = adapter
        self.background = None
        self.status_code = 200
        self._response_done = False
        self._client_gone = False

    async def __call__(self, scope, receive, send) -> None:  # type: ignore[override]
        ctx = self.ctx
        self.rec.open_requests_at_start = ctx.open_requests
        ctx.open_requests += 1
        ctx.live.on_request_start(self.rec)
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._watch_disconnect, receive, tg.cancel_scope)
                try:
                    await self._run(send)
                finally:
                    tg.cancel_scope.cancel()
        finally:
            ctx.open_requests -= 1
            self._finalize()

    async def _watch_disconnect(self, receive, cancel_scope: anyio.CancelScope) -> None:
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                if not self._response_done:
                    self._client_gone = True
                    cancel_scope.cancel()
                return

    async def _run(self, send) -> None:
        rec, ctx = self.rec, self.ctx
        backend = ctx.backend
        headers = {"content-type": "application/json", "accept-encoding": "identity"}
        headers.update(backend.auth_headers())
        if self.stream:
            headers["accept"] = "text/event-stream"
        request = ctx.client.build_request(
            "POST", backend.base_url + self.path, content=json.dumps(self.body).encode(), headers=headers
        )
        resp: httpx.Response | None = None
        try:
            try:
                resp = await ctx.client.send(request, stream=True)
            except httpx.HTTPError as exc:
                rec.outcome = "upstream_unreachable"
                rec.error_type = type(exc).__name__
                ctype, body = self.adapter.error(502, "upstream unavailable", "proxy_upstream_unavailable")
                await self._send_buffered(send, 502, ctype, body)
                return
            rec.status_code = resp.status_code
            rec.first_byte_ms = rec.ms(time.monotonic())
            ctype = resp.headers.get("content-type", "")
            if self.stream and resp.status_code == 200 and ctype.startswith("text/event-stream"):
                await self._relay_stream(resp, send)
            else:
                await self._relay_buffered(resp, send, ctype)
        except anyio.get_cancelled_exc_class():
            raise
        except OSError:
            self._client_gone = True
        except Exception as exc:  # never log payloads, only the failure type
            log.error("proxy failure: %s", type(exc).__name__, exc_info=log.isEnabledFor(logging.DEBUG))
            rec.outcome = rec.outcome or "proxy_error"
            rec.error_type = rec.error_type or type(exc).__name__
            if not self._response_done:
                try:
                    ctype, body = self.adapter.error(500, "proxy error", "proxy_error")
                    await self._send_buffered(send, 500, ctype, body)
                except Exception:
                    pass
        finally:
            if resp is not None:
                with anyio.CancelScope(shield=True):
                    await resp.aclose()

    async def _relay_buffered(self, resp: httpx.Response, send, ctype: str) -> None:
        rec = self.rec
        try:
            data = await resp.aread()
        except httpx.HTTPError as exc:
            rec.outcome = "upstream_interrupted"
            rec.error_type = type(exc).__name__
            etype, body = self.adapter.error(502, "upstream response interrupted", "proxy_upstream_interrupted")
            await self._send_buffered(send, 502, etype, body)
            return
        now = time.monotonic()
        obj = None
        try:
            obj = json.loads(data)
        except ValueError:
            rec.malformed_events += 1
        if resp.status_code == 200:
            rec.outcome = "ok"
            if isinstance(obj, dict):
                inspect_completion(rec, obj, now)
                if rec.error_type:
                    rec.outcome = "error_event"
        else:
            rec.outcome = "upstream_error"
            if isinstance(obj, dict):
                apply_error(rec, obj.get("error", obj))
            elif rec.error_type is None:
                rec.error_type = f"http_{resp.status_code}"
        status, out_ctype, out = self.adapter.body(resp.status_code, obj, data, ctype, rec)
        await self._send_buffered(send, status, out_ctype, out)

    def _handle_event(self, raw: bytes, data: bytes | None, now: float) -> bytes:
        rec, adapter = self.rec, self.adapter
        if data is None:
            return adapter.comment(raw)
        if data == b"[DONE]":
            rec.done_marker = True
            return adapter.done(rec)
        try:
            obj = json.loads(data)
        except ValueError:
            rec.malformed_events += 1
            return adapter.malformed(raw)
        if not isinstance(obj, dict):
            return adapter.malformed(raw)
        observe_stream_object(rec, obj, now)
        return adapter.event(obj, raw, rec)

    async def _relay_stream(self, resp: httpx.Response, send) -> None:
        rec, adapter = self.rec, self.adapter
        splitter = SSESplitter()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", adapter.stream_content_type),
                    (b"cache-control", b"no-cache"),
                    (b"x-accel-buffering", b"no"),
                    (b"x-proxy-request-id", rec.request_id.encode()),
                ],
            }
        )
        try:
            async for chunk in resp.aiter_raw():
                now = time.monotonic()
                out = b"".join(self._handle_event(raw, data, now) for raw, data in splitter.feed(chunk))
                if out:
                    await send({"type": "http.response.body", "body": out, "more_body": True})
        except httpx.HTTPError as exc:
            rec.outcome = "upstream_interrupted"
            rec.error_type = type(exc).__name__
            final = adapter.interrupted(rec)
        else:
            now = time.monotonic()
            final = b"".join(self._handle_event(raw, data, now) for raw, data in splitter.tail())
            if rec.error_type:
                rec.outcome = "error_event"
            elif rec.done_marker:
                rec.outcome = "ok"
            else:
                rec.outcome = "incomplete_stream"
            final += adapter.end(rec)
        self._response_done = True
        await send({"type": "http.response.body", "body": final, "more_body": False})

    async def _send_buffered(self, send, status: int, ctype: str, body: bytes) -> None:
        self._response_done = True
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", ctype.encode("latin-1")),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-proxy-request-id", self.rec.request_id.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    def _finalize(self) -> None:
        rec = self.rec
        if self._client_gone:
            rec.cancelled = True
            rec.outcome = "client_cancelled"
        if rec.outcome is None:
            rec.outcome = "unknown"
        finalize(rec, time.monotonic(), self.ctx.backend.context_length)
        self.ctx.store.insert("requests", rec.row())
        self.ctx.live.on_request_done(rec)
        log.info(
            "%s %s key=%s model=%s stream=%s status=%s outcome=%s in=%s out=%s %.0fms",
            rec.api, rec.endpoint, rec.actor_id, rec.model, int(rec.stream), rec.status_code,
            rec.outcome, rec.prompt_tokens, rec.completion_tokens, rec.end_ms or 0.0,
        )
