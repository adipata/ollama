"""Authentication middleware.

* ``/api/*`` and ``/v1/*``  -> a proxy API key (Bearer or x-api-key)
* ``/admin/api/*``           -> the admin token (Bearer)
* everything else            -> open (liveness, health, static admin pages; none carry data)

The two credentials are not interchangeable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .keys import token_equal

if TYPE_CHECKING:
    from .app import Context


def _header(scope, name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key == name:
            return value.decode("latin-1")
    return ""


def bearer(scope) -> str:
    value = _header(scope, b"authorization")
    if value[:7].lower() == "bearer ":
        return value[7:].strip()
    return ""


def error_body(path: str, status: int, message: str, etype: str = "invalid_request_error") -> bytes:
    """Error in the dialect of the route: OpenAI object under /v1, Ollama string elsewhere."""
    if path.startswith("/v1"):
        code = "invalid_api_key" if status == 401 else None
        return json.dumps({"error": {"message": message, "type": etype, "param": None, "code": code}}).encode()
    return json.dumps({"error": message}).encode()


class AuthMiddleware:
    def __init__(self, app, ctx: "Context"):
        self.app = app
        self.ctx = ctx

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        path: str = scope["path"]
        if path.startswith("/admin/api/"):
            token = bearer(scope)
            if not token or not token_equal(token, self.ctx.admin_token):
                return await self._reject(scope, send, path, "invalid admin token")
        elif path in ("/api", "/v1") or path.startswith(("/api/", "/v1/")):
            token = bearer(scope) or _header(scope, b"x-api-key").strip()
            found = self.ctx.keys.lookup(token)
            if found is None:
                self.ctx.rejected += 1
                return await self._reject(scope, send, path, "invalid or missing API key")
            scope.setdefault("state", {})
            scope["state"]["key_id"], scope["state"]["key_name"] = found
        return await self.app(scope, receive, send)

    async def _reject(self, scope, send, path: str, message: str) -> None:
        body = error_body(path, 401, message, "authentication_error")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
