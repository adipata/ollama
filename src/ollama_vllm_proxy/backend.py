"""The upstream vLLM backend: where it is, whether it is usable, what it serves.

In *external mode* (``VLLM_BASE_URL`` set) the backend is a fixed URL that is
assumed to be up. In *managed mode* the pod deployment code moves the backend
through its states and calls :meth:`Backend.set_ready` once vLLM answers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger("ovp.backend")

# States of the managed backend, and the message clients get while it is not ready.
STATE_MESSAGES = {
    "no_pod": "backend not deployed: no pod connected",
    "connected": "backend not deployed: pod connected, nothing deployed yet",
    "deploying": "backend is deploying",
    "stopped": "backend stopped",
    "failed": "backend deployment failed",
    "tunnel_down": "backend unreachable: SSH tunnel is down",
}


class BackendUnavailable(Exception):
    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.message = message
        self.status = status


class Backend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.external = settings.external_mode
        self.state = "ready" if self.external else "no_pod"
        self.detail = ""
        self.base_url = settings.vllm_base_url
        self.api_key = settings.vllm_api_key
        self.served_model = settings.default_model
        self.context_length: int | None = settings.context_length
        self.label: str | None = None  # deployment label recorded with every request
        self.gpu_url = ""  # GPU helper base URL ("" = none)
        if self.external and settings.gpu_exporter_url.lower() not in ("", "off"):
            self.gpu_url = settings.gpu_exporter_url.rstrip("/")
        self.ready_since: float | None = time.time() if self.external else None
        self._models: list[dict[str, Any]] = []
        self._models_at = 0.0

    # -- state (managed mode) ---------------------------------------------------

    def set_state(self, state: str, detail: str = "") -> None:
        if state != self.state:
            log.info("backend state: %s -> %s %s", self.state, state, detail)
        self.state, self.detail = state, detail
        if state != "ready":
            self.ready_since = None

    def set_ready(
        self,
        *,
        base_url: str,
        api_key: str,
        served_model: str,
        context_length: int | None,
        label: str | None = None,
        gpu_url: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.served_model = served_model
        self.context_length = context_length
        self.label = label
        self.gpu_url = gpu_url.rstrip("/")
        self._models_at = 0.0
        self.ready_since = time.time()
        self.set_state("ready")

    def set_endpoints(self, *, base_url: str = "", gpu_url: str = "") -> None:
        """Update tunnel endpoints without changing the state (reconnects, helper start)."""
        if base_url:
            self.base_url = base_url.rstrip("/")
        self.gpu_url = gpu_url.rstrip("/")

    @property
    def ready(self) -> bool:
        return self.state == "ready" and bool(self.base_url)

    def require_ready(self) -> None:
        if not self.ready:
            message = STATE_MESSAGES.get(self.state, f"backend not ready ({self.state})")
            if self.detail:
                message = f"{message} ({self.detail})"
            raise BackendUnavailable(message)

    def auth_headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "external" if self.external else "managed",
            "state": self.state,
            "detail": self.detail,
            "served_model": self.served_model or None,
            "context_length": self.context_length,
            "label": self.label,
            "ready_since": self.ready_since,
            "gpu_helper": bool(self.gpu_url),
        }

    # -- model list -------------------------------------------------------------

    async def models(self, client: httpx.AsyncClient, max_age: float = 10.0) -> list[dict[str, Any]]:
        """Upstream /v1/models, cached briefly. Fills in served model and context length."""
        self.require_ready()
        if self._models and time.monotonic() - self._models_at < max_age:
            return self._models
        try:
            resp = await client.get(self.base_url + "/v1/models", headers=self.auth_headers(), timeout=10)
        except httpx.HTTPError as exc:
            raise BackendUnavailable(f"upstream unavailable ({type(exc).__name__})", 502) from None
        if resp.status_code != 200:
            raise BackendUnavailable(f"upstream returned {resp.status_code} for the model list", 502)
        try:
            data = resp.json().get("data") or []
        except ValueError:
            data = []
        self._models = [m for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]
        self._models_at = time.monotonic()
        if self._models:
            ids = [m["id"] for m in self._models]
            if not self.served_model or self.served_model not in ids:
                self.served_model = ids[0]
            if self.context_length is None:
                for m in self._models:
                    if m["id"] == self.served_model and isinstance(m.get("max_model_len"), int):
                        self.context_length = m["max_model_len"]
        return self._models


def resolve_model(backend: Backend, settings: Settings, name: Any, known: list[str]) -> str:
    """Map the model name a client sent to the served model id.

    Strips ``:latest``, applies aliases, then either rejects unknown names
    (STRICT_MODELS) or falls back to the served model.
    """
    default = backend.served_model or (known[0] if known else "")
    if not isinstance(name, str) or not name.strip():
        if default:
            return default
        raise BackendUnavailable("no model available", 503)
    name = name.strip()
    candidates = [name]
    if name.endswith(":latest"):
        candidates.append(name[: -len(":latest")])
    for cand in list(candidates):
        if cand in settings.model_aliases:
            candidates.append(settings.model_aliases[cand])
    for cand in candidates:
        if cand in known or cand == backend.served_model:
            return cand
    if settings.strict_models or not default:
        raise BackendUnavailable(f"model '{name}' not found", 404)
    return default
