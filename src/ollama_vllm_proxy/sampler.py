"""Engine sampler: scrapes vLLM /metrics and the GPU helper, and records the inventory.

Every ``SAMPLE_INTERVAL_S`` the ``vllm:*`` series (without histogram buckets
and without the ``model_name`` label, which can be the weights path), the
helper's ``gpu:*`` series and ``proxy:open_requests`` go to ``engine_samples``
and to the in-memory ring. Every 30 s the static facts (GPU inventory, vLLM
version) are fetched and stored only when they changed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from typing import TYPE_CHECKING, Any

import httpx

from .live import Sample

if TYPE_CHECKING:
    from .app import Context

log = logging.getLogger("ovp.sampler")

INVENTORY_INTERVAL_S = 30.0
_DROP_LABELS = {"model_name"}
_INSERT = "INSERT INTO engine_samples (ts, deployment_id, name, labels, value) VALUES (?, ?, ?, ?, ?)"
_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)\s*(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+)?\s*$")
_LABEL = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')
_ESCAPES = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,63}$")


def _unescape(value: str) -> str:
    return re.sub(r'\\[\\"n]', lambda m: _ESCAPES[m.group(0)], value) if "\\" in value else value


def parse_metrics(text: str, prefix: str = "vllm:") -> list[Sample]:
    """Prometheus text format -> [(name, labels as sorted JSON, value)] for one metric prefix.

    Histogram buckets and ``_created`` timestamps are dropped, as are values
    that are not finite numbers.
    """
    out: list[Sample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        m = _LINE.match(line)
        if m is None:
            continue
        name, raw_labels, raw_value = m.groups()
        if name.endswith(("_bucket", "_created")):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = {k: _unescape(v) for k, v in _LABEL.findall(raw_labels or "") if k not in _DROP_LABELS}
        out.append((name, json.dumps(labels, sort_keys=True), value))
    return out


def split_info(parsed: list[Sample]) -> tuple[list[Sample], dict[str, list[dict[str, str]]]]:
    """Separate ``*_info`` metrics (static configuration carried in labels) from the time series."""
    series: list[Sample] = []
    info: dict[str, list[dict[str, str]]] = {}
    for name, labels, value in parsed:
        if name.endswith("_info"):
            info.setdefault(name, []).append(json.loads(labels))
        else:
            series.append((name, labels, value))
    return series, info


class Sampler:
    def __init__(self, ctx: "Context"):
        self.ctx = ctx
        self._stored: dict[str, tuple[str | None, str]] = {}  # kind -> (deployment label, JSON) last written
        self._task: asyncio.Task | None = None

    # -- inventory ---------------------------------------------------------------

    def load_stored(self) -> None:
        """Remember what is already in the database so a restart does not duplicate rows."""
        rows = self.ctx.store.read(
            "SELECT kind, deployment_id, data FROM inventory WHERE id IN (SELECT MAX(id) FROM inventory GROUP BY kind)"
        )
        for row in rows:
            self._stored[row["kind"]] = (row["deployment_id"], row["data"])

    def record_inventory(self, kind: str, data: Any) -> bool:
        """Show ``data`` on the live page and store it if it differs from the last stored row."""
        ctx = self.ctx
        ctx.live.hardware[kind] = data
        entry = (ctx.backend.label, json.dumps(data, sort_keys=True))
        if self._stored.get(kind) == entry:
            return False
        self._stored[kind] = entry
        ctx.store.insert("inventory", {"ts": time.time(), "deployment_id": entry[0], "kind": kind, "data": entry[1]})
        return True

    async def inventory_once(self) -> None:
        ctx = self.ctx
        backend = ctx.backend
        if backend.gpu_url:
            try:
                r = await ctx.client.get(backend.gpu_url + "/inventory", timeout=20)
                inv = r.json() if r.status_code == 200 else None
                if isinstance(inv, dict):
                    inv.pop("generated_at", None)  # changes on every call; the rest only when something changed
                    self.record_inventory("gpu_inventory", inv)
            except (httpx.HTTPError, ValueError) as exc:
                log.debug("inventory fetch failed: %s", type(exc).__name__)
        else:
            ctx.live.hardware.pop("gpu_inventory", None)
        if backend.ready:
            try:
                r = await ctx.client.get(backend.base_url + "/version", headers=backend.auth_headers(), timeout=10)
                version = str(r.json().get("version", "")) if r.status_code == 200 else ""
                if _VERSION_RE.match(version):
                    self.record_inventory("vllm_version", {"version": version})
            except (httpx.HTTPError, ValueError, AttributeError) as exc:
                log.debug("version fetch failed: %s", type(exc).__name__)
        else:
            for kind in [k for k in ctx.live.hardware if k != "gpu_inventory"]:
                del ctx.live.hardware[kind]

    # -- samples -----------------------------------------------------------------

    async def _scrape(self, url: str, headers: dict[str, str], prefix: str, timeout: float) -> list[Sample]:
        r = await self.ctx.client.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise ValueError(f"http_{r.status_code}")
        return await asyncio.to_thread(parse_metrics, r.text, prefix)

    async def sample_once(self) -> None:
        ctx = self.ctx
        backend, live = ctx.backend, ctx.live
        ts = time.time()
        engine: list[Sample] | None = None
        gpu: list[Sample] | None = None
        if backend.ready:
            try:
                parsed = await self._scrape(backend.base_url + "/metrics", backend.auth_headers(), "vllm:", 5)
                engine, info = split_info(parsed)
                for kind, label_sets in info.items():
                    self.record_inventory(kind, label_sets)
                live.metrics_ok, live.metrics_error = True, None
            except (httpx.HTTPError, ValueError) as exc:
                live.metrics_ok = False
                live.metrics_error = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        else:
            live.metrics_ok = live.metrics_error = None
        if backend.gpu_url:
            try:
                parsed = await self._scrape(backend.gpu_url + "/metrics", {}, "gpu:", 3)
                gpu = [s for s in parsed if s[0] != "gpu:exporter_up"]
                live.gpu_ok = True
            except (httpx.HTTPError, ValueError):
                live.gpu_ok = False
        else:
            live.gpu_ok = None
        if engine is not None or gpu is not None:
            # Nothing is written while there is no backend at all, so an idle install does not grow.
            label = backend.label
            rows = [(ts, label, "proxy:open_requests", "{}", float(ctx.open_requests))]
            rows.extend((ts, label, n, lb, v) for n, lb, v in (engine or []))
            rows.extend((ts, label, n, lb, v) for n, lb, v in (gpu or []))
            ctx.store.submit(_INSERT, rows, many=True)
        live.on_scrape(ts, engine, ctx.open_requests, gpu)

    async def run(self) -> None:
        interval = max(0.1, self.ctx.settings.sample_interval_s)
        next_tick = time.monotonic()
        last_inventory = -math.inf
        was_ready = False
        try:
            await asyncio.to_thread(self.load_stored)
        except Exception as exc:  # the sampler must come up even if this read fails
            log.warning("could not read stored inventory: %s", type(exc).__name__)
        while True:
            try:
                now = time.monotonic()
                ready = self.ctx.backend.ready
                # A backend that just became ready is probed at once, not up to 30 s later.
                if now - last_inventory > INVENTORY_INTERVAL_S or ready != was_ready:
                    last_inventory, was_ready = now, ready
                    await self.inventory_once()
                await self.sample_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("sampler iteration failed: %s: %s", type(exc).__name__, exc)
            next_tick += interval
            delay = next_tick - time.monotonic()
            if delay < 0:  # fell behind (slow scrape): do not burst to catch up
                next_tick = time.monotonic()
                delay = 0
            await asyncio.sleep(delay)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="engine-sampler")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
