"""In-memory view of recent activity for the live monitoring page.

Holds about 25 minutes of engine/GPU samples, the last 2,000 finished requests
and the requests currently in flight, so the page can poll every sample
interval without touching SQLite. Longer windows are read from the database
(``export.history``) and passed to :meth:`Live.snapshot`, which shapes both
sources the same way. Metadata only: nothing here ever sees prompt or
completion text.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from typing import Any, Iterable, Sequence

from .observe import RequestRecord

Sample = tuple[str, str, float]  # (series name, labels as sorted JSON, value)

# helper metric -> short field name; a point carries them as g<index>_<field>
GPU_FIELDS = {
    "gpu:utilization_percent": "util",
    "gpu:memory_bandwidth_percent": "mem_bw",
    "gpu:memory_used_mib": "mem_used",
    "gpu:memory_total_mib": "mem_total",
    "gpu:power_watts": "power",
    "gpu:power_limit_watts": "power_limit",
    "gpu:temperature_celsius": "temp",
    "gpu:memory_temperature_celsius": "mem_temp",
    "gpu:sm_clock_mhz": "sm_clock",
    "gpu:memory_clock_mhz": "mem_clock",
    "gpu:pcie_link_gen": "pcie_gen",
    "gpu:pcie_link_width": "pcie_width",
    "gpu:pcie_rx_mb_per_second": "rx",
    "gpu:pcie_tx_mb_per_second": "tx",
}
GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "open": "proxy:open_requests",
}
RATES = {"prompt_tps": "vllm:prompt_tokens_total", "output_tps": "vllm:generation_tokens_total"}
KV_NAMES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
PREEMPTIONS = "vllm:num_preemptions_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
COUNTER_NAMES = (*RATES.values(), PREEMPTIONS, PREFIX_HITS, PREFIX_QUERIES)
SERIES_NAMES = (*GAUGES.values(), *KV_NAMES, *COUNTER_NAMES, *GPU_FIELDS)
# When several samples are merged into one chart point these keep their peak; everything else is averaged.
PEAK_KEYS = {"running", "waiting", "open", "kv"}
PEAK_GPU_FIELDS = {"mem_used", "mem_total", "temp", "mem_temp", "power_limit"}

# Columns of `requests` the page and the per-key statistics need.
REQUEST_COLS = (
    "t_arrival_wall", "actor_id", "session_id", "api", "endpoint", "stream", "status_code", "outcome",
    "finish_reason", "messages_count", "prompt_tokens", "cached_tokens", "completion_tokens",
    "reasoning_tokens", "open_requests_at_start", "first_payload_ms", "first_text_ms", "first_reasoning_ms",
    "first_tool_ms", "end_ms", "max_gap_ms", "decode_tps", "delivered_tps", "context_fraction",
)

RING_SECONDS = 1500.0  # about 25 minutes of samples
MAX_RECENT = 2000
MAX_POINTS = 360  # a chart never gets more points than this
MAX_REQUEST_ROWS = 600  # finished requests sent to the page (scatter and table)


def pct(values: Iterable[float | None], q: float) -> float | None:
    """Nearest-rank percentile, ignoring missing values."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[max(1, math.ceil(q / 100.0 * len(vals))) - 1]


def is_peak(key: str) -> bool:
    return key in PEAK_KEYS or (key[:1] == "g" and key.split("_", 1)[-1] in PEAK_GPU_FIELDS)


def gpu_index(labels: str) -> int | None:
    try:
        return int(json.loads(labels).get("gpu"))
    except (TypeError, ValueError, AttributeError):
        return None


def aggregate(samples: Iterable[Sample]) -> dict[str, float]:
    """One scrape -> {series name: value summed over label sets, "kv": max, "g0_util": ...}."""
    cur: dict[str, float] = {}
    for name, labels, value in samples:
        if value is None or not math.isfinite(value):
            continue
        field = GPU_FIELDS.get(name)
        if field is not None:
            idx = gpu_index(labels)
            if idx is not None:
                cur[f"g{idx}_{field}"] = value
        elif name in KV_NAMES:
            cur["kv"] = max(cur.get("kv", 0.0), value)
        else:
            cur[name] = cur.get(name, 0.0) + value
    return cur


def make_point(
    ts: float, cur: dict[str, float], prev: tuple[float, dict[str, float]] | None, max_gap: float
) -> dict[str, Any]:
    """Chart point from one scrape and the previous one (counters become rates and deltas).

    A counter that went backwards (engine restart) or a gap longer than
    ``max_gap`` yields no rate instead of a misleading one.
    """
    point: dict[str, Any] = {"t": round(ts, 3), "kv": _r(cur.get("kv"))}
    for key, name in GAUGES.items():
        point[key] = _r(cur.get(name))
    deltas: dict[str, float] = {}
    dt = 0.0
    if prev is not None and 0 < ts - prev[0] <= max_gap:
        dt = ts - prev[0]
        for name in COUNTER_NAMES:
            if name in cur and name in prev[1] and cur[name] >= prev[1][name]:
                deltas[name] = cur[name] - prev[1][name]
    for key, name in RATES.items():
        point[key] = _r(deltas[name] / dt) if name in deltas else None
    point["preemptions"] = _r(deltas.get(PREEMPTIONS))
    queries = deltas.get(PREFIX_QUERIES)
    point["prefix_hit"] = round(min(1.0, deltas.get(PREFIX_HITS, 0.0) / queries), 4) if queries else None
    for key, value in cur.items():
        if key[:1] == "g" and key[1:2].isdigit():
            point[key] = _r(value)
    return point


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def downsample(points: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Merge points into at most ``n`` time buckets (peaks kept for queues, cache, memory, temperature)."""
    if len(points) <= n:
        return list(points)
    t0, t1 = points[0]["t"], points[-1]["t"]
    width = (t1 - t0) / n or 1.0
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in points:
        buckets[min(n - 1, int((p["t"] - t0) / width))].append(p)
    out = []
    for i in sorted(buckets):
        group = buckets[i]
        merged: dict[str, Any] = {"t": group[-1]["t"]}
        for key in {k for p in group for k in p} - {"t"}:
            vals = [p[key] for p in group if p.get(key) is not None]
            if not vals:
                merged[key] = None
            elif key == "preemptions":
                merged[key] = sum(vals)
            else:
                merged[key] = _r(max(vals) if is_peak(key) else sum(vals) / len(vals))
        out.append(merged)
    return out


class GpuStats:
    """Running mean/max/min per GPU field, fed from raw samples or from SQL aggregates."""

    def __init__(self) -> None:
        self._acc: dict[tuple[int, str], list[float]] = {}  # (gpu, field) -> [sum, n, max, min]

    def add(self, gpu: int, field: str, mean: float, n: int = 1, hi: float | None = None, lo: float | None = None):
        hi = mean if hi is None else hi
        lo = mean if lo is None else lo
        acc = self._acc.get((gpu, field))
        if acc is None:
            self._acc[(gpu, field)] = [mean * n, n, hi, lo]
        else:
            acc[0] += mean * n
            acc[1] += n
            acc[2] = max(acc[2], hi)
            acc[3] = min(acc[3], lo)

    def add_point(self, point: dict[str, Any]) -> None:
        for key, value in point.items():
            if value is not None and key[:1] == "g" and key[1:2].isdigit():
                idx, _, field = key[1:].partition("_")
                self.add(int(idx), field, value)

    def rows(self, points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Per-GPU summary; the utilization p90 comes from the chart points."""
        out = []
        for gpu in sorted({g for g, _ in self._acc}):

            def stat(field: str, which: int) -> float | None:
                acc = self._acc.get((gpu, field))
                if acc is None:
                    return None
                return _r(acc[0] / acc[1]) if which == 0 else acc[which]

            util = self._acc.get((gpu, "util"))
            out.append({
                "gpu": gpu,
                "samples": int(util[1]) if util else 0,
                "util_mean": stat("util", 0), "util_max": stat("util", 2),
                "util_p90": pct([p.get(f"g{gpu}_util") for p in points], 90),
                "mem_used_max_mib": stat("mem_used", 2), "mem_total_mib": stat("mem_total", 2),
                "power_mean": stat("power", 0), "power_max": stat("power", 2),
                "power_limit": stat("power_limit", 2),
                "temp_max": stat("temp", 2), "sm_clock_min": stat("sm_clock", 3),
                "rx_mean": stat("rx", 0), "rx_max": stat("rx", 2),
                "tx_mean": stat("tx", 0), "tx_max": stat("tx", 2),
            })
        return out


def is_error(outcome: str | None) -> bool:
    return outcome not in ("ok", "client_cancelled")


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts, latency percentiles and token totals for a set of request rows."""

    def col(name: str) -> list[float]:
        return [r[name] for r in rows if r.get(name) is not None]

    known = [r for r in rows if r.get("prompt_tokens") and r.get("cached_tokens") is not None]
    prompt_known = sum(r["prompt_tokens"] for r in known)
    return {
        "n": len(rows),
        "ok": sum(r.get("outcome") == "ok" for r in rows),
        "cancelled": sum(r.get("outcome") == "client_cancelled" for r in rows),
        "errors": sum(is_error(r.get("outcome")) for r in rows),
        "first_p50": pct(col("first_payload_ms"), 50), "first_p90": pct(col("first_payload_ms"), 90),
        "text_p50": pct(col("first_text_ms"), 50), "tool_p50": pct(col("first_tool_ms"), 50),
        "end_p50": pct(col("end_ms"), 50), "end_p90": pct(col("end_ms"), 90),
        "gap_p90": pct(col("max_gap_ms"), 90),
        "decode_p50": pct(col("decode_tps"), 50), "delivered_p50": pct(col("delivered_tps"), 50),
        "prompt_tokens": sum(col("prompt_tokens")), "completion_tokens": sum(col("completion_tokens")),
        "cached_ratio": round(sum(r["cached_tokens"] for r in known) / prompt_known, 4) if prompt_known else None,
        "prompt_max": max(col("prompt_tokens"), default=None),
        "ctx_max": max(col("context_fraction"), default=None),
    }


def per_key(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[r.get("actor_id") or "(none)"].append(r)
    return [{"key": name, **summarize(rs)} for name, rs in sorted(by_key.items())]


def phase(rec: RequestRecord) -> str:
    """What an open request is doing right now, from the timestamps recorded so far."""
    if not rec.stream:
        return "generating (not streamed)"
    if rec.first_payload_ms is None:
        return "queued/prefill"
    if rec.first_tool_ms is not None:
        return "tool call"
    if rec.first_text_ms is not None:
        return "answering"
    return "reasoning"


class Live:
    def __init__(self, ctx: Any = None):
        self.ctx = ctx
        interval = max(0.1, ctx.settings.sample_interval_s) if ctx is not None else 1.5
        self.interval = interval
        self.series: deque[dict[str, Any]] = deque(maxlen=int(min(6000, max(200, RING_SECONDS / interval))))
        self.recent: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT)
        self.open: dict[str, RequestRecord] = {}
        self.hardware: dict[str, Any] = {}  # latest inventory per kind, set by the sampler
        self.metrics_ok: bool | None = None  # None = not scraped (backend not ready)
        self.metrics_error: str | None = None
        self.gpu_ok: bool | None = None  # None = no GPU helper configured
        self._prev: tuple[float, dict[str, float]] | None = None

    # -- feeds -----------------------------------------------------------------

    def on_scrape(
        self,
        ts: float,
        engine: Iterable[Sample] | None,
        open_requests: int,
        gpu: Iterable[Sample] | None = None,
    ) -> None:
        """One sampler tick; ``engine``/``gpu`` are None when that source was not scraped."""
        cur = aggregate([*(engine or ()), *(gpu or ())])
        cur[GAUGES["open"]] = float(open_requests)
        self.series.append(make_point(ts, cur, self._prev, max_gap=max(15.0, self.interval * 10)))
        self._prev = (ts, cur) if engine is not None else None

    def on_request_start(self, rec: RequestRecord) -> None:
        self.open[rec.request_id] = rec

    def on_request_done(self, rec: RequestRecord) -> None:
        self.open.pop(rec.request_id, None)
        row = {name: getattr(rec, name) for name in REQUEST_COLS}
        row["stream"] = int(bool(rec.stream))
        self.recent.append(row)

    # -- snapshot --------------------------------------------------------------

    def covers(self, since: float) -> bool:
        """True when the in-memory ring holds everything that happened after ``since``."""
        if time.time() - since > RING_SECONDS * 0.9 or len(self.recent) == self.recent.maxlen:
            return False
        started = self.ctx.started_at if self.ctx is not None else 0.0
        return started <= since or bool(self.series and self.series[0]["t"] <= since + 2 * self.interval)

    def in_flight(self) -> list[dict[str, Any]]:
        mono = time.monotonic()
        rows = [
            {
                "key": rec.actor_id,
                "session": rec.session_id,
                "api": rec.api,
                "phase": phase(rec),
                "elapsed_ms": round((mono - rec.t0) * 1000.0, 1),
                "first_ms": rec.first_payload_ms,
                "content_events": rec.content_events,
                "messages": rec.messages_count,
                "max_tokens": rec.requested_max_tokens,
                "stream": bool(rec.stream),
            }
            for rec in list(self.open.values())
        ]
        return sorted(rows, key=lambda r: -r["elapsed_ms"])

    def current(self) -> dict[str, Any]:
        """Latest engine and GPU values for the tiles, whatever window the charts show."""
        open_now = self.ctx.open_requests if self.ctx is not None else len(self.open)
        out: dict[str, Any] = {"open": open_now, "running": None, "waiting": None, "kv": None,
                               "output_tps": None, "prompt_tps": None, "gpus": []}
        points = list(self.series)[-3:]
        if not points or time.time() - points[-1]["t"] > max(10.0, self.interval * 5):
            return out
        last = points[-1]
        out.update({k: last.get(k) for k in ("running", "waiting", "kv")})
        for key in RATES:  # mean of the last few samples: single deltas are jumpy
            vals = [p[key] for p in points if p.get(key) is not None]
            out[key] = round(sum(vals) / len(vals), 3) if vals else None
        gpus: dict[int, dict[str, Any]] = {}
        for key, value in last.items():
            if value is not None and key[:1] == "g" and key[1:2].isdigit():
                idx, _, field = key[1:].partition("_")
                gpus.setdefault(int(idx), {"gpu": int(idx)})[field] = value
        out["gpus"] = [gpus[i] for i in sorted(gpus)]
        return out

    def snapshot(self, window_s: float, history: dict[str, Any] | None = None) -> dict[str, Any]:
        """Everything the live page shows for the last ``window_s`` seconds.

        ``history`` (from ``export.history``) replaces the in-memory series and
        requests for windows the ring does not cover.
        """
        now = time.time()
        since = now - window_s
        if history is None:
            raw = [p for p in self.series if p["t"] >= since]
            stats = GpuStats()
            for p in raw:
                stats.add_point(p)
            series = downsample(raw, MAX_POINTS)
            gpus = stats.rows(series)
            requests = [r for r in self.recent if r["t_arrival_wall"] >= since]
            source = "memory"
        else:
            series, gpus, requests, source = history["series"], history["gpus"], history["requests"], "database"

        in_flight = self.in_flight()
        keys = per_key(requests)
        flying: dict[str, int] = defaultdict(int)
        for f in in_flight:
            flying[f["key"] or "(none)"] += 1
        for k in keys:
            k["in_flight"] = flying.get(k["key"], 0)

        def series_max(key: str) -> float | None:
            return max((p[key] for p in series if p.get(key) is not None), default=None)

        def series_mean(key: str) -> float | None:
            vals = [p[key] for p in series if p.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        total = summarize(requests)
        ctx = self.ctx
        return {
            "now": now,
            "window_s": window_s,
            "source": source,
            "sample_interval_s": self.interval,
            "retention_days": ctx.settings.retention_days if ctx is not None else 0,
            "backend": ctx.backend.describe() if ctx is not None else None,
            "metrics_ok": self.metrics_ok,
            "metrics_error": self.metrics_error,
            "gpu_ok": self.gpu_ok,
            "store": {"queued": ctx.store.queued, "dropped": ctx.store.dropped, "failed": ctx.store.failed}
            if ctx is not None else None,
            "rejected": ctx.rejected if ctx is not None else 0,
            "current": self.current(),
            "series": series,
            "gpus": gpus,
            "hardware": self.hardware,
            "in_flight": in_flight,
            "keys": keys,
            "requests": list(requests[-MAX_REQUEST_ROWS:]),
            "totals": {
                "completed": total["n"], "ok": total["ok"], "cancelled": total["cancelled"],
                "errors": total["errors"], "prompt_tokens": total["prompt_tokens"],
                "completion_tokens": total["completion_tokens"],
                "preemptions": round(sum(p.get("preemptions") or 0 for p in series)),
                "running_max": series_max("running"), "waiting_max": series_max("waiting"),
                "kv_max": series_max("kv"), "output_tps_mean": series_mean("output_tps"),
                "prompt_tps_mean": series_mean("prompt_tps"),
            },
        }
