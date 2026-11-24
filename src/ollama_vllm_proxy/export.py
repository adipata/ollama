"""Monitoring history from SQLite, and the downloadable report built from it.

A report covers a time range, optionally narrowed to one API key or one
``X-Session-Id``. The HTML is one self-contained file: inline CSS and
server-rendered SVG charts, no scripts and no external requests, so it opens
offline and prints cleanly. The full data set is embedded as JSON in
``<script type="application/json" id="report-data">`` (a data block, never
executed); ``--format json`` returns only that data.

    ollama-vllm-proxy export --last 2h -o report.html
    ollama-vllm-proxy export --from 2026-11-24T09:00 --to 2026-11-24T12:00 --key laptop --format json
    ollama-vllm-proxy export --list
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Sequence

from .live import (
    COUNTER_NAMES,
    GPU_FIELDS,
    MAX_POINTS,
    REQUEST_COLS,
    SERIES_NAMES,
    GpuStats,
    aggregate,
    gpu_index,
    is_peak,
    make_point,
    per_key,
    summarize,
)

Read = Callable[..., list[dict[str, Any]]]  # Store.read

FORMAT = "ovp-report/1"
REPORT_POINTS = 720  # engine samples are merged down to this many chart points
MAX_HTML_ROWS = 5000  # requests listed in the HTML table; the embedded JSON always has all of them
_GAUGE_KEY = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
              "proxy:open_requests": "open", "vllm:kv_cache_usage_perc": "kv", "vllm:gpu_cache_usage_perc": "kv"}


class ExportError(ValueError):
    pass


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def engine_history(
    read: Read, t0: float, t1: float, max_points: int = MAX_POINTS
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Engine and GPU chart points for a time range, plus the per-GPU summary.

    The samples are merged into time buckets by SQLite, so a 30-day range costs
    one index scan and never materialises millions of rows in Python. Counters
    keep their last value per bucket; rates are taken between buckets.
    """
    width = max(t1 - t0, 1e-3) / max(1, max_points)
    marks = ",".join("?" * len(SERIES_NAMES))
    rows = read(
        "SELECT CAST((ts - ?) / ? AS INTEGER) AS b, name, labels, AVG(value) AS mean, MAX(value) AS hi, "
        "MIN(value) AS lo, COUNT(*) AS n, MAX(ts) AS ts FROM engine_samples "
        f"WHERE ts >= ? AND ts <= ? AND value IS NOT NULL AND name IN ({marks}) "
        "GROUP BY b, name, labels ORDER BY b",
        (t0, width, t0, t1, *SERIES_NAMES),
    )
    stats = GpuStats()
    buckets: dict[int, tuple[float, list[tuple[str, str, float]]]] = {}
    for row in rows:
        name = row["name"]
        field = GPU_FIELDS.get(name)
        if field is not None:
            idx = gpu_index(row["labels"])
            if idx is not None:
                stats.add(idx, field, row["mean"], row["n"], row["hi"], row["lo"])
            peak = is_peak(f"g0_{field}")
        else:
            peak = name in COUNTER_NAMES or is_peak(_GAUGE_KEY.get(name, ""))
        ts, samples = buckets.get(row["b"], (0.0, []))
        samples.append((name, row["labels"], row["hi"] if peak else row["mean"]))
        buckets[row["b"]] = (max(ts, row["ts"]), samples)
    points: list[dict[str, Any]] = []
    prev: tuple[float, dict[str, float]] | None = None
    for b in sorted(buckets):
        ts, samples = buckets[b]
        cur = aggregate(samples)
        points.append(make_point(ts, cur, prev, max_gap=max(60.0, width * 5)))
        if any(name in cur for name in COUNTER_NAMES):
            prev = (ts, cur)
    return points, stats.rows(points)


def load_requests(
    read: Read, t0: float, t1: float, key: str | None = None, session: str | None = None, cols: str = "*"
) -> list[dict[str, Any]]:
    sql = f"SELECT {cols} FROM requests WHERE t_arrival_wall >= ? AND t_arrival_wall <= ?"
    params: list[Any] = [t0, t1]
    if key:
        # The row holds the key name at request time; also match the key's current name after a rename.
        sql += " AND (actor_id = ? OR key_id IN (SELECT id FROM api_keys WHERE name = ?))"
        params += [key, key]
    if session:
        sql += " AND session_id = ?"
        params.append(session)
    return read(sql + " ORDER BY t_arrival_wall", params)


def history(read: Read, t0: float, t1: float, max_points: int = MAX_POINTS) -> dict[str, Any]:
    """What ``Live.snapshot`` needs for a window the in-memory ring does not cover."""
    series, gpus = engine_history(read, t0, t1, max_points)
    return {"series": series, "gpus": gpus, "requests": load_requests(read, t0, t1, cols=", ".join(REQUEST_COLS))}


def options(read: Read, t0: float | None = None, t1: float | None = None) -> dict[str, Any]:
    """Keys, sessions and deployment labels that have data, and the time span covered (for the picker)."""
    lo, hi = (0.0 if t0 is None else t0), (1e18 if t1 is None else t1)
    where, params = "t_arrival_wall >= ? AND t_arrival_wall <= ?", (lo, hi)
    keys = read(
        "SELECT actor_id AS key, COUNT(*) AS requests, MIN(t_arrival_wall) AS first, MAX(t_arrival_wall) AS last "
        f"FROM requests WHERE {where} AND actor_id IS NOT NULL GROUP BY actor_id ORDER BY actor_id", params)
    sessions = read(
        "SELECT session_id AS session, COUNT(*) AS requests, GROUP_CONCAT(DISTINCT actor_id) AS keys, "
        "MIN(t_arrival_wall) AS first, MAX(t_arrival_wall) AS last "
        f"FROM requests WHERE {where} AND session_id IS NOT NULL GROUP BY session_id ORDER BY last DESC LIMIT 200",
        params)
    labels: dict[str, dict[str, Any]] = {}
    for row in read(
        "SELECT deployment_id AS label, COUNT(*) AS requests, MIN(t_arrival_wall) AS first, "
        f"MAX(t_arrival_wall) AS last FROM requests WHERE {where} AND deployment_id IS NOT NULL "
        "GROUP BY deployment_id", params):
        labels[row["label"]] = row
    for row in read(
        "SELECT deployment_id AS label, MIN(ts) AS first, MAX(ts) AS last FROM engine_samples "
        "WHERE name = 'proxy:open_requests' AND ts >= ? AND ts <= ? AND deployment_id IS NOT NULL "
        "GROUP BY deployment_id", params):
        cur = labels.setdefault(row["label"], {"label": row["label"], "requests": 0, "first": row["first"],
                                               "last": row["last"]})
        cur["first"], cur["last"] = min(cur["first"], row["first"]), max(cur["last"], row["last"])
    spans = read(
        f"SELECT MIN(t_arrival_wall) AS first, MAX(t_arrival_wall) AS last, COUNT(*) AS n FROM requests WHERE {where}",
        params)[0]
    samples = read("SELECT MIN(ts) AS first, MAX(ts) AS last FROM engine_samples WHERE ts >= ? AND ts <= ?", params)[0]
    firsts = [v for v in (spans["first"], samples["first"]) if v is not None]
    lasts = [v for v in (spans["last"], samples["last"]) if v is not None]
    return {
        "span": {"first": min(firsts, default=None), "last": max(lasts, default=None)},
        "requests": spans["n"],
        "unnamed_session_requests": read(
            f"SELECT COUNT(*) AS n FROM requests WHERE {where} AND session_id IS NULL", params)[0]["n"],
        "keys": keys,
        "sessions": sessions,
        "deployments": sorted(labels.values(), key=lambda r: r["last"], reverse=True),
    }


def build_report(
    read: Read,
    t0: float,
    t1: float,
    key: str | None = None,
    session: str | None = None,
    *,
    served_model: str | None = None,
    context_length: int | None = None,
    proxy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The report data set for ``[t0, t1]`` (epoch seconds).

    ``key`` and ``session`` narrow the requests only: the engine and GPU series
    are engine-wide and still show the load caused by everyone.
    """
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        raise ExportError("the time range is empty: 'to' must be later than 'from'")
    reqs = load_requests(read, t0, t1, key, session)
    for r in reqs:
        try:
            r["srv_metrics"] = json.loads(r.pop("srv_metrics_json", None) or "null")
        except ValueError:
            r["srv_metrics"] = None

    # Charts span the part of the range that has data, so a wide range with a short burst stays readable.
    first_req = reqs[0]["t_arrival_wall"] if reqs else None
    last_req = max(r["t_arrival_wall"] + (r["end_ms"] or 0) / 1000.0 for r in reqs) if reqs else None
    lo, hi = first_req, last_req
    if not (key or session) or not reqs:
        ext = read("SELECT MIN(ts) AS first, MAX(ts) AS last FROM engine_samples WHERE ts >= ? AND ts <= ?",
                   (t0, t1))[0]
        if ext["first"] is not None:
            lo = ext["first"] if lo is None else min(lo, ext["first"])
            hi = ext["last"] if hi is None else max(hi, ext["last"])
    if lo is None or hi is None:
        e0, e1 = t0, t1
    else:
        pad = max(10.0, (hi - lo) * 0.02)
        e0, e1 = max(t0, lo - pad), min(t1, hi + pad)
        if e1 - e0 < 1.0:
            e0, e1 = e0 - 30.0, e1 + 30.0
    engine, gpus = engine_history(read, e0, e1, REPORT_POINTS)

    labels = [r["deployment_id"] for r in reqs if r.get("deployment_id")]
    labels += [row["deployment_id"] for row in read(
        "SELECT deployment_id, MAX(ts) AS last FROM engine_samples WHERE name = 'proxy:open_requests' "
        "AND ts >= ? AND ts <= ? AND deployment_id IS NOT NULL GROUP BY deployment_id ORDER BY last", (e0, e1))]
    deployments = list(dict.fromkeys(labels))
    last_label = deployments[-1] if deployments else None
    models = Counter(r["model"] for r in reqs if r.get("model"))
    model = served_model or (models.most_common(1)[0][0] if models else None)

    hardware: dict[str, Any] = {}
    for row in read(
        "SELECT kind, data FROM inventory WHERE id IN "
        "(SELECT MAX(id) FROM inventory WHERE ts <= ? AND deployment_id IS ? GROUP BY kind)", (e1 + 60.0, last_label)):
        try:
            hardware[row["kind"]] = json.loads(row["data"])
        except ValueError:
            pass

    def col(name: str) -> list[float]:
        return [p[name] for p in engine if p.get(name) is not None]

    def mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    ctx_lens = [r["served_max_model_len"] for r in reqs if r.get("served_max_model_len")]
    return {
        "format": FORMAT,
        "generated_at": time.time(),
        "title": ", ".join(deployments) or model or "Proxy report",
        "deployments": deployments,
        "served_model": model,
        "context_length": max(ctx_lens) if ctx_lens else context_length,
        "filter": {"key": key or None, "session": session or None},
        "range": {"from": t0, "to": t1},
        "window": {"start": e0, "end": e1, "duration_s": e1 - e0, "first_request": first_req,
                   "last_request": last_req},
        "summary": {
            **summarize(reqs),
            "running_max": max(col("running"), default=None), "waiting_max": max(col("waiting"), default=None),
            "open_max": max(col("open"), default=None), "kv_max": max(col("kv"), default=None),
            "preemptions": round(sum(col("preemptions"))),
            "output_tps_mean": mean(col("output_tps")), "output_tps_max": max(col("output_tps"), default=None),
            "prompt_tps_mean": mean(col("prompt_tps")), "prompt_tps_max": max(col("prompt_tps"), default=None),
        },
        "keys": per_key(reqs),
        "engine": engine,
        "gpus": gpus,
        "hardware": hardware,
        "proxy": proxy,
        "requests": reqs,
    }


def filename(report: dict[str, Any], ext: str) -> str:
    stamp = lambda ts: time.strftime("%Y%m%dT%H%MZ", time.gmtime(ts))  # noqa: E731
    f = report["filter"]
    parts = ["ovp-report", report["title"][:80], f"{stamp(report['window']['start'])}-{stamp(report['window']['end'])}"]
    parts += [x for x in (f["key"], f["session"]) if x]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "_".join(parts)) + "." + ext


def to_json(report: dict[str, Any], indent: int | None = None) -> str:
    return json.dumps(report, indent=indent, separators=None if indent else (",", ":"), default=str)


# --------------------------------------------------------------------------
# HTML rendering (no scripts; every value goes through html.escape)
# --------------------------------------------------------------------------

_E = html.escape
DASH = "–"
_COLORS = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6"]
_OTHER = "--text-muted"


def _ms(v: float | None) -> str:
    if v is None:
        return DASH
    if v < 1000:
        return f"{v:.0f} ms"
    if v < 10_000:
        return f"{v / 1000:.2f} s"
    if v < 120_000:
        return f"{v / 1000:.1f} s"
    return f"{int(v // 60000)}m {round((v % 60000) / 1000)}s"


def _int(v: float | None) -> str:
    return DASH if v is None else f"{round(v):,}"


def _num(v: float | None, d: int = 1) -> str:
    return DASH if v is None else f"{v:,.{d}f}"


def _pctf(v: float | None, d: int = 0) -> str:
    return DASH if v is None else f"{v * 100:.{d}f}%"


def _dur(s: float | None) -> str:
    if s is None:
        return DASH
    s = int(round(s))
    days, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    if days:
        return f"{days}d {h:02d}h"
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


class _Clock:
    """Renders times in the viewer's zone, given as minutes east of UTC."""

    def __init__(self, tz_offset_min: int):
        self.off = tz_offset_min * 60
        sign = "+" if tz_offset_min >= 0 else "-"
        a = abs(tz_offset_min)
        self.label = "UTC" if tz_offset_min == 0 else f"UTC{sign}{a // 60:02d}:{a % 60:02d}"

    def t(self, ts: float | None, fmt: str = "%H:%M:%S") -> str:
        return DASH if ts is None else time.strftime(fmt, time.gmtime(ts + self.off))

    def dt(self, ts: float | None) -> str:
        return self.t(ts, "%Y-%m-%d %H:%M:%S")


def _nice(max_v: float, count: int = 4) -> tuple[float, list[float]]:
    if not max_v > 0:
        max_v = 1.0
    raw = max_v / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(f * mag for f in (1, 2, 2.5, 5, 10) if f * mag >= raw)
    top = math.ceil(max_v / step - 1e-9) * step
    return top, [round(i * step, 10) for i in range(int(round(top / step)) + 1)]


_STEPS = (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 172800, 604800)


def _time_ticks(t0: float, t1: float, clock: _Clock, count: int = 7) -> tuple[list[float], str]:
    span = t1 - t0
    step = next((s for s in _STEPS if span / s <= count), 1209600)
    first = math.ceil((t0 + clock.off) / step) * step - clock.off
    out, t = [], first
    while t <= t1:
        out.append(t)
        t += step
    fmt = "%H:%M:%S" if step < 60 else ("%H:%M" if span <= 86400 else ("%m-%d %H:%M" if step < 86400 else "%m-%d"))
    return out, fmt


_W_HALF = 680  # line charts sit two per row; draw near their on-screen size so text stays legible
_W_FULL = 1400


def _line_chart(title: str, sub: str, engine: list[dict], series: list[tuple[str, str, str]],
                t0: float, t1: float, clock: _Clock, y_max: float | None = None, min_top: float = 1.0,
                y_fmt=lambda v: _num(v, 0 if float(v).is_integer() else 1), scale: float = 1.0) -> str:
    W, H = _W_HALF, 220
    multi = len(series) > 1
    ml, mr, mt, mb = 52, (100 if multi else 14), 10, 26
    pw, ph = W - ml - mr, H - mt - mb
    span = (t1 - t0) or 1.0
    x = lambda t: ml + (t - t0) / span * pw  # noqa: E731
    vmax = max((p[k] * scale for p in engine for k, _, _ in series if p.get(k) is not None), default=0.0)
    top, ticks = _nice(y_max if y_max is not None else max(vmax, min_top))
    y = lambda v: mt + ph - (min(v, top) / top) * ph  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_E(title)}">']
    for tv in ticks:
        cls = "baseline" if tv == 0 else "gridline"
        out.append(f'<line class="{cls}" x1="{ml}" x2="{W - mr}" y1="{y(tv):.1f}" y2="{y(tv):.1f}"/>'
                   f'<text class="tick" x="{ml - 8}" y="{y(tv) + 4:.1f}" text-anchor="end">{_E(y_fmt(tv))}</text>')
    tticks, tfmt = _time_ticks(t0, t1, clock)
    for tt in tticks:
        out.append(f'<text class="tick" x="{x(tt):.1f}" y="{H - 6}" text-anchor="middle">{clock.t(tt, tfmt)}</text>')
    has_data = any(p.get(k) is not None for p in engine for k, _, _ in series)
    if not has_data:
        out.append(f'<text class="empty" x="{ml + pw / 2}" y="{mt + ph / 2}" text-anchor="middle">'
                   "No samples in this range</text>")
    # Break lines across gaps (proxy restart, tunnel down) instead of drawing across them.
    dts = sorted(b["t"] - a["t"] for a, b in zip(engine, engine[1:]))
    gap = (dts[len(dts) // 2] * 5) if dts else math.inf
    ends = []
    for key, label, color in series:
        segments: list[list[tuple[float, float]]] = []
        pen, last, prev_t = False, None, None
        for p in engine:
            v = p.get(key)
            if v is None or (prev_t is not None and p["t"] - prev_t > gap):
                pen = False
            if v is not None:
                if not pen:
                    segments.append([])
                segments[-1].append((x(p["t"]), y(v * scale)))
                pen, last = True, v * scale
            prev_t = p["t"]
        d = "".join("M" + "L".join(f"{px:.1f},{py:.1f}" for px, py in seg) for seg in segments if len(seg) > 1)
        if d:
            out.append(f'<path d="{d}" fill="none" stroke="var({color})" stroke-width="2" '
                       'stroke-linejoin="round" stroke-linecap="round"/>')
        for seg in segments:  # a sample with a gap on both sides would otherwise be invisible
            if len(seg) == 1:
                out.append(f'<circle cx="{seg[0][0]:.1f}" cy="{seg[0][1]:.1f}" r="2" fill="var({color})"/>')
        if multi and last is not None:
            ends.append([y(last), label])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 14:
            ends[i][0] = ends[i - 1][0] + 14
    for ey, label in ends:
        out.append(f'<text class="dlabel" x="{W - mr + 8}" y="{ey + 4:.1f}">{_E(label)}</text>')
    out.append("</svg>")
    legend = ""
    if multi:
        legend = '<div class="legend">' + "".join(
            f'<span><span class="key-line" style="background:var({c})"></span>{_E(lbl)}</span>'
            for _, lbl, c in series) + "</div>"
    return (f'<section class="card"><div class="card-head"><h2>{_E(title)}</h2><span class="sub">{_E(sub)}</span>'
            f'</div>{legend}<div class="plot">{"".join(out)}</div></section>')


def key_colors(keys: Sequence[dict[str, Any]]) -> dict[str, str]:
    """A colour per key for the scatter; beyond six keys the least active ones share a muted colour."""
    ranked = sorted(keys, key=lambda k: (-k["n"], k["key"]))
    limit = len(_COLORS) if len(ranked) <= len(_COLORS) else len(_COLORS) - 1
    return {k["key"]: (_COLORS[i] if i < limit else _OTHER) for i, k in enumerate(ranked)}


def _scatter(reqs: list[dict], keys: list[dict], t0: float, t1: float, clock: _Clock) -> str:
    W, H = _W_FULL, 260
    ml, mr, mt, mb = 64, 16, 10, 26
    pw, ph = W - ml - mr, H - mt - mb
    span = (t1 - t0) or 1.0
    pts = [r for r in reqs if r.get("first_payload_ms") and r["first_payload_ms"] > 0]
    lo = min([100.0] + [r["first_payload_ms"] for r in pts])
    hi = max([10_000.0] + [r["first_payload_ms"] for r in pts])
    l0, l1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    x = lambda t: ml + (t - t0) / span * pw  # noqa: E731
    y = lambda v: mt + ph - (math.log10(v) - l0) / (l1 - l0) * ph  # noqa: E731
    colors = key_colors(keys)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Time to first content per request">']
    for e in range(l0, l1 + 1):
        v = 10.0 ** e
        out.append(f'<line class="{"baseline" if e == l0 else "gridline"}" x1="{ml}" x2="{W - mr}" y1="{y(v):.1f}" '
                   f'y2="{y(v):.1f}"/><text class="tick" x="{ml - 8}" y="{y(v) + 4:.1f}" text-anchor="end">'
                   f'{_ms(v)}</text>')
    tticks, tfmt = _time_ticks(t0, t1, clock, 12)
    for tt in tticks:
        out.append(f'<text class="tick" x="{x(tt):.1f}" y="{H - 6}" text-anchor="middle">{clock.t(tt, tfmt)}</text>')
    if not pts:
        out.append(f'<text class="empty" x="{ml + pw / 2}" y="{mt + ph / 2}" text-anchor="middle">'
                   "No requests with first content</text>")
    for r in pts:
        name = r.get("actor_id") or "(none)"
        color = colors.get(name, _OTHER)
        cx, cy = x(r["t_arrival_wall"]), y(r["first_payload_ms"])
        tip = (f'{clock.t(r["t_arrival_wall"])} · {name}'
               f'{" · " + r["session_id"] if r.get("session_id") else ""}\n'
               f'first content {_ms(r["first_payload_ms"])} · end {_ms(r["end_ms"])}\n'
               f'prompt {_int(r["prompt_tokens"])} (cached {_int(r["cached_tokens"])}) · '
               f'output {_int(r["completion_tokens"])}\noutcome {r["outcome"] or DASH}')
        attrs = (f'r="4" fill="var({color})" stroke="var(--surface-1)" stroke-width="1.5"' if r["outcome"] == "ok"
                 else f'r="3.5" fill="var(--surface-1)" stroke="var({color})" stroke-width="2"')
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" {attrs}><title>{_E(tip)}</title></circle>')
    out.append("</svg>")
    legend = "".join(f'<span><span class="key-dot" style="background:var({c})"></span>{_E(k)}</span>'
                     for k, c in colors.items() if c != _OTHER)
    if _OTHER in colors.values():
        legend += f'<span><span class="key-dot" style="background:var({_OTHER})"></span>Other keys</span>'
    return ('<section class="card"><div class="card-head"><h2>Time to first content per request</h2>'
            '<span class="sub">proxy clock, log scale; one dot per request, coloured by API key</span></div>'
            f'<div class="legend">{legend}<span><span class="key-hollow"></span>Not completed OK</span></div>'
            f'<div class="plot">{"".join(out)}</div></section>')


def _table(headers: list[str], rows: list[list[str]], text_cols: Sequence[int] = (), empty: str = "None.") -> str:
    if not rows:
        return f'<div class="none">{_E(empty)}</div>'
    th = "".join(f'<th{" class=text" if i in text_cols else ""}>{_E(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td{" class=text" if i in text_cols else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _outcome(o: str | None) -> str:
    icon = ("good", "●") if o == "ok" else (("warning", "▲") if o == "client_cancelled" else ("critical", "✕"))
    return f'<span class="icon {icon[0]}" aria-hidden="true">{icon[1]}</span>{_E(o or DASH)}'


def _tile(label: str, value: str, sub: str = "", unit: str = "", alert: bool = False) -> str:
    flag = '<span class="flag"><span class="icon" aria-hidden="true">✕ </span>attention</span>' if alert else ""
    return (f'<div class="tile{" alert" if alert else ""}"><div class="label">{_E(label)}</div>'
            f'<div class="value">{_E(value)}{f"<span class=unit>{_E(unit)}</span>" if unit else ""}{flag}</div>'
            f'<div class="sub">{_E(sub) or "&nbsp;"}</div></div>')


_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--wash:rgba(11,11,11,.05);--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#8a5cd6;--series-5:#c9a00c;--series-6:#d6458b;
--good:#0ca30c;--warning:#b97f00;--critical:#d03b3b;--font:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;
--text-secondary:#c3c2b7;--text-muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--wash:rgba(255,255,255,.06);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#9d74e6;--series-5:#d8b030;--series-6:#e2609d;
--good:#3cc23c;--warning:#fab219;--critical:#ef6a6a}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--wash:rgba(255,255,255,.06);--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#9d74e6;--series-5:#d8b030;--series-6:#e2609d;
--good:#3cc23c;--warning:#fab219;--critical:#ef6a6a}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--page);color:var(--text-primary);font:14px/1.45 var(--font)}
.wrap{max-width:1440px;margin:0 auto;padding:20px 16px 48px}
header{margin-bottom:16px}h1{font-size:22px;font-weight:600;margin:0 0 4px;overflow-wrap:anywhere}h2{font-size:14px;font-weight:600;margin:0}
.meta{color:var(--text-secondary);font-size:13px;overflow-wrap:anywhere}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 24px;margin:12px 0 0;font-size:13px}
.facts div{display:flex;gap:8px;min-width:0}.facts dt{color:var(--text-secondary);flex:0 0 auto}.facts dd{margin:0;overflow-wrap:anywhere}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.tile,.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:10px}
.tile{padding:12px 14px;min-width:0}.tile .label{color:var(--text-secondary);font-size:12px}
.tile .value{font-size:24px;font-weight:600;line-height:1.2;margin-top:2px;white-space:nowrap}
.tile .value .unit{font-size:13px;font-weight:400;color:var(--text-secondary);margin-left:3px}
.tile .flag{font-size:12px;font-weight:400;color:var(--text-secondary);margin-left:6px}.tile .flag .icon{color:var(--critical)}
.tile .sub{color:var(--text-muted);font-size:12px;margin-top:2px}.tile.alert{border-color:var(--critical)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px;margin-bottom:12px}
.card{padding:14px 16px 12px;min-width:0;margin-bottom:12px}.grid .card{margin-bottom:0}
.card-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;margin-bottom:8px}.card-head .sub{color:var(--text-muted);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--text-secondary);margin:0 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.key-line{width:14px;height:2px;border-radius:1px;display:inline-block}.key-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.key-hollow{width:8px;height:8px;border-radius:50%;display:inline-block;border:2px solid var(--text-muted)}
.plot svg{display:block;width:100%;height:auto}.plot text{font-family:var(--font)}
.tick{fill:var(--text-muted);font-size:11px;font-variant-numeric:tabular-nums}
.gridline{stroke:var(--grid);stroke-width:1;shape-rendering:crispEdges}.baseline{stroke:var(--axis);stroke-width:1;shape-rendering:crispEdges}
.dlabel{fill:var(--text-secondary);font-size:12px}.empty{fill:var(--text-muted);font-size:13px}
details{margin-top:6px}summary{cursor:pointer;color:var(--text-secondary);font-size:13px;width:fit-content}
.tablewrap{overflow-x:auto;margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--grid)}
th{color:var(--text-secondary);font-weight:600;font-size:12px;background:var(--surface-1)}
td{font-variant-numeric:tabular-nums}th:first-child,td:first-child,td.text,th.text{text-align:left}
tbody tr:hover{background:var(--wash)}td .icon{margin-right:4px}
.icon.critical{color:var(--critical)}.icon.warning{color:var(--warning)}.icon.good{color:var(--good)}
.none{color:var(--text-muted);font-size:13px;padding:6px 0}
.note{color:var(--text-muted);font-size:12px;margin-top:24px}
.hw-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:8px 24px;margin-bottom:6px}
.hw-grid .facts{grid-template-columns:1fr;margin:0}
h3{font-size:13px;font-weight:600;margin:14px 0 0;color:var(--text-secondary)}
pre.raw{font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--page);border:1px solid var(--grid);
border-radius:8px;padding:8px 10px;overflow-x:auto;white-space:pre;margin:6px 0 0;color:var(--text-secondary)}
@media print{:root,:root[data-theme]{color-scheme:light;--page:#fff;--surface-1:#fff;--text-primary:#000;--text-secondary:#444;--text-muted:#777;
--grid:#ddd;--axis:#bbb;--ring:#ccc;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#8a5cd6;--series-5:#c9a00c;--series-6:#d6458b;
--good:#0ca30c;--warning:#b97f00;--critical:#d03b3b}
.wrap{max-width:none;padding:0}.card,.tile{break-inside:avoid}.tablewrap{overflow:visible}table{font-size:8.5px}th,td{padding:2px 4px}
pre.raw{white-space:pre-wrap}details>summary{display:none}}
"""

_LINK_TEXT = {
    "SYS": "PCIe across CPU sockets (crosses the NUMA interconnect)",
    "NODE": "PCIe across host bridges, same NUMA node", "PHB": "PCIe through the CPU host bridge",
    "PXB": "PCIe through several PCIe switches", "PIX": "PCIe through one PCIe switch",
}


def _link_text(link: str) -> str:
    return f"NVLink ({link})" if re.match(r"NV\d+", link) else f"{link} · {_LINK_TEXT.get(link, 'see topology')}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _gpu_tiles(gpus: list[dict]) -> list[str]:
    if not gpus:
        return []
    utils = [g["util_mean"] for g in gpus if g["util_mean"] is not None]
    mem = [(g["mem_used_max_mib"], g["mem_total_mib"] or 0) for g in gpus if g["mem_used_max_mib"] is not None]
    pcie = [max(g["rx_max"] or 0, g["tx_max"] or 0) for g in gpus]
    power = [g["power_mean"] for g in gpus if g["power_mean"] is not None]
    return [
        _tile("GPU utilization mean", _int(_mean(utils)),
              " · ".join(f"GPU{g['gpu']} {_int(g['util_mean'])}%" for g in gpus), unit="%"),
        _tile("GPU memory peak", _num(sum(m[0] for m in mem) / 1024, 1) if mem else DASH,
              f"of {_num(sum(m[1] for m in mem) / 1024, 1)} GiB" if mem else "", unit="GiB"),
        _tile("GPU power mean", _int(sum(power)) if power else DASH,
              " · ".join(f"GPU{g['gpu']} max {_int(g['power_max'])} W" for g in gpus), unit="W"),
        _tile("PCIe peak", _int(max(pcie)) if pcie else DASH, "highest RX or TX on any GPU", unit="MB/s"),
    ]


def _gpu_charts(report: dict[str, Any], e0: float, e1: float, clock: _Clock) -> str:
    gpus, engine = report["gpus"], report["engine"]
    if not gpus:
        return ""

    def per(field: str) -> list[tuple[str, str, str]]:
        return [(f"g{g['gpu']}_{field}", f"GPU {g['gpu']}", _COLORS[i % 6]) for i, g in enumerate(gpus)]

    pcie = []
    for i, g in enumerate(gpus):
        pcie += [(f"g{g['gpu']}_rx", f"GPU {g['gpu']} RX", _COLORS[(2 * i) % 6]),
                 (f"g{g['gpu']}_tx", f"GPU {g['gpu']} TX", _COLORS[(2 * i + 1) % 6])]
    mem_top = max((g["mem_total_mib"] or 0) for g in gpus) / 1024 or None
    power_top = max((g["power_limit"] or 0) for g in gpus) or None
    return '<div class="grid">' + "".join([
        _line_chart("GPU utilization", "percent of time a kernel was running, per GPU", engine, per("util"),
                    e0, e1, clock, y_max=100, y_fmt=lambda v: f"{v:.0f}%"),
        _line_chart("GPU memory used", "GiB per GPU (vLLM reserves most of it at startup)", engine, per("mem_used"),
                    e0, e1, clock, y_max=mem_top, scale=1 / 1024),
        _line_chart("GPU power", "watts per GPU; axis top is the power limit", engine, per("power"),
                    e0, e1, clock, y_max=power_top, min_top=50, y_fmt=lambda v: f"{v:.0f} W"),
        _line_chart("GPU temperature", "°C per GPU", engine, per("temp"), e0, e1, clock, min_top=90,
                    y_fmt=lambda v: f"{v:.0f}°"),
        _line_chart("SM clock", "MHz per GPU; drops under power or thermal limits", engine, per("sm_clock"),
                    e0, e1, clock, min_top=500),
        _line_chart("PCIe traffic", "MB/s per GPU; with tensor parallelism this includes GPU-to-GPU traffic",
                    engine, pcie, e0, e1, clock, min_top=10),
    ]) + "</div>"


def _facts_dl(pairs: list[tuple[str, Any]]) -> str:
    return '<dl class="facts">' + "".join(
        f"<div><dt>{_E(k)}</dt><dd>{_E(str(v))}</dd></div>" for k, v in pairs if v not in (None, "")) + "</dl>"


def _digits(value: Any) -> int | None:
    return int(value) if str(value or "").isdigit() else None


def _hardware_html(report: dict[str, Any]) -> str:
    hw, gpus_sum = report.get("hardware") or {}, report.get("gpus") or []
    inv = hw.get("gpu_inventory") or {}
    cache = (hw.get("vllm:cache_config_info") or [{}])[0]
    version = (hw.get("vllm_version") or {}).get("version")
    max_len = report.get("context_length")
    head = '<section class="card"><div class="card-head"><h2>Hardware and vLLM</h2>'
    if not inv and not cache and not version and not gpus_sum:
        return (head + '</div><div class="none">No GPU helper data or vLLM configuration was recorded for this '
                "range.</div></section>")
    gpus = inv.get("gpus") or []
    links = (inv.get("topology") or {}).get("links") or []
    host = inv.get("host") or {}
    left = [
        ("GPUs", f"{len(gpus)} × {gpus[0].get('name')}" if gpus else None),
        ("Driver / CUDA", f"{gpus[0].get('driver_version', DASH)} / {inv.get('cuda_version') or DASH}" if gpus
         else None),
        ("GPU-to-GPU", "; ".join(f"{ln['a']}↔{ln['b']}: {_link_text(ln['link'])}" for ln in links) if links
         else ("single GPU" if len(gpus) == 1 else None)),
        ("NVLink", ("active" if inv.get("nvlink_active") else "inactive") if gpus else None),
        ("Host CPU", f"{host.get('cpu_model') or DASH} · {host.get('logical_cpus') or DASH} logical CPUs" if host
         else None),
        ("Host RAM", f"{host['memory_gib']:.0f} GiB" if host.get("memory_gib") else None),
    ]
    block, blocks = _digits(cache.get("block_size")), _digits(cache.get("num_gpu_blocks"))
    kv_tokens = _digits(cache.get("kv_cache_size_tokens")) or (block * blocks if block and blocks else None)
    conc = cache.get("kv_cache_max_concurrency")
    if not conc and kv_tokens and max_len:
        conc = kv_tokens / max_len
    try:
        conc_text = f"{float(conc):.2f}× requests at {int(max_len):,} tokens each" if conc and max_len else None
    except (TypeError, ValueError):
        conc_text = None
    try:
        share = _pctf(float(cache["gpu_memory_utilization"])) if cache.get("gpu_memory_utilization") else None
    except ValueError:
        share = None
    right = [
        ("vLLM", version),
        ("Served model", report.get("served_model")),
        ("Context length", f"{int(max_len):,} tokens" if max_len else None),
        ("KV cache size", f"{kv_tokens:,} tokens" if kv_tokens else None),
        ("Max concurrency", conc_text),
        ("KV cache dtype", cache.get("cache_dtype")),
        ("Block size / blocks", f"{block} tokens / {blocks:,}" if block and blocks else None),
        ("GPU memory share", share),
        ("Prefix caching", {"True": "on", "False": "off"}.get(cache.get("enable_prefix_caching"))),
    ]
    parts = [f'<div class="hw-grid"><div>{_facts_dl(left)}</div><div>{_facts_dl(right)}</div></div>']
    if gpus:
        def num(g: dict, field: str) -> float | None:
            try:
                return float(g[field])
            except (KeyError, TypeError, ValueError):
                return None

        rows = [[_E(str(g.get("index", ""))), _E(str(g.get("name", ""))),
                 f"{num(g, 'memory.total') / 1024:.1f} GiB" if num(g, "memory.total") else DASH,
                 _E(str(g.get("ecc.mode.current", DASH))),
                 f"{num(g, 'power.limit'):.0f} W" if num(g, "power.limit") else DASH,
                 _E(f"Gen{g.get('pcie.link.gen.max')} x{g.get('pcie.link.width.max')}") if g.get("pcie.link.gen.max")
                 else DASH,
                 _E(f"{g.get('clocks.max.sm')} MHz") if g.get("clocks.max.sm") else DASH,
                 _E(str(g.get("pci.bus_id", ""))), _E(str(g.get("vbios_version", "")))] for g in gpus]
        parts.append(_table(["GPU", "Name", "Memory", "ECC", "Power limit", "PCIe max", "Max SM clock", "Bus ID",
                             "VBIOS"], rows, (1, 3, 7, 8)))
    if gpus_sum:
        def pctv(v):
            return f"{_int(v)}%" if v is not None else DASH

        rows = [[f"GPU {g['gpu']}", pctv(g["util_mean"]), pctv(g["util_p90"]),
                 f"{_num((g['mem_used_max_mib'] or 0) / 1024, 1)} / {_num((g['mem_total_mib'] or 0) / 1024, 1)} GiB"
                 if g["mem_used_max_mib"] is not None else DASH,
                 f"{_int(g['power_mean'])} / {_int(g['power_max'])} W" if g["power_mean"] is not None else DASH,
                 f"{_int(g['temp_max'])}°" if g["temp_max"] is not None else DASH,
                 f"{_int(g['sm_clock_min'])} MHz" if g["sm_clock_min"] is not None else DASH,
                 f"{_int(g['rx_mean'])} / {_int(g['rx_max'])}" if g["rx_mean"] is not None else DASH,
                 f"{_int(g['tx_mean'])} / {_int(g['tx_max'])}" if g["tx_mean"] is not None else DASH,
                 _int(g["samples"])] for g in gpus_sum]
        parts.append("<h3>GPU history over this range</h3>" + _table(
            ["GPU", "Util mean", "Util p90", "Memory peak", "Power mean / max", "Temp max", "SM clock min",
             "PCIe RX mean / max MB/s", "PCIe TX mean / max MB/s", "Samples"], rows))
    log_lines = [str(line) for lines in (inv.get("vllm_log") or {}).values() for line in lines]
    if log_lines:
        parts.append(f'<details open><summary>vLLM startup lines ({_E(str(inv.get("vllm_log_path") or "log"))})'
                     f'</summary><pre class="raw">{_E(chr(10).join(log_lines))}</pre></details>')
    if inv.get("topology_text"):
        parts.append('<details open><summary>nvidia-smi topo -m</summary>'
                     f'<pre class="raw">{_E(str(inv["topology_text"]))}</pre></details>')
    return (head + '<span class="sub">latest snapshot recorded in this range; the table summarizes the GPU history'
            "</span></div>" + "".join(parts) + "</section>")


def render_html(report: dict[str, Any], tz_offset_min: int = 0) -> str:
    clock = _Clock(tz_offset_min)
    f, w, s, rng = report["filter"], report["window"], report["summary"], report["range"]
    reqs, engine, keys = report["requests"], report["engine"], report["keys"]
    e0, e1 = w["start"], w["end"]
    scope = " · ".join(x for x in (
        f"key {f['key']}" if f["key"] else None,
        f"session {f['session']}" if f["session"] else None) if x) or "all keys and sessions"
    title = f"Report · {report['title']}"
    proxy = report.get("proxy") or {}

    facts = [
        ("Deployment", ", ".join(report["deployments"]) or DASH), ("Served model", report.get("served_model") or DASH),
        ("Scope", scope),
        ("Requested range", f"{clock.dt(rng['from'])} to {clock.dt(rng['to'])}"),
        ("Charts from", clock.dt(e0)), ("Charts to", clock.dt(e1)), ("Duration", _dur(w["duration_s"])),
        ("Keys", ", ".join(k["key"] for k in keys) or DASH),
        ("Generated", f"{clock.dt(report['generated_at'])} {clock.label}"),
    ]
    facts_html = "".join(f"<div><dt>{_E(k)}</dt><dd>{_E(v)}</dd></div>" for k, v in facts)

    drops = (proxy.get("dropped") or 0) + (proxy.get("failed") or 0)
    err_sub = "upstream or proxy"
    if proxy:
        err_sub = f"writer: {proxy.get('dropped', 0)} dropped, {proxy.get('failed', 0)} failed since start"
    tiles = "".join([
        _tile("Requests", _int(s["n"]), f"{s['ok']} ok · {s['cancelled']} cancelled"),
        _tile("Errors", _int(s["errors"]), err_sub, alert=s["errors"] > 0 or drops > 0),
        _tile("Open requests max", _int(s["open_max"]), "at the proxy, all keys"),
        _tile("Engine running max", _int(s["running_max"]), f"waiting max {_int(s['waiting_max'])}",
              alert=(s["waiting_max"] or 0) > 0),
        _tile("KV cache max", _pctf(s["kv_max"], 1), f"{_int(s['preemptions'])} preemptions",
              alert=(s["preemptions"] or 0) > 0),
        _tile("Output mean", _num(s["output_tps_mean"]), f"peak {_num(s['output_tps_max'])}", unit="tok/s"),
        _tile("Prompt mean", _num(s["prompt_tps_mean"]), f"peak {_num(s['prompt_tps_max'])}", unit="tok/s"),
        _tile("First content p50", _ms(s["first_p50"]), f"p90 {_ms(s['first_p90'])}"),
        _tile("End p50", _ms(s["end_p50"]), f"p90 {_ms(s['end_p90'])}"),
        _tile("Decode p50", _num(s["decode_p50"]), f"delivered {_num(s['delivered_p50'])}", unit="tok/s"),
        _tile("Output tokens", _int(s["completion_tokens"]), f"prompt {_int(s['prompt_tokens'])}"),
        _tile("Prompt cached", _pctf(s["cached_ratio"]), "prefix-cache hit share"),
        _tile("Largest prompt", _int(s["prompt_max"]), f"{_pctf(s['ctx_max'], 1)} of context", unit="tok"),
        *_gpu_tiles(report["gpus"]),
    ])

    charts = '<div class="grid">' + "".join([
        _line_chart("Concurrency", "engine-wide requests (all keys)", engine,
                    [("running", "Running", "--series-1"), ("waiting", "Waiting", "--series-2"),
                     ("open", "Open at proxy", "--series-3")], e0, e1, clock, min_top=4),
        _line_chart("KV cache used", "percent of GPU cache blocks", engine, [("kv", "KV cache", "--series-1")],
                    e0, e1, clock, y_max=100, scale=100, y_fmt=lambda v: f"{v:.0f}%"),
        _line_chart("Output tokens/s", "engine-wide, from counter deltas", engine,
                    [("output_tps", "Output tok/s", "--series-1")], e0, e1, clock, min_top=10),
        _line_chart("Prompt tokens/s", "engine-wide, includes cached prompt tokens", engine,
                    [("prompt_tps", "Prompt tok/s", "--series-1")], e0, e1, clock, min_top=10),
        _line_chart("Prefix cache hit rate", "share of looked-up prompt tokens served from cache", engine,
                    [("prefix_hit", "Hit rate", "--series-1")], e0, e1, clock, y_max=100, scale=100,
                    y_fmt=lambda v: f"{v:.0f}%"),
    ]) + "</div>" + _gpu_charts(report, e0, e1, clock) + _scatter(reqs, keys, e0, e1, clock)

    key_rows = [[_E(k["key"]), _int(k["n"]), _int(k["ok"]), _int(k["cancelled"]), _int(k["errors"]),
                 _ms(k["first_p50"]), _ms(k["first_p90"]), _ms(k["text_p50"]), _ms(k["tool_p50"]),
                 _ms(k["end_p50"]), _ms(k["end_p90"]), _ms(k["gap_p90"]), _num(k["decode_p50"]),
                 _num(k["delivered_p50"]), _pctf(k["cached_ratio"]), _int(k["completion_tokens"]),
                 _pctf(k["ctx_max"], 1)] for k in keys]
    keys_html = _table(["Key", "Requests", "OK", "Cancelled", "Errors", "First p50", "First p90", "Text p50",
                        "Tool p50", "End p50", "End p90", "Gap p90", "Decode p50 tok/s", "Delivered p50 tok/s",
                        "Cached", "Output tok", "Context max"], key_rows, (0,), "No requests in this scope.")

    multi_day = time.gmtime(e0 + clock.off)[:3] != time.gmtime(e1 + clock.off)[:3]
    shown = reqs[-MAX_HTML_ROWS:]
    req_rows = []
    for r in shown:
        cached = (_pctf(r["cached_tokens"] / r["prompt_tokens"])
                  if r["prompt_tokens"] and r["cached_tokens"] is not None else DASH)
        req_rows.append([
            clock.t(r["t_arrival_wall"], "%m-%d %H:%M:%S" if multi_day else "%H:%M:%S"),
            _E(r["actor_id"] or DASH), _E(r["session_id"] or DASH), _E(r["api"] or DASH), _outcome(r["outcome"]),
            _E(r["finish_reason"] or DASH), _int(r["messages_count"]), _int(r["prompt_tokens"]), cached,
            _int(r["completion_tokens"]), _int(r["reasoning_tokens"]), _int(r["open_requests_at_start"]),
            _ms(r["first_payload_ms"]), _ms(r["first_text_ms"]), _ms(r["first_tool_ms"]), _ms(r["end_ms"]),
            _ms(r["max_gap_ms"]), _num(r["decode_tps"]), _num(r["delivered_tps"]), _pctf(r["context_fraction"], 1),
        ])
    req_html = _table(["Time", "Key", "Session", "API", "Outcome", "Finish", "Msgs", "Prompt tok", "Cached",
                       "Output tok", "Reasoning tok", "Open at start", "First content", "First text", "First tool",
                       "End", "Max gap", "Decode tok/s", "Delivered tok/s", "Context"],
                      req_rows, (1, 2, 3, 4, 5), "No requests in this scope.")
    req_sub = f"{len(reqs):,} in arrival order"
    if len(shown) < len(reqs):
        req_sub = f"the last {len(shown):,} of {len(reqs):,}; all of them are in the embedded JSON"
    open_attr = " open" if len(reqs) <= 200 else ""

    data = to_json(report).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{_E(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
<h1>{_E(title)}</h1>
<div class="meta">Ollama to vLLM proxy · metadata only (no prompts, completions or tool arguments) · times in {clock.label}</div>
<dl class="facts">{facts_html}</dl>
</header>
<div class="tiles">{tiles}</div>
{_hardware_html(report)}
{charts}
<section class="card"><div class="card-head"><h2>Per key</h2><span class="sub">latencies on the proxy clock</span></div>{keys_html}</section>
<section class="card"><div class="card-head"><h2>Requests</h2><span class="sub">{_E(req_sub)}</span></div>
<details{open_attr}><summary>Show {len(shown):,} requests</summary>{req_html}</details></section>
<p class="note">Engine and GPU charts are engine-wide: a report filtered to one key or session still shows the load caused by all keys. The full data set is embedded in this file as JSON (element <code>#report-data</code>, format <code>{_E(report["format"])}</code>).</p>
</div>
<script type="application/json" id="report-data">{data}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_SPAN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw]?)\s*$")
_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def parse_span(text: str) -> float:
    """'90m', '2h', '7d' or plain seconds -> seconds."""
    m = _SPAN_RE.match(text)
    if m is None or float(m.group(1)) <= 0:
        raise ExportError(f"cannot read '{text}' as a duration; use for example 30m, 2h or 7d")
    return float(m.group(1)) * _UNITS[m.group(2)]


def parse_time(text: str) -> float:
    """Epoch seconds, or an ISO date/time (local time unless it carries an offset)."""
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.strip()).timestamp()
    except ValueError:
        raise ExportError(f"cannot read '{text}' as a time; use epoch seconds or e.g. 2026-11-24T09:30") from None


def resolve_range(t_from: str | None, t_to: str | None, last: str | None, now: float | None = None) -> tuple[float, float]:
    now = time.time() if now is None else now
    t1 = parse_time(t_to) if t_to else now
    if t_from and last:
        raise ExportError("give either --from or --last, not both")
    if t_from:
        t0 = parse_time(t_from)
    else:
        t0 = t1 - parse_span(last or "24h")
    if t1 <= t0:
        raise ExportError("the time range is empty: 'to' must be later than 'from'")
    return t0, t1


def local_tz_offset_min() -> int:
    offset = datetime.now().astimezone().utcoffset()
    return int(offset.total_seconds() // 60) if offset is not None else 0


def _print_options(data: dict[str, Any]) -> None:
    def when(ts: float | None) -> str:
        return DASH if ts is None else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

    span = data["span"]
    print(f"data from {when(span['first'])} to {when(span['last'])} (local time), {data['requests']} requests")
    for title, rows, name in (("keys", data["keys"], "key"), ("sessions", data["sessions"], "session"),
                              ("deployment labels", data["deployments"], "label")):
        print(f"\n{title}:" + ("" if rows else " none"))
        for row in rows:
            print(f"  {row[name]:<44} {row['requests']:>7} requests  {when(row['first'])} to {when(row['last'])}")


def _cli(args: argparse.Namespace, settings: Any) -> int:
    from .store import Store

    store = Store(settings.db_path)
    try:
        if args.list:
            if args.t_from or args.last or args.t_to:
                t0, t1 = resolve_range(args.t_from, args.t_to, args.last)
                _print_options(options(store.read, t0, t1))
            else:
                _print_options(options(store.read))
            return 0
        t0, t1 = resolve_range(args.t_from, args.t_to, args.last)
        report = build_report(store.read, t0, t1, args.key, args.session)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.stop()
    tz = args.tz_offset if args.tz_offset is not None else local_tz_offset_min()
    text = render_html(report, tz) if args.format == "html" else to_json(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.output} ({len(report['requests'])} requests)", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


def add_cli(subparsers: Any) -> None:
    """Register ``ollama-vllm-proxy export`` (works directly on the SQLite file)."""
    p = subparsers.add_parser(
        "export", help="write a monitoring report for a time range",
        description="Write a self-contained HTML report (or its JSON data) for a time range.")
    p.add_argument("--from", dest="t_from", metavar="TIME", help="start: epoch seconds or ISO time, e.g. 2026-11-24T09:00")
    p.add_argument("--to", dest="t_to", metavar="TIME", help="end (default: now)")
    p.add_argument("--last", metavar="SPAN", help="range ending at --to, e.g. 30m, 2h, 7d (default: 24h)")
    p.add_argument("--key", metavar="NAME", help="only requests made with this API key")
    p.add_argument("--session", metavar="ID", help="only requests with this X-Session-Id")
    p.add_argument("--format", choices=("html", "json"), default="html")
    p.add_argument("--tz-offset", type=int, default=None, metavar="MINUTES",
                   help="minutes east of UTC for displayed times (default: this machine's zone)")
    p.add_argument("--list", action="store_true", help="list keys, sessions and deployment labels that have data")
    p.add_argument("-o", "--output", metavar="FILE", help="write to this file (default: standard output)")
    p.set_defaults(func=_cli)
