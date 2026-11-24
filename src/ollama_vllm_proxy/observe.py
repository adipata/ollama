"""Per-request records and in-memory inspection of upstream responses.

Only metadata leaves this module: timings, counts, finish reasons and token
usage. Prompt text, completions, reasoning text and tool arguments are looked
at in memory to decide *whether* an event carries content, and then dropped.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable

OPAQUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


def opaque(value: Any) -> str | None:
    """Accept a short opaque identifier; anything else becomes None."""
    if isinstance(value, str) and OPAQUE_RE.match(value):
        return value
    return None


def nonneg_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def nonneg_num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value) or value < 0:
        return None
    return value


@dataclass
class RequestRecord:
    request_id: str
    deployment_id: str | None = None  # deployment label, e.g. rtx-pro-6000x1-qwen38-27b-bf16-131072
    key_id: int | None = None
    actor_id: str | None = None  # API key name at request time
    api: str = "openai"  # "openai" (/v1) or "ollama" (/api)
    session_id: str | None = None
    endpoint: str = "chat.completions"
    model: str | None = None
    stream: bool = False
    requested_max_tokens: int | None = None
    messages_count: int | None = None
    tools_count: int | None = None
    include_usage_forced: bool = False
    t_arrival_wall: float = 0.0
    open_requests_at_start: int | None = None
    # outcome
    status_code: int | None = None
    outcome: str | None = None
    finish_reason: str | None = None
    cancelled: bool = False
    error_type: str | None = None
    context_error: bool = False
    done_marker: bool = False
    # proxy monotonic timings, milliseconds since request arrival
    first_byte_ms: float | None = None
    first_payload_ms: float | None = None
    first_text_ms: float | None = None
    first_reasoning_ms: float | None = None
    first_tool_ms: float | None = None
    last_content_ms: float | None = None
    end_ms: float | None = None
    max_gap_ms: float | None = None
    content_events: int = 0
    tool_call_deltas: int = 0
    malformed_events: int = 0
    # final usage
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    # server-reported per-request metrics (vLLM --enable-per-request-metrics)
    srv_queue_s: float | None = None
    srv_ttft_s: float | None = None
    srv_generation_s: float | None = None
    srv_mean_itl_s: float | None = None
    srv_tokens_per_s: float | None = None
    srv_metrics_json: str | None = None
    # derived
    decode_tps: float | None = None
    delivered_tps: float | None = None
    served_max_model_len: int | None = None
    context_fraction: float | None = None
    cached_fraction: float | None = None
    t0: float = field(default=0.0, repr=False)

    def ms(self, now: float) -> float:
        return round((now - self.t0) * 1000.0, 3)

    def row(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("t0")
        for key, value in data.items():
            if isinstance(value, bool):
                data[key] = int(value)
        return data


def requests_ddl() -> str:
    """CREATE TABLE statement generated from RequestRecord so they never drift."""
    cols = []
    for f in fields(RequestRecord):
        if f.name == "t0":
            continue
        t = str(f.type)
        sql_type = "TEXT" if "str" in t else "REAL" if "float" in t else "INTEGER"
        suffix = " PRIMARY KEY" if f.name == "request_id" else ""
        cols.append(f"  {f.name} {sql_type}{suffix}")
    return "CREATE TABLE IF NOT EXISTS requests (\n" + ",\n".join(cols) + "\n);"


# --------------------------------------------------------------------------
# Field extraction helpers
# --------------------------------------------------------------------------

_CONTEXT_MARKERS = (
    "maximum context length",
    "max_model_len",
    "context length",
    "context window",
    "prompt is too long",
    "too many tokens",
)

_SRV_FIELDS: dict[str, tuple[str, ...]] = {
    "srv_queue_s": ("queue_time", "queued_time", "queue_duration", "time_in_queue"),
    "srv_ttft_s": ("ttft", "time_to_first_token", "first_token_time", "first_token_latency"),
    "srv_generation_s": ("generation_time", "decode_time", "generation_duration"),
    "srv_mean_itl_s": ("mean_itl", "mean_inter_token_latency", "inter_token_latency", "mean_tpot"),
    "srv_tokens_per_s": ("tokens_per_second", "tokens_per_s"),
}
_UNIT_SUFFIXES = (("", 1.0), ("_s", 1.0), ("_sec", 1.0), ("_seconds", 1.0), ("_ms", 1e-3))


def apply_usage(rec: RequestRecord, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    rec.prompt_tokens = nonneg_int(usage.get("prompt_tokens"))
    rec.completion_tokens = nonneg_int(usage.get("completion_tokens"))
    rec.total_tokens = nonneg_int(usage.get("total_tokens"))
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        rec.cached_tokens = nonneg_int(ptd.get("cached_tokens"))
    ctd = usage.get("completion_tokens_details")
    if isinstance(ctd, dict):
        rec.reasoning_tokens = nonneg_int(ctd.get("reasoning_tokens"))


def apply_server_metrics(rec: RequestRecord, metrics: Any) -> None:
    """Keep numeric server metrics only; map well-known names to columns.

    The exact key names depend on the installed vLLM build; the raw numeric
    map is preserved in srv_metrics_json so unmapped fields are not lost.
    """
    if not isinstance(metrics, dict):
        return
    flat: dict[str, float] = {}

    def walk(prefix: str, obj: dict, depth: int) -> None:
        for key, value in obj.items():
            if not isinstance(key, str) or not OPAQUE_RE.match(key):
                continue
            name = f"{prefix}{key}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    flat[name] = value
            elif isinstance(value, dict) and depth < 2:
                walk(name + ".", value, depth + 1)

    walk("", metrics, 0)
    if not flat:
        return
    rec.srv_metrics_json = json.dumps(flat, sort_keys=True)[:4000]
    for key, value in flat.items():
        base = key.rsplit(".", 1)[-1].lower()
        for col, names in _SRV_FIELDS.items():
            if getattr(rec, col) is not None:
                continue
            for name in names:
                for suffix, scale in _UNIT_SUFFIXES:
                    if base == name + suffix:
                        v = nonneg_num(value)
                        if v is not None:
                            setattr(rec, col, v * (1.0 if col == "srv_tokens_per_s" else scale))
                        break


def apply_error(rec: RequestRecord, err: Any) -> None:
    if isinstance(err, dict):
        etype = err.get("type") or err.get("code") or "error"
        message = err.get("message")
    else:
        etype, message = "error", err
    rec.error_type = opaque(str(etype)[:64]) or "error"
    if isinstance(message, str) and any(m in message.lower() for m in _CONTEXT_MARKERS):
        rec.context_error = True


def _nonempty(v: Any) -> bool:
    return isinstance(v, str) and v != ""


def _visible(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def mark_content(
    rec: RequestRecord,
    now: float,
    *,
    visible: bool = False,
    reasoning: bool = False,
    tool: bool = False,
) -> None:
    t = rec.ms(now)
    if rec.first_payload_ms is None:
        rec.first_payload_ms = t
    if visible and rec.first_text_ms is None:
        rec.first_text_ms = t
    if reasoning and rec.first_reasoning_ms is None:
        rec.first_reasoning_ms = t
    if tool and rec.first_tool_ms is None:
        rec.first_tool_ms = t
    if rec.last_content_ms is None:
        rec.max_gap_ms = 0.0
    else:
        gap = round(t - rec.last_content_ms, 3)
        if rec.max_gap_ms is None or gap > rec.max_gap_ms:
            rec.max_gap_ms = gap
    rec.last_content_ms = t
    rec.content_events += 1


def _observe_message(rec: RequestRecord, msg: dict, now: float) -> None:
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    tool_calls = msg.get("tool_calls")
    has_tool = isinstance(tool_calls, list) and len(tool_calls) > 0
    has_text = _nonempty(content) or _nonempty(msg.get("refusal"))
    has_reasoning = _nonempty(reasoning)
    if has_tool:
        rec.tool_call_deltas += len(tool_calls)
    if has_text or has_reasoning or has_tool:
        mark_content(
            rec,
            now,
            visible=_visible(content) or _visible(msg.get("refusal")),
            reasoning=has_reasoning,
            tool=has_tool,
        )


def _finish_reason(rec: RequestRecord, choice: dict) -> None:
    fr = choice.get("finish_reason")
    if isinstance(fr, str) and fr:
        rec.finish_reason = opaque(fr[:32]) or "other"


def inspect_completion(rec: RequestRecord, obj: dict, now: float) -> None:
    """Inspect a non-streaming chat completion body."""
    if obj.get("error"):
        apply_error(rec, obj["error"])
    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message")
            if isinstance(msg, dict):
                _observe_message(rec, msg, now)
            elif _nonempty(choice.get("text")):
                mark_content(rec, now, visible=_visible(choice.get("text")))
            _finish_reason(rec, choice)
    if obj.get("usage"):
        apply_usage(rec, obj["usage"])
    if "metrics" in obj:
        apply_server_metrics(rec, obj["metrics"])


class SSESplitter:
    """Splits an upstream SSE byte stream into complete events.

    ``feed`` returns ``(raw_event, data)`` pairs, where ``raw_event`` is the
    event without its terminating blank line and ``data`` is the joined
    ``data:`` payload, or None for comments, keepalives and name-only events.
    """

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, chunk: bytes) -> list[tuple[bytes, bytes | None]]:
        self._buf += chunk
        if b"\r" in self._buf:
            self._buf = self._buf.replace(b"\r\n", b"\n")
        out: list[tuple[bytes, bytes | None]] = []
        while True:
            idx = self._buf.find(b"\n\n")
            if idx < 0:
                break
            event = self._buf[:idx]
            self._buf = self._buf[idx + 2 :]
            out.append((event, self._data(event)))
        return out

    def tail(self) -> list[tuple[bytes, bytes | None]]:
        rest, self._buf = self._buf.rstrip(b"\n"), b""
        return [(rest, self._data(rest))] if rest.strip() else []

    @staticmethod
    def _data(event: bytes) -> bytes | None:
        lines = []
        for line in event.split(b"\n"):
            if line.startswith(b"data:"):
                value = line[5:]
                lines.append(value[1:] if value.startswith(b" ") else value)
        return b"\n".join(lines).strip() if lines else None


def observe_stream_object(rec: RequestRecord, obj: dict, now: float) -> None:
    """Record what one parsed streaming chunk carries (never its content)."""
    if obj.get("error"):
        apply_error(rec, obj["error"])
    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                _observe_message(rec, delta, now)
            elif _nonempty(choice.get("text")):
                mark_content(rec, now, visible=_visible(choice.get("text")))
            _finish_reason(rec, choice)
    if obj.get("usage"):
        apply_usage(rec, obj["usage"])
    if "metrics" in obj:
        apply_server_metrics(rec, obj["metrics"])


def is_usage_only(obj: dict) -> bool:
    return bool(obj.get("usage")) and obj.get("choices") == []


def finalize(rec: RequestRecord, now: float, served_max_model_len: int | None) -> None:
    rec.end_ms = rec.ms(now)
    wall_s = rec.end_ms / 1000.0
    ct = rec.completion_tokens
    if ct is not None and wall_s > 0:
        rec.delivered_tps = round(ct / wall_s, 3)
    if rec.outcome == "ok" and ct is not None and ct > 1:
        gen_s = rec.srv_generation_s
        if not gen_s and rec.first_payload_ms is not None and rec.last_content_ms is not None:
            # Proxy-side estimate: first content event to last content event.
            gen_s = (rec.last_content_ms - rec.first_payload_ms) / 1000.0
        if gen_s and gen_s > 0:
            rec.decode_tps = round((ct - 1) / gen_s, 3)
    rec.served_max_model_len = served_max_model_len
    if served_max_model_len and rec.prompt_tokens is not None and rec.outcome == "ok":
        rec.context_fraction = round(
            (rec.prompt_tokens + (ct or 0)) / served_max_model_len, 6
        )
    if rec.prompt_tokens and rec.cached_tokens is not None:
        rec.cached_fraction = round(rec.cached_tokens / rec.prompt_tokens, 6)
