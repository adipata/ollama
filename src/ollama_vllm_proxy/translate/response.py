"""OpenAI (vLLM) response -> Ollama response, streaming (NDJSON) and buffered."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from ..observe import RequestRecord


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _line(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


def done_reason(finish_reason: str | None) -> str:
    return "length" if finish_reason == "length" else "stop"


def error_message(err: Any, default: str = "upstream error") -> str:
    if isinstance(err, dict):
        inner = err.get("error", err)
        if isinstance(inner, dict):
            message = inner.get("message")
            return message if isinstance(message, str) and message else default
        if isinstance(inner, str) and inner:
            return inner
    if isinstance(err, str) and err:
        return err
    return default


def _stats(rec: RequestRecord, usage: dict[str, Any] | None) -> dict[str, Any]:
    """Ollama's final counters; durations are nanoseconds on the proxy clock."""
    total_ms = rec.ms(time.monotonic())
    first = rec.first_payload_ms if rec.first_payload_ms is not None else total_ms
    last = rec.last_content_ms if rec.last_content_ms is not None else total_ms
    usage = usage or {}
    return {
        "total_duration": int(total_ms * 1e6),
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
        "prompt_eval_duration": int(first * 1e6),
        "eval_count": int(usage.get("completion_tokens") or 0),
        "eval_duration": int(max(last - first, 0.0) * 1e6),
    }


class OllamaAdapter:
    """Translates an upstream chat/completions response for /api/chat or /api/generate."""

    stream_content_type = b"application/x-ndjson"

    def __init__(self, kind: str, client_model: str):
        self.kind = kind  # "chat" or "generate"
        self.model = client_model
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self._calls: dict[int, dict[str, Any]] = {}  # tool-call index -> {id, name, args}
        self._emitted_calls: set[int] = set()
        self._failed = False
        self._finished = False

    # -- line builders ----------------------------------------------------------

    def _chunk(self, *, content: str = "", thinking: str = "", tool_calls: list | None = None) -> dict[str, Any]:
        base: dict[str, Any] = {"model": self.model, "created_at": now_iso()}
        if self.kind == "chat":
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if thinking:
                message["thinking"] = thinking
            if tool_calls:
                message["tool_calls"] = tool_calls
            base["message"] = message
        else:
            base["response"] = content
            if thinking:
                base["thinking"] = thinking
        base["done"] = False
        return base

    def _final(self, rec: RequestRecord, content: str = "", thinking: str = "", tool_calls: list | None = None) -> dict:
        obj = self._chunk(content=content, thinking=thinking, tool_calls=tool_calls)
        obj["done"] = True
        obj["done_reason"] = done_reason(self.finish_reason)
        if self.kind == "generate":
            obj["context"] = []
        obj.update(_stats(rec, self.usage))
        return obj

    def _ollama_calls(self, indexes: list[int]) -> list[dict[str, Any]]:
        out = []
        for i in indexes:
            call = self._calls[i]
            self._emitted_calls.add(i)
            entry: dict[str, Any] = {
                "function": {"name": call["name"], "arguments": parse_arguments(call["args"])}
            }
            if call.get("id"):
                entry["id"] = call["id"]
            out.append(entry)
        return out

    def _flush_calls(self, below: int | None = None) -> bytes:
        ready = sorted(
            i for i in self._calls if i not in self._emitted_calls and (below is None or i < below)
        )
        if not ready or self.kind != "chat":
            return b""
        return _line(self._chunk(tool_calls=self._ollama_calls(ready)))

    # -- streaming ----------------------------------------------------------------

    def comment(self, raw: bytes) -> bytes:
        return b""

    def malformed(self, raw: bytes) -> bytes:
        return b""

    def event(self, obj: dict, raw: bytes, rec: RequestRecord) -> bytes:
        if obj.get("error"):
            self._failed = True
            return _line({"error": error_message(obj)})
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        out = b""
        for choice in obj.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {"content": choice.get("text")} if isinstance(choice.get("text"), str) else {}
            for tc in delta.get("tool_calls") or ():
                if not isinstance(tc, dict):
                    continue
                index = tc.get("index") if isinstance(tc.get("index"), int) else len(self._calls)
                out += self._flush_calls(below=index)  # earlier calls are complete once a new one starts
                call = self._calls.setdefault(index, {"id": None, "name": "", "args": ""})
                if isinstance(tc.get("id"), str) and tc["id"]:
                    call["id"] = tc["id"]
                fn = tc.get("function")
                if isinstance(fn, dict):
                    if isinstance(fn.get("name"), str):
                        call["name"] += fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        call["args"] += fn["arguments"]
            thinking = delta.get("reasoning_content") or delta.get("reasoning")
            content = delta.get("content")
            thinking = thinking if isinstance(thinking, str) else ""
            content = content if isinstance(content, str) else ""
            if thinking or content:
                out += _line(self._chunk(content=content, thinking=thinking))
            if isinstance(choice.get("finish_reason"), str) and choice["finish_reason"]:
                self.finish_reason = choice["finish_reason"]
                out += self._flush_calls()
        return out

    def done(self, rec: RequestRecord) -> bytes:
        if self._failed or self._finished:
            return b""
        self._finished = True
        return self._flush_calls() + _line(self._final(rec))

    def end(self, rec: RequestRecord) -> bytes:
        if self._failed or self._finished:
            return b""
        self._finished = True
        return _line({"error": "upstream stream ended unexpectedly"})

    def interrupted(self, rec: RequestRecord) -> bytes:
        return _line({"error": "upstream stream interrupted"})

    # -- buffered -------------------------------------------------------------------

    def body(self, status: int, obj: Any, raw: bytes, ctype: str, rec: RequestRecord) -> tuple[int, str, bytes]:
        if status != 200 or not isinstance(obj, dict) or obj.get("error"):
            status = status if status != 200 else 502
            return status, "application/json", json.dumps({"error": error_message(obj)}).encode()
        self.usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
        content, thinking, calls = "", "", None
        choices = obj.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            self.finish_reason = choice.get("finish_reason")
            msg = choice.get("message")
            if isinstance(msg, dict):
                content = msg.get("content") if isinstance(msg.get("content"), str) else ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                thinking = reasoning if isinstance(reasoning, str) else ""
                for i, tc in enumerate(msg.get("tool_calls") or ()):
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(fn, dict):
                        self._calls[i] = {"id": tc.get("id"), "name": fn.get("name") or "", "args": fn.get("arguments")}
                if self._calls and self.kind == "chat":
                    calls = self._ollama_calls(sorted(self._calls))
            elif isinstance(choice.get("text"), str):
                content = choice["text"]
        final = self._final(rec, content=content, thinking=thinking, tool_calls=calls)
        return 200, "application/json", json.dumps(final, ensure_ascii=False).encode()

    def error(self, status: int, message: str, etype: str) -> tuple[str, bytes]:
        return "application/json", json.dumps({"error": message}).encode()


def load_response(kind: str, model: str, reason: str) -> dict[str, Any]:
    """Reply to an empty prompt (preload) or keep_alive: 0 (unload); no upstream call."""
    base: dict[str, Any] = {"model": model, "created_at": now_iso()}
    if kind == "chat":
        base["message"] = {"role": "assistant", "content": ""}
    else:
        base["response"] = ""
    base.update({"done": True, "done_reason": reason})
    return base
