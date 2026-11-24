"""Ollama request -> OpenAI (vLLM) request.

Everything here is deterministic: the same Ollama conversation always produces
the same upstream payload, which keeps vLLM's prefix cache effective.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class BadRequest(ValueError):
    pass


_SAME_NAME_OPTIONS = ("temperature", "top_p", "seed", "presence_penalty", "frequency_penalty", "top_k", "min_p")


def apply_options(options: Any, payload: dict[str, Any]) -> None:
    if not isinstance(options, dict):
        return
    for name in _SAME_NAME_OPTIONS:
        if options.get(name) is not None:
            payload[name] = options[name]
    if options.get("repeat_penalty") is not None:
        payload["repetition_penalty"] = options["repeat_penalty"]
    num_predict = options.get("num_predict")
    if isinstance(num_predict, int) and not isinstance(num_predict, bool) and num_predict > 0:
        payload["max_tokens"] = num_predict
    stop = options.get("stop")
    if isinstance(stop, str) and stop:
        payload["stop"] = [stop]
    elif isinstance(stop, list) and stop:
        payload["stop"] = [s for s in stop if isinstance(s, str)]
    # num_ctx, num_gpu, num_thread, mirostat*, ... have no meaning for a fixed vLLM server.


def apply_format(fmt: Any, payload: dict[str, Any]) -> None:
    if fmt in (None, ""):
        return
    if fmt == "json":
        payload["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt}}
    else:
        raise BadRequest("format must be \"json\" or a JSON schema object")


def apply_think(think: Any, payload: dict[str, Any]) -> None:
    """think: true/false or "low"/"medium"/"high" (any level means on)."""
    if think is None:
        return
    enabled = bool(think) if not isinstance(think, str) else think.lower() not in ("", "false", "none", "off")
    payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = enabled


def _image_part(data: Any) -> dict[str, Any]:
    if not isinstance(data, str) or not data:
        raise BadRequest("images must be base64 strings")
    if data.startswith("data:"):
        url = data
    else:
        mime = "image/png"
        if data.startswith("/9j/"):
            mime = "image/jpeg"
        elif data.startswith("R0lG"):
            mime = "image/gif"
        elif data.startswith("UklG"):
            mime = "image/webp"
        url = f"data:{mime};base64,{data}"
    return {"type": "image_url", "image_url": {"url": url}}


def _content(text: Any, images: Any) -> Any:
    text = text if isinstance(text, str) else ""
    if isinstance(images, list) and images:
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        parts.extend(_image_part(i) for i in images)
        return parts
    return text


def _tool_call_id(index: int, name: str, arguments: str) -> str:
    digest = hashlib.sha256(f"{name}\x00{arguments}".encode()).hexdigest()[:10]
    return f"call_{index}_{digest}"


def convert_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise BadRequest("messages must be a list")
    out: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []  # (tool_call_id, tool name) not yet answered
    call_index = 0
    for msg in messages:
        if not isinstance(msg, dict):
            raise BadRequest("each message must be an object")
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise BadRequest(f"unsupported message role: {role!r}")
        if role == "tool":
            wanted_id = msg.get("tool_call_id")
            wanted_name = msg.get("tool_name") or msg.get("name")
            match = None
            for i, (cid, cname) in enumerate(pending):
                if (wanted_id and cid == wanted_id) or (not wanted_id and wanted_name and cname == wanted_name):
                    match = i
                    break
            if match is None and pending and not wanted_id:
                match = 0
            if match is not None:
                call_id = pending.pop(match)[0]
            else:
                call_id = wanted_id if isinstance(wanted_id, str) and wanted_id else f"call_orphan_{len(out)}"
            content = msg.get("content")
            out.append({"role": "tool", "tool_call_id": call_id, "content": content if isinstance(content, str) else ""})
            continue
        item: dict[str, Any] = {"role": role, "content": _content(msg.get("content"), msg.get("images"))}
        tool_calls = msg.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            pending = []
            converted = []
            for call in tool_calls:
                fn = call.get("function") if isinstance(call, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    raise BadRequest("tool_calls entries need function.name")
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args if args is not None else {}, separators=(",", ":"), sort_keys=True)
                call_id = call.get("id") if isinstance(call.get("id"), str) and call.get("id") else None
                call_id = call_id or _tool_call_id(call_index, fn["name"], args)
                call_index += 1
                pending.append((call_id, fn["name"]))
                converted.append(
                    {"id": call_id, "type": "function", "function": {"name": fn["name"], "arguments": args}}
                )
            item["tool_calls"] = converted
        # messages[].thinking (assistant history) is dropped on purpose.
        out.append(item)
    return out


def chat_payload(body: dict[str, Any], served_model: str, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": served_model,
        "messages": convert_messages(body.get("messages") or []),
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    apply_format(body.get("format"), payload)
    apply_think(body.get("think"), payload)
    apply_options(body.get("options"), payload)
    return payload


def generate_payload(body: dict[str, Any], served_model: str, stream: bool) -> tuple[str, dict[str, Any]]:
    """Returns (upstream path, payload) for /api/generate."""
    prompt = body.get("prompt")
    prompt = prompt if isinstance(prompt, str) else ""
    suffix = body.get("suffix")
    if body.get("raw") or (isinstance(suffix, str) and suffix):
        payload: dict[str, Any] = {"model": served_model, "prompt": prompt, "stream": stream}
        if isinstance(suffix, str) and suffix:
            payload["suffix"] = suffix
        if stream:
            payload["stream_options"] = {"include_usage": True}
        apply_options(body.get("options"), payload)
        return "/v1/completions", payload
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt, "images": body.get("images")})
    chat_body = {
        "messages": messages,
        "format": body.get("format"),
        "think": body.get("think"),
        "options": body.get("options"),
    }
    return "/v1/chat/completions", chat_payload(chat_body, served_model, stream)
