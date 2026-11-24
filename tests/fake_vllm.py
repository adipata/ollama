"""A small stand-in for vLLM's OpenAI server, for tests and local runs.

    python tests/fake_vllm.py --port 8000 --api-key test-key

Behaviour is chosen by the text of the last user message:
  contains "tool"     -> a tool call, streamed in fragments
  contains "think"    -> reasoning deltas, then text
  contains "overflow" -> HTTP 400 context-length error
  contains "midfail"  -> an error event in the middle of the stream
  contains "slow"     -> long pause before the first token (cancellation tests)
  anything else       -> "Hello from fake vLLM."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

MODEL = "dev-test"
MAX_MODEL_LEN = 65536


class State:
    def __init__(self, api_key: str = "test-key", delay: float = 0.0):
        self.api_key = api_key
        self.delay = delay
        self.requests: list[dict] = []  # every chat/completions body received
        self.headers: list[dict] = []
        self.cancelled = 0
        self.completed = 0


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _last_user_text(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content or ""
    return body.get("prompt") or ""


def _chunk(delta: dict, finish=None, model=MODEL) -> dict:
    return {
        "id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


USAGE = {"prompt_tokens": 21, "completion_tokens": 7, "total_tokens": 28,
         "prompt_tokens_details": {"cached_tokens": 16}}


def make_app(state: State | None = None) -> Starlette:
    state = state or State()

    def authorized(request: Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {state.api_key}"

    def unauthorized() -> Response:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    async def models(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        return JSONResponse({"object": "list", "data": [
            {"id": MODEL, "object": "model", "created": 0, "owned_by": "vllm", "root": "/workspace/model",
             "max_model_len": MAX_MODEL_LEN}]})

    async def health(request: Request) -> Response:
        return Response(status_code=200)

    async def version(request: Request) -> Response:
        return JSONResponse({"version": "0.34.0"})

    async def metrics(request: Request) -> Response:
        text = (
            '# HELP vllm:num_requests_running Number of requests running\n'
            f'vllm:num_requests_running{{model_name="{MODEL}"}} 1.0\n'
            f'vllm:num_requests_waiting{{model_name="{MODEL}"}} 0.0\n'
            f'vllm:kv_cache_usage_perc{{model_name="{MODEL}"}} 0.12\n'
            f'vllm:prompt_tokens_total{{model_name="{MODEL}"}} {1000 + 50 * state.completed}\n'
            f'vllm:generation_tokens_total{{model_name="{MODEL}"}} {400 + 20 * state.completed}\n'
            f'vllm:prefix_cache_queries_total{{model_name="{MODEL}"}} {2000 + 40 * state.completed}\n'
            f'vllm:prefix_cache_hits_total{{model_name="{MODEL}"}} {1500 + 30 * state.completed}\n'
            f'vllm:num_preemptions_total{{model_name="{MODEL}"}} 0.0\n'
            f'vllm:request_success_total{{finished_reason="stop",model_name="{MODEL}"}} {state.completed}\n'
            f'vllm:cache_config_info{{block_size="16",cache_dtype="fp8",gpu_memory_utilization="0.9",'
            f'num_gpu_blocks="20000",model_name="{MODEL}"}} 1.0\n'
        )
        return PlainTextResponse(text)

    async def chat(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        body = await request.json()
        state.requests.append(body)
        state.headers.append(dict(request.headers))
        text = _last_user_text(body).lower()
        if "overflow" in text:
            return JSONResponse({"error": {"message": "This model's maximum context length is 65536 tokens. "
                                           "However, you requested 70000 tokens.", "type": "BadRequestError",
                                           "param": None, "code": 400}}, status_code=400)
        has_tool_result = any(m.get("role") == "tool" for m in body.get("messages") or [])
        want_tool = "tool" in text and bool(body.get("tools")) and not has_tool_result
        thinking = "think" in text and (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False
        answer = "The weather is sunny." if has_tool_result else "Hello from fake vLLM."
        tool_args = '{"city": "Paris", "unit": "celsius"}'

        if not body.get("stream"):
            message: dict = {"role": "assistant", "content": answer}
            finish = "stop"
            if thinking:
                message["reasoning_content"] = "Let me think. "
            if want_tool:
                message["content"] = None
                message["tool_calls"] = [{"id": "call_abc", "type": "function",
                                          "function": {"name": "get_weather", "arguments": tool_args}}]
                finish = "tool_calls"
            state.completed += 1
            return JSONResponse({"id": "chatcmpl-fake", "object": "chat.completion", "model": MODEL,
                                 "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                 "usage": USAGE})

        include_usage = (body.get("stream_options") or {}).get("include_usage")

        async def gen():
            try:
                yield _sse(_chunk({"role": "assistant", "content": ""}))
                if "slow" in text:
                    await asyncio.sleep(30)
                if state.delay:
                    await asyncio.sleep(state.delay)
                if thinking:
                    for piece in ("Let me ", "think. "):
                        yield _sse(_chunk({"reasoning_content": piece}))
                        await asyncio.sleep(state.delay)
                if want_tool:
                    yield _sse(_chunk({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                                       "function": {"name": "get_weather", "arguments": ""}}]}))
                    for piece in (tool_args[:9], tool_args[9:20], tool_args[20:]):
                        yield _sse(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}))
                    yield _sse(_chunk({}, finish="tool_calls"))
                else:
                    words = answer.split(" ")
                    for i, word in enumerate(words):
                        if "midfail" in text and i == 2:
                            yield _sse({"error": {"message": "engine crashed", "type": "InternalServerError", "code": 500}})
                            return
                        yield _sse(_chunk({"content": word + (" " if i < len(words) - 1 else "")}))
                        await asyncio.sleep(state.delay)
                    yield _sse(_chunk({}, finish="stop"))
                if include_usage:
                    yield _sse({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": MODEL,
                                "choices": [], "usage": USAGE})
                yield b"data: [DONE]\n\n"
                state.completed += 1
            except asyncio.CancelledError:
                state.cancelled += 1
                raise

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def completions(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        body = await request.json()
        state.requests.append(body)
        text = "raw completion"
        if not body.get("stream"):
            state.completed += 1
            return JSONResponse({"id": "cmpl-fake", "object": "text_completion", "model": MODEL,
                                 "choices": [{"index": 0, "text": text, "finish_reason": "stop"}], "usage": USAGE})

        async def gen():
            for word in ("raw ", "completion"):
                yield _sse({"id": "cmpl-fake", "object": "text_completion", "model": MODEL,
                            "choices": [{"index": 0, "text": word, "finish_reason": None}]})
            yield _sse({"id": "cmpl-fake", "object": "text_completion", "model": MODEL,
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                yield _sse({"id": "cmpl-fake", "model": MODEL, "choices": [], "usage": USAGE})
            yield b"data: [DONE]\n\n"
            state.completed += 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def tokenize(request: Request) -> Response:
        if not authorized(request):
            return unauthorized()
        body = await request.json()
        return JSONResponse({"count": 3, "max_model_len": MAX_MODEL_LEN, "tokens": [1, 2, 3], "echo_model": body.get("model")})

    app = Starlette(routes=[
        Route("/v1/models", models), Route("/health", health), Route("/version", version),
        Route("/metrics", metrics),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/completions", completions, methods=["POST"]),
        Route("/v1/tokenize", tokenize, methods=["POST"]),
    ])
    app.state.fake = state
    return app


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default="test-key")
    ap.add_argument("--delay", type=float, default=0.05, help="seconds between streamed tokens")
    a = ap.parse_args()
    uvicorn.run(make_app(State(a.api_key, a.delay)), host=a.host, port=a.port, log_level="warning")
