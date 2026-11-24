"""Proxy behaviour against the in-process fake vLLM."""

from __future__ import annotations

import json
import sqlite3

from conftest import UPSTREAM_KEY


def ndjson(resp) -> list[dict]:
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def sse(resp) -> list:
    out = []
    for block in resp.text.split("\n\n"):
        if block.startswith("data: "):
            data = block[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


CHAT = {"model": "dev-test", "messages": [{"role": "user", "content": "hi"}]}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]


# -- liveness and auth ----------------------------------------------------------


async def test_liveness_is_open(env):
    r = await env.anon.get("/")
    assert r.status_code == 200 and r.text == "Ollama is running"
    assert (await env.anon.head("/")).status_code == 200
    assert (await env.anon.get("/healthz")).json()["backend"] == "ready"


async def test_every_api_route_needs_a_key(env):
    for method, path in [("GET", "/api/tags"), ("GET", "/api/version"), ("POST", "/api/chat"), ("GET", "/api/ps"),
                         ("POST", "/api/show"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions"),
                         ("POST", "/v1/anything")]:
        r = await env.anon.request(method, path, json={})
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"] == "Bearer"
    assert (await env.anon.get("/api/tags")).json() == {"error": "invalid or missing API key"}
    v1 = (await env.anon.get("/v1/models")).json()
    assert v1["error"]["type"] == "authentication_error" and v1["error"]["code"] == "invalid_api_key"
    assert env.ctx.rejected >= 8
    assert env.rows() == []  # rejected calls never create rows


async def test_key_variants(env):
    assert (await env.anon.get("/api/tags", headers={"x-api-key": env.key})).status_code == 200
    assert (await env.anon.get("/api/tags", headers={"Authorization": "Bearer sk-ovp-wrong"})).status_code == 401
    # the admin token is not an API key, and an API key is not an admin token
    assert (await env.anon.get("/api/tags", headers={"Authorization": f"Bearer {env.ctx.admin_token}"})).status_code == 401
    assert (await env.http.get("/admin/api/keys")).status_code == 401
    assert (await env.anon.get("/admin/api/keys")).status_code == 401


async def test_disable_and_revoke_take_effect(env):
    keys = (await env.admin.get("/admin/api/keys")).json()["keys"]
    key_id = keys[0]["id"]
    assert (await env.admin.patch(f"/admin/api/keys/{key_id}", json={"enabled": False})).json()["status"] == "disabled"
    assert (await env.http.get("/api/tags")).status_code == 401
    await env.admin.patch(f"/admin/api/keys/{key_id}", json={"enabled": True})
    assert (await env.http.get("/api/tags")).status_code == 200
    assert (await env.admin.delete(f"/admin/api/keys/{key_id}")).json()["status"] == "revoked"
    assert (await env.http.get("/api/tags")).status_code == 401
    assert (await env.admin.patch(f"/admin/api/keys/{key_id}", json={"enabled": True})).status_code == 400


async def test_key_is_shown_once_and_never_stored(env):
    r = await env.admin.post("/admin/api/keys", json={"name": "laptop"})
    assert r.status_code == 201
    key = r.json()["key"]
    assert key.startswith("sk-ovp-") and len(key) == 39
    listing = (await env.admin.get("/admin/api/keys")).text
    assert key not in listing and "key_sha256" not in listing
    assert (await env.admin.post("/admin/api/keys", json={"name": "laptop"})).status_code == 400
    assert (await env.admin.post("/admin/api/keys", json={"name": "bad name!"})).status_code == 400
    env.ctx.store.flush()
    dump = "\n".join(sqlite3.connect(env.settings.db_path).iterdump())
    assert key not in dump


async def test_admin_pages_carry_no_data_and_are_hardened(env):
    r = await env.anon.get("/admin")
    assert r.status_code == 200 and "API keys" in r.text
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert env.ctx.admin_token not in r.text
    assert (await env.anon.get("/admin/static/common.js")).status_code == 200
    assert (await env.anon.get("/admin/static/..%2Fapp.py")).status_code == 404


# -- /v1 pass-through -------------------------------------------------------------


async def test_v1_chat_non_streaming_swaps_auth(env):
    r = await env.http.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello from fake vLLM."
    assert env.fake.headers[-1]["authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert env.key not in json.dumps(env.fake.headers)
    row = env.rows()[-1]
    assert (row["api"], row["endpoint"], row["outcome"], row["actor_id"]) == ("openai", "chat.completions", "ok", "tests")
    assert (row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"]) == (21, 7, 16)


async def test_v1_stream_hides_forced_usage_chunk(env):
    r = await env.http.post("/v1/chat/completions", json={**CHAT, "stream": True})
    events = sse(r)
    assert events[-1] == "[DONE]"
    assert all(e == "[DONE]" or e.get("choices") for e in events)  # no usage-only chunk
    assert "".join(e["choices"][0]["delta"].get("content") or "" for e in events[:-1]) == "Hello from fake vLLM."
    assert env.fake.requests[-1]["stream_options"] == {"include_usage": True}
    row = env.rows()[-1]
    assert row["outcome"] == "ok" and row["completion_tokens"] == 7 and row["include_usage_forced"] == 1
    assert row["first_payload_ms"] is not None and row["end_ms"] >= row["first_payload_ms"]

    r = await env.http.post("/v1/chat/completions", json={**CHAT, "stream": True, "stream_options": {"include_usage": True}})
    assert any(e != "[DONE]" and e.get("usage") for e in sse(r))


async def test_v1_model_alias_and_name_restored(tmp_path):
    from conftest import Env

    e = Env(tmp_path, model_aliases={"qwen3:latest": "dev-test"})
    try:
        r = await e.http.post("/v1/chat/completions", json={**CHAT, "model": "qwen3:latest"})
        assert r.json()["model"] == "qwen3:latest"
        assert e.fake.requests[-1]["model"] == "dev-test"
        ids = [m["id"] for m in (await e.http.get("/v1/models")).json()["data"]]
        assert ids == ["dev-test", "qwen3:latest"]
        r = await e.http.post("/v1/chat/completions", json={**CHAT, "model": "whatever"})
        assert r.status_code == 200 and e.fake.requests[-1]["model"] == "dev-test"  # falls back by default
    finally:
        await e.aclose()


async def test_strict_models_rejects_unknown(tmp_path):
    from conftest import Env

    e = Env(tmp_path, strict_models=True)
    try:
        assert (await e.http.post("/v1/chat/completions", json={**CHAT, "model": "nope"})).status_code == 404
        r = await e.http.post("/api/chat", json={**CHAT, "model": "nope"})
        assert r.status_code == 404 and "not found" in r.json()["error"]
        assert (await e.http.post("/api/chat", json={**CHAT, "model": "dev-test:latest", "stream": False})).status_code == 200
    finally:
        await e.aclose()


async def test_v1_reasoning_field_normalised(env):
    body = {"model": "dev-test", "messages": [{"role": "user", "content": "think please"}]}
    msg = (await env.http.post("/v1/chat/completions", json=body)).json()["choices"][0]["message"]
    assert msg["reasoning"] == msg["reasoning_content"] == "Let me think. "


async def test_v1_upstream_error_passes_through(env):
    body = {"model": "dev-test", "messages": [{"role": "user", "content": "overflow"}]}
    r = await env.http.post("/v1/chat/completions", json=body)
    assert r.status_code == 400 and "maximum context length" in r.json()["error"]["message"]
    row = env.rows()[-1]
    assert row["outcome"] == "upstream_error" and row["context_error"] == 1


async def test_v1_generic_passthrough_and_completions(env):
    r = await env.http.post("/v1/tokenize", json={"model": "dev-test:latest", "prompt": "abc"})
    assert r.status_code == 200 and r.json()["count"] == 3 and r.json()["echo_model"] == "dev-test"
    assert (await env.http.get("/v1/nope")).status_code == 404
    r = await env.http.post("/v1/completions", json={"model": "dev-test", "prompt": "x"})
    assert r.json()["choices"][0]["text"] == "raw completion"
    assert (await env.http.post("/v1/embeddings", json={"input": "x"})).status_code == 501


# -- /api (Ollama) ------------------------------------------------------------------


async def test_ollama_discovery(env):
    assert (await env.http.get("/api/version")).json() == {"version": "0.12.0"}
    tags = (await env.http.get("/api/tags")).json()["models"]
    assert tags[0]["name"] == "dev-test" and len(tags[0]["digest"]) == 64
    assert (await env.http.get("/api/ps")).json()["models"][0]["context_length"] == 65536
    show = (await env.http.post("/api/show", json={"model": "dev-test:latest"})).json()
    assert show["capabilities"] == ["completion", "tools", "thinking"]
    assert show["model_info"]["dev.context_length"] == 65536


async def test_ollama_chat_non_streaming(env):
    r = await env.http.post("/api/chat", json={**CHAT, "stream": False})
    body = r.json()
    assert body["message"] == {"role": "assistant", "content": "Hello from fake vLLM."}
    assert body["done"] is True and body["done_reason"] == "stop" and body["model"] == "dev-test"
    assert body["prompt_eval_count"] == 21 and body["eval_count"] == 7 and body["total_duration"] > 0
    assert body["created_at"].endswith("Z")
    sent = env.fake.requests[-1]
    assert sent["stream"] is False and "stream_options" not in sent


async def test_ollama_chat_streams_ndjson_by_default(env):
    r = await env.http.post("/api/chat", json=CHAT)
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = ndjson(r)
    assert [l["done"] for l in lines].count(True) == 1 and lines[-1]["done"] is True
    assert "".join(l["message"]["content"] for l in lines) == "Hello from fake vLLM."
    assert lines[-1]["eval_count"] == 7 and lines[-1]["done_reason"] == "stop"
    assert env.fake.requests[-1]["stream_options"] == {"include_usage": True}
    row = env.rows()[-1]
    assert (row["api"], row["endpoint"], row["outcome"], row["stream"]) == ("ollama", "chat", "ok", 1)


async def test_ollama_options_format_and_think_mapping(env):
    body = {
        "model": "dev-test", "stream": False, "think": False, "format": "json", "keep_alive": "5m",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi", "images": ["iVBORw0KGgo="]}],
        "options": {"num_predict": 99, "temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "seed": 7,
                    "stop": "END", "repeat_penalty": 1.1, "presence_penalty": 0.1, "frequency_penalty": 0.2,
                    "num_ctx": 4096, "num_gpu": 1},
    }
    assert (await env.http.post("/api/chat", json=body)).status_code == 200
    sent = env.fake.requests[-1]
    assert sent["max_tokens"] == 99 and sent["temperature"] == 0.2 and sent["top_k"] == 40 and sent["min_p"] == 0.05
    assert sent["stop"] == ["END"] and sent["repetition_penalty"] == 1.1 and sent["seed"] == 7
    assert "num_ctx" not in sent and "keep_alive" not in sent and "options" not in sent
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["messages"][1]["content"][1]["image_url"]["url"] == "data:image/png;base64,iVBORw0KGgo="

    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    await env.http.post("/api/chat", json={**CHAT, "stream": False, "format": schema, "options": {"num_predict": -1}})
    sent = env.fake.requests[-1]
    assert sent["response_format"]["json_schema"]["schema"] == schema and "max_tokens" not in sent
    assert (await env.http.post("/api/chat", json={**CHAT, "format": 5})).status_code == 400


async def test_ollama_thinking_stream(env):
    body = {"model": "dev-test", "think": True, "messages": [{"role": "user", "content": "think please"}]}
    lines = ndjson(await env.http.post("/api/chat", json=body))
    assert "".join(l["message"].get("thinking", "") for l in lines) == "Let me think. "
    assert "".join(l["message"]["content"] for l in lines) == "Hello from fake vLLM."
    assert env.fake.requests[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    row = env.rows()[-1]
    assert row["first_reasoning_ms"] is not None and row["first_text_ms"] >= row["first_reasoning_ms"]


async def test_ollama_tool_call_round_trip(env):
    ask = {"model": "dev-test", "tools": TOOLS, "messages": [{"role": "user", "content": "use the tool"}]}
    lines = ndjson(await env.http.post("/api/chat", json=ask))
    calls = [l for l in lines if l["message"].get("tool_calls")]
    assert len(calls) == 1  # fragments are accumulated into one complete call
    call = calls[0]["message"]["tool_calls"][0]
    assert call["function"] == {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
    assert lines[-1]["done"] is True and lines[-1]["done_reason"] == "stop"

    # non-streaming gives the same shape
    msg = (await env.http.post("/api/chat", json={**ask, "stream": False})).json()["message"]
    assert msg["tool_calls"][0]["function"]["arguments"] == {"city": "Paris", "unit": "celsius"}

    # send the result back the way ollama clients do (no ids, tool_name)
    history = ask["messages"] + [
        {"role": "assistant", "content": "", "tool_calls": [{"function": call["function"]}]},
        {"role": "tool", "tool_name": "get_weather", "content": "sunny"},
    ]
    final = (await env.http.post("/api/chat", json={**ask, "messages": history, "stream": False})).json()
    assert final["message"]["content"] == "The weather is sunny."
    sent = env.fake.requests[-1]["messages"]
    assistant, tool = sent[1], sent[2]
    assert assistant["tool_calls"][0]["type"] == "function"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris", "unit": "celsius"}
    assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
    # deterministic ids: the same history serialises identically (prefix cache stays warm)
    await env.http.post("/api/chat", json={**ask, "messages": history, "stream": False})
    assert env.fake.requests[-1]["messages"] == sent


async def test_ollama_generate(env):
    r = await env.http.post("/api/generate", json={"model": "dev-test", "prompt": "hi", "system": "be brief", "stream": False})
    body = r.json()
    assert body["response"] == "Hello from fake vLLM." and body["done"] is True and body["context"] == []
    assert [m["role"] for m in env.fake.requests[-1]["messages"]] == ["system", "user"]

    lines = ndjson(await env.http.post("/api/generate", json={"model": "dev-test", "prompt": "hi"}))
    assert "".join(l["response"] for l in lines) == "Hello from fake vLLM." and lines[-1]["done"] is True

    lines = ndjson(await env.http.post("/api/generate", json={"model": "dev-test", "prompt": "x", "raw": True}))
    assert "".join(l["response"] for l in lines) == "raw completion"
    assert env.fake.requests[-1]["prompt"] == "x" and "messages" not in env.fake.requests[-1]


async def test_ollama_load_and_unload_shortcuts(env):
    before = len(env.fake.requests)
    assert (await env.http.post("/api/generate", json={"model": "dev-test"})).json()["done_reason"] == "load"
    assert (await env.http.post("/api/generate", json={"model": "dev-test", "keep_alive": 0})).json()["done_reason"] == "unload"
    assert (await env.http.post("/api/chat", json={"model": "dev-test", "messages": []})).json()["done_reason"] == "load"
    assert len(env.fake.requests) == before  # no upstream call


async def test_ollama_errors(env):
    body = {"model": "dev-test", "messages": [{"role": "user", "content": "overflow"}]}
    r = await env.http.post("/api/chat", json=body)
    assert r.status_code == 400 and "maximum context length" in r.json()["error"]
    lines = ndjson(await env.http.post("/api/chat", json={"model": "dev-test", "messages": [{"role": "user", "content": "midfail"}]}))
    assert lines[-1] == {"error": "engine crashed"} and not any(l.get("done") for l in lines)
    assert env.rows()[-1]["outcome"] == "error_event"
    assert (await env.http.post("/api/chat", content=b"not json")).status_code == 400
    assert (await env.http.post("/api/chat", json={"messages": []})).status_code == 400


async def test_ollama_stubs(env):
    lines = ndjson(await env.http.post("/api/pull", json={"model": "dev-test"}))
    assert lines[-1] == {"status": "success"}
    assert (await env.http.post("/api/pull", json={"model": "llama3"})).status_code == 404
    assert (await env.http.post("/api/embed", json={"model": "dev-test", "input": "x"})).status_code == 501
    for method, path in [("POST", "/api/push"), ("POST", "/api/create"), ("POST", "/api/copy"), ("DELETE", "/api/delete")]:
        assert (await env.http.request(method, path, json={})).status_code == 501


async def test_session_header_is_recorded_only_when_opaque(env):
    await env.http.post("/api/chat", json={**CHAT, "stream": False}, headers={"X-Session-Id": "solo-1"})
    await env.http.post("/api/chat", json={**CHAT, "stream": False}, headers={"X-Session-Id": "has spaces; drop"})
    rows = env.rows()
    assert [r["session_id"] for r in rows[-2:]] == ["solo-1", None]


async def test_no_prompt_or_completion_text_in_database(env):
    secret = "ZEBRA-PROMPT-MARKER"
    await env.http.post("/api/chat", json={"model": "dev-test", "stream": False, "messages": [{"role": "user", "content": secret}]})
    await env.http.post("/v1/chat/completions", json={"model": "dev-test", "messages": [{"role": "user", "content": secret}]})
    env.ctx.store.flush()
    dump = "\n".join(sqlite3.connect(env.settings.db_path).iterdump())
    assert secret not in dump and "Hello from fake vLLM" not in dump and UPSTREAM_KEY not in dump


# -- managed mode without a backend -------------------------------------------------


async def test_managed_mode_without_backend_returns_503(managed):
    r = await managed.http.post("/api/chat", json=CHAT)
    assert r.status_code == 503 and "backend not deployed" in r.json()["error"]
    r = await managed.http.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 503 and r.json()["error"]["type"] == "service_unavailable"
    assert (await managed.http.get("/api/tags")).json() == {"models": []}
    assert (await managed.anon.get("/healthz")).json()["backend"] == "not_ready"
    status = (await managed.admin.get("/admin/api/status")).json()
    assert status["backend"]["state"] == "no_pod" and status["backend"]["mode"] == "managed"

    # the deployment code switches the backend on
    managed.ctx.backend.set_ready(base_url="http://vllm.test", api_key=UPSTREAM_KEY, served_model="dev-test",
                                  context_length=131072, label="rtx-pro-6000x1-test-bf16-131072")
    r = await managed.http.post("/api/chat", json={**CHAT, "stream": False})
    assert r.status_code == 200
    assert managed.rows()[-1]["deployment_id"] == "rtx-pro-6000x1-test-bf16-131072"
    assert (await managed.http.get("/api/ps")).json()["models"][0]["context_length"] == 131072
