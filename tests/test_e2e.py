"""Real sockets: uvicorn proxy + uvicorn fake vLLM, driven by the official client libraries."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fake_vllm import State, make_app

from ollama_vllm_proxy.app import Context, create_app
from ollama_vllm_proxy.config import Settings


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, app, port: int):
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return
            time.sleep(0.02)
        raise RuntimeError("server did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture
def live(tmp_path):
    fake = State(api_key="upstream-secret", delay=0.01)
    up_port, proxy_port = free_port(), free_port()
    upstream = Server(make_app(fake), up_port)
    upstream.start()
    settings = Settings(data_dir=str(tmp_path / "data"), vllm_base_url=f"http://127.0.0.1:{up_port}",
                        vllm_api_key="upstream-secret", port=proxy_port, retention_days=0)
    ctx = Context(settings)
    key = ctx.keys.create("e2e")["key"]
    proxy = Server(create_app(ctx=ctx), proxy_port)
    proxy.start()
    yield {"url": f"http://127.0.0.1:{proxy_port}", "key": key, "fake": fake, "ctx": ctx}
    proxy.stop()
    upstream.stop()


def test_openai_sdk(live):
    from openai import OpenAI

    client = OpenAI(base_url=live["url"] + "/v1", api_key=live["key"])
    assert [m.id for m in client.models.list().data] == ["dev-test"]
    out = client.chat.completions.create(model="dev-test", messages=[{"role": "user", "content": "hi"}])
    assert out.choices[0].message.content == "Hello from fake vLLM."
    stream = client.chat.completions.create(model="dev-test", messages=[{"role": "user", "content": "hi"}], stream=True)
    assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices) == "Hello from fake vLLM."


def test_ollama_python_client(live):
    from ollama import Client, ResponseError

    client = Client(host=live["url"], headers={"Authorization": f"Bearer {live['key']}"})
    assert [m.model for m in client.list().models] == ["dev-test"]
    assert "tools" in client.show("dev-test").capabilities
    reply = client.chat(model="dev-test", messages=[{"role": "user", "content": "hi"}])
    assert reply.message.content == "Hello from fake vLLM." and reply.done and reply.eval_count == 7
    text = "".join(part.message.content for part in client.chat(
        model="dev-test", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert text == "Hello from fake vLLM."

    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get the weather for a city."""
        return "sunny"

    messages = [{"role": "user", "content": "use the tool"}]
    first = client.chat(model="dev-test", messages=messages, tools=[get_weather])
    call = first.message.tool_calls[0]
    assert call.function.name == "get_weather" and call.function.arguments == {"city": "Paris", "unit": "celsius"}
    messages += [first.message, {"role": "tool", "tool_name": call.function.name, "content": get_weather(**call.function.arguments)}]
    final = client.chat(model="dev-test", messages=messages, tools=[get_weather])
    assert final.message.content == "The weather is sunny."

    thought = client.chat(model="dev-test", messages=[{"role": "user", "content": "think please"}], think=True)
    assert thought.message.thinking == "Let me think. "
    assert client.generate(model="dev-test", prompt="hi").response == "Hello from fake vLLM."

    with pytest.raises(ResponseError) as err:
        Client(host=live["url"]).list()
    assert err.value.status_code == 401


def test_client_disconnect_cancels_upstream(live):
    fake, ctx = live["fake"], live["ctx"]
    headers = {"Authorization": f"Bearer {live['key']}"}
    body = {"model": "dev-test", "messages": [{"role": "user", "content": "slow please"}]}
    with httpx.Client(timeout=5) as client:
        with client.stream("POST", live["url"] + "/api/chat", json=body, headers=headers) as resp:
            assert resp.status_code == 200
            time.sleep(0.3)  # upstream is now parked before its first token
    for _ in range(100):
        if fake.cancelled:
            break
        time.sleep(0.05)
    assert fake.cancelled == 1  # vLLM would abort the generation here
    ctx.store.flush()
    row = ctx.store.read("SELECT outcome, cancelled FROM requests")[-1]
    assert row == {"outcome": "client_cancelled", "cancelled": 1}
    assert ctx.open_requests == 0
