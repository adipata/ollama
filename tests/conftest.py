"""Fixtures: the proxy app wired to the fake vLLM, both in-process (ASGI)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_vllm import State, make_app  # noqa: E402

from ollama_vllm_proxy.app import Context, create_app  # noqa: E402
from ollama_vllm_proxy.config import Settings  # noqa: E402

UPSTREAM_KEY = "test-key"


class Env:
    def __init__(self, tmp_path, external: bool = True, **overrides):
        self.fake = State(api_key=UPSTREAM_KEY)
        self.fake_app = make_app(self.fake)
        values = dict(data_dir=str(tmp_path / "data"), retention_days=0)
        if external:
            values.update(vllm_base_url="http://vllm.test", vllm_api_key=UPSTREAM_KEY)
        values.update(overrides)
        self.settings = Settings(**values)
        upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.fake_app))
        self.ctx = Context(self.settings, client=upstream)
        self.ctx.store.start()
        self.app = create_app(ctx=self.ctx)
        self.key = self.ctx.keys.create("tests")["key"]
        self.http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://proxy.test",
            headers={"Authorization": f"Bearer {self.key}"},
        )
        self.anon = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://proxy.test")
        self.admin = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://proxy.test",
            headers={"Authorization": f"Bearer {self.ctx.admin_token}"},
        )

    def rows(self, sql: str = "SELECT * FROM requests ORDER BY t_arrival_wall"):
        self.ctx.store.flush()
        return self.ctx.store.read(sql)

    async def aclose(self):
        for client in (self.http, self.anon, self.admin):
            await client.aclose()
        await self.ctx.aclose()


@pytest.fixture
async def env(tmp_path):
    e = Env(tmp_path)
    yield e
    await e.aclose()


@pytest.fixture
async def managed(tmp_path):
    """Managed mode with nothing deployed."""
    e = Env(tmp_path, external=False)
    yield e
    await e.aclose()
