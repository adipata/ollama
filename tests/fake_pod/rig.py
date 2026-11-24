"""Test rig: the proxy application in managed mode wired to a FakePod."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from ollama_vllm_proxy.app import Context, create_app
from ollama_vllm_proxy.config import Settings
from ollama_vllm_proxy.pod.deploy import PodManager, PodSettings

from .server import FakeHF, FakePod


class Rig:
    def __init__(self, tmp_path: Path, gpus: str = "1x96"):
        self.pod = FakePod(tmp_path / "pod", gpus=gpus)
        self.settings = Settings(data_dir=str(tmp_path / "data"), retention_days=0,
                                 ssh_key_path=str(self.pod.key_path))
        self.hf = FakeHF()
        self.boots = 0

    async def start(self) -> "Rig":
        await self.pod.start()
        self.boot()
        return self

    def boot(self) -> None:
        """Create the proxy side (again): a new Context and PodManager on the same data directory."""
        self.boots += 1
        self.ctx = Context(self.settings)  # real upstream client: requests go through the SSH forward
        self.ctx.store.start()
        self.cfg = PodSettings(remote_root=str(self.pod.root), vllm_port=self.pod.vllm_port,
                               helper_port=self.pod.helper_port, poll_interval_s=0.1, ready_timeout_s=30,
                               reconnect_wait_s=20, stop_grace_s=3)
        self.mgr = PodManager(self.ctx, self.cfg, self.hf)
        self.mgr.ssh_backoff = (0.05, 0.1, 0.2)
        self.ctx.extras["pod"] = self.mgr
        self.app = create_app(ctx=self.ctx)
        self.key = self.ctx.keys.create(f"tests-{self.boots}")["key"]
        transport = httpx.ASGITransport(app=self.app)
        self.admin = httpx.AsyncClient(transport=transport, base_url="http://proxy.test", timeout=60,
                                       headers={"Authorization": f"Bearer {self.ctx.admin_token}"})
        self.http = httpx.AsyncClient(transport=transport, base_url="http://proxy.test", timeout=60,
                                      headers={"Authorization": f"Bearer {self.key}"})

    async def shutdown_proxy(self) -> None:
        await self.mgr.shutdown()
        await self.admin.aclose()
        await self.http.aclose()
        await self.ctx.aclose()

    async def close(self) -> None:
        await self.shutdown_proxy()
        await self.pod.close()

    # -- shortcuts ----------------------------------------------------------------------

    async def connect(self) -> str:
        """Paste the command, read the fingerprint, accept it."""
        resp = await self.admin.put("/admin/api/pod", json={"ssh_command": self.pod.ssh_command})
        assert resp.status_code == 200, resp.text
        first = (await self.admin.post("/admin/api/pod/connect", json={})).json()
        assert first["state"] == "confirm", first
        done = await self.admin.post("/admin/api/pod/connect", json={"accept_fingerprint": first["fingerprint"]})
        assert done.status_code == 200 and done.json()["state"] == "connected", done.text
        return first["fingerprint"]

    def profile_id(self, name: str = "qwen3.8-27b") -> int:
        return next(p["id"] for p in self.mgr.profiles.list() if p["name"] == name)

    async def status(self) -> dict[str, Any]:
        return (await self.admin.get("/admin/api/deploy/status")).json()

    async def wait_idle(self, timeout: float = 40.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while self.mgr.busy:
            assert time.monotonic() < deadline, f"deployment still running: {await self.status()}"
            await asyncio.sleep(0.05)
        return await self.status()

    async def deploy(self, expect: int = 202, **body: Any) -> dict[str, Any]:
        body.setdefault("profile_id", self.profile_id())
        resp = await self.admin.post("/admin/api/deploy", json=body)
        assert resp.status_code == expect, resp.text
        if expect != 202:
            return resp.json()
        return await self.wait_idle()

    async def wait_for(self, predicate, timeout: float = 20.0, what: str = "condition") -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, f"timed out waiting for {what}"
            await asyncio.sleep(0.05)

    def steps(self, status: dict[str, Any]) -> dict[str, str]:
        return {s["id"]: s["status"] for s in status["deployment"]["steps"]}

    def vllm_argv(self) -> list[str]:
        return self.pod.records("vllm")[-1]["argv"]
