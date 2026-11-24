"""The pod manager: connection state, the deployment state machine, Stop / Restart / Cancel.

A deployment is a sequence of steps whose status, timestamps and output are
kept in ``deployments.steps``. Every step is written as "look at what is on
the pod, then do only what is missing", which gives three properties at once:
clicking Deploy again skips finished work, a dropped SSH connection is waited
out and the step carries on, and a proxy restart resumes by running the same
sequence again.

Long-running remote work (install, download, vLLM, the GPU helper) is started
detached by ``remote/launch.sh`` and tracked by PID and exit-status files under
``<remote_root>/ovp/``. Secrets reach those processes only through 0600 files
read inside the process; they are never part of a command line.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets as pysecrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from . import fit
from . import profiles as prof
from .connstr import Connection, ConnStrError, key_problem, parse
from .logs import redact
from .ssh import HostKeyChanged, PodSSH, Result, SSHError, SSHUnavailable, quote_argv

log = logging.getLogger("ovp.pod")

REMOTE_DIR = Path(__file__).resolve().parent / "remote"
REMOTE_FILES = (
    "common.sh", "launch.sh", "job.sh", "stop.sh", "status.sh", "preflight.sh", "sizes.sh", "install_vllm.sh",
    "start_download.sh", "start_vllm.sh", "start_helper.sh", "delete_models.sh", "gpu_exporter.py",
)
STEPS = (
    ("connect", "Connect"),
    ("preflight", "Preflight and plan"),
    ("helper", "Install helper"),
    ("install", "Install vLLM"),
    ("download", "Download model"),
    ("secrets", "Write secrets"),
    ("start", "Start vLLM"),
    ("ready", "Wait until ready"),
    ("activate", "Activate"),
)
SECRET_VLLM_KEY = "vllm_api_key"
SECRET_HF_TOKEN = "hf_token"
_OUTPUT_LIMIT = 6000
_CONTEXT_LIMIT_PATTERNS = (
    re.compile(r"estimated maximum model length is (\d+)"),
    re.compile(r"maximum number of tokens that can be stored in KV cache \((\d+)\)"),
)
_ROOT = re.compile(r"^/[A-Za-z0-9._/-]{1,200}$")


class PodError(Exception):
    """An action cannot be carried out; ``status`` is the HTTP status for the API."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


class StepFailed(Exception):
    pass


@dataclass
class PodSettings:
    """Settings of the pod module. Read from the environment until they move into config.Settings."""

    remote_root: str = "/workspace"
    vllm_port: int = 8000
    helper_port: int = 9400
    ready_timeout_s: float = 1800.0  # model load can take many minutes
    poll_interval_s: float = 2.0
    reconnect_wait_s: float = 900.0  # how long a running step waits for the SSH connection to return
    stop_grace_s: int = 20

    @classmethod
    def from_env(cls) -> "PodSettings":
        e = os.environ
        out = cls(
            remote_root=e.get("POD_REMOTE_ROOT", "/workspace").rstrip("/") or "/workspace",
            vllm_port=int(e.get("POD_VLLM_PORT", "8000")),
            helper_port=int(e.get("POD_HELPER_PORT", "9400")),
            ready_timeout_s=float(e.get("POD_READY_TIMEOUT_S", "1800")),
            poll_interval_s=float(e.get("POD_POLL_INTERVAL_S", "2")),
        )
        out.check()
        return out

    def check(self) -> None:
        if not _ROOT.match(self.remote_root) or ".." in self.remote_root.split("/"):
            raise ValueError(f"POD_REMOTE_ROOT must be a plain absolute path, got {self.remote_root!r}")
        for port in (self.vllm_port, self.helper_port):
            if not 1 <= port <= 65535:
                raise ValueError("POD_VLLM_PORT and POD_HELPER_PORT must be valid ports")

    @property
    def ovp(self) -> str:
        return f"{self.remote_root}/ovp"

    @property
    def venv(self) -> str:
        return f"{self.remote_root}/venv"

    @property
    def vllm_log(self) -> str:
        return f"{self.remote_root}/vllm.log"


def parse_kv(text: str) -> dict[str, list[str]]:
    """key=value lines (keys may repeat) as printed by preflight.sh."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            out.setdefault(key, []).append(value.strip())
    return out


def context_limit_from_log(text: str) -> int | None:
    """The context vLLM says would fit, when it refused to start for lack of KV cache."""
    for pattern in _CONTEXT_LIMIT_PATTERNS:
        found = pattern.findall(text)
        if found:
            return int(found[-1])
    return None


def new_steps() -> list[dict[str, Any]]:
    return [{"id": sid, "title": title, "status": "pending", "started_at": None, "ended_at": None, "output": ""}
            for sid, title in STEPS]


class PodManager:
    def __init__(self, ctx: Any, cfg: PodSettings | None = None, hf: fit.HuggingFace | None = None):
        self.ctx = ctx
        self.cfg = cfg or PodSettings.from_env()
        self.hf = hf or fit.HuggingFace()
        self.profiles = prof.ProfileStore(ctx.store)
        self.disabled = bool(ctx.backend.external)
        self.ssh: PodSSH | None = None
        self.ssh_backoff: tuple[float, ...] = (1, 2, 5, 10, 20, 30)
        self.deployment: dict[str, Any] | None = None
        self.pod_facts: dict[str, Any] = {}  # GPUs and limits read at connect / preflight
        self._task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._subs: set[asyncio.Queue] = set()
        self._connect_lock = asyncio.Lock()
        self._ready_before_drop = False
        self._helper_ok = False
        self._started_jobs: set[str] = set()

    # ------------------------------------------------------------------ lifecycle

    async def startup(self) -> None:
        self.profiles.seed()
        self.deployment = self._load_last_deployment()
        if self.disabled:
            return
        row = self.pod_row()
        if row and row.get("fingerprint") and row.get("state") == "connected":
            # The key confirmed before the restart is still pinned: reconnect on our own, but only to that key.
            self.ctx.backend.set_state("no_pod", "reconnecting to the pod after a proxy restart")
            self._spawn(self._auto_connect(row["fingerprint"]))
        elif self.deployment and self.deployment["state"] == "running":
            self._finish(self.deployment, "failed", "interrupted by a proxy restart while no pod was connected")

    async def shutdown(self) -> None:
        """Stop local activity. Nothing on the pod is touched: vLLM keeps running."""
        tasks = [t for t in (self._task, *self._bg) if t and not t.done()]
        self._shutting_down = True
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.ssh is not None:
            await self.ssh.close()
        await self.hf.aclose()

    _shutting_down = False

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    async def _auto_connect(self, fingerprint: str) -> None:
        delays = (0, 2, 5, 10, 20, 30, 60, 60, 60, 60)
        for delay in delays:
            await asyncio.sleep(delay)
            try:
                result = await self.connect(fingerprint)
            except PodError as exc:
                self.ctx.backend.set_state("no_pod", f"reconnect after restart failed: {exc.message}")
                self.publish()
                continue
            if result["state"] != "connected":
                self._forget_fingerprint()
                self.ctx.backend.set_state("no_pod", "the pod's host key changed: confirm the new fingerprint")
                break
            return
        dep = self.deployment
        if dep and dep["state"] == "running" and not self.busy:
            self._finish(dep, "failed", "interrupted by a proxy restart; the pod could not be reached again")
        self.publish()

    # ------------------------------------------------------------------ small helpers

    def require_enabled(self) -> None:
        if self.disabled:
            raise PodError("pod management is disabled: the proxy runs in external mode (VLLM_BASE_URL is set)", 409)

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def secret_values(self) -> list[str]:
        """Values that must never leave the process in a log line or step output."""
        store = self.ctx.store
        return [v for v in (store.get_secret(SECRET_VLLM_KEY), store.get_secret(SECRET_HF_TOKEN)) if v]

    def pod_row(self) -> dict[str, Any] | None:
        return self.ctx.store.read_one("SELECT * FROM pod WHERE id = 1")

    def _set_pod_state(self, state: str) -> None:
        self.ctx.store.write("UPDATE pod SET state = ?, updated_at = ? WHERE id = 1", (state, time.time()))

    def _forget_fingerprint(self) -> None:
        self.ctx.store.write("UPDATE pod SET fingerprint = NULL, state = 'saved', updated_at = ? WHERE id = 1",
                             (time.time(),))

    # ------------------------------------------------------------------ events

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self) -> None:
        """Wake the SSE listeners. They fetch the snapshot themselves, so a slow one only skips updates."""
        for q in self._subs:
            if q.empty():
                q.put_nowait(True)

    # ------------------------------------------------------------------ status

    def describe(self) -> dict[str, Any]:
        row = self.pod_row()
        pod: dict[str, Any] | None = None
        if row:
            ssh = self.ssh
            pod = {
                "ssh_command": row["ssh_command"], "host": row["host"], "port": row["port"], "user": row["user"],
                "key_path": row["key_path"], "key_problem": key_problem(row["key_path"]),
                "fingerprint": row["fingerprint"],
                "state": ssh.state if ssh else "saved",
                "detail": ssh.detail if ssh else "",
                "connected": bool(ssh and ssh.up),
                "facts": self.pod_facts if ssh else {},
            }
        dep = self.deployment
        return {
            "external": self.disabled,
            "backend": self.ctx.backend.describe(),
            "pod": pod,
            "remote_root": self.cfg.remote_root,
            "busy": self.busy,
            "hf_token_set": bool(self.ctx.store.get_secret(SECRET_HF_TOKEN)),
            "deployment": self._public_deployment(dep) if dep else None,
        }

    @staticmethod
    def _public_deployment(dep: dict[str, Any]) -> dict[str, Any]:
        plan = dep.get("plan") or {}
        return {
            "id": dep["id"], "profile_id": dep["profile_id"], "profile_name": dep["profile_name"],
            "label": dep.get("label"), "state": dep["state"], "kind": plan.get("kind", "deploy"),
            "error": plan.get("error", ""), "offer": plan.get("offer"),
            "started_at": dep["started_at"], "ended_at": dep.get("ended_at"),
            "steps": dep["steps"],
            "plan": {k: plan.get(k) for k in ("variant", "repo", "context", "tensor_parallel", "gpu_name", "gpu_count",
                                              "forced", "command", "download_bytes", "context_requested")},
        }

    # ------------------------------------------------------------------ deployment rows

    def _load_last_deployment(self) -> dict[str, Any] | None:
        row = self.ctx.store.read_one("SELECT * FROM deployments ORDER BY id DESC LIMIT 1")
        if not row:
            return None
        for col, default in (("snapshot", {}), ("plan", {}), ("steps", [])):
            try:
                row[col] = json.loads(row[col]) if row[col] else default
            except ValueError:
                row[col] = default
        return row

    def _new_deployment(self, profile: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        dep = {
            "profile_id": profile["id"], "profile_name": profile["name"], "label": plan.get("label"),
            "snapshot": prof.snapshot(profile), "plan": plan, "steps": new_steps(), "state": "running",
            "started_at": time.time(), "ended_at": None,
        }
        dep["id"] = self.ctx.store.write(
            "INSERT INTO deployments (profile_id, profile_name, label, snapshot, plan, steps, state, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (dep["profile_id"], dep["profile_name"], dep["label"], json.dumps(dep["snapshot"]),
             json.dumps(plan), json.dumps(dep["steps"]), dep["state"], dep["started_at"]),
        )
        return dep

    def _save(self, dep: dict[str, Any]) -> None:
        self.ctx.store.write(
            "UPDATE deployments SET label = ?, plan = ?, steps = ?, state = ?, ended_at = ? WHERE id = ?",
            (dep.get("label"), json.dumps(dep["plan"]), json.dumps(dep["steps"]), dep["state"],
             dep.get("ended_at"), dep["id"]),
        )
        self.publish()

    def _finish(self, dep: dict[str, Any], state: str, error: str = "") -> None:
        dep["state"] = state
        dep["ended_at"] = time.time()
        dep["plan"]["error"] = redact(error, self.secret_values())
        for step in dep["steps"]:
            if step["status"] == "running":
                # The other half of the install/download pair: no longer followed, still running on the pod.
                step["status"] = "cancelled"
                step["ended_at"] = time.time()
                if state == "failed":
                    step["output"] = (step["output"] + "\nno longer followed; if it is still running on the pod, "
                                      "Deploy again picks it up").strip()
        self._save(dep)

    # ------------------------------------------------------------------ connection

    def set_connection(self, ssh_command: Any) -> dict[str, Any]:
        """Parse and store a pasted connection string. A new string always needs a new confirmation."""
        self.require_enabled()
        if self.busy:
            raise PodError("a deployment is running: cancel it before changing the connection", 409)
        if self.ssh is not None and self.ssh.state != "closed":
            raise PodError("disconnect from the current pod before entering another connection string", 409)
        try:
            conn = parse(ssh_command, default_key=self.ctx.settings.ssh_key_path)
        except ConnStrError as exc:
            raise PodError(str(exc), 400) from None
        self.ctx.store.write(
            "INSERT OR REPLACE INTO pod (id, ssh_command, host, port, user, key_path, fingerprint, state, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?, NULL, 'saved', ?)",
            (ssh_command.strip(), conn.host, conn.port, conn.user, conn.key_path, time.time()),
        )
        self.pod_facts = {}
        self.publish()
        return {**conn.as_dict(), "key_problem": key_problem(conn.key_path)}

    async def connect(self, accept_fingerprint: str | None = None) -> dict[str, Any]:
        """Connect, or return the fingerprint that has to be confirmed first.

        Nothing is sent to the pod until the fingerprint passed in equals the
        one the server presents now. A stored fingerprint (same pod start,
        confirmed earlier) counts as accepted; a different key never does.
        """
        self.require_enabled()
        async with self._connect_lock:
            row = self.pod_row()
            if not row:
                raise PodError("enter the pod's SSH command first", 400)
            if self.ssh is not None and self.ssh.up:
                return {"state": "connected", "fingerprint": self.ssh.fingerprint}
            problem = key_problem(row["key_path"])
            if problem:
                raise PodError(problem, 400)
            target = Connection(user=row["user"], host=row["host"], port=row["port"], key_path=row["key_path"])
            ssh = PodSSH(target, vllm_port=self.cfg.vllm_port, helper_port=self.cfg.helper_port,
                         on_event=self._ssh_event, backoff=self.ssh_backoff)
            try:
                found = await ssh.probe()
            except SSHError as exc:
                raise PodError(str(exc), 502) from None
            accepted = accept_fingerprint if accept_fingerprint is not None else row.get("fingerprint")
            if accepted != found:
                changed = bool(accepted)
                if changed:
                    self._forget_fingerprint()
                return {"state": "confirm", "fingerprint": found, "changed": changed}
            if self.ssh is not None:
                await self.ssh.close()
                self.ssh = None
            try:
                await ssh.connect(found)
            except HostKeyChanged as exc:
                return {"state": "confirm", "fingerprint": exc.found, "changed": True}
            except SSHError as exc:
                raise PodError(str(exc), 502) from None
            self.ssh = ssh
            self._helper_ok = False
            self.ctx.store.write("UPDATE pod SET fingerprint = ?, state = 'connected', updated_at = ? WHERE id = 1",
                                 (found, time.time()))
            log.info("connected to pod %s:%s (host key %s)", target.host, target.port, found)
            try:
                await self._after_connect()
            except SSHError as exc:
                await ssh.close()
                self.ssh = None
                raise PodError(f"connected, but the pod could not be prepared: {exc}", 502) from None
            self.publish()
            return {"state": "connected", "fingerprint": found}

    async def _after_connect(self) -> None:
        """Upload the scripts, look at what is already on the pod and pick the backend state."""
        assert self.ssh is not None
        await self._upload_scripts()
        await self._read_gpus()
        dep = self.deployment
        if dep and dep["state"] == "running" and not self.busy:
            log.info("resuming deployment %s after a restart", dep["id"])
            self._start_task(dep)
            return
        if dep and dep["state"] == "ready" and await self._reactivate():
            return
        self.ctx.backend.set_state("connected")

    async def _upload_scripts(self) -> None:
        assert self.ssh is not None
        files = [(f"{self.cfg.ovp}/{name}", (REMOTE_DIR / name).read_bytes(), 0o700 if name.endswith(".sh") else 0o600)
                 for name in REMOTE_FILES]
        await self.ssh.upload(files)

    async def _read_gpus(self) -> None:
        assert self.ssh is not None
        result = await self.ssh.run(fit.NVIDIA_SMI_QUERY, timeout=30)
        gpus = fit.parse_gpus(result.stdout) if result.ok else []
        self.pod_facts = {
            "gpus": [g.as_dict() for g in gpus],
            "gpu_summary": fit.describe_gpus(gpus),
            "gpu_error": "" if result.ok else (result.text()[:300] or "nvidia-smi failed"),
        }

    async def _ssh_event(self, state: str, detail: str) -> None:
        backend = self.ctx.backend
        if state == "down":
            self._helper_ok = False
            if backend.state == "ready":
                self._ready_before_drop = True
                backend.set_state("tunnel_down", "reconnecting")
        elif state == "up":
            assert self.ssh is not None
            backend.set_endpoints(base_url=self.ssh.url("vllm"), gpu_url="")
            await self._helper_check()
            if self._ready_before_drop or backend.state == "tunnel_down":
                self._ready_before_drop = False
                if not await self._reactivate():
                    backend.set_state("stopped", "vLLM is no longer answering on the pod")
        elif state == "hostkey_changed":
            self._ready_before_drop = False
            self._forget_fingerprint()
            if self.busy and self._task:
                self._cancel_reason = "the pod presented a different host key; nothing more was sent"
                self._task.cancel()
            backend.set_state("no_pod", "the pod's host key changed: confirm the new fingerprint")
        self.publish()

    async def disconnect(self) -> None:
        """Close SSH. Whatever runs on the pod keeps running."""
        self.require_enabled()
        if self.busy and self._task:
            self._cancel_reason = "disconnected; remote processes were left running"
            self._kill_on_cancel = False
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        if self.ssh is not None:
            await self.ssh.close()
            self.ssh = None
        if self.pod_row():
            self._set_pod_state("disconnected")
        self._ready_before_drop = False
        self.pod_facts = {}
        self.ctx.backend.set_endpoints(gpu_url="")
        self.ctx.backend.set_state("no_pod")
        self.publish()

    # ------------------------------------------------------------------ remote primitives

    async def _r(self, argv: list[str], timeout: float = 60.0) -> Result:
        """Run an idempotent remote command; a dropped connection is waited out and the command repeated."""
        while True:
            ssh = self.ssh
            if ssh is None or ssh.state in ("closed", "hostkey_changed"):
                raise StepFailed("the pod is not connected")
            try:
                return await ssh.run(argv, timeout=timeout)
            except SSHUnavailable as exc:
                if ssh.state in ("closed", "hostkey_changed"):
                    raise StepFailed(str(exc)) from None
                log.info("ssh unavailable during a step; waiting for the reconnect")
                if not await ssh.wait_up(self.cfg.reconnect_wait_s):
                    raise StepFailed("the SSH connection did not come back") from None
            except SSHError as exc:
                raise StepFailed(str(exc)) from None

    def _script(self, name: str, *args: Any) -> list[str]:
        return ["bash", f"{self.cfg.ovp}/{name}", *[str(a) for a in args]]

    async def _jobs(self) -> dict[str, dict[str, Any]]:
        result = await self._r(self._script("status.sh", self.cfg.ovp))
        jobs: dict[str, dict[str, Any]] = {n: {"pid": None, "alive": False, "exit": None}
                                           for n in ("vllm", "install", "download", "helper")}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[0] == "job" and parts[1] in jobs:
                jobs[parts[1]] = {"pid": int(parts[2]) if parts[2].isdigit() else None, "alive": parts[3] == "1",
                                  "exit": int(parts[4]) if parts[4].isdigit() else None}
        return jobs

    async def _launch(self, name: str, logfile: str, secrets_file: str, command: list[str]) -> None:
        result = await self._r(self._script("launch.sh", name, self.cfg.ovp, logfile, secrets_file, "--", *command))
        if result.rc not in (0, 3):  # 3: already running (a repeated launch after a reconnect)
            raise StepFailed(f"could not start {name}: {result.text()[:500]}")
        self._started_jobs.add(name)

    async def _stop_job(self, name: str) -> None:
        await self._r(self._script("stop.sh", name, self.cfg.ovp, self.cfg.stop_grace_s),
                      timeout=self.cfg.stop_grace_s + 30)

    async def _tail(self, path: str, lines: int = 40) -> str:
        result = await self._r(["tail", "-n", str(lines), "--", path])
        return redact(result.stdout, self.secret_values()).rstrip() if result.ok else ""

    async def _sizes(self, dirs: list[str]) -> dict[str, tuple[int, str]]:
        result = await self._r(self._script("sizes.sh", self.cfg.remote_root, *dirs))
        out: dict[str, tuple[int, str]] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "size":
                out[parts[1]] = (int(parts[2]) if parts[2].isdigit() else 0, parts[3])
        return out

    async def _http_status(self, url: str, headers: dict[str, str] | None = None, timeout: float = 5.0):
        """GET through the tunnel; returns the response or None when nothing answers."""
        if not url:
            return None
        try:
            return await self.ctx.client.get(url, headers=headers or {}, timeout=timeout)
        except httpx.HTTPError:
            return None

    async def _helper_check(self) -> bool:
        """Point the monitoring at the GPU helper when it answers through the tunnel."""
        ssh = self.ssh
        url = ssh.url("helper") if ssh else ""
        resp = await self._http_status(f"{url}/healthz") if url else None
        self._helper_ok = bool(resp is not None and resp.status_code == 200)
        self.ctx.backend.set_endpoints(gpu_url=url if self._helper_ok else "")
        return self._helper_ok

    def _gpu_url(self) -> str:
        return self.ssh.url("helper") if self.ssh and self._helper_ok else ""

    # ------------------------------------------------------------------ plan (also used by the API)

    def hf_token(self) -> str | None:
        return self.ctx.store.get_secret(SECRET_HF_TOKEN)

    async def repo_infos(self, profile_data: dict[str, Any]) -> dict[str, fit.RepoInfo]:
        token = self.hf_token()
        variants = profile_data["variants"]
        infos = await asyncio.gather(*(self.hf.fetch(v["repo"], v.get("revision") or "", token) for v in variants))
        return {v["label"]: info for v, info in zip(variants, infos)}

    def catalog(self) -> list[dict[str, Any]]:
        raw = self.ctx.store.get_state("gpu_catalog")
        if raw:
            try:
                return fit.validate_catalog(json.loads(raw))
            except (ValueError, fit.CatalogError):
                log.warning("stored GPU catalog is invalid; using the default")
        return fit.default_catalog()

    def set_catalog(self, value: Any) -> list[dict[str, Any]]:
        if value is None:
            self.ctx.store.write("DELETE FROM state WHERE key = 'gpu_catalog'")
            return fit.default_catalog()
        try:
            catalog = fit.validate_catalog(value)
        except fit.CatalogError as exc:
            raise PodError(str(exc), 400) from None
        self.ctx.store.set_state("gpu_catalog", json.dumps(catalog))
        return catalog

    def command_preview(self, profile_data: dict[str, Any], variant: dict[str, Any], context: int, tp: int) -> str:
        argv = prof.serve_args(profile_data, variant, model_dir=f"{self.cfg.remote_root}/{variant['dir']}",
                               context=context, tensor_parallel=tp, port=self.cfg.vllm_port)
        return "vllm " + quote_argv(argv)

    async def plan_for(self, profile_id: int, gpu: str | None = None, count: int | None = None) -> dict[str, Any]:
        """The plan table for the connected pod, or for ``count`` × a catalog GPU."""
        try:
            profile = self.profiles.get_valid(profile_id)
        except prof.ProfileError as exc:
            raise PodError(str(exc), 404 if "no such" in str(exc) else 400) from None
        data = profile["data"]
        if gpu:
            entry = next((e for e in self.catalog() if e["name"].lower() == gpu.lower()), None)
            if entry is None:
                raise PodError(f"'{gpu}' is not in the GPU catalog", 404)
            n = count or 1
            if n not in prof.TP_SIZES or n > int(entry["max_count"]):
                raise PodError(f"count must be one of 1, 2, 4, 8 and at most {entry['max_count']} for {entry['name']}")
            gpus, hypothetical = fit.hypothetical_gpus(entry, n), True
        else:
            if self.ssh is None or not self.ssh.up:
                raise PodError("no pod connected: connect first, or ask for a hypothetical GPU with ?gpu=&count=", 409)
            try:
                await self._read_gpus()
            except SSHError as exc:
                raise PodError(str(exc), 502) from None
            gpus = [fit.Gpu(**g) for g in self.pod_facts.get("gpus", [])]
            hypothetical = False
        infos = await self.repo_infos(data)
        plan = fit.build_plan(data, infos, gpus, hypothetical=hypothetical)
        for row, variant in zip(plan["rows"], data["variants"]):
            row["model_dir"] = f"{self.cfg.remote_root}/{variant['dir']}"
            row["command"] = (self.command_preview(data, variant, row["context"], row["tensor_parallel"])
                              if row["tensor_parallel"] else "")
        plan["profile_id"] = profile["id"]
        if not hypothetical and self.pod_facts.get("gpu_error"):
            plan["gpu_error"] = self.pod_facts["gpu_error"]
        return plan

    async def sizing(self, profile_id: int) -> dict[str, Any]:
        try:
            profile = self.profiles.get_valid(profile_id)
        except prof.ProfileError as exc:
            raise PodError(str(exc), 404 if "no such" in str(exc) else 400) from None
        infos = await self.repo_infos(profile["data"])
        errors = [i.error for i in infos.values() if not i.ok]
        return {"profile_id": profile["id"], "rows": fit.sizing_table(profile["data"], infos, self.catalog()),
                "errors": errors,
                "variants": [{"variant": v["label"], "repo": v["repo"],
                              "download_bytes": infos[v["label"]].total_bytes if infos[v["label"]].ok else None,
                              "disk_request_gb": fit.disk_request_gb(infos[v["label"]].total_bytes)
                              if infos[v["label"]].ok else None} for v in profile["data"]["variants"]]}

    # ------------------------------------------------------------------ actions

    async def deploy(self, body: dict[str, Any]) -> dict[str, Any]:
        self.require_enabled()
        if self.busy:
            raise PodError("a deployment is already running", 409)
        if self.ssh is None or self.ssh.state in ("closed", "hostkey_changed"):
            raise PodError("no pod connected: connect first", 409)
        profile_id = body.get("profile_id")
        if isinstance(profile_id, bool) or not isinstance(profile_id, int):
            raise PodError("profile_id is required")
        try:
            profile = self.profiles.get_valid(profile_id)
        except prof.ProfileError as exc:
            raise PodError(str(exc), 404 if "no such" in str(exc) else 400) from None
        request: dict[str, Any] = {"force": body.get("force") is True}
        variant = body.get("variant")
        if variant is not None:
            if not any(v["label"] == variant for v in profile["data"]["variants"]):
                raise PodError(f"the profile has no variant '{variant}'")
            request["variant"] = variant
        for key, low, high in (("context", 1024, 4_194_304), ("tensor_parallel", 1, 8)):
            value = body.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise PodError(f"{key} must be a whole number between {low} and {high}")
            request[key] = value
        if request.get("tensor_parallel") is not None and request["tensor_parallel"] not in prof.TP_SIZES:
            raise PodError("tensor_parallel must be 1, 2, 4 or 8")
        dep = self._new_deployment(profile, {"kind": "deploy", "request": request})
        self._start_task(dep)
        return self.describe()

    async def restart_vllm(self, context: int | None = None) -> dict[str, Any]:
        """Start vLLM again with the last deployment's profile and plan; nothing is installed or downloaded."""
        self.require_enabled()
        if self.busy:
            raise PodError("a deployment is running: cancel it first", 409)
        if self.ssh is None or self.ssh.state in ("closed", "hostkey_changed"):
            raise PodError("no pod connected: connect first", 409)
        last = self.deployment
        if not last or not (last.get("plan") or {}).get("variant"):
            raise PodError("nothing has been deployed to restart: use Deploy", 409)
        plan = {k: v for k, v in last["plan"].items() if k not in ("error", "offer", "request")}
        if context is not None:
            if isinstance(context, bool) or not isinstance(context, int) or not 1024 <= context <= 4_194_304:
                raise PodError("context must be a whole number of tokens (at least 1024)")
            plan["context"] = context
        plan["kind"] = "restart"
        profile = {"id": last["profile_id"], "name": last["profile_name"], "data": last["snapshot"].get("data", {})}
        try:
            profile["name"], profile["data"] = prof.validate(profile["name"], profile["data"])
        except prof.ProfileError as exc:
            raise PodError(f"the stored profile is not valid any more: {exc}") from None
        self._refresh_plan_command(profile["data"], plan)
        dep = self._new_deployment(profile, plan)
        for step in dep["steps"]:
            if step["id"] in ("preflight", "helper", "install", "download"):
                step["status"] = "skipped"
                step["output"] = f"restart: using the plan of deployment #{last['id']}"
        self._start_task(dep)
        return self.describe()

    async def cancel(self) -> dict[str, Any]:
        self.require_enabled()
        if not self.busy or self._task is None:
            raise PodError("no deployment is running", 409)
        self._cancel_reason = "cancelled"
        self._kill_on_cancel = True
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        return self.describe()

    async def stop_vllm(self) -> dict[str, Any]:
        self.require_enabled()
        if self.busy:
            raise PodError("a deployment is running: cancel it instead", 409)
        if self.ssh is None or not self.ssh.up:
            raise PodError("no pod connected", 409)
        self.ctx.backend.set_state("stopped", "stopped from the Pod page")
        try:
            await self._stop_job("vllm")
        except StepFailed as exc:
            raise PodError(str(exc), 502) from None
        dep = self.deployment
        if dep and dep["state"] == "ready":
            dep["state"] = "stopped"
            dep["ended_at"] = time.time()
            self._save(dep)
        self.publish()
        return self.describe()

    async def delete_models(self, dirs: Any = None) -> dict[str, Any]:
        """Remove downloaded model directories under the remote root. vLLM must not be running."""
        self.require_enabled()
        if self.busy:
            raise PodError("a deployment is running: cancel it first", 409)
        if self.ssh is None or not self.ssh.up:
            raise PodError("no pod connected", 409)
        known: list[str] = []
        for profile in self.profiles.list():
            known += [v.get("dir") for v in profile["data"].get("variants", []) if isinstance(v, dict)]
        if self.deployment:
            known += [v.get("dir") for v in self.deployment["snapshot"].get("data", {}).get("variants", [])]
        known = sorted({d for d in known if isinstance(d, str) and prof._DIR.match(d) and d not in prof.RESERVED_DIRS})
        if dirs is None:
            targets = known
        else:
            if not isinstance(dirs, list) or any(d not in known for d in dirs):
                raise PodError("dirs must be model directories of a profile")
            targets = dirs
        try:
            if (await self._jobs())["vllm"]["alive"]:
                raise PodError("vLLM is running: stop it before deleting model files", 409)
            await self._stop_job("download")
            result = await self._r(self._script("delete_models.sh", self.cfg.remote_root, *targets), timeout=600)
        except StepFailed as exc:
            raise PodError(str(exc), 502) from None
        deleted = [{"dir": p[1], "bytes": int(p[2]) * 1024} for p in (line.split() for line in result.stdout.splitlines())
                   if len(p) == 3 and p[0] == "deleted" and p[2].isdigit()]
        return {"deleted": deleted}

    # ------------------------------------------------------------------ the state machine

    _cancel_reason = ""
    _kill_on_cancel = True

    def _start_task(self, dep: dict[str, Any]) -> None:
        self.deployment = dep
        self._cancel_reason = ""
        self._kill_on_cancel = True
        self._started_jobs = set()
        self._task = asyncio.ensure_future(self._run(dep))

    async def _run(self, dep: dict[str, Any]) -> None:
        backend = self.ctx.backend
        current = {"title": ""}

        async def step(step_id: str, fn: Callable[[dict[str, Any]], Awaitable[str | None]]) -> None:
            entry = next(s for s in dep["steps"] if s["id"] == step_id)
            if entry["status"] == "skipped" and dep["plan"].get("kind") == "restart":
                return
            entry.update(status="running", started_at=time.time(), ended_at=None, output="")
            entry.pop("progress", None)
            current["title"] = entry["title"]
            backend.set_state("deploying", entry["title"])
            self._save(dep)
            try:
                outcome = await fn(entry)
            except StepFailed as exc:
                entry["status"] = "failed"
                entry["ended_at"] = time.time()
                self._out(entry, f"FAILED: {exc}")
                raise StepFailed(f"{entry['title']}: {exc}") from None
            entry["status"] = outcome or "done"
            entry["ended_at"] = time.time()
            self._save(dep)

        try:
            dep["state"] = "running"
            dep["plan"].pop("error", None)
            dep["plan"].pop("offer", None)
            await step("connect", lambda s: self._s_connect(dep, s))
            await step("preflight", lambda s: self._s_preflight(dep, s))
            await step("helper", lambda s: self._s_helper(dep, s))
            # Both are network-bound and independent; the download only waits for the `hf` tool.
            pair = [asyncio.ensure_future(step("install", lambda s: self._s_install(dep, s))),
                    asyncio.ensure_future(step("download", lambda s: self._s_download(dep, s)))]
            try:
                await asyncio.wait(pair, return_when=asyncio.FIRST_EXCEPTION)
            finally:
                # On a failure or a cancel the other one stops being followed; its remote job is not touched here.
                for task in pair:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pair, return_exceptions=True)
            for task in pair:
                if not task.cancelled() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
            await step("secrets", lambda s: self._s_secrets(dep, s))
            await step("start", lambda s: self._s_start(dep, s))
            await step("ready", lambda s: self._s_ready(dep, s))
            await step("activate", lambda s: self._s_activate(dep, s))
            self._finish(dep, "ready")
            log.info("deployment %s is serving: %s", dep["id"], dep.get("label"))
        except StepFailed as exc:
            message = redact(str(exc), self.secret_values())
            log.warning("deployment %s failed: %s", dep["id"], message)
            self._finish(dep, "failed", message)
            backend.set_state("failed", message[:300])
        except asyncio.CancelledError:
            if self._shutting_down:
                raise  # leave the row as "running": the next start resumes it
            reason = self._cancel_reason or "cancelled"
            if self._kill_on_cancel:
                await self._kill_started(dep)
            for entry in dep["steps"]:
                if entry["status"] == "running":
                    entry["status"] = "cancelled"
                    entry["ended_at"] = time.time()
            self._finish(dep, "cancelled", reason)
            if backend.state == "deploying":
                backend.set_state("connected" if self.ssh and self.ssh.up else "no_pod", f"deployment {reason}")
        except Exception as exc:  # a bug must not leave the page stuck on "deploying"
            log.exception("deployment %s crashed", dep["id"])
            self._finish(dep, "failed", f"internal error: {type(exc).__name__}")
            backend.set_state("failed", f"{current['title']}: internal error")
        finally:
            self.publish()

    async def _kill_started(self, dep: dict[str, Any]) -> None:
        """Cancel: stop the remote processes this deployment started (never a vLLM it merely found running)."""
        ssh = self.ssh
        if ssh is None or not ssh.up:
            return
        for name in ("install", "download", "vllm"):
            if name in self._started_jobs:
                with contextlib.suppress(SSHError, asyncio.TimeoutError):
                    await asyncio.shield(ssh.run(self._script("stop.sh", name, self.cfg.ovp, 5), timeout=30))

    def _out(self, step: dict[str, Any], text: str, replace: bool = False) -> None:
        text = redact(text, self.secret_values())
        merged = text if replace else (step["output"] + ("\n" if step["output"] else "") + text)
        step["output"] = merged[-_OUTPUT_LIMIT:]
        self.publish()

    # -- step 1 ---------------------------------------------------------------------

    async def _s_connect(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        ssh = self.ssh
        if ssh is None or ssh.state in ("closed", "hostkey_changed"):
            raise StepFailed("the pod is not connected: use Connect and confirm the host key")
        if not await ssh.wait_up(self.cfg.reconnect_wait_s):
            raise StepFailed("the SSH connection is down")
        try:
            await self._upload_scripts()
        except SSHError as exc:
            raise StepFailed(str(exc)) from None
        t = ssh.target
        self._out(step, f"connected to {t.user}@{t.host}:{t.port}, host key {ssh.fingerprint}")
        return None

    # -- step 2 ---------------------------------------------------------------------

    async def _s_preflight(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        cfg, data = self.cfg, dep["snapshot"]["data"]
        request = dep["plan"].get("request") or {}
        result = await self._r(self._script("preflight.sh", cfg.remote_root, cfg.venv, data["python_version"],
                                            cfg.vllm_port, cfg.helper_port), timeout=90)
        facts = parse_kv(result.stdout)
        one = lambda key, default="": (facts.get(key) or [default])[0]  # noqa: E731

        if one("nvidia_smi") != "ok":
            detail = "; ".join(facts.get("nvidia_smi_error", [])) or "nvidia-smi is not available"
            raise StepFailed(f"cannot read the pod's GPUs: {detail}")
        gpus = fit.parse_gpus("\n".join(facts.get("gpu", [])))
        if not gpus:
            raise StepFailed("the pod reports no NVIDIA GPU")
        self._out(step, f"GPUs: {fit.describe_gpus(gpus)}; compute capability "
                        f"{', '.join(sorted({f'{g.compute_cap:g}' if g.compute_cap else '?' for g in gpus}))}")
        disk_free = int(one("disk_avail_kb", "0") or 0) * 1024 if one("disk_avail_kb").isdigit() else None
        runpod = {k[len("runpod_"):]: v[0] for k, v in facts.items() if k.startswith("runpod_")}
        self.pod_facts = {
            "gpus": [g.as_dict() for g in gpus], "gpu_summary": fit.describe_gpus(gpus), "gpu_error": "",
            "disk_free_bytes": disk_free, "mem_limit": one("mem_limit"), "runpod": runpod,
        }

        jobs = await self._jobs()
        problems: list[str] = []
        if one("uv") in ("", "missing"):
            problems.append("`uv` was not found on the pod (expected in the Runpod base image)")
        elif one("python") != "ok":
            problems.append(f"Python {data['python_version']} is not available through uv on the pod")
        for name, key, port in (("vLLM", "vllm", cfg.vllm_port), ("the GPU helper", "helper", cfg.helper_port)):
            if one(f"port_{key}") == "busy" and not jobs[key]["alive"]:
                problems.append(f"port {port} on the pod is in use by another process; {name} needs it")
        shm = one("shm_avail_kb")
        if shm.isdigit() and int(shm) < 1024 * 1024:
            self._out(step, f"warning: /dev/shm has only {int(shm) / 1024:.0f} MB free; vLLM may need more")
        mem = one("mem_limit")
        if mem.isdigit() and int(mem) < 2**50:  # cgroup v1 reports a huge number for "no limit"
            self._out(step, f"container memory limit: {int(mem) / fit.GIB:.0f} GiB")
        if disk_free is not None:
            self._out(step, f"free disk under {cfg.remote_root}: {disk_free / fit.GB:.1f} GB")

        variants = data["variants"]
        sizes = await self._sizes([v["dir"] for v in variants])
        infos = await self.repo_infos(data)
        env_installed = one("venv_vllm") == data["vllm_version"]
        present = {v["label"]: sizes.get(v["dir"], (0, "-"))[0] for v in variants}
        table = fit.build_plan(data, infos, gpus, disk_free_bytes=disk_free, present=present, env_installed=env_installed)
        for row in table["rows"]:
            self._out(step, f"  {row['variant']}: {row['status']} - {row['reason']}")

        force = bool(request.get("force"))
        label = request.get("variant") or table["auto"]
        if label is None:
            if not force:
                raise StepFailed("no variant of this profile fits this pod (see the plan); choose a variant and "
                                 "'deploy anyway' to try regardless")
            label = variants[0]["label"]
        variant = next(v for v in variants if v["label"] == label)
        info = infos.get(label)
        row = fit.evaluate_variant(data, variant, info, gpus, tensor_parallel=request.get("tensor_parallel"),
                                   context=request.get("context"), disk_free_bytes=disk_free,
                                   present_bytes=present.get(label, 0), env_installed=env_installed)
        if info is not None and info.error_kind in ("auth", "notfound"):
            hint = " Set a Hugging Face token with access under 'Hugging Face token' on this page." \
                if info.error_kind == "auth" else ""
            raise StepFailed(info.error + "." + hint)
        if not row["fits"]:
            if not force:
                raise StepFailed(f"{label}: {row['reason']}. Use 'deploy anyway' to try regardless.")
            if not row["forceable"]:
                raise StepFailed(f"{label}: {row['reason']}. This cannot be overridden.")
            self._out(step, f"deploying anyway although the estimate says: {row['reason']}")
        if not row["tensor_parallel"]:
            raise StepFailed(f"{label}: {row['reason'] or 'no usable GPU'}")

        marker = f"{variant['repo']}@{variant.get('revision') or ''}"
        need_download = sizes.get(variant["dir"], (0, "-"))[1] != marker
        if one("net_pypi") == "fail" and not env_installed:
            problems.append("pypi.org is not reachable from the pod; vLLM cannot be installed")
        if one("net_huggingface") == "fail" and need_download:
            problems.append("huggingface.co is not reachable from the pod; the model cannot be downloaded")
        if problems:
            raise StepFailed("; ".join(problems))

        used = row["gpu_indices"]
        plan = dep["plan"]
        plan.update(
            variant=label, repo=variant["repo"], revision=variant.get("revision") or "", dir=variant["dir"],
            context=row["context"], context_requested=row["context"], tensor_parallel=row["tensor_parallel"],
            gpu_name=row["gpu_name"], gpu_count=len(used), gpu_indices=used, gpu_total=len(gpus),
            forced=force and not row["fits"], download_bytes=row["download_bytes"], table=table["rows"],
            vllm_version=data["vllm_version"],
        )
        self._refresh_plan_command(data, plan)
        dep["label"] = plan["label"]
        self._out(step, f"plan: {label} on {len(used)}× {row['gpu_name']}, tensor-parallel {row['tensor_parallel']}, "
                        f"context {row['context']:,}")
        self.ctx.store.insert("inventory", {
            "ts": time.time(), "deployment_id": plan["label"], "kind": "pod",
            "data": json.dumps({"runpod": runpod, "gpus": self.pod_facts["gpus"], "mem_limit": mem,
                                "disk_free_bytes": disk_free, "plan": {k: plan[k] for k in (
                                    "variant", "repo", "context", "tensor_parallel", "gpu_name", "gpu_count")}}),
        })

        # A different configuration is about to replace the one that is serving: stop it now.
        if jobs["vllm"]["alive"] and await self._running_config() != self._wanted_config(dep):
            self._out(step, "another configuration is running: stopping vLLM (in-flight requests are cut)")
            await self._stop_job("vllm")
        return None

    def _refresh_plan_command(self, data: dict[str, Any], plan: dict[str, Any]) -> None:
        """Recompute label and command after the variant, context or tensor-parallel size changed."""
        variant = next(v for v in data["variants"] if v["label"] == plan["variant"])
        argv = prof.serve_args(data, variant, model_dir=f"{self.cfg.remote_root}/{variant['dir']}",
                               context=plan["context"], tensor_parallel=plan["tensor_parallel"],
                               port=self.cfg.vllm_port)
        plan["argv"] = argv
        plan["command"] = "vllm " + quote_argv(argv)
        plan["label"] = prof.deployment_label(plan.get("gpu_name") or "gpu", plan.get("gpu_count") or 1,
                                              variant, plan["context"])

    def _vllm_vars(self, dep: dict[str, Any]) -> dict[str, str]:
        """Non-secret variables of the vLLM process: the profile's own, and the GPU selection if partial."""
        plan = dep["plan"]
        env = dict(dep["snapshot"]["data"].get("env") or {})
        if plan.get("gpu_total") and plan.get("gpu_count") and plan["gpu_count"] < plan["gpu_total"]:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in plan["gpu_indices"])
        return env

    def _wanted_config(self, dep: dict[str, Any]) -> str:
        plan = dep["plan"]
        return json.dumps({"argv": plan["argv"], "env": self._vllm_vars(dep), "vllm": plan.get("vllm_version")},
                          sort_keys=True)

    async def _running_config(self) -> str:
        result = await self._r(["cat", "--", f"{self.cfg.ovp}/vllm.json"])
        return result.stdout.strip() if result.ok else ""

    # -- step 3 ---------------------------------------------------------------------

    async def _s_helper(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        cfg = self.cfg
        jobs = await self._jobs()
        if jobs["helper"]["alive"] and await self._helper_check():
            self._out(step, "the GPU helper is already running")
            return "skipped"
        if jobs["helper"]["alive"]:
            await self._stop_job("helper")
        await self._launch("helper", f"{cfg.ovp}/helper.log", "",
                           self._script("start_helper.sh", cfg.ovp, cfg.helper_port, cfg.vllm_log))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            if await self._helper_check():
                self._out(step, f"GPU helper answering on the pod's 127.0.0.1:{cfg.helper_port} through the tunnel")
                return None
            if not (await self._jobs())["helper"]["alive"]:
                break
        raise StepFailed("the GPU helper did not start:\n" + await self._tail(f"{cfg.ovp}/helper.log", 20))

    # -- steps 4 and 5 --------------------------------------------------------------

    async def _wait_job(self, step: dict[str, Any], name: str, logfile: str,
                        on_poll: Callable[[], Awaitable[None]] | None = None) -> None:
        """Follow a detached job until it exits; its exit status decides."""
        while True:
            job = (await self._jobs())[name]
            if on_poll:
                await on_poll()
            if job["alive"]:
                tail = await self._tail(logfile, 12)
                if tail:
                    self._out(step, tail, replace=True)
                await asyncio.sleep(self.cfg.poll_interval_s)
                continue
            tail = await self._tail(logfile, 40)
            if job["exit"] == 0:
                self._out(step, tail, replace=True)
                return
            code = f"exit status {job['exit']}" if job["exit"] is not None else "the process disappeared"
            self._out(step, tail, replace=True)
            raise StepFailed(f"{name} failed ({code}); see the {name} log")

    async def _installed_version(self) -> tuple[str, bool]:
        cfg = self.cfg
        result = await self._r(["bash", "-c", 'for d in "$1"/lib/python*/site-packages/vllm-*.dist-info; do '
                                              '[ -d "$d" ] && basename "$d"; done; [ -x "$1/bin/hf" ] && echo hf-ok',
                                "ovp", cfg.venv])
        version = ""
        for line in result.stdout.splitlines():
            m = re.match(r"^vllm-(.+)\.dist-info$", line.strip())
            if m:
                version = m.group(1)
        return version, "hf-ok" in result.stdout.split()

    async def _s_install(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        cfg, data = self.cfg, dep["snapshot"]["data"]
        logfile = f"{cfg.ovp}/install.log"
        job = (await self._jobs())["install"]
        if not job["alive"]:
            version, has_hf = await self._installed_version()
            if version == data["vllm_version"] and has_hf and job["exit"] in (0, None):
                self._out(step, f"vLLM {version} is already installed in {cfg.venv}")
                return "skipped"
            await self._launch("install", logfile, "", self._script("install_vllm.sh", cfg.venv,
                                                                    data["python_version"], data["vllm_version"]))
        else:
            self._out(step, "an installation is already running: following it")
        await self._wait_job(step, "install", logfile)
        version, _ = await self._installed_version()
        if version != data["vllm_version"]:
            raise StepFailed(f"the environment has vLLM '{version or 'none'}', expected {data['vllm_version']}")
        log_text = await self._tail(logfile, 60)
        found = re.findall(r"^OVP_VERSIONS (.+)$", log_text, re.M)
        if found:
            versions = dict(p.split("=", 1) for p in found[-1].split() if "=" in p)
            self.ctx.store.insert("inventory", {"ts": time.time(), "deployment_id": dep.get("label"),
                                                "kind": "vllm_env", "data": json.dumps(versions)})
        return None

    async def _s_download(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        cfg, plan = self.cfg, dep["plan"]
        logfile = f"{cfg.ovp}/download.log"
        target_dir = f"{cfg.remote_root}/{plan['dir']}"
        marker = f"{plan['repo']}@{plan['revision']}"
        total = plan.get("download_bytes") or 0

        async def progress() -> None:
            done = (await self._sizes([plan["dir"]])).get(plan["dir"], (0, "-"))[0]
            step["progress"] = {"done": done, "total": total}
            self.publish()

        size, found = (await self._sizes([plan["dir"]])).get(plan["dir"], (0, "-"))
        job = (await self._jobs())["download"]
        running_target = ""
        if job["alive"]:
            result = await self._r(["cat", "--", f"{cfg.ovp}/download.target"])
            running_target = result.stdout.strip()
        if found == marker and not (job["alive"] and running_target == marker):
            step["progress"] = {"done": size, "total": total or size}
            self._out(step, f"{plan['repo']} is already present in {target_dir}")
            return "skipped"
        if job["alive"] and running_target != marker:
            self._out(step, "stopping the download of another model")
            await self._stop_job("download")
            job = {"alive": False}
        if not job["alive"]:
            # The `hf` tool is installed first by the install step; wait for it rather than for all of vLLM.
            while True:
                _, has_hf = await self._installed_version()
                if has_hf:
                    break
                install = (await self._jobs())["install"]
                if not install["alive"] and install["exit"] not in (0, None):
                    raise StepFailed("the download tool is not installed because the install step failed")
                if not install["alive"] and install["exit"] == 0:
                    raise StepFailed("the environment has no `hf` command")
                self._out(step, "waiting for the download tool to be installed…", replace=True)
                await asyncio.sleep(min(self.cfg.poll_interval_s, 1.0))
            token = self.hf_token()
            secrets_file = ""
            if token:
                await self._upload_secret("hf.env", {"HF_TOKEN": token})
                secrets_file = "hf.env"
            else:
                await self._r(["rm", "-f", "--", f"{cfg.ovp}/hf.env"])
            command = self._script("start_download.sh", cfg.ovp, cfg.venv, plan["repo"], target_dir)
            if plan["revision"]:
                command.append(plan["revision"])
            await self._launch("download", logfile, secrets_file, command)
        else:
            self._out(step, "this download is already running: following it")
        await self._wait_job(step, "download", logfile, on_poll=progress)
        await progress()
        return None

    # -- steps 6 to 9 ---------------------------------------------------------------

    async def _upload_secret(self, name: str, values: dict[str, str]) -> None:
        """NAME=VALUE lines in a 0600 file under ovp/, by SFTP: the values never touch a command line."""
        await self._upload_private(name, "".join(f"{k}={v}\n" for k, v in values.items()).encode())

    async def _upload_private(self, name: str, body: bytes) -> None:
        while True:
            ssh = self.ssh
            if ssh is None or ssh.state in ("closed", "hostkey_changed"):
                raise StepFailed("the pod is not connected")
            try:
                await ssh.upload([(f"{self.cfg.ovp}/{name}", body, 0o600)])
                return
            except SSHUnavailable:
                if not await ssh.wait_up(self.cfg.reconnect_wait_s):
                    raise StepFailed("the SSH connection did not come back") from None

    async def _s_secrets(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        plan = dep["plan"]
        if "argv" not in plan:
            raise StepFailed("no plan")
        jobs = await self._jobs()
        stored = self.ctx.store.get_secret(SECRET_VLLM_KEY)
        reuse = (plan.get("kind") != "restart" and jobs["vllm"]["alive"] and bool(stored)
                 and await self._running_config() == self._wanted_config(dep))
        plan["reuse"] = reuse
        if reuse:
            self._out(step, "vLLM is already running with this configuration: keeping it and its key")
            return "skipped"
        if jobs["vllm"]["alive"]:
            self._out(step, "stopping the running vLLM")
            await self._stop_job("vllm")
        key = "vk-" + pysecrets.token_urlsafe(36)
        self.ctx.store.set_secret(SECRET_VLLM_KEY, key)
        values = {"VLLM_API_KEY": key}
        token = self.hf_token()
        if token:
            values["HF_TOKEN"] = token
        await self._upload_secret("vllm.env", values)
        await self._upload_secret("vllm.vars", self._vllm_vars(dep))
        self._out(step, f"wrote {self.cfg.ovp}/vllm.env (mode 600): a new vLLM API key"
                        + (" and the Hugging Face token" if token else ""))
        return None

    async def _s_start(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        plan = dep["plan"]
        if plan.get("reuse"):
            self._out(step, "already running")
            return "skipped"
        await self._start_vllm(dep)
        self._out(step, plan["command"])
        return None

    async def _start_vllm(self, dep: dict[str, Any]) -> None:
        cfg, plan = self.cfg, dep["plan"]
        await self._upload_private("vllm.json", self._wanted_config(dep).encode() + b"\n")
        await self._launch("vllm", cfg.vllm_log, "vllm.env", self._script("start_vllm.sh", cfg.venv, *plan["argv"]))

    async def _s_ready(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        cfg, plan, data = self.cfg, dep["plan"], dep["snapshot"]["data"]
        deadline = time.monotonic() + cfg.ready_timeout_s
        retries = 0
        while True:
            ssh = self.ssh
            resp = await self._http_status(f"{ssh.url('vllm')}/health") if ssh and ssh.up else None
            if resp is not None and resp.status_code == 200:
                self._out(step, "vLLM answers /health")
                return None
            job = (await self._jobs())["vllm"]
            if not job["alive"]:
                text = await self._tail(cfg.vllm_log, 200)
                limit = context_limit_from_log(text)
                last = "\n".join(text.splitlines()[-25:])
                if limit and limit < plan["context"]:
                    if limit >= data["context_min"] and retries < 3 and limit >= 1024:
                        retries += 1
                        self._out(step, f"vLLM reports that a context of {plan['context']:,} does not fit its KV "
                                        f"cache; restarting with {limit:,}")
                        plan["context"] = limit
                        self._refresh_plan_command(data, plan)
                        dep["label"] = plan["label"]
                        await self._start_vllm(dep)
                        self._save(dep)
                        continue
                    plan["offer"] = {"context": limit}
                    self._out(step, last)
                    raise StepFailed(f"vLLM reports that the context does not fit: at most {limit:,} tokens, which "
                                     f"is below the profile's minimum of {data['context_min']:,}. "
                                     "You can restart with the smaller context.")
                self._out(step, last, replace=True)
                code = f"exit status {job['exit']}" if job["exit"] is not None else "the process disappeared"
                raise StepFailed(f"vLLM exited before it was ready ({code}); the last log lines are shown above")
            if time.monotonic() > deadline:
                raise StepFailed(f"vLLM was not ready after {cfg.ready_timeout_s / 60:.0f} minutes; it is still "
                                 "running - see the vLLM log, then Restart or Stop it")
            tail = await self._tail(cfg.vllm_log, 10)
            if tail:
                self._out(step, tail, replace=True)
            await asyncio.sleep(min(cfg.poll_interval_s, 2.0))

    async def _verify_protected(self, key: str) -> tuple[bool, str, dict[str, Any] | None]:
        """401 without the key and 200 with it. Returns (ok, problem, model list body)."""
        ssh = self.ssh
        base = ssh.url("vllm") if ssh else ""
        if not base:
            return False, "the tunnel is down", None
        anonymous = await self._http_status(f"{base}/v1/models", timeout=10)
        if anonymous is None:
            return False, "vLLM does not answer through the tunnel", None
        if anonymous.status_code not in (401, 403):
            return False, (f"vLLM answered {anonymous.status_code} to a request without a key: refusing to "
                           "activate an unprotected server"), None
        with_key = await self._http_status(f"{base}/v1/models", {"authorization": f"Bearer {key}"}, timeout=10)
        if with_key is None or with_key.status_code != 200:
            code = with_key.status_code if with_key is not None else "no answer"
            return False, f"vLLM did not accept the generated key ({code})", None
        try:
            body = with_key.json()
        except ValueError:
            body = None
        return True, "", body if isinstance(body, dict) else None

    async def _s_activate(self, dep: dict[str, Any], step: dict[str, Any]) -> str | None:
        plan, data = dep["plan"], dep["snapshot"]["data"]
        key = self.ctx.store.get_secret(SECRET_VLLM_KEY) or ""
        ok, problem, body = await self._verify_protected(key)
        if not ok:
            if "unprotected" in problem:
                await self._stop_job("vllm")
            raise StepFailed(problem)
        self._out(step, "/v1/models: 401 without the key, 200 with it")
        models = [m for m in (body or {}).get("data") or [] if isinstance(m, dict)]
        served = data["served_model_name"]
        entry = next((m for m in models if m.get("id") == served), None)
        if entry is None:
            raise StepFailed(f"vLLM does not list the served model '{served}' "
                             f"(it lists: {', '.join(str(m.get('id')) for m in models) or 'nothing'})")
        actual = entry.get("max_model_len")
        if isinstance(actual, int) and actual != plan["context"]:
            self._out(step, f"note: vLLM reports a context of {actual:,}, the plan asked for {plan['context']:,}; "
                            "using the value vLLM reports")
            plan["context"] = actual
            self._refresh_plan_command(data, plan)
            dep["label"] = plan["label"]
        await self._helper_check()
        self._activate(dep, key)
        self._out(step, f"serving '{served}' with a context of {plan['context']:,} as {plan['label']}")
        return None

    def _activate(self, dep: dict[str, Any], key: str) -> None:
        assert self.ssh is not None
        plan = dep["plan"]
        self.ctx.backend.set_ready(
            base_url=self.ssh.url("vllm"), api_key=key,
            served_model=dep["snapshot"]["data"]["served_model_name"],
            context_length=plan["context"], label=plan["label"], gpu_url=self._gpu_url(),
        )

    async def _reactivate(self) -> bool:
        """After a reconnect or a proxy restart: serve again if the deployed vLLM is still there and protected."""
        dep = self.deployment
        key = self.ctx.store.get_secret(SECRET_VLLM_KEY)
        ssh = self.ssh
        if not dep or dep["state"] != "ready" or not key or ssh is None or not ssh.up:
            return False
        health = await self._http_status(f"{ssh.url('vllm')}/health")
        if health is None or health.status_code != 200:
            return False
        ok, problem, _ = await self._verify_protected(key)
        if not ok:
            log.warning("not re-activating the backend: %s", problem)
            return False
        await self._helper_check()
        self._activate(dep, key)
        return True
