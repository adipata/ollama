"""An in-process stand-in for a Runpod pod: an asyncssh server that runs commands locally.

Commands received over SSH are executed with a controlled environment whose
PATH starts with a directory of stub executables (``uv``, ``vllm``, ``hf``,
``nvidia-smi``, ``curl``, ``df``). SFTP serves the real file system (the
"remote root" is a temp directory) and direct-tcpip forwarding is allowed, so
the proxy's port forwards reach the stub vLLM and the real GPU helper.

Standalone, for looking at the Pod page by hand:

    python tests/fake_pod/server.py --dir /some/scratch --port 19022 --hf-port 19080
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import sys
from pathlib import Path
from typing import Any

import asyncssh

from ollama_vllm_proxy.pod import fit

STUBS = Path(__file__).resolve().parent / "stubs"
TESTS = Path(__file__).resolve().parent.parent

GPUS = {
    "1x96": ["0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 12.0"],
    "2x48": ["0, NVIDIA L40S, 46068, 8.9", "1, NVIDIA L40S, 46068, 8.9"],
    "1x48": ["0, NVIDIA L40S, 46068, 8.9"],
    "2x24": ["0, NVIDIA RTX 4090, 24564, 8.9", "1, NVIDIA RTX 4090, 24564, 8.9"],
    "1x24": ["0, NVIDIA RTX PRO 4000 Blackwell, 24467, 12.0"],
    "1x80": ["0, NVIDIA H100 80GB HBM3, 81559, 9.0"],
    "2xA6000": ["0, NVIDIA RTX A6000, 49140, 8.6", "1, NVIDIA RTX A6000, 49140, 8.6"],
    "mixed": ["0, NVIDIA L40S, 46068, 8.9", "1, NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0",
              "2, NVIDIA L40S, 46068, 8.9"],
}

# A plausible config for the (fictional) default model: 2*64*8*128*2 bytes = 256 KiB of KV cache per token.
MODEL_CONFIG = {"num_hidden_layers": 64, "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
                "hidden_size": 5120, "torch_dtype": "bfloat16"}
REPOS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3.8-27B": {"weights": 54_000_000_000, "other": 20_000_000, "config": MODEL_CONFIG},
    "Qwen/Qwen3.8-27B-FP8": {"weights": 29_000_000_000, "other": 20_000_000, "config": MODEL_CONFIG},
    "Tiny/Tiny-1B": {"weights": 2_000_000_000, "other": 5_000_000,
                     "config": {"num_hidden_layers": 16, "num_attention_heads": 16, "hidden_size": 1024}},
}
GOOD_TOKEN = "hf_GoodTokenGoodTokenGoodToken1234"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeHF(fit.HuggingFace):
    """Repository facts without the network. ``gated`` repos need GOOD_TOKEN."""

    def __init__(self, repos: dict[str, dict[str, Any]] | None = None):
        super().__init__(endpoint="http://hf.invalid")
        self.repos = dict(REPOS if repos is None else repos)
        self.gated: set[str] = set()
        self.down = False
        self.calls: list[tuple[str, str, bool]] = []

    async def fetch(self, repo: str, revision: str = "", token: str | None = None) -> fit.RepoInfo:
        self.calls.append((repo, revision, bool(token)))
        info = fit.RepoInfo(repo=repo, revision=revision)
        if self.down:
            info.error_kind, info.error = "network", f"{repo}: cannot read the repository from Hugging Face (stub)"
        elif repo not in self.repos:
            info.error_kind, info.error = "notfound", f"{repo}: no such repository or revision on Hugging Face"
        elif repo in self.gated and token != GOOD_TOKEN:
            info.error_kind = "auth"
            info.error = f"{repo}: the files are gated and not readable" + ("" if token else " (no Hugging Face token is set)")
        else:
            spec = self.repos[repo]
            info.weight_bytes, info.total_bytes = spec["weights"], spec["weights"] + spec["other"]
            info.config = spec["config"]
        return info


def hf_http_app(repos: dict[str, dict[str, Any]] | None = None):
    """The same facts behind Hugging Face's HTTP API shape (for POD_HF_ENDPOINT and the client's own tests)."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    table = REPOS if repos is None else repos

    def allowed(request, spec) -> bool:
        return not spec.get("gated") or request.headers.get("authorization") == f"Bearer {GOOD_TOKEN}"

    async def model(request):
        repo = f"{request.path_params['owner']}/{request.path_params['name']}"
        spec = table.get(repo)
        if spec is None:
            return JSONResponse({"error": "Repository not found"}, status_code=404)
        return JSONResponse({"id": repo, "gated": bool(spec.get("gated")), "siblings": [
            {"rfilename": "config.json", "size": 1200},
            {"rfilename": "tokenizer.json", "size": spec["other"] - 1200},
            {"rfilename": "model-00001-of-00002.safetensors", "size": spec["weights"] // 2,
             "lfs": {"size": spec["weights"] // 2}},
            {"rfilename": "model-00002-of-00002.safetensors", "lfs": {"size": spec["weights"] - spec["weights"] // 2}},
        ]})

    async def config(request):
        repo = f"{request.path_params['owner']}/{request.path_params['name']}"
        spec = table.get(repo)
        if spec is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if not allowed(request, spec):
            return JSONResponse({"error": "Access to this model is restricted"}, status_code=401)
        return JSONResponse(spec["config"])

    return Starlette(routes=[
        Route("/api/models/{owner}/{name}", model),
        Route("/api/models/{owner}/{name}/revision/{rev:path}", model),
        Route("/{owner}/{name}/resolve/{rev}/config.json", config),
    ])


class _Server(asyncssh.SSHServer):
    def __init__(self, pod: FakePod):
        self.pod = pod
        self.conn: asyncssh.SSHServerConnection | None = None

    def connection_made(self, conn):
        self.conn = conn
        self.pod.connections.add(conn)

    def connection_lost(self, exc):
        self.pod.connections.discard(self.conn)

    def begin_auth(self, username):
        return True

    def public_key_auth_supported(self):
        return True

    def validate_public_key(self, username, key):
        self.pod.auth_attempts += 1
        return key.public_data == self.pod.client_key.public_data

    def password_auth_supported(self):
        return False

    def connection_requested(self, dest_host, dest_port, orig_host, orig_port):
        self.pod.forwards.append((dest_host, dest_port))
        return True  # direct-tcpip: asyncssh opens the connection itself


class FakePod:
    def __init__(self, directory: Path, gpus: str = "1x96", port: int = 0):
        self.dir = Path(directory)
        self.root = self.dir / "workspace"
        self.bin = self.dir / "bin"
        self.state = self.dir / "state"
        self.home = self.dir / "home"
        for d in (self.root, self.bin, self.state, self.home, self.dir / "keys"):
            d.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.vllm_port = free_port()
        self.helper_port = free_port()
        self.client_key = asyncssh.generate_private_key("ssh-ed25519")
        self.key_path = self.dir / "keys" / "id_ed25519"
        self.key_path.write_bytes(self.client_key.export_private_key())
        self.key_path.chmod(0o600)
        self.host_key = asyncssh.generate_private_key("ssh-ed25519")
        self.commands: list[str] = []
        self.connections: set[Any] = set()
        self.forwards: list[tuple[str, int]] = []
        self.auth_attempts = 0
        self._listener: asyncssh.SSHAcceptor | None = None
        self._procs: set[asyncio.subprocess.Process] = set()
        self.env = {
            "PATH": f"{self.bin}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            # What a real pod has in its environment and must never reach vLLM or a log:
            "RUNPOD_API_KEY": "rpa_PODSECRETPODSECRETPODSECRET99",
            "JUPYTER_PASSWORD": "jupyter-pw-do-not-leak",
            "RUNPOD_POD_ID": "fakepod0001",
            "RUNPOD_DC_ID": "EU-TEST-1",
            "RUNPOD_GPU_COUNT": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "NVIDIA_VISIBLE_DEVICES": "all",
        }
        self._install_stubs()
        self.set_gpus(gpus)
        self.set("df_avail_kb", str(400 * 1000 * 1000))  # 400 GB free unless a test says otherwise
        self.set("hf_sizes.json", json.dumps({r: s["weights"] + s["other"] for r, s in REPOS.items()}))

    # -- configuration ----------------------------------------------------------------

    def _install_stubs(self) -> None:
        for src in STUBS.iterdir():
            if src.name.startswith("__"):
                continue
            text = src.read_text().replace("#!PYTHON", f"#!{sys.executable}").replace('"STUBS"', repr(str(self.bin)))
            text = text.replace('"TESTS"', repr(str(TESTS)))
            dst = self.bin / src.name
            dst.write_text(text)
            dst.chmod(0o755)

    def set(self, name: str, value: str | None) -> None:
        path = self.state / name
        if value is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(value)

    def set_gpus(self, spec: str | list[str] | None) -> None:
        if spec is None:
            self.set("gpus.csv", None)
            return
        lines = GPUS[spec] if isinstance(spec, str) else spec
        self.set("gpus.csv", "\n".join(lines) + "\n")

    def records(self, name: str) -> list[dict[str, Any]]:
        path = self.state / f"{name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @property
    def ssh_command(self) -> str:
        return f"ssh root@127.0.0.1 -p {self.port} -i {self.key_path}"

    @property
    def fingerprint(self) -> str:
        return self.host_key.get_fingerprint("sha256")

    # -- server -------------------------------------------------------------------------

    async def start(self) -> None:
        self._listener = await asyncssh.create_server(
            lambda: _Server(self), "127.0.0.1", self.port, server_host_keys=[self.host_key],
            process_factory=self._process, sftp_factory=True, allow_scp=False, encoding=None,
            keepalive_interval=0,
        )
        self.port = self._listener.get_port()

    def drop_connections(self) -> None:
        """Cut every open SSH connection, as a network drop would. Remote processes are not touched."""
        for conn in list(self.connections):
            conn.abort()

    async def stop(self) -> None:
        """Stop listening and cut the connections (the pod is unreachable)."""
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.close()
        self.drop_connections()
        if listener is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(listener.wait_closed(), timeout=5)
        await asyncio.sleep(0)

    async def new_host_key(self) -> None:
        """Come back on the same port with another host key, like a restarted pod."""
        await self.stop()
        self.host_key = asyncssh.generate_private_key("ssh-ed25519")
        await self.start()

    async def _process(self, process: asyncssh.SSHServerProcess) -> None:
        command = process.command
        if command is None:
            process.exit(1)
            return
        self.commands.append(command)
        proc = await asyncio.create_subprocess_shell(
            command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=self.env, cwd=str(self.home), start_new_session=True,
        )
        self._procs.add(proc)

        async def pump_out(reader, writer) -> None:
            with contextlib.suppress(Exception):
                while data := await reader.read(65536):
                    writer.write(data)

        async def pump_in() -> None:
            try:
                while data := await process.stdin.read(65536):
                    proc.stdin.write(data)
                    await proc.stdin.drain()
            except (asyncssh.SignalReceived, asyncssh.BreakReceived):
                self._kill(proc)  # the client terminated the command (log tail)
            except Exception:
                pass
            with contextlib.suppress(Exception):
                proc.stdin.close()

        pumps = [asyncio.ensure_future(pump_out(proc.stdout, process.stdout)),
                 asyncio.ensure_future(pump_out(proc.stderr, process.stderr)),
                 asyncio.ensure_future(pump_in())]
        closed = asyncio.ensure_future(process.channel.wait_closed())
        waiter = asyncio.ensure_future(proc.wait())
        try:
            await asyncio.wait({closed, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if not waiter.done():
                self._kill(proc)  # the channel went away: a real sshd would hang up the command too
                await waiter
            await asyncio.wait(pumps[:2], timeout=5)
            with contextlib.suppress(Exception):
                process.exit(proc.returncode or 0)
        finally:
            self._procs.discard(proc)
            for task in (*pumps, closed):
                task.cancel()

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)

    # -- cleanup --------------------------------------------------------------------------

    def job_pid(self, name: str) -> int | None:
        try:
            pid = int((self.root / "ovp" / f"{name}.pid").read_text().strip())
            os.kill(pid, 0)
            if "\nState:\tZ" in Path(f"/proc/{pid}/status").read_text():
                return None  # killed, not yet reaped by this machine's init
            return pid
        except (OSError, ValueError):
            return None

    def kill_jobs(self) -> None:
        """Kill the detached processes the deployment started (they outlive SSH by design)."""
        for pidfile in (self.root / "ovp").glob("*.pid"):
            try:
                pid = int(pidfile.read_text().strip())
                os.killpg(pid, signal.SIGKILL)
            except (OSError, ValueError):
                pass

    async def close(self) -> None:
        await self.stop()
        for proc in list(self._procs):
            self._kill(proc)
        self.kill_jobs()


async def _main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="scratch directory for the pod's files, stubs and keys")
    ap.add_argument("--port", type=int, default=19022)
    ap.add_argument("--hf-port", type=int, default=19080, help="fake Hugging Face API (POD_HF_ENDPOINT)")
    ap.add_argument("--gpus", default="1x96", choices=sorted(GPUS))
    ap.add_argument("--slow", action="store_true", help="slow install, download and model load (to watch progress)")
    a = ap.parse_args()
    directory = Path(a.dir)
    if directory.exists():
        shutil.rmtree(directory)
    pod = FakePod(directory, gpus=a.gpus, port=a.port)
    if a.slow:
        pod.set("uv_delay", "6")
        pod.set("hf_steps", "40")
        pod.set("hf_delay", "0.4")
        pod.set("vllm_start_delay", "6")
    await pod.start()
    hf = uvicorn.Server(uvicorn.Config(hf_http_app(), host="127.0.0.1", port=a.hf_port, log_level="warning"))
    info = {"ssh_command": pod.ssh_command, "fingerprint": pod.fingerprint, "remote_root": str(pod.root),
            "vllm_port": pod.vllm_port, "helper_port": pod.helper_port,
            "hf_endpoint": f"http://127.0.0.1:{a.hf_port}"}
    (directory / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)
    try:
        await hf.serve()
    finally:
        await pod.close()


if __name__ == "__main__":
    sys.path.insert(0, str(TESTS))
    asyncio.run(_main())
