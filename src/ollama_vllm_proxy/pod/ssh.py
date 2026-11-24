"""The SSH connection to the pod: host-key pinning, reconnects, port forwards, commands, uploads.

Every pod start has a new host key, so there is no known_hosts file. Instead
the caller first asks for the key's fingerprint (:meth:`PodSSH.probe`), shows
it to the user, and only then calls :meth:`PodSSH.connect` with the accepted
fingerprint. From then on the connection only ever talks to a server holding
that exact key: a reconnect that meets another key stops and reports it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import asyncssh

from .connstr import Connection

log = logging.getLogger("ovp.pod.ssh")

KEEPALIVE_INTERVAL = 15
KEEPALIVE_COUNT = 3
CONNECT_TIMEOUT = 20.0


class SSHError(Exception):
    """A remote operation could not be carried out (connection down, timeout, ...)."""


class SSHUnavailable(SSHError):
    """There is no usable connection right now; the operation may be retried after a reconnect."""


class HostKeyChanged(SSHError):
    def __init__(self, expected: str, found: str):
        super().__init__(f"the pod presented a different host key ({found}) than the one confirmed ({expected})")
        self.expected, self.found = expected, found


@dataclass
class Result:
    rc: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def text(self) -> str:
        return (self.stdout + ("\n" + self.stderr if self.stderr.strip() else "")).strip()


def quote_argv(argv: Sequence[str]) -> str:
    """The only way remote command text is produced: every argument quoted, nothing interpolated."""
    return " ".join(shlex.quote(str(a)) for a in argv)


class PodSSH:
    def __init__(
        self,
        target: Connection,
        *,
        vllm_port: int = 8000,
        helper_port: int = 9400,
        on_event: Callable[[str, str], Awaitable[None] | None] | None = None,
        backoff: Sequence[float] = (1, 2, 5, 10, 20, 30),
    ):
        self.target = target
        self.remote_ports = {"vllm": vllm_port, "helper": helper_port}
        self.local_ports: dict[str, int] = {}
        self.fingerprint: str | None = None
        self.state = "closed"  # closed | up | down | hostkey_changed
        self.detail = ""
        self.backoff = tuple(backoff)
        self._on_event = on_event
        self._conn: asyncssh.SSHClientConnection | None = None
        self._listeners: list[Any] = []
        self._watcher: asyncio.Task | None = None
        self._up = asyncio.Event()
        self._closing = False

    # -- host key -----------------------------------------------------------------

    async def _server_key(self) -> asyncssh.SSHKey:
        t = self.target
        try:
            key = await asyncio.wait_for(
                asyncssh.get_server_host_key(t.host, t.port, config=None), timeout=CONNECT_TIMEOUT
            )
        except TimeoutError:
            raise SSHUnavailable(f"no answer from {t.host}:{t.port} within {CONNECT_TIMEOUT:.0f} s") from None
        except (OSError, asyncssh.Error) as exc:
            raise SSHUnavailable(f"cannot reach {t.host}:{t.port}: {_reason(exc)}") from None
        if key is None:
            raise SSHUnavailable(f"{t.host}:{t.port} did not present a host key")
        return key

    async def probe(self) -> str:
        """Fetch the server's host-key fingerprint. Nothing is authenticated or sent."""
        return (await self._server_key()).get_fingerprint("sha256")

    # -- connect / close ----------------------------------------------------------

    def _auth_options(self) -> dict[str, Any]:
        """Key-based only. An encrypted key is used through ssh-agent; no passphrase is ever asked for."""
        path = self.target.key_path
        try:
            key = asyncssh.read_private_key(path)
        except (asyncssh.KeyEncryptionError, asyncssh.KeyImportError) as exc:
            if "assphrase" not in str(exc) and not isinstance(exc, asyncssh.KeyEncryptionError):
                raise SSHError(f"cannot read the private key {path}: {exc}") from None
            if not os.environ.get("SSH_AUTH_SOCK"):
                raise SSHError(
                    f"the key {path} is protected by a passphrase: add it to ssh-agent (ssh-add) and start "
                    "the proxy with SSH_AUTH_SOCK set; the proxy never asks for a passphrase"
                ) from None
            pub = path + ".pub"
            return {
                "client_keys": [path],
                "ignore_encrypted": True,
                "agent_identities": [pub] if os.path.isfile(pub) else None,
            }
        except OSError as exc:
            raise SSHError(f"cannot read the private key {path}: {exc.strerror or exc}") from None
        return {"client_keys": [key], "agent_path": None}

    async def _open(self) -> None:
        """One connection attempt against the pinned key, then the port forwards."""
        assert self.fingerprint
        key = await self._server_key()
        found = key.get_fingerprint("sha256")
        if found != self.fingerprint:
            raise HostKeyChanged(self.fingerprint, found)
        t = self.target
        try:
            conn = await asyncssh.connect(
                t.host,
                t.port,
                username=t.user,
                known_hosts=([key], [], []),  # exactly the confirmed key, nothing else
                config=None,  # never read ~/.ssh/config: no ProxyCommand, no aliases
                password=None,
                preferred_auth=["publickey"],
                agent_forwarding=False,
                x509_trusted_certs=None,
                keepalive_interval=KEEPALIVE_INTERVAL,
                keepalive_count_max=KEEPALIVE_COUNT,
                connect_timeout=CONNECT_TIMEOUT,
                login_timeout=CONNECT_TIMEOUT,
                **self._auth_options(),
            )
        except asyncssh.HostKeyNotVerifiable:
            raise HostKeyChanged(self.fingerprint, "another key") from None
        except asyncssh.PermissionDenied:
            raise SSHError(
                f"the pod refused the key {t.key_path} for user {t.user}: check that its public half is on "
                "your Runpod account"
            ) from None
        except TimeoutError:
            raise SSHUnavailable(f"timed out connecting to {t.host}:{t.port}") from None
        except (OSError, asyncssh.Error) as exc:
            raise SSHUnavailable(f"cannot connect to {t.host}:{t.port}: {_reason(exc)}") from None
        try:
            listeners = []
            for name, remote_port in self.remote_ports.items():
                listener = None
                wanted = self.local_ports.get(name, 0)
                for local_port in dict.fromkeys((wanted, 0)):  # same local port after a reconnect if free
                    try:
                        listener = await conn.forward_local_port("127.0.0.1", local_port, "127.0.0.1", remote_port)
                        break
                    except OSError:
                        continue
                if listener is None:
                    raise SSHError("cannot open a local loopback port for the tunnel")
                listeners.append(listener)
                self.local_ports[name] = listener.get_port()
        except BaseException:
            conn.abort()
            raise
        self._conn, self._listeners = conn, listeners

    async def connect(self, fingerprint: str) -> None:
        """Connect to the server whose key has this (user-confirmed) fingerprint."""
        await self.close()
        self._closing = False
        self.fingerprint = fingerprint
        await self._open()
        self._set("up")
        self._watcher = asyncio.create_task(self._watch(), name="pod-ssh-watch")

    async def close(self) -> None:
        self._closing = True
        watcher, self._watcher = self._watcher, None
        if watcher and watcher is not asyncio.current_task():
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watcher
        conn, self._conn = self._conn, None
        self._listeners = []
        if conn is not None:
            conn.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(conn.wait_closed(), timeout=5)
        self.state, self.detail = "closed", ""
        self._up.clear()

    def _set(self, state: str, detail: str = "") -> None:
        self.state, self.detail = state, detail
        if state == "up":
            self._up.set()
        else:
            self._up.clear()

    async def _notify(self, state: str, detail: str = "") -> None:
        if self._on_event is None:
            return
        try:
            result = self._on_event(state, detail)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # a listener must never kill the reconnect loop
            log.exception("ssh event handler failed")

    async def _watch(self) -> None:
        """Wait for the connection to end, then reconnect with backoff against the pinned key."""
        while not self._closing:
            conn = self._conn
            if conn is not None:
                await conn.wait_closed()
            if self._closing:
                return
            self._conn, self._listeners = None, []
            self._set("down", "connection lost")
            log.warning("ssh connection to %s:%s lost; reconnecting", self.target.host, self.target.port)
            await self._notify("down", "connection lost")
            attempt = 0
            while not self._closing:
                await asyncio.sleep(self.backoff[min(attempt, len(self.backoff) - 1)])
                attempt += 1
                try:
                    await self._open()
                except HostKeyChanged as exc:
                    # Never reconnect to another key: stop and make the user confirm again.
                    log.error("ssh: %s", exc)
                    self._set("hostkey_changed", str(exc))
                    await self._notify("hostkey_changed", str(exc))
                    return
                except SSHError as exc:
                    self.detail = f"reconnecting (attempt {attempt}): {exc}"
                    continue
                self._set("up")
                log.info("ssh connection to %s:%s re-established", self.target.host, self.target.port)
                await self._notify("up", "")
                break

    # -- use ------------------------------------------------------------------------

    @property
    def up(self) -> bool:
        return self.state == "up" and self._conn is not None

    async def wait_up(self, timeout: float) -> bool:
        if self.up:
            return True
        if self.state in ("closed", "hostkey_changed"):
            return False
        try:
            await asyncio.wait_for(self._up.wait(), timeout)
        except TimeoutError:
            return False
        return self.up

    def url(self, name: str) -> str:
        port = self.local_ports.get(name)
        return f"http://127.0.0.1:{port}" if port and self.up else ""

    def _require(self) -> asyncssh.SSHClientConnection:
        if not self.up or self._conn is None:
            raise SSHUnavailable("the SSH connection is down")
        return self._conn

    async def run(self, argv: Sequence[str], *, timeout: float = 60.0, input: str | None = None) -> Result:
        """Run one remote command built from an argument list; never raises on a non-zero exit."""
        conn = self._require()
        try:
            done = await asyncio.wait_for(
                conn.run(quote_argv(argv), check=False, input=input, encoding="utf-8", errors="replace"),
                timeout,
            )
        except TimeoutError:
            raise SSHError(f"remote command timed out after {timeout:.0f} s: {argv[0]}") from None
        except (OSError, asyncssh.Error) as exc:
            raise SSHUnavailable(f"the SSH connection failed during a command: {_reason(exc)}") from None
        rc = done.exit_status
        if rc is None:  # killed by a signal, or the channel closed without a status
            rc = -1
        return Result(rc, str(done.stdout or ""), str(done.stderr or ""))

    async def start(self, argv: Sequence[str]) -> asyncssh.SSHClientProcess:
        """Start a long-running remote command (log tail); the caller reads its stdout and closes it."""
        conn = self._require()
        try:
            return await conn.create_process(quote_argv(argv), encoding="utf-8", errors="replace")
        except (OSError, asyncssh.Error) as exc:
            raise SSHUnavailable(f"the SSH connection failed: {_reason(exc)}") from None

    async def upload(self, files: Sequence[tuple[str, bytes, int]]) -> None:
        """Write files by SFTP. Each is created with its final mode and renamed into place."""
        conn = self._require()
        try:
            async with conn.start_sftp_client() as sftp:
                made: set[str] = set()
                for path, data, mode in files:
                    directory = os.path.dirname(path)
                    if directory not in made:
                        await sftp.makedirs(directory, exist_ok=True)
                        made.add(directory)
                    tmp = path + ".tmp"
                    with contextlib.suppress(asyncssh.SFTPError):
                        await sftp.remove(tmp)
                    async with sftp.open(tmp, "wb", attrs=asyncssh.SFTPAttrs(permissions=mode)) as fh:
                        await fh.write(data)
                    await sftp.chmod(tmp, mode)
                    await sftp.posix_rename(tmp, path)
        except (OSError, asyncssh.Error) as exc:
            raise SSHUnavailable(f"file upload failed: {_reason(exc)}") from None


def _reason(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    return text if len(text) < 300 else text[:300] + "…"
