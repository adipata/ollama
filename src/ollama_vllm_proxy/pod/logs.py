"""Remote log files relayed to the browser: ``tail -F`` over an SSH channel, as server-sent events.

Lines are redacted before they leave this module and are never stored.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import AsyncIterator, Iterable
from typing import Any

from .ssh import PodSSH, SSHError

SOURCES = ("vllm", "install", "download", "helper")
TAIL_LINES = 500
_MAX_LINE = 4000

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(Bearer\s+)(?!\*\*\*)[A-Za-z0-9._~+/=-]+"), r"\1***"),
    (re.compile(r"(?i)(--api-key[ =])\S+"), r"\1***"),
    # name = value / "name": "value" for anything that sounds like a credential
    (re.compile(r"""(?i)((?:api[_-]?key|token|passw(?:or)?d|secret|authorization)["']?\s*[:=]\s*)(["']?)"""
                r"""(?!\*\*\*|Bearer\b)[^\s"',}]+\2"""), r"\1\2***\2"),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}"), "hf_***"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "sk-***"),
    (re.compile(r"\brpa_[A-Za-z0-9]{16,}"), "rpa_***"),
]


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove known secret values and anything that looks like a key or token."""
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "***")
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def clean(line: str, secrets: Iterable[str] = ()) -> str:
    line = _ANSI.sub("", line.rstrip("\r\n")).replace("\r", " ")
    if len(line) > _MAX_LINE:
        line = line[:_MAX_LINE] + " …"
    return redact(line, secrets)


def log_path(remote_root: str, source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"unknown log source '{source}'")
    return f"{remote_root}/vllm.log" if source == "vllm" else f"{remote_root}/ovp/{source}.log"


def sse(payload: Any) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


async def follow(ssh: PodSSH, path: str, secrets: Iterable[str] = (), lines: int = TAIL_LINES) -> AsyncIterator[list[str]]:
    """Yield batches of redacted lines: the last ``lines`` of the file, then whatever is appended.

    Ends when the SSH channel ends. The remote ``tail`` is terminated when the
    consumer stops iterating.
    """
    secrets = [s for s in secrets if s]
    proc = await ssh.start(["tail", "-n", str(lines), "-F", path])
    try:
        pending: asyncio.Task | None = None
        batch: list[str] = []
        while True:
            if pending is None:
                pending = asyncio.ensure_future(proc.stdout.readline())
            # Collect lines that arrive close together into one event.
            done, _ = await asyncio.wait({pending}, timeout=0.15 if batch else None)
            if not done:
                yield batch
                batch = []
                continue
            line = pending.result()
            pending = None
            if not line:
                break
            batch.append(clean(line, secrets))
            if len(batch) >= 200:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        if pending is not None:
            pending.cancel()
        with contextlib.suppress(Exception):
            proc.terminate()
        with contextlib.suppress(Exception):
            proc.close()


async def stream(manager: Any, source: str) -> AsyncIterator[bytes]:
    """SSE body for one log source. Reconnects the tail when the SSH connection comes back."""
    path = log_path(manager.cfg.remote_root, source)
    first = True
    while True:
        ssh = manager.ssh
        if ssh is None or ssh.state in ("closed", "hostkey_changed"):
            yield sse({"info": "No pod connected."})
            return
        if not ssh.up:
            yield sse({"info": "SSH connection is down; waiting for it to come back…"})
            if not await ssh.wait_up(600):
                return
            continue
        try:
            yield sse({"reset": True, "path": path} if first else {"reset": True, "path": path, "info": "reconnected"})
            first = False
            async for batch in follow(ssh, path, manager.secret_values()):
                yield sse({"lines": batch})
        except SSHError as exc:
            yield sse({"info": str(exc)})
        except (asyncio.CancelledError, GeneratorExit):
            raise
        await asyncio.sleep(1.0)


async def full_text(ssh: PodSSH, path: str, secrets: Iterable[str] = ()) -> str:
    """The whole log file, redacted (for the download link)."""
    result = await ssh.run(["cat", "--", path], timeout=120)
    if not result.ok:
        return ""
    secrets = [s for s in secrets if s]
    return "\n".join(clean(line, secrets) for line in result.stdout.splitlines()) + "\n"
