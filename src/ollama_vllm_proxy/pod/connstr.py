"""Parser for the ``ssh`` command Runpod shows under "SSH over exposed TCP".

The pasted text is only tokenised (``shlex``) and read; it is never executed.
Exactly three things are understood - ``user@host``, ``-p <port>`` and
``-i <key>`` - and anything else is rejected by name, so an option such as
``-o ProxyCommand=...`` cannot be smuggled in and silently ignored.
"""

from __future__ import annotations

import os
import re
import shlex
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

_HOST = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?)*"
                   r"|[0-9A-Fa-f:]{2,39})$")
_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,31}$")


class ConnStrError(ValueError):
    """The pasted command cannot be used; the message says which part."""


@dataclass
class Connection:
    user: str
    host: str
    port: int
    key_path: str

    def as_dict(self) -> dict:
        return asdict(self)


def expand_key_path(path: str, home: str | None = None) -> str:
    """Expand a leading ``~`` against the home of the user running the proxy."""
    home = home or str(Path.home())
    if path == "~":
        return home
    if path.startswith("~/"):
        return str(Path(home) / path[2:])
    if path.startswith("~"):
        raise ConnStrError(f"key path '{path}': only '~/' (your own home directory) is supported")
    return str(Path(path).absolute()) if not os.path.isabs(path) else path


def parse(command: str, default_key: str = "~/.ssh/id_ed25519", home: str | None = None) -> Connection:
    """Parse ``ssh root@1.2.3.4 -p 10909 -i ~/.ssh/id_ed25519`` (options in any order)."""
    if not isinstance(command, str) or not command.strip():
        raise ConnStrError("paste the ssh command Runpod shows under Connect → SSH over exposed TCP")
    if len(command) > 1024 or any(ord(c) < 32 for c in command.strip()):
        raise ConnStrError("the connection string must be a single line")
    try:
        tokens = shlex.split(command.strip())
    except ValueError as exc:
        raise ConnStrError(f"cannot read the command: {exc}") from None
    if tokens and tokens[0] == "ssh":
        tokens = tokens[1:]
    target: str | None = None
    port: str | None = None
    key: str | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-p", "-i"):
            if i + 1 >= len(tokens):
                raise ConnStrError(f"option '{tok}' needs a value")
            value = tokens[i + 1]
            i += 2
        elif tok[:2] in ("-p", "-i") and len(tok) > 2:
            tok, value = tok[:2], tok[2:]
            i += 1
        elif tok.startswith("-"):
            raise ConnStrError(
                f"option '{tok}' is not supported: only user@host, -p <port> and -i <key file> are accepted"
            )
        elif target is None:
            target = tok
            i += 1
            continue
        else:
            raise ConnStrError(
                f"unexpected '{tok}' after the host: a remote command is not accepted, paste only the ssh command"
            )
        if tok == "-p":
            if port is not None:
                raise ConnStrError("option '-p' is given twice")
            port = value
        else:
            if key is not None:
                raise ConnStrError("option '-i' is given twice")
            key = value
    if target is None:
        raise ConnStrError("no host found: expected something like 'ssh root@213.173.99.7 -p 10909 -i ~/.ssh/id_ed25519'")
    user, sep, host = target.rpartition("@")
    if not sep:
        user = "root"
    if not _USER.match(user):
        raise ConnStrError(f"user name '{user}' is not valid")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not _HOST.match(host):
        raise ConnStrError(f"host '{host}' is not a valid host name or address")
    if host.lower().endswith("ssh.runpod.io"):
        raise ConnStrError(
            "this is Runpod's proxied SSH, which supports neither port forwarding nor file transfer: "
            "use the command under 'SSH over exposed TCP' instead"
        )
    if port is None:
        port_n = 22
    else:
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ConnStrError(f"port '{port}' is not a valid port number")
        port_n = int(port)
    key_path = expand_key_path(key if key is not None else default_key, home)
    if any(ord(c) < 32 for c in key_path):
        raise ConnStrError("key path contains control characters")
    return Connection(user=user, host=host, port=port_n, key_path=key_path)


def key_problem(path: str) -> str | None:
    """Why the private key file cannot be used, or None. The key itself is never read here."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return f"key file {path} does not exist"
    except OSError as exc:
        return f"key file {path} cannot be read ({exc.strerror})"
    if not stat.S_ISREG(st.st_mode):
        return f"{path} is not a regular file"
    if st.st_mode & 0o077:
        return f"key file {path} is readable by other users (mode {st.st_mode & 0o777:o}); run: chmod 600 {shlex.quote(path)}"
    if not os.access(path, os.R_OK):
        return f"key file {path} is not readable by the proxy"
    return None
