"""The pasted Runpod ssh command: parsed, never executed (PLAN 12.4)."""

from __future__ import annotations

import os

import pytest

from ollama_vllm_proxy.pod.connstr import ConnStrError, key_problem, parse

HOME = "/home/someone"


def p(text, **kw):
    kw.setdefault("home", HOME)
    return parse(text, **kw)


def test_runpod_form():
    c = p("ssh root@213.173.99.7 -p 10909 -i ~/.ssh/id_ed25519")
    assert (c.user, c.host, c.port, c.key_path) == ("root", "213.173.99.7", 10909, f"{HOME}/.ssh/id_ed25519")


@pytest.mark.parametrize("text", [
    "ssh -i ~/.ssh/id_ed25519 -p 10909 root@213.173.99.7",
    "ssh -p 10909 root@213.173.99.7 -i ~/.ssh/id_ed25519",
    "  ssh   -p10909 -i~/.ssh/id_ed25519 root@213.173.99.7  ",
    "root@213.173.99.7 -p 10909 -i '~/.ssh/id_ed25519'",
])
def test_reordered_options(text):
    c = p(text)
    assert (c.user, c.host, c.port, c.key_path) == ("root", "213.173.99.7", 10909, f"{HOME}/.ssh/id_ed25519")


def test_missing_parts_fall_back():
    c = p("ssh pod.example.com", default_key="~/keys/runpod")
    assert (c.user, c.host, c.port, c.key_path) == ("root", "pod.example.com", 22, f"{HOME}/keys/runpod")
    assert p("ssh ubuntu@10.0.0.5 -p 2222", default_key="/etc/keys/k").key_path == "/etc/keys/k"
    assert p("ssh root@10.0.0.5 -i /abs/key").port == 22


@pytest.mark.parametrize("text, named", [
    ("ssh root@1.2.3.4 -p 22 -o ProxyCommand='nc evil 1'", "-o"),
    ("ssh -o StrictHostKeyChecking=no root@1.2.3.4", "-o"),
    ("ssh -L 8000:localhost:8000 root@1.2.3.4", "-L"),
    ("ssh -J jump@host root@1.2.3.4", "-J"),
    ("ssh -N root@1.2.3.4", "-N"),
    ("ssh -A root@1.2.3.4", "-A"),
    ("ssh root@1.2.3.4 -p 22 uname -a", "uname"),
    ("ssh root@1.2.3.4 'rm -rf /'", "rm -rf /"),
    ("ssh root@1.2.3.4 -p", "-p"),
    ("ssh root@1.2.3.4 -p 22 -p 23", "-p"),
    ("ssh root@1.2.3.4 -p 99999", "99999"),
    ("ssh root@1.2.3.4 -p abc", "abc"),
    ("ssh root@1.2.3.4; touch /tmp/pwned", "touch"),
    ("ssh root@1.2.3.4;", "1.2.3.4;"),
    ("ssh 'root@$(id)'", "$(id)"),
    ("ssh ro`ot@1.2.3.4", "ro`ot"),
    ("ssh root@1.2.3.4 -i 'unbalanced", "cannot read"),
    ("ssh bad user@1.2.3.4", "user@1.2.3.4"),
    ("ssh root@1.2.3.4 -i ~other/.ssh/key", "~other"),
    ("ssh abc123@ssh.runpod.io -i ~/.ssh/id_ed25519", "exposed TCP"),
    ("ssh", "no host"),
    ("", "paste"),
    ("ssh root@1.2.3.4\nrm -rf /", "single line"),
])
def test_rejected_with_a_message_naming_the_problem(text, named):
    with pytest.raises(ConnStrError) as err:
        p(text)
    assert named in str(err.value)


def test_key_file_checks(tmp_path):
    key = tmp_path / "id"
    assert "does not exist" in key_problem(str(key))
    key.write_text("x")
    os.chmod(key, 0o644)
    assert "chmod 600" in key_problem(str(key))
    os.chmod(key, 0o600)
    assert key_problem(str(key)) is None
    assert "not a regular file" in key_problem(str(tmp_path))
