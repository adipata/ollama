"""Command line and settings."""

from __future__ import annotations

import os
import stat

from ollama_vllm_proxy.__main__ import main
from ollama_vllm_proxy.config import Settings
from ollama_vllm_proxy.keys import KeyStore, load_admin_token
from ollama_vllm_proxy.store import Store


def test_keys_cli_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert main(["keys", "create", "laptop"]) == 0
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert key.startswith("sk-ovp-")
    store = Store(str(tmp_path / "proxy.sqlite3"))
    keys = KeyStore(store)
    assert keys.lookup(key) is not None
    assert main(["keys", "disable", "laptop"]) == 0
    keys.invalidate()
    assert keys.lookup(key) is None
    assert main(["keys", "enable", "laptop"]) == 0
    assert main(["keys", "revoke", "laptop"]) == 0
    assert main(["keys", "list"]) == 0
    assert "revoked" in capsys.readouterr().out
    assert main(["keys", "revoke", "nobody"]) == 1
    assert main(["keys", "create", "laptop"]) == 0  # a revoked name can be reused
    store.stop()


def test_admin_token_file_is_private_and_stable(tmp_path):
    path = str(tmp_path / "d" / "admin_token.txt")
    token = load_admin_token(path)
    assert token.startswith("adm-") and load_admin_token(path) == token
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert load_admin_token(path, rotate=True) != token
    assert load_admin_token(path, override="from-env") == "from-env"


def test_database_file_is_private(tmp_path):
    store = Store(str(tmp_path / "data" / "proxy.sqlite3"))
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(tmp_path / "data").st_mode) == 0o700
    store.stop()


def test_settings_from_env():
    s = Settings.from_env({"PROXY_PORT": "1234", "MODEL_ALIASES": "a=b, c:latest=d", "STRICT_MODELS": "true",
                           "VLLM_BASE_URL": "http://x:8000/", "CONTEXT_LENGTH": "4096"})
    assert s.port == 1234 and s.model_aliases == {"a": "b", "c:latest": "d"} and s.strict_models
    assert s.external_mode and s.vllm_base_url == "http://x:8000" and s.context_length == 4096
    assert not Settings.from_env({}).external_mode
    import re

    assert re.match(Settings().origin_regex(), "http://localhost:3000")
    assert not re.match(Settings().origin_regex(), "https://evil.example")
    assert re.match(Settings(allowed_origins="https://ui.example").origin_regex(), "https://ui.example")
    assert Settings(allowed_origins="*").origin_regex() is None
