"""API keys: generation, hashing, CRUD and a small lookup cache.

Only the SHA-256 of a key is stored. Keys are 32 random base62 characters, so
a plain hash is sufficient (there is nothing to brute-force).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import string
import time
from pathlib import Path
from typing import Any

from .store import Store

KEY_PREFIX = "sk-ovp-"
_ALPHABET = string.ascii_letters + string.digits
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class KeyError_(ValueError):
    """Invalid key operation (bad name, duplicate, unknown id)."""


def generate_key() -> str:
    return KEY_PREFIX + "".join(secrets.choice(_ALPHABET) for _ in range(32))


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _public(row: dict[str, Any]) -> dict[str, Any]:
    status = "revoked" if row["revoked_at"] else ("active" if row["enabled"] else "disabled")
    return {
        "id": row["id"],
        "name": row["name"],
        "prefix": row["key_prefix"],
        "enabled": bool(row["enabled"]),
        "status": status,
        "created_at": row["created_at"],
        "revoked_at": row["revoked_at"],
    }


class KeyStore:
    """CRUD on api_keys plus a hash -> (id, name) cache for the request path."""

    CACHE_TTL_S = 3.0  # picks up changes made by the CLI while the server runs

    def __init__(self, store: Store):
        self.store = store
        self._cache: dict[str, tuple[int, str]] = {}
        self._loaded_at = 0.0

    # -- request path ---------------------------------------------------------

    def invalidate(self) -> None:
        self._loaded_at = 0.0

    def _load(self) -> None:
        rows = self.store.read(
            "SELECT id, name, key_sha256 FROM api_keys WHERE enabled = 1 AND revoked_at IS NULL"
        )
        self._cache = {r["key_sha256"]: (r["id"], r["name"]) for r in rows}
        self._loaded_at = time.monotonic()

    def lookup(self, token: str) -> tuple[int, str] | None:
        if not token or not token.startswith(KEY_PREFIX):
            return None
        if time.monotonic() - self._loaded_at > self.CACHE_TTL_S:
            self._load()
        return self._cache.get(hash_key(token))

    def count_active(self) -> int:
        row = self.store.read_one(
            "SELECT COUNT(*) AS n FROM api_keys WHERE enabled = 1 AND revoked_at IS NULL"
        )
        return int(row["n"]) if row else 0

    # -- management -------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        rows = self.store.read("SELECT * FROM api_keys ORDER BY revoked_at IS NOT NULL, created_at DESC")
        return [_public(r) for r in rows]

    def create(self, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not NAME_RE.match(name):
            raise KeyError_("name must be 1-64 characters: letters, digits, '.', '_', ':' or '-'")
        key = generate_key()
        try:
            key_id = self.store.write(
                "INSERT INTO api_keys (name, key_prefix, key_sha256, enabled, created_at) VALUES (?, ?, ?, 1, ?)",
                (name, key[: len(KEY_PREFIX) + 4], hash_key(key), time.time()),
            )
        except sqlite3.IntegrityError:
            raise KeyError_(f"a key named '{name}' already exists") from None
        self.invalidate()
        out = self.get(key_id)
        out["key"] = key  # the only time the full key is available
        return out

    def get(self, key_id: int) -> dict[str, Any]:
        row = self.store.read_one("SELECT * FROM api_keys WHERE id = ?", (key_id,))
        if row is None:
            raise KeyError_("no such key")
        return _public(row)

    def find(self, name_or_id: str) -> dict[str, Any]:
        """Resolve an active key by id or name (CLI helper)."""
        if str(name_or_id).isdigit():
            return self.get(int(name_or_id))
        row = self.store.read_one(
            "SELECT * FROM api_keys WHERE name = ? ORDER BY revoked_at IS NOT NULL, id DESC", (name_or_id,)
        )
        if row is None:
            raise KeyError_("no such key")
        return _public(row)

    def update(self, key_id: int, *, name: str | None = None, enabled: bool | None = None) -> dict[str, Any]:
        current = self.get(key_id)
        if current["status"] == "revoked":
            raise KeyError_("a revoked key cannot be changed")
        if name is not None:
            name = name.strip()
            if not NAME_RE.match(name):
                raise KeyError_("name must be 1-64 characters: letters, digits, '.', '_', ':' or '-'")
            try:
                self.store.write("UPDATE api_keys SET name = ? WHERE id = ?", (name, key_id))
            except sqlite3.IntegrityError:
                raise KeyError_(f"a key named '{name}' already exists") from None
        if enabled is not None:
            self.store.write("UPDATE api_keys SET enabled = ? WHERE id = ?", (1 if enabled else 0, key_id))
        self.invalidate()
        return self.get(key_id)

    def revoke(self, key_id: int) -> dict[str, Any]:
        self.get(key_id)
        self.store.write(
            "UPDATE api_keys SET enabled = 0, revoked_at = COALESCE(revoked_at, ?) WHERE id = ?",
            (time.time(), key_id),
        )
        self.invalidate()
        return self.get(key_id)


# ---------------------------------------------------------------------------
# Admin token
# ---------------------------------------------------------------------------


def load_admin_token(path: str, override: str = "", rotate: bool = False) -> str:
    """Return the admin token, creating the 0600 token file on first use."""
    if override and not rotate:
        return override
    p = Path(path)
    if p.is_file() and not rotate:
        token = p.read_text(encoding="utf-8").strip()
        if token:
            return token
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = "adm-" + secrets.token_urlsafe(32)
    fd = os.open(str(p), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token + "\n")
    os.chmod(str(p), 0o600)
    return token


def token_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
