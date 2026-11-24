"""Settings read from the environment (optionally from a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DEFAULT_ORIGINS = (
    r"^(https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(:\d+)?"
    r"|app://.*|file://.*|tauri://.*|vscode-webview://.*|vscode-file://.*)$"
)


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, existing variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _aliases(value: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (value or "").split(","):
        if "=" in part:
            alias, _, target = part.partition("=")
            if alias.strip() and target.strip():
                out[alias.strip()] = target.strip()
    return out


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 11434
    vllm_base_url: str = ""  # external mode when set
    vllm_api_key: str = ""
    ssh_key_path: str = "~/.ssh/id_ed25519"
    default_model: str = ""
    model_aliases: dict[str, str] = field(default_factory=dict)
    strict_models: bool = False
    context_length: int | None = None
    ollama_version: str = "0.12.0"
    allowed_origins: str = ""  # comma-separated extra origins; "*" allows all
    log_level: str = "info"
    data_dir: str = "./data"
    admin_token: str = ""
    sample_interval_s: float = 1.5
    gpu_exporter_url: str = ""  # external mode only; "off" disables
    retention_days: int = 30

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        ctx = e.get("CONTEXT_LENGTH", "").strip()
        return cls(
            host=e.get("PROXY_HOST", "127.0.0.1"),
            port=int(e.get("PROXY_PORT", "11434")),
            vllm_base_url=e.get("VLLM_BASE_URL", "").rstrip("/"),
            vllm_api_key=e.get("VLLM_API_KEY", ""),
            ssh_key_path=e.get("SSH_KEY_PATH", "~/.ssh/id_ed25519"),
            default_model=e.get("DEFAULT_MODEL", ""),
            model_aliases=_aliases(e.get("MODEL_ALIASES")),
            strict_models=_bool(e.get("STRICT_MODELS")),
            context_length=int(ctx) if ctx else None,
            ollama_version=e.get("OLLAMA_VERSION", "0.12.0"),
            allowed_origins=e.get("ALLOWED_ORIGINS", ""),
            log_level=e.get("LOG_LEVEL", "info"),
            data_dir=e.get("DATA_DIR", "./data"),
            admin_token=e.get("ADMIN_TOKEN", ""),
            sample_interval_s=float(e.get("SAMPLE_INTERVAL_S", "1.5")),
            gpu_exporter_url=e.get("GPU_EXPORTER_URL", ""),
            retention_days=int(e.get("RETENTION_DAYS", "30")),
        )

    @property
    def external_mode(self) -> bool:
        return bool(self.vllm_base_url)

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "proxy.sqlite3")

    @property
    def admin_token_path(self) -> str:
        return str(Path(self.data_dir) / "admin_token.txt")

    def origin_regex(self) -> str | None:
        """Regex for allowed CORS origins, or None when everything is allowed."""
        extra = [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
        if "*" in extra:
            return None
        if not extra:
            return DEFAULT_ORIGINS
        import re

        return DEFAULT_ORIGINS[:-2] + "|" + "|".join(re.escape(o) for o in extra) + ")$"
