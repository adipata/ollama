"""Deployment profiles: what to serve. How it runs on a given pod is decided by ``fit``.

A profile is structured data (JSON in ``profiles.data``), never a free-form
command. Every field is validated against a strict pattern when it is saved
and again when it is used, and the ``vllm serve`` command is built as an
argument list in which the application-owned options always come last.
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

DEFAULT_PROFILE_NAME = "qwen3.8-27b"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$")
_DIR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FLAG = re.compile(r"^--[a-z0-9][a-z0-9-]{0,62}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
_VERSION = re.compile(r"^[0-9]+(\.[0-9]+){1,3}([a-z0-9.+-]{0,16})?$")
_PYTHON = re.compile(r"^3\.[0-9]{1,2}$")

# Set by the application on every start; a profile can neither change nor repeat them.
# --config is excluded as well because a config file could set any of the others.
OWNED_FLAGS = frozenset({
    "--host", "--port", "--served-model-name", "--max-model-len", "--tensor-parallel-size", "--api-key",
    "--uds", "--config", "--model", "--gpu-memory-utilization", "--ssl-keyfile", "--ssl-certfile",
    "--ssl-ca-certs", "--root-path",
})
# Names the launchers set themselves or that would change which code runs.
RESERVED_ENV = frozenset({
    "VLLM_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "LD_AUDIT", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "HF_HOME", "BASH_ENV", "ENV", "IFS",
    "RUNPOD_API_KEY", "JUPYTER_PASSWORD",
})
RESERVED_DIRS = frozenset({"ovp", "venv", ".hf", "vllm.log"})
TP_SIZES = (1, 2, 4, 8)


class ProfileError(ValueError):
    """A profile field is invalid; the message names the field."""


def default_profile() -> dict[str, Any]:
    """The profile of PLAN section 12.3."""
    return {
        "served_model_name": "dev-test",
        "variants": [
            {"label": "BF16", "repo": "Qwen/Qwen3.8-27B", "revision": "", "dir": "qwen3.8-27b-bf16", "args": []},
            {"label": "FP8", "repo": "Qwen/Qwen3.8-27B-FP8", "revision": "", "dir": "qwen3.8-27b-fp8",
             "args": [{"flag": "--kv-cache-dtype", "value": "fp8"}]},
        ],
        "context_wanted": 131072,
        "context_min": 65536,
        "args": [
            {"flag": "--enable-prefix-caching", "value": ""},
            {"flag": "--enable-auto-tool-choice", "value": ""},
            {"flag": "--tool-call-parser", "value": "qwen3_coder"},
            {"flag": "--reasoning-parser", "value": "qwen3"},
        ],
        "gpu_memory_utilization": None,
        "vllm_version": "0.34.0",
        "python_version": "3.12",
        "env": {},
    }


def _text(value: Any, field: str, pattern: re.Pattern[str], what: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise ProfileError(f"{field}: {what}")
    return value


def _int(value: Any, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ProfileError(f"{field}: must be a whole number between {low} and {high}")
    return value


def _args(value: Any, field: str) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 64:
        raise ProfileError(f"{field}: must be a list of at most 64 flag/value pairs")
    out = []
    seen: set[str] = set()
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProfileError(f"{field}[{i}]: must be an object with 'flag' and 'value'")
        flag = _text(item.get("flag"), f"{field}[{i}].flag", _FLAG, "must look like --some-option (lower case, no '=')")
        if flag in OWNED_FLAGS:
            raise ProfileError(f"{field}[{i}].flag: {flag} is set by the application and cannot be given in a profile")
        if flag in seen:
            raise ProfileError(f"{field}[{i}].flag: {flag} is given twice")
        seen.add(flag)
        val = item.get("value")
        val = "" if val is None else val
        if not isinstance(val, str) or len(val) > 256 or any(ord(c) < 32 or ord(c) == 127 for c in val):
            raise ProfileError(f"{field}[{i}].value: must be a single line of at most 256 characters")
        # A value that looks like an option would be read by vLLM as one, not as this flag's value.
        if val.startswith("-") and not re.match(r"^-[0-9.]+$", val):
            raise ProfileError(f"{field}[{i}].value: a value cannot start with '-'")
        out.append({"flag": flag, "value": val})
    return out


def validate(name: Any, data: Any) -> tuple[str, dict[str, Any]]:
    """Return the cleaned (name, data) or raise ProfileError. Unknown fields are dropped."""
    name = _text(name, "name", _NAME, "use letters, digits, '.', '_' and '-' (at most 64 characters)")
    if not isinstance(data, dict):
        raise ProfileError("profile must be an object")
    out: dict[str, Any] = {}
    out["served_model_name"] = _text(
        data.get("served_model_name"), "served_model_name", _NAME,
        "use letters, digits, '.', '_' and '-' (at most 64 characters)",
    )
    variants = data.get("variants")
    if not isinstance(variants, list) or not 1 <= len(variants) <= 8:
        raise ProfileError("variants: between 1 and 8 variants are required")
    out["variants"] = []
    labels: set[str] = set()
    dirs: set[str] = set()
    for i, var in enumerate(variants):
        f = f"variants[{i}]"
        if not isinstance(var, dict):
            raise ProfileError(f"{f}: must be an object")
        label = _text(var.get("label"), f"{f}.label", _LABEL, "use letters, digits, '.', '_' and '-' (at most 32)")
        repo = _text(var.get("repo"), f"{f}.repo", _REPO, "must be a Hugging Face repo id such as Qwen/Qwen3.8-27B")
        revision = var.get("revision") or ""
        if revision:
            _text(revision, f"{f}.revision", _REVISION, "must be a branch, tag or commit id")
        directory = _text(var.get("dir"), f"{f}.dir", _DIR,
                          "must be a plain directory name (letters, digits, '.', '_', '-'), not a path")
        if directory in RESERVED_DIRS or directory in (".", ".."):
            raise ProfileError(f"{f}.dir: '{directory}' is reserved")
        if label.lower() in labels:
            raise ProfileError(f"{f}.label: '{label}' is used twice")
        if directory in dirs:
            raise ProfileError(f"{f}.dir: '{directory}' is used twice")
        labels.add(label.lower())
        dirs.add(directory)
        out["variants"].append({"label": label, "repo": repo, "revision": revision, "dir": directory,
                                "args": _args(var.get("args"), f"{f}.args")})
    out["context_wanted"] = _int(data.get("context_wanted"), "context_wanted", 1024, 4_194_304)
    out["context_min"] = _int(data.get("context_min", out["context_wanted"]), "context_min", 1024, 4_194_304)
    if out["context_min"] > out["context_wanted"]:
        raise ProfileError("context_min: cannot be larger than the wanted context")
    out["args"] = _args(data.get("args"), "args")
    for i, var in enumerate(out["variants"]):
        common = {a["flag"] for a in out["args"]}
        for a in var["args"]:
            if a["flag"] in common:
                raise ProfileError(f"variants[{i}].args: {a['flag']} is already a common argument")
    util = data.get("gpu_memory_utilization")
    if util in ("", None):
        out["gpu_memory_utilization"] = None
    else:
        if isinstance(util, bool) or not isinstance(util, (int, float)) or not 0.1 <= float(util) <= 1.0:
            raise ProfileError("gpu_memory_utilization: must be between 0.1 and 1.0, or empty for the vLLM default")
        out["gpu_memory_utilization"] = round(float(util), 3)
    out["vllm_version"] = _text(data.get("vllm_version"), "vllm_version", _VERSION, "must be a version such as 0.34.0")
    out["python_version"] = _text(data.get("python_version"), "python_version", _PYTHON, "must be like 3.12")
    env = data.get("env") or {}
    if not isinstance(env, dict) or len(env) > 32:
        raise ProfileError("env: must be an object with at most 32 variables")
    out["env"] = {}
    for key, val in env.items():
        _text(key, f"env.{key}", _ENV_NAME, "variable names are upper case letters, digits and '_'")
        if key in RESERVED_ENV or key.startswith(("LD_", "RUNPOD_")):
            raise ProfileError(f"env.{key}: this variable is set by the application and cannot be overridden")
        if not isinstance(val, str) or len(val) > 512 or any(ord(c) < 32 or ord(c) == 127 for c in val):
            raise ProfileError(f"env.{key}: the value must be a single line of at most 512 characters")
        out["env"][key] = val
    return name, out


def kv_cache_dtype(profile: dict[str, Any], variant: dict[str, Any]) -> str:
    """'fp8' when the variant runs with an FP8 KV cache, else 'auto' (the model's dtype)."""
    for arg in [*profile.get("args", []), *variant.get("args", [])]:
        if arg["flag"] == "--kv-cache-dtype":
            return arg["value"].lower() or "auto"
    return "auto"


def is_fp8(profile: dict[str, Any], variant: dict[str, Any]) -> bool:
    """Whether the variant depends on 8-bit floating point (weights or KV cache)."""
    text = (variant["label"] + " " + variant["repo"]).lower()
    return "fp8" in text or kv_cache_dtype(profile, variant).startswith("fp8")


def serve_args(
    profile: dict[str, Any],
    variant: dict[str, Any],
    *,
    model_dir: str,
    context: int,
    tensor_parallel: int,
    port: int = 8000,
) -> list[str]:
    """Arguments after ``vllm``. The key is not here: vLLM reads VLLM_API_KEY from its environment.

    The application-owned options come last, so even if an earlier value were
    read as an option, the later occurrence would win.
    """
    if tensor_parallel not in TP_SIZES:
        raise ProfileError(f"tensor-parallel size {tensor_parallel} is not one of {TP_SIZES}")
    _int(context, "context", 1024, 4_194_304)
    args = ["serve", model_dir]
    for arg in [*profile.get("args", []), *variant.get("args", [])]:
        if arg["flag"] in OWNED_FLAGS or not _FLAG.match(arg["flag"]):
            raise ProfileError(f"argument {arg['flag']} is not allowed")
        args.append(arg["flag"])
        if arg.get("value"):
            args.append(arg["value"])
    util = profile.get("gpu_memory_utilization")
    if util is not None:
        args += ["--gpu-memory-utilization", f"{float(util):g}"]
    args += [
        "--served-model-name", profile["served_model_name"],
        "--max-model-len", str(context),
        "--tensor-parallel-size", str(tensor_parallel),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    return args


def slug(text: str) -> str:
    """Lower-case label part: 'RTX PRO 6000 Blackwell' -> 'rtx-pro-6000-blackwell'."""
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


_GPU_NOISE = re.compile(r"\b(nvidia|geforce|quadro|blackwell|generation|workstation edition|server edition)\b", re.I)


def gpu_slug(name: str) -> str:
    return slug(_GPU_NOISE.sub(" ", name)) or "gpu"


def model_slug(repo: str) -> str:
    """'Qwen/Qwen3.8-27B-FP8' -> 'qwen38-27b' (owner, dots and a trailing quantisation suffix removed)."""
    base = repo.split("/")[-1].lower()
    base = re.sub(r"[-_](fp8|bf16|awq|gptq|int8|int4|fp16)$", "", base)
    return slug(base.replace(".", ""))


def deployment_label(gpu_name: str, gpu_count: int, variant: dict[str, Any], context: int) -> str:
    """Same form as the reference reports, e.g. ``rtx-pro-6000x1-qwen38-27b-bf16-131072``."""
    return f"{gpu_slug(gpu_name)}x{gpu_count}-{model_slug(variant['repo'])}-{slug(variant['label'])}-{context}"


# -- storage ----------------------------------------------------------------------


class ProfileStore:
    """Profile rows in SQLite. Reads re-validate, so a hand-edited row cannot bypass the patterns."""

    def __init__(self, store: Any):
        self.store = store

    def seed(self) -> None:
        if not self.store.read_one("SELECT id FROM profiles LIMIT 1"):
            self.create(DEFAULT_PROFILE_NAME, default_profile())

    @staticmethod
    def _row(row: dict[str, Any]) -> dict[str, Any]:
        try:
            data = json.loads(row["data"])
        except ValueError:
            data = {}
        return {"id": row["id"], "name": row["name"], "data": data,
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def list(self) -> list[dict[str, Any]]:
        return [self._row(r) for r in self.store.read("SELECT * FROM profiles ORDER BY name COLLATE NOCASE")]

    def get(self, profile_id: int) -> dict[str, Any] | None:
        row = self.store.read_one("SELECT * FROM profiles WHERE id = ?", (profile_id,))
        return self._row(row) if row else None

    def get_valid(self, profile_id: int) -> dict[str, Any]:
        """The profile, validated again; raises ProfileError when missing or invalid."""
        row = self.get(profile_id)
        if row is None:
            raise ProfileError("no such profile")
        row["name"], row["data"] = validate(row["name"], row["data"])
        return row

    def _name_taken(self, name: str, except_id: int | None = None) -> bool:
        row = self.store.read_one("SELECT id FROM profiles WHERE name = ? COLLATE NOCASE", (name,))
        return bool(row) and row["id"] != except_id

    def create(self, name: Any, data: Any) -> dict[str, Any]:
        name, data = validate(name, data)
        if self._name_taken(name):
            raise ProfileError(f"name: a profile named '{name}' already exists")
        now = time.time()
        new_id = self.store.write(
            "INSERT INTO profiles (name, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, json.dumps(data), now, now),
        )
        return self.get(new_id) or {}

    def update(self, profile_id: int, name: Any, data: Any) -> dict[str, Any]:
        if self.get(profile_id) is None:
            raise ProfileError("no such profile")
        name, data = validate(name, data)
        if self._name_taken(name, profile_id):
            raise ProfileError(f"name: a profile named '{name}' already exists")
        self.store.write(
            "UPDATE profiles SET name = ?, data = ?, updated_at = ? WHERE id = ?",
            (name, json.dumps(data), time.time(), profile_id),
        )
        return self.get(profile_id) or {}

    def delete(self, profile_id: int) -> bool:
        return self.store.write_count("DELETE FROM profiles WHERE id = ?", (profile_id,)) > 0


def snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    """A detached copy stored with a deployment."""
    return copy.deepcopy({"id": profile["id"], "name": profile["name"], "data": profile["data"]})
