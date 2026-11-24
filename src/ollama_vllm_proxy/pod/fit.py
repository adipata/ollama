"""The fit plan: which variant of a profile runs on a set of GPUs, and with what context.

Nothing here is tied to a list of GPU models. The plan works from what the pod
reports (count, memory per GPU, compute capability) and from the model's
repository (weight size, ``config.json``). The same calculation runs for a
hypothetical configuration from the catalog, to help choose what to rent.

The numbers are estimates used to choose and to fail early; vLLM makes the
real decision at startup (see ``deploy``: restart with a smaller context).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from . import profiles as prof

log = logging.getLogger("ovp.pod.fit")

GIB = 1024**3
GB = 1000**3
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90  # vLLM's default when the profile does not set one
GPU_OVERHEAD_BYTES = int(1.5 * GIB)  # per GPU: CUDA context, activations, fragmentation (PLAN 16.11)
ENV_BYTES = 9 * GB  # the vLLM virtual environment (measured: 8.9 GB)
DISK_MARGIN_BYTES = 16 * GB  # caches, logs and headroom; with the environment: model + 25 GB
MIN_COMPUTE_CAP = 8.9  # Ada and newer; below it FP8 runs in an unverified mode
NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"]

GENERATION_COMPUTE_CAP = {"blackwell": 12.0, "hopper": 9.0, "ada": 8.9, "ampere": 8.6}

# Runpod hardware list, November 2026: (generation, name, GB per GPU, max per pod, MIG slice)
_CATALOG_SEED = [
    ("Blackwell", "B300", 288, 8, False), ("Blackwell", "B200", 180, 1, False),
    ("Blackwell", "RTX PRO 6000", 96, 1, False), ("Blackwell", "RTX PRO 6000 WK", 96, 8, False),
    ("Blackwell", "RTX 5090", 32, 8, False), ("Blackwell", "RTX PRO 4500", 32, 8, False),
    ("Blackwell", "RTX PRO 4500 SE", 32, 8, False), ("Blackwell", "RTX PRO 4000", 24, 1, False),
    ("Blackwell", "PRO 6000 MIG 48GB", 48, 1, True), ("Blackwell", "PRO 6000 MIG 24GB", 24, 1, True),
    ("Hopper", "H200 NVL", 143, 8, False), ("Hopper", "H200 SXM", 141, 2, False),
    ("Hopper", "H100 NVL", 94, 8, False), ("Hopper", "H100 SXM", 80, 2, False), ("Hopper", "H100 PCIe", 80, 8, False),
    ("Ada", "L40", 48, 8, False), ("Ada", "L40S", 48, 8, False), ("Ada", "RTX 6000 Ada", 48, 8, False),
    ("Ada", "RTX 4090", 24, 1, False), ("Ada", "L4", 24, 8, False), ("Ada", "RTX 4000 Ada", 20, 8, False),
    ("Ada", "RTX 2000 Ada", 16, 1, False),
]


def default_catalog() -> list[dict[str, Any]]:
    return [{"generation": g, "name": n, "memory_gb": m, "max_count": c, "mig": mig}
            for g, n, m, c, mig in _CATALOG_SEED]


class CatalogError(ValueError):
    pass


def validate_catalog(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 200:
        raise CatalogError("the catalog must be a list of 1 to 200 GPUs")
    out, seen = [], set()
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise CatalogError(f"entry {i + 1}: must be an object")
        name, gen = item.get("name"), item.get("generation", "")
        if not isinstance(name, str) or not re.match(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,47}$", name):
            raise CatalogError(f"entry {i + 1}: name must be 1-48 letters, digits, spaces, '.', '_', '+', '-'")
        if not isinstance(gen, str) or not re.match(r"^[A-Za-z0-9 ._-]{0,32}$", gen):
            raise CatalogError(f"entry {i + 1} ({name}): generation must be a short word")
        mem, count = item.get("memory_gb"), item.get("max_count", 1)
        if isinstance(mem, bool) or not isinstance(mem, (int, float)) or not 1 <= mem <= 4096:
            raise CatalogError(f"entry {i + 1} ({name}): memory_gb must be a number between 1 and 4096")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 16:
            raise CatalogError(f"entry {i + 1} ({name}): max_count must be between 1 and 16")
        if name.lower() in seen:
            raise CatalogError(f"entry {i + 1}: '{name}' is listed twice")
        seen.add(name.lower())
        out.append({"generation": gen, "name": name, "memory_gb": mem, "max_count": count,
                    "mig": bool(item.get("mig", False))})
    return out


# -- GPUs ---------------------------------------------------------------------------


@dataclass
class Gpu:
    index: int
    name: str
    memory_mib: int
    compute_cap: float | None = None
    mig: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_gpus(text: str) -> list[Gpu]:
    """Parse the CSV printed by NVIDIA_SMI_QUERY. Lines that do not parse are skipped."""
    gpus: list[Gpu] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        mem = re.match(r"^\d+", parts[2])
        if not mem:
            continue
        cap: float | None = None
        if len(parts) > 3:
            try:
                cap = float(parts[3])
            except ValueError:
                cap = None
        name = parts[1][:80]
        gpus.append(Gpu(int(parts[0]), name, int(mem.group(0)), cap, mig=bool(re.search(r"\bMIG\b", name, re.I))))
    return gpus


def hypothetical_gpus(entry: dict[str, Any], count: int) -> list[Gpu]:
    """GPUs as a pod of this catalog type would report them (memory slightly under the nominal size)."""
    cap = GENERATION_COMPUTE_CAP.get(str(entry.get("generation", "")).split(",")[0].strip().lower())
    mib = int(float(entry["memory_gb"]) * 1000)
    return [Gpu(i, entry["name"], mib, cap, bool(entry.get("mig"))) for i in range(count)]


def identical_group(gpus: list[Gpu]) -> list[Gpu]:
    """The largest group of identical GPUs (ties: the one with more memory)."""
    groups: dict[tuple[str, int], list[Gpu]] = {}
    for g in gpus:
        groups.setdefault((g.name, g.memory_mib), []).append(g)
    if not groups:
        return []
    return max(groups.values(), key=lambda grp: (len(grp), grp[0].memory_mib))


def auto_tensor_parallel(group: list[Gpu]) -> int:
    if not group or group[0].mig:
        return 1 if group else 0
    return max(s for s in prof.TP_SIZES if s <= len(group))


def describe_gpus(gpus: list[Gpu]) -> str:
    if not gpus:
        return "no GPU"
    parts = []
    seen: dict[tuple[str, int], int] = {}
    for g in gpus:
        seen[(g.name, g.memory_mib)] = seen.get((g.name, g.memory_mib), 0) + 1
    for (name, mib), n in seen.items():
        parts.append(f"{n}× {name} ({mib / 1024:.0f} GiB)")
    return " + ".join(parts)


# -- model repository ---------------------------------------------------------------


@dataclass
class RepoInfo:
    repo: str
    revision: str = ""
    weight_bytes: int = 0
    total_bytes: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    error_kind: str = ""  # "" | auth | notfound | network

    @property
    def ok(self) -> bool:
        return not self.error


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


def sizes_from_siblings(siblings: list[dict[str, Any]]) -> tuple[int, int]:
    """(weight bytes, total bytes). Safetensors win when a repo carries several formats."""
    total, by_suffix = 0, {s: 0 for s in _WEIGHT_SUFFIXES}
    for sib in siblings:
        name = str(sib.get("rfilename") or "")
        size = sib.get("size")
        if not isinstance(size, int):
            size = (sib.get("lfs") or {}).get("size") if isinstance(sib.get("lfs"), dict) else 0
        size = size if isinstance(size, int) else 0
        total += size
        for suffix in _WEIGHT_SUFFIXES:
            if name.endswith(suffix):
                by_suffix[suffix] += size
    weights = by_suffix[".safetensors"] or sum(by_suffix.values())
    return weights, total


class HuggingFace:
    """Repository facts from the Hugging Face API, fetched from the proxy host. Replaceable in tests."""

    def __init__(self, endpoint: str | None = None, ttl: float = 600.0, client: httpx.AsyncClient | None = None):
        self.endpoint = (endpoint or os.environ.get("POD_HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
        self.ttl = ttl
        self._cache: dict[tuple[str, str, bool], tuple[float, RepoInfo]] = {}
        self._client = client

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # Unlike the upstream client this one honours HTTPS_PROXY: huggingface.co is on the internet.
            self._client = httpx.AsyncClient(timeout=20.0, follow_redirects=True, trust_env=True)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def forget(self) -> None:
        self._cache.clear()

    async def fetch(self, repo: str, revision: str = "", token: str | None = None) -> RepoInfo:
        key = (repo, revision, bool(token))
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        info = await self._fetch(repo, revision, token)
        if info.ok:
            self._cache[key] = (time.monotonic(), info)
        return info

    async def _fetch(self, repo: str, revision: str, token: str | None) -> RepoInfo:
        info = RepoInfo(repo=repo, revision=revision)
        headers = {"authorization": f"Bearer {token}"} if token else {}
        rev_path = f"/revision/{revision}" if revision else ""
        hint = "" if token else " (no Hugging Face token is set)"
        try:
            meta = await self._http().get(f"{self.endpoint}/api/models/{repo}{rev_path}",
                                          params={"blobs": "true"}, headers=headers)
            if meta.status_code in (401, 403):
                info.error_kind = "auth"
                info.error = f"{repo}: access denied by Hugging Face{hint}; the repo is gated, private or missing"
                return info
            if meta.status_code == 404:
                info.error_kind, info.error = "notfound", f"{repo}: no such repository or revision on Hugging Face"
                return info
            if meta.status_code != 200:
                info.error_kind, info.error = "network", f"{repo}: Hugging Face answered {meta.status_code}"
                return info
            body = meta.json()
            siblings = body.get("siblings") if isinstance(body, dict) else None
            info.weight_bytes, info.total_bytes = sizes_from_siblings(siblings if isinstance(siblings, list) else [])
            cfg = await self._http().get(f"{self.endpoint}/{repo}/resolve/{revision or 'main'}/config.json",
                                         headers=headers)
            if cfg.status_code in (401, 403):
                info.error_kind = "auth"
                info.error = (f"{repo}: the files are gated and not readable{hint}; accept the model's terms on "
                              "Hugging Face and set a token with access on this page")
                return info
            if cfg.status_code == 200:
                parsed = cfg.json()
                info.config = parsed if isinstance(parsed, dict) else {}
        except (httpx.HTTPError, ValueError) as exc:
            info.error_kind = "network"
            info.error = f"{repo}: cannot read the repository from Hugging Face ({type(exc).__name__})"
        return info


_DTYPE_BYTES = {"float32": 4, "float": 4, "float16": 2, "half": 2, "bfloat16": 2, "float8": 1, "fp8": 1}


def kv_bytes_per_token(config: dict[str, Any], kv_dtype: str = "auto") -> int | None:
    """Bytes of KV cache one token occupies: 2 (K and V) × layers × KV heads × head dim × dtype size."""
    cfg = config
    if "num_hidden_layers" not in cfg and isinstance(cfg.get("text_config"), dict):
        cfg = {**config, **cfg["text_config"]}
    layers, heads = cfg.get("num_hidden_layers"), cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim")
    if not head_dim and isinstance(cfg.get("hidden_size"), int) and isinstance(heads, int) and heads:
        head_dim = cfg["hidden_size"] // heads
    if not all(isinstance(v, int) and v > 0 for v in (layers, kv_heads, head_dim)):
        return None
    if kv_dtype.lower().startswith("fp8"):
        size = 1
    else:
        size = _DTYPE_BYTES.get(str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16").lower(), 2)
    return 2 * layers * kv_heads * head_dim * size


# -- the plan -----------------------------------------------------------------------


def _gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


def disk_request_gb(download_bytes: int) -> int:
    """Container disk to ask Runpod for: model + 25 GB, rounded up to 5 GB."""
    return int(math.ceil((download_bytes + ENV_BYTES + DISK_MARGIN_BYTES) / (5 * GB)) * 5)


def evaluate_variant(
    profile: dict[str, Any],
    variant: dict[str, Any],
    info: RepoInfo | None,
    gpus: list[Gpu],
    *,
    tensor_parallel: int | None = None,
    context: int | None = None,
    disk_free_bytes: int | None = None,
    present_bytes: int = 0,
    env_installed: bool = False,
) -> dict[str, Any]:
    """One row of the plan table. ``tensor_parallel`` and ``context`` are the user's overrides."""
    group = identical_group(gpus)
    wanted, minimum = profile["context_wanted"], profile["context_min"]
    row: dict[str, Any] = {
        "variant": variant["label"], "repo": variant["repo"], "dir": variant["dir"],
        "weights_bytes": None, "download_bytes": None, "tensor_parallel": None, "gpu_indices": [],
        "gpu_name": group[0].name if group else "", "usable_bytes": None, "kv_bytes": None,
        "kv_bytes_per_token": None, "max_context": None, "context": context or wanted,
        "disk_needed_bytes": None, "disk_request_gb": None, "status": "no", "fits": False,
        "forceable": True, "reason": "", "warnings": [],
    }
    problems: list[str] = []  # blocking, but "deploy anyway" may override
    fatal: list[str] = []  # cannot be overridden

    if not group:
        fatal.append("the pod reports no NVIDIA GPU")
    auto_tp = auto_tensor_parallel(group)
    tp = tensor_parallel if tensor_parallel is not None else auto_tp
    if group:
        if tp not in prof.TP_SIZES:
            fatal.append(f"tensor-parallel size {tp} is not one vLLM accepts (1, 2, 4, 8)")
            tp = auto_tp
        elif tp > len(group):
            fatal.append(f"tensor-parallel size {tp} needs {tp} identical GPUs; the pod has {len(group)}")
            tp = auto_tp
        elif group[0].mig and tp != 1:
            fatal.append("a MIG slice is used as a single GPU only")
            tp = 1
    used = group[:tp]
    row["tensor_parallel"] = tp or None
    row["gpu_indices"] = [g.index for g in used]
    if len(gpus) > len(used) and used:
        row["warnings"].append(f"uses {len(used)} of {len(gpus)} GPUs (identical group / tensor-parallel size)")

    if used:
        cap = used[0].compute_cap
        if cap is None:
            row["warnings"].append("the GPU did not report its compute capability; generation not checked")
        elif cap < MIN_COMPUTE_CAP:
            text = (f"compute capability {cap:g} is below {MIN_COMPUTE_CAP}: this GPU generation lacks native "
                    "8-bit floating point and is not supported")
            if prof.is_fp8(profile, variant):
                fatal.append(text)
            else:
                problems.append(text + " (a BF16 variant can be deployed anyway)")

    util = profile.get("gpu_memory_utilization") or DEFAULT_GPU_MEMORY_UTILIZATION
    usable = sum(int(g.memory_mib * 1024 * 1024 * util) - GPU_OVERHEAD_BYTES for g in used)
    row["usable_bytes"] = max(usable, 0) if used else None

    if info is None or not info.ok:
        row["status"] = "unknown"
        row["reason"] = (info.error if info else "repository not read") or "repository not read"
        row["forceable"] = not fatal and not (info and info.error_kind in ("auth", "notfound"))
        if fatal:
            row["reason"] = "; ".join(fatal + [row["reason"]])
        return row

    row["weights_bytes"], row["download_bytes"] = info.weight_bytes, info.total_bytes
    row["disk_request_gb"] = disk_request_gb(info.total_bytes)
    row["disk_needed_bytes"] = (max(info.total_bytes - present_bytes, 0) + (0 if env_installed else ENV_BYTES)
                                + DISK_MARGIN_BYTES)
    per_token = kv_bytes_per_token(info.config, prof.kv_cache_dtype(profile, variant))
    row["kv_bytes_per_token"] = per_token

    if used:
        kv = usable - info.weight_bytes
        row["kv_bytes"] = kv
        if info.weight_bytes <= 0:
            row["warnings"].append("the repository lists no weight files; sizes are unknown")
        if kv <= 0:
            problems.append(f"the weights ({_gb(info.weight_bytes)}) do not fit in the usable GPU memory "
                            f"({_gb(max(usable, 0))} on {len(used)} GPU{'s' if len(used) != 1 else ''})")
            row["max_context"] = 0
        elif per_token:
            largest = int(kv // per_token)
            row["max_context"] = largest
            if context is None:
                chosen = min(wanted, largest)
                if chosen < wanted and chosen >= 1024:
                    chosen -= chosen % 1024
                row["context"] = chosen
                if chosen < minimum:
                    problems.append(f"the largest context that fits ({largest:,} tokens) is below the profile's "
                                    f"minimum ({minimum:,})")
            elif context > largest:
                problems.append(f"a context of {context:,} tokens needs more KV cache than is left "
                                f"(estimate: at most {largest:,})")
        else:
            row["warnings"].append("config.json does not describe the attention layout; the context is not checked")
        if used[0].mig:
            row["warnings"].append("MIG slice: untested with vLLM")

    if disk_free_bytes is not None and row["disk_needed_bytes"] > disk_free_bytes:
        problems.append(f"needs {_gb(row['disk_needed_bytes'])} of free disk, the pod has {_gb(disk_free_bytes)}: "
                        f"use a container disk of at least {row['disk_request_gb']} GB")

    if fatal or problems:
        row["status"], row["fits"] = "no", False
        row["forceable"] = not fatal
        row["reason"] = "; ".join(fatal + problems)
    else:
        row["fits"] = True
        row["status"] = "untested" if used and used[0].mig else "fits"
        ctx = row["context"]
        row["reason"] = (f"{_gb(info.weight_bytes)} of weights on {len(used)} GPU{'s' if len(used) != 1 else ''}, "
                         f"{_gb(row['kv_bytes'])} left for the KV cache")
        if context is None and ctx < wanted:
            row["reason"] += f"; context reduced from {wanted:,} to {ctx:,}"
    return row


def build_plan(
    profile: dict[str, Any],
    infos: dict[str, RepoInfo],
    gpus: list[Gpu],
    *,
    hypothetical: bool = False,
    disk_free_bytes: int | None = None,
    present: dict[str, int] | None = None,
    env_installed: bool = False,
) -> dict[str, Any]:
    """The plan table for all variants, with the automatic choice (first fitting variant)."""
    rows = [
        evaluate_variant(profile, v, infos.get(v["label"]), gpus, disk_free_bytes=disk_free_bytes,
                         present_bytes=(present or {}).get(v["label"], 0), env_installed=env_installed)
        for v in profile["variants"]
    ]
    auto = next((r["variant"] for r in rows if r["fits"]), None)
    group = identical_group(gpus)
    return {
        "hypothetical": hypothetical,
        "gpus": [g.as_dict() for g in gpus],
        "gpu_summary": describe_gpus(gpus),
        "gpu_name": group[0].name if group else "",
        "gpu_group_size": len(group),
        "tensor_parallel_options": [s for s in prof.TP_SIZES if s <= len(group)] if group and not group[0].mig
        else ([1] if group else []),
        "context_wanted": profile["context_wanted"],
        "context_min": profile["context_min"],
        "rows": rows,
        "auto": auto,
    }


def sizing_table(profile: dict[str, Any], infos: dict[str, RepoInfo], catalog: list[dict[str, Any]]) -> list[dict]:
    """What each catalog configuration would get: variant, tensor-parallel size, context, disk to request."""
    out = []
    for entry in catalog:
        counts = [1] if entry.get("mig") else [c for c in prof.TP_SIZES if c <= int(entry.get("max_count", 1))]
        for count in counts:
            plan = build_plan(profile, infos, hypothetical_gpus(entry, count), hypothetical=True)
            row = next((r for r in plan["rows"] if r["variant"] == plan["auto"]), None)
            item = {"gpu": entry["name"], "generation": entry.get("generation", ""), "count": count,
                    "memory_gb": entry["memory_gb"], "variant": None, "tensor_parallel": None, "context": None,
                    "disk_request_gb": None, "status": "no", "reason": ""}
            if row:
                item.update(variant=row["variant"], tensor_parallel=row["tensor_parallel"], context=row["context"],
                            disk_request_gb=row["disk_request_gb"], status=row["status"], reason=row["reason"])
            else:
                first = plan["rows"][0] if plan["rows"] else None
                item["status"] = "unknown" if first and first["status"] == "unknown" else "no"
                item["reason"] = first["reason"] if first else "no variant"
            out.append(item)
    return out


def dumps(plan: dict[str, Any]) -> str:
    return json.dumps(plan, separators=(",", ":"))
