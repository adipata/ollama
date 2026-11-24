#!/usr/bin/env python3
"""GPU and vLLM-startup exporter for the Runpod Pod (standard library only).

Runs next to vLLM and listens on 127.0.0.1 only; the local tunnel container
forwards it to the measurement proxy, which records the history.

    python3 gpu_exporter.py --vllm-log /workspace/vllm.log          # :9400

  GET /metrics    Prometheus text, one series per GPU (label gpu="<index>"):
                  utilization, memory, power, temperatures, clocks, PCIe link
                  and PCIe RX/TX throughput. Sampled every --interval seconds.
  GET /inventory  JSON snapshot: GPU identities and limits, ECC, driver/CUDA,
                  `nvidia-smi topo -m`, NVLink state, host CPU/RAM, and the
                  relevant vLLM startup lines (KV cache size, max concurrency,
                  attention backend, non-default args with secrets redacted).
  GET /healthz    {"ok": true}

Both nvidia-smi samplers are long-running child processes (-lms / dmon), so a
scrape never waits on nvidia-smi.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "1"

# Live per-GPU fields: (nvidia-smi field, metric name, help)
LIVE_FIELDS = [
    ("utilization.gpu", "gpu:utilization_percent", "GPU (SM) utilization"),
    ("utilization.memory", "gpu:memory_bandwidth_percent", "Memory controller utilization"),
    ("memory.used", "gpu:memory_used_mib", "Framebuffer memory used (MiB)"),
    ("memory.total", "gpu:memory_total_mib", "Framebuffer memory total (MiB)"),
    ("power.draw", "gpu:power_watts", "Power draw (W)"),
    ("power.limit", "gpu:power_limit_watts", "Enforced power limit (W)"),
    ("temperature.gpu", "gpu:temperature_celsius", "GPU temperature"),
    ("temperature.memory", "gpu:memory_temperature_celsius", "Memory temperature"),
    ("clocks.sm", "gpu:sm_clock_mhz", "SM clock (MHz)"),
    ("clocks.mem", "gpu:memory_clock_mhz", "Memory clock (MHz)"),
    ("pcie.link.gen.current", "gpu:pcie_link_gen", "Current PCIe generation"),
    ("pcie.link.width.current", "gpu:pcie_link_width", "Current PCIe lane width"),
]
STATIC_FIELDS = [
    "index", "name", "uuid", "pci.bus_id", "driver_version", "vbios_version", "memory.total",
    "power.limit", "power.max_limit", "clocks.max.sm", "clocks.max.memory", "pcie.link.gen.max",
    "pcie.link.width.max", "ecc.mode.current", "compute_cap", "persistence_mode",
]
LOG_PATTERNS = [
    ("non_default_args", re.compile(r"non-default args")),
    ("kv_cache_size", re.compile(r"GPU KV cache size")),
    ("available_kv_memory", re.compile(r"Available KV cache memory")),
    ("model_loading", re.compile(r"Model loading took")),
    ("attention_backend", re.compile(r"attention backend", re.I)),
    ("init_engine", re.compile(r"init engine")),
    ("memory_utilization_note", re.compile(r"gpu-memory-utilization=")),
    ("nccl", re.compile(r"NCCL version|custom allreduce|P2P", re.I)),
]
_REDACT = [
    (re.compile(r"('api_key':\s*)'[^']*'"), r"\1'***'"),
    (re.compile(r"(--api-key[ =])\S+"), r"\1***"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "hf_***"),
    (re.compile(r"(Bearer\s+)\S+", re.I), r"\1***"),
]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_INVALID_FIELD = re.compile(r'"([^"]+)" is not a valid field')


def redact(line: str) -> str:
    for pattern, repl in _REDACT:
        line = pattern.sub(repl, line)
    return line


def to_number(text: str) -> float | None:
    text = text.strip()
    if not text or text.startswith("[") or text in ("N/A", "-"):
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        return None


def parse_dmon(lines, state: dict[str, dict[str, float]]) -> None:
    """Fold `nvidia-smi dmon -s t` output (header "# gpu rxpci txpci") into state."""
    columns: list[str] | None = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            names = line.lstrip("#").split()
            # dmon prints a name row ("# gpu rxpci txpci") and a unit row ("# Idx MB/s MB/s")
            if names and names[0].lower() == "gpu":
                columns = [n.lower() for n in names]
            continue
        if not columns:
            continue
        parts = line.split()
        if len(parts) != len(columns):
            continue
        row = dict(zip(columns, parts))
        gpu = row.get("gpu")
        if gpu is None or not gpu.isdigit():
            continue
        cur = state.setdefault(gpu, {})
        for src, dst in (("rxpci", "rx"), ("txpci", "tx")):
            v = to_number(row.get(src, ""))
            if v is not None:
                cur[dst] = v
        cur["ts"] = time.time()


def parse_topology(text: str) -> dict:
    """Pull the GPU-to-GPU link types out of `nvidia-smi topo -m`."""
    text = _ANSI.sub("", text)
    rows = [l.split("\t") for l in text.splitlines() if l.strip()]
    if not rows:
        return {"links": [], "cpu_affinity": {}}
    header = [c.strip() for c in rows[0]]
    if header and header[0] == "":  # the header row starts with a tab
        header = header[1:]
    gpus = [c for c in header if re.fullmatch(r"GPU\d+", c)]
    links, affinity = [], {}
    for row in rows[1:]:
        cells = [c.strip() for c in row]
        if not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        a = cells[0]
        values = dict(zip(header, cells[1:]))
        for b in gpus:
            if b > a and b in values:
                links.append({"a": a, "b": b, "link": values[b]})
        if "CPU Affinity" in values:
            affinity[a] = {"cpus": values.get("CPU Affinity"), "numa": values.get("NUMA Affinity")}
    return {"links": links, "cpu_affinity": affinity}


class LogScanner:
    """Incrementally scans the vLLM log for startup lines; a new startup resets them."""

    def __init__(self, path: str | None):
        self.path = path
        self.offset = 0
        self.lines: dict[str, list[str]] = {}
        self.lock = threading.Lock()

    def scan(self) -> dict[str, list[str]]:
        with self.lock:
            if not self.path or not os.path.isfile(self.path):
                return {}
            size = os.path.getsize(self.path)
            if size < self.offset:  # truncated: vLLM restarted with ">"
                self.offset, self.lines = 0, {}
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read(64 * 1024 * 1024)
                self.offset += len(chunk)
            for raw in chunk.decode("utf-8", "replace").splitlines():
                for key, pattern in LOG_PATTERNS:
                    if pattern.search(raw):
                        if key == "non_default_args":
                            self.lines = {}  # a new vLLM startup begins
                        bucket = self.lines.setdefault(key, [])
                        line = redact(_ANSI.sub("", raw.strip()))[:1500]
                        if line not in bucket and len(bucket) < 20:
                            bucket.append(line)
                        break
            return {k: list(v) for k, v in self.lines.items()}


def run_text(cmd: list[str], timeout: float = 15) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "") + (out.stderr if out.returncode else "")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error: {type(exc).__name__}: {exc}"


def query_fields(fields: list[str], extra: list[str] | None = None) -> tuple[list[str], subprocess.CompletedProcess | None]:
    """Drop fields this driver rejects, then return (fields, result) of one query."""
    fields = list(fields)
    for _ in range(len(fields)):
        cmd = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"] + (extra or [])
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return fields, None
        if res.returncode == 0:
            return fields, res
        bad = _INVALID_FIELD.search(res.stdout + res.stderr)
        if not bad or bad.group(1) not in fields:
            return fields, res
        fields.remove(bad.group(1))
    return fields, None


class Sampler:
    def __init__(self, interval: float, log_path: str | None):
        self.interval = interval
        self.live: dict[str, dict[str, float]] = {}
        self.pcie: dict[str, dict[str, float]] = {}
        self.lock = threading.Lock()
        self.errors: dict[str, str] = {}
        self.logs = LogScanner(log_path)
        self._inventory: dict | None = None
        self._inventory_at = 0.0
        self.fields = ["index"] + [f for f, _, _ in LIVE_FIELDS]

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            self.errors["nvidia-smi"] = "not found on PATH"
            return
        self.fields, _ = query_fields(self.fields)
        threading.Thread(target=self._loop_query, name="query", daemon=True).start()
        threading.Thread(target=self._loop_dmon, name="dmon", daemon=True).start()

    def _loop_query(self) -> None:
        ms = max(200, int(self.interval * 1000))
        while True:
            cmd = ["nvidia-smi", f"--query-gpu={','.join(self.fields)}", "--format=csv,noheader,nounits", f"-lms={ms}"]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in proc.stdout:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != len(self.fields):
                        self.errors["query"] = line.strip()[:300]
                        continue
                    row = dict(zip(self.fields, parts))
                    vals = {f: to_number(row[f]) for f in self.fields[1:]}
                    vals["ts"] = time.time()
                    with self.lock:
                        self.live[row["index"]] = vals
                    self.errors.pop("query", None)
                proc.wait()
            except OSError as exc:
                self.errors["query"] = str(exc)
            time.sleep(5)

    def _loop_dmon(self) -> None:
        delay = max(1, int(round(self.interval)))
        while True:
            try:
                proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "t", "-d", str(delay)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                header: list[str] = []
                for line in proc.stdout:
                    if line.lstrip().startswith("#"):
                        if line.lstrip("# \t").lower().startswith("gpu"):
                            header = [line]
                        continue
                    with self.lock:
                        parse_dmon(header + [line], self.pcie)
                proc.wait()
            except OSError as exc:
                self.errors["dmon"] = str(exc)
            time.sleep(5)

    def metrics_text(self) -> str:
        now = time.time()
        stale = max(10.0, self.interval * 5)
        with self.lock:
            live = {g: dict(v) for g, v in self.live.items() if now - v["ts"] < stale}
            pcie = {g: dict(v) for g, v in self.pcie.items() if now - v.get("ts", 0) < stale}
        out = []
        for field, name, help_text in LIVE_FIELDS:
            rows = [(g, v.get(field)) for g, v in sorted(live.items(), key=lambda x: int(x[0]))]
            rows = [(g, val) for g, val in rows if val is not None]
            if rows:
                out.append(f"# HELP {name} {help_text}\n# TYPE {name} gauge")
                out += [f'{name}{{gpu="{g}"}} {val}' for g, val in rows]
        for key, name, help_text in (("rx", "gpu:pcie_rx_mb_per_second", "PCIe receive throughput (MB/s)"),
                                     ("tx", "gpu:pcie_tx_mb_per_second", "PCIe transmit throughput (MB/s)")):
            rows = [(g, v[key]) for g, v in sorted(pcie.items(), key=lambda x: int(x[0])) if key in v]
            if rows:
                out.append(f"# HELP {name} {help_text}\n# TYPE {name} gauge")
                out += [f'{name}{{gpu="{g}"}} {val}' for g, val in rows]
        out.append("# TYPE gpu:exporter_up gauge\ngpu:exporter_up 1")
        return "\n".join(out) + "\n"

    def inventory(self) -> dict:
        now = time.time()
        if self._inventory is None or now - self._inventory_at > 60:
            self._inventory = self._collect_static()
            self._inventory_at = now
        inv = dict(self._inventory)
        inv["vllm_log"] = self.logs.scan()
        inv["vllm_log_path"] = self.logs.path
        inv["errors"] = dict(self.errors)
        inv["generated_at"] = now
        return inv

    def _collect_static(self) -> dict:
        gpus = []
        fields, res = query_fields(STATIC_FIELDS)
        if res is not None and res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == len(fields):
                    gpus.append(dict(zip(fields, parts)))
        smi = run_text(["nvidia-smi"])
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi)
        topo = _ANSI.sub("", run_text(["nvidia-smi", "topo", "-m"]))
        nvlink = run_text(["nvidia-smi", "nvlink", "-s"])
        return {
            "exporter_version": VERSION,
            "gpus": gpus,
            "cuda_version": cuda.group(1) if cuda else None,
            "topology_text": topo.strip(),
            "topology": parse_topology(topo),
            "nvlink_text": nvlink.strip()[:4000],
            "nvlink_active": bool(re.search(r"Link \d+:\s*[\d.]+ GB/s", nvlink)),
            "host": host_info(),
        }


def host_info() -> dict:
    info = {"kernel": platform.release(), "logical_cpus": os.cpu_count(), "python": platform.python_version()}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            m = re.search(r"^model name\s*:\s*(.+)$", fh.read(), re.M)
            info["cpu_model"] = m.group(1).strip() if m else None
    except OSError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            m = re.search(r"^MemTotal:\s*(\d+) kB", fh.read(), re.M)
            info["memory_gib"] = round(int(m.group(1)) / 1024 / 1024, 1) if m else None
    except OSError:
        pass
    return info


def make_handler(sampler: Sampler):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gpu-exporter/" + VERSION

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/metrics":
                self._send(200, sampler.metrics_text().encode(), "text/plain; version=0.0.4")
            elif path == "/inventory":
                self._send(200, json.dumps(sampler.inventory()).encode(), "application/json")
            elif path == "/healthz":
                self._send(200, b'{"ok": true}', "application/json")
            else:
                self._send(404, b'{"error": "not found"}', "application/json")

        def log_message(self, fmt, *args) -> None:  # no access log
            pass

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; keep it local)")
    ap.add_argument("--port", type=int, default=9400)
    ap.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds")
    ap.add_argument("--vllm-log", default="/workspace/vllm.log", help="vLLM log file to scan for startup lines")
    args = ap.parse_args(argv)
    sampler = Sampler(args.interval, args.vllm_log)
    sampler.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sampler))
    print(f"gpu_exporter listening on http://{args.host}:{args.port} (log: {args.vllm_log})", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
