"""Monitoring: metrics parsing, the sampler, the live snapshot, the report and its routes."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import pytest
from conftest import Env
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from ollama_vllm_proxy import export
from ollama_vllm_proxy.__main__ import main as cli_main
from ollama_vllm_proxy.live import MAX_POINTS, Live, downsample, pct, phase
from ollama_vllm_proxy.observe import RequestRecord
from ollama_vllm_proxy.routes.monitor import monitor_of
from ollama_vllm_proxy.sampler import parse_metrics, split_info

FIXTURES = Path(__file__).parent / "fixtures"
LABEL = "rtx-pro-6000x2-qwen38-27b-bf16-131072"
CHAT = {"model": "dev-test", "messages": [{"role": "user", "content": "hi"}]}


# -- a stand-in for pod/gpu_exporter.py -------------------------------------------


class FakeGpu:
    def __init__(self):
        self.metrics = (FIXTURES / "gpu_metrics.txt").read_text()
        self.up = True
        self.calls = 0
        self.inventory = {
            "exporter_version": "1",
            "gpus": [
                {"index": str(i), "name": "NVIDIA RTX PRO 6000 Blackwell", "pci.bus_id": f"00000000:0{i + 1}:00.0",
                 "driver_version": "580.95.05", "vbios_version": "98.02.2E.00.03", "memory.total": "97887",
                 "power.limit": "600.00", "clocks.max.sm": "2430", "pcie.link.gen.max": "5",
                 "pcie.link.width.max": "16", "ecc.mode.current": "Enabled"}
                for i in range(2)
            ],
            "cuda_version": "13.0",
            "topology_text": "\tGPU0\tGPU1\tCPU Affinity\nGPU0\t X \tNODE\t0-31\nGPU1\tNODE\t X \t0-31",
            "topology": {"links": [{"a": "GPU0", "b": "GPU1", "link": "NODE"}], "cpu_affinity": {}},
            "nvlink_active": False,
            "host": {"cpu_model": "AMD EPYC 9355", "logical_cpus": 64, "memory_gib": 503.5},
            "vllm_log": {"kv_cache_size": ["INFO GPU KV cache size: 262,144 tokens"],
                         "non_default_args": ["INFO non-default args: {'api_key': '***', 'port': 8000}"]},
            "vllm_log_path": "/workspace/vllm.log",
            "errors": {},
        }

    def app(self) -> Starlette:
        async def metrics(request):
            self.calls += 1
            return PlainTextResponse(self.metrics) if self.up else PlainTextResponse("down", status_code=503)

        async def inventory(request):
            return JSONResponse({**self.inventory, "generated_at": time.time()})

        return Starlette(routes=[Route("/metrics", metrics), Route("/inventory", inventory)])


class MonEnv(Env):
    """The proxy with a fake vLLM and a fake GPU helper reachable through one shared client."""

    def __init__(self, tmp_path, **kw):
        super().__init__(tmp_path, **kw)
        self.gpu = FakeGpu()
        self._old_client = self.ctx.client
        self.ctx.client = httpx.AsyncClient(mounts={
            "all://gpu.test": httpx.ASGITransport(app=self.gpu.app()),
            "all://vllm.test": httpx.ASGITransport(app=self.fake_app),
        })
        self.sampler = monitor_of(self.ctx).sampler

    def with_gpu(self) -> None:
        self.ctx.backend.gpu_url = "http://gpu.test"
        self.ctx.backend.label = LABEL

    def samples(self, where: str = "1=1") -> list[dict]:
        self.ctx.store.flush()
        return self.ctx.store.read(f"SELECT * FROM engine_samples WHERE {where} ORDER BY ts, name, labels")

    async def aclose(self):
        await self._old_client.aclose()
        await super().aclose()


@pytest.fixture
async def mon(tmp_path):
    e = MonEnv(tmp_path)
    yield e
    await e.aclose()


# -- metrics parsing ---------------------------------------------------------------


def test_parse_vllm_metrics_fixture():
    parsed = parse_metrics((FIXTURES / "vllm_metrics.txt").read_text())
    names = [n for n, _, _ in parsed]
    assert all(n.startswith("vllm:") for n in names)  # other families are ignored
    assert not any(n.endswith(("_bucket", "_created")) for n in names)  # histograms buckets are dropped
    assert "vllm:time_to_first_token_seconds_sum" in names and "vllm:no_labels_total" in names
    assert "vllm:lora_requests_info" not in names  # NaN is not stored
    assert all("model_name" not in labels and "/workspace" not in labels for _, labels, _ in parsed)
    by = {(n, labels): v for n, labels, v in parsed}
    assert by[("vllm:num_requests_running", '{"engine": "0"}')] == 3.0
    assert by[("vllm:prompt_tokens_total", '{"engine": "0"}')] == 1234500.0
    assert by[("vllm:generation_tokens_total", '{"engine": "0"}')] == 98765.0  # trailing timestamp ignored
    assert by[("vllm:request_success_total", '{"engine": "0", "finished_reason": "length"}')] == 2.0
    assert by[("vllm:no_labels_total", "{}")] == 7.0

    series, info = split_info(parsed)
    assert not any(n.endswith("_info") for n, _, _ in series)
    cache = info["vllm:cache_config_info"][0]
    assert cache["kv_cache_size_tokens"] == "262144" and cache["cache_dtype"] == "auto"
    assert cache["note"] == 'a "quoted" value, with a comma and a \\ backslash'


def test_parse_gpu_metrics_fixture():
    parsed = parse_metrics((FIXTURES / "gpu_metrics.txt").read_text(), "gpu:")
    by = {(n, labels): v for n, labels, v in parsed}
    assert by[("gpu:utilization_percent", '{"gpu": "1"}')] == 91.0
    assert by[("gpu:power_watts", '{"gpu": "0"}')] == 412.5
    assert ("gpu:exporter_up", "{}") in by  # the sampler, not the parser, leaves this one out


# -- sampler -------------------------------------------------------------------------


async def test_sampler_writes_engine_and_gpu_series(mon):
    mon.with_gpu()
    await mon.sampler.sample_once()
    rows = mon.samples()
    names = {r["name"] for r in rows}
    assert {"proxy:open_requests", "vllm:num_requests_running", "vllm:kv_cache_usage_perc",
            "vllm:generation_tokens_total", "gpu:utilization_percent", "gpu:pcie_tx_mb_per_second"} <= names
    assert "gpu:exporter_up" not in names and "vllm:cache_config_info" not in names
    assert not any(n.endswith("_bucket") for n in names)
    assert all("model_name" not in r["labels"] for r in rows)
    assert {r["deployment_id"] for r in rows} == {LABEL}
    assert len({r["ts"] for r in rows}) == 1  # one scrape, one timestamp
    success = [r for r in rows if r["name"] == "vllm:request_success_total"]
    assert json.loads(success[0]["labels"]) == {"finished_reason": "stop"}
    gpu1 = [r for r in rows if r["name"] == "gpu:utilization_percent" and r["labels"] == '{"gpu": "1"}']
    assert gpu1[0]["value"] == 91.0
    # the upstream key is sent to vLLM, and the cache config went to the inventory instead
    assert mon.ctx.live.metrics_ok is True and mon.ctx.live.gpu_ok is True
    inv = mon.rows("SELECT kind, data FROM inventory")
    assert [r["kind"] for r in inv] == ["vllm:cache_config_info"]
    assert json.loads(inv[0]["data"])[0]["cache_dtype"] == "fp8"

    point = mon.ctx.live.series[-1]
    assert point["running"] == 1.0 and point["kv"] == 0.12 and point["open"] == 0
    assert point["g0_util"] == 87.0 and point["g1_mem_used"] == 90624.0
    assert point["output_tps"] is None  # a rate needs two samples


async def test_sampler_rates_between_two_samples(mon):
    await mon.sampler.sample_once()
    for _ in range(3):
        assert (await mon.http.post("/v1/chat/completions", json=CHAT)).status_code == 200
    await asyncio.sleep(0.3)
    await mon.sampler.sample_once()
    a, b = list(mon.ctx.live.series)[-2:]
    dt = b["t"] - a["t"]
    assert b["output_tps"] == pytest.approx(60 / dt, rel=0.05)  # the fake adds 20 tokens per request
    assert b["prompt_tps"] == pytest.approx(150 / dt, rel=0.05)
    assert b["prefix_hit"] == 0.75 and b["preemptions"] == 0


async def test_sampler_survives_failing_sources(mon):
    mon.with_gpu()
    mon.gpu.up = False
    mon.ctx.backend.api_key = "wrong"  # the fake /metrics is open, so break the URL instead
    mon.ctx.backend.base_url = "http://vllm.test/nowhere"
    await mon.sampler.sample_once()
    live = mon.ctx.live
    assert live.metrics_ok is False and live.metrics_error == "http_404" and live.gpu_ok is False
    assert mon.samples() == []  # nothing scraped, nothing written
    assert live.series[-1]["running"] is None and live.series[-1]["open"] == 0


async def test_sampler_without_backend_writes_nothing(managed):
    sampler = monitor_of(managed.ctx).sampler
    await sampler.sample_once()
    await sampler.inventory_once()
    managed.ctx.store.flush()
    assert managed.ctx.store.read("SELECT COUNT(*) AS n FROM engine_samples")[0]["n"] == 0
    assert managed.ctx.live.metrics_ok is None and managed.ctx.live.gpu_ok is None
    snap = managed.ctx.live.snapshot(300)
    assert snap["backend"]["state"] == "no_pod" and snap["gpus"] == [] and snap["hardware"] == {}
    assert snap["totals"]["completed"] == 0 and snap["in_flight"] == []


async def test_gpu_helper_is_scraped_before_vllm_is_ready(managed):
    gpu = FakeGpu()
    await managed.ctx.client.aclose()
    managed.ctx.client = httpx.AsyncClient(mounts={"all://gpu.test": httpx.ASGITransport(app=gpu.app())})
    managed.ctx.backend.set_state("deploying")
    managed.ctx.backend.set_endpoints(gpu_url="http://gpu.test")
    sampler = monitor_of(managed.ctx).sampler
    await sampler.sample_once()
    await sampler.inventory_once()
    managed.ctx.store.flush()
    names = {r["name"] for r in managed.ctx.store.read("SELECT DISTINCT name FROM engine_samples")}
    assert "gpu:utilization_percent" in names and "proxy:open_requests" in names
    assert not any(n.startswith("vllm:") for n in names)
    assert managed.ctx.live.hardware["gpu_inventory"]["cuda_version"] == "13.0"
    assert managed.ctx.live.current()["gpus"][1]["util"] == 91.0


async def test_inventory_is_stored_only_when_it_changes(mon):
    mon.with_gpu()
    for _ in range(3):
        await mon.sampler.inventory_once()
        await mon.sampler.sample_once()
    rows = mon.rows("SELECT kind, deployment_id, data FROM inventory ORDER BY id")
    assert sorted(r["kind"] for r in rows) == ["gpu_inventory", "vllm:cache_config_info", "vllm_version"]
    assert {r["deployment_id"] for r in rows} == {LABEL}
    stored = {r["kind"]: json.loads(r["data"]) for r in rows}
    assert stored["vllm_version"] == {"version": "0.34.0"}
    assert "generated_at" not in stored["gpu_inventory"]
    assert stored["gpu_inventory"]["gpus"][0]["name"] == "NVIDIA RTX PRO 6000 Blackwell"

    mon.gpu.inventory["vllm_log"]["model_loading"] = ["INFO Model loading took 51.2 GiB"]
    await mon.sampler.inventory_once()
    await mon.sampler.inventory_once()
    kinds = [r["kind"] for r in mon.rows("SELECT kind FROM inventory ORDER BY id")]
    assert kinds.count("gpu_inventory") == 2 and kinds.count("vllm_version") == 1
    assert "model_loading" in mon.ctx.live.hardware["gpu_inventory"]["vllm_log"]

    # A restarted proxy reads what is stored and does not write the same content again.
    fresh = type(mon.sampler)(mon.ctx)
    fresh.load_stored()
    await fresh.inventory_once()
    assert len(mon.rows("SELECT id FROM inventory")) == len(kinds)


# -- live snapshot -------------------------------------------------------------------


def _rec(rid: str, **kw) -> RequestRecord:
    base = dict(request_id=rid, actor_id="laptop", key_id=1, api="ollama", stream=True, messages_count=2,
                t_arrival_wall=time.time(), t0=time.monotonic())
    base.update(kw)
    return RequestRecord(**base)


def test_phase_follows_the_timestamps():
    assert phase(_rec("a")) == "queued/prefill"
    assert phase(_rec("a", first_byte_ms=5.0)) == "queued/prefill"
    assert phase(_rec("a", first_payload_ms=9.0, first_reasoning_ms=9.0)) == "reasoning"
    assert phase(_rec("a", first_payload_ms=9.0, first_reasoning_ms=9.0, first_text_ms=40.0)) == "answering"
    assert phase(_rec("a", first_payload_ms=9.0, first_text_ms=40.0, first_tool_ms=50.0)) == "tool call"
    assert phase(_rec("a", stream=False)) == "generating (not streamed)"


async def test_live_snapshot_in_flight_and_finished(mon):
    live = mon.ctx.live
    thinking = _rec("r1", session_id="s-1", first_payload_ms=120.0, first_reasoning_ms=120.0, content_events=4,
                    requested_max_tokens=512, t0=time.monotonic() - 2.0)
    queued = _rec("r2", actor_id="ci", api="openai")
    live.on_request_start(thinking)
    live.on_request_start(queued)
    mon.ctx.open_requests = 2
    snap = live.snapshot(300)
    assert [f["phase"] for f in snap["in_flight"]] == ["reasoning", "queued/prefill"]  # longest running first
    first = snap["in_flight"][0]
    assert first["key"] == "laptop" and first["session"] == "s-1" and first["elapsed_ms"] >= 2000
    assert first["first_ms"] == 120.0 and first["max_tokens"] == 512 and first["content_events"] == 4
    assert snap["current"]["open"] == 2 and snap["totals"]["completed"] == 0

    thinking.outcome, thinking.end_ms, thinking.completion_tokens = "ok", 900.0, 30
    thinking.prompt_tokens, thinking.cached_tokens, thinking.first_text_ms = 100, 80, 300.0
    live.on_request_done(thinking)
    queued.outcome = "upstream_error"
    live.on_request_done(queued)
    snap = live.snapshot(300)
    assert snap["in_flight"] == [] and snap["source"] == "memory"
    assert snap["totals"]["completed"] == 2 and snap["totals"]["ok"] == 1 and snap["totals"]["errors"] == 1
    keys = {k["key"]: k for k in snap["keys"]}
    assert keys["laptop"]["first_p50"] == 120.0 and keys["laptop"]["cached_ratio"] == 0.8
    assert keys["ci"]["errors"] == 1 and keys["ci"]["in_flight"] == 0
    assert snap["requests"][0]["session_id"] == "s-1" and snap["requests"][0]["stream"] == 1
    json.dumps(snap)  # everything in it is JSON-serialisable
    # nothing that could carry prompt or completion text
    assert not {"prompt", "content", "messages"} & set(snap["requests"][0])


async def test_live_snapshot_after_real_requests(mon):
    mon.with_gpu()
    await mon.sampler.sample_once()
    await mon.http.post("/api/chat", json={**CHAT, "stream": False}, headers={"X-Session-Id": "sess-a"})
    r = await mon.http.post("/v1/chat/completions", json={**CHAT, "messages": [{"role": "user", "content": "overflow"}]})
    assert r.status_code == 400
    await mon.sampler.sample_once()
    snap = mon.ctx.live.snapshot(120)
    assert snap["totals"]["completed"] == 2 and snap["totals"]["errors"] == 1
    assert snap["keys"][0]["key"] == "tests" and snap["keys"][0]["n"] == 2
    assert [r["api"] for r in snap["requests"]] == ["ollama", "openai"]
    assert snap["requests"][0]["session_id"] == "sess-a"
    assert len(snap["series"]) == 2 and snap["current"]["running"] == 1.0
    assert [g["gpu"] for g in snap["gpus"]] == [0, 1]
    assert snap["gpus"][0]["util_mean"] == 87.0 and snap["gpus"][1]["power_max"] == 398.25
    assert snap["gpus"][0]["samples"] == 2 and snap["gpus"][0]["sm_clock_min"] == 2400.0


def test_ring_is_bounded_and_downsampled():
    live = Live()
    now = time.time()
    for i in range(3000):
        live.on_scrape(now - 3000 + i, [("vllm:num_requests_running", "{}", float(i % 7)),
                                         ("vllm:generation_tokens_total", "{}", 10.0 * i)], 0)
    assert len(live.series) == live.series.maxlen == 1000  # 25 minutes at 1.5 s
    snap = live.snapshot(900)
    assert 0 < len(snap["series"]) <= MAX_POINTS
    assert max(p["running"] for p in snap["series"]) == 6.0  # peaks survive the merge
    rates = [p["output_tps"] for p in snap["series"] if p["output_tps"] is not None]
    assert rates and all(r == pytest.approx(10.0) for r in rates)
    assert pct([1, 2, 3, 4, None], 50) == 2 and pct([], 90) is None
    merged = downsample([{"t": float(i), "preemptions": 1, "open": i} for i in range(100)], 10)
    assert len(merged) == 10 and sum(p["preemptions"] for p in merged) == 100 and merged[-1]["open"] == 99


def test_counter_reset_and_gaps_give_no_rate():
    live = Live()
    t = time.time() - 100
    live.on_scrape(t, [("vllm:generation_tokens_total", "{}", 500.0)], 0)
    live.on_scrape(t + 1.5, [("vllm:generation_tokens_total", "{}", 20.0)], 0)  # vLLM restarted
    live.on_scrape(t + 3.0, [("vllm:generation_tokens_total", "{}", 50.0)], 0)
    live.on_scrape(t + 4.5, None, 0)  # scrape failed
    live.on_scrape(t + 6.0, [("vllm:generation_tokens_total", "{}", 5000.0)], 0)
    assert [p["output_tps"] for p in live.series] == [None, None, 20.0, None, None]


# -- report --------------------------------------------------------------------------

T0 = 1_780_000_000.0  # a fixed instant so the seeded data is reproducible


def seed(store, t0: float = T0) -> None:
    """Ten minutes of engine and GPU samples every 2 s, and 6 requests from two keys."""
    rows = []
    for i in range(300):
        ts = t0 + 2.0 * i
        rows += [
            (ts, LABEL, "proxy:open_requests", "{}", float(i % 3)),
            (ts, LABEL, "vllm:num_requests_running", "{}", 2.0 if i < 150 else 4.0),
            (ts, LABEL, "vllm:num_requests_waiting", "{}", 1.0 if i == 200 else 0.0),
            (ts, LABEL, "vllm:kv_cache_usage_perc", "{}", 0.25 + (0.5 if i == 100 else 0.0)),
            (ts, LABEL, "vllm:generation_tokens_total", "{}", 100.0 * i),  # 50 tok/s
            (ts, LABEL, "vllm:prompt_tokens_total", "{}", 1000.0 * i),  # 500 tok/s
            (ts, LABEL, "vllm:prefix_cache_queries_total", "{}", 1000.0 * i),
            (ts, LABEL, "vllm:prefix_cache_hits_total", "{}", 600.0 * i),
            (ts, LABEL, "vllm:num_preemptions_total", "{}", 3.0 if i >= 250 else 0.0),
            (ts, LABEL, "vllm:time_to_first_token_seconds_sum", "{}", 1.0 * i),
        ]
        for g in (0, 1):
            labels = json.dumps({"gpu": str(g)})
            rows += [
                (ts, LABEL, "gpu:utilization_percent", labels, 80.0 + 10 * g + (i % 2)),
                (ts, LABEL, "gpu:memory_used_mib", labels, 90000.0 + i),
                (ts, LABEL, "gpu:memory_total_mib", labels, 97887.0),
                (ts, LABEL, "gpu:power_watts", labels, 400.0 + g),
                (ts, LABEL, "gpu:power_limit_watts", labels, 600.0),
                (ts, LABEL, "gpu:temperature_celsius", labels, 70.0 + (5 if i == 42 else 0)),
                (ts, LABEL, "gpu:sm_clock_mhz", labels, 2400.0 - (300 if i == 10 else 0)),
                (ts, LABEL, "gpu:pcie_rx_mb_per_second", labels, 10.0),
                (ts, LABEL, "gpu:pcie_tx_mb_per_second", labels, 2.0),
            ]
    with store._lock:
        store._sync.executemany(
            "INSERT INTO engine_samples (ts, deployment_id, name, labels, value) VALUES (?, ?, ?, ?, ?)", rows)
    reqs = [
        # (offset, key, session, outcome, first_ms, end_ms, prompt, cached, completion)
        (60, "laptop", "sess-a", "ok", 200.0, 2000.0, 1000, 800, 100),
        (120, "laptop", "sess-a", "ok", 400.0, 4000.0, 2000, 1000, 200),
        (180, "laptop", "sess-b", "client_cancelled", 300.0, 900.0, None, None, None),
        (240, "ci<script>", None, "ok", 1000.0, 9000.0, 500, 0, 300),
        (300, "ci<script>", None, "upstream_error", None, 50.0, None, None, None),
        (360, "laptop", "sess-b", "ok", 600.0, 3000.0, 4000, 4000, 50),
    ]
    for i, (off, key, session, outcome, first, end, prompt, cached, completion) in enumerate(reqs):
        rec = RequestRecord(
            request_id=f"req-{i}", deployment_id=LABEL, key_id=1 if key == "laptop" else 2, actor_id=key,
            api="ollama" if i % 2 else "openai", session_id=session, model="dev-test", stream=True,
            messages_count=2 + i, t_arrival_wall=t0 + off, open_requests_at_start=i % 2,
            status_code=200 if outcome != "upstream_error" else 502, outcome=outcome,
            finish_reason="stop" if outcome == "ok" else None, first_payload_ms=first,
            first_text_ms=first + 10 if first else None, first_tool_ms=250.0 if i == 0 else None, end_ms=end,
            max_gap_ms=40.0 * (i + 1), prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached,
            reasoning_tokens=5 if i == 0 else None, decode_tps=40.0 + i if outcome == "ok" else None,
            delivered_tps=30.0 + i if outcome == "ok" else None, served_max_model_len=131072,
            context_fraction=(prompt + completion) / 131072 if prompt else None,
            srv_metrics_json='{"queue_time": 0.01}' if i == 0 else None,
        )
        row = rec.row()
        cols = list(row)
        store.write(f"INSERT INTO requests ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    tuple(row[c] for c in cols))
    inventory = FakeGpu().inventory
    for kind, data in (("gpu_inventory", inventory), ("vllm_version", {"version": "0.34.0"}),
                       ("vllm:cache_config_info", [{"block_size": "16", "num_gpu_blocks": "16384",
                                                    "cache_dtype": "auto", "gpu_memory_utilization": "0.9",
                                                    "enable_prefix_caching": "True"}])):
        store.write("INSERT INTO inventory (ts, deployment_id, kind, data) VALUES (?, ?, ?, ?)",
                    (t0 + 1, LABEL, kind, json.dumps(data)))


def test_report_of_a_seeded_database(env):
    seed(env.ctx.store)
    report = export.build_report(env.ctx.store.read, T0 - 3600, T0 + 3600)
    assert report["format"] == "ovp-report/1" and report["title"] == LABEL and report["deployments"] == [LABEL]
    assert report["served_model"] == "dev-test" and report["context_length"] == 131072
    w = report["window"]
    assert T0 - 15 <= w["start"] <= T0 and T0 + 598 <= w["end"] <= T0 + 615  # trimmed to where the data is

    s = report["summary"]
    assert (s["n"], s["ok"], s["cancelled"], s["errors"]) == (6, 4, 1, 1)
    assert s["first_p50"] == 400.0 and s["first_p90"] == 1000.0  # of 200, 300, 400, 600, 1000
    assert s["end_p50"] == 2000.0 and s["end_p90"] == 9000.0
    assert s["prompt_tokens"] == 7500 and s["completion_tokens"] == 650
    assert s["cached_ratio"] == pytest.approx(5800 / 7500, abs=1e-4)
    assert s["prompt_max"] == 4000 and s["ctx_max"] == pytest.approx(4050 / 131072)
    assert s["running_max"] == 4.0 and s["waiting_max"] == 1.0 and s["open_max"] == 2.0
    assert s["kv_max"] == 0.75 and s["preemptions"] == 3
    assert s["output_tps_mean"] == pytest.approx(50.0, rel=0.01)
    assert s["prompt_tps_max"] == pytest.approx(500.0, rel=0.01)

    keys = {k["key"]: k for k in report["keys"]}
    assert set(keys) == {"laptop", "ci<script>"}
    assert keys["laptop"]["n"] == 4 and keys["laptop"]["cancelled"] == 1 and keys["laptop"]["tool_p50"] == 250.0
    assert keys["ci<script>"]["errors"] == 1 and keys["ci<script>"]["decode_p50"] == 43.0

    engine = report["engine"]
    assert len(engine) == 300  # fewer samples than chart points: nothing is merged
    assert engine[5]["output_tps"] == pytest.approx(50.0) and engine[5]["prefix_hit"] == 0.6
    assert engine[5]["g1_util"] in (90.0, 91.0) and "vllm:time_to_first_token_seconds_sum" not in engine[5]
    g0, g1 = report["gpus"]
    assert g0["samples"] == 300 and g0["util_mean"] == 80.5 and g1["util_max"] == 91.0
    assert g0["mem_used_max_mib"] == 90299.0 and g0["mem_total_mib"] == 97887.0
    assert g1["power_mean"] == 401.0 and g0["power_limit"] == 600.0
    assert g0["temp_max"] == 75.0 and g0["sm_clock_min"] == 2100.0 and g0["rx_mean"] == 10.0 and g0["tx_max"] == 2.0
    assert report["hardware"]["vllm_version"] == {"version": "0.34.0"}
    assert report["hardware"]["gpu_inventory"]["gpus"][1]["index"] == "1"
    assert len(report["requests"]) == 6 and report["requests"][0]["srv_metrics"] == {"queue_time": 0.01}
    assert "ovp-report_rtx-pro-6000x2" in export.filename(report, "html")


def test_report_filters_by_key_session_and_time(env):
    seed(env.ctx.store)
    read = env.ctx.store.read
    by_key = export.build_report(read, T0, T0 + 600, key="laptop")
    assert by_key["summary"]["n"] == 4 and [k["key"] for k in by_key["keys"]] == ["laptop"]
    assert by_key["filter"] == {"key": "laptop", "session": None}
    assert len(by_key["engine"]) > 100 and by_key["gpus"]  # engine-wide data is still there

    by_session = export.build_report(read, T0, T0 + 600, session="sess-a")
    assert [r["request_id"] for r in by_session["requests"]] == ["req-0", "req-1"]
    w = by_session["window"]  # charts zoom in on the session
    assert T0 + 45 <= w["start"] <= T0 + 60 and T0 + 124 <= w["end"] <= T0 + 140
    both = export.build_report(read, T0, T0 + 600, key="laptop", session="sess-b")
    assert [r["request_id"] for r in both["requests"]] == ["req-2", "req-5"]

    part = export.build_report(read, T0 + 100, T0 + 250)
    assert [r["request_id"] for r in part["requests"]] == ["req-1", "req-2", "req-3"]
    assert all(T0 + 100 <= p["t"] <= T0 + 250 for p in part["engine"]) and part["summary"]["preemptions"] == 0

    empty = export.build_report(read, T0 + 5000, T0 + 6000, key="nobody")
    assert empty["summary"]["n"] == 0 and empty["engine"] == [] and empty["gpus"] == []
    assert empty["title"] == "Proxy report" and "No requests in this scope." in export.render_html(empty)
    with pytest.raises(export.ExportError):
        export.build_report(read, T0 + 10, T0)

    # a key that was renamed later is still found under its current name
    env.ctx.store.write("INSERT INTO api_keys (id, name, key_prefix, key_sha256, created_at) VALUES (2, 'ci', 'x', 'y', 0)")
    assert export.build_report(read, T0, T0 + 600, key="ci")["summary"]["n"] == 2


def test_long_ranges_are_merged_in_sql(env):
    seed(env.ctx.store)
    points, gpus = export.engine_history(env.ctx.store.read, T0, T0 + 600, max_points=30)
    assert 28 <= len(points) <= 31
    rates = [p["output_tps"] for p in points if p["output_tps"] is not None]
    assert len(rates) >= 28 and all(r == pytest.approx(50.0, rel=0.01) for r in rates)
    assert max(p["kv"] for p in points) == 0.75 and max(p["waiting"] for p in points) == 1.0  # peaks survive
    assert sum(p["preemptions"] or 0 for p in points) == 3
    assert max(p["g0_temp"] for p in points) == 75.0
    assert gpus[0]["util_mean"] == 80.5 and gpus[0]["samples"] == 300 and gpus[0]["sm_clock_min"] == 2100.0
    hist = export.history(env.ctx.store.read, T0, T0 + 600)
    assert len(hist["requests"]) == 6 and "request_id" not in hist["requests"][0]


def test_html_report_is_self_contained(env):
    seed(env.ctx.store)
    report = export.build_report(env.ctx.store.read, T0, T0 + 600, proxy={"dropped": 0, "failed": 0, "rejected": 2})
    page = export.render_html(report, tz_offset_min=120)
    # no executable script: the only <script> is the JSON data block
    scripts = re.findall(r"<script\b[^>]*>", page)
    assert scripts == ['<script type="application/json" id="report-data">']
    assert not re.search(r"\son[a-z]+\s*=", page) and "javascript:" not in page
    # no external requests of any kind
    assert not re.search(r"https?://|//[a-z]|url\(|@import|<link|<img|<iframe|\ssrc=|\shref=", page)
    # the key name with markup in it is escaped everywhere, including inside the data block
    assert "<script>" not in page.replace('<script type="application/json" id="report-data">', "")
    assert "ci&lt;script&gt;" in page
    data = json.loads(re.search(r'id="report-data">(.*?)</script>', page, re.S).group(1))
    assert data["summary"] == json.loads(json.dumps(report["summary"])) and len(data["requests"]) == 6
    assert data["keys"][0]["key"] == "ci<script>"
    # content: sections, charts, viewer time zone, print and dark styles
    for text in ("Hardware and vLLM", "Concurrency", "KV cache used", "Output tokens/s", "Prompt tokens/s",
                 "Prefix cache hit rate", "GPU utilization", "GPU memory used", "GPU power", "GPU temperature",
                 "SM clock", "PCIe traffic", "GPU history over this range", "Time to first content per request",
                 "Per key", "Requests", "nvidia-smi topo -m", "vLLM startup lines", "NVIDIA RTX PRO 6000 Blackwell",
                 "580.95.05 / 13.0", "262,144 tokens", "2.00× requests at 131,072 tokens each", "UTC+02:00"):
        assert text in page, text
    assert page.count("<svg") == 12 and "@media print" in page and "prefers-color-scheme:dark" in page
    assert page.count("<circle") >= 5 and 'fill="var(--surface-1)" stroke="var(--series-' in page  # hollow = not OK
    assert time.strftime("%H:%M:%S", time.gmtime(T0 + 60 + 7200)) in page


def test_report_without_gpu_data_omits_gpu_sections(env):
    store = env.ctx.store
    store.write("INSERT INTO engine_samples (ts, deployment_id, name, labels, value) VALUES (?, NULL, ?, '{}', 1)",
                (T0, "vllm:num_requests_running"))
    store.write("INSERT INTO engine_samples (ts, deployment_id, name, labels, value) VALUES (?, NULL, ?, '{}', 2)",
                (T0 + 30, "vllm:num_requests_running"))
    report = export.build_report(store.read, T0 - 60, T0 + 60, served_model="dev-test")
    page = export.render_html(report)
    assert report["title"] == "dev-test" and report["gpus"] == []  # falls back to the served model
    assert "GPU utilization" not in page and "PCIe traffic" not in page and page.count("<svg") == 6
    assert "No GPU helper data or vLLM configuration" in page


# -- CLI -----------------------------------------------------------------------------


def test_cli_export(env, tmp_path, monkeypatch, capsys):
    seed(env.ctx.store)
    monkeypatch.setenv("DATA_DIR", env.settings.data_dir)
    monkeypatch.chdir(tmp_path)  # no stray .env
    out = tmp_path / "report.html"
    assert cli_main(["export", "--from", str(T0), "--to", str(T0 + 600), "--key", "laptop", "--tz-offset", "60",
                     "-o", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert "Report · " + LABEL in page and "UTC+01:00" in page and "key laptop" in page
    assert "wrote" in capsys.readouterr().err

    assert cli_main(["export", "--to", str(T0 + 600), "--last", "10m", "--session", "sess-b", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["filter"]["session"] == "sess-b" and len(data["requests"]) == 2

    assert cli_main(["export", "--list"]) == 0
    listing = capsys.readouterr().out
    assert "laptop" in listing and "sess-a" in listing and LABEL in listing and "6 requests" in listing

    assert cli_main(["export", "--from", "yesterday"]) == 1
    assert "cannot read 'yesterday'" in capsys.readouterr().err
    assert cli_main(["export", "--from", str(T0), "--last", "1h"]) == 1


def test_cli_time_arguments():
    assert export.parse_span("90m") == 5400 and export.parse_span("2h") == 7200 and export.parse_span("7d") == 604800
    assert export.parse_time("1780000000") == 1_780_000_000.0
    assert export.parse_time("2026-11-24T09:30:00+00:00") == 1795512600.0
    assert export.resolve_range(None, None, None, now=1000.0) == (1000.0 - 86400, 1000.0)
    assert export.resolve_range(None, "500", "5m") == (200.0, 500.0)
    with pytest.raises(export.ExportError):
        export.parse_span("soon")


# -- routes --------------------------------------------------------------------------


async def test_monitoring_api_needs_the_admin_token(env):
    for path in ("/admin/api/live", "/admin/api/live?window=120", "/admin/api/export", "/admin/api/export/options"):
        assert (await env.anon.get(path)).status_code == 401, path
        assert (await env.http.get(path)).status_code == 401, path  # an API key is not an admin token
        assert (await env.admin.get(path)).status_code == 200, path


async def test_static_page_is_served_without_inline_script(env):
    page = await env.anon.get("/admin/monitor")
    assert page.status_code == 200 and "script-src 'self'" in page.headers["content-security-policy"]
    assert re.findall(r"<script\b[^>]*>", page.text) == [
        '<script src="/admin/static/common.js">', '<script src="/admin/static/monitor.js">']
    assert not re.search(r"\son[a-z]+=", page.text) and 'id="login"' in page.text
    for name in ("monitor.js", "monitor.css"):
        assert (await env.anon.get(f"/admin/static/{name}")).status_code == 200


async def test_live_route(mon):
    mon.with_gpu()
    await mon.sampler.sample_once()
    await mon.http.post("/api/chat", json={**CHAT, "stream": False})
    await asyncio.sleep(0.5)  # longer than one chart point of a 2-minute window
    await mon.sampler.sample_once()
    await mon.sampler.inventory_once()
    mon.ctx.store.flush()

    # The proxy started a moment ago, so the ring cannot vouch for 2 minutes: SQLite answers.
    snap = (await mon.admin.get("/admin/api/live?window=120")).json()
    assert snap["source"] == "database" and snap["window_s"] == 120
    assert snap["totals"]["completed"] == 1 and snap["keys"][0]["key"] == "tests"
    assert len(snap["series"]) == 2 and snap["series"][1]["output_tps"] > 0
    assert snap["gpus"][0]["util_mean"] == 87.0 and snap["current"]["gpus"][0]["util"] == 87.0
    assert snap["backend"]["state"] == "ready" and snap["backend"]["label"] == LABEL
    assert snap["hardware"]["vllm_version"] == {"version": "0.34.0"}
    assert snap["store"] == {"queued": 0, "dropped": 0, "failed": 0} and snap["metrics_ok"] is True

    # Once the ring covers the window the same data comes from memory.
    mon.ctx.started_at -= 3600
    snap = (await mon.admin.get("/admin/api/live?window=120")).json()
    assert snap["source"] == "memory" and snap["totals"]["completed"] == 1 and len(snap["series"]) == 2
    assert snap["refresh_s"] == 1.5 and snap["gpus"][0]["util_mean"] == 87.0

    day = (await mon.admin.get("/admin/api/live?window=86400")).json()
    assert day["source"] == "database" and day["totals"]["completed"] == 1 and day["refresh_s"] == 60
    assert (await mon.admin.get("/admin/api/live?window=1")).json()["window_s"] == 30  # clamped
    assert (await mon.admin.get("/admin/api/live?window=1e12")).json()["window_s"] == 365 * 86400
    assert (await mon.admin.get("/admin/api/live?window=abc")).status_code == 400


async def test_export_routes(env):
    seed(env.ctx.store)
    opts = (await env.admin.get("/admin/api/export/options")).json()
    assert [k["key"] for k in opts["keys"]] == ["ci<script>", "laptop"] and opts["requests"] == 6
    assert {s["session"]: s["requests"] for s in opts["sessions"]} == {"sess-a": 2, "sess-b": 2}
    assert opts["unnamed_session_requests"] == 2
    assert opts["deployments"][0]["label"] == LABEL and opts["deployments"][0]["requests"] == 6
    assert opts["span"]["first"] == T0 and opts["span"]["last"] == T0 + 598

    q = f"from={T0}&to={T0 + 600}"
    r = await env.admin.get(f"/admin/api/export?{q}&format=html&tz_offset=-300&key=laptop")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert re.fullmatch(r'attachment; filename="ovp-report_rtx-pro-6000x2[A-Za-z0-9._-]+_laptop\.html"',
                        r.headers["content-disposition"])
    assert "UTC-05:00" in r.text and "key laptop" in r.text and "writer: 0 dropped" in r.text

    r = await env.admin.get(f"/admin/api/export?{q}&format=json&session=sess-a")
    data = r.json()
    assert r.headers["content-disposition"].endswith('_sess-a.json"')
    assert data["summary"]["n"] == 2 and data["proxy"] == {"dropped": 0, "failed": 0, "rejected": 0}

    assert (await env.admin.get("/admin/api/export?format=pdf")).status_code == 400
    assert (await env.admin.get("/admin/api/export?from=x")).status_code == 400
    r = await env.admin.get(f"/admin/api/export?from={T0 + 10}&to={T0}")
    assert r.status_code == 400 and "later" in r.json()["error"]


async def test_export_includes_rows_still_in_the_writer_queue(env):
    await env.http.post("/api/chat", json={**CHAT, "stream": False})
    data = (await env.admin.get("/admin/api/export?format=json")).json()  # default range: the last hour
    assert data["summary"]["n"] == 1 and data["requests"][0]["actor_id"] == "tests"
    assert data["title"] == "dev-test"
    assert not {"prompt", "content", "messages"} & set(data["requests"][0])
