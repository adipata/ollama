"""The deployment state machine against the in-process SSH "pod" (PLAN 12.4-12.6, 15)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import stat

import pytest
from fake_pod.rig import Rig
from fake_pod.server import GOOD_TOKEN

from ollama_vllm_proxy.pod import logs as podlogs
from ollama_vllm_proxy.pod.deploy import SECRET_VLLM_KEY
from ollama_vllm_proxy.routes.pod import status_events

CHAT = {"model": "dev-test", "messages": [{"role": "user", "content": "hi"}], "stream": False}
ALL_DONE = {s: "done" for s in ("connect", "preflight", "helper", "install", "download", "secrets", "start",
                                "ready", "activate")}


@pytest.fixture
async def rig(tmp_path):
    r = await Rig(tmp_path).start()
    yield r
    await r.close()


def step(status, step_id):
    return next(s for s in status["deployment"]["steps"] if s["id"] == step_id)


def gone(pid: int) -> bool:
    """The process is dead (or a zombie nobody reaped yet)."""
    try:
        return "\nState:\tZ" in open(f"/proc/{pid}/status").read()
    except OSError:
        return True


async def chat_ok(rig) -> bool:
    resp = await rig.http.post("/api/chat", json=CHAT)
    return resp.status_code == 200 and resp.json()["message"]["content"] == "Hello from fake vLLM."


# -- connection and host key --------------------------------------------------------------


async def test_connection_string_is_parsed_and_shown(rig):
    resp = await rig.admin.put("/admin/api/pod", json={"ssh_command": "ssh root@213.173.99.7 -p 10909 -i ~/.ssh/nope"})
    body = resp.json()
    assert resp.status_code == 200
    assert (body["parsed"]["host"], body["parsed"]["port"], body["parsed"]["user"]) == ("213.173.99.7", 10909, "root")
    assert body["parsed"]["key_path"] == os.path.expanduser("~/.ssh/nope")
    assert "does not exist" in body["parsed"]["key_problem"]
    bad = await rig.admin.put("/admin/api/pod", json={"ssh_command": "ssh root@1.2.3.4 -o ProxyCommand=evil"})
    assert bad.status_code == 400 and "'-o'" in bad.json()["error"]
    # the default key comes from SSH_KEY_PATH when the command has no -i
    resp = await rig.admin.put("/admin/api/pod", json={"ssh_command": f"ssh root@127.0.0.1 -p {rig.pod.port}"})
    assert resp.json()["parsed"]["key_path"] == str(rig.pod.key_path)
    # connecting with an unusable key file is refused before anything is sent
    await rig.admin.put("/admin/api/pod", json={"ssh_command": "ssh root@127.0.0.1 -i /nonexistent/key"})
    refused = await rig.admin.post("/admin/api/pod/connect", json={})
    assert refused.status_code == 400 and "does not exist" in refused.json()["error"]


async def test_host_key_must_be_confirmed_before_anything_is_sent(rig):
    await rig.admin.put("/admin/api/pod", json={"ssh_command": rig.pod.ssh_command})
    first = (await rig.admin.post("/admin/api/pod/connect", json={})).json()
    assert first["state"] == "confirm" and first["fingerprint"] == rig.pod.fingerprint
    assert first["fingerprint"].startswith("SHA256:")
    # a fingerprint that is not the server's is never accepted
    wrong = (await rig.admin.post("/admin/api/pod/connect", json={"accept_fingerprint": "SHA256:" + "A" * 43})).json()
    assert wrong["state"] == "confirm" and wrong["fingerprint"] == rig.pod.fingerprint
    assert rig.pod.commands == [] and rig.pod.auth_attempts == 0  # no login, no command, no upload so far
    assert (await rig.status())["backend"]["state"] == "no_pod"
    done = (await rig.admin.post("/admin/api/pod/connect", json={"accept_fingerprint": first["fingerprint"]})).json()
    assert done["state"] == "connected"
    status = await rig.status()
    assert status["backend"]["state"] == "connected" and status["pod"]["connected"]
    assert status["pod"]["fingerprint"] == rig.pod.fingerprint
    assert "RTX PRO 6000" in status["pod"]["facts"]["gpu_summary"]
    assert rig.pod.forwards == []  # nothing forwarded until something is deployed
    # entering a connection string again discards the confirmation
    await rig.admin.post("/admin/api/pod/disconnect")
    await rig.admin.put("/admin/api/pod", json={"ssh_command": rig.pod.ssh_command})
    assert (await rig.admin.post("/admin/api/pod/connect", json={})).json()["state"] == "confirm"


async def test_reconnect_to_a_different_host_key_stops_and_asks_again(rig):
    await rig.connect()
    old = rig.pod.fingerprint
    await rig.pod.new_host_key()  # the pod was restarted: same address, new key
    await rig.wait_for(lambda: rig.mgr.ssh.state == "hostkey_changed", what="host key change detection")
    status = await rig.status()
    assert status["backend"]["state"] == "no_pod" and "host key changed" in status["backend"]["detail"]
    assert status["pod"]["fingerprint"] is None
    commands_before = len(rig.pod.commands)
    again = (await rig.admin.post("/admin/api/pod/connect", json={})).json()
    assert again["state"] == "confirm" and again["fingerprint"] == rig.pod.fingerprint != old
    # presenting the old confirmation does not help
    stale = (await rig.admin.post("/admin/api/pod/connect", json={"accept_fingerprint": old})).json()
    assert stale["state"] == "confirm" and stale["changed"]
    assert len(rig.pod.commands) == commands_before
    ok = (await rig.admin.post("/admin/api/pod/connect", json={"accept_fingerprint": again["fingerprint"]})).json()
    assert ok["state"] == "connected"


async def test_remote_commands_are_quoted_argument_lists(rig):
    await rig.connect()
    result = await rig.mgr.ssh.run(["printf", "%s|", "$(id)", "a;b", "it's", "`x`", "two words", ""])
    assert result.stdout == "$(id)|a;b|it's|`x`|two words||"


# -- the full sequence ---------------------------------------------------------------------


async def test_full_deployment_and_inference_through_the_tunnel(rig, caplog):
    caplog.set_level(logging.DEBUG)
    await rig.connect()
    assert (await rig.http.post("/api/chat", json=CHAT)).status_code == 503
    status = await rig.deploy()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    assert rig.steps(status) == ALL_DONE
    assert all(s["started_at"] and s["ended_at"] >= s["started_at"] for s in status["deployment"]["steps"])
    assert status["backend"]["state"] == "ready" and status["backend"]["gpu_helper"]
    assert status["backend"]["label"] == "rtx-pro-6000x1-qwen38-27b-bf16-131072"
    assert status["backend"]["context_length"] == 131072
    dl = step(status, "download")
    assert dl["progress"]["total"] == 54_020_000_000 and dl["progress"]["done"] >= dl["progress"]["total"]
    assert "rtx-pro-6000x1" in step(status, "activate")["output"]

    # an inference request goes proxy -> SSH forward -> stub vLLM
    assert await chat_ok(rig)
    show = await rig.http.post("/api/show", json={"model": "dev-test"})
    assert show.status_code == 200 and "131072" in show.text
    row = rig.ctx.store.read_one("SELECT * FROM deployments")
    assert row["state"] == "ready" and row["label"] == status["backend"]["label"]
    assert json.loads(row["snapshot"])["data"]["served_model_name"] == "dev-test"
    assert json.loads(row["plan"])["tensor_parallel"] == 1

    # what the stub vLLM was started with
    run = rig.pod.records("vllm")[-1]
    argv = run["argv"]
    root = str(rig.pod.root)
    assert argv[:2] == ["serve", f"{root}/qwen3.8-27b-bf16"]
    assert argv[-10:] == ["--served-model-name", "dev-test", "--max-model-len", "131072", "--tensor-parallel-size",
                          "1", "--host", "127.0.0.1", "--port", str(rig.pod.vllm_port)]
    assert "--tool-call-parser" in argv and "--api-key" not in argv
    # clean environment: the pod's own secrets are not inherited, the key arrives by environment
    key = rig.ctx.store.get_secret(SECRET_VLLM_KEY)
    assert run["env"]["VLLM_API_KEY"] == key and len(key) > 30
    assert "RUNPOD_API_KEY" not in run["env"] and "JUPYTER_PASSWORD" not in run["env"]
    assert "RUNPOD_POD_ID" not in run["env"] and run["env"]["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert run["env"]["HF_HOME"] == f"{root}/.hf" and "HF_TOKEN" not in run["env"]
    assert run["nofile"] >= min(65535, os.sysconf("SC_OPEN_MAX")) or run["nofile"] > 1024
    hf_env = rig.pod.records("hf")[-1]["env"]
    assert "RUNPOD_API_KEY" not in hf_env and hf_env["HF_HUB_ENABLE_HF_TRANSFER"] == "1" and "HF_TOKEN" not in hf_env
    assert stat.S_IMODE(os.stat(f"{root}/ovp/vllm.env").st_mode) == 0o600
    assert f"VLLM_API_KEY={key}" in open(f"{root}/ovp/vllm.env").read()

    # the key is nowhere it should not be
    assert all(key not in command for command in rig.pod.commands)
    pid = rig.pod.job_pid("vllm")
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                cmdline = open(f"/proc/{name}/cmdline", "rb").read()
            except OSError:
                continue
            assert key.encode() not in cmdline
    assert pid and b"job.sh" in open(f"/proc/{pid}/cmdline", "rb").read()
    for path in ("/admin/api/deploy/status", "/admin/api/pod", "/admin/api/profiles", "/admin/api/status",
                 "/admin/api/logs/download?source=vllm", "/admin/api/settings/hf-token"):
        resp = await rig.admin.get(path)
        assert resp.status_code == 200 and key not in resp.text, path
    assert key not in caplog.text
    assert "rpa_PODSECRET" not in json.dumps(status) and "jupyter-pw" not in json.dumps(status)
    # the six RUNPOD_* facts are recorded, nothing else from the pod's environment
    rig.ctx.store.flush()
    inventory = json.loads(rig.ctx.store.read_one("SELECT data FROM inventory WHERE kind = 'pod'")["data"])
    assert inventory["runpod"] == {"RUNPOD_POD_ID": "fakepod0001", "RUNPOD_DC_ID": "EU-TEST-1", "RUNPOD_GPU_COUNT": "1"}
    # both forwards were used: vLLM and the GPU helper
    assert {port for _, port in rig.pod.forwards} == {rig.pod.vllm_port, rig.pod.helper_port}


async def test_logs_are_relayed_redacted(rig):
    await rig.connect()
    await rig.deploy()
    key = rig.ctx.store.get_secret(SECRET_VLLM_KEY)
    raw = open(f"{rig.pod.root}/vllm.log").read()
    assert key in raw  # the stub echoes its key, as a real server's argument dump could
    lines: list[str] = []
    gen = podlogs.follow(rig.mgr.ssh, f"{rig.pod.root}/vllm.log", rig.mgr.secret_values())
    async for batch in gen:
        lines += batch
        if any("Application startup complete" in line for line in lines):
            break
    await gen.aclose()
    text = "\n".join(lines)
    assert "vLLM API server version 0.34.0" in text and "'api_key': '***'" in text and "Bearer ***" in text
    assert key not in text
    # the SSE body for each source, and the full-log download
    events: list[dict] = []
    stream = podlogs.stream(rig.mgr, "install")
    async for chunk in stream:
        events.append(json.loads(chunk.decode().removeprefix("data: ")))
        if any("vllm==0.34.0" in line for e in events for line in e.get("lines", [])):
            break
    await stream.aclose()
    assert events[0]["reset"] and events[0]["path"].endswith("/ovp/install.log")
    for source, needle in (("vllm", "Application startup complete"), ("download", "download complete"),
                           ("install", "OVP_VERSIONS"), ("helper", "")):
        resp = await rig.admin.get(f"/admin/api/logs/download?source={source}")
        assert resp.status_code == 200 and needle in resp.text and key not in resp.text
        assert "attachment" in resp.headers["content-disposition"]
    assert (await rig.admin.get("/admin/api/logs/stream?source=secrets")).status_code == 400
    # following stops when the consumer goes away: no tail process is left behind on the pod
    await asyncio.sleep(0.3)
    assert not [p for p in rig.pod._procs if p.returncode is None]


async def test_redaction_of_key_like_values():
    red = podlogs.redact
    assert red("Authorization: Bearer abc.DEF-123") == "Authorization: Bearer ***"
    assert red("args: {'api_key': 'vk-abcdef123456', 'port': 8000}") == "args: {'api_key': '***', 'port': 8000}"
    assert red("vllm serve --api-key s3cr3tvalue --port 8000") == "vllm serve --api-key *** --port 8000"
    assert red("HF_TOKEN=hf_abcdefghijklmnopqrstuvwx") == "HF_TOKEN=***"
    assert red("using hf_abcdefghijklmnopqrstuvwx now") == "using hf_*** now"
    assert red("RUNPOD_API_KEY=rpa_ABCDEFGHIJKLMNOPQRSTUVWX") == "RUNPOD_API_KEY=***"
    assert red("password: hunter2") == "password: ***"
    assert red("the exact value 1234567890 appears", ["1234567890"]) == "the exact value *** appears"
    assert red("prompt tokens: 1200, max_tokens=64") == "prompt tokens: 1200, max_tokens=64"
    assert podlogs.clean("\x1b[32mINFO\x1b[0m ok\r\n") == "INFO ok"


async def test_status_events_stream(rig):
    await rig.connect()
    gen = status_events(rig.mgr, heartbeat=0.2)
    first = json.loads((await gen.__anext__()).decode().removeprefix("data: "))
    assert first["backend"]["state"] == "connected" and first["deployment"] is None
    resp = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id()})
    assert resp.status_code == 202
    seen: list[dict] = []
    async for chunk in gen:
        if chunk.startswith(b"data: "):
            seen.append(json.loads(chunk[6:]))
            if seen[-1]["backend"]["state"] == "ready":
                break
    await gen.aclose()
    assert any(s["backend"]["state"] == "deploying" for s in seen)
    details = {s["backend"]["detail"] for s in seen if s["backend"]["state"] == "deploying"}
    assert "Wait until ready" in details or "Start vLLM" in details or "Install vLLM" in details
    assert seen[-1]["deployment"]["state"] == "ready" and not rig.mgr._subs


# -- the plan drives the command -------------------------------------------------------------


@pytest.mark.parametrize("gpus, directory, tp, label_start", [
    ("2x48", "qwen3.8-27b-bf16", "2", "l40sx2-qwen38-27b-bf16-"),
    ("1x48", "qwen3.8-27b-fp8", "1", "l40sx1-qwen38-27b-fp8-"),
    ("2x24", "qwen3.8-27b-fp8", "2", "rtx-4090x2-qwen38-27b-fp8-"),
])
async def test_gpu_configuration_decides_variant_tp_and_context(tmp_path, gpus, directory, tp, label_start):
    rig = await Rig(tmp_path, gpus=gpus).start()
    try:
        await rig.connect()
        plan = (await rig.admin.get(f"/admin/api/profiles/{rig.profile_id()}/plan")).json()
        auto = next(r for r in plan["rows"] if r["variant"] == plan["auto"])
        status = await rig.deploy()
        assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
        argv = rig.vllm_argv()
        assert argv[1] == f"{rig.pod.root}/{directory}"
        assert argv[argv.index("--tensor-parallel-size") + 1] == tp
        context = int(argv[argv.index("--max-model-len") + 1])
        assert context == auto["context"] and 65536 <= context < 131072
        assert ("--kv-cache-dtype" in argv) == directory.endswith("fp8")
        assert status["backend"]["label"] == f"{label_start}{context}"
        assert status["backend"]["context_length"] == context
        assert auto["command"] == status["deployment"]["plan"]["command"]  # the previewed command is the one run
        assert await chat_ok(rig)
    finally:
        await rig.close()


async def test_mixed_gpus_select_the_identical_group(tmp_path):
    rig = await Rig(tmp_path, gpus="mixed").start()
    try:
        await rig.connect()
        status = await rig.deploy()
        assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
        run = rig.pod.records("vllm")[-1]
        assert run["env"]["CUDA_VISIBLE_DEVICES"] == "0,2"
        assert run["argv"][run["argv"].index("--tensor-parallel-size") + 1] == "2"
    finally:
        await rig.close()


async def test_overrides_and_deploy_anyway(tmp_path):
    rig = await Rig(tmp_path, gpus="1x48").start()
    try:
        await rig.connect()
        # the estimate rejects BF16 on one 48 GB card: refused before anything is installed or downloaded
        status = await rig.deploy(variant="BF16")
        assert status["deployment"]["state"] == "failed"
        assert "deploy anyway" in status["deployment"]["error"] and "54.0 GB" in status["deployment"]["error"]
        assert rig.steps(status)["preflight"] == "failed" and rig.steps(status)["install"] == "pending"
        assert rig.pod.records("hf") == [] and rig.pod.records("uv") == [] or all(
            r["argv"][:2] == ["python", "find"] for r in rig.pod.records("uv"))
        assert status["backend"]["state"] == "failed" and "Preflight" in status["backend"]["detail"]
        # with the explicit override it is attempted, with the chosen context and tensor-parallel size
        status = await rig.deploy(variant="BF16", context=70000, tensor_parallel=1, force=True)
        assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
        assert status["deployment"]["plan"]["forced"] is True
        argv = rig.vllm_argv()
        assert argv[1].endswith("qwen3.8-27b-bf16") and argv[argv.index("--max-model-len") + 1] == "70000"
        bad = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id(), "tensor_parallel": 3})
        assert bad.status_code == 400
        bad = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id(), "variant": "nope"})
        assert bad.status_code == 400
        assert (await rig.admin.post("/admin/api/deploy", json={"profile_id": 999})).status_code == 404
    finally:
        await rig.close()


async def test_unsupported_generation_stops_in_preflight(tmp_path):
    rig = await Rig(tmp_path, gpus="2xA6000").start()
    try:
        await rig.connect()
        status = await rig.deploy()
        assert status["deployment"]["state"] == "failed" and "no variant" in status["deployment"]["error"]
        assert "8.6" in step(status, "preflight")["output"]
        status = await rig.deploy(variant="FP8", force=True)  # FP8 on Ampere cannot be forced
        assert status["deployment"]["state"] == "failed" and "cannot be overridden" in status["deployment"]["error"]
        status = await rig.deploy(variant="BF16", force=True)  # BF16 can
        assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
        assert status["backend"]["label"].startswith("rtx-a6000x2-qwen38-27b-bf16-")
    finally:
        await rig.close()


# -- failure paths, step by step ----------------------------------------------------------------


async def test_deploy_needs_a_connection(rig):
    resp = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id()})
    assert resp.status_code == 409 and "connect" in resp.json()["error"]
    assert (await rig.admin.post("/admin/api/vllm/stop")).status_code == 409
    assert (await rig.admin.post("/admin/api/vllm/restart")).status_code == 409
    assert (await rig.admin.post("/admin/api/deploy/cancel")).status_code == 409


@pytest.mark.parametrize("breakage, message", [
    ("no_gpu", "cannot read the pod's GPUs"),
    ("no_uv", "`uv` was not found"),
    ("no_python", "Python 3.12 is not available"),
    ("offline", "pypi.org is not reachable"),
    ("small_disk", "container disk of at least 80 GB"),
    ("port_busy", "in use by another process"),
    ("tiny_gpu", "no variant of this profile fits"),
    ("hf_down", "no variant of this profile fits"),
    ("gated", "Hugging Face token"),
])
async def test_preflight_failures_stop_before_anything_is_downloaded(rig, breakage, message):
    await rig.connect()
    listener = None
    if breakage == "no_gpu":
        rig.pod.set_gpus(None)
    elif breakage == "no_uv":
        (rig.pod.bin / "uv").unlink()
    elif breakage == "no_python":
        rig.pod.set("uv_mode", "no_python")
    elif breakage == "offline":
        rig.pod.set("offline", "1")
    elif breakage == "small_disk":
        rig.pod.set("df_avail_kb", str(30 * 1000 * 1000))
    elif breakage == "port_busy":
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", rig.pod.vllm_port))
        listener.listen(1)
    elif breakage == "tiny_gpu":
        rig.pod.set_gpus("1x24")
    elif breakage == "hf_down":
        rig.hf.down = True
    elif breakage == "gated":
        rig.hf.gated.add("Qwen/Qwen3.8-27B")
    try:
        status = await rig.deploy(**({"variant": "BF16"} if breakage == "gated" else {}))
    finally:
        if listener:
            listener.close()
    assert status["deployment"]["state"] == "failed"
    assert message in status["deployment"]["error"], status["deployment"]["error"]
    assert rig.steps(status)["preflight"] == "failed"
    assert rig.steps(status)["helper"] == rig.steps(status)["download"] == "pending"
    assert rig.pod.records("hf") == [] and rig.pod.records("vllm") == []
    assert status["backend"]["state"] == "failed"
    assert (await rig.http.post("/api/chat", json=CHAT)).status_code == 503


async def test_gated_repo_works_with_the_token_and_the_token_stays_secret(rig):
    await rig.connect()
    rig.hf.gated.update({"Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-FP8"})
    assert (await rig.admin.get("/admin/api/settings/hf-token")).json() == {"set": False}
    assert (await rig.admin.put("/admin/api/settings/hf-token", json={"token": "has space"})).status_code == 400
    put = await rig.admin.put("/admin/api/settings/hf-token", json={"token": GOOD_TOKEN})
    assert put.json() == {"set": True} and GOOD_TOKEN not in put.text
    assert (await rig.admin.get("/admin/api/settings/hf-token")).json() == {"set": True}
    status = await rig.deploy()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    assert status["hf_token_set"] is True and GOOD_TOKEN not in json.dumps(status)
    assert rig.pod.records("hf")[-1]["env"]["HF_TOKEN"] == GOOD_TOKEN
    assert rig.pod.records("vllm")[-1]["env"]["HF_TOKEN"] == GOOD_TOKEN
    root = rig.pod.root
    for name in ("hf.env", "vllm.env"):
        assert stat.S_IMODE(os.stat(root / "ovp" / name).st_mode) == 0o600
    assert all(GOOD_TOKEN not in command for command in rig.pod.commands)
    for source in podlogs.SOURCES:
        assert GOOD_TOKEN not in (await rig.admin.get(f"/admin/api/logs/download?source={source}")).text
    assert (await rig.admin.delete("/admin/api/settings/hf-token")).json() == {"set": False}
    assert rig.ctx.store.get_secret("hf_token") is None


async def test_helper_failure(rig):
    await rig.connect()
    rig.pod.set("python3_fail", "1")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "failed" and "GPU helper did not start" in status["deployment"]["error"]
    assert rig.steps(status)["helper"] == "failed" and "stub failure" in step(status, "helper")["output"]
    assert rig.steps(status)["install"] == "pending"


async def test_install_failure(rig):
    await rig.connect()
    rig.pod.set("uv_mode", "fail")
    rig.pod.set("hf_delay", "0.3")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "failed"
    assert "Install vLLM: install failed (exit status 1)" in status["deployment"]["error"]
    assert "could not be installed" in step(status, "install")["output"]
    steps = rig.steps(status)
    assert steps["install"] == "failed" and steps["download"] == "cancelled" and steps["secrets"] == "pending"
    assert status["backend"]["state"] == "failed" and "Install vLLM" in status["backend"]["detail"]
    # fixing the cause and deploying again completes; the download that kept running is picked up
    rig.pod.set("uv_mode", "ok")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    assert len(rig.pod.records("hf")) == 1


async def test_download_failure(rig):
    await rig.connect()
    rig.pod.set("hf_mode", "fail")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "failed"
    assert "Download model: download failed (exit status 1)" in status["deployment"]["error"]
    assert "GatedRepoError" in step(status, "download")["output"]
    assert rig.steps(status)["secrets"] == "pending" and rig.pod.records("vllm") == []


async def test_vllm_crash_shows_the_last_log_lines(rig):
    await rig.connect()
    rig.pod.set("vllm_mode", "crash")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "failed"
    assert "vLLM exited before it was ready (exit status 1)" in status["deployment"]["error"]
    out = step(status, "ready")["output"]
    assert "CUDA out of memory" in out and rig.ctx.store.get_secret(SECRET_VLLM_KEY) not in out
    assert rig.steps(status)["ready"] == "failed" and rig.steps(status)["activate"] == "pending"
    assert status["backend"]["state"] == "failed" and (await rig.http.post("/api/chat", json=CHAT)).status_code == 503


async def test_restart_with_smaller_context_when_vllm_says_it_does_not_fit(rig):
    await rig.connect()
    rig.pod.set("vllm_mode", "nofit:98304")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    runs = [r["argv"][r["argv"].index("--max-model-len") + 1] for r in rig.pod.records("vllm")]
    assert runs == ["131072", "98304"]
    assert "restarting with 98,304" in step(status, "ready")["output"]
    assert status["backend"]["context_length"] == 98304
    assert status["backend"]["label"] == "rtx-pro-6000x1-qwen38-27b-bf16-98304"
    assert status["deployment"]["plan"]["context"] == 98304 and status["deployment"]["plan"]["context_requested"] == 131072
    assert await chat_ok(rig)


async def test_context_below_the_minimum_is_offered_not_applied(rig):
    await rig.connect()
    rig.pod.set("vllm_mode", "nofit:40000")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "failed"
    assert "at most 40,000 tokens" in status["deployment"]["error"] and "minimum of 65,536" in status["deployment"]["error"]
    assert status["deployment"]["offer"] == {"context": 40000}
    assert len(rig.pod.records("vllm")) == 1
    # the offered restart
    resp = await rig.admin.post("/admin/api/vllm/restart", json={"context": 40000})
    assert resp.status_code == 202
    status = await rig.wait_idle()
    assert status["deployment"]["state"] == "ready" and status["deployment"]["kind"] == "restart"
    assert status["backend"]["context_length"] == 40000 and await chat_ok(rig)
    assert len(rig.pod.records("hf")) == 1  # no second download


async def test_unprotected_vllm_is_never_activated(rig):
    await rig.connect()
    rig.pod.set("vllm_mode", "open")  # a server that answers 200 without a key
    status = await rig.deploy()
    assert status["deployment"]["state"] == "failed"
    assert "refusing to activate an unprotected server" in status["deployment"]["error"]
    assert rig.steps(status)["activate"] == "failed" and status["backend"]["state"] == "failed"
    assert rig.pod.job_pid("vllm") is None  # and it was stopped
    assert (await rig.http.post("/api/chat", json=CHAT)).status_code == 503


# -- resume, cancel, stop, restart ----------------------------------------------------------------


async def test_dropped_connection_during_a_step_is_waited_out(rig):
    await rig.connect()
    rig.pod.set("hf_delay", "0.25")
    resp = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id()})
    assert resp.status_code == 202
    await rig.wait_for(lambda: rig.pod.records("hf"), what="the download to start")
    pid = rig.pod.job_pid("download")
    rig.pod.drop_connections()
    await rig.wait_for(lambda: rig.mgr.ssh.state == "down", what="the drop to be noticed")
    status = await rig.wait_idle()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    assert len(rig.pod.records("hf")) == 1 and pid  # the same download carried on; it was not started again
    assert await chat_ok(rig)


async def test_tunnel_down_and_back_without_redeploy(rig):
    await rig.connect()
    await rig.deploy()
    starts = len(rig.pod.records("vllm"))
    await rig.pod.stop()  # the pod is unreachable for a while
    await rig.wait_for(lambda: rig.ctx.backend.state == "tunnel_down", what="tunnel_down")
    down = await rig.http.post("/api/chat", json=CHAT)
    assert down.status_code == 503 and "tunnel" in down.text
    await rig.pod.start()
    await rig.wait_for(lambda: rig.ctx.backend.state == "ready", what="ready again")
    assert await chat_ok(rig)
    assert len(rig.pod.records("vllm")) == starts
    assert (await rig.status())["backend"]["gpu_helper"]


async def test_resume_after_a_proxy_restart_mid_deployment(rig):
    await rig.connect()
    rig.pod.set("hf_delay", "0.25")
    resp = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id()})
    assert resp.status_code == 202
    await rig.wait_for(lambda: rig.pod.records("hf"), what="the download to start")
    await rig.shutdown_proxy()  # the proxy process goes away mid-download
    assert rig.pod.job_pid("download")  # the download does not care
    rig.boot()
    assert rig.ctx.store.read_one("SELECT state FROM deployments")["state"] == "running"
    await rig.mgr.startup()  # reconnects to the pinned host key on its own and resumes
    await rig.wait_for(lambda: rig.mgr.busy or rig.ctx.backend.state == "ready", what="the resume")
    status = await rig.wait_idle()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    assert status["deployment"]["id"] == 1 and rig.ctx.store.read_one("SELECT COUNT(*) AS n FROM deployments")["n"] == 1
    assert len(rig.pod.records("hf")) == 1
    assert len([r for r in rig.pod.records("uv") if r["argv"][:1] == ["venv"]]) == 1
    assert await chat_ok(rig)


async def test_proxy_restart_while_serving_reactivates_without_touching_vllm(rig):
    await rig.connect()
    await rig.deploy()
    pid, key = rig.pod.job_pid("vllm"), rig.ctx.store.get_secret(SECRET_VLLM_KEY)
    await rig.shutdown_proxy()
    rig.boot()
    assert rig.ctx.backend.state == "no_pod"
    await rig.mgr.startup()
    await rig.wait_for(lambda: rig.ctx.backend.state == "ready", what="reactivation")
    assert rig.pod.job_pid("vllm") == pid and rig.ctx.store.get_secret(SECRET_VLLM_KEY) == key
    assert len(rig.pod.records("vllm")) == 1 and await chat_ok(rig)
    assert rig.ctx.backend.label == "rtx-pro-6000x1-qwen38-27b-bf16-131072" and rig.ctx.backend.gpu_url


async def test_proxy_restart_does_not_reconnect_to_another_host_key(rig):
    await rig.connect()
    await rig.shutdown_proxy()
    await rig.pod.new_host_key()
    commands = len(rig.pod.commands)
    rig.boot()
    await rig.mgr.startup()
    await rig.wait_for(lambda: not rig.mgr._bg, what="the reconnect attempt to end")
    assert rig.ctx.backend.state == "no_pod" and "host key changed" in rig.ctx.backend.detail
    assert rig.mgr.ssh is None and len(rig.pod.commands) == commands
    assert rig.mgr.pod_row()["fingerprint"] is None


async def test_cancel_kills_the_remote_process(rig):
    await rig.connect()
    rig.pod.set("hf_delay", "0.5")
    resp = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id()})
    assert resp.status_code == 202
    again = await rig.admin.post("/admin/api/deploy", json={"profile_id": rig.profile_id()})
    assert again.status_code == 409  # one deployment at a time
    await rig.wait_for(lambda: rig.pod.job_pid("download"), what="the download to start")
    pid = rig.pod.job_pid("download")
    resp = await rig.admin.post("/admin/api/deploy/cancel")
    assert resp.status_code == 200
    status = resp.json()
    assert status["deployment"]["state"] == "cancelled" and not status["busy"]
    assert rig.steps(status)["download"] == "cancelled" and rig.steps(status)["secrets"] == "pending"
    assert rig.pod.job_pid("download") is None
    assert gone(pid)
    assert status["backend"]["state"] == "connected" and "cancelled" in status["backend"]["detail"]
    assert rig.pod.records("vllm") == []
    # and a later Deploy resumes the partial download
    rig.pod.set("hf_delay", "0.01")
    status = await rig.deploy()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]


async def test_stop_and_restart_vllm(rig):
    await rig.connect()
    await rig.deploy()
    pid, key = rig.pod.job_pid("vllm"), rig.ctx.store.get_secret(SECRET_VLLM_KEY)
    blocked = await rig.admin.post("/admin/api/pod/delete-models", json={})
    assert blocked.status_code == 409 and "stop it" in blocked.json()["error"]
    status = (await rig.admin.post("/admin/api/vllm/stop")).json()
    assert status["backend"]["state"] == "stopped" and status["deployment"]["state"] == "stopped"
    assert rig.pod.job_pid("vllm") is None
    assert gone(pid)
    stopped = await rig.http.post("/api/chat", json=CHAT)
    assert stopped.status_code == 503 and "stopped" in stopped.text
    assert rig.pod.job_pid("helper")  # the helper keeps running

    resp = await rig.admin.post("/admin/api/vllm/restart")
    assert resp.status_code == 202
    status = await rig.wait_idle()
    assert status["deployment"]["state"] == "ready" and status["deployment"]["kind"] == "restart"
    steps = rig.steps(status)
    assert steps["install"] == steps["download"] == steps["preflight"] == "skipped"
    assert steps["secrets"] == steps["start"] == steps["ready"] == steps["activate"] == "done"
    assert len(rig.pod.records("hf")) == 1 and len(rig.pod.records("vllm")) == 2
    assert rig.pod.records("vllm")[0]["argv"] == rig.pod.records("vllm")[1]["argv"]
    assert rig.ctx.store.get_secret(SECRET_VLLM_KEY) != key  # every start gets a fresh key
    assert await chat_ok(rig)
    # Restart while it is serving replaces the process
    pid = rig.pod.job_pid("vllm")
    await rig.admin.post("/admin/api/vllm/restart")
    await rig.wait_idle()
    assert rig.pod.job_pid("vllm") not in (None, pid) and await chat_ok(rig)


async def test_redeploy_is_idempotent(rig):
    await rig.connect()
    await rig.deploy()
    pid, key = rig.pod.job_pid("vllm"), rig.ctx.store.get_secret(SECRET_VLLM_KEY)
    helper = rig.pod.job_pid("helper")
    counts = {name: len(rig.pod.records(name)) for name in ("hf", "vllm")}
    installs = len([r for r in rig.pod.records("uv") if r["argv"][:1] == ["pip"]])
    status = await rig.deploy()
    assert status["deployment"]["state"] == "ready" and status["deployment"]["id"] == 2
    steps = rig.steps(status)
    assert [steps[s] for s in ("helper", "install", "download", "secrets", "start")] == ["skipped"] * 5
    assert steps["ready"] == steps["activate"] == "done"
    assert rig.pod.job_pid("vllm") == pid and rig.pod.job_pid("helper") == helper
    assert rig.ctx.store.get_secret(SECRET_VLLM_KEY) == key
    assert {name: len(rig.pod.records(name)) for name in ("hf", "vllm")} == counts
    assert len([r for r in rig.pod.records("uv") if r["argv"][:1] == ["pip"]]) == installs
    assert await chat_ok(rig)
    # vLLM died in the meantime: Deploy starts it again, and only it
    os.killpg(pid, 9)
    status = await rig.deploy()
    assert status["deployment"]["state"] == "ready"
    assert rig.steps(status)["download"] == "skipped" and rig.steps(status)["start"] == "done"
    assert len(rig.pod.records("vllm")) == 2 and len(rig.pod.records("hf")) == 1 and await chat_ok(rig)


async def test_profile_switch_stops_the_running_vllm_first(rig):
    await rig.connect()
    await rig.deploy()
    pid = rig.pod.job_pid("vllm")
    data = rig.mgr.profiles.list()[0]["data"]
    data["variants"] = [{"label": "BF16", "repo": "Tiny/Tiny-1B", "revision": "v2", "dir": "tiny-1b", "args": []}]
    data["context_wanted"], data["context_min"] = 32768, 8192
    created = await rig.admin.post("/admin/api/profiles", json={"name": "tiny", "data": data})
    assert created.status_code == 201
    rig.pod.set("hf_delay", "0.2")
    resp = await rig.admin.post("/admin/api/deploy", json={"profile_id": created.json()["id"]})
    assert resp.status_code == 202
    await rig.wait_for(lambda: len(rig.pod.records("hf")) == 2, what="the second download")
    assert rig.pod.job_pid("vllm") is None  # stopped before the new model is even downloaded
    assert gone(pid)
    status = await rig.wait_idle()
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    argv = rig.vllm_argv()
    assert argv[1] == f"{rig.pod.root}/tiny-1b" and argv[argv.index("--max-model-len") + 1] == "32768"
    assert rig.pod.records("hf")[-1]["argv"][-2:] == ["--revision", "v2"]
    assert rig.steps(status)["install"] == "skipped" and rig.steps(status)["download"] == "done"
    assert status["backend"]["label"] == "rtx-pro-6000x1-tiny-1b-bf16-32768" and await chat_ok(rig)
    # delete model files: refused while serving, done after Stop
    await rig.admin.post("/admin/api/vllm/stop")
    deleted = (await rig.admin.post("/admin/api/pod/delete-models", json={})).json()["deleted"]
    assert {d["dir"] for d in deleted} == {"qwen3.8-27b-bf16", "tiny-1b"}
    assert not (rig.pod.root / "tiny-1b").exists() and (rig.pod.root / "venv").exists()
    bad = await rig.admin.post("/admin/api/pod/delete-models", json={"dirs": ["venv"]})
    assert bad.status_code == 400


async def test_disconnect_leaves_the_pod_running(rig):
    await rig.connect()
    await rig.deploy()
    pid = rig.pod.job_pid("vllm")
    status = (await rig.admin.post("/admin/api/pod/disconnect")).json()
    assert status["backend"]["state"] == "no_pod" and not status["pod"]["connected"]
    await rig.wait_for(lambda: not rig.pod.connections, what="the SSH connection to close")
    assert rig.pod.job_pid("vllm") == pid
    assert (await rig.http.post("/api/chat", json=CHAT)).status_code == 503
    # same pod start, key already confirmed: Connect does not ask again and serving resumes
    again = (await rig.admin.post("/admin/api/pod/connect", json={})).json()
    assert again["state"] == "connected"
    assert (await rig.status())["backend"]["state"] == "ready" and await chat_ok(rig)


# -- hostile profile values ----------------------------------------------------------------------


async def test_hostile_profile_values_are_rejected_or_inert(rig):
    await rig.connect()
    base = rig.mgr.profiles.list()[0]["data"]
    for field, value in (("repo", "Qwen/Qwen3.8-27B; touch pwned"), ("repo", "Qwen/$(touch pwned)"),
                         ("dir", "../pwned"), ("dir", "x; touch pwned"), ("revision", "main`touch pwned`")):
        data = json.loads(json.dumps(base))
        data["variants"][0][field] = value
        resp = await rig.admin.post("/admin/api/profiles", json={"name": "evil", "data": data})
        assert resp.status_code == 400 and field in resp.json()["error"], (field, resp.text)
    for name in ("x; touch pwned", "$(touch pwned)", "a'b"):
        resp = await rig.admin.post("/admin/api/profiles", json={"name": name, "data": base})
        assert resp.status_code == 400
    data = json.loads(json.dumps(base))
    data["served_model_name"] = "dev-test\"; touch pwned; \""
    assert (await rig.admin.post("/admin/api/profiles", json={"name": "evil", "data": data})).status_code == 400
    data = json.loads(json.dumps(base))
    data["args"].append({"flag": "--host", "value": "0.0.0.0"})
    resp = await rig.admin.post("/admin/api/profiles", json={"name": "evil", "data": data})
    assert resp.status_code == 400 and "--host" in resp.json()["error"]

    # free-text values may contain shell syntax; they must arrive as literal single arguments
    nasty_arg = "$(touch pwned-arg); `touch pwned-tick`; 'q' \"dq\" && touch pwned-and | tee x"
    nasty_env = "a'; touch pwned-env; ' $(touch pwned-env2) \\"
    data = json.loads(json.dumps(base))
    data["args"].append({"flag": "--chat-template", "value": nasty_arg})
    data["env"] = {"MY_NOTE": nasty_env, "OMP_NUM_THREADS": "4"}
    created = await rig.admin.post("/admin/api/profiles", json={"name": "nasty", "data": data})
    assert created.status_code == 201, created.text
    status = await rig.deploy(profile_id=created.json()["id"])
    assert status["deployment"]["state"] == "ready", status["deployment"]["error"]
    run = rig.pod.records("vllm")[-1]
    assert run["argv"][run["argv"].index("--chat-template") + 1] == nasty_arg
    assert run["env"]["MY_NOTE"] == nasty_env and run["env"]["OMP_NUM_THREADS"] == "4"
    assert run["argv"][-4:] == ["--host", "127.0.0.1", "--port", str(rig.pod.vllm_port)]
    leaked = [str(p) for p in rig.pod.dir.rglob("pwned*")] + [p for p in os.listdir(".") if p.startswith("pwned")]
    assert leaked == []
    assert await chat_ok(rig)


# -- API odds and ends --------------------------------------------------------------------------------


async def test_profile_crud_and_plan_endpoints(rig):
    listing = (await rig.admin.get("/admin/api/profiles")).json()
    assert [p["name"] for p in listing["profiles"]] == ["qwen3.8-27b"] and listing["remote_root"] == str(rig.pod.root)
    pid = listing["profiles"][0]["id"]
    data = listing["profiles"][0]["data"]
    dup = await rig.admin.post("/admin/api/profiles", json={"name": "qwen3.8-27b", "data": data})
    assert dup.status_code == 400 and "already exists" in dup.json()["error"]
    copy = (await rig.admin.post("/admin/api/profiles", json={"name": "copy", "data": data})).json()
    data["context_wanted"] = 100000
    upd = await rig.admin.put(f"/admin/api/profiles/{copy['id']}", json={"name": "copy2", "data": data})
    assert upd.status_code == 200 and upd.json()["name"] == "copy2" and upd.json()["data"]["context_wanted"] == 100000
    assert (await rig.admin.put("/admin/api/profiles/999", json={"name": "x", "data": data})).status_code == 404
    assert (await rig.admin.delete(f"/admin/api/profiles/{copy['id']}")).status_code == 200
    assert (await rig.admin.delete(f"/admin/api/profiles/{copy['id']}")).status_code == 404

    # plan: needs a pod, or a hypothetical configuration
    assert (await rig.admin.get(f"/admin/api/profiles/{pid}/plan")).status_code == 409
    hypo = (await rig.admin.get(f"/admin/api/profiles/{pid}/plan?gpu=L40S&count=2")).json()
    assert hypo["hypothetical"] and hypo["auto"] == "BF16" and hypo["rows"][0]["tensor_parallel"] == 2
    assert "--tensor-parallel-size 2" in hypo["rows"][0]["command"] and "--api-key" not in hypo["rows"][0]["command"]
    assert (await rig.admin.get(f"/admin/api/profiles/{pid}/plan?gpu=RTX%20PRO%206000&count=2")).status_code == 400
    assert (await rig.admin.get(f"/admin/api/profiles/{pid}/plan?gpu=GTX%20970")).status_code == 404
    assert (await rig.admin.get("/admin/api/profiles/999/plan?gpu=L40S")).status_code == 404
    sizing = (await rig.admin.get(f"/admin/api/profiles/{pid}/sizing")).json()
    assert {v["variant"]: v["disk_request_gb"] for v in sizing["variants"]} == {"BF16": 80, "FP8": 55}
    assert any(r["gpu"] == "H100 NVL" and r["count"] == 1 and r["variant"] == "BF16" for r in sizing["rows"])

    await rig.connect()
    live = (await rig.admin.get(f"/admin/api/profiles/{pid}/plan")).json()
    assert not live["hypothetical"] and live["auto"] == "BF16" and live["rows"][0]["context"] == 131072
    assert live["rows"][0]["command"].startswith(f"vllm serve {rig.pod.root}/qwen3.8-27b-bf16 ")

    # the catalog is editable
    catalog = (await rig.admin.get("/admin/api/gpu-catalog")).json()["catalog"]
    assert len(catalog) == 22
    put = await rig.admin.put("/admin/api/gpu-catalog", json={"catalog": [
        {"name": "Future 512", "generation": "Next", "memory_gb": 512, "max_count": 2}]})
    assert put.status_code == 200
    sizing = (await rig.admin.get(f"/admin/api/profiles/{pid}/sizing")).json()
    assert [(r["gpu"], r["count"]) for r in sizing["rows"]] == [("Future 512", 1), ("Future 512", 2)]
    assert (await rig.admin.put("/admin/api/gpu-catalog", json={"catalog": [{"name": "x"}]})).status_code == 400
    reset = await rig.admin.put("/admin/api/gpu-catalog", json={"catalog": None})
    assert len(reset.json()["catalog"]) == 22


async def test_admin_token_is_required(rig):
    from httpx import ASGITransport, AsyncClient

    anon = AsyncClient(transport=ASGITransport(app=rig.app), base_url="http://proxy.test")
    for method, path in (("GET", "/admin/api/pod"), ("POST", "/admin/api/deploy"), ("GET", "/admin/api/profiles"),
                         ("GET", "/admin/api/deploy/events"), ("GET", "/admin/api/logs/stream?source=vllm"),
                         ("PUT", "/admin/api/settings/hf-token")):
        assert (await anon.request(method, path)).status_code == 401, path
        assert (await rig.http.request(method, path)).status_code == 401, path  # an API key is not an admin token
    page = await anon.get("/admin/pod")
    assert page.status_code == 200 and "script-src 'self'" in page.headers["content-security-policy"]
    for name in ("pod.js", "pod.css"):
        assert (await anon.get(f"/admin/static/{name}")).status_code == 200
    await anon.aclose()


async def test_external_mode_disables_pod_management(tmp_path):
    from conftest import Env

    env = Env(tmp_path)  # VLLM_BASE_URL set
    try:
        status = (await env.admin.get("/admin/api/pod")).json()
        assert status["external"] is True and status["backend"]["state"] == "ready"
        for method, path, body in (
            ("PUT", "/admin/api/pod", {"ssh_command": "ssh root@1.2.3.4"}),
            ("POST", "/admin/api/pod/connect", {}),
            ("POST", "/admin/api/pod/disconnect", {}),
            ("POST", "/admin/api/deploy", {"profile_id": 1}),
            ("POST", "/admin/api/deploy/cancel", {}),
            ("POST", "/admin/api/vllm/stop", {}),
            ("POST", "/admin/api/vllm/restart", {}),
            ("PUT", "/admin/api/settings/hf-token", {"token": "hf_xxxxxxxxxxxxxxxxxxxx"}),
            ("POST", "/admin/api/pod/delete-models", {}),
        ):
            resp = await env.admin.request(method, path, json=body)
            assert resp.status_code == 409 and "external mode" in resp.json()["error"], path
        assert env.ctx.backend.state == "ready"
    finally:
        await env.aclose()
