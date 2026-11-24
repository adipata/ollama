"""The fit plan for real and hypothetical GPU configurations (PLAN 12.3)."""

from __future__ import annotations

import httpx
import pytest
from fake_pod.server import GOOD_TOKEN, GPUS, MODEL_CONFIG, REPOS, FakeHF, hf_http_app

from ollama_vllm_proxy.pod import fit
from ollama_vllm_proxy.pod import profiles as prof


def gpus(spec):
    return fit.parse_gpus("\n".join(GPUS[spec]))


async def plan(spec, profile=None, **kw):
    profile = profile or prof.default_profile()
    hf = FakeHF()
    infos = {v["label"]: await hf.fetch(v["repo"]) for v in profile["variants"]}
    return fit.build_plan(profile, infos, gpus(spec) if isinstance(spec, str) else spec, **kw)


def row(p, label):
    return next(r for r in p["rows"] if r["variant"] == label)


def test_parse_nvidia_smi_csv():
    out = fit.parse_gpus("0, NVIDIA RTX PRO 6000 Blackwell, 97887, 12.0\n1, NVIDIA L40S, 46068 MiB, 8.9\n"
                         "garbage\n2, Old Card, 8192, [N/A]\n")
    assert [(g.index, g.name, g.memory_mib, g.compute_cap) for g in out] == [
        (0, "NVIDIA RTX PRO 6000 Blackwell", 97887, 12.0), (1, "NVIDIA L40S", 46068, 8.9), (2, "Old Card", 8192, None)]
    assert fit.parse_gpus("No devices were found\n") == []
    assert fit.parse_gpus("0, NVIDIA RTX PRO 6000 MIG 48GB, 48000, 12.0")[0].mig


def test_kv_cache_cost_per_token():
    assert fit.kv_bytes_per_token(MODEL_CONFIG) == 2 * 64 * 8 * 128 * 2
    assert fit.kv_bytes_per_token(MODEL_CONFIG, "fp8") == 2 * 64 * 8 * 128
    assert fit.kv_bytes_per_token({"num_hidden_layers": 12, "num_attention_heads": 12, "hidden_size": 768,
                                   "torch_dtype": "float32"}) == 2 * 12 * 12 * 64 * 4
    assert fit.kv_bytes_per_token({"text_config": {"num_hidden_layers": 2, "num_attention_heads": 4,
                                                   "hidden_size": 64}}) == 2 * 2 * 4 * 16 * 2
    assert fit.kv_bytes_per_token({}) is None


@pytest.mark.parametrize("spec, variant, tp, context", [
    ("1x96", "BF16", 1, 131072),   # one RTX PRO 6000: full precision, full context
    ("2x48", "BF16", 2, None),     # two 48 GB cards: BF16 across both, context reduced
    ("1x48", "FP8", 1, None),      # one 48 GB card: falls back to FP8
    ("2x24", "FP8", 2, None),
    ("1x80", "BF16", 1, None),
])
async def test_auto_selection(spec, variant, tp, context):
    p = await plan(spec)
    assert p["auto"] == variant
    chosen = row(p, variant)
    assert chosen["fits"] and chosen["tensor_parallel"] == tp
    assert 65536 <= chosen["context"] <= 131072 and chosen["context"] <= chosen["max_context"]
    if context:
        assert chosen["context"] == context
    else:
        assert chosen["context"] < 131072 and chosen["context"] % 1024 == 0


async def test_nothing_fits_a_24gb_card():
    p = await plan("1x24")
    assert p["auto"] is None
    assert "do not fit" in row(p, "BF16")["reason"] and not row(p, "FP8")["fits"]
    assert all(r["forceable"] for r in p["rows"])


async def test_single_48gb_explains_why_bf16_is_out():
    p = await plan("1x48")
    bf16 = row(p, "BF16")
    assert bf16["status"] == "no" and "54.0 GB" in bf16["reason"]
    assert row(p, "FP8")["weights_bytes"] == 29_000_000_000


async def test_ampere_is_unsupported_but_bf16_can_be_forced():
    p = await plan("2xA6000")
    assert p["auto"] is None
    bf16, fp8 = row(p, "BF16"), row(p, "FP8")
    assert "8.6" in bf16["reason"] and bf16["forceable"]
    assert not fp8["forceable"]


async def test_mig_slice_is_single_gpu_and_untested():
    mig = fit.parse_gpus("0, NVIDIA RTX PRO 6000 MIG 48GB, 48000, 12.0\n1, NVIDIA RTX PRO 6000 MIG 48GB, 48000, 12.0")
    p = await plan(mig)
    fp8 = row(p, "FP8")
    assert p["tensor_parallel_options"] == [1]
    assert fp8["tensor_parallel"] == 1 and fp8["status"] == "untested" and fp8["fits"]
    assert p["auto"] == "FP8"


async def test_mixed_gpus_use_the_largest_identical_group():
    p = await plan("mixed")
    chosen = row(p, p["auto"])
    assert chosen["tensor_parallel"] == 2 and chosen["gpu_indices"] == [0, 2] and "L40S" in chosen["gpu_name"]
    assert any("2 of 3" in w for w in chosen["warnings"])


async def test_overrides():
    profile = prof.default_profile()
    info = await FakeHF().fetch("Qwen/Qwen3.8-27B-FP8")
    two = gpus("2x48")
    one_gpu = fit.evaluate_variant(profile, profile["variants"][1], info, two, tensor_parallel=1)
    assert one_gpu["fits"] and one_gpu["gpu_indices"] == [0]
    too_big = fit.evaluate_variant(profile, profile["variants"][1], info, gpus("1x48"), context=131072)
    assert not too_big["fits"] and too_big["forceable"] and "131,072" in too_big["reason"]
    small = fit.evaluate_variant(profile, profile["variants"][1], info, gpus("1x48"), context=32768)
    assert small["fits"] and small["context"] == 32768
    bad_tp = fit.evaluate_variant(profile, profile["variants"][1], info, two, tensor_parallel=4)
    assert not bad_tp["fits"] and not bad_tp["forceable"]


async def test_disk_shortfall_is_reported_with_the_size_to_request():
    p = await plan("1x96", disk_free_bytes=30 * fit.GB)
    bf16 = row(p, "BF16")
    assert not bf16["fits"] and "container disk of at least 80 GB" in bf16["reason"]
    assert bf16["disk_request_gb"] == 80 and row(p, "FP8")["disk_request_gb"] == 55
    # with the model already present and the environment installed, the same disk is enough
    q = await plan("1x96", disk_free_bytes=30 * fit.GB, present={"BF16": 54_020_000_000}, env_installed=True)
    assert row(q, "BF16")["fits"]


async def test_gpu_memory_utilization_changes_the_result():
    profile = prof.default_profile()
    profile["gpu_memory_utilization"] = 0.5
    p = await plan("1x96", profile)
    assert p["auto"] == "FP8"


async def test_unreadable_repo_is_unknown_not_a_guess():
    hf = FakeHF()
    hf.gated.add("Qwen/Qwen3.8-27B")
    profile = prof.default_profile()
    infos = {v["label"]: await hf.fetch(v["repo"]) for v in profile["variants"]}
    p = fit.build_plan(profile, infos, gpus("1x96"))
    assert row(p, "BF16")["status"] == "unknown" and not row(p, "BF16")["forceable"]
    assert p["auto"] == "FP8"


async def test_sizing_table_from_the_catalog():
    profile = prof.default_profile()
    hf = FakeHF()
    infos = {v["label"]: await hf.fetch(v["repo"]) for v in profile["variants"]}
    table = fit.sizing_table(profile, infos, fit.default_catalog())
    by = {(r["gpu"], r["count"]): r for r in table}
    assert by[("RTX PRO 6000", 1)]["variant"] == "BF16" and by[("RTX PRO 6000", 1)]["context"] == 131072
    assert by[("RTX PRO 6000", 1)]["disk_request_gb"] == 80
    assert by[("L40S", 1)]["variant"] == "FP8" and by[("L40S", 2)]["variant"] == "BF16"
    assert by[("L40S", 2)]["tensor_parallel"] == 2
    assert by[("RTX PRO 4000", 1)]["variant"] is None and by[("RTX PRO 4000", 1)]["status"] == "no"
    assert by[("PRO 6000 MIG 48GB", 1)]["status"] == "untested"
    assert ("RTX PRO 6000", 2) not in by and ("PRO 6000 MIG 48GB", 2) not in by  # max per pod respected
    assert ("B300", 8) in by and ("H200 SXM", 2) in by and ("H200 SXM", 4) not in by
    names = {e["name"] for e in fit.default_catalog()}
    assert not names & {"A40", "RTX A6000", "RTX 3090", "MI300X"}  # unsupported generations are left out
    assert len(names) == 22


def test_catalog_validation():
    ok = fit.validate_catalog([{"name": "X1", "generation": "Ada", "memory_gb": 48, "max_count": 4}])
    assert ok[0]["mig"] is False
    for bad in ([], "x", [{"name": "", "memory_gb": 1}], [{"name": "a", "memory_gb": 0}],
                [{"name": "a", "memory_gb": 8, "max_count": 99}],
                [{"name": "a", "memory_gb": 8}, {"name": "A", "memory_gb": 8}],
                [{"name": "<script>", "memory_gb": 8}]):
        with pytest.raises(fit.CatalogError):
            fit.validate_catalog(bad)


async def test_hugging_face_client_reads_sizes_and_config():
    repos = {**REPOS, "Gated/Model": {**REPOS["Tiny/Tiny-1B"], "gated": True}}
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=hf_http_app(repos)))
    hf = fit.HuggingFace(endpoint="http://hf.test", client=client)
    info = await hf.fetch("Qwen/Qwen3.8-27B")
    assert info.ok and info.weight_bytes == 54_000_000_000 and info.total_bytes == 54_020_000_000
    assert info.config["num_hidden_layers"] == 64
    assert (await hf.fetch("Qwen/Qwen3.8-27B", "v1")).ok
    missing = await hf.fetch("No/Such")
    assert missing.error_kind == "notfound"
    gated = await hf.fetch("Gated/Model")
    assert gated.error_kind == "auth" and "no Hugging Face token" in gated.error
    assert (await hf.fetch("Gated/Model", token="hf_wrongwrongwrongwrong")).error_kind == "auth"
    assert (await hf.fetch("Gated/Model", token=GOOD_TOKEN)).ok
    await hf.aclose()


def test_weight_size_prefers_safetensors():
    weights, total = fit.sizes_from_siblings([
        {"rfilename": "model.safetensors", "size": 100}, {"rfilename": "pytorch_model.bin", "size": 100},
        {"rfilename": "README.md", "size": 5}, {"rfilename": "x.safetensors", "lfs": {"size": 50}}])
    assert (weights, total) == (150, 255)
