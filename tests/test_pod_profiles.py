"""Profiles: validation, the default profile, the vllm command (PLAN 12.3)."""

from __future__ import annotations

import copy

import pytest

from ollama_vllm_proxy.pod import profiles as prof


def test_default_profile_matches_the_plan():
    name, d = prof.validate(prof.DEFAULT_PROFILE_NAME, prof.default_profile())
    assert name == "qwen3.8-27b"
    assert d["served_model_name"] == "dev-test"
    assert [(v["label"], v["repo"], v["dir"]) for v in d["variants"]] == [
        ("BF16", "Qwen/Qwen3.8-27B", "qwen3.8-27b-bf16"), ("FP8", "Qwen/Qwen3.8-27B-FP8", "qwen3.8-27b-fp8")]
    assert (d["context_wanted"], d["context_min"]) == (131072, 65536)
    assert [(a["flag"], a["value"]) for a in d["args"]] == [
        ("--enable-prefix-caching", ""), ("--enable-auto-tool-choice", ""),
        ("--tool-call-parser", "qwen3_coder"), ("--reasoning-parser", "qwen3")]
    assert d["variants"][0]["args"] == [] and d["variants"][1]["args"] == [{"flag": "--kv-cache-dtype", "value": "fp8"}]
    assert d["gpu_memory_utilization"] is None
    assert (d["vllm_version"], d["python_version"], d["env"]) == ("0.34.0", "3.12", {})


def test_serve_command_for_the_default_profile():
    d = prof.default_profile()
    args = prof.serve_args(d, d["variants"][1], model_dir="/workspace/qwen3.8-27b-fp8", context=98304,
                           tensor_parallel=2)
    assert args == [
        "serve", "/workspace/qwen3.8-27b-fp8", "--enable-prefix-caching", "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--kv-cache-dtype", "fp8",
        "--served-model-name", "dev-test", "--max-model-len", "98304", "--tensor-parallel-size", "2",
        "--host", "127.0.0.1", "--port", "8000"]
    assert "--api-key" not in args  # the key travels in the environment, never on the command line
    d["gpu_memory_utilization"] = 0.92
    assert "--gpu-memory-utilization" in prof.serve_args(d, d["variants"][0], model_dir="/w/m", context=4096,
                                                         tensor_parallel=1)


@pytest.mark.parametrize("flag", sorted(prof.OWNED_FLAGS))
def test_application_owned_flags_cannot_be_given(flag):
    d = prof.default_profile()
    d["args"].append({"flag": flag, "value": "0.0.0.0"})
    with pytest.raises(prof.ProfileError) as err:
        prof.validate("p", d)
    assert flag in str(err.value)
    d = prof.default_profile()
    d["variants"][0]["args"] = [{"flag": flag, "value": "x"}]
    with pytest.raises(prof.ProfileError):
        prof.validate("p", d)


def _with(path, value):
    d = prof.default_profile()
    target = d
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return d


@pytest.mark.parametrize("path, value", [
    (["served_model_name"], "dev test"),
    (["served_model_name"], "dev;id"),
    (["served_model_name"], "$(id)"),
    (["variants", 0, "repo"], "Qwen/Qwen3.8-27B; rm -rf /"),
    (["variants", 0, "repo"], "Qwen/$(touch x)"),
    (["variants", 0, "repo"], "../../etc/passwd"),
    (["variants", 0, "repo"], "'Qwen/Q'"),
    (["variants", 0, "repo"], "--local-dir/x"),
    (["variants", 0, "revision"], "main; id"),
    (["variants", 0, "revision"], "--force"),
    (["variants", 0, "dir"], "../outside"),
    (["variants", 0, "dir"], "/etc"),
    (["variants", 0, "dir"], "a b"),
    (["variants", 0, "dir"], "ovp"),
    (["variants", 0, "dir"], "venv"),
    (["variants", 0, "dir"], "x`id`"),
    (["variants", 0, "label"], "B F"),
    (["variants", 1, "label"], "bf16"),  # duplicate (case-insensitive)
    (["variants", 1, "dir"], "qwen3.8-27b-bf16"),  # duplicate directory
    (["variants"], []),
    (["args"], [{"flag": "--host=0.0.0.0", "value": ""}]),
    (["args"], [{"flag": "--Enable", "value": ""}]),
    (["args"], [{"flag": "-p", "value": "1"}]),
    (["args"], [{"flag": "--x; id", "value": ""}]),
    (["args"], [{"flag": "--enable-prefix-caching", "value": "--host"}]),  # a value that would be read as an option
    (["args"], [{"flag": "--seed", "value": "-x"}]),
    (["args"], [{"flag": "--seed", "value": "1\n2"}]),
    (["args"], [{"flag": "--seed", "value": "1"}, {"flag": "--seed", "value": "2"}]),
    (["args"], "--host 0.0.0.0"),
    (["variants", 1, "args"], [{"flag": "--reasoning-parser", "value": "x"}]),  # already a common argument
    (["context_wanted"], 100),
    (["context_wanted"], "131072"),
    (["context_min"], 200000),
    (["gpu_memory_utilization"], 1.5),
    (["vllm_version"], "0.34.0; id"),
    (["vllm_version"], "latest"),
    (["python_version"], "3.12 --seed"),
    (["env"], {"lower": "x"}),
    (["env"], {"VLLM_API_KEY": "mine"}),
    (["env"], {"LD_PRELOAD": "/tmp/x.so"}),
    (["env"], {"PATH": "/tmp"}),
    (["env"], {"RUNPOD_API_KEY": "x"}),
    (["env"], {"A": "line1\nB=2"}),
    (["env"], {"A=B": "x"}),
])
def test_invalid_fields_are_rejected(path, value):
    with pytest.raises(prof.ProfileError):
        prof.validate("p", _with(path, value))


@pytest.mark.parametrize("name", ["", "a b", "x/y", "-lead", "n" * 65, "a;b", None, 5])
def test_invalid_names(name):
    with pytest.raises(prof.ProfileError):
        prof.validate(name, prof.default_profile())


def test_values_with_shell_characters_are_kept_literal():
    # Free-text values are allowed to contain anything printable: they are passed as single arguments.
    d = prof.default_profile()
    d["args"].append({"flag": "--chat-template", "value": "$(touch /tmp/x); `id` 'q' \"d\""})
    d["env"] = {"MY_NOTE": "a'; touch /tmp/y; '"}
    _, clean = prof.validate("p", copy.deepcopy(d))
    args = prof.serve_args(clean, clean["variants"][0], model_dir="/w/m", context=4096, tensor_parallel=1)
    assert "$(touch /tmp/x); `id` 'q' \"d\"" in args
    # even so, the owned options are the last ones
    assert args[-10:] == ["--served-model-name", "dev-test", "--max-model-len", "4096", "--tensor-parallel-size",
                          "1", "--host", "127.0.0.1", "--port", "8000"]


def test_serve_args_refuses_bad_input_even_without_validation():
    d = prof.default_profile()
    with pytest.raises(prof.ProfileError):
        prof.serve_args(d, d["variants"][0], model_dir="/w/m", context=4096, tensor_parallel=3)
    d["args"].append({"flag": "--port", "value": "1"})
    with pytest.raises(prof.ProfileError):
        prof.serve_args(d, d["variants"][0], model_dir="/w/m", context=4096, tensor_parallel=1)


@pytest.mark.parametrize("gpu, count, variant, context, label", [
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 1, 0, 131072, "rtx-pro-6000x1-qwen38-27b-bf16-131072"),
    ("NVIDIA L40S", 2, 0, 112640, "l40sx2-qwen38-27b-bf16-112640"),
    ("NVIDIA RTX A6000", 2, 1, 65536, "rtx-a6000x2-qwen38-27b-fp8-65536"),
    ("NVIDIA H100 80GB HBM3", 1, 1, 131072, "h100-80gb-hbm3x1-qwen38-27b-fp8-131072"),
])
def test_deployment_label(gpu, count, variant, context, label):
    d = prof.default_profile()
    assert prof.deployment_label(gpu, count, d["variants"][variant], context) == label


def test_fp8_detection():
    d = prof.default_profile()
    assert not prof.is_fp8(d, d["variants"][0]) and prof.is_fp8(d, d["variants"][1])
    assert prof.kv_cache_dtype(d, d["variants"][1]) == "fp8" and prof.kv_cache_dtype(d, d["variants"][0]) == "auto"
