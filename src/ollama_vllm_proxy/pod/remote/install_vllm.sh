#!/bin/bash
# install_vllm.sh VENV PYTHON_VERSION VLLM_VERSION
# Creates the environment with the image's uv and installs vLLM into it.
# The download tools go in first so the model download can start in parallel.
# --no-cache: uv would otherwise keep a second 9 GB copy in its cache.
set -u
venv=$1; pyver=$2; version=$3
step() { echo "== $(date -u +%H:%M:%S) $*"; }

if [ ! -x "$venv/bin/python" ]; then
  step "creating $venv (Python $pyver)"
  uv venv "$venv" --python "$pyver" || exit 1
fi
step "installing huggingface_hub and hf_transfer"
uv pip install --no-cache --python "$venv/bin/python" huggingface_hub hf_transfer || exit 1
step "installing vllm==$version (about 9 GB)"
uv pip install --no-cache --python "$venv/bin/python" "vllm==$version" || exit 1
step "verifying"
"$venv/bin/vllm" --version || exit 1
"$venv/bin/python" - <<'PY'
from importlib import metadata
parts = []
for name in ("vllm", "torch", "transformers", "huggingface_hub"):
    try:
        parts.append(f"{name}={metadata.version(name)}")
    except Exception:
        parts.append(f"{name}=unknown")
print("OVP_VERSIONS " + " ".join(parts))
PY
step "done"
