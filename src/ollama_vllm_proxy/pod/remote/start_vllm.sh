#!/bin/bash
# start_vllm.sh VENV ARG...   (ARG... is everything after `vllm`, built by the proxy)
# The API key is not an argument: vLLM reads VLLM_API_KEY from the environment,
# so it never shows in `ps` or in the log's argument dump.
set -u
venv=$1; shift
# The image's open-file limit is 1024, too low for a server with many streams.
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
if [ -z "${VLLM_API_KEY:-}" ]; then
  echo "refusing to start vLLM without VLLM_API_KEY" >&2
  exit 78
fi
exec "$venv/bin/vllm" "$@"
