#!/bin/bash
# start_helper.sh OVP_DIR PORT VLLM_LOG  - GPU helper on loopback, with the system python3
set -u
exec python3 "$1/gpu_exporter.py" --host 127.0.0.1 --port "$2" --vllm-log "$3"
