#!/bin/bash
# preflight.sh ROOT VENV PYTHON_VERSION VLLM_PORT HELPER_PORT
# Prints key=value facts about the pod; decisions are made by the proxy.
set -u
root=$1; venv=$2; pyver=$3; vport=$4; hport=$5
. "${root:?}/ovp/common.sh"

if command -v nvidia-smi >/dev/null 2>&1; then
  if out=$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader,nounits 2>&1); then
    echo "nvidia_smi=ok"
    printf '%s\n' "$out" | while IFS= read -r line; do echo "gpu=$line"; done
  else
    echo "nvidia_smi=error"
    printf '%s\n' "$out" | head -n 3 | while IFS= read -r line; do echo "nvidia_smi_error=$line"; done
  fi
else
  echo "nvidia_smi=missing"
fi

echo "disk_avail_kb=$(df -Pk "$root" 2>/dev/null | awk 'NR==2 {print $4}')"
echo "shm_avail_kb=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print $4}')"
mem=unknown
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  [ -r "$f" ] && { read -r mem < "$f"; break; }
done
echo "mem_limit=$mem"
echo "ulimit_n=$(ulimit -n)"

if uvp=$(command -v uv 2>/dev/null); then
  echo "uv=$uvp"
  if uv python find "$pyver" >/dev/null 2>&1; then echo "python=ok"; else echo "python=missing"; fi
else
  echo "uv=missing"
  echo "python=unknown"
fi

port_state() {
  if (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; then echo busy; else echo free; fi
}
echo "port_vllm=$(port_state "$vport")"
echo "port_helper=$(port_state "$hport")"

net() {
  if curl -sS -o /dev/null -m 10 -I "$1" >/dev/null 2>&1; then echo ok; else echo fail; fi
}
tmp_hf="$root/ovp/.net_hf.$$"; tmp_pypi="$root/ovp/.net_pypi.$$"
net https://huggingface.co > "$tmp_hf" &
net https://pypi.org/simple/ > "$tmp_pypi" &
wait
echo "net_huggingface=$(cat "$tmp_hf")"
echo "net_pypi=$(cat "$tmp_pypi")"
rm -f -- "$tmp_hf" "$tmp_pypi"

ver=
for d in "$venv"/lib/python*/site-packages/vllm-*.dist-info; do
  [ -d "$d" ] || continue
  ver=${d##*/vllm-}; ver=${ver%.dist-info}
done
echo "venv_vllm=$ver"
if [ -x "$venv/bin/hf" ]; then echo "venv_hf=ok"; else echo "venv_hf=missing"; fi

# Only these six pod variables are recorded; never the API key or passwords.
for v in RUNPOD_POD_ID RUNPOD_DC_ID RUNPOD_GPU_NAME RUNPOD_GPU_COUNT RUNPOD_MEM_GB RUNPOD_CPU_COUNT; do
  [ -n "${!v:-}" ] && echo "runpod_$v=${!v}"
done
exit 0
