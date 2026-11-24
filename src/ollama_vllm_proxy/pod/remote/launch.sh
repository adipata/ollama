#!/bin/bash
# launch.sh NAME OVP_DIR LOG SECRETS_FILE -- COMMAND [ARG...]
#
# Starts COMMAND detached (setsid nohup), so it survives the SSH session and a
# proxy restart, with a clean whitelisted environment: the pod's own variables
# (RUNPOD_API_KEY, JUPYTER_PASSWORD, ...) are not inherited. Secrets are not
# passed here; job.sh reads them from a 0600 file inside the new process.
# Writes OVP_DIR/NAME.pid. Exit 3: the job is already running.
set -u
name=$1; ovp=$2; logf=$3; secrets=$4; shift 4
[ "${1:-}" = "--" ] && shift
. "${ovp:?}/common.sh"
valid_job "$name" || exit 2
case "$secrets" in ''|vllm.env|hf.env) ;; *) echo "bad secrets file" >&2; exit 2 ;; esac
if pid=$(job_pid "$name" "$ovp") && [ ! -e "$ovp/$name.exit" ]; then
  echo "$name is already running (pid $pid)"
  exit 3
fi
rm -f "${ovp:?}/${name:?}.exit" "${ovp:?}/${name:?}.pid"
[ -f "$logf" ] && mv -f "$logf" "$logf.1"
: > "$logf"

root=$(dirname "$ovp")
envargs=("PATH=$PATH" "HOME=${HOME:-/root}" "LANG=C.UTF-8" "HF_HOME=$root/.hf")
[ -n "${LD_LIBRARY_PATH:-}" ] && envargs+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
while IFS= read -r -d '' kv; do
  case "$kv" in NVIDIA_*|CUDA_*) envargs+=("$kv") ;; esac
done < <(env -0)

setsid nohup env -i "${envargs[@]}" /bin/bash "$ovp/job.sh" "$name" "$ovp" "$logf" "$secrets" "$@" \
  </dev/null >/dev/null 2>&1 &
echo $! > "$ovp/$name.pid"
echo "started $name (pid $!)"
