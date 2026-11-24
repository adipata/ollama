# Sourced by the other scripts. Arguments are data: nothing here evaluates them.
# Non-interactive SSH sessions get a minimal PATH; add the places uv may live in.
for d in /usr/local/bin "$HOME/.local/bin" "$HOME/.cargo/bin"; do
  case ":$PATH:" in *":$d:"*) ;; *) [ -d "$d" ] && PATH="$PATH:$d" ;; esac
done
export PATH

# job_pid NAME OVP_DIR -> prints the PID if the job's process is alive
job_pid() {
  local pid=""
  [ -s "$2/$1.pid" ] && read -r pid < "$2/$1.pid"
  case "$pid" in ''|*[!0-9]*) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

valid_job() {
  case "$1" in vllm|install|download|helper) return 0 ;; *) echo "unknown job '$1'" >&2; return 1 ;; esac
}
