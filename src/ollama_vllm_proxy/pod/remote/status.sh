#!/bin/bash
# status.sh OVP_DIR -> one line per job: "job NAME PID ALIVE EXIT" ("-" when unknown)
set -u
ovp=$1
. "${ovp:?}/common.sh"
for name in vllm install download helper; do
  pid=-; alive=0; code=-
  [ -s "$ovp/$name.pid" ] && read -r pid < "$ovp/$name.pid"
  case "$pid" in ''|*[!0-9]*) pid=- ;; esac
  [ -s "$ovp/$name.exit" ] && read -r code < "$ovp/$name.exit"
  case "$code" in ''|*[!0-9]*) code=- ;; esac
  if [ "$pid" != - ] && [ "$code" = - ] && pid_alive "$pid"; then alive=1; fi
  echo "job $name $pid $alive $code"
done
