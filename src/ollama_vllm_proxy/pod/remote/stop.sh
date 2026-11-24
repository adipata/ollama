#!/bin/bash
# stop.sh NAME OVP_DIR [GRACE_SECONDS]
# Terminates a job started by launch.sh: its whole process group, TERM then KILL.
set -u
name=$1; ovp=$2; grace=${3:-20}
. "${ovp:?}/common.sh"
valid_job "$name" || exit 2
case "$grace" in ''|*[!0-9]*) grace=20 ;; esac
if ! pid=$(job_pid "$name" "$ovp"); then
  echo "$name is not running"
  exit 0
fi
kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
i=0
while pid_alive "$pid" && [ "$i" -lt $((grace * 5)) ]; do sleep 0.2; i=$((i + 1)); done
if pid_alive "$pid"; then
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
  sleep 0.2
fi
# The wrapper died with its group, so record the status for it.
[ -e "$ovp/$name.exit" ] || echo 143 > "$ovp/$name.exit"
echo "stopped $name (pid $pid)"
