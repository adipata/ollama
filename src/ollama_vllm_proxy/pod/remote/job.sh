#!/bin/bash
# job.sh NAME OVP_DIR LOG SECRETS_FILE COMMAND [ARG...]
#
# Runs inside the clean environment made by launch.sh. Loads NAME.vars (profile
# variables) and, when asked, the secrets file, with the shell's own `export`,
# so no value ever appears on a command line. The files are read as NAME=VALUE
# lines and never sourced. Records the command's exit status in NAME.exit.
set -u
name=$1; ovp=$2; logf=$3; secrets=$4; shift 4

load_vars() {
  [ -f "$1" ] || return 0
  local line key
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in *=*) ;; *) continue ;; esac
    key=${line%%=*}
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    export "$key=${line#*=}"
  done < "$1"
}

load_vars "$ovp/$name.vars"
[ -n "$secrets" ] && load_vars "$ovp/$secrets"

"$@" >>"$logf" 2>&1
rc=$?
echo "$rc" > "$ovp/$name.exit.tmp" && mv -f "$ovp/$name.exit.tmp" "$ovp/$name.exit"
exit "$rc"
