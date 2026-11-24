#!/bin/bash
# delete_models.sh ROOT DIR...  - removes model directories directly under ROOT
# Only plain names are accepted; the application's own directories are refused.
set -u
root=$1; shift
for d in "$@"; do
  case "$d" in ''|*/*|.*|ovp|venv) echo "skipped $d"; continue ;; esac
  if [ -d "${root:?}/$d" ]; then
    kb=$(du -sk "$root/$d" 2>/dev/null | awk '{print $1}')
    rm -rf -- "${root:?}/${d:?}" && echo "deleted $d ${kb:-0}"
  fi
done
exit 0
