#!/bin/bash
# sizes.sh ROOT DIR... -> "size DIR BYTES MARKER" per directory
# MARKER is repo@revision once the download completed, else "-".
set -u
root=$1; shift
for d in "$@"; do
  case "$d" in ''|*/*|.*) continue ;; esac
  bytes=0; marker=-
  if [ -d "$root/$d" ]; then
    bytes=$(du -sb "$root/$d" 2>/dev/null | awk '{print $1}')
    [ -s "$root/$d/.ovp-download" ] && read -r marker < "$root/$d/.ovp-download"
  fi
  echo "size $d ${bytes:-0} ${marker:--}"
done
exit 0
