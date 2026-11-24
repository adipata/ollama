#!/bin/bash
# start_download.sh OVP_DIR VENV REPO DIR [REVISION]
# Downloads one model repository; resumable. HF_TOKEN, when set, comes from the
# environment prepared by job.sh. The Hugging Face variables are set here
# explicitly instead of being inherited from the image.
set -u
ovp=$1; venv=$2; repo=$3; dir=$4; rev=${5:-}
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_TELEMETRY=1
printf '%s@%s\n' "$repo" "$rev" > "$ovp/download.target"
rm -f -- "${dir:?}/.ovp-download"
echo "== $(date -u +%H:%M:%S) hf download $repo -> $dir"
if [ -n "$rev" ]; then
  "$venv/bin/hf" download "$repo" --local-dir "$dir" --revision "$rev" || exit 1
else
  "$venv/bin/hf" download "$repo" --local-dir "$dir" || exit 1
fi
printf '%s@%s\n' "$repo" "$rev" > "$dir/.ovp-download"
echo "== $(date -u +%H:%M:%S) download complete"
