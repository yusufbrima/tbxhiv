#!/usr/bin/env bash
# Copy the selected MIMIC-CXR-JPG images from PhysioNet's Google Cloud Storage mirror
# (much faster than HTTPS from a GCP VM), in chunks, shrinking each chunk like 03_*.sh.
# One-time setup: link a Google account on physionet.org/settings/cloud, request GCS access
# on the MIMIC-CXR-JPG page, then `gcloud auth login` with that account on this machine.
# The bucket is requester-pays: transfer is billed to GCP_BILLING_PROJECT.
# Usage: bash scripts/mimic/03b_download_images_gcs.sh [parallel_downloads]   (default 32)
set -euo pipefail
source "$(dirname "$0")/env.sh"
PROJECT="${GCP_BILLING_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
[[ -n "$PROJECT" ]] || { echo "Set GCP_BILLING_PROJECT (requester-pays bucket)"; exit 1; }
URLS="$TBMXAI_MIMIC_OUT/selection/image_urls.txt"
[[ -f "$URLS" ]] || { echo "Run scripts/mimic/02_select.sh first"; exit 1; }
WORKERS="${1:-32}"
LOGS="$TBMXAI_MIMIC_OUT/selection/download_logs"
mkdir -p "$LOGS"
CHUNK="${MIMIC_DOWNLOAD_CHUNK:-2000}"
rm -f "$LOGS"/gcs_chunk_*
split -l "$CHUNK" -d -a 4 "$URLS" "$LOGS/gcs_chunk_"
cd "$TBMXAI_REPO"
for chunk in "$LOGS"/gcs_chunk_????; do
  uv run tbmxai mimic fetch-gcs "$chunk" "$MIMIC_DIR" --project "$PROJECT" --workers "$WORKERS"
  if [[ -n "${MIMIC_MAX_SIDE:-}" ]]; then
    uv run tbmxai mimic shrink-images "$chunk" "$MIMIC_DIR" \
        --max-side "$MIMIC_MAX_SIDE" --workers "${MIMIC_SHRINK_WORKERS:-4}"
  fi
  echo "$(date -u +%H:%M:%S) $(basename "$chunk") done; $(du -sh "$MIMIC_DIR" | cut -f1) on disk"
  rm -f "$chunk"
done
echo "Done. Next: bash scripts/mimic/04_build.sh"
