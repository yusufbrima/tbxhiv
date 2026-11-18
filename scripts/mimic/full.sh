#!/usr/bin/env bash
# Build the full eligible MIMIC cohort (configs/mimic/build_full.yaml): select (inheriting the
# development sample's patient split) -> copy missing images from PhysioNet's GCS mirror
# (requester-pays, billed to the gcloud project) -> build EHR features with a DuckDB memory cap.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
source scripts/mimic/env.sh
export TBMXAI_MIMIC_OUT="$HOME/data/tbmxai/mimic_full" MIMIC_N_PATIENTS=100000
export MIMIC_DUCKDB_MEMORY="${MIMIC_DUCKDB_MEMORY:-6GB}"
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; exit 1; fi; }
step "select" uv run tbmxai mimic select configs/mimic/build_full.yaml --memory-limit "$MIMIC_DUCKDB_MEMORY"
echo "images listed: $(wc -l < "$TBMXAI_MIMIC_OUT/selection/image_urls.txt")"
step "download images (GCS)" bash scripts/mimic/03b_download_images_gcs.sh 32
step "build" uv run tbmxai mimic build configs/mimic/build_full.yaml --memory-limit "$MIMIC_DUCKDB_MEMORY"
echo "=== $(date -u +%H:%M) full cohort done"
