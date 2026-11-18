#!/usr/bin/env bash
# Decode + resize all subset CXRs once (short side 512 px) so training does not re-decode
# full-resolution JPEGs every epoch. Optional but strongly recommended (few CPU cores).
# Usage: bash scripts/mimic/04b_cache_images.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
TBMXAI_MIMIC_OUT="${TBMXAI_INDEX_COHORT:-$HOME/data/tbmxai/mimic_full_index}" uv run tbmxai cache-images configs/pneumonia/cxr.yaml
