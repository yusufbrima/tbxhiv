#!/usr/bin/env bash
# Build features (all EHR groups), labels, image manifest and reports for the subset.
# Usage: bash scripts/mimic/04_build.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
uv run tbmxai mimic build configs/mimic/build.yaml ${MIMIC_DUCKDB_MEMORY:+--memory-limit "$MIMIC_DUCKDB_MEMORY"}
