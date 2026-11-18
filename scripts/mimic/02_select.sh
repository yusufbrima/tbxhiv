#!/usr/bin/env bash
# Sample the patient subset (patient-level) and write the list of JPGs to download.
# Usage: bash scripts/mimic/02_select.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
uv run tbmxai mimic select configs/mimic/build.yaml ${MIMIC_DUCKDB_MEMORY:+--memory-limit "$MIMIC_DUCKDB_MEMORY"}
echo "Image list: $TBMXAI_MIMIC_OUT/selection/image_urls.txt ($(wc -l < "$TBMXAI_MIMIC_OUT/selection/image_urls.txt") images)"
