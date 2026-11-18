#!/usr/bin/env bash
# Run the whole MIMIC workflow on a FAKE MIMIC tree (no credentials, ~1 minute).
# Usage: bash scripts/mimic/00_dry_run.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
DRY="${TMPDIR:-/tmp}/tbmxai_mimic_dry_run"
rm -rf "$DRY"
export MIMIC_DIR="$DRY/physionet" TBMXAI_MIMIC_OUT="$DRY/subset" TBMXAI_OUTPUT_ROOT="$DRY/runs"
export MIMIC_N_PATIENTS=150
cd "$TBMXAI_REPO"
uv run tbmxai mimic fake "$MIMIC_DIR" --n-patients 200
uv run tbmxai mimic select configs/mimic/build.yaml
uv run tbmxai mimic build configs/mimic/build.yaml
echo "Dry run finished. Outputs in $DRY"
