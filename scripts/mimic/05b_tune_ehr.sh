#!/usr/bin/env bash
# Optuna search for the EHR XGBoost model (scored on dev_split val), then refit + test once.
# Usage: bash scripts/mimic/05b_tune_ehr.sh [target ...]   (default: label_pneumonia label_hiv_pneumonia)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
targets=("$@")
(( ${#targets[@]} )) || targets=(label_pneumonia label_hiv_pneumonia)
for t in "${targets[@]}"; do
  echo "== tuning EHR XGBoost: $t"
  TARGET="$t" uv run tbmxai tune configs/mimic/tabular_xgb.yaml \
      --output-dir "$TBMXAI_OUTPUT_ROOT/mimic_ehr_tuned/xgb_$t"
done
