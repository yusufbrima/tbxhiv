#!/usr/bin/env bash
# EHR XGBoost (tuned settings) with and without the pre-specified workup features, on the
# index-radiograph cohort (scripts/cohort/build_index_cohort.py). CPU only.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="${TBMXAI_OUTPUT_ROOT:-$HOME/tbmxai_runs}"; O="$R/pneumonia"
export TBMXAI_MIMIC_OUT="${TBMXAI_INDEX_COHORT:-$HOME/data/tbmxai/mimic_full_index}" TBMXAI_PROTOCOL=
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
for m in ehr_xgb ehr_xgb_noworkup; do
  step "train $m" uv run tbmxai run "configs/pneumonia/$m.yaml" --output-dir "$O/$m"
done
echo "=== $(date -u +%H:%M) train_ehr done"
