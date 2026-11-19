#!/usr/bin/env bash
# EHR XGBoost on the index-radiograph cohort with film-time features
# (scripts/cohort/build_available_features.py): 30-trial search on validation patients, then the
# models with and without the pre-specified workup features. CPU only.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="${TBMXAI_OUTPUT_ROOT:-$HOME/tbmxai_runs}"; O="$R/pneumonia"
export TBMXAI_MIMIC_OUT="${TBMXAI_INDEX_COHORT:-$HOME/data/tbmxai/mimic_full_index_available}" TBMXAI_PROTOCOL=
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; exit 1; fi; }
step "tune ehr" uv run tbmxai tune configs/pneumonia/ehr_xgb_tune.yaml --output-dir "$O/ehr_tune"
step "apply tuned settings" uv run python scripts/experiments/apply_tuned_params.py "$O/ehr_tune/tuning/best_params.json" configs/pneumonia/ehr_xgb.yaml configs/pneumonia/ehr_xgb_noworkup.yaml
for m in ehr_xgb ehr_xgb_noworkup; do
  step "train $m" uv run tbmxai run "configs/pneumonia/$m.yaml" --output-dir "$O/$m"
done
echo "=== $(date -u +%H:%M) train_ehr done"
