#!/usr/bin/env bash
# Image model, end-to-end fusion and fusion without workup features, three seeds each, on the
# index-radiograph cohort. Needs the MIMIC-pretrained ConvNeXt-T (scripts/mimic/06_pretrain.sh).
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
R="${TBMXAI_OUTPUT_ROOT:-$HOME/tbmxai_runs}"; O="$R/pneumonia"
export TBMXAI_MIMIC_OUT="${TBMXAI_INDEX_COHORT:-$HOME/data/tbmxai/mimic_full_index}" TBMXAI_PROTOCOL=
step() { local name="$1"; shift; echo "=== $(date -u +%H:%M) START $name"; if "$@"; then echo "=== $(date -u +%H:%M) OK $name"; else echo "=== $(date -u +%H:%M) FAILED $name"; fi; }
for m in cxr fusion fusion_noworkup; do
  step "train $m" uv run tbmxai cv "configs/pneumonia/$m.yaml" --folds 0 --seed 1 --seed 2 --seed 3 --output-dir "$O/$m"
  rm -rf -- "$O/$m"/seed*/checkpoints
done
echo "=== $(date -u +%H:%M) train_cxr_fusion done"
