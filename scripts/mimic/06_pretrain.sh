#!/usr/bin/env bash
# Supervised multi-label pretraining on the MIMIC selection (14 CheXpert findings).
# Needs: 02_select.sh + images (03_*.sh or 03b_*_gcs.sh). Not 04_build (EHR features).
# Usage: bash scripts/mimic/06_pretrain.sh [backbone ...]   (default: densenet121 convnext_tiny)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
backbones=("$@")
(( ${#backbones[@]} )) || backbones=(densenet121 convnext_tiny)
uv run tbmxai mimic dev-split configs/mimic/build.yaml   # shared train/val/test patients (once)
uv run tbmxai pretrain "configs/pretrain/mimic_chexpert_${backbones[0]}.yaml" --cache-only
for bb in "${backbones[@]}"; do
  echo "== pretraining $bb"
  uv run tbmxai pretrain "configs/pretrain/mimic_chexpert_$bb.yaml" --resume
done
echo "Done → $TBMXAI_OUTPUT_ROOT/pretrain/*/backbone.pt"
