#!/usr/bin/env bash
# Show which MIMIC tables are present and selection/build progress.
# Usage: bash scripts/mimic/status.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$TBMXAI_REPO"
uv run tbmxai mimic status configs/mimic/build.yaml
