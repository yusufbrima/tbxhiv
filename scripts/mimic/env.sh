# Settings for the MIMIC workflow. Edit the values below once (in any editor).
# Every script in this folder sources this file. Nothing here is committed data.

# Your PhysioNet username (credentialed access + signed DUAs for MIMIC-IV,
# MIMIC-IV-ED, MIMIC-CXR and MIMIC-CXR-JPG are required).
export PHYSIONET_USER="${PHYSIONET_USER:-your_physionet_username}"

# Where the raw PhysioNet downloads go (outside the repo).
export MIMIC_DIR="${MIMIC_DIR:-$HOME/data/physionet}"

# Where the tbmxai-ready subset (features/labels/manifests) is written.
export TBMXAI_MIMIC_OUT="${TBMXAI_MIMIC_OUT:-$HOME/data/tbmxai/mimic_subset}"

# Where experiment runs are written.
export TBMXAI_OUTPUT_ROOT="${TBMXAI_OUTPUT_ROOT:-$HOME/tbmxai_runs}"

# Subset size (patients) and dataset versions.
export MIMIC_N_PATIENTS="${MIMIC_N_PATIENTS:-20000}"
export MIMIC_IV_VERSION="${MIMIC_IV_VERSION:-3.1}"
export MIMIC_IV_ED_VERSION="${MIMIC_IV_ED_VERSION:-2.2}"
export MIMIC_CXR_JPG_VERSION="${MIMIC_CXR_JPG_VERSION:-2.1.0}"
export MIMIC_CXR_VERSION="${MIMIC_CXR_VERSION:-2.1.0}"

# Shrink each downloaded CXR to this longest side (px) right after download (empty = keep the
# originals, ~1.9 MB each). 1024 px keeps headroom over the 256–512 px used for training at
# ~10 % of the size, so a 20,000-patient subset (~82k images) fits on a 250 GB disk.
export MIMIC_MAX_SIDE="${MIMIC_MAX_SIDE-1024}"

# DuckDB memory cap for select/build. Larger joins spill to disk beyond it.
# This VM has 15 GB RAM; raise on bigger machines.
export MIMIC_DUCKDB_MEMORY="${MIMIC_DUCKDB_MEMORY:-10GB}"

# Optional cache of resized CXRs (decoded once, reused every epoch); see 04b_cache_images.sh.
export TBMXAI_IMAGE_CACHE="${TBMXAI_IMAGE_CACHE:-$HOME/data/tbmxai/cxr_cache}"

# Repository root (for `uv run`), resolved from this file's location.
export TBMXAI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
