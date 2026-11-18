#!/usr/bin/env bash
# Download ONLY the small files that 02_select.sh needs (< ~100 MB), so selection can run
# while 01_download_tables.sh is still fetching the large EHR tables. Safe to run in
# parallel: both scripts resume/skip files that are already complete.
# Usage: bash scripts/mimic/01a_download_select_inputs.sh   (asks for your PhysioNet password)
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Use ~/.netrc if set up (scripts/mimic/setup_login.sh), otherwise ask for the password.
if grep -qs "^machine physionet.org " "$HOME/.netrc"; then AUTH=(); else
  AUTH=(--user "$PHYSIONET_USER" --ask-password); fi
BASE="https://physionet.org/files"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
{
  for t in patients admissions diagnoses_icd; do
    echo "$BASE/mimiciv/$MIMIC_IV_VERSION/hosp/$t.csv.gz"; done
  for t in edstays diagnosis; do
    echo "$BASE/mimic-iv-ed/$MIMIC_IV_ED_VERSION/ed/$t.csv.gz"; done
  for t in metadata split chexpert; do
    echo "$BASE/mimic-cxr-jpg/$MIMIC_CXR_JPG_VERSION/mimic-cxr-2.0.0-$t.csv.gz"; done
} > "$LIST"
mkdir -p "$MIMIC_DIR"
echo "Downloading the $(wc -l < "$LIST") files needed by 02_select.sh into $MIMIC_DIR"
wget --continue --timestamping --force-directories --no-host-directories --cut-dirs=1 \
     "${AUTH[@]}" \
     --directory-prefix "$MIMIC_DIR" --input-file "$LIST"
echo "Done. Next: bash scripts/mimic/02_select.sh"
