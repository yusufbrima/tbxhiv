#!/usr/bin/env bash
# Download the MIMIC metadata + EHR tables needed by tbmxai (~8–12 GB compressed).
# Usage: bash scripts/mimic/01_download_tables.sh      (asks for your PhysioNet password once)
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Use ~/.netrc if set up (scripts/mimic/setup_login.sh), otherwise ask for the password.
if grep -qs "^machine physionet.org " "$HOME/.netrc"; then AUTH=(); else
  AUTH=(--user "$PHYSIONET_USER" --ask-password); fi
BASE="https://physionet.org/files"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
{
  for t in patients admissions diagnoses_icd labevents d_labitems omr prescriptions microbiologyevents \
           transfers services procedures_icd; do
    echo "$BASE/mimiciv/$MIMIC_IV_VERSION/hosp/$t.csv.gz"; done
  for t in icustays chartevents d_items; do
    echo "$BASE/mimiciv/$MIMIC_IV_VERSION/icu/$t.csv.gz"; done
  for t in edstays triage vitalsign diagnosis; do
    echo "$BASE/mimic-iv-ed/$MIMIC_IV_ED_VERSION/ed/$t.csv.gz"; done
  for t in metadata split chexpert negbio; do
    echo "$BASE/mimic-cxr-jpg/$MIMIC_CXR_JPG_VERSION/mimic-cxr-2.0.0-$t.csv.gz"; done
  echo "$BASE/mimic-cxr/$MIMIC_CXR_VERSION/mimic-cxr-reports.zip"
} > "$LIST"
mkdir -p "$MIMIC_DIR"
echo "Downloading $(wc -l < "$LIST") files into $MIMIC_DIR as $PHYSIONET_USER"
wget --continue --timestamping --force-directories --no-host-directories --cut-dirs=1 \
     "${AUTH[@]}" \
     --directory-prefix "$MIMIC_DIR" --input-file "$LIST"
echo "Done. Check with: bash scripts/mimic/status.sh"
