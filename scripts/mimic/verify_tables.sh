#!/usr/bin/env bash
# Check every downloaded MIMIC table for corruption (gzip -t; zip CRC check via Python).
# A corrupt file must be deleted and downloaded again (01_download_tables.sh).
# Usage: bash scripts/mimic/verify_tables.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
bad=0; n=0
while IFS= read -r -d '' f; do
    n=$((n + 1))
    case "$f" in
        *.gz) gzip -t "$f" 2> /dev/null ;;
        *.zip) python3 -c "import sys, zipfile; sys.exit(zipfile.ZipFile(sys.argv[1]).testzip() is not None)" "$f" > /dev/null 2>&1 ;;
    esac || { echo "CORRUPT  $f"; bad=$((bad + 1)); }
done < <(find "$MIMIC_DIR" \( -name '*.csv.gz' -o -name '*.zip' \) -print0)
if (( bad )); then
    echo "$bad of $n files are corrupt: delete them and re-run scripts/mimic/01_download_tables.sh"
    exit 1
fi
echo "All $n table files OK"
