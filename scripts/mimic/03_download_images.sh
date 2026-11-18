#!/usr/bin/env bash
# Download only the selected MIMIC-CXR-JPG images, with several parallel connections.
# Resumable: re-run to continue; finished files are skipped.
# Usage: bash scripts/mimic/03_download_images.sh [parallel_connections]   (default 4)
set -euo pipefail
source "$(dirname "$0")/env.sh"
# Use ~/.netrc if set up (scripts/mimic/setup_login.sh), otherwise ask for the password.
if grep -qs "^machine physionet.org " "$HOME/.netrc"; then AUTH=(); else
  read -r -s -p "PhysioNet password for $PHYSIONET_USER: " PW; echo
  AUTH=(--user "$PHYSIONET_USER" --password "$PW"); fi
URLS="$TBMXAI_MIMIC_OUT/selection/image_urls.txt"
[[ -f "$URLS" ]] || { echo "Run scripts/mimic/02_select.sh first"; exit 1; }
JOBS="${1:-4}"
LOGS="$TBMXAI_MIMIC_OUT/selection/download_logs"
mkdir -p "$LOGS" "$MIMIC_DIR"

# Only request images that are missing or incomplete. A JPEG is complete when it ends with
# the end-of-image marker FF D9 (an interrupted download is truncated and lacks it).
TODO="$LOGS/todo.txt"
incomplete() {
  python3 - "$URLS" "$MIMIC_DIR" <<'PY'
import sys
from pathlib import Path
urls, root = sys.argv[1], Path(sys.argv[2])
prefix = "https://physionet.org/files/"
for url in Path(urls).read_text().split():
    f = root / url[len(prefix):]
    try:
        with f.open("rb") as fh:
            fh.seek(-2, 2)
            ok = fh.read(2) == b"\xff\xd9"
    except OSError:
        ok = False
    if not ok:
        print(url)
PY
}
incomplete > "$TODO"
echo "$(wc -l < "$TODO") of $(wc -l < "$URLS") images still to download, $JOBS parallel connections"
[[ -s "$TODO" ]] || { echo "All images present."; exit 0; }

# Download in chunks. With MIMIC_MAX_SIDE set (scripts/mimic/env.sh), each finished chunk is
# shrunk in place before the next one starts, so originals never pile up on disk.
CHUNK="${MIMIC_DOWNLOAD_CHUNK:-2000}"
split -l "$CHUNK" -d -a 4 "$TODO" "$LOGS/chunk_"
for chunk in "$LOGS"/chunk_????; do
  rm -f "$LOGS"/part_??
  split -n "l/$JOBS" -d "$chunk" "$LOGS/part_"
  for part in "$LOGS"/part_??; do
    wget --continue --force-directories --no-host-directories --cut-dirs=1 --no-verbose \
         "${AUTH[@]}" --directory-prefix "$MIMIC_DIR" --input-file "$part" \
         >> "$part.log" 2>&1 &
  done
  wait
  if [[ -n "${MIMIC_MAX_SIDE:-}" ]]; then
    (cd "$TBMXAI_REPO" && uv run tbmxai mimic shrink-images "$chunk" "$MIMIC_DIR" \
        --max-side "$MIMIC_MAX_SIDE" --workers "${MIMIC_SHRINK_WORKERS:-4}")
  fi
  echo "$(date -u +%H:%M:%S) chunk $(basename "$chunk") done ($(wc -l < "$chunk") images); $(du -sh "$MIMIC_DIR" | cut -f1) on disk"
  rm -f "$chunk"
done
rm -f "$LOGS"/part_?? "$TODO"
MISSING=$(incomplete | wc -l)
echo "Finished. Missing or incomplete images: $MISSING (re-run this script to retry). Logs: $LOGS"
