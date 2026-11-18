#!/usr/bin/env bash
# Store your PhysioNet login in ~/.netrc (permissions 600, readable only by you) so the
# download scripts run without a password prompt. Run once: bash scripts/mimic/setup_login.sh
set -euo pipefail
read -r -p "PhysioNet username: " USERNAME
read -r -s -p "PhysioNet password (hidden): " PASSWORD; echo
NETRC="$HOME/.netrc"
touch "$NETRC"; chmod 600 "$NETRC"
grep -v "^machine physionet.org " "$NETRC" > "$NETRC.tmp" || true
printf 'machine physionet.org login %s password %s\n' "$USERNAME" "$PASSWORD" >> "$NETRC.tmp"
mv "$NETRC.tmp" "$NETRC"; chmod 600 "$NETRC"
echo "Saved to $NETRC. Download scripts will no longer ask for a password."
