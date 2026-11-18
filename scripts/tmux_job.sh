#!/usr/bin/env bash
# Run a long job in its own window of the tmux session "tbmxai", so it survives closing the
# laptop / SSH disconnects. Output is also appended to $TBMXAI_LOGS/<name>.log.
#
# Usage:  bash scripts/tmux_job.sh NAME COMMAND [ARGS...]
#   e.g.  bash scripts/tmux_job.sh train bash scripts/mimic/05_train.sh cxr_densenet label_tb
# Watch:  tmux attach -t tbmxai     (Ctrl-b n / Ctrl-b <number> switch windows, Ctrl-b d detach)
set -euo pipefail
[[ $# -ge 2 ]] || { echo "Usage: bash scripts/tmux_job.sh NAME COMMAND [ARGS...]" >&2; exit 2; }
command -v tmux > /dev/null || { echo "tmux is not installed" >&2; exit 1; }
SESSION="${TBMXAI_TMUX_SESSION:-tbmxai}"
LOGS="${TBMXAI_LOGS:-$HOME/data/tbmxai/logs}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="$1"; shift
mkdir -p "$LOGS"
cmd="$(printf '%q ' "$@")"
# keep the window open after the job ends, so its last output stays visible
run="$cmd 2>&1 | tee -a $(printf '%q' "$LOGS/$name.log"); echo; echo '[$name finished]'; exec bash"
if tmux has-session -t "$SESSION" 2> /dev/null; then
    tmux new-window -t "$SESSION" -n "$name" -c "$REPO" "$run"
else
    tmux new-session -d -s "$SESSION" -n "$name" -c "$REPO" "$run"
    tmux set -t "$SESSION" history-limit 20000 > /dev/null
fi
echo "Started '$name' in tmux session '$SESSION' (log: $LOGS/$name.log). Watch: tmux attach -t $SESSION"
