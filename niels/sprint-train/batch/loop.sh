#!/bin/bash
# Tick the big batch every 10 minutes for up to $1 hours (default 48), logging to loop.log. Extra args go to batch.py tick.
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
hours=${1:-48}; shift
end=$(( $(date +%s) + hours * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "=== $(date '+%m-%d %H:%M') tick" >> niels/sprint-train/batch/loop.log
  caffeinate -is ~/.venv/bin/python niels/sprint-train/batch/batch.py tick "$@" 2>&1 \
    | grep -v -E '^\s*$|\[JudgeRunner\]|^(__init__|_client|files\.py|jobs\.py|utils\.py)' \
    >> niels/sprint-train/batch/loop.log
  caffeinate -is sleep 600
done
