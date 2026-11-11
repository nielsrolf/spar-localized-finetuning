#!/bin/bash
# Tick the derisking driver every 5 minutes for up to $1 hours (default 10), logging to loop.log.
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
end=$(( $(date +%s) + ${1:-10} * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "=== $(date +%H:%M) tick" >> niels/sprint-train/derisking/loop.log
  ~/.venv/bin/python niels/sprint-train/derisking/driver.py tick 2>&1 \
    | grep -v -E '^\s*$|\[JudgeRunner\]|^(__init__|_client|files\.py|jobs\.py|utils\.py)' \
    >> niels/sprint-train/derisking/loop.log
  sleep 300
done
