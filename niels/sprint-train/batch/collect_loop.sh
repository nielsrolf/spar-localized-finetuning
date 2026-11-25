#!/bin/bash
# Collect and judge finished evals of the big batch every 10 minutes for up to $1 hours (default 48), log in collect.log.
# Runs next to loop.sh --skip-collect, so that submitting never waits for judging. One collect at a time.
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
hours=${1:-48}
end=$(( $(date +%s) + hours * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  if ! pgrep -f "derisk.py collect" > /dev/null; then
    echo "=== $(date '+%m-%d %H:%M') collect" >> niels/sprint-train/batch/collect.log
    EVAL_HOME=niels/sprint-train/batch caffeinate -is ~/.venv/bin/python niels/sprint-train/derisking/derisk.py collect 2>&1 \
      | grep -v -E '^\s*$|\[JudgeRunner\]|^(__init__|_client|files\.py|jobs\.py|utils\.py)' | cut -c1-300 \
      >> niels/sprint-train/batch/collect.log
  fi
  caffeinate -is sleep 600
done
