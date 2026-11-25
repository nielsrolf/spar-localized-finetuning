#!/bin/bash
# Collect and judge finished evals of the big batch, 8 evals in parallel, every 5 minutes for up to $1 hours
# (default 48). Log: collect.log. Runs next to loop.sh --skip-collect, so that submitting never waits for judging.
cd "$(dirname "$0")/../../.."
set -a; source .env; set +a
export EVAL_HOME=niels/sprint-train/batch
hours=${1:-48}
end=$(( $(date +%s) + hours * 3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  if ! pgrep -f "derisk.py collect" > /dev/null; then
    python3 -c "
import json, os
for label in json.load(open('$EVAL_HOME/state.json')):
    if not os.path.exists(f'$EVAL_HOME/evals/{label}/summary.json'): print(label)" \
    | xargs -P 8 -I{} sh -c 'caffeinate -is ~/.venv/bin/python niels/sprint-train/derisking/derisk.py collect --label "{}" 2>&1 | grep -E "^\{|Error|: \[|restarted|giving up" | cut -c1-200 | sed "s|^|{} |"' \
      >> niels/sprint-train/batch/collect.log
    echo "=== $(date '+%m-%d %H:%M') collect pass done" >> niels/sprint-train/batch/collect.log
  fi
  caffeinate -is sleep 300
done
