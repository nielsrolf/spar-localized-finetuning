#!/bin/bash
# Collect and judge one eval label (EVAL_HOME picks the batch or derisking); used by collect_loop.sh via xargs.
# Extra arguments go to derisk.py collect (for example --force).
label=$1; shift
~/.venv/bin/python niels/sprint-train/derisking/derisk.py collect --label "$label" "$@" 2>&1 \
  | grep -E '^\{|Error|: \[|restarted|giving up' | cut -c1-200 | sed "s|^|$label |"
