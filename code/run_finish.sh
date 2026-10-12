#!/bin/bash
# Remaining scale-up steps (restarted after the earlier chain hit its time limit).
cd "$(dirname "$0")"
PY=.venv/bin/python
run() { local log=logs/$1.log; shift; echo "=== $(date +%H:%M:%S) start $log"
  env $ENVV $PY tools/memrun.py --limit-gb 3.5 -- $PY "$@" > "$log" 2>&1; echo "=== $(date +%H:%M:%S) exit $? $(tail -1 "$log")"; }
ENVV="N_MAGE=12000" run text_diverse_12k -m text.train_diverse
ENVV="N_MAGE=12000" run text_calibrate_12k -m text.calibrate
ENVV="" run market_modern market.py
echo "=== $(date +%H:%M:%S) finish done"
