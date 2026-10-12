#!/bin/bash
# Scale-up runs, strictly one at a time, after the GenImage jobs.
cd "$(dirname "$0")"
PY=.venv/bin/python
run() { local log=logs/$1.log; shift; echo "=== $(date +%H:%M:%S) start $log"
  env $ENVV $PY tools/memrun.py --limit-gb 3.5 -- $PY "$@" > "$log" 2>&1; echo "=== $(date +%H:%M:%S) exit $? $(tail -1 "$log")"; }
ENVV="" run code_12k -m code_det.train --n-train 12000 --n-test 1000 --epochs 3 --name code_codebert_supcon_12k
ENVV="N_MAGE=12000" run text_diverse_12k -m text.train_diverse
ENVV="N_MAGE=12000" run text_calibrate_12k -m text.calibrate
ENVV="" run market_modern market.py
echo "=== $(date +%H:%M:%S) scale done"
