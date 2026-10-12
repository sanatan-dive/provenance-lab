#!/bin/bash
# Final experiments, strictly one at a time.
cd "$(dirname "$0")"
PY=.venv/bin/python
run() { local log=logs/$1.log; shift; echo "=== $(date +%H:%M:%S) start $log"
  $PY tools/memrun.py --limit-gb 3.5 -- $PY "$@" > "$log" 2>&1; echo "=== $(date +%H:%M:%S) exit $? $(tail -1 "$log")"; }
run market   market.py
run ood_text ood_text.py
run code_holdout_gpt_supcon -m code_det.train --epochs 3 --holdout-generator gpt --name code_holdout_gpt_supcon
run code_holdout_gpt_ce     -m code_det.train --epochs 3 --holdout-generator gpt --supcon-weight 0 --name code_holdout_gpt_ce
$PY summarize.py > /dev/null
echo "=== $(date +%H:%M:%S) final done"
