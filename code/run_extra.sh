#!/bin/bash
# Unseen-generator experiment: train code detectors without GPT-written code, test on all generators.
cd "$(dirname "$0")"
PY=.venv/bin/python
run() { local log=logs/$1.log; shift; echo "=== $(date +%H:%M:%S) start $log"
  $PY tools/memrun.py --limit-gb 3 -- $PY -m "$@" > "$log" 2>&1; echo "=== $(date +%H:%M:%S) exit $? $(tail -1 "$log")"; }
run code_holdout_gpt_supcon code_det.train --epochs 3 --holdout-generator gpt --name code_holdout_gpt_supcon
run code_holdout_gpt_ce     code_det.train --epochs 3 --holdout-generator gpt --supcon-weight 0 --name code_holdout_gpt_ce
run ood_text ood_text
run replot replot
$PY summarize.py
echo "=== $(date +%H:%M:%S) extra done"
