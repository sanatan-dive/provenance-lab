#!/bin/bash
# Retrain HC3 text models on whitespace-normalised data (removes the literal "\n" / line-break shortcut).
cd "$(dirname "$0")"
PY=.venv/bin/python
run() { local log=logs/$1.log; shift; echo "=== $(date +%H:%M:%S) start $log"
  $PY tools/memrun.py --limit-gb 3.5 -- $PY "$@" > "$log" 2>&1; echo "=== $(date +%H:%M:%S) exit $? $(tail -1 "$log")"; }
run text_main_clean     -m text.train --epochs 3
run text_ce_clean       -m text.train --epochs 3 --supcon-weight 0 --name text_bert_ce
run ood_text_clean      ood_text.py
run baselines_clean     baselines.py
echo "=== $(date +%H:%M:%S) textA done"
