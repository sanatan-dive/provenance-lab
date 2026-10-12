#!/bin/bash
# Train every detector one at a time under the memory watchdog, then build the summary table.
# Main models first (so the demo works early), ablations after.
#   ./run_all.sh            # everything
#   ./run_all.sh main       # main models only
cd "$(dirname "$0")"
mkdir -p logs
PY=.venv/bin/python
run() {  # run <log-name> <module> [args...]
  local log=logs/$1.log; shift
  echo "=== $(date +%H:%M:%S) start $log"
  $PY tools/memrun.py --limit-gb 3 -- $PY -m "$@" > "$log" 2>&1
  echo "=== $(date +%H:%M:%S) exit $? $(tail -1 "$log")"
}

run image_main  image.train    --epochs 4
run code_main   code_det.train --epochs 3
run text_main   text.train     --epochs 3
run audio_main  audio.train    --epochs 15

if [ "$1" != "main" ]; then
  run image_nosupcon image.train    --epochs 4  --supcon-weight 0 --name image_vit_fft_ce
  run image_nofreq   image.train    --epochs 4  --no-freq-branch  --name image_vit_only_supcon
  run audio_nosupcon audio.train    --epochs 15 --supcon-weight 0 --name audio_cnn_ce
  run code_nosupcon  code_det.train --epochs 3  --supcon-weight 0 --name code_codebert_ce
  run text_nosupcon  text.train     --epochs 3  --supcon-weight 0 --name text_bert_ce
fi

$PY summarize.py
echo "=== $(date +%H:%M:%S) all done"
