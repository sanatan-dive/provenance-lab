"""Run a command under a memory watchdog.

    .venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python -m text.train --smoke

Kills the whole process tree if its RSS exceeds --limit-gb, or if the Mac's available
memory drops below --min-free-gb. Also caps PyTorch MPS (GPU) memory via env vars.
"""
import argparse
import os
import subprocess
import sys
import time

import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--limit-gb", type=float, default=3.0)
ap.add_argument("--min-free-gb", type=float, default=1.2)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

env = dict(os.environ)
env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.35")  # hard cap on MPS allocations (fraction of RAM)
env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")  # must be <= high ratio
env.setdefault("TOKENIZERS_PARALLELISM", "false")
env.setdefault("OMP_NUM_THREADS", "4")

proc = subprocess.Popen(cmd, env=env)
p = psutil.Process(proc.pid)
peak = 0.0
while proc.poll() is None:
    try:
        rss = sum(c.memory_info().rss for c in [p] + p.children(recursive=True)) / 1e9
    except psutil.Error:
        rss = 0.0
    peak = max(peak, rss)
    free = psutil.virtual_memory().available / 1e9
    if rss > a.limit_gb or free < a.min_free_gb:
        print(f"\n[memrun] KILLED: rss={rss:.2f}GB (limit {a.limit_gb}), system available={free:.2f}GB "
              f"(min {a.min_free_gb})", file=sys.stderr, flush=True)
        for c in p.children(recursive=True) + [p]:
            try:
                c.kill()
            except psutil.Error:
                pass
        sys.exit(137)
    time.sleep(0.5)
print(f"[memrun] exit={proc.returncode} peak_rss={peak:.2f}GB", file=sys.stderr, flush=True)
sys.exit(proc.returncode)
