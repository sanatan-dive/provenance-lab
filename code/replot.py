"""Redraw every run's ROC, confusion and learning-curve plots in the shared style.

Runs finished before predictions were saved (image and code main models) are re-scored from
their checkpoints on the same test set; the recomputed AUC is checked against the stored one.
    .venv/bin/python replot.py
"""
import glob
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from common.metrics import plot_roc_confusion
from common.train import plot_curves


def rescore(mod):
    if mod == "image":
        from image.predict import predict_image
        z = np.load("data/image/cifake_test_n2000_seed0.npz")
        probs = []
        for img in z["images"]:
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            buf.seek(0)
            probs.append(predict_image(buf))
        return z["labels"], np.array(probs)
    if mod == "code":
        from code_det.predict import predict_code
        d = pd.read_parquet("data/code/test_1000_s0.parquet")
        return d.label.to_numpy(), np.array([predict_code(c) for c in d.code])
    raise ValueError(mod)


MAIN = {"image": "image_vit_fft_supcon", "code": "code_codebert_supcon"}

for f in sorted(glob.glob("results/*/*_metrics.json")):
    m = json.loads(Path(f).read_text())
    if m.get("smoke"):
        continue
    out, name = Path(f).parent, m["name"]
    if m.get("history"):
        plot_curves(m["history"], out, name)
    preds = out / f"{name}_preds.npz"
    if not preds.exists() and MAIN.get(out.name) == name:
        y, p = rescore(out.name)
        auc = roc_auc_score(y, p)
        print(f"{name}: recomputed AUC {auc:.4f} vs stored {m['auc_roc']:.4f}")
        np.savez(preds, y_true=y, y_prob=p)
    if preds.exists():
        z = np.load(preds)
        plot_roc_confusion(z["y_true"], z["y_prob"], m["auc_roc"], out, name)
        print("replotted", name)
