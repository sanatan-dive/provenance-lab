"""Calibrate the diverse essay model's decision threshold on MAGE *validation* data, then apply it to the test set.

The threshold is chosen on 1,000 MAGE validation texts (OPT excluded, so the held-out family stays unseen)
to maximise balanced accuracy. The MAGE test set is touched exactly once, after the threshold is fixed.
    .venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python -m text.calibrate
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from common.train import get_device, predict
from ood_text import load_mage
from text.data import DATA_DIR, _detok, load_splits
from text.features import FEATURE_NAMES, cached_features, standardize
from text.model import ENCODER_NAME, TextDetector
from text.train import make_loader
from text.train_diverse import FAMILIES, HOLDOUT, NAME, RESULT, generator


def mage_val(n=500, seed=0):
    f = hf_hub_download("yaful/MAGE", "default/validation/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
    d = pq.read_table(f, columns=["text", "src"]).to_pandas()
    d["label"] = d.src.str.contains("_machine_").astype(int)
    d = d[~((d.label == 1) & generator(d.src).str.contains(HOLDOUT, case=False))]
    d["text"] = d.text.map(_detok)
    d = d[d.text.str.split().str.len() >= 20]
    return pd.concat([g.sample(n, random_state=seed) for _, g in d.groupby("label")]).reset_index(drop=True)


def main():
    ck = torch.load(f"results/text/{NAME}.pt", map_location="cpu", weights_only=False)
    model = TextDetector(feat_dim=len(FEATURE_NAMES), use_handcrafted=True)
    model.load_state_dict(ck["state_dict"])
    dev = get_device()
    model.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)

    val, test = mage_val(), load_mage()
    _, hc3_te, tag = load_splits(4000, 1000, 0)
    f_val, f_te, f_hc3 = cached_features([(val.text, DATA_DIR / "features_mage_val_1000_noopt_s0.npy"),
                                          (test.text, DATA_DIR / "features_mage_test_1000_s0.npy"),
                                          (hc3_te.text, DATA_DIR / f"features_{tag}_test.npy")])
    probs = {}
    for name, df, f in (("val", val, f_val), ("test", test, f_te), ("hc3", hc3_te, f_hc3)):
        loader = make_loader(df.text, standardize(f, ck["feat_mean"], ck["feat_std"]), df.label, tok, 256, 16, False)
        probs[name] = predict(model, loader, dev)

    yv, pv = probs["val"]
    grid = np.linspace(0.05, 0.99, 95)
    bal = lambda y, p, t: 0.5 * (((p >= t) & (y == 1)).sum() / (y == 1).sum() + ((p < t) & (y == 0)).sum() / (y == 0).sum())
    thr = float(max(grid, key=lambda t: bal(yv, pv, t)))

    yt, pt = probs["test"]
    pred = pt >= thr
    gens = np.where(test.label == 1, generator(test.src), "human")
    fam = {}
    for name, pat in FAMILIES:
        mask = (yt == 1) & pd.Series(gens).str.contains(pat, case=False, regex=True).to_numpy()
        fam[name] = {"n": int(mask.sum()), "recall": float(pred[mask].mean())}
    yh, ph = probs["hc3"]
    res = {"threshold": thr, "chosen_on": "MAGE validation, 1,000 texts, OPT excluded",
           "mage_accuracy": float((pred == yt).mean()), "mage_balanced_accuracy": float(bal(yt, pt, thr)),
           "human_specificity": float((~pred[yt == 0]).mean()), "hc3_accuracy": float(((ph >= thr) == yh).mean()),
           "ai_recall_by_family": fam}
    d = json.loads(RESULT.read_text())
    d["calibrated"] = res
    RESULT.write_text(json.dumps(d, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
