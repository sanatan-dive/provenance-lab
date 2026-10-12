"""Fix for the out-of-distribution failure: train the essay detector on diverse data.

The HC3-only model learned "ChatGPT style" and dropped to ~57% on MAGE (many LLMs, many topics).
This trains the same model on HC3 + a MAGE *training* subset that covers many generators and domains,
with one whole generator family (OPT) held out, then evaluates on:
  - the same 1,000 MAGE *test* texts used by ood_text.py (per family; OPT never seen in training)
  - the HC3 test set
    .venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python -m text.train_diverse
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from common.train import fit, get_device, predict
from ood_text import load_mage
from text.data import DATA_DIR, _detok, load_splits
from text.features import FEATURE_NAMES, cached_features, fit_stats, standardize
from text.model import ENCODER_NAME, TextDetector
from text.train import make_loader

HOLDOUT = "opt"  # every OPT model (opt_125m ... opt_30b, opt_iml) is excluded from training
N_MAGE = int(os.environ.get("N_MAGE", 4000))  # MAGE training texts; the 4,000 run is the original
SUFFIX = "" if N_MAGE == 4000 else f"_{N_MAGE // 1000}k"
NAME = f"text_bert_diverse{SUFFIX}"
RESULT = Path(f"results/ood/mage_diverse{SUFFIX}.json")
FAMILIES = [("GPT-3.5 / davinci", "gpt-3.5|davinci"), ("LLaMA", r"^\d+B$|llama"), ("OPT (held out)", "opt"),
            ("FLAN-T5 / T0", "flan|t0"), ("GLM / BLOOM / GPT-J / NeoX", "GLM|bloom|gpt_j|neox")]


def generator(src):
    return src.str.replace(r"^.*_machine_[a-z]+_", "", regex=True)


def mage_train(seed=0):
    path = DATA_DIR / f"mage_train_{N_MAGE}_no{HOLDOUT}_s{seed}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    f = hf_hub_download("yaful/MAGE", "default/train/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
    # Memory-safe: pick rows from the small `src` column, then stream `text` and keep only those rows.
    src = pq.read_table(f, columns=["src"]).column("src").to_pandas()
    meta = pd.DataFrame({"src": src, "label": src.str.contains("_machine_").astype(int)})
    meta = meta[~((meta.label == 1) & generator(meta.src).str.contains(HOLDOUT, case=False))]
    # oversample 1.5x so rows dropped by the 20-word filter can be replaced
    pick = pd.concat([g.sample(int(N_MAGE * 0.75), random_state=seed) for _, g in meta.groupby("label")])
    want, rows, pos = set(pick.index), {}, 0
    for b in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["text"]):
        for t in b.column("text").to_pylist():
            if pos in want:
                rows[pos] = _detok(t)
            pos += 1
    pick = pick.assign(text=pd.Series(rows))
    pick = pick[pick.text.str.split().str.len() >= 20]
    out = pd.concat([g.head(N_MAGE // 2) for _, g in pick.groupby("label")])
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)[["text", "label", "src"]]
    out.to_parquet(path)
    return out


def main():
    torch.manual_seed(0)
    hc3_tr, hc3_te, tag = load_splits(4000, 1000, 0)
    mg_tr = mage_train()
    assert not generator(mg_tr[mg_tr.label == 1].src).str.contains(HOLDOUT, case=False).any()
    test = load_mage()  # same 1,000 MAGE test texts as ood_text.py
    print("MAGE train generators:", generator(mg_tr[mg_tr.label == 1].src).nunique(), "| held out:", HOLDOUT, flush=True)

    train = pd.concat([hc3_tr[["text", "label"]], mg_tr[["text", "label"]]], ignore_index=True)
    f_hc3, f_mg, f_te, f_hc3te = cached_features([
        (hc3_tr.text, DATA_DIR / f"features_{tag}_train.npy"),
        (mg_tr.text, DATA_DIR / f"features_mage_train_{N_MAGE}_no{HOLDOUT}_s0.npy"),
        (test.text, DATA_DIR / "features_mage_test_1000_s0.npy"),
        (hc3_te.text, DATA_DIR / f"features_{tag}_test.npy")])
    f_tr = np.concatenate([f_hc3, f_mg])
    mean, std = fit_stats(f_tr)

    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    tr_loader = make_loader(train.text, standardize(f_tr, mean, std), train.label, tok, 256, 16, True)
    te_loader = make_loader(test.text, standardize(f_te, mean, std), test.label, tok, 256, 16, False)
    model = TextDetector(feat_dim=len(FEATURE_NAMES), use_handcrafted=True)
    model, m = fit(model, tr_loader, te_loader, "results/text", NAME, epochs=2, supcon_weight=0.5,
                   extra={"modality": "text", "dataset": "HC3 + MAGE-train (no OPT); tested on MAGE test",
                          "n_train": len(train), "use_handcrafted": True, "max_len": 256})
    torch.save({"state_dict": model.cpu().state_dict(), "feat_mean": mean, "feat_std": std,
                "use_handcrafted": True, "max_len": 256}, Path("results/text") / f"{NAME}.pt")

    dev = get_device()
    model.to(dev)
    y, p = predict(model, te_loader, dev)
    pred = p >= 0.5
    gens = np.where(test.label == 1, generator(test.src), "human")
    fam = {}
    for name, pat in FAMILIES:
        mask = (test.label.to_numpy() == 1) & pd.Series(gens).str.contains(pat, case=False, regex=True).to_numpy()
        if mask.any():
            fam[name] = {"n": int(mask.sum()), "recall": float(pred[mask].mean())}
    hc3_loader = make_loader(hc3_te.text, standardize(f_hc3te, mean, std), hc3_te.label, tok, 256, 16, False)
    yh, ph = predict(model, hc3_loader, dev)
    res = {"mage_accuracy": m["accuracy"], "mage_auc": m["auc_roc"], "hc3_accuracy": float(((ph >= 0.5) == yh).mean()),
           "human_specificity": float((~pred[test.label.to_numpy() == 0]).mean()), "ai_recall_by_family": fam,
           "holdout_family": HOLDOUT}
    RESULT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
