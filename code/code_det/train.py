"""Train the source-code AI detector.

    .venv/bin/python -m code_det.train [--smoke] [--no-handcrafted] ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.train import fit, get_device, predict

from .data import DATA_DIR, load_split
from .features import extract_all, fit_stats, standardize
from .model import ENCODER, CodeDetector

OUT_DIR = Path("results/code")


def cached_feats(split, df, n, seed):
    path = DATA_DIR / f"feats_{split}_{n}_s{seed}.npy"
    if not path.exists():
        np.save(path, extract_all(df["code"].tolist()))
    return np.load(path)


def make_loader(df, feats, tok, max_len, batch_size, shuffle):
    rows = list(zip(df["code"].tolist(), feats, df["label"].tolist()))

    def collate(batch):
        codes, f, y = zip(*batch)
        enc = tok(list(codes), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "feats": torch.tensor(np.stack(f)), "label": torch.tensor(y, dtype=torch.int64)}

    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def per_generator(model, loader, gens, smoke):
    y, p = predict(model, loader, get_device(), max_batches=2 if smoke else None)
    gens = np.asarray(gens)[: len(y)]
    pred = (p >= 0.5).astype(int)
    out = {}
    for g in sorted(set(gens)):
        m = gens == g
        # human: accuracy = specificity; AI generator: accuracy = recall on that generator
        out[str(g)] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--supcon-weight", type=float, default=0.5)
    ap.add_argument("--no-handcrafted", action="store_true")
    ap.add_argument("--name", default="code_codebert_supcon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-generator", default=None,
                    help="drop this LLM from training (test still contains it): unseen-generator test")
    a = ap.parse_args()
    if a.smoke:
        a.n_train, a.n_test = 64, 32
    torch.manual_seed(a.seed)

    train_df = load_split("train", a.n_train, a.seed)
    test_df = load_split("test", a.n_test, a.seed)
    print("train generators:", train_df["model"].value_counts().to_dict(), flush=True)
    print("test generators:", test_df["model"].value_counts().to_dict(), flush=True)

    tr_raw = cached_feats("train", train_df, a.n_train, a.seed)
    te_raw = cached_feats("test", test_df, a.n_test, a.seed)
    if a.holdout_generator:
        # Remove the held-out LLM from training, and drop as many human rows to stay balanced.
        keep = (train_df["model"] != a.holdout_generator).to_numpy().copy()
        n_drop = int((~keep).sum())
        human_idx = np.flatnonzero(train_df["label"].to_numpy() == 0)
        keep[np.random.default_rng(a.seed).choice(human_idx, n_drop, replace=False)] = False
        train_df, tr_raw = train_df[keep].reset_index(drop=True), tr_raw[keep]
        print(f"holdout {a.holdout_generator}: train now", train_df["model"].value_counts().to_dict(), flush=True)
    mean, std = fit_stats(tr_raw)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    train_loader = make_loader(train_df, standardize(tr_raw, mean, std), tok, a.max_len, a.batch_size, True)
    test_loader = make_loader(test_df, standardize(te_raw, mean, std), tok, a.max_len, a.batch_size, False)

    model = CodeDetector(use_handcrafted=not a.no_handcrafted)
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # common.train.plot_curves does not create it
    extra = {"n_train": len(train_df), "use_handcrafted": not a.no_handcrafted, "max_len": a.max_len,
             "holdout_generator": a.holdout_generator}
    model, _ = fit(model, train_loader, test_loader, out_dir=str(OUT_DIR), name=a.name, epochs=a.epochs,
                   supcon_weight=a.supcon_weight, smoke=a.smoke, extra=extra)

    name = a.name + ("_smoke" if a.smoke else "")
    pg = per_generator(model, test_loader, test_df["model"].tolist(), a.smoke)
    (OUT_DIR / f"{name}_per_generator.json").write_text(json.dumps(pg, indent=2))
    print("per-generator:", pg, flush=True)

    if not a.smoke:
        torch.save({"state_dict": model.state_dict(), "feat_mean": mean, "feat_std": std,
                    "use_handcrafted": not a.no_handcrafted, "max_len": a.max_len},
                   OUT_DIR / f"{a.name}.pt")


if __name__ == "__main__":
    main()
