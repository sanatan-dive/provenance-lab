"""Voice detector v2: frozen wav2vec 2.0 features + logistic regression.

Self-supervised speech models capture fine acoustic detail that vocoders get wrong, and their
features are a standard strong baseline for spoof detection. The encoder stays frozen (no fine-tuning),
so only a small linear head is trained, on exactly the same 3,000 / 1,000 speaker-disjoint split.
    .venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python -m audio.w2v
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from common.metrics import evaluate_and_save
from common.train import get_device

from .data import ROOT, decode_rows

MODEL = "facebook/wav2vec2-base"
LAYERS = (3, 6, 9, 12)  # lower layers keep acoustic detail; mean-pool each, then concatenate
NAME = "audio_w2v_lr"
OUT = Path("results/audio")


def train_waves():
    path = ROOT / "train_wave_3000_s0.npy"
    if not path.exists():
        meta = pd.read_parquet(ROOT / "train_meta_3000_s0.parquet")
        _, _, waves = decode_rows(meta, keep_wave=True)  # same decode + silence trim as training
        np.save(path, waves)
    return np.load(path, mmap_mode="r")


class Embedder:
    def __init__(self):
        self.dev = get_device()
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL)
        self.model = Wav2Vec2Model.from_pretrained(MODEL).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, waves, batch=16):
        out = []
        for i in range(0, len(waves), batch):
            x = np.asarray(waves[i:i + batch], dtype=np.float32)
            x = x / 32767 if np.abs(x).max() > 1.5 else x  # int16 caches -> [-1, 1]
            inp = self.fe(list(x), sampling_rate=16000, return_tensors="pt").input_values.to(self.dev)
            hs = self.model(inp, output_hidden_states=True).hidden_states
            out.append(torch.cat([hs[l].mean(1) for l in LAYERS], dim=1).float().cpu().numpy())
            if self.dev.type == "mps":
                torch.mps.empty_cache()
        return np.concatenate(out)


def cached_emb(name, waves, emb):
    path = ROOT / f"w2v_{name}.npy"
    if not path.exists():
        np.save(path, emb(waves))
    return np.load(path)


def main():
    mtr = pd.read_parquet(ROOT / "train_meta_3000_s0.parquet")
    mte = pd.read_parquet(ROOT / "test_meta_1000_s0.parquet")
    wtr, wte = train_waves(), np.load(ROOT / "test_wave_1000_s0.npy", mmap_mode="r")
    emb = Embedder()
    xtr, xte = cached_emb("train_3000", wtr, emb), cached_emb("test_1000", wte, emb)

    # Choose C on a speaker-disjoint split of the training set (test set untouched).
    spk = mtr.speaker_id.unique()
    val_spk = set(np.random.default_rng(0).choice(spk, len(spk) // 5, replace=False))
    va = mtr.speaker_id.isin(val_spk).to_numpy()
    best = max((0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0), key=lambda c: make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=3000)).fit(xtr[~va], mtr.label[~va]).score(xtr[va], mtr.label[va]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000)).fit(xtr, mtr.label)

    p = clf.predict_proba(xte)[:, 1]
    tr_acc = clf.score(xtr, mtr.label)
    m = evaluate_and_save(mte.label.to_numpy(), p, OUT, NAME,
                          extra={"model": MODEL, "layers": list(LAYERS), "C": best, "n_train": len(mtr),
                                 "train_acc_best": tr_acc, "supcon_weight": 0.0, "epochs_run": 1, "best_epoch": 1,
                                 "train_seconds": 0, "smoke": False})
    m["generalization_gap"] = tr_acc - m["accuracy"]
    attacks = mte.attack.fillna("bonafide").to_numpy()
    correct = (p >= 0.5) == mte.label.to_numpy()
    per = {str(a): {"n": int((attacks == a).sum()), "accuracy": float(correct[attacks == a].mean())} for a in np.unique(attacks)}
    (OUT / f"{NAME}_metrics.json").write_text(json.dumps(m, indent=2))
    (OUT / f"{NAME}_per_attack.json").write_text(json.dumps(per, indent=2))
    with open(OUT / f"{NAME}.pkl", "wb") as f:
        pickle.dump(clf, f)
    print({k: m[k] for k in ("accuracy", "f1", "auc_roc", "C", "generalization_gap")})
    print("per-attack:", {k: round(v["accuracy"], 2) for k, v in per.items()})


if __name__ == "__main__":
    main()
