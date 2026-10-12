"""Train the voice spoof detector.

    .venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python -m audio.train --smoke
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.train import fit, get_device, predict
from .data import AudioDataset, build
from .features import FEAT_DIM
from .model import AudioCNN

OUT = "results/audio"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--supcon-weight", type=float, default=0.5)
    ap.add_argument("--no-handcrafted", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="audio_cnn_supcon")
    a = ap.parse_args()
    if a.smoke:
        a.n_train, a.n_test = 128, 64  # 128 -> 3 train batches after fit()'s 10% val split
    torch.manual_seed(a.seed)

    d = build(a.n_train, a.n_test, a.seed)
    (mtr, ftr, metatr), (mte, fte, metate) = d["train"], d["test"]
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6  # standardize with train stats only
    norm = lambda f: ((f - mu) / sd).astype(np.float32)
    train_dl = DataLoader(AudioDataset(mtr, norm(ftr), metatr.label), batch_size=a.batch_size, shuffle=True)
    test_dl = DataLoader(AudioDataset(mte, norm(fte), metate.label), batch_size=a.batch_size, shuffle=False)

    model = AudioCNN(FEAT_DIM, use_handcrafted=not a.no_handcrafted)
    extra = {"n_train": len(metatr), "n_test": len(metate), "use_handcrafted": not a.no_handcrafted,
             "train_speakers": int(metatr.speaker_id.nunique()), "test_speakers": int(metate.speaker_id.nunique()),
             "lr": a.lr, "batch_size": a.batch_size}
    model, m = fit(model, train_dl, test_dl, OUT, a.name, epochs=a.epochs, lr=a.lr, head_lr=a.lr,
                   supcon_weight=a.supcon_weight, smoke=a.smoke, extra=extra, max_grad_norm=1.0, patience=4)
    if a.smoke:
        return

    torch.save({"state_dict": model.state_dict(), "feat_mean": mu, "feat_std": sd,
                "use_handcrafted": not a.no_handcrafted}, Path(OUT) / f"{a.name}.pt")
    y, p = predict(model, test_dl, get_device())  # test loader is unshuffled -> aligned with metate
    correct = (p >= 0.5).astype(int) == y
    attacks = metate["attack"].fillna("bonafide").to_numpy()
    per = {str(att): {"n": int((attacks == att).sum()), "accuracy": float(correct[attacks == att].mean())}
           for att in np.unique(attacks)}
    (Path(OUT) / f"{a.name}_per_attack.json").write_text(json.dumps(dict(sorted(per.items())), indent=2))
    print("per-attack accuracy:", {k: round(v["accuracy"], 3) for k, v in sorted(per.items())})


if __name__ == "__main__":
    main()
