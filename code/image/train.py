"""Train the dual-branch image detector on CIFAKE.

    .venv/bin/python -m image.train --smoke
    .venv/bin/python -m image.train  # 6000 train / 2000 test, 2 epochs
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.train import fit

from .data import CifakeDataset, load_split
from .model import VIT_NAME, ImageDetector

OUT = "results/image"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--supcon-weight", type=float, default=0.5)
    ap.add_argument("--no-handcrafted", action="store_true")
    ap.add_argument("--no-freq-branch", action="store_true")
    ap.add_argument("--name", default="image_vit_fft_supcon")
    args = ap.parse_args()
    if args.smoke:  # 3 train batches + 2 test batches at batch size 32
        args.n_train, args.n_test = 96, 64

    torch.manual_seed(0)
    train_ds = CifakeDataset(*load_split("train", args.n_train))
    test_ds = CifakeDataset(*load_split("test", args.n_test), train_ds.feat_mean, train_ds.feat_std)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    use_hc, use_freq = not args.no_handcrafted, not args.no_freq_branch
    model = ImageDetector(use_handcrafted=use_hc, use_freq=use_freq)
    # Pretrained ViT trains at lr=2e-5; the from-scratch FreqCNN and head train at head_lr.
    config = {"vit": VIT_NAME, "use_handcrafted": use_hc, "use_freq": use_freq,
              "n_train": len(train_ds), "n_test": len(test_ds), "batch_size": args.batch_size}
    model, _ = fit(model, train_dl, test_dl, out_dir=OUT, name=args.name, epochs=args.epochs,
                   lr=2e-5, fast_prefixes=("head.", "freq."), supcon_weight=args.supcon_weight, smoke=args.smoke, extra=config)

    if not args.smoke:
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "feat_mean": train_ds.feat_mean, "feat_std": train_ds.feat_std},
                   Path(OUT) / f"{args.name}.pt")


if __name__ == "__main__":
    main()
