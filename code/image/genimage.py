"""Image detector v2: modern generators at full resolution (Tiny-GenImage).

Tiny-GenImage pairs real ImageNet photos with images from 7 generators (Midjourney, SD 1.5, ADM,
GLIDE, VQDM, Wukong, BigGAN; this release has no SD 1.4 images). Known shortcut in GenImage: real photos are JPEG, generated images PNG,
at different sizes. To stop the model from learning file format, every image is resized (shorter side
256, centre crop 224) and re-encoded as JPEG q=95 -- the same normalisation runs on uploads.

    .venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python -m image.genimage
    .venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python -m image.genimage --holdout Midjourney
"""
import argparse
import glob
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common.train import fit, get_device, predict

from .features import handcrafted_features, spectrum_input
from .model import VIT_NAME, ImageDetector

HF = Path("data/genimage/hf")
CACHE = Path("data/genimage")
OUT = "results/image"
GENS = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]


def normalise(img: Image.Image) -> bytes:
    """Shorter side -> 256, centre crop 224, JPEG q=95. Applied to every train/test image and upload."""
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    buf = io.BytesIO()
    img.crop((l, t, l + 224, t + 224)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def decode(b: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def build(split, per_gen, seed=0):
    """Balanced subset: `per_gen` images from each of the 8 generators + the same total of real photos."""
    path = CACHE / f"{split}_{per_gen}_s{seed}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    files = sorted(glob.glob(str(HF / "default" / split / "*.parquet")))
    meta = []
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["label", "generator"])
        meta += [(fi, r, lab, g) for r, (lab, g) in enumerate(zip(t.column("label").to_pylist(), t.column("generator").to_pylist()))]
    meta = np.array(meta)
    rng = np.random.default_rng(seed)
    pick = []
    for g in range(1, 9):
        idx = np.flatnonzero((meta[:, 2] == 1) & (meta[:, 3] == g))
        pick += list(rng.choice(idx, min(per_gen, len(idx)), replace=False))
    real = np.flatnonzero(meta[:, 2] == 0)
    pick += list(rng.choice(real, len(pick), replace=False))
    want = {(int(meta[i, 0]), int(meta[i, 1])): i for i in pick}
    out = {}
    for fi, f in enumerate(files):
        row = 0
        for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["image"]):
            for im in b.column("image").to_pylist():
                if (fi, row) in want:
                    out[want[(fi, row)]] = normalise(Image.open(io.BytesIO(im["bytes"])))
                row += 1
        print(f"  {split}: {len(out)}/{len(pick)} decoded", flush=True)
    order = rng.permutation(pick)
    data = {"jpeg": [out[i] for i in order], "label": meta[order, 2].astype(np.int64),
            "generator": [GENS[g] for g in meta[order, 3]]}
    path.write_bytes(pickle.dumps(data))
    return data


def feats_of(jpegs, chunk=256):
    return np.concatenate([handcrafted_features(np.stack([decode(b) for b in jpegs[i:i + chunk]]))
                           for i in range(0, len(jpegs), chunk)])


class GenImageDataset(Dataset):
    def __init__(self, data, mean, std, augment=False):
        self.jpeg, self.labels, self.augment = data["jpeg"], data["label"], augment
        self.feats = torch.from_numpy((data["feats"] - mean) / std).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = decode(self.jpeg[i])
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
        return {"pixels": torch.from_numpy(img).permute(2, 0, 1), "spectrum": torch.from_numpy(spectrum_input(img[None])),
                "feats": self.feats[i], "label": torch.tensor(self.labels[i])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gen-train", type=int, default=1000)
    ap.add_argument("--per-gen-test", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--holdout", default=None, help="generator excluded from training (still in test)")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or ("image_genimage" + (f"_no{a.holdout.lower()}" if a.holdout else ""))
    torch.manual_seed(0)

    tr, te = build("train", a.per_gen_train), build("validation", a.per_gen_test)
    for d, tag in ((tr, "train"), (te, "test")):
        fpath = CACHE / f"feats_{tag}_{a.per_gen_train if tag == 'train' else a.per_gen_test}.npy"
        if not fpath.exists():
            np.save(fpath, feats_of(d["jpeg"]))
        d["feats"] = np.load(fpath)
    if a.holdout:  # drop the held-out generator and as many real photos, to stay balanced
        g = np.array(tr["generator"])
        drop = np.flatnonzero(g == a.holdout)
        real = np.flatnonzero(g == "Real")[: len(drop)]
        keep = np.setdiff1d(np.arange(len(g)), np.concatenate([drop, real]))
        tr = {"jpeg": [tr["jpeg"][i] for i in keep], "label": tr["label"][keep],
              "generator": [tr["generator"][i] for i in keep], "feats": tr["feats"][keep]}
    mean, std = tr["feats"].mean(0), tr["feats"].std(0) + 1e-6
    print("train", len(tr["label"]), "test", len(te["label"]), "held out:", a.holdout, flush=True)

    train_dl = DataLoader(GenImageDataset(tr, mean, std, augment=True), batch_size=32, shuffle=True, drop_last=True)
    test_dl = DataLoader(GenImageDataset(te, mean, std), batch_size=32)
    model = ImageDetector(use_handcrafted=True, use_freq=True)
    config = {"vit": VIT_NAME, "use_handcrafted": True, "use_freq": True, "dataset": "Tiny-GenImage",
              "n_train": len(tr["label"]), "n_test": len(te["label"]), "holdout": a.holdout, "batch_size": 32}
    model, _ = fit(model, train_dl, test_dl, out_dir=OUT, name=name, epochs=a.epochs, lr=2e-5,
                   fast_prefixes=("head.", "freq."), supcon_weight=0.5, extra=config)
    torch.save({"state_dict": model.state_dict(), "config": config, "feat_mean": mean, "feat_std": std},
               Path(OUT) / f"{name}.pt")

    y, p = predict(model, test_dl, get_device())
    gens = np.array(te["generator"])
    per = {g: {"n": int((gens == g).sum()), "accuracy": float(((p >= 0.5) == y)[gens == g].mean())} for g in GENS if (gens == g).any()}
    Path(OUT, f"{name}_per_generator.json").write_text(json.dumps(per, indent=2))
    print("per-generator:", {k: round(v["accuracy"], 3) for k, v in per.items()})


if __name__ == "__main__":
    main()
