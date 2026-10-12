"""CIFAKE (dragonintelligence/CIFAKE-image-dataset) loading.

The HF parquet files are sorted by class (all FAKE rows, then all REAL rows), so a
"first N rows" slice would be single-class. We download the two parquet shards
(~50 MB total), then draw a class-balanced random subset per split with a fixed seed.

Raw ClassLabel names are ['FAKE', 'REAL'] (FAKE=0, REAL=1). Our convention is
1 = AI-generated, 0 = real, so label = 1 if name == 'FAKE' else 0.
"""
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from .features import handcrafted_features, log_spectrum

REPO = "dragonintelligence/CIFAKE-image-dataset"
CACHE = Path("data/image")


def _ai_label_map(table):
    """Map raw label ids to 1 = AI / 0 = real using the ClassLabel names in the parquet metadata."""
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    assert set(names) == {"FAKE", "REAL"}, names
    return np.array([1 if n == "FAKE" else 0 for n in names], dtype=np.int64)


def load_split(split, n, seed=0):
    """Return (images uint8 [n,32,32,3], labels int64 [n]) — balanced, shuffled, cached."""
    out = CACHE / f"cifake_{split}_n{n}_seed{seed}.npz"
    if out.exists():
        d = np.load(out)
        return d["images"], d["labels"]
    path = hf_hub_download(REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset",
                           local_dir=CACHE / "hf")
    pf = pq.ParquetFile(path)
    labels_tbl = pf.read(columns=["label"])  # labels only; image bytes are read per row group below
    labels = _ai_label_map(labels_tbl)[labels_tbl.column("label").to_numpy()]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), n // 2, replace=False) for c in (0, 1)])
    rng.shuffle(idx)
    images = np.empty((len(idx), 32, 32, 3), dtype=np.uint8)
    starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
    for g in np.unique(np.searchsorted(starts, idx, side="right") - 1):
        sel = np.flatnonzero((idx >= starts[g]) & (idx < starts[g + 1]))
        col = pf.read_row_group(int(g), columns=["image"]).column("image")
        for k in sel:
            images[k] = np.asarray(Image.open(io.BytesIO(col[int(idx[k] - starts[g])]["bytes"].as_py()))
                                   .convert("RGB").resize((32, 32)))
    labels = labels[idx]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, images=images, labels=labels)
    return images, labels


class CifakeDataset(Dataset):
    """Precomputes the FFT spectrum and standardized handcrafted features for every image."""

    def __init__(self, images, labels, feat_mean=None, feat_std=None):
        self.pixels = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()  # uint8 [N,3,32,32]
        self.spectrum = torch.from_numpy(log_spectrum(images)).unsqueeze(1)    # float [N,1,32,32]
        feats = handcrafted_features(images)
        if feat_mean is None:
            feat_mean, feat_std = feats.mean(0), feats.std(0) + 1e-6
        self.feat_mean, self.feat_std = feat_mean, feat_std
        self.feats = torch.from_numpy((feats - feat_mean) / feat_std).float()
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"pixels": self.pixels[i], "spectrum": self.spectrum[i], "feats": self.feats[i],
                "label": self.labels[i]}
