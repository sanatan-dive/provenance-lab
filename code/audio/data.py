"""ASVspoof 2019 LA (eval partition, published on HF as split `test`) -> balanced, speaker-disjoint subset.

Memory-safe two-pass build (shards are ~500 MB each and are never loaded whole):
  pass 1: read only `label` + `notes` per shard -> pick rows (balanced, speakers split train/test)
  pass 2: stream `audio` in small batches, decode only the selected rows -> log-mel + features
Caches (float16) live in data/audio/. Attack ids (A07..A19) come from the official CM protocol file.
Label convention: 1 = spoof (AI), 0 = bonafide (human). The dataset's ClassLabel is the same.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .features import FEAT_DIM, decode, wave_to_inputs

ROOT = Path("data/audio")
RAW = ROOT / "raw"
N_SHARDS = 9
SHARD_URL = ("https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/resolve/"
             "refs%2Fconvert%2Fparquet/default/test/{:04d}.parquet")
PROTOCOL = ROOT / "ASVspoof2019.LA.cm.eval.trl.txt"
PROTOCOL_URL = ("https://huggingface.co/datasets/Nemez1z/asvspoof-2019-la/resolve/main/"
                "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")


def _download(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {dest.name} ...", flush=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def shard_path(k):
    return _download(SHARD_URL.format(k), RAW / f"{k}.parquet")


def attack_ids():
    _download(PROTOCOL_URL, PROTOCOL)
    p = pd.read_csv(PROTOCOL, sep=" ", header=None, names=["spk", "utt", "_", "attack", "key"])
    return dict(zip(p.utt, p.attack.replace("-", "bonafide")))


def scan(k):
    """Pass 1: label + notes only for shard k."""
    rows, pf = [], pq.ParquetFile(shard_path(k))
    for b in pf.iter_batches(batch_size=1024, columns=["label", "notes"]):
        for lab, note in zip(b.column("label").to_pylist(), b.column("notes").to_pylist()):
            n = json.loads(note)
            rows.append((k, len(rows), n["utterance_id"], n["speaker_id"], lab))
    return pd.DataFrame(rows, columns=["shard", "row", "utt", "speaker_id", "label"])


def choose(meta, n_train, n_test, seed):
    """Speaker-disjoint split, then a balanced 50/50 sample per side. Returns None if too few bonafide."""
    rng = np.random.default_rng(seed)
    spk = np.array(meta.speaker_id.unique(), dtype=object)
    rng.shuffle(spk)
    bona = meta[meta.label == 0].speaker_id.value_counts()
    test_spk, nb = [], 0
    for s in spk:  # fill the test side with whole speakers until it has enough bonafide
        if nb >= n_test // 2:
            break
        test_spk.append(s)
        nb += bona.get(s, 0)
    is_test = meta.speaker_id.isin(test_spk)
    out = {}
    for split, part, n in (("train", meta[~is_test], n_train), ("test", meta[is_test], n_test)):
        picks = []
        for lab, k in ((0, n // 2), (1, n - n // 2)):
            pool = part[part.label == lab]
            if len(pool) < k:
                return None
            picks.append(pool.sample(k, random_state=int(rng.integers(1 << 31))))
        out[split] = pd.concat(picks).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def decode_rows(sel, keep_wave=False):
    """Pass 2: stream audio in small batches; decode only the selected rows (kept in `sel` order)."""
    n = len(sel)
    mels = np.zeros((n, 80, 401), np.float16)
    feats = np.zeros((n, FEAT_DIM), np.float32)
    waves = np.zeros((n, 64000), np.int16) if keep_wave else None
    pos = {(s, r): i for i, (s, r) in enumerate(zip(sel.shard, sel.row))}
    done = 0
    for k in sorted(sel.shard.unique()):
        row = 0
        for b in pq.ParquetFile(shard_path(k)).iter_batches(batch_size=64, columns=["audio"]):
            for a in b.column("audio").to_pylist():
                i = pos.get((k, row))
                row += 1
                if i is None:
                    continue
                w = decode(a["bytes"])
                m, f = wave_to_inputs(w)
                mels[i], feats[i] = m, f
                if keep_wave:
                    waves[i] = np.clip(w * 32767, -32768, 32767).astype(np.int16)
                done += 1
                if done % 500 == 0:
                    print(f"  decoded {done}/{n}", flush=True)
    assert done == n, f"decoded {done} of {n} selected rows"
    return mels, feats, waves


def build(n_train=3000, n_test=1000, seed=0):
    """Returns {split: (mel f16 [N,80,401], feats f32 [N,88], meta DataFrame)}; cached after first call."""
    tag = lambda split, kind, ext: ROOT / f"{split}_{kind}_{n_train if split == 'train' else n_test}_s{seed}.{ext}"
    if all(tag(s, k, e).exists() for s in ("train", "test") for k, e in
           (("mel", "npy"), ("feat", "npy"), ("meta", "parquet"))):
        return {s: (np.load(tag(s, "mel", "npy")), np.load(tag(s, "feat", "npy")),
                    pd.read_parquet(tag(s, "meta", "parquet"))) for s in ("train", "test")}
    attacks = attack_ids()
    metas, sel = [], None
    for k in range(N_SHARDS):  # add shards until there is enough bonafide on both sides of the split
        metas.append(scan(k))
        sel = choose(pd.concat(metas, ignore_index=True), n_train, n_test, seed)
        if sel is not None:
            break
    assert sel is not None, "not enough bonafide even with all shards"
    print(f"using shards 0..{k}", flush=True)
    out = {}
    for split, df in sel.items():
        df["attack"] = df.utt.map(attacks)
        print(f"{split}: {len(df)} rows, {df.speaker_id.nunique()} speakers, "
              f"labels {df.label.value_counts().to_dict()}", flush=True)
        mels, feats, waves = decode_rows(df, keep_wave=split == "test")  # test waves: web demo samples
        np.save(tag(split, "mel", "npy"), mels)
        np.save(tag(split, "feat", "npy"), feats)
        df.to_parquet(tag(split, "meta", "parquet"))
        if waves is not None:
            np.save(tag(split, "wave", "npy"), waves)
        out[split] = (mels, feats, df)
    tr, te = set(out["train"][2].speaker_id), set(out["test"][2].speaker_id)
    assert not tr & te, "speaker leakage"
    return out


class AudioDataset(Dataset):
    def __init__(self, mels, feats, labels):
        self.mels, self.feats = mels, torch.from_numpy(feats)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"mel": torch.from_numpy(self.mels[i].astype(np.float32))[None],
                "feats": self.feats[i], "label": self.labels[i]}
