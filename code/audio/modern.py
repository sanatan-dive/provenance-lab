"""Voice detector v3: add modern TTS / voice cloning (ElevenLabs, Amazon Polly, Hume AI, ...) to ASVspoof.

Data: garystafford/deepfake-audio-detection v4 -- 933 real clips cut from 14 YouTube recordings and
933 synthetic clips from 96 generations on 6 platforms. Clips from one recording or one generation are
near-duplicates, so the split is by group (recording id / generation id), never by clip.

Reports:
  1. the ASVspoof-only wav2vec model (audio/w2v.py) on the modern test set: how much does it drop?
  2. a model trained on ASVspoof + modern train groups, tested on both test sets.
    .venv/bin/python tools/memrun.py --limit-gb 3.5 -- .venv/bin/python -m audio.modern
"""
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.metrics import evaluate_and_save

from .data import ROOT
from .features import decode
from .w2v import Embedder, cached_emb

OUT = Path("results/audio")
NAME = "audio_w2v_modern"


def load_modern(seed=0, test_frac=0.25):
    path = ROOT / "modern_s0.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    d = snapshot_download("garystafford/deepfake-audio-detection", repo_type="dataset", local_dir="data/voice_modern")
    rows = []
    for f in sorted(glob.glob(f"{d}/data/*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["audio", "label"]):
            for a, lab in zip(b.column("audio").to_pylist(), b.column("label").to_pylist()):
                w = decode(a["bytes"])  # 16 kHz mono, silence trimmed, first 4 s (same as ASVspoof)
                rows.append((a["path"], "_".join(a["path"].split("_")[:2]), a["path"].split("_")[0], lab,
                             np.clip(w * 32767, -32768, 32767).astype(np.int16)))
    meta = pd.DataFrame([r[:4] for r in rows], columns=["file", "group", "source", "label"])
    waves = np.stack([r[4] for r in rows])
    rng = np.random.default_rng(seed)
    test_groups = set()
    for lab in (0, 1):  # per class, add random groups until ~25% of that class's clips are in test
        g = meta[meta.label == lab].groupby("group").size()
        order, n = rng.permutation(g.index), 0
        for grp in order:
            if n >= test_frac * g.sum():
                break
            test_groups.add(grp)
            n += g[grp]
    meta["split"] = np.where(meta.group.isin(test_groups), "test", "train")
    out = {"meta": meta, "waves": waves}
    path.write_bytes(pickle.dumps(out))
    return out


def fit_lr(x, y, groups):
    """C chosen on a group-disjoint validation split of the training data."""
    ug = np.unique(groups)
    val = set(np.random.default_rng(0).choice(ug, max(1, len(ug) // 5), replace=False))
    va = np.isin(groups, list(val))
    best = max((0.003, 0.01, 0.03, 0.1, 0.3), key=lambda c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
               .fit(x[~va], y[~va]).score(x[va], y[va]))
    return make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=3000)).fit(x, y), best


def main():
    mod = load_modern()
    meta, waves = mod["meta"], mod["waves"]
    tr, te = (meta.split == "train").to_numpy(), (meta.split == "test").to_numpy()
    print("modern:", meta.groupby(["split", "label"]).size().to_dict(), "test groups:",
          sorted(meta[te].group.unique()), flush=True)
    emb = Embedder()
    x_mod = cached_emb("modern_all", waves, emb)
    asv_tr = pd.read_parquet(ROOT / "train_meta_3000_s0.parquet")
    asv_te = pd.read_parquet(ROOT / "test_meta_1000_s0.parquet")
    x_asv_tr, x_asv_te = np.load(ROOT / "w2v_train_3000.npy"), np.load(ROOT / "w2v_test_1000.npy")

    # 1. Existing ASVspoof-only model on modern fakes
    with open(OUT / "audio_w2v_lr.pkl", "rb") as f:
        old = pickle.load(f)
    p_old = old.predict_proba(x_mod[te])[:, 1]
    y_mod = meta.label.to_numpy()

    # 2. ASVspoof + modern training groups
    x = np.concatenate([x_asv_tr, x_mod[tr]])
    y = np.concatenate([asv_tr.label.to_numpy(), y_mod[tr]])
    groups = np.concatenate(["asv_" + asv_tr.speaker_id.astype(str).to_numpy(), meta.group.to_numpy()[tr]])
    clf, c = fit_lr(x, y, groups)
    p_mod, p_asv = clf.predict_proba(x_mod[te])[:, 1], clf.predict_proba(x_asv_te)[:, 1]

    y_all = np.concatenate([asv_te.label.to_numpy(), y_mod[te]])
    m = evaluate_and_save(y_all, np.concatenate([p_asv, p_mod]), OUT, NAME,
                          extra={"C": c, "n_train": len(y), "train_acc_best": clf.score(x, y), "supcon_weight": 0.0,
                                 "epochs_run": 1, "best_epoch": 1, "train_seconds": 0, "smoke": False,
                                 "dataset": "ASVspoof 2019 + modern TTS (ElevenLabs, Polly, Hume, ...)"})
    m["generalization_gap"] = m["train_acc_best"] - m["accuracy"]
    (OUT / f"{NAME}_metrics.json").write_text(json.dumps(m, indent=2))
    src = meta.source.to_numpy()[te]
    res = {
        "modern_test_n": int(te.sum()),
        "old_model_on_modern": float(((p_old >= 0.5) == y_mod[te]).mean()),
        "new_model_on_modern": float(((p_mod >= 0.5) == y_mod[te]).mean()),
        "new_model_on_asvspoof": float(((p_asv >= 0.5) == asv_te.label.to_numpy()).mean()),
        "old_model_on_asvspoof": json.loads((OUT / "audio_w2v_lr_metrics.json").read_text())["accuracy"],
        "by_source_new": {s: float(((p_mod >= 0.5) == y_mod[te])[src == s].mean()) for s in np.unique(src)},
        "by_source_old": {s: float(((p_old >= 0.5) == y_mod[te])[src == s].mean()) for s in np.unique(src)},
    }
    Path("results/ood/voice_modern.json").write_text(json.dumps(res, indent=2))
    with open(OUT / f"{NAME}.pkl", "wb") as f:
        pickle.dump(clf, f)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
