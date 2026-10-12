"""Simple baselines on exactly the same train/test splits as the deep models.

The point: show what simple, well-known methods already achieve, so the deep models' gain is measured
against something, not against chance.
  - Perplexity threshold (essays): one number per text, the idea behind early AI-text detectors.
  - TF-IDF + logistic regression (essays, code): the classic text-classification baseline.
  - Forensic features + logistic regression (all four): our hand-crafted features without any deep model.
CPU only, runs in seconds from cached features.  .venv/bin/python baselines.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from image.features import handcrafted_features


def score(y, p, thr=0.5):
    return {"accuracy": accuracy_score(y, p >= thr), "f1": f1_score(y, p >= thr), "auc_roc": roc_auc_score(y, p)}


def logreg(xtr, ytr, xte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    return clf.fit(xtr, ytr).predict_proba(xte)[:, 1]


def tfidf(ttr, ytr, tte, **kw):
    clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, **kw),
                        LogisticRegression(max_iter=5000, C=4.0))
    return clf.fit(ttr, ytr).predict_proba(tte)[:, 1]


out = {}

# Essays
tr = pd.read_parquet("data/text/hc3_tr4000_te1000_s0_train.parquet")
te = pd.read_parquet("data/text/hc3_tr4000_te1000_s0_test.parquet")
ftr = np.load("data/text/features_hc3_tr4000_te1000_s0_train.npy")
fte = np.load("data/text/features_hc3_tr4000_te1000_s0_test.npy")
# Perplexity threshold: lower mean token loss = more predictable = more likely AI. Threshold picked on train.
ppl_tr, ppl_te = -ftr[:, 0], -fte[:, 0]
cands = np.quantile(ppl_tr, np.linspace(0.02, 0.98, 97))
thr = max(cands, key=lambda t: accuracy_score(tr.label, ppl_tr >= t))
out["text"] = {
    "Perplexity threshold": {**score(te.label, (ppl_te >= thr).astype(float)), "auc_roc": roc_auc_score(te.label, ppl_te)},
    "TF-IDF + logistic regression": score(te.label, tfidf(tr.text, tr.label, te.text, ngram_range=(1, 2))),
    "Forensic features + logistic regression": score(te.label, logreg(ftr, tr.label, fte)),
}

# Code
tr = pd.read_parquet("data/code/train_12000_s0.parquet")  # same 12k training set as the deployed code model
te = pd.read_parquet("data/code/test_1000_s0.parquet")
ftr, fte = np.load("data/code/feats_train_12000_s0.npy"), np.load("data/code/feats_test_1000_s0.npy")
out["code"] = {
    "TF-IDF (char n-grams) + logistic regression": score(te.label, tfidf(tr.code, tr.label, te.code, analyzer="char_wb", ngram_range=(2, 4))),
    "Forensic features + logistic regression": score(te.label, logreg(ftr, tr.label, fte)),
}

# Images
ztr, zte = np.load("data/image/cifake_train_n6000_seed0.npz"), np.load("data/image/cifake_test_n2000_seed0.npz")
out["image"] = {
    "Forensic features + logistic regression": score(zte["labels"], logreg(handcrafted_features(ztr["images"]), ztr["labels"], handcrafted_features(zte["images"]))),
}

# Voice
mtr, mte = pd.read_parquet("data/audio/train_meta_3000_s0.parquet"), pd.read_parquet("data/audio/test_meta_1000_s0.parquet")
out["audio"] = {
    "Forensic features + logistic regression": score(mte.label, logreg(np.load("data/audio/train_feat_3000_s0.npy"), mtr.label, np.load("data/audio/test_feat_1000_s0.npy"))),
}

out = {m: {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in d.items()} for m, d in out.items()}
Path("results/baselines").mkdir(parents=True, exist_ok=True)
Path("results/baselines/baselines.json").write_text(json.dumps(out, indent=2))
for m, d in out.items():
    for k, v in d.items():
        print(f"{m:6s} {k:45s} acc {v['accuracy']:.3f}  f1 {v['f1']:.3f}  auc {v['auc_roc']:.3f}")
