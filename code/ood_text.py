"""Out-of-distribution test: essay detectors trained on HC3, evaluated on MAGE without retraining.

MAGE (Li et al., 2024) has text from 10 domains written by humans and 27 LLMs (GPT-3.5, LLaMA,
FLAN-T5, OPT, GLM, BLOOM, ...). Our models only ever saw HC3 ChatGPT answers, so this measures how
accuracy holds up on unseen generators and unseen topics, which is the realistic setting.
    .venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python ood_text.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from text.data import _detok

N_PER_CLASS = 500
OUT = Path("results/ood")


def load_mage(seed=0):
    path = hf_hub_download("yaful/MAGE", "default/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
    d = pq.read_table(path, columns=["text", "src"]).to_pandas()
    d["label"] = d.src.str.contains("_machine_").astype(int)  # 1 = AI (MAGE's own label is 1 = human)
    d["text"] = d.text.map(_detok)
    d = d[d.text.str.split().str.len() >= 20]
    d["domain"] = d.src.str.split("_").str[0]
    d["generator"] = np.where(d.label == 1, d.src.str.replace(r"^.*_machine_[a-z]+_", "", regex=True), "human")
    return pd.concat([g.sample(N_PER_CLASS, random_state=seed) for _, g in d.groupby("label")]).reset_index(drop=True)


def score(y, p):
    return {"accuracy": accuracy_score(y, p >= 0.5), "f1": f1_score(y, p >= 0.5), "auc_roc": roc_auc_score(y, p)}


def main():
    mage = load_mage()
    print("MAGE sample:", mage.label.value_counts().to_dict(), "domains:", mage.domain.nunique(),
          "generators:", mage.generator.nunique() - 1, flush=True)

    # Our model (same code path as the web app)
    from text.predict import predict_text
    p_ours = np.array([predict_text(t, "results/text/text_bert_supcon.pt") for t in mage.text])  # the HC3-only model

    # Baselines trained on the same HC3 training set
    tr = pd.read_parquet("data/text/hc3_tr4000_te1000_s0_train.parquet")
    tfidf = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50000, ngram_range=(1, 2)),
                          LogisticRegression(max_iter=5000, C=4.0)).fit(tr.text, tr.label)
    p_tfidf = tfidf.predict_proba(mage.text)[:, 1]
    from text.features import LMScorer, compute_features
    f_mage = compute_features(list(mage.text), LMScorer())
    f_tr = np.load("data/text/features_hc3_tr4000_te1000_s0_train.npy")
    # Perplexity threshold fitted on HC3 train (lower loss = more AI-like)
    cands = np.quantile(-f_tr[:, 0], np.linspace(0.02, 0.98, 97))
    thr = max(cands, key=lambda t: accuracy_score(tr.label, -f_tr[:, 0] >= t))
    p_ppl = (-f_mage[:, 0] >= thr).astype(float)

    hc3 = json.loads(Path("results/text/text_bert_supcon_metrics.json").read_text())
    base = json.loads(Path("results/baselines/baselines.json").read_text())["text"]
    res = {
        "n": len(mage), "domains": int(mage.domain.nunique()), "generators": int(mage.generator.nunique() - 1),
        "models": {
            "Ours (BERT + forensic + SupCon)": {"hc3": hc3["accuracy"], "mage": score(mage.label, p_ours)},
            "TF-IDF + logistic regression": {"hc3": base["TF-IDF + logistic regression"]["accuracy"], "mage": score(mage.label, p_tfidf)},
            "Perplexity threshold": {"hc3": base["Perplexity threshold"]["accuracy"],
                                     "mage": {**score(mage.label, p_ppl), "auc_roc": roc_auc_score(mage.label, -f_mage[:, 0])}},
        },
    }
    pred = p_ours >= 0.5
    res["ours_by_domain"] = {dom: float((pred[g.index] == g.label).mean()) for dom, g in mage.groupby("domain")}
    ai = mage[mage.label == 1]
    res["ours_ai_recall_by_family"] = {}
    for fam, pat in [("GPT-3.5 / davinci", "gpt-3.5|davinci"), ("LLaMA", r"^\d+B$|llama"), ("OPT", "opt"),
                     ("FLAN-T5 / T0", "flan|t0"), ("GLM / BLOOM / GPT-J / NeoX", "GLM|bloom|gpt_j|neox")]:
        g = ai[ai.generator.str.contains(pat, case=False, regex=True)]
        if len(g):
            res["ours_ai_recall_by_family"][fam] = {"n": int(len(g)), "recall": float(pred[g.index].mean())}
    res["ours_human_specificity"] = float((~pred[mage.label == 0]).mean())

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mage.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
