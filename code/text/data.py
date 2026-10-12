"""HC3 (Hello-SimpleAI/HC3, config "all") -> balanced (text, label) subsets, split by question.

Label convention: 1 = AI-generated (ChatGPT answer), 0 = human answer.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

DATA_DIR = Path("data/text")
MIN_WORDS = 20
TEST_QUESTION_FRAC = 0.2


def _detok(s):
    # HC3 human answers (esp. reddit_eli5) are PTB-tokenized ("it 's", "word ,"). Undo the most
    # common artifacts so the model can't separate classes on tokenization spacing alone.
    s = re.sub(r" ([.,!?;:%)\]])", r"\1", s)
    s = re.sub(r"([(\[$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    # HC3 ChatGPT answers carry literal "\n" escapes and real line breaks; human answers were flattened
    # when the dataset was collected. Both are collection artefacts, so normalise all whitespace.
    s = s.replace("\\n", " ")
    return " ".join(s.split())


def _load_rows():
    path = hf_hub_download("Hello-SimpleAI/HC3", "all/train/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet")
    q = pd.read_parquet(path, columns=["id", "source", "human_answers", "chatgpt_answers"])
    q["qid"] = q["source"] + "_" + q["id"].astype(str)
    rows = []
    for qid, source, hum, gpt in zip(q["qid"], q["source"], q["human_answers"], q["chatgpt_answers"]):
        rows += [(qid, source, _detok(t), 0) for t in hum]
        rows += [(qid, source, _detok(t), 1) for t in gpt]
    df = pd.DataFrame(rows, columns=["qid", "source", "text", "label"])
    return df[df["text"].str.split().str.len() >= MIN_WORDS].reset_index(drop=True)


def _balanced(df, n, rng):
    parts = [g.sample(n // 2, random_state=rng) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)


def load_splits(n_train=4000, n_test=1000, seed=0):
    """Return (train_df, test_df) with columns qid, source, text, label. Cached under data/text/."""
    tag = f"hc3_tr{n_train}_te{n_test}_s{seed}"
    tr_p, te_p = DATA_DIR / f"{tag}_train.parquet", DATA_DIR / f"{tag}_test.parquet"
    if tr_p.exists() and te_p.exists():
        return pd.read_parquet(tr_p), pd.read_parquet(te_p), tag
    df = _load_rows()
    rng = np.random.RandomState(seed)
    qids = df["qid"].unique().copy()
    rng.shuffle(qids)
    test_q = set(qids[: int(len(qids) * TEST_QUESTION_FRAC)])
    is_test = df["qid"].isin(test_q)
    train, test = _balanced(df[~is_test], n_train, rng), _balanced(df[is_test], n_test, rng)
    assert not set(train["qid"]) & set(test["qid"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(tr_p)
    test.to_parquet(te_p)
    return train, test, tag
