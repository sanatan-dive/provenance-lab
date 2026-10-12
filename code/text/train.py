"""Train the text (HC3 essay/answer) AI-content detector.

    .venv/bin/python -m text.train            # full run
    .venv/bin/python -m text.train --smoke    # pipeline check
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common.train import fit
from text.data import DATA_DIR, load_splits
from text.features import FEATURE_NAMES, cached_features, fit_stats, standardize
from text.model import ENCODER_NAME, TextDetector

OUT_DIR = "results/text"


def make_loader(texts, feats, labels, tok, max_len, batch_size, shuffle):
    items = list(zip(texts, feats, labels))

    def collate(b):
        t, f, y = zip(*b)
        # fixed shape keeps the MPS allocator cache from growing across varying batch lengths
        enc = tok(list(t), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "feats": torch.stack([torch.from_numpy(x) for x in f]),
                "label": torch.tensor(y, dtype=torch.int64)}

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--supcon-weight", type=float, default=0.5)
    ap.add_argument("--no-handcrafted", action="store_true")
    ap.add_argument("--name", default="text_bert_supcon")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        a.n_train, a.n_test = 64, 32
    torch.manual_seed(a.seed)

    train, test, tag = load_splits(a.n_train, a.n_test, a.seed)
    print(f"train {len(train)} rows / {train.qid.nunique()} questions, "
          f"test {len(test)} rows / {test.qid.nunique()} questions", flush=True)
    f_tr, f_te = cached_features([(train.text, DATA_DIR / f"features_{tag}_train.npy"),
                                  (test.text, DATA_DIR / f"features_{tag}_test.npy")])
    mean, std = fit_stats(f_tr)
    f_tr, f_te = standardize(f_tr, mean, std), standardize(f_te, mean, std)

    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    tr_loader = make_loader(train.text, f_tr, train.label, tok, a.max_len, a.batch_size, True)
    te_loader = make_loader(test.text, f_te, test.label, tok, a.max_len, a.batch_size, False)
    model = TextDetector(feat_dim=len(FEATURE_NAMES), use_handcrafted=not a.no_handcrafted)
    model, _ = fit(model, tr_loader, te_loader, OUT_DIR, a.name, epochs=a.epochs,
                   supcon_weight=a.supcon_weight, smoke=a.smoke,
                   extra={"modality": "text", "dataset": "HC3/all", "n_train": len(train),
                          "use_handcrafted": not a.no_handcrafted, "max_len": a.max_len})
    if not a.smoke:
        path = Path(OUT_DIR) / f"{a.name}.pt"
        torch.save({"state_dict": model.cpu().state_dict(), "feat_mean": mean, "feat_std": std,
                    "use_handcrafted": not a.no_handcrafted, "max_len": a.max_len}, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
