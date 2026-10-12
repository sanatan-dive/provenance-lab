"""Handcrafted forensic features for text: perplexity, burstiness, stylometry.

Per text (FEATURE_NAMES order):
  ppl_mean      mean token log-loss under distilgpt2
  ppl_std       std of token log-loss
  sent_len_cv   burstiness: std / mean of sentence lengths (words)
  sent_ppl_std  burstiness: std of per-sentence mean log-loss
  ttr           type-token ratio
  avg_word_len  mean characters per word
  punct_rate    punctuation characters per word
  avg_sent_len  mean words per sentence
"""
import re
import string

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.train import get_device

FEATURE_NAMES = ["ppl_mean", "ppl_std", "sent_len_cv", "sent_ppl_std", "ttr", "avg_word_len",
                 "punct_rate", "avg_sent_len"]
LM_NAME = "distilgpt2"
LM_MAX_TOKENS = 512
SENT_RE = re.compile(r"[^.!?\n]+[.!?]*")
PUNCT = set(string.punctuation)


def _sentences(text):
    return [(m.start(), m.end()) for m in SENT_RE.finditer(text) if m.group().split()]


def _stylometry(text, sents):
    words = text.split()
    sent_lens = np.array([len(text[a:b].split()) for a, b in sents] or [len(words)], dtype=float)
    return {
        "sent_len_cv": sent_lens.std() / sent_lens.mean(),
        "ttr": len({w.lower() for w in words}) / len(words),
        "avg_word_len": np.mean([len(w) for w in words]),
        "punct_rate": sum(c in PUNCT for c in text) / len(words),
        "avg_sent_len": sent_lens.mean(),
    }


class LMScorer:
    def __init__(self, device=None):
        self.device = device or get_device()
        self.tok = AutoTokenizer.from_pretrained(LM_NAME)
        self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(LM_NAME).to(self.device).eval()

    @torch.no_grad()
    def token_losses(self, texts):
        """Per text: (array of per-token log-loss, array of each scored token's char start offset)."""
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=LM_MAX_TOKENS, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.lm(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none").cpu()
        mask = enc["attention_mask"][:, 1:].bool().cpu()
        return [(loss[i][mask[i]].numpy(), offsets[i, 1:, 0][mask[i]].numpy()) for i in range(len(texts))]


def compute_features(texts, scorer=None, batch_size=4, verbose=False):
    scorer = scorer or LMScorer()
    out = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        for text, (loss, starts) in zip(batch, scorer.token_losses(batch)):
            sents = _sentences(text)
            sent_ppl = [loss[(starts >= a) & (starts < b)].mean() for a, b in sents
                        if ((starts >= a) & (starts < b)).any()]
            f = _stylometry(text, sents)
            f.update(ppl_mean=loss.mean(), ppl_std=loss.std(),
                     sent_ppl_std=np.std(sent_ppl) if sent_ppl else 0.0)
            out.append([f[k] for k in FEATURE_NAMES])
        if scorer.device.type == "mps":
            torch.mps.empty_cache()  # vocab-sized logits vary in shape per batch; don't let the cache pile up
        if verbose and (s // batch_size) % 100 == 0:
            print(f"features {s + len(batch)}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=np.float32)


def cached_features(splits):
    """splits: list of (texts, cache_path). Loads distilgpt2 only if a cache is missing, then frees it."""
    missing = [(t, p) for t, p in splits if not p.exists()]
    if missing:
        scorer = LMScorer()
        for texts, path in missing:
            np.save(path, compute_features(list(texts), scorer, verbose=True))
        del scorer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return [np.load(p) for _, p in splits]


def standardize(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def fit_stats(train_feats):
    return train_feats.mean(0), train_feats.std(0) + 1e-6
