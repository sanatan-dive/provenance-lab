"""Inference: predict_code(src) -> probability the code is AI-generated."""
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoTokenizer

from common.train import get_device

from .features import extract_all, standardize
from .model import ENCODER, CodeDetector

CKPT = "results/code/code_codebert_supcon_12k.pt"  # 12k-sample model (4k model kept for the ablation)


@lru_cache(maxsize=1)
def _load(ckpt=CKPT):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = CodeDetector(use_handcrafted=c["use_handcrafted"])
    model.load_state_dict(c["state_dict"])
    model.to(get_device()).eval()
    return model, AutoTokenizer.from_pretrained(ENCODER), c


@torch.no_grad()
def predict_code(src: str) -> float:
    model, tok, c = _load()
    dev = get_device()
    enc = tok([src], truncation=True, max_length=c["max_len"], return_tensors="pt")
    feats = torch.tensor(standardize(extract_all([src]), c["feat_mean"], c["feat_std"]))
    batch = {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev),
             "feats": feats.to(dev)}
    logit, _ = model(batch)
    return float(torch.sigmoid(logit)[0])
