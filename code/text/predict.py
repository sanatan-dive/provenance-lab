"""predict_text(text) -> probability the text is AI-generated.

Uses the diverse-data model (HC3 + MAGE training data). Its decision threshold was calibrated on MAGE
validation data (text/calibrate.py); the raw score is rescaled so that threshold maps to 0.5, which keeps
the app's 30% / 70% "uncertain" band meaningful.
"""
import json
from pathlib import Path
from functools import lru_cache

import torch
from transformers import AutoTokenizer

from common.train import get_device
from text.features import FEATURE_NAMES, LMScorer, compute_features, standardize
from text.model import ENCODER_NAME, TextDetector

DEFAULT_CKPT = "results/text/text_bert_diverse_12k.pt"
CALIBRATION = Path("results/ood/mage_diverse_12k.json")


def _threshold():
    return json.loads(CALIBRATION.read_text())["calibrated"]["threshold"] if CALIBRATION.exists() else 0.5


def _rescale(p, t):
    return p / (2 * t) if p < t else 0.5 + (p - t) / (2 * (1 - t))


@lru_cache(maxsize=2)
def _load(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TextDetector(feat_dim=len(FEATURE_NAMES), use_handcrafted=ck["use_handcrafted"])
    model.load_state_dict(ck["state_dict"])
    return model.to(get_device()).eval(), AutoTokenizer.from_pretrained(ENCODER_NAME), ck, LMScorer()


@torch.no_grad()
def predict_text(text: str, ckpt_path: str = DEFAULT_CKPT) -> float:
    model, tok, ck, scorer = _load(ckpt_path)
    device = get_device()
    feats = standardize(compute_features([text], scorer), ck["feat_mean"], ck["feat_std"])
    enc = tok([text], truncation=True, max_length=ck["max_len"], return_tensors="pt")
    batch = {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device),
             "feats": torch.from_numpy(feats).to(device)}
    logit, _ = model(batch)
    p = torch.sigmoid(logit).item()
    return _rescale(p, _threshold()) if ckpt_path == DEFAULT_CKPT else p
