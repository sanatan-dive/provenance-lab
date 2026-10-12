"""predict_image(path) -> probability the image is AI-generated.

Uses the GenImage model (image/genimage.py): trained at 224 px on 7 modern generators. Every input is
normalised exactly like the training data (shorter side 256, centre crop 224, JPEG q=95).
predict_image_cifake keeps the original 32x32 CIFAKE model for comparison.
"""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from common.train import get_device

from .features import handcrafted_features, spectrum_input
from .model import ImageDetector

CKPT = "results/image/image_genimage.pt"
CIFAKE_CKPT = "results/image/image_vit_fft_supcon.pt"


@lru_cache(maxsize=2)
def _load(ckpt=CKPT):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = c["config"]
    model = ImageDetector(use_handcrafted=cfg["use_handcrafted"], use_freq=cfg["use_freq"], vit_name=cfg["vit"])
    model.load_state_dict(c["state_dict"])
    return model.to(get_device()).eval(), c["feat_mean"], c["feat_std"]


def prepare(path) -> np.ndarray:
    """Upload -> the 224x224 RGB array the model was trained on."""
    from .genimage import decode, normalise
    return decode(normalise(Image.open(path)))


@torch.no_grad()
def scores(images: np.ndarray, ckpt=CKPT) -> np.ndarray:
    """Batch of uint8 [N,H,W,3] -> AI probabilities."""
    model, mean, std = _load(ckpt)
    dev = get_device()
    batch = {"pixels": torch.from_numpy(images).permute(0, 3, 1, 2).to(dev),
             "spectrum": torch.from_numpy(spectrum_input(images)).unsqueeze(1).to(dev),
             "feats": torch.from_numpy((handcrafted_features(images) - mean) / std).float().to(dev)}
    logit, _ = model(batch)
    return torch.sigmoid(logit).float().cpu().numpy()


def predict_image(path) -> float:
    return float(scores(prepare(path)[None])[0])


def predict_image_cifake(path) -> float:
    """Original CIFAKE model: every input is shrunk to 32x32 (only meaningful for CIFAKE-like images)."""
    img = np.asarray(Image.open(path).convert("RGB").resize((32, 32), Image.BICUBIC))[None]
    return float(scores(img, CIFAKE_CKPT)[0])
