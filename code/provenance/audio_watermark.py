"""Layer-1 provenance: Meta AudioSeal invisible audio watermark (16-bit message variant).

Runs on CPU (models are tiny, ~15 MB). AudioSeal operates at 16 kHz; inputs at other
rates are resampled to 16 kHz.

    from provenance.audio_watermark import embed, detect
    wm = embed(wav, sr)          # wav: 1-D float numpy/tensor, returns 1-D float32 numpy @ sr
    p = detect(wm, sr)           # probability in [0,1] that the clip is watermarked
"""
from functools import lru_cache

import numpy as np
import torch
import torchaudio.functional as AF
from audioseal import AudioSeal

SR = 16000
THRESHOLD = 0.5  # detect() >= THRESHOLD  ->  watermarked


@lru_cache(maxsize=1)
def _models():
    gen = AudioSeal.load_generator("audioseal_wm_16bits").eval()
    det = AudioSeal.load_detector("audioseal_detector_16bits").eval()
    return gen, det


def _to16k(wav, sr):
    x = torch.as_tensor(np.asarray(wav, dtype=np.float32)).reshape(1, 1, -1)
    return AF.resample(x, sr, SR) if sr != SR else x


@torch.no_grad()
def embed(wav, sr, message=None):
    gen, _ = _models()
    x = _to16k(wav, sr)
    y = x + gen.get_watermark(x, SR, message=message)
    if sr != SR:
        y = AF.resample(y, SR, sr)
    return y.reshape(-1).numpy()


@torch.no_grad()
def detect(wav, sr) -> float:
    _, det = _models()
    prob, _message = det.detect_watermark(_to16k(wav, sr), SR)
    return float(prob)
