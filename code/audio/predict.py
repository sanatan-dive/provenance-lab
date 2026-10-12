"""predict_audio(path) -> probability the clip is synthetic / voice-cloned.

Uses the wav2vec 2.0 detector trained on ASVspoof + modern TTS (audio/modern.py): frozen wav2vec 2.0
features + logistic regression.
The spectrogram CNN (predict_audio_cnn) is kept as the comparison model.
"""
import pickle
from functools import lru_cache

import numpy as np
import torch

from .features import FEAT_DIM, decode, wave_to_inputs
from .model import AudioCNN

CKPT = "results/audio/audio_cnn_supcon.pt"
W2V_CLF = "results/audio/audio_w2v_modern.pkl"  # ASVspoof + modern TTS (audio/modern.py)


@lru_cache(maxsize=1)
def _load_w2v():
    from .w2v import Embedder
    with open(W2V_CLF, "rb") as f:
        return Embedder(), pickle.load(f)


def predict_audio(path: str) -> float:
    emb, clf = _load_w2v()
    wave = decode(path)  # same decode + silence trim + 4 s window as training
    return float(clf.predict_proba(emb(wave[None]))[0, 1])


@lru_cache(maxsize=1)
def _load(ckpt=CKPT):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = AudioCNN(FEAT_DIM, use_handcrafted=c["use_handcrafted"])
    model.load_state_dict(c["state_dict"])
    return model.eval(), c["feat_mean"], c["feat_std"]


@torch.no_grad()
def predict_audio_cnn(path: str) -> float:
    model, mu, sd = _load()
    mel, feats = wave_to_inputs(decode(path))  # any format librosa reads -> 16 kHz mono, first 4 s
    batch = {"mel": torch.from_numpy(mel)[None, None],
             "feats": torch.from_numpy(((feats - mu) / sd).astype(np.float32))[None]}
    logit, _ = model(batch)
    return float(torch.sigmoid(logit)[0])
