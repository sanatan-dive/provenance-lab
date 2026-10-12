"""Real held-out test samples for the web demo (never seen in training).

Each sample comes back with its true label, so the demo can show whether the model got it right.
"""
import base64
import glob
import io
import random

import numpy as np
import pandas as pd


def _pick(labels, want):
    idx = [i for i, y in enumerate(labels) if want is None or int(y) == want]
    return random.choice(idx)


def get_sample(modality, want=None):
    """want: 1 = AI, 0 = human, None = either."""
    if modality == "text":
        f = max(glob.glob("data/text/hc3_tr*_te*_s0_test.parquet"), key=lambda p: int(p.split("_te")[1].split("_")[0]))
        d = pd.read_parquet(f)
        r = d.iloc[_pick(d.label, want)]
        return {"modality": "text", "label": int(r.label), "content": r.text,
                "source": f"HC3 ({r.source})"}
    if modality == "code":
        f = max(glob.glob("data/code/test_*_s0.parquet"), key=lambda p: int(p.split("test_")[1].split("_")[0]))
        d = pd.read_parquet(f)
        r = d.iloc[_pick(d.label, want)]
        return {"modality": "code", "label": int(r.label), "content": r.code,
                "source": f"CoDET-M4 (written by {r.model})"}
    if modality == "image":
        import pickle
        d = pickle.loads(open("data/genimage/validation_250_s0.pkl", "rb").read())
        i = _pick(d["label"], want)
        gen = d["generator"][i]
        return {"modality": "image", "label": int(d["label"][i]), "filename": f"genimage_test_{i}.jpg",
                "data_url": "data:image/jpeg;base64," + base64.b64encode(d["jpeg"][i]).decode(),
                "source": "GenImage, real ImageNet photo" if gen == "Real" else f"GenImage, made by {gen}"}
    if modality == "audio" and random.random() < 0.5:  # half the samples come from modern TTS
        import pickle

        import soundfile as sf
        m = pickle.loads(open("data/audio/modern_s0.pkl", "rb").read())
        test = np.flatnonzero(m["meta"].split == "test")
        i = test[_pick(m["meta"].label.to_numpy()[test], want)]
        names = {"el": "ElevenLabs", "po": "Amazon Polly", "hu": "Hume AI", "yt": "a real YouTube recording"}
        src = m["meta"].source.iloc[i]
        buf = io.BytesIO()
        sf.write(buf, m["waves"][i], 16000, format="WAV", subtype="PCM_16")
        return {"modality": "audio", "label": int(m["meta"].label.iloc[i]), "filename": f"modern_test_{i}.wav",
                "data_url": "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode(),
                "source": f"Modern voices: {names.get(src, 'a TTS platform (' + src + ')')}"}
    if modality == "audio":
        import soundfile as sf
        wf = max(glob.glob("data/audio/test_wave_*_s*.npy"), key=lambda p: int(p.split("test_wave_")[1].split("_")[0]))
        meta_f = wf.replace("test_wave_", "test_meta_").replace(".npy", "")
        meta = pd.read_parquet(meta_f + ".parquet") if glob.glob(meta_f + ".parquet") else pd.DataFrame(dict(np.load(meta_f + ".npz")))
        waves = np.load(wf, mmap_mode="r")
        i = _pick(meta.label, want)
        buf = io.BytesIO()
        sf.write(buf, np.asarray(waves[i]), 16000, format="WAV", subtype="PCM_16")
        return {"modality": "audio", "label": int(meta.label.iloc[i]), "filename": f"asvspoof_test_{i}.wav",
                "data_url": "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode(),
                "source": "ASVspoof 2019 LA"}
    raise ValueError(modality)
