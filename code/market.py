"""Compare our detectors with popular public detectors on exactly the same held-out test sets.

Every public model is used as released (no fine-tuning). Its AI/human labels are read from the
model's own config, and the mapping is printed so it can be checked.
Fairness notes are stored with each result: some public models were trained on the same benchmark
(home advantage), and every model, ours included, is out of domain on MAGE.
    .venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python market.py
"""
import gc
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import pipeline

from common.train import get_device

OUT = Path("results/market")
AI_WORDS = ("fake", "artificial", "ai", "chatgpt", "machine", "generated", "spoof", "gpt")
HUMAN_WORDS = ("real", "human", "hum", "bonafide", "bona-fide", "authentic")

TEXT_MODELS = {
    "openai-community/roberta-base-openai-detector": "",
    "fakespot-ai/roberta-base-ai-text-detection-v1": "",
    "Hello-SimpleAI/chatgpt-detector-roberta": "Trained on HC3 itself: home advantage on HC3; MAGE is the fair test.",
}
IMAGE_MODELS = {
    "umm-maybe/AI-image-detector": "",
    "dima806/ai_vs_real_image_detection": "Trained on CIFAKE itself (per its model card): home advantage, possible test-set overlap.",
    "Ateeqq/ai-vs-human-image-detector": "",
}
AUDIO_MODELS = {
    "mo-thecreator/Deepfake-audio-detection": "",
    "MelodyMachine/Deepfake-audio-detection-V2": "",
}


def ai_label(id2label):
    """Name of the label that means AI-generated, decided from the label names."""
    labels = [str(v) for v in id2label.values()]
    low = {l: l.lower() for l in labels}
    ai = [l for l in labels if any(w == low[l] or w in low[l].replace("_", " ").split() or low[l].startswith(w) for w in AI_WORDS)
          and not any(low[l].startswith(h) for h in HUMAN_WORDS)]
    if len(ai) != 1:
        raise ValueError(f"can't tell which label is AI: {labels}")
    return ai[0]


def ai_probs(pipe, inputs, batch_size=16, **kw):
    target = ai_label(pipe.model.config.id2label)
    print(f"  labels {list(pipe.model.config.id2label.values())} -> AI = {target!r}", flush=True)
    out = pipe(inputs, batch_size=batch_size, top_k=None, **kw)
    return np.array([next(d["score"] for d in r if d["label"] == target) for r in out]), target


def score(y, p):
    return {"accuracy": float(accuracy_score(y, p >= 0.5)), "f1": float(f1_score(y, p >= 0.5)),
            "auc_roc": float(roc_auc_score(y, p))}


PARTIAL = OUT / "market_partial.json"


def save_partial(mod, res):
    allres = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    allres[mod] = res
    PARTIAL.write_text(json.dumps(allres, indent=2))


def run(task, mod, models, sets, **kw):
    """Resumable: models already scored on every set (in market_partial.json) are skipped."""
    res = (json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}).get(mod, {})
    dev = 0 if get_device().type == "mps" else -1
    for name, note in models.items():
        if all(k in res.get(name, {}) for k in sets):
            print(f"{task}: {name} (cached)", flush=True)
            continue
        print(f"{task}: {name}", flush=True)
        try:
            pipe = pipeline(task, model=name, device="mps" if dev == 0 else "cpu")
            res[name] = {"note": note}
            for set_name, (inputs, y) in sets.items():
                p, target = ai_probs(pipe, inputs, **kw)
                res[name][set_name] = score(y, p)
                res[name]["ai_label"] = target
                print(f"  {set_name}: {res[name][set_name]}", flush=True)
        except Exception as e:  # a public model that won't load or has unclear labels is reported, not hidden
            res[name] = {"note": note, "error": f"{type(e).__name__}: {e}"[:300]}
            print("  ERROR", res[name]["error"], flush=True)
        finally:
            pipe = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        save_partial(mod, res)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}

    # Essays: HC3 test + MAGE (out of domain for everyone)
    hc3 = pd.read_parquet("data/text/hc3_tr4000_te1000_s0_test.parquet")
    from ood_text import load_mage
    mage = load_mage()
    out["text"] = run("text-classification", "text", TEXT_MODELS,
                      {"hc3": (list(hc3.text), hc3.label.to_numpy()), "mage": (list(mage.text), mage.label.to_numpy())},
                      truncation=True, max_length=512)

    # Images: CIFAKE test + GenImage test (modern generators, 224 px, JPEG-normalised like our training data)
    import pickle
    z = np.load("data/image/cifake_test_n2000_seed0.npz")
    imgs = [Image.fromarray(a) for a in z["images"]]
    g = pickle.loads(open("data/genimage/validation_250_s0.pkl", "rb").read())
    gimgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in g["jpeg"]]
    out["image"] = run("image-classification", "image", IMAGE_MODELS,
                       {"cifake": (imgs, z["labels"]), "genimage": (gimgs, g["label"])})

    # Voice: ASVspoof test clips (16 kHz, 4 s)
    waves = np.load("data/audio/test_wave_1000_s0.npy").astype(np.float32) / 32767
    meta = pd.read_parquet("data/audio/test_meta_1000_s0.parquet")
    mod = pickle.loads(open("data/audio/modern_s0.pkl", "rb").read())
    te = (mod["meta"].split == "test").to_numpy()
    mwaves = mod["waves"][te].astype(np.float32) / 32767
    out["audio"] = run("audio-classification", "audio", AUDIO_MODELS,
                       {"asvspoof": ([{"raw": w, "sampling_rate": 16000} for w in waves], meta.label.to_numpy()),
                        "modern": ([{"raw": w, "sampling_rate": 16000} for w in mwaves], mod["meta"].label.to_numpy()[te])},
                       batch_size=8)

    (OUT / "market.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
