"""Robustness of Layer-1 provenance signals under common edits (real data, real models).

Audio: 50 real bonafide speech clips (ASVspoof2019 LA eval) -> AudioSeal embed -> attacks.
Text : 30 HC3 questions as prompts -> gpt2 KGW / SynthID-style watermarked continuations
       -> edit attacks; FPR on 100 real HC3 human answers.
Image: C2PA manifest survival after re-encode / crop of signed JPEGs.

Run (under the memory watchdog):
  .venv/bin/python tools/memrun.py --limit-gb 3 -- .venv/bin/python -m provenance.robustness
Outputs results/provenance/robustness.json and robustness.png.
"""
import io
import json
import os
import random
import re
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from provenance import audio_watermark as aw
from provenance import text_watermark as tw
from provenance.c2pa_check import check_c2pa

torch.set_num_threads(4)
OUT = "results/provenance"
CLIPS = "data/provenance/asv_bonafide"
SR = 16000
N_AUDIO, N_TEXT, N_HUMAN = 50, 30, 100
rng = random.Random(0)
np_rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- audio data
def load_bonafide(n=N_AUDIO):
    """Stream row groups of one ASVspoof2019 LA parquet shard until n bonafide clips."""
    os.makedirs(CLIPS, exist_ok=True)
    files = sorted(f for f in os.listdir(CLIPS) if f.endswith(".wav"))
    if len(files) < n:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        f = HfFileSystem().open("datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/data/"
                                "test-00000-of-00009.parquet", "rb")
        pf = pq.ParquetFile(f)
        got = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["audio", "label"]).to_pylist()
            for row in t:
                if row["label"] != 0:  # 0 = bonafide
                    continue
                wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                assert sr == SR
                name = os.path.splitext(os.path.basename(row["audio"]["path"]))[0]
                sf.write(f"{CLIPS}/{name}.wav", wav, SR)
                got += 1
                if got >= n:
                    break
            del t
            if got >= n:
                break
        files = sorted(f for f in os.listdir(CLIPS) if f.endswith(".wav"))
    return [sf.read(f"{CLIPS}/{f}", dtype="float32")[0] for f in files[:n]]


# ----------------------------------------------------------------------------- audio attacks
def add_noise(x, snr_db):
    p = np.mean(x ** 2) / (10 ** (snr_db / 10))
    return (x + np_rng.normal(0, np.sqrt(p), x.shape)).astype(np.float32)


def _t(x):
    return torch.from_numpy(x).reshape(1, -1)


def mp3(x, kbps):
    with tempfile.TemporaryDirectory() as d:
        sf.write(f"{d}/a.wav", x, SR)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{d}/a.wav", "-b:a", f"{kbps}k",
                        f"{d}/a.mp3"], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{d}/a.mp3", "-ar", str(SR),
                        "-ac", "1", f"{d}/b.wav"], check=True)
        y = sf.read(f"{d}/b.wav", dtype="float32")[0]
    return y


AUDIO_ATTACKS = {
    "none": lambda x: x,
    "noise SNR20": lambda x: add_noise(x, 20),
    "noise SNR10": lambda x: add_noise(x, 10),
    "resample 8k": lambda x: AF.resample(AF.resample(_t(x), SR, 8000), 8000, SR)[0].numpy(),
    "lowpass 3kHz": lambda x: AF.lowpass_biquad(_t(x), SR, 3000.0)[0].numpy(),
    "mp3 32kbps": lambda x: mp3(x, 32),
    "crop 50%": lambda x: x[len(x) // 4: len(x) // 4 + len(x) // 2],
    # 1.25x speed (pitch shifts too): play 16k samples resampled to 12.8k as if 16k
    "speed 1.25x": lambda x: AF.resample(_t(x), SR, int(SR / 1.25))[0].numpy(),
    "amplitude x0.5": lambda x: 0.5 * x,
}


def run_audio():
    clips = load_bonafide()
    print(f"[audio] {len(clips)} bonafide clips, mean {np.mean([len(c) for c in clips]) / SR:.2f}s")
    wm = [aw.embed(c, SR) for c in clips]
    snr = [10 * np.log10(np.sum(c ** 2) / np.sum((w - c) ** 2)) for c, w in zip(clips, wm)]
    res = {"n_clips": len(clips), "threshold": aw.THRESHOLD, "watermark_snr_db_mean": float(np.mean(snr)),
           "attacks": {}}
    for name, atk in AUDIO_ATTACKS.items():
        pw = [aw.detect(atk(w), SR) for w in wm]
        pc = [aw.detect(atk(c), SR) for c in clips]
        r = {"detection_rate": float(np.mean(np.array(pw) >= aw.THRESHOLD)),
             "fpr": float(np.mean(np.array(pc) >= aw.THRESHOLD)),
             "mean_prob_wm": float(np.mean(pw)), "mean_prob_clean": float(np.mean(pc))}
        res["attacks"][name] = r
        print(f"[audio] {name:15s} TPR={r['detection_rate']:.2f} FPR={r['fpr']:.2f} "
              f"p_wm={r['mean_prob_wm']:.3f} p_clean={r['mean_prob_clean']:.3f}")
    return res


# ----------------------------------------------------------------------------- text data
def load_hc3():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Hello-SimpleAI/HC3", "open_qa/train/0000.parquet", repo_type="dataset",
                        revision="refs/convert/parquet")
    df = pd.read_parquet(p, columns=["question", "human_answers"])
    df = df.sample(frac=1.0, random_state=0)
    prompts = [q.strip() for q in df["question"] if 20 < len(q.strip()) < 200][:N_TEXT]
    human = [a[0].strip() for a in df["human_answers"] if len(a) and len(a[0].split()) >= 60][:N_HUMAN]
    return prompts, human


# ----------------------------------------------------------------------------- text attacks
def delete_words(t, frac):
    w = t.split()
    return " ".join(x for x in w if rng.random() >= frac)


_SYN_CACHE = {}


def _synonym(word):
    from nltk.corpus import wordnet as wn
    lw = word.lower()
    if lw not in _SYN_CACHE:
        cands = {l.name() for s in wn.synsets(lw) for l in s.lemmas()}
        _SYN_CACHE[lw] = sorted(c for c in cands if c.lower() != lw and "_" not in c)
    c = _SYN_CACHE[lw]
    return rng.choice(c) if c else None


def synonym_sub(t, frac):
    out = []
    for tok in t.split():
        m = re.fullmatch(r"([A-Za-z]{4,})([^A-Za-z]*)", tok)
        s = _synonym(m.group(1)) if m and rng.random() < frac else None
        out.append(s + m.group(2) if s else tok)
    return " ".join(out)


def shuffle_within_sentences(t):
    sents = re.split(r"(?<=[.!?])\s+", t)
    out = []
    for s in sents:
        w = s.split()
        rng.shuffle(w)
        out.append(" ".join(w))
    return " ".join(out)


def truncate_half(t):
    w = t.split()
    return " ".join(w[: len(w) // 2])


TEXT_ATTACKS = {
    "none": lambda t: t,
    "delete 10%": lambda t: delete_words(t, 0.10),
    "delete 30%": lambda t: delete_words(t, 0.30),
    "synonym 30%": lambda t: synonym_sub(t, 0.30),
    "shuffle in-sentence": shuffle_within_sentences,
    "truncate 50%": truncate_half,
}


def run_text():
    import nltk
    nltk.download("wordnet", quiet=True)
    prompts, human = load_hc3()
    print(f"[text] {len(prompts)} prompts, {len(human)} human answers")
    res = {"model": tw.MODEL, "z_threshold": tw.Z_THRESHOLD, "n_prompts": len(prompts),
           "n_human": len(human), "schemes": {}, "samples": {}}
    for scheme in ("kgw", "synthid"):
        t0 = time.time()
        gens = [tw.generate_watermarked(p, scheme=scheme, seed=i) for i, p in enumerate(prompts)]
        print(f"[text] {scheme}: generated {len(gens)} in {time.time() - t0:.0f}s")
        human_z = [tw.detect(h, scheme)[1] for h in human]
        sc = {"fpr_human": float(np.mean(np.array(human_z) > tw.Z_THRESHOLD)),
              "mean_z_human": float(np.mean(human_z)), "attacks": {}}
        for name, atk in TEXT_ATTACKS.items():
            z = [tw.detect(atk(g), scheme)[1] for g in gens]
            sc["attacks"][name] = {"detection_rate": float(np.mean(np.array(z) > tw.Z_THRESHOLD)),
                                   "mean_z": float(np.mean(z))}
            print(f"[text] {scheme} {name:20s} TPR={sc['attacks'][name]['detection_rate']:.2f} "
                  f"mean z={np.mean(z):.2f}")
        print(f"[text] {scheme} FPR on human={sc['fpr_human']:.2f} mean z={sc['mean_z_human']:.2f}")
        res["schemes"][scheme] = sc
        res["samples"][scheme] = [{"prompt": p, "generation": g} for p, g in zip(prompts[:3], gens[:3])]
    return res


# ----------------------------------------------------------------------------- image / C2PA
def run_c2pa():
    from PIL import Image
    res = {}
    for f in ["data/provenance/C.jpg", "data/provenance/A_ai_signed.jpg"]:
        im = Image.open(f).convert("RGB")
        r = {"original": check_c2pa(f)["has_manifest"]}
        with tempfile.TemporaryDirectory() as d:
            im.save(f"{d}/q90.jpg", quality=90)
            r["re-encode JPEG q90"] = check_c2pa(f"{d}/q90.jpg")["has_manifest"]
            im.crop((0, 0, im.width // 2, im.height // 2)).save(f"{d}/crop.png")
            r["crop + PNG"] = check_c2pa(f"{d}/crop.png")["has_manifest"]
        res[os.path.basename(f)] = r
        print(f"[c2pa] {f}: {r}")
    return res


# ----------------------------------------------------------------------------- plot
def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    a = res["audio"]["attacks"]
    names = list(a)
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x - 0.2, [a[n]["detection_rate"] for n in names], 0.4, label="detection rate (watermarked)",
           color="#2b6cb0")
    ax.bar(x + 0.2, [a[n]["fpr"] for n in names], 0.4, label="false positive rate (clean)", color="#e07b39")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_title(f"AudioSeal on {res['audio']['n_clips']} real ASVspoof2019 bonafide clips")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1]
    s = res["text"]["schemes"]
    tn = list(TEXT_ATTACKS)
    x = np.arange(len(tn) + 1)
    for i, (scheme, col) in enumerate([("kgw", "#2b6cb0"), ("synthid", "#38a169")]):
        vals = [s[scheme]["attacks"][n]["detection_rate"] for n in tn] + [s[scheme]["fpr_human"]]
        ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, color=col,
               label="KGW green-list" if scheme == "kgw" else "SynthID-Text (mean-g)")
    ax.set_xticks(x, tn + ["FPR (human HC3)"], rotation=30, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_title(f"gpt2 text watermarks, {res['text']['n_prompts']} HC3 prompts (z>{tw.Z_THRESHOLD:g})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/robustness.png", dpi=150)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = {"c2pa": run_c2pa(), "audio": run_audio(), "text": run_text()}
    res["runtime_s"] = round(time.time() - t0, 1)
    with open(f"{OUT}/robustness.json", "w") as f:
        json.dump(res, f, indent=2)
    plot(res)
    print(f"saved {OUT}/robustness.json and robustness.png in {res['runtime_s']}s")


if __name__ == "__main__":
    main()
