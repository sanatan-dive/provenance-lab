"""Layer-1 provenance: text watermarks, generated with gpt2 (124M) on CPU.

Two schemes, both real implementations shipped in transformers 5.19:

* "kgw"     - Kirchenbauer et al. 2023 "green-list" watermark (`WatermarkingConfig`).
              Detection = transformers `WatermarkDetector` one-proportion z-test on the
              fraction of green tokens. No training, only the shared hashing key.
* "synthid" - Google SynthID-Text tournament-sampling watermark
              (`SynthIDTextWatermarkingConfig`). transformers also ships
              `SynthIDTextWatermarkDetector`, but that wraps a *trained* Bayesian
              detector (Google only released one for gemma-2b, not gpt2). Instead we use
              the training-free "mean g-value" score from the SynthID paper
              (Dathathri et al., Nature 2024): under H0 (no watermark) every g-value is
              an independent fair coin, so mean(g) ~ 0.5 and we can z-test it.
              Real Gemini SynthID detection needs Google's private keys - not possible.

    from provenance.text_watermark import generate_watermarked, detect
    t = generate_watermarked("The history of the bicycle")
    is_wm, z = detect(t)                         # kgw by default
    t2 = generate_watermarked("...", scheme="synthid"); detect(t2, scheme="synthid")
"""
from functools import lru_cache
import math

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig,
                          WatermarkDetector, WatermarkingConfig)
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

MODEL = "gpt2"
DEVICE = "cpu"
Z_THRESHOLD = 4.0  # one-sided z > 4  ->  ~3e-5 theoretical FPR (KGW paper default)

KGW_CFG = WatermarkingConfig(greenlist_ratio=0.25, bias=2.0, hashing_key=15485863,
                             seeding_scheme="lefthash", context_width=1)
SYNTHID_CFG = SynthIDTextWatermarkingConfig(
    keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585],
    ngram_len=5)


@lru_cache(maxsize=1)
def _lm():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(DEVICE).eval()
    return tok, model


@torch.no_grad()
def generate_watermarked(prompt, scheme="kgw", max_new_tokens=150, seed=None):
    tok, model = _lm()
    if seed is not None:
        torch.manual_seed(seed)
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    cfg = KGW_CFG if scheme == "kgw" else SYNTHID_CFG
    out = model.generate(**enc, watermarking_config=cfg, do_sample=True, top_k=0, temperature=1.0,
                         max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


@lru_cache(maxsize=1)
def _kgw_detector():
    _, model = _lm()
    return WatermarkDetector(model.config, DEVICE, KGW_CFG, ignore_repeated_ngrams=True)


@lru_cache(maxsize=1)
def _synthid_processor():
    c = SYNTHID_CFG
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=c.ngram_len, keys=c.keys, sampling_table_size=c.sampling_table_size,
        sampling_table_seed=c.sampling_table_seed, context_history_size=c.context_history_size,
        device=torch.device(DEVICE))


def _synthid_z(ids):
    proc = _synthid_processor()
    g = proc.compute_g_values(ids)                                    # (1, L-n+1, depth)
    mask = proc.compute_context_repetition_mask(ids)                  # (1, L-n+1)
    eos = proc.compute_eos_token_mask(ids, _lm()[0].eos_token_id)[:, proc.ngram_len - 1:]
    m = (mask * eos).bool()
    vals = g[m].float()                                               # (k, depth)
    n = vals.numel()
    if n == 0:
        return 0.0
    return (vals.mean().item() - 0.5) / math.sqrt(0.25 / n)


def detect(text, scheme="kgw", z_threshold=Z_THRESHOLD):
    """Return (is_watermarked, z_score). Needs >= a few tokens; short text gives z ~ 0."""
    tok, _ = _lm()
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
    min_len = 2 if scheme == "kgw" else SYNTHID_CFG.ngram_len
    if ids.shape[1] < min_len:
        return False, 0.0
    if scheme == "kgw":
        z = float(_kgw_detector()(ids, z_threshold=z_threshold, return_dict=True).z_score[0])
    else:
        z = _synthid_z(ids)
    return z > z_threshold, z


if __name__ == "__main__":
    for s in ("kgw", "synthid"):
        t = generate_watermarked("The most important thing about learning to cook is", scheme=s, seed=0)
        print(f"[{s}] {t[:200]!r}...\n  watermarked text -> {detect(t, s)}")
        print(f"  human text       -> {detect(__doc__, s)}")
