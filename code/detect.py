"""Unified two-layer detector.

Layer 1 (provenance): C2PA content credentials / invisible watermarks. If found, that is
                      verifiable proof and we stop.
Layer 2 (forensic):   the trained deep-learning detector for the input's modality.

Usage:
    .venv/bin/python detect.py path/to/file.jpg
    .venv/bin/python detect.py path/to/script.py
    .venv/bin/python detect.py --text "some essay text ..."
"""
import argparse
import json
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
CODE_EXT = {".py"}


def modality_of(path):
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in CODE_EXT:
        return "code"
    return "text"


def forensic_score(content, modality):
    if modality == "text":
        from text.predict import predict_text
        return predict_text(content)
    if modality == "code":
        from code_det.predict import predict_code
        return predict_code(content)
    if modality == "image":
        from image.predict import predict_image
        return predict_image(content)
    from audio.predict import predict_audio
    return predict_audio(content)


def detect(content, modality):
    """content: a file path for image/audio, the raw string for text/code."""
    from provenance.check import check_provenance

    prov = check_provenance(content, modality)
    # A C2PA manifest only proves AI origin if it declares it (e.g. trainedAlgorithmicMedia);
    # a manifest from a camera or photo editor is provenance, not an AI verdict.
    c2pa_not_ai = prov.get("source") == "c2pa" and not (prov.get("details") or {}).get("ai_generated")
    if prov.get("found") and not c2pa_not_ai:
        return {"modality": modality, "verdict": "AI-generated", "layer": "provenance",
                "confidence": 1.0, "provenance": prov}
    p = float(forensic_score(content, modality))
    if p >= 0.7:
        verdict = "AI-generated"
    elif p <= 0.3:
        verdict = "Human"
    else:
        verdict = "Uncertain (needs human review)"
    from explain import explain
    return {"modality": modality, "verdict": verdict, "layer": "forensic",
            "ai_probability": round(p, 4), "provenance": prov, "explanation": explain(content, modality)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--text")
    a = ap.parse_args()
    if a.text:
        print(json.dumps(detect(a.text, "text"), indent=2, default=str))
    else:
        m = modality_of(a.path)
        content = Path(a.path).read_text() if m in ("text", "code") else a.path
        print(json.dumps(detect(content, m), indent=2, default=str))
