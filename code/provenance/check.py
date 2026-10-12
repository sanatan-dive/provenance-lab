"""Layer-1 entry point: look for verifiable provenance before running Layer-2 detectors.

    check_provenance("data/provenance/A_ai_signed.jpg", "image")
    check_provenance("clip.wav", "audio")
    check_provenance("some generated text ...", "text")

`found` means a verifiable signal was present (C2PA manifest, AudioSeal watermark,
text watermark). If found is False, defer to the Layer-2 learned detector.

Run:  .venv/bin/python -m provenance.check image data/provenance/A_ai_signed.jpg
"""
import sys

from provenance.c2pa_check import check_c2pa


def check_provenance(path_or_text, modality) -> dict:
    if modality == "image" or modality == "video":
        c = check_c2pa(path_or_text)
        return {"found": c["has_manifest"], "source": "c2pa" if c["has_manifest"] else None, "details": c}

    if modality == "audio":
        import soundfile as sf
        from provenance.audio_watermark import THRESHOLD, detect
        c = check_c2pa(path_or_text)
        if c["has_manifest"]:
            return {"found": True, "source": "c2pa", "details": c}
        wav, sr = sf.read(path_or_text, dtype="float32", always_2d=True)
        p = detect(wav.mean(axis=1), sr)
        found = p >= THRESHOLD
        return {"found": found, "source": "audioseal" if found else None,
                "details": {"c2pa": c, "audioseal_prob": p}}

    if modality in ("text", "code"):
        from provenance.text_watermark import detect
        details = {}
        for scheme in ("kgw", "synthid"):
            is_wm, z = detect(path_or_text, scheme)
            details[scheme] = {"watermarked": is_wm, "z_score": z}
            if is_wm:
                return {"found": True, "source": "text_watermark", "details": {"scheme": scheme, **details}}
        return {"found": False, "source": None, "details": details}

    raise ValueError(f"unknown modality {modality!r}")


if __name__ == "__main__":
    modality, arg = sys.argv[1], sys.argv[2]
    print(check_provenance(arg, modality))
