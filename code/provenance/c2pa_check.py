"""Layer-1 provenance: read C2PA Content Credentials from a media file.

Uses c2pa-python 0.38 (c2pa-rs 0.91) API: `c2pa.Reader.try_create(path)` returns
None when the file carries no manifest; `reader.json()` gives the manifest store.

Run:  .venv/bin/python -m provenance.c2pa_check data/provenance/C.jpg data/provenance/A.jpg
"""
import json
import sys

import c2pa

# IPTC digital source types that declare generative-AI involvement.
AI_SOURCE_TYPES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia",
                   "compositeSynthetic"}
# Non-learned synthetic media (procedural / computer generated) - reported separately.
SYNTHETIC_SOURCE_TYPES = {"algorithmicMedia", "digitalCreation", "virtualRecording"}


def _source_types(obj):
    """Collect every digitalSourceType value anywhere in an assertion tree."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "digitalSourceType" and isinstance(v, str):
                found.append(v.rstrip("/").rsplit("/", 1)[-1])
            else:
                found += _source_types(v)
    elif isinstance(obj, list):
        for v in obj:
            found += _source_types(v)
    return found


def check_c2pa(path) -> dict:
    out = {"has_manifest": False, "claim_generator": None, "ai_generated": False,
           "synthetic": False, "digital_source_types": [], "validation_state": None,
           "assertions": [], "error": None}
    try:
        reader = c2pa.Reader.try_create(str(path))
    except c2pa.C2paError as e:  # unsupported format or corrupt manifest
        out["error"] = str(e)
        return out
    if reader is None:
        return out
    with reader:
        store = json.loads(reader.json())
        out["validation_state"] = str(reader.get_validation_state())
    active = store["manifests"][store["active_manifest"]]
    out["has_manifest"] = True
    out["claim_generator"] = active.get("claim_generator") or ", ".join(
        f"{g.get('name')}/{g.get('version')}" for g in active.get("claim_generator_info", []))
    out["assertions"] = [a["label"] for a in active.get("assertions", [])]
    types = _source_types(active.get("assertions", []))
    out["digital_source_types"] = sorted(set(types))
    out["ai_generated"] = any(t in AI_SOURCE_TYPES for t in types)
    out["synthetic"] = out["ai_generated"] or any(t in SYNTHETIC_SOURCE_TYPES for t in types)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, json.dumps(check_c2pa(p), indent=2))
