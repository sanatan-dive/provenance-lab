"""End-to-end check of the running web app: real held-out samples through /api/sample and /api/detect.

    .venv/bin/python tools/e2e_api.py [n_per_kind]      (server must be running on :8000)
Writes results/e2e_api.json: every request, the verdict, the true label, and latency.
"""
import base64
import json
import sys
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=300) as r:
        return json.load(r)


def post_detect(fields, file=None):
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if file:
        name, data = file
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/api/detect", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


rows = []
for kind in ("text", "code", "image", "audio"):
    for i in range(N):
        s = get(f"/api/sample?modality={kind}&label={i % 2}")
        t0 = time.time()
        if "content" in s:
            r = post_detect({"modality": kind, "text": s["content"]})
        else:
            r = post_detect({"modality": kind}, (s["filename"], base64.b64decode(s["data_url"].split(",", 1)[1])))
        said = 1 if r["verdict"].startswith("AI") else 0 if r["verdict"] == "Human" else None
        rows.append({"kind": kind, "true": s["label"], "verdict": r["verdict"], "layer": r["layer"],
                     "ai_probability": r.get("ai_probability"), "correct": said == s["label"] if said is not None else None,
                     "seconds": round(time.time() - t0, 2), "source": s["source"]})
        print(f"{kind:5s} true={'AI   ' if s['label'] else 'human'} -> {r['verdict']:32s} p={r.get('ai_probability')} "
              f"{'OK' if said == s['label'] else 'UNSURE' if said is None else 'WRONG'} ({rows[-1]['seconds']}s)", flush=True)

decided = [r for r in rows if r["correct"] is not None]
summary = {"requests": len(rows), "errors": 0, "decided": len(decided), "correct": sum(r["correct"] for r in decided),
           "uncertain": len(rows) - len(decided), "rows": rows}
open("results/e2e_api.json", "w").write(json.dumps(summary, indent=2))
print(f"\n{summary['correct']}/{summary['decided']} decided correctly, {summary['uncertain']} flagged uncertain, "
      f"{summary['requests']} requests, 0 errors")
