"""Web app backend.

    .venv/bin/python server.py   ->   http://127.0.0.1:8000
"""
import json
import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detect import detect, modality_of
from samples import get_sample

ROOT = Path(__file__).parent
app = FastAPI(title="Multimodal AI-content detector")


@app.post("/api/detect")
async def api_detect(modality: str = Form(None), text: str = Form(None), file: UploadFile = File(None)):
    try:
        if file is not None:
            suffix = Path(file.filename).suffix.lower()
            m = modality or modality_of(file.filename)
            if m in ("text", "code"):
                content = (await file.read()).decode("utf-8", errors="replace")
                return detect(content, m)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(file.file, tmp)
            try:
                return detect(tmp.name, m)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        return detect(text, modality or "text")
    except FileNotFoundError as e:
        return JSONResponse({"error": f"This detector hasn't been trained yet ({e.filename or e})."}, 503)


@app.get("/api/sample")
def api_sample(modality: str, label: int = None):
    try:
        return get_sample(modality, label)
    except FileNotFoundError:
        return JSONResponse({"error": "Test samples for this kind of content aren't prepared yet."}, 503)


@app.get("/api/metrics")
def api_metrics():
    out = []
    for f in sorted((ROOT / "results").glob("*/*_metrics.json")):
        m = json.loads(f.read_text())
        if m.get("smoke"):
            continue
        roc = f.with_name(f.name.replace("_metrics.json", "_roc.png"))
        m["modality"] = f.parent.name
        m["roc"] = f"/results/{f.parent.name}/{roc.name}" if roc.exists() else None
        curves = f.with_name(f.name.replace("_metrics.json", "_curves.png"))
        m["curves"] = f"/results/{f.parent.name}/{curves.name}" if curves.exists() else None
        m.pop("history", None)
        pg = f.with_name(f.name.replace("_metrics.json", "_per_generator.json"))
        if pg.exists():
            m["per_generator"] = json.loads(pg.read_text())
        out.append(m)
    def opt(rel):
        f = ROOT / rel
        return json.loads(f.read_text()) if f.exists() else None
    return {"runs": out, "robustness": opt("results/provenance/robustness.json"),
            "baselines": opt("results/baselines/baselines.json"), "ood": opt("results/ood/mage.json"),
            "market": opt("results/market/market.json"), "ood_diverse": opt("results/ood/mage_diverse_12k.json"),
            "voice_modern": opt("results/ood/voice_modern.json")}


(ROOT / "results").mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=ROOT / "results"), name="results")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
