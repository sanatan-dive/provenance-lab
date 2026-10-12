"""Evidence behind a verdict, shown in the web app.

text   -> AI probability for each sentence (scored by the same essay model), so the most AI-like
          sentences can be highlighted.
image  -> occlusion heatmap: grey out each patch of a 7x7 grid and measure how much the AI
          probability drops. Patches whose removal changes the score most are what the model used.
audio  -> the log-mel spectrogram the detector listened to.
code   -> per-function AI probability for files with several functions.
"""
import ast
import base64
import io
import re

import numpy as np


def _data_url(buf):
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _png_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return _data_url(buf)


def _png_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", dpi=110, bbox_inches="tight", pad_inches=0.05)
    return _data_url(buf)


def explain_text(text):
    from text.predict import predict_text
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    if len(sents) < 2:
        return None
    return {"kind": "sentences", "items": [{"text": s, "p": round(predict_text(s), 3) if len(s.split()) >= 6 else None}
                                           for s in sents[:40]]}


def explain_code(src):
    from code_det.predict import predict_code
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    lines = src.splitlines()
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if len(funcs) < 2:
        return None
    return {"kind": "blocks", "items": [{"name": f.name, "line": f.lineno,
                                         "p": round(predict_code("\n".join(lines[f.lineno - 1:f.end_lineno])), 3)}
                                        for f in funcs[:20]]}


def explain_image(path, grid=7):
    from PIL import Image

    from image.predict import prepare, scores
    img = prepare(path)
    base = float(scores(img[None])[0])
    step = img.shape[0] // grid
    fill = img.reshape(-1, 3).mean(0).astype(np.uint8)
    batch = []
    for i in range(grid):
        for j in range(grid):
            x = img.copy()
            x[i * step:(i + 1) * step, j * step:(j + 1) * step] = fill
            batch.append(x)
    occluded = np.concatenate([scores(np.stack(batch[k:k + 16])) for k in range(0, len(batch), 16)])
    # importance = how much the score moves toward the other class when the patch is hidden
    imp = np.abs(base - occluded).reshape(grid, grid)
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)  # min-max so only the decisive patches stand out
    heat = np.array(Image.fromarray((imp * 255).astype(np.uint8)).resize(img.shape[:2][::-1], Image.BICUBIC)) / 255.0
    heat = heat ** 2  # suppress the weak patches
    violet = np.array([90, 63, 208], dtype=np.float32)
    dim = img.astype(np.float32) * (0.55 + 0.45 * heat[..., None])  # darken what didn't matter
    overlay = dim * (1 - 0.45 * heat[..., None]) + violet * 0.45 * heat[..., None]
    return {"kind": "heatmap", "image": _png_image(Image.fromarray(overlay.clip(0, 255).astype(np.uint8))),
            "note": "Violet areas changed the verdict most when hidden."}


def explain_audio(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from audio.features import decode, logmel
    mel = logmel(decode(path))
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma", extent=[0, 4, 0, mel.shape[0]])
    ax.grid(False)
    ax.set_xlabel("seconds"); ax.set_ylabel("mel band"); ax.set_title("Spectrogram analysed", loc="left", fontsize=10)
    fig.tight_layout()
    url = _png_figure(fig)
    plt.close(fig)
    return {"kind": "spectrogram", "image": url}


def explain(content, modality):
    return {"text": explain_text, "code": explain_code, "image": explain_image, "audio": explain_audio}[modality](content)
