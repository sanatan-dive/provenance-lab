"""Report figures: architecture and pipeline diagrams plus result charts, drawn from code/results.

    ../../code/.venv/bin/python make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = Path(__file__).resolve().parents[2] / "code/results"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
J = lambda rel: json.loads((RES / rel).read_text())

INK, SOFT, UV, UVW, GREEN, ORANGE, GREY, LINE = "#13202A", "#4A5A66", "#5A3FD0", "#ECE8FB", "#1D7A57", "#C24A12", "#9AA8B1", "#CDD6DB"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": LINE,
                     "axes.labelcolor": SOFT, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.color": SOFT, "ytick.color": SOFT, "legend.frameon": False})


def box(ax, x, y, w, h, title, sub, fill, fg="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=fill))
    ax.text(x + 0.15, y + h - 0.28, title, color=fg, fontsize=11, fontweight="bold", va="top")
    ax.text(x + 0.15, y + h - 0.68, sub, color=fg if fg == "white" else SOFT, fontsize=8.5, va="top", wrap=True, linespacing=1.4)


def arrow(ax, a, b, label=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=SOFT, lw=1.4))
    if label:
        horizontal = abs(a[1] - b[1]) < 0.01
        ax.text((a[0] + b[0]) / 2 + (0 if horizontal else 0.1), (a[1] + b[1]) / 2 + (0.22 if horizontal else 0), label,
                fontsize=8, color=SOFT, va="center", ha="center" if horizontal else "left")


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off")
    box(ax, 0.2, 2.1, 2.2, 1.5, "Input", "Essay, Python file,\nimage or voice clip", "#E6ECEE", INK)
    box(ax, 3.2, 3.4, 3.6, 1.7, "Layer 1: provenance", "C2PA content credentials\nAudioSeal audio watermark\nText watermark (green-list, SynthID)", UV)
    box(ax, 3.2, 0.4, 3.6, 2.2, "Layer 2: trained detector", "Deep encoder + forensic features\nBERT · CodeBERT · ViT+FFT · wav2vec 2.0\nThreshold: >70% AI, <30% human", INK)
    box(ax, 8.2, 3.4, 3.4, 1.7, "AI-generated (proof)", "Verifiable signal found;\nstop here", ORANGE)
    box(ax, 8.2, 0.4, 3.4, 2.2, "Verdict + evidence", "Human / AI / uncertain\nConfidence score\nHighlights, heatmap, spectrogram", "#E6ECEE", INK)
    arrow(ax, (2.4, 2.85), (3.2, 4.1))
    arrow(ax, (6.8, 4.25), (8.2, 4.25), "signal found")
    arrow(ax, (5.0, 3.4), (5.0, 2.6), "nothing found")
    arrow(ax, (6.8, 1.5), (8.2, 1.5))
    fig.tight_layout(); fig.savefig(OUT / "architecture.png", dpi=200); plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 12.5); ax.set_ylim(0.3, 2.0); ax.axis("off")
    steps = [("Data", "Public benchmarks,\nleakage-free splits"), ("Preprocess", "Clean text, trim audio,\nnormalise images"),
             ("Features", "Perplexity, AST, FFT,\nMFCC / phase"), ("Encode", "Pretrained encoder\nfine-tuned or frozen"),
             ("Train", "BCE + SupCon loss,\nearly stopping"), ("Evaluate", "Held-out test, OOD,\nbaselines, public tools")]
    for i, (t, s) in enumerate(steps):
        x = 0.15 + i * 2.07
        box(ax, x, 0.4, 1.8, 1.5, f"{i + 1}. {t}", s, UVW, INK)
        if i < len(steps) - 1:
            arrow(ax, (x + 1.8, 1.15), (x + 2.07, 1.15))
    fig.tight_layout(); fig.savefig(OUT / "pipeline.png", dpi=200); plt.close(fig)


def bars(ax, labels, series, colors, title, ylim=(0, 100), fmt="{:.1f}"):
    n = len(series)
    w = 0.8 / n
    for k, (name, vals) in enumerate(series):
        xs = [i + (k - (n - 1) / 2) * w for i in range(len(labels))]
        bs = ax.bar(xs, vals, w * 0.92, label=name, color=colors[k])
        for b_, v in zip(bs, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 1, fmt.format(v), ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)), labels); ax.set_ylim(ylim[0], ylim[1] * 1.18); ax.set_title(title, loc="left")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E6ECEE"); ax.set_axisbelow(True)
    if n > 1:
        ax.legend(loc="upper left", ncol=2, fontsize=8)


def results_chart():
    t = J("text/text_bert_supcon_metrics.json"); c = J("code/code_codebert_supcon_12k_metrics.json")
    i = J("image/image_genimage_metrics.json"); vm = J("ood/voice_modern.json"); div = J("ood/mage_diverse_12k.json")["calibrated"]
    labels = ["Essays\n(HC3)", "Essays\n(MAGE, unseen)", "Code\n(CoDET-M4)", "Images\n(GenImage)", "Voice\n(modern TTS)", "Voice\n(ASVspoof)"]
    vals = [t["accuracy"], div["mage_accuracy"], c["accuracy"], i["accuracy"], vm["new_model_on_modern"], vm["new_model_on_asvspoof"]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars(ax, labels, [("Accuracy", [v * 100 for v in vals])], [UV], "Test accuracy of the deployed detectors (%)")
    fig.tight_layout(); fig.savefig(OUT / "results.png", dpi=200); plt.close(fig)


def public_chart():
    m = J("market/market.json"); div = J("ood/mage_diverse_12k.json")["calibrated"]
    i = J("image/image_genimage_metrics.json"); vm = J("ood/voice_modern.json")
    best = lambda mod, s: max(r[s]["accuracy"] for r in m[mod].values() if s in r) * 100
    labels = ["Essays (MAGE)", "Images (GenImage)", "Voice (modern TTS)"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars(ax, labels, [("Best public detector", [best("text", "mage"), best("image", "genimage"), best("audio", "modern")]),
                      ("Our system", [div["mage_accuracy"] * 100, i["accuracy"] * 100, vm["new_model_on_modern"] * 100])],
         [GREY, UV], "Accuracy on current AI content (%)")
    fig.tight_layout(); fig.savefig(OUT / "public_compare.png", dpi=200); plt.close(fig)


def generalisation_chart():
    o = J("ood/mage.json"); div = J("ood/mage_diverse_12k.json")["calibrated"]; vm = J("ood/voice_modern.json")
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bars(axs[0], ["HC3 (seen)", "MAGE (unseen)"], [("Trained on HC3 only", [o["models"][next(iter(o["models"]))]["hc3"] * 100, o["models"][next(iter(o["models"]))]["mage"]["accuracy"] * 100]),
                                                   ("Diverse data + calibration", [div["hc3_accuracy"] * 100, div["mage_accuracy"] * 100])],
         [GREY, UV], "Essays: seen vs unseen AI models (%)")
    bars(axs[1], ["ASVspoof 2019", "Modern TTS"], [("Trained on 2019 data only", [vm["old_model_on_asvspoof"] * 100, vm["old_model_on_modern"] * 100]),
                                                   ("+ modern TTS", [vm["new_model_on_asvspoof"] * 100, vm["new_model_on_modern"] * 100])],
         [GREY, UV], "Voice: 2019 attacks vs modern clones (%)")
    fig.tight_layout(); fig.savefig(OUT / "generalisation.png", dpi=200); plt.close(fig)


def genimage_chart():
    g = J("image/image_genimage_per_generator.json"); h = J("image/image_genimage_nomidjourney_per_generator.json")
    keys = list(g)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    bars(ax, ["Real" if k == "Real" else k for k in keys], [("All generators in training", [g[k]["accuracy"] * 100 for k in keys]),
                                                            ("Midjourney held out", [h[k]["accuracy"] * 100 for k in keys])],
         [UV, ORANGE], "Images: % correct per source (GenImage test)", fmt="{:.0f}")
    fig.tight_layout(); fig.savefig(OUT / "genimage.png", dpi=200); plt.close(fig)


def watermark_chart():
    r = J("provenance/robustness.json")
    a = r["audio"]["attacks"]; t = r["text"]["schemes"]["kgw"]["attacks"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, d, title in ((axs[0], a, "AudioSeal: % of watermarked clips detected"), (axs[1], t, "Text watermark: % detected")):
        ks = list(d); vs = [d[k]["detection_rate"] * 100 for k in ks]
        ax.barh(ks, vs, color=[UV if v >= 50 else ORANGE for v in vs]); ax.set_xlim(0, 105); ax.set_title(title, loc="left")
        for y, v in enumerate(vs):
            ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "watermarks.png", dpi=200); plt.close(fig)


for f in (architecture, pipeline, results_chart, public_chart, generalisation_chart, genimage_chart, watermark_chart):
    f()
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
