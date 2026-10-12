import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UV, INK, SOFT, LINE = "#5A3FD0", "#13202A", "#4A5A66", "#CDD6DB"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": LINE, "axes.labelcolor": SOFT, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6ECEE",
    "xtick.color": SOFT, "ytick.color": SOFT, "legend.frameon": False, "figure.facecolor": "white",
})
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


def evaluate_and_save(y_true, y_prob, out_dir, name, extra=None):
    """Label convention everywhere: 1 = AI-generated, 0 = human."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    m = {
        "name": name,
        "n_test": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
    }
    if extra:
        m.update(extra)
    (out / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
    np.savez(out / f"{name}_preds.npz", y_true=y_true, y_prob=y_prob)
    plot_roc_confusion(y_true, y_prob, m["auc_roc"], out, name)
    return m


def plot_roc_confusion(y_true, y_prob, auc, out, name):
    out = Path(out)
    y_pred = (y_prob >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(4, 3.4))
    plt.fill_between(fpr, tpr, color=UV, alpha=0.08)
    plt.plot(fpr, tpr, color=UV, lw=2.2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color=LINE, lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve", loc="left"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out / f"{name}_roc.png", dpi=170); plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(3.6, 3.2))
    plt.imshow(cm, cmap="Purples")
    plt.grid(False)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)
    plt.xticks([0, 1], ["Human", "AI"]); plt.yticks([0, 1], ["Human", "AI"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(name); plt.tight_layout()
    plt.savefig(out / f"{name}_confusion.png", dpi=170); plt.close()
