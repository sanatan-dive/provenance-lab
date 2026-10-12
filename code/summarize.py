"""Collect every results/*/*_metrics.json (excluding smoke runs) into results/summary.md."""
import json
from pathlib import Path

rows = []
for f in sorted(Path("results").glob("*/*_metrics.json")):
    m = json.loads(f.read_text())
    if m.get("smoke"):
        continue
    rows.append((f.parent.name, m))

lines = [
    "| Modality | Run | Contrastive weight | Test samples | Accuracy | Precision | Recall | F1 | AUC-ROC "
    "| Train acc | Gap | Best epoch | Train time |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for modality, m in rows:
    lines.append(
        f"| {modality} | {m['name']} | {m.get('supcon_weight', '-')} | {m['n_test']} | {m['accuracy']:.3f} | "
        f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['auc_roc']:.3f} | "
        f"{m.get('train_acc_best', 0):.3f} | {m.get('generalization_gap', 0):+.3f} | "
        f"{m.get('best_epoch', '-')}/{m.get('epochs_run', '-')} | {m.get('train_seconds', 0) // 60} min |"
    )
Path("results/summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
