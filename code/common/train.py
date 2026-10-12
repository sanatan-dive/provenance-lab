"""Generic train/eval loop shared by all modalities.

A modality supplies:
  - model: nn.Module whose forward(batch) returns (logit, z)
  - DataLoaders yielding dict batches that contain a "label" tensor (1 = AI, 0 = human)
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import metrics as _style  # noqa: F401  (applies the shared plot style)
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .metrics import evaluate_and_save
from .supcon import supcon_loss


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device, max_batches=None):
    model.eval()
    ys, ps = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, device)
        logit, _ = model(batch)
        ys.append(batch["label"].cpu())
        ps.append(torch.sigmoid(logit).float().cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def eval_loss_acc(model, loader, device, max_batches=None):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, device)
        logit, _ = model(batch)
        y = batch["label"]
        tot += F.binary_cross_entropy_with_logits(logit, y.float(), reduction="sum").item()
        correct += ((logit > 0).long() == y).sum().item()
        n += len(y)
    return tot / n, correct / n


def split_val(train_loader, frac=0.1, seed=0):
    """Hold out `frac` of the training set as a validation set (same batch size and collate)."""
    ds = train_loader.dataset
    n_val = max(1, int(len(ds) * frac))
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
    kw = dict(batch_size=train_loader.batch_size, collate_fn=train_loader.collate_fn, num_workers=0)
    return DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, shuffle=False, **kw)


def plot_curves(hist, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ep = [h["epoch"] for h in hist]
    best = min(hist, key=lambda h: h["val_loss"])["epoch"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.0))
    for a, key, title in ((ax[0], "loss", "Loss (BCE)"), (ax[1], "acc", "Accuracy")):
        a.plot(ep, [h[f"train_{key}"] for h in hist], "o-", color="#9AA8B1", lw=2, label="train")
        a.plot(ep, [h[f"val_{key}"] for h in hist], "o-", color="#5A3FD0", lw=2, label="validation")
        a.axvline(best, color="#1D7A57", ls=":", lw=1.2, label=f"best epoch ({best})")
        a.set_title(title, loc="left"); a.set_xlabel("epoch"); a.set_xticks(ep)
    ax[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{name}_curves.png", dpi=170); plt.close(fig)


def fit(model, train_loader, test_loader, out_dir, name, epochs=2, lr=2e-5, head_lr=1e-3,
        supcon_weight=0.5, smoke=False, extra=None, fast_prefixes=("head.",), patience=2, max_grad_norm=None):
    """Train with BCE + supcon_weight * SupCon, then evaluate on the test set.

    10% of the training set is held out for validation. After each epoch we record train/val
    loss and accuracy; the weights from the epoch with the lowest validation loss are restored
    (early stopping after `patience` epochs without improvement). Learning curves are saved
    as evidence the model is neither over- nor underfitting.

    smoke=True runs 3 train batches and 2 test batches, only to prove the pipeline works.
    fast_prefixes: parameter-name prefixes trained from scratch at head_lr (the rest are
    pretrained and use lr).
    """
    device = get_device()
    model.to(device)
    train_loader, val_loader = split_val(train_loader)
    head_params = [p for n, p in model.named_parameters() if n.startswith(fast_prefixes)]
    enc_params = [p for n, p in model.named_parameters() if not n.startswith(fast_prefixes) and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr}, {"params": head_params, "lr": head_lr}],
                            weight_decay=0.01)
    max_train = 3 if smoke else None
    max_eval = 2 if smoke else None
    epochs = 1 if smoke else epochs
    t0 = time.time()
    hist, best, best_loss, bad = [], None, float("inf"), 0
    for ep in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            if max_train and i >= max_train:
                break
            batch = to_device(batch, device)
            y = batch["label"].float()
            logit, z = model(batch)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if supcon_weight > 0:
                loss = loss + supcon_weight * supcon_loss(z, batch["label"])
            opt.zero_grad()
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            if i % 20 == 0:
                print(f"[{name}] epoch {ep + 1}/{epochs} step {i}/{len(train_loader)} "
                      f"loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        # Train metrics in eval mode on a same-size sample, so they compare fairly with validation.
        tr_loss, tr_acc = eval_loss_acc(model, train_loader, device, max_batches=max_eval or len(val_loader))
        va_loss, va_acc = eval_loss_acc(model, val_loader, device, max_batches=max_eval)
        hist.append({"epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": va_loss, "val_acc": va_acc})
        print(f"[{name}] epoch {ep + 1}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}", flush=True)
        if va_loss < best_loss:
            best_loss, bad = va_loss, 0
            best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stopping: no val improvement for {patience} epochs", flush=True)
                break
    best_epoch = min(hist, key=lambda h: h["val_loss"])["epoch"]
    model.load_state_dict(best)
    run_name = name + ("_smoke" if smoke else "")
    plot_curves(hist, out_dir, run_name)
    y, p = predict(model, test_loader, device, max_batches=max_eval)
    info = {"epochs_run": len(hist), "best_epoch": best_epoch, "supcon_weight": supcon_weight,
            "train_seconds": round(time.time() - t0), "smoke": smoke, "history": hist,
            "train_acc_best": hist[best_epoch - 1]["train_acc"], "val_acc_best": hist[best_epoch - 1]["val_acc"]}
    if extra:
        info.update(extra)
    m = evaluate_and_save(y, p, out_dir, run_name, extra=info)
    m["generalization_gap"] = m["train_acc_best"] - m["accuracy"]
    (Path(out_dir) / f"{run_name}_metrics.json").write_text(json.dumps(m, indent=2))
    print({k: v for k, v in m.items() if k != "history"}, flush=True)
    return model, m
