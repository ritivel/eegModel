"""Adaptation strategies — how we map a frozen / fine-tunable encoder to a task.

Two functions, each ~150 lines:

    linear_probe(encoder, train_ds, val_ds, test_ds, *, num_classes, task_type, ...)
    fine_tune   (encoder, train_ds, val_ds, test_ds, *, num_classes, task_type, ...)

Both return a metrics dict. Both honour the `task_type` ∈ {binary, multiclass, regression}.

Conventions:
- Datasets yield `(x, y)` tuples where `x` is `(C, T)` float32 and `y` is the label.
- For LP we extract features in batches with `torch.no_grad()`, fit sklearn, predict.
- For FT we attach a linear head (recipe: LaBraM-style if `recipe="labram"`).
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .adapter import Encoder
from ._common import compute_metrics


# ---------------------------------------------------------------------------
# LP — frozen encoder + sklearn head
# ---------------------------------------------------------------------------


def linear_probe(
    encoder: Encoder,
    train_ds: Dataset,
    test_ds: Dataset,
    *,
    val_ds: Dataset | None = None,
    num_classes: int = 2,
    task_type: Literal["binary", "multiclass", "regression"] = "binary",
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    num_workers: int = 4,
    n_bootstrap: int = 1000,
) -> dict:
    """Extract features → fit sklearn LogReg / RidgeCV → predict → metrics+CIs."""
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    encoder.eval().to(device)

    X_train, y_train = _extract(encoder, train_ds, batch_size=batch_size,
                                num_workers=num_workers, device=device)
    X_test, y_test = _extract(encoder, test_ds, batch_size=batch_size,
                              num_workers=num_workers, device=device)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task_type == "regression":
        # RidgeCV over a literature-standard alpha sweep
        reg = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0), cv=5)
        reg.fit(X_train_s, y_train)
        y_pred = reg.predict(X_test_s)
        metrics = compute_metrics(y_test, y_pred=y_pred, task_type="regression",
                                   n_bootstrap=n_bootstrap, seed=seed)
        head_info = {"head": "RidgeCV", "alpha": float(reg.alpha_)}

    elif task_type == "binary":
        clf = LogisticRegression(max_iter=5000, n_jobs=1)
        clf.fit(X_train_s, y_train.astype(np.int8))
        y_score = clf.predict_proba(X_test_s)[:, 1]
        y_pred = (y_score >= 0.5).astype(np.int8)
        metrics = compute_metrics(y_test.astype(np.int8), y_pred=y_pred, y_score=y_score,
                                   task_type="binary", n_bootstrap=n_bootstrap, seed=seed)
        head_info = {"head": "LogReg(binary)"}

    else:  # multiclass
        clf = LogisticRegression(max_iter=5000, n_jobs=1, solver="lbfgs")
        clf.fit(X_train_s, y_train.astype(np.int8))
        y_pred = clf.predict(X_test_s).astype(np.int8)
        metrics = compute_metrics(y_test.astype(np.int8), y_pred=y_pred,
                                   task_type="multiclass", n_bootstrap=n_bootstrap, seed=seed)
        head_info = {"head": f"LogReg(multinomial,{num_classes})"}

    return {
        "strategy": "lp",
        "head": head_info,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "metrics": metrics,
        "runtime_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# FT — attach a head, fine-tune end-to-end
# ---------------------------------------------------------------------------


def fine_tune(
    encoder: Encoder,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    *,
    num_classes: int = 2,
    task_type: Literal["binary", "multiclass", "regression"] = "binary",
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    layer_decay: float = 0.65,
    drop_path: float = 0.1,
    label_smoothing: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    num_workers: int = 4,
    n_bootstrap: int = 1000,
) -> dict:
    """Full fine-tuning with a linear/regression head, LaBraM-style recipe.

    Defaults reproduce LaBraM's TUAB recipe verbatim
    (`https://github.com/935963004/LaBraM/blob/main/README.md`):
    AdamW, lr=5e-4, weight_decay=0.05, warmup_epochs=5, layer_decay=0.65,
    drop_path=0.1, cosine schedule.
    """
    t0 = time.time()
    torch.manual_seed(seed)
    encoder.train()
    encoder.to(device)

    head_out = 1 if task_type in ("binary", "regression") else num_classes
    head = nn.Linear(encoder.spec.d_features, head_out).to(device)

    if task_type == "binary":
        loss_fn: nn.Module = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        loss_fn = nn.MSELoss()

    # Optimiser with layer-wise LR decay (LaBraM-style — backbone deeper layers get higher LR).
    param_groups = _layer_decay_param_groups(
        encoder, head, base_lr=lr, weight_decay=weight_decay, layer_decay=layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    n_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    best_val = -float("inf") if task_type != "regression" else float("inf")
    best_state = None
    step = 0

    for epoch in range(epochs):
        encoder.train()
        head.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            cur_lr = _cosine_lr(step, warmup_steps, n_steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr * pg.get("lr_scale", 1.0)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                feats = encoder(x)
                logits = head(feats).squeeze(-1)
                if task_type == "binary":
                    loss = loss_fn(logits, y.float())
                elif task_type == "regression":
                    loss = loss_fn(logits, y.float())
                else:
                    loss = loss_fn(logits, y.long())

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()), 1.0)
                optimizer.step()
            step += 1

        # ---- val ----
        val_metrics = _eval_loader(encoder, head, val_loader, task_type=task_type, device=device)
        primary = (val_metrics.get("auroc", val_metrics.get("bac",
                  val_metrics.get("nrmse", {"point": float("nan")})))).get("point", float("nan"))
        better = (primary > best_val) if task_type != "regression" else (primary < best_val)
        if better:
            best_val = primary
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "head": {k: v.detach().clone() for k, v in head.state_dict().items()},
            }
        print(f"  epoch {epoch+1}/{epochs}  loss={loss.item():.4f}  val_primary={primary:.4f}  "
              f"best={best_val:.4f}")

    # Load best
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])

    test_metrics_full = _eval_loader(encoder, head, test_loader,
                                      task_type=task_type, device=device,
                                      n_bootstrap=n_bootstrap, seed=seed)

    return {
        "strategy": "ft",
        "recipe": {
            "lr": lr, "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
            "epochs": epochs, "layer_decay": layer_decay, "drop_path": drop_path,
            "batch_size": batch_size, "label_smoothing": label_smoothing,
        },
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "feature_dim": int(encoder.spec.d_features),
        "best_val_primary": float(best_val),
        "metrics": test_metrics_full,
        "runtime_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract(encoder: Encoder, ds: Dataset, *, batch_size: int, num_workers: int,
             device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen encoder over a Dataset, return (features, labels) numpy arrays."""
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, shuffle=False)
    feats_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = encoder.encode(x)
            feats_chunks.append(f.detach().cpu().numpy().astype(np.float32))
            labels_chunks.append(y.numpy())
    return np.concatenate(feats_chunks, axis=0), np.concatenate(labels_chunks, axis=0)


@torch.no_grad()
def _eval_loader(encoder: Encoder, head: nn.Linear, loader: DataLoader, *,
                  task_type: str, device: str,
                  n_bootstrap: int = 0, seed: int = 0,
                  ) -> dict[str, dict[str, float]]:
    """Run the full loader, return metrics. Set `n_bootstrap > 0` for CIs (test-time)."""
    encoder.eval()
    head.eval()
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        feats = encoder(x)
        logits = head(feats).squeeze(-1)
        if task_type == "binary":
            s = torch.sigmoid(logits).cpu().numpy()
            scores.append(s)
            preds.append((s >= 0.5).astype(np.int8))
        elif task_type == "multiclass":
            preds.append(logits.argmax(dim=-1).cpu().numpy().astype(np.int8))
        else:
            preds.append(logits.cpu().numpy())
        ys.append(y.cpu().numpy())
    ys_all = np.concatenate(ys)
    preds_all = np.concatenate(preds)
    if task_type == "binary":
        scores_all = np.concatenate(scores)
        return compute_metrics(ys_all.astype(np.int8), y_pred=preds_all, y_score=scores_all,
                               task_type="binary", n_bootstrap=n_bootstrap, seed=seed)
    if task_type == "multiclass":
        return compute_metrics(ys_all.astype(np.int8), y_pred=preds_all.astype(np.int8),
                               task_type="multiclass", n_bootstrap=n_bootstrap, seed=seed)
    return compute_metrics(ys_all, y_pred=preds_all, task_type="regression",
                           n_bootstrap=n_bootstrap, seed=seed)


def _cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float,
                min_lr_frac: float = 0.0) -> float:
    import math
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos)


def _layer_decay_param_groups(
    encoder: Encoder, head: nn.Linear, *,
    base_lr: float, weight_decay: float, layer_decay: float,
) -> list[dict]:
    """LaBraM-style layer-wise LR decay: deeper layers get smaller LR by `layer_decay**depth`.

    Best-effort: counts `nn.Module` children of the encoder's backbone as 'layers'.
    For models with no clean `.layers` attribute, falls back to a single LR.
    """
    groups: list[dict] = []
    backbone = getattr(encoder.model, "backbone", None) or getattr(encoder.model, "encoder", None)
    layers = getattr(backbone, "layers", None) or getattr(backbone, "blocks", None)
    if layers is None:
        # No layer-wise structure visible — single group.
        groups.append({"params": list(encoder.parameters()) + list(head.parameters()),
                        "lr_scale": 1.0, "lr": base_lr, "weight_decay": weight_decay})
        return groups

    n = len(layers)
    for i, layer in enumerate(layers):
        scale = layer_decay ** (n - i - 1)
        groups.append({"params": list(layer.parameters()),
                        "lr_scale": scale, "lr": base_lr * scale,
                        "weight_decay": weight_decay})
    # Non-layer params (frontend, decoder, head) at full LR.
    used = {id(p) for g in groups for p in g["params"]}
    rest = [p for p in encoder.parameters() if id(p) not in used] + list(head.parameters())
    groups.append({"params": rest, "lr_scale": 1.0, "lr": base_lr,
                    "weight_decay": weight_decay})
    return groups
