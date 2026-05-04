"""Generic SSL pretraining loop, paradigm-agnostic.

The same loop runs G0 MAE / G1 AR / G2 MAR (and any future paradigm
plumbed through :mod:`exp03.paradigms`) — only the model config differs.
Built on:

* `accelerate <https://huggingface.co/docs/accelerate>`_ for DDP /
  multi-GPU readiness, mixed-precision (bf16), and gradient
  accumulation. We deliberately avoid HF :class:`Trainer` because our
  model output dict doesn't fit the Trainer's ``outputs.loss`` /
  ``outputs.logits`` contract — accelerate gives us the same wins
  (DDP, mixed precision, checkpointing) with a custom loop.

* `wandb <https://docs.wandb.ai/>`_ for metrics, gradient histograms,
  and config provenance. The ``WANDB_DISABLED=true`` env var disables
  it (useful for smoke tests).

* :mod:`exp03.eval` ``run_protocol_a`` for the end-of-training frozen-
  probe metrics (HBN 6-task BAC + WF1 + k-NN top-1 + CBCL R²), which
  get logged to wandb under ``eval/*``.

Recipe references:

* MAE (He et al. 2022 — `facebookresearch/mae <https://github.com/facebookresearch/mae>`_):
  ``main_pretrain.py`` / ``engine_pretrain.py`` are the canonical
  single-process / DDP loop. Cosine LR schedule with warmup, AdamW
  (β=0.9, 0.95, wd=0.05), per-iteration LR adjustment, ``norm_pix_loss``
  flag, accum_iter for effective batch size.
* ARM (Ren et al. ICLR 2025 — `OliverRensu/ARM <https://github.com/OliverRensu/ARM>`_):
  same recipe as MAE; we treat AR as an MAE-shaped problem with
  zero mask ratio.
* MAR (Li et al. NeurIPS 2024 — `LTH14/mar <https://github.com/LTH14/mar>`_):
  also follows MAE's recipe; the only paradigm-specific knob is
  ``--diffusion_batch_mul`` (replicate each sample N times to amortise
  the diffusion-noise sampling across multiple noise levels per
  example). We expose this as ``TrainConfig.diffusion_batch_mul``;
  default 1 (off) — turn on for the actual exp17 runs per the MAR
  paper's recipe.

Usage:

    >>> from exp03.train import TrainConfig, train
    >>> cfg = TrainConfig(paradigm="mae", max_steps=1000, batch_size=32,
    ...                   data_root="/opt/dlami/nvme/eeg/derived/hbn_minimal_500hz",
    ...                   output_dir="/opt/dlami/nvme/eeg/runs/exp17/g0_seed0",
    ...                   wandb_project="exp03",
    ...                   wandb_run_name="exp17-g0-seed0")
    >>> train(cfg)

CLI: see :mod:`exp03.cli` ``train`` subcommand.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as data_mod
from . import model as model_mod
from . import storage as storage_mod


# =============================================================================
# Train config
# =============================================================================


@dataclass
class TrainConfig:
    """All knobs for one training run (one paradigm × one seed).

    Defaults are scaled for a *small-scale validation cell*: 1000 steps
    on a single GPU at batch 32. For the actual exp17 cells, bump
    ``max_steps`` (or set ``max_tokens``) per the README's iso-data
    budget of 35M tokens-seen. The CLI exposes overrides for every
    field.
    """

    # --- model -----------------------------------------------------------
    paradigm: str = "mae"                                # "mae" | "ar" | "mar"
    backbone_kind: str = "mamba2"                        # "mamba2" | "transformer"
    backbone_layers: int = 6
    backbone_d_model: int = 256
    backbone_bidirectional: bool | None = None           # default: True for mae/mar, False for ar
    decoder_layers: int = 2
    mask_ratio: float = 0.50
    window_samples: int = 2000

    # --- MAR-specific knobs (ignored for other paradigms) ---------------
    diffloss_d: int = 3
    diffloss_w: int = 1024
    num_diffusion_steps: int = 1000
    diffusion_batch_mul: int = 1                         # MAR's batch-replication trick

    # --- data ------------------------------------------------------------
    data_root: Path | None = None                        # required at runtime
    max_windows_per_shard: int | None = None             # cap iid windows per parquet shard
    subject_filter: tuple[str, ...] | None = None        # train-split subjects (None = all)

    # --- optimisation ---------------------------------------------------
    max_steps: int = 1000
    batch_size: int = 32                                  # per-GPU
    accum_iter: int = 1
    lr: float = 1e-4
    blr: float | None = None                              # base LR; if set, lr = blr * eff_batch / 256
    weight_decay: float = 0.05
    adam_betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    precision: str = "bf16"                               # "fp32" | "bf16" | "fp16"

    # --- runtime / logging ----------------------------------------------
    seed: int = 0
    output_dir: Path | None = None                        # required at runtime
    log_every: int = 20
    ckpt_every: int = 0                                   # 0 = checkpoint at end only
    eval_at_end: bool = True
    eval_max_subjects: int = 50
    eval_max_windows_per_shard: int = 20
    num_workers: int = 2                                  # DataLoader workers (set 0 for macOS dev / stdin scripts)

    # --- wandb -----------------------------------------------------------
    wandb_project: str = "exp03"
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: str = "online"                            # "online" | "offline" | "disabled"

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d


# =============================================================================
# Helpers
# =============================================================================


def _build_model_config(t: TrainConfig) -> model_mod.ModelConfig:
    """Map a :class:`TrainConfig` onto the lower-level :class:`ModelConfig`."""
    bidi = t.backbone_bidirectional
    if bidi is None:
        bidi = (t.paradigm != "ar")  # AR is unidirectional by spec
    return model_mod.ModelConfig(
        frontend=model_mod.FrontendConfig(d_model=t.backbone_d_model),
        backbone=model_mod.BackboneConfig(
            kind=t.backbone_kind,
            n_layers=t.backbone_layers,
            d_model=t.backbone_d_model,
            bidirectional=bidi,
        ),
        decoder=model_mod.DecoderConfig(
            kind="mamba2",                                # G0 uses Mamba decoder; G1/G2 unused
            n_layers=t.decoder_layers,
            d_model=t.backbone_d_model,
        ),
        mask=model_mod.MaskConfig(mask_ratio=t.mask_ratio),
        paradigm=model_mod.ParadigmConfig(kind=t.paradigm),
        target=model_mod.TargetConfig(kind="raw"),
        window_samples=t.window_samples,
    )


def _cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                      base_lr: float, min_lr_frac: float = 0.0) -> float:
    """MAE / DiT-style cosine LR with linear warmup.

    ``min_lr_frac`` is the floor as a fraction of ``base_lr`` (default 0).
    """
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)


def _build_data_loader(t: TrainConfig, accelerator) -> Iterator[dict]:
    """Wrap :class:`data.ParquetWindowDataset` in a torch DataLoader.

    Returns an *iterator* (not the DataLoader itself) — IterableDataset
    iteration restarts on every epoch, which is fine for our token-
    based budget where we never count epochs.
    """
    if t.data_root is None or not Path(t.data_root).exists():
        raise FileNotFoundError(f"data_root not found: {t.data_root}")
    sub_filter = set(t.subject_filter) if t.subject_filter else None
    ds = data_mod.ParquetWindowDataset(
        derived_root=Path(t.data_root),
        subject_filter=sub_filter,
        max_windows_per_shard=t.max_windows_per_shard,
        shuffle_within_shard=True,
        rng_seed=t.seed,
    )
    loader = DataLoader(
        ds,
        batch_size=t.batch_size,
        num_workers=t.num_workers,
        collate_fn=data_mod.collate_signal_batch,
        pin_memory=t.num_workers > 0,
        drop_last=True,
    )
    return loader


# =============================================================================
# Main entry point
# =============================================================================


def train(t: TrainConfig) -> dict[str, Any]:
    """Run one training cell. Returns the final metrics dict."""
    if t.output_dir is None:
        raise ValueError("TrainConfig.output_dir is required")
    output_dir = Path(t.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- accelerate setup ----------------------------------------------
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    accelerator = Accelerator(
        mixed_precision={"fp32": "no", "bf16": "bf16", "fp16": "fp16"}[t.precision],
        gradient_accumulation_steps=t.accum_iter,
    )
    set_seed(t.seed, device_specific=True)

    # ---- wandb (rank 0 only) -------------------------------------------
    wandb = None
    if accelerator.is_main_process and t.wandb_mode != "disabled":
        try:
            import wandb as _wandb
            _wandb.init(
                project=t.wandb_project,
                name=t.wandb_run_name,
                tags=list(t.wandb_tags),
                mode=t.wandb_mode,
                config=t.to_dict(),
                dir=str(output_dir),
            )
            wandb = _wandb
        except Exception as e:                                     # noqa: BLE001
            print(f"[train] wandb init failed: {e}; continuing without wandb")
            wandb = None

    def log(metrics: dict, step: int):
        if wandb is not None:
            wandb.log(metrics, step=step)
        if accelerator.is_main_process and step % t.log_every == 0:
            entry = {"step": step, **{k: round(v, 6) if isinstance(v, float) else v
                                       for k, v in metrics.items()}}
            print(json.dumps(entry, default=str))

    # ---- build model + optimiser ---------------------------------------
    mcfg = _build_model_config(t)
    if accelerator.is_main_process:
        print(f"[train] paradigm={t.paradigm} backbone={t.backbone_kind} "
              f"d_model={t.backbone_d_model} bidir={mcfg.backbone.bidirectional}")
    m = model_mod.build_model(mcfg)
    if accelerator.is_main_process:
        p = model_mod.count_params(m)
        print(f"[train] params: total={p['total']:,} trainable={p['trainable']:,}")

    # MAE-style absolute LR computation: lr = blr * eff_batch / 256
    eff_batch = t.batch_size * t.accum_iter * accelerator.num_processes
    if t.blr is not None:
        base_lr = t.blr * eff_batch / 256.0
    else:
        base_lr = t.lr

    optimizer = torch.optim.AdamW(
        m.parameters(),
        lr=base_lr, betas=t.adam_betas, weight_decay=t.weight_decay,
    )

    # ---- build data + prepare with accelerate --------------------------
    loader = _build_data_loader(t, accelerator)
    m, optimizer, loader = accelerator.prepare(m, optimizer, loader)

    # ---- train loop ----------------------------------------------------
    m.train()
    step = 0
    start = time.time()
    losses_window: list[float] = []
    data_iter = iter(loader)
    while step < t.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)                                # rewind IterableDataset
            batch = next(data_iter)

        x = batch["signal"]                                          # (B, T_samples) float32

        # MAR-specific: replicate each sample to get more noise levels per example
        if t.paradigm == "mar" and t.diffusion_batch_mul > 1:
            x = x.repeat(t.diffusion_batch_mul, 1)

        # Per-iteration cosine LR
        lr = _cosine_warmup_lr(step, t.max_steps, t.warmup_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with accelerator.accumulate(m):
            out = m(x, compute_loss=True)
            loss = out["loss"]
            accelerator.backward(loss)
            if accelerator.sync_gradients and t.grad_clip_norm > 0:
                accelerator.clip_grad_norm_(m.parameters(), t.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses_window.append(loss.item())
        if step % t.log_every == 0 or step == t.max_steps - 1:
            comps = out.get("components", {})
            metrics = {
                "train/loss": float(loss.item()),
                "train/loss_avg20": float(sum(losses_window[-20:]) / max(len(losses_window[-20:]), 1)),
                "train/lr": lr,
                "train/step_per_s": (step + 1) / max(time.time() - start, 1e-3),
                "train/eff_batch": eff_batch,
            }
            for k, v in comps.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    metrics[f"train/{k}"] = float(v)
            log(metrics, step)

        if t.ckpt_every > 0 and step > 0 and step % t.ckpt_every == 0 and accelerator.is_main_process:
            ckpt_path = output_dir / f"ckpt_step{step}.pt"
            unwrapped = accelerator.unwrap_model(m)
            torch.save({"step": step, "state_dict": unwrapped.state_dict(),
                        "config": t.to_dict()}, ckpt_path)
            print(f"[train] checkpoint -> {ckpt_path}")

        step += 1

    elapsed = time.time() - start
    if accelerator.is_main_process:
        print(f"[train] done: {step} steps in {elapsed:.1f}s "
              f"({step / max(elapsed, 1e-3):.1f} step/s)")

    # ---- final checkpoint ----------------------------------------------
    final_state: dict[str, Any] = {}
    if accelerator.is_main_process:
        ckpt_path = output_dir / "ckpt_final.pt"
        unwrapped = accelerator.unwrap_model(m)
        torch.save({"step": step, "state_dict": unwrapped.state_dict(),
                    "config": t.to_dict()}, ckpt_path)
        final_state["ckpt"] = str(ckpt_path)
        print(f"[train] final checkpoint -> {ckpt_path}")

    # ---- end-of-training eval (Protocol A frozen-probe) ----------------
    metrics_a: dict[str, Any] = {}
    if t.eval_at_end and t.data_root is not None and accelerator.is_main_process:
        from . import eval as eval_mod
        unwrapped = accelerator.unwrap_model(m).eval()
        try:
            extracted = eval_mod.extract_features(
                unwrapped, Path(t.data_root),
                max_subjects=t.eval_max_subjects,
                max_windows_per_shard=t.eval_max_windows_per_shard,
                seed=t.seed,
                device=accelerator.device.type,
            )
            metrics_a = eval_mod.run_protocol_a(extracted, seed=t.seed)
            wandb_metrics: dict[str, float] = {}
            for name, sub in metrics_a.items():
                if isinstance(sub, dict) and "point" in sub:
                    wandb_metrics[f"eval/{name}/point"] = float(sub["point"])
                    if "ci_low_95" in sub:
                        wandb_metrics[f"eval/{name}/ci_low_95"] = float(sub["ci_low_95"])
                        wandb_metrics[f"eval/{name}/ci_high_95"] = float(sub["ci_high_95"])
            log(wandb_metrics, step)
            print("[train] eval/Protocol A:")
            for k, v in metrics_a.items():
                print(f"  {k}: {v}")
        except Exception as e:                                     # noqa: BLE001
            print(f"[train] eval-at-end failed: {e}")
            metrics_a = {"error": str(e)}

    # ---- write summary -------------------------------------------------
    if accelerator.is_main_process:
        summary = {
            "config": t.to_dict(),
            "step": step,
            "elapsed_s": elapsed,
            "step_per_s": step / max(elapsed, 1e-3),
            "final_state": final_state,
            "eval_protocol_a": metrics_a,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"[train] summary -> {output_dir/'summary.json'}")

    if wandb is not None:
        wandb.finish()

    return {"step": step, "elapsed_s": elapsed, "metrics": metrics_a,
            "output_dir": str(output_dir)}
