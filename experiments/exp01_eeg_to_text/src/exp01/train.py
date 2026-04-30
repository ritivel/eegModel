"""3-stage training: alignment -> frozen-LM SFT -> LoRA SFT.

Single entry point ``train(cfg)`` returns the path to the final checkpoint.

Everything that helps debug a bad run is logged:
  - per-step:   loss, grad_norm, lr, stage  (W&B + log.jsonl)
  - per N step: K dev sample generations    (W&B Table + sample_gens.jsonl)
  - per stage:  encoder feature stats       (W&B + stats.jsonl)
  - per stage:  checkpoint                  (model_stage{1,2,3}.pt)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data, decoder, preprocessing, storage
from .config import CellConfig
from .model import EEG2Text


# ============================================================================
# Collator
# ============================================================================


def _collate(rows: list[dict], tokenizer, *, target_sr: int | None,
             max_text_tokens: int = 96, min_T: int = 200):
    """Right-pad to (B, Cmax, Tmax). Per-row time resample to ``target_sr``
    (when set) so mixed-source batches share a single sampling rate. Time
    axis is also padded up to ``min_T`` (default 200 — REVE's patch_size,
    one second at 200 Hz) so even short sentences make it through encoders
    that require a minimum window.
    """
    Bsz = len(rows)
    eeg_arrs = []
    for r in rows:
        a = r["eeg"]                               # (C, T) float32
        if target_sr is not None and abs(r["sr"] - target_sr) > 0.5:
            new_T = max(1, int(round(a.shape[1] * target_sr / r["sr"])))
            x = torch.from_numpy(a).unsqueeze(0)   # (1, C, T)
            x = torch.nn.functional.interpolate(x, size=new_T, mode="linear", align_corners=False)
            a = x.squeeze(0).numpy()
        eeg_arrs.append(a)

    Cmax = max(a.shape[0] for a in eeg_arrs)
    Tmax = max(min_T, max(a.shape[1] for a in eeg_arrs))
    eeg = torch.zeros(Bsz, Cmax, Tmax, dtype=torch.float32)
    for i, a in enumerate(eeg_arrs):
        c, t = a.shape
        eeg[i, :c, :t] = torch.from_numpy(a)

    channels = max((r["channels"] for r in rows), key=len)
    while len(channels) < Cmax:
        channels = list(channels) + [f"ch{len(channels):03d}"]

    text = [r["text"] for r in rows]
    enc = tokenizer(
        text, padding=True, truncation=True, max_length=max_text_tokens, return_tensors="pt"
    )
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {
        "eeg": eeg,
        "sr": float(target_sr if target_sr is not None else rows[0]["sr"]),
        "channels": list(channels),
        "text": text,
        "subject_ids": [r["participant_id"] for r in rows],
        "datasets": [r["dataset"] for r in rows],
        "text_input_ids": enc["input_ids"],
        "text_attention_mask": enc["attention_mask"],
        "labels": labels,
    }


# ============================================================================
# Training loop
# ============================================================================


def train(cfg: CellConfig) -> Path | None:
    storage.ensure_dirs()
    run_dir = storage.cell_run_dir(cfg.cell_id)
    log = open(run_dir / "log.jsonl", "a")
    samples_log = open(run_dir / "sample_gens.jsonl", "a")
    stats_log = open(run_dir / "stats.jsonl", "a")

    fold = data.load_fold(cfg.fold)

    # noise_test reuses the eeg checkpoint; train nothing.
    if cfg.input == "noise_test":
        twin = run_dir.parent / cfg.cell_id.replace("_noise_test_", "_eeg_")
        (run_dir / "uses_checkpoint_from.txt").write_text(str(twin))
        return None

    # Per-row preprocessing pipeline. Default ``v1`` is a no-op (preserves
    # the Apr-30 pilot's behaviour); ``v2`` resolves to encoder-specific
    # REVE/TFM recipes (bandpass + notch + 200 Hz polyphase resample +
    # per-recording z-score + 15-σ clip). See ``preprocessing.for_encoder``.
    pp_spec = (preprocessing.for_encoder(cfg.preprocess, cfg.encoder)
               if cfg.preprocess != "v1" else None)

    # Subject-independent: train sees only training subjects; inputs become
    # noise only when ``cfg.input == "noise_train"`` (Jo et al. §4.3).
    train_ds = data.EEGSentenceDataset(
        sources=data.ALL_SOURCES,
        subject_filter=fold.train_subjects,
        sentence_filter=fold.train_sent_hashes,
        noise="gauss" if cfg.input == "noise_train" else None,
        preprocess=pp_spec,
    )
    dev_ds = data.EEGSentenceDataset(
        sources=data.ZUCO_SOURCES,
        subject_filter=fold.dev_subjects,
        sentence_filter=fold.dev_sent_hashes,
        noise="gauss" if cfg.input == "noise_train" else None,
        preprocess=pp_spec,
    )

    model = EEG2Text(cfg).to("cuda")
    tokenizer = model.dec.tokenizer
    target_sr = model.encoder.spec.native_sr

    def loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: _collate(b, tokenizer, target_sr=target_sr),
            pin_memory=True, persistent_workers=cfg.num_workers > 0,
        )

    train_dl = loader(train_ds, True)
    dev_dl = loader(dev_ds, False)

    # ---- W&B ---------------------------------------------------------------
    wb = _wandb_init(cfg, n_train=len(train_ds), n_dev=len(dev_ds))

    log.write(json.dumps({
        "event": "init", "n_train": len(train_ds), "n_dev": len(dev_ds),
        "encoder_feature_dim": model.encoder.spec.feature_dim,
        "encoder_native_sr": model.encoder.spec.native_sr,
        "decoder_vocab_size": model.dec.vocab_size,
        "vocab_offset": model.vocab_offset,
        "trainable_params_stage1": _count(model.trainables_stage1()),
    }) + "\n")
    log.flush()

    # Pin a fixed dev sample bank for periodic generations (debug-friendly).
    dev_samples = _dev_sample_bank(dev_dl, k=16)

    # ---- Stage 1: alignment ------------------------------------------------
    # Effective contrastive batch size: bs × grad_accum (we average over
    # grad_accum sub-batches; the contrastive loss is computed per sub-batch).
    align_w = cfg.stage1_align_weight if cfg.batch_size >= 2 else 0.0
    _run_stage(
        model=model, dl=train_dl,
        params=model.trainables_stage1(),
        steps=cfg.stage1_steps, lr=cfg.stage1_lr,
        grad_accum=cfg.grad_accum, log=log, samples_log=samples_log,
        stats_log=stats_log, wb=wb, name="stage1",
        dev_samples=dev_samples, generate_every=max(1, cfg.stage1_steps // 4),
        align_weight=align_w, align_temperature=cfg.stage1_align_temperature,
        rvq_commit_weight=cfg.rvq_commit_weight,
    )
    _save(model, cfg, run_dir / "model_stage1.pt", log, wb, name="stage1")

    # ---- Stage 2: frozen-LM SFT --------------------------------------------
    _run_stage(
        model=model, dl=train_dl,
        params=model.trainables_stage2(),
        steps=cfg.stage2_steps, lr=cfg.stage2_lr,
        grad_accum=cfg.grad_accum, log=log, samples_log=samples_log,
        stats_log=stats_log, wb=wb, name="stage2",
        dev_samples=dev_samples, generate_every=max(1, cfg.stage2_steps // 4),
        align_weight=0.0, align_temperature=cfg.stage1_align_temperature,
        rvq_commit_weight=cfg.rvq_commit_weight,
    )
    _save(model, cfg, run_dir / "model_stage2.pt", log, wb, name="stage2")

    # ---- Stage 3: LoRA SFT (optional) --------------------------------------
    if cfg.use_lora_in_stage3:
        model.dec.model = decoder.attach_lora(model.dec.model, r=cfg.lora_r, alpha=cfg.lora_alpha)
        _run_stage(
            model=model, dl=train_dl,
            params=model.trainables_stage3(),
            steps=cfg.stage3_steps, lr=cfg.stage3_lr,
            grad_accum=cfg.grad_accum, log=log, samples_log=samples_log,
            stats_log=stats_log, wb=wb, name="stage3",
            dev_samples=dev_samples, generate_every=max(1, cfg.stage3_steps // 4),
            align_weight=0.0, align_temperature=cfg.stage1_align_temperature,
            rvq_commit_weight=0.0,
        )

    # ---- Save final --------------------------------------------------------
    final_ckpt = run_dir / "model.pt"
    _save(model, cfg, final_ckpt, log, wb, name="final")
    log.write(json.dumps({"event": "done", "ckpt": str(final_ckpt)}) + "\n")
    for f in (log, samples_log, stats_log):
        f.close()
    # Note: don't call wb.finish() here. evaluate_cell() reuses the active
    # run to log eval metrics + the predictions table; cli/pilot orchestrates
    # finishing per cell via wandb.finish() between cells.
    return final_ckpt


# ============================================================================
# Stage runner
# ============================================================================


def _run_stage(
    *,
    model, dl, params, steps: int, lr: float, grad_accum: int,
    log, samples_log, stats_log, wb,
    name: str, dev_samples: list[dict], generate_every: int,
    align_weight: float = 0.0,
    align_temperature: float = 0.07,
    rvq_commit_weight: float = 0.0,
):
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=max(1, steps // 20)
    )
    model.train()
    it = iter(_inf(dl))
    t0 = time.time()

    log.write(json.dumps({
        "event": "stage_start", "stage": name, "steps": steps, "lr": lr,
        "trainable_params": _count(params),
        "align_weight": align_weight,
        "rvq_commit_weight": rvq_commit_weight,
    }) + "\n"); log.flush()

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        align_acc = 0.0
        commit_acc = 0.0
        for _ in range(grad_accum):
            batch = next(it)
            batch = {
                k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            out = model(
                eeg=batch["eeg"], sr=batch["sr"], channels=batch["channels"],
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                labels=batch["labels"],
            )
            lm_loss = out.loss
            extra = []

            aux = getattr(model, "_last_aux", None) or {}
            if align_weight > 0 and aux.get("bridge_pooled") is not None \
               and aux["bridge_pooled"].size(0) >= 2:
                a_loss = _infonce_align(
                    aux["bridge_pooled"], aux["text_pooled"], align_temperature,
                )
                extra.append(align_weight * a_loss)
                align_acc += float(a_loss.detach())
            commit = aux.get("commit_loss", 0.0)
            if rvq_commit_weight > 0 and isinstance(commit, torch.Tensor):
                extra.append(rvq_commit_weight * commit)
                commit_acc += float(commit.detach())

            total = lm_loss + sum(extra) if extra else lm_loss
            (total / grad_accum).backward()
            loss_acc += float(lm_loss.detach())
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        sched.step()
        lr_now = float(sched.get_last_lr()[0])

        if step % 10 == 0 or step == 1:
            payload = {
                "stage": name, "step": step,
                "loss": loss_acc / grad_accum,
                "align_loss": align_acc / grad_accum,
                "commit_loss": commit_acc / grad_accum,
                "grad_norm": gnorm,
                "lr": lr_now,
                "elapsed": round(time.time() - t0, 1),
            }
            log.write(json.dumps(payload) + "\n"); log.flush()
            if wb is not None:
                wb_payload = {f"{name}/loss": payload["loss"],
                              f"{name}/grad_norm": gnorm,
                              f"{name}/lr": lr_now,
                              "stage": _stage_idx(name), "step": step}
                if align_weight > 0:
                    wb_payload[f"{name}/align_loss"] = payload["align_loss"]
                if rvq_commit_weight > 0:
                    wb_payload[f"{name}/commit_loss"] = payload["commit_loss"]
                wb.log(wb_payload)

        # Periodic sample generations (debug)
        if step % generate_every == 0 or step == steps:
            _log_dev_samples(model, dev_samples, samples_log, wb,
                             stage=name, step=step)
            _log_feature_stats(model, dev_samples, stats_log, wb,
                               stage=name, step=step)
            model.train()  # generate() flips to eval


def _infonce_align(bridge_pooled: torch.Tensor, text_pooled: torch.Tensor,
                   temperature: float) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE between bridge and text pooled vectors.

    bridge_pooled: (B, d) — has gradient through the bridge / RVQ / new
                   embed rows.
    text_pooled:   (B, d) — frozen text-token embeddings (no grad).
    Returns a scalar loss that the trainer scales by ``align_weight``.

    Compute in fp32 for numerical stability (Gemma runs in bf16).
    """
    b = F.normalize(bridge_pooled.float(), dim=-1)
    t = F.normalize(text_pooled.float(), dim=-1)
    logits = b @ t.t() / max(temperature, 1e-6)
    targets = torch.arange(b.size(0), device=b.device)
    return 0.5 * (F.cross_entropy(logits, targets)
                  + F.cross_entropy(logits.t(), targets))


# ============================================================================
# Helpers — generation samples + feature stats during training
# ============================================================================


def _dev_sample_bank(dev_dl: DataLoader, *, k: int) -> list[dict]:
    """Pull a small fixed bank of dev batches for repeated sample generation."""
    bank, n = [], 0
    for batch in dev_dl:
        bank.append(batch)
        n += batch["eeg"].size(0)
        if n >= k:
            break
    return bank


@torch.no_grad()
def _log_dev_samples(model, banks: list[dict], samples_log, wb, *, stage: str, step: int):
    rows = []
    for b in banks:
        b_gpu = {kk: (v.to("cuda") if torch.is_tensor(v) else v) for kk, v in b.items()}
        try:
            gen = model.generate(eeg=b_gpu["eeg"], sr=b_gpu["sr"], channels=b_gpu["channels"])
        except Exception as e:
            gen = [f"<gen-error: {type(e).__name__}: {e}>"] * len(b["text"])
        for ref, hyp, sid, dset in zip(b["text"], gen, b["subject_ids"], b["datasets"]):
            rows.append({"stage": stage, "step": step, "subject": sid, "dataset": dset,
                         "ref": ref, "hyp": hyp})
    for r in rows:
        samples_log.write(json.dumps(r) + "\n")
    samples_log.flush()
    if wb is not None and rows:
        import wandb
        tbl = wandb.Table(columns=["stage", "step", "subject", "dataset", "ref", "hyp"],
                          data=[[r["stage"], r["step"], r["subject"], r["dataset"], r["ref"], r["hyp"]]
                                for r in rows])
        wb.log({f"samples/{stage}_step{step}": tbl})


@torch.no_grad()
def _log_feature_stats(model, banks: list[dict], stats_log, wb, *, stage: str, step: int):
    """Encoder output mean/std/norm — catches dead encoders or NaNs early."""
    feats_all = []
    for b in banks:
        b_gpu = {kk: (v.to("cuda") if torch.is_tensor(v) else v) for kk, v in b.items()}
        try:
            f = model.encoder.encode(b_gpu["eeg"], b_gpu["sr"], b_gpu["channels"])
            feats_all.append(f.flatten().float().cpu())
        except Exception as e:
            stats_log.write(json.dumps({"stage": stage, "step": step,
                                        "encode_error": f"{type(e).__name__}: {e}"}) + "\n")
            return
    if not feats_all:
        return
    flat = torch.cat(feats_all)
    payload = {
        "stage": stage, "step": step,
        "feat_mean": float(flat.mean()),
        "feat_std": float(flat.std()),
        "feat_abs_max": float(flat.abs().max()),
        "feat_nonzero_frac": float((flat != 0).float().mean()),
    }
    stats_log.write(json.dumps(payload) + "\n"); stats_log.flush()
    if wb is not None:
        wb.log({f"{stage}/feat_mean": payload["feat_mean"],
                f"{stage}/feat_std": payload["feat_std"],
                f"{stage}/feat_abs_max": payload["feat_abs_max"],
                "step": step})


# ============================================================================
# Wandb / save / misc
# ============================================================================


def _wandb_init(cfg: CellConfig, *, n_train: int, n_dev: int):
    if "WANDB_API_KEY" not in os.environ:
        print("[train] WANDB_API_KEY not set; W&B logging disabled.", flush=True)
        return None
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "exp01-eeg-to-text"),
        name=cfg.cell_id,
        group=f"{cfg.encoder}_{cfg.bridge}",
        tags=[cfg.encoder, cfg.bridge, cfg.input, f"fold{cfg.fold}"],
        config={**asdict(cfg), "n_train": n_train, "n_dev": n_dev},
        dir=str(storage.WANDB_DIR),
        resume="allow",
    )
    return run


def _save(model, cfg: CellConfig, path: Path, log, wb, *, name: str):
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg)}, path)
    size_mb = path.stat().st_size / 1024 / 1024
    log.write(json.dumps({"event": "save", "stage": name, "path": str(path),
                          "size_mb": round(size_mb, 1)}) + "\n")
    log.flush()
    if wb is not None:
        wb.log({f"checkpoint/{name}_size_mb": size_mb})


def _count(params) -> int:
    return sum(p.numel() for p in params if p.requires_grad)


def _stage_idx(name: str) -> int:
    return {"stage1": 1, "stage2": 2, "stage3": 3}.get(name, 0)


def _inf(dl):
    while True:
        for b in dl:
            yield b
