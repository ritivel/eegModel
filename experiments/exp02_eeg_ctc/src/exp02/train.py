"""Single-stage end-to-end CTC trainer for exp02.

Training loop shape (no curriculum, no stages — Wav2Vec2 / Willett 2023
standard, see ``design.md``)::

    model = EEG2CTC(cfg, vocab).cuda()
    head_opt = AdamW(head.params,    lr=cfg.head_lr,    weight_decay=cfg.weight_decay)
    enc_opt  = AdamW(encoder.params, lr=cfg.encoder_lr, weight_decay=cfg.weight_decay)
    sched_h  = LinearWarmupCosineDecay(head_opt, warmup=cfg.warmup_steps, total=cfg.total_steps)
    sched_e  = LinearWarmupCosineDecay(enc_opt,  warmup=cfg.warmup_steps, total=cfg.total_steps)

    for step in range(cfg.total_steps):
        if step == 0:
            model.set_encoder_trainable(False)
        elif step == cfg.encoder_warmup_freeze_steps + 1:
            model.set_encoder_trainable(True)

        batch = next(it)
        feats = model.encoder_features(batch.eeg, batch.sr, batch.channels)
        out_a = model.head_forward(feats, aed_target_ids=...)

        loss = ctc_loss(out_a.logits, batch.targets) + maybe label-prior
        if intermediate_ctc:  loss += sum(ctc_loss(li, batch.targets))
        if cr_ctc:            feats_b = ...; out_b = ...; loss += KL(out_a, out_b)
        if aed:               loss = (1-λ) * loss + λ * cross_entropy(out.aed_logits, batch.targets_shifted)

        loss.backward(); clip_grad; head_opt.step(); enc_opt.step(); sched_h.step(); sched_e.step()

The encoder warmup-freeze is the closest thing to a "stage" we have, and it's
just a flag toggle inside one training loop — not a separate stage with its
own loss / its own data loader / its own LR schedule.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_common import augment, preprocessing, splits
from eeg_common.data import ALL_SOURCES, EEGSentenceDataset, ZUCO_SOURCES

from . import chars as chars_mod
from . import storage
from .chars import BLANK_ID
from .config import CTCConfig
from .model import EEG2CTC
from .text_augment import ParaphraseLookup


# ============================================================================
# Collator
# ============================================================================


def _collate(rows: list[dict], *, target_sr: int | None,
             min_T: int = 200, max_T_seconds: float = 12.0,
             paraphrase_lookup: "ParaphraseLookup | None" = None,
             text_aug_prob: float = 0.0,
             text_aug_rng: "np.random.Generator | None" = None):
    """Right-pad EEG to (B, Cmax, Tmax) and collect texts.

    With ``paraphrase_lookup`` set and ``text_aug_prob > 0``, replaces each
    row's reference text with a random paraphrase of itself with the given
    probability. The original text is kept in ``orig_text`` for diagnostics.
    """
    Bsz = len(rows)
    eeg_arrs = []
    for r in rows:
        a = r["eeg"]
        if target_sr is not None and abs(r["sr"] - target_sr) > 0.5:
            new_T = max(1, int(round(a.shape[1] * target_sr / r["sr"])))
            x = torch.from_numpy(a).unsqueeze(0)
            x = torch.nn.functional.interpolate(x, size=new_T, mode="linear",
                                                align_corners=False)
            a = x.squeeze(0).numpy()
        sr_eff = float(target_sr if target_sr is not None else r["sr"])
        max_T = int(round(max_T_seconds * sr_eff))
        if a.shape[1] > max_T:
            a = a[:, :max_T]
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

    orig_text = [r["text"] for r in rows]
    text = list(orig_text)
    if paraphrase_lookup is not None and text_aug_prob > 0 and text_aug_rng is not None:
        for i, t in enumerate(orig_text):
            if float(text_aug_rng.random()) < text_aug_prob:
                text[i] = paraphrase_lookup.sample(t, rng=text_aug_rng)

    return {
        "eeg": eeg,
        "sr": float(target_sr if target_sr is not None else rows[0]["sr"]),
        "channels": list(channels),
        "text": text,
        "orig_text": orig_text,
        "subject_ids": [r["participant_id"] for r in rows],
        "datasets": [r["dataset"] for r in rows],
    }


# ============================================================================
# LR schedule
# ============================================================================


def _linear_warmup_cosine_decay(step: int, *, warmup: int, total: int) -> float:
    """LR multiplier that ramps 0 -> 1 over ``warmup`` then cosine-decays
    1 -> 0 over the remaining steps.
    """
    if step < warmup:
        return float(step) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ============================================================================
# Loss components
# ============================================================================


def _ctc_loss_with_prior(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    target_lengths: torch.LongTensor,
    *,
    label_prior: torch.Tensor | None = None,
    label_prior_weight: float = 0.0,
) -> torch.Tensor:
    """CTC loss with optional label-prior subtraction (Zeyer 2021 §7).

    The label-prior trick replaces ``log p(y|x)`` with
    ``log p(y|x) - α * log p_prior(y)`` before the CTC marginalisation,
    discouraging the model from collapsing onto the marginal label
    distribution (which is what produces blank-collapse / peaky behavior).
    """
    B, T, V = logits.shape
    log_probs = F.log_softmax(logits.float(), dim=-1)
    if label_prior is not None and label_prior_weight > 0:
        log_probs = log_probs - label_prior_weight * label_prior.unsqueeze(0).unsqueeze(0)
        # Re-normalise so the modified distribution still sums to 1 (keeps
        # CTC's forward-backward numerically well-behaved).
        log_probs = F.log_softmax(log_probs, dim=-1)
    log_probs_T = log_probs.transpose(0, 1)            # (T, B, V)
    input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
    return F.ctc_loss(
        log_probs_T, targets, input_lengths, target_lengths,
        blank=BLANK_ID, zero_infinity=True,
    )


def _cr_ctc_kl(logits_a: torch.Tensor, logits_b: torch.Tensor,
               *, temperature: float = 1.0) -> torch.Tensor:
    """Symmetric KL between two CTC log-prob distributions (Yao 2024 ICLR 2025).

    Both inputs are (B, T, V). Returns scalar KL(a || b) + KL(b || a).
    """
    log_p_a = F.log_softmax(logits_a.float() / temperature, dim=-1)
    log_p_b = F.log_softmax(logits_b.float() / temperature, dim=-1)
    p_a = log_p_a.exp()
    p_b = log_p_b.exp()
    kl_ab = (p_a * (log_p_a - log_p_b)).sum(dim=-1).mean()
    kl_ba = (p_b * (log_p_b - log_p_a)).sum(dim=-1).mean()
    return 0.5 * (kl_ab + kl_ba)


# ============================================================================
# Train entry point
# ============================================================================


def train(cfg: CTCConfig) -> Path:
    """Train one cell. Returns path to the final checkpoint."""
    torch.manual_seed(cfg.seed)

    storage.ensure_dirs()
    run_dir = storage.cell_run_dir(cfg.cell_id)
    log_path = run_dir / "log.jsonl"
    samples_path = run_dir / "sample_gens.jsonl"
    stats_path = run_dir / "stats.jsonl"

    # ---- vocabulary ----
    if cfg.vocab == "bpe1k":
        if not storage.BPE_MODEL.exists():
            raise FileNotFoundError(
                f"BPE model not found at {storage.BPE_MODEL}. "
                "Run `exp02 build-bpe` first."
            )
        vocab = chars_mod.load_vocab("bpe1k", bpe_model_path=str(storage.BPE_MODEL))
    else:
        vocab = chars_mod.load_vocab(cfg.vocab)

    # ---- data ----
    fold = splits.load_fold(storage.STORAGE, cfg.fold)
    pp_spec = (preprocessing.for_encoder(cfg.preprocess, cfg.encoder)
               if cfg.preprocess != "v1" else None)
    sa_kwargs = (
        dict(n_time_masks=cfg.specaug_n_time_masks,
             time_mask_ms=cfg.specaug_time_mask_ms,
             n_chan_masks=cfg.specaug_n_chan_masks,
             chan_mask_max=cfg.specaug_chan_mask_max)
        if cfg.specaugment else None
    )
    drop_sources = tuple(s.strip() for s in (cfg.drop_sources or "").split(",") if s.strip())
    quality_kwargs = dict(
        drop_sources=drop_sources,
        min_text_chars=cfg.min_text_chars,
        max_text_chars=(cfg.max_text_chars or None),
        max_seconds=(cfg.max_seconds or None),
        drop_nan_rows=cfg.drop_nan_rows,
        drop_zero_rows=cfg.drop_zero_rows,
    )

    train_ds = EEGSentenceDataset(
        storage.STORAGE,
        sources=ALL_SOURCES,
        subject_filter=fold.train_subjects,
        sentence_filter=fold.train_sent_hashes,
        noise="gauss" if cfg.input == "noise_train" else None,
        preprocess=pp_spec,
        specaugment=sa_kwargs,
        **quality_kwargs,
    )
    dev_ds = EEGSentenceDataset(
        storage.STORAGE,
        sources=ZUCO_SOURCES,
        subject_filter=fold.dev_subjects,
        sentence_filter=fold.dev_sent_hashes,
        noise="gauss" if cfg.input == "noise_train" else None,
        preprocess=pp_spec,
        eval_only=True,
        # Dev keeps the same quality knobs so we evaluate apples-to-apples,
        # but DON'T drop sources at dev time — dev is ZuCo-only already.
        min_text_chars=cfg.min_text_chars,
        max_text_chars=(cfg.max_text_chars or None),
        max_seconds=(cfg.max_seconds or None),
        drop_nan_rows=cfg.drop_nan_rows,
        drop_zero_rows=cfg.drop_zero_rows,
    )

    # ---- text augmentation (paraphrase lookup; loaded once on main proc) ----
    paraphrase_lookup: ParaphraseLookup | None = None
    if cfg.text_aug_prob > 0 and cfg.text_aug_paraphrase_path:
        path = Path(cfg.text_aug_paraphrase_path)
        if not path.exists():
            raise FileNotFoundError(
                f"text-aug paraphrase parquet not found at {path}. "
                "Build with `exp02 build-paraphrases` first."
            )
        paraphrase_lookup = ParaphraseLookup(str(path))
    text_aug_rng = (np.random.default_rng(seed=cfg.seed)
                    if paraphrase_lookup is not None else None)

    # ---- signal augmentation config (per-step GPU-side) ----
    signal_aug_cfg = augment.SignalAugmentConfig(
        time_shift_max_frac=cfg.signal_aug_time_shift_max_frac,
        channel_dropout_p=cfg.signal_aug_channel_dropout_p,
        channel_dropout_frac=cfg.signal_aug_channel_dropout_frac,
        freq_mask_p=cfg.signal_aug_freq_mask_p,
        freq_mask_n=cfg.signal_aug_freq_mask_n,
        freq_mask_max_hz=cfg.signal_aug_freq_mask_max_hz,
        time_warp_p=cfg.signal_aug_time_warp_p,
        time_warp_segments=cfg.signal_aug_time_warp_segments,
        time_warp_factor_low=cfg.signal_aug_time_warp_factor_low,
        time_warp_factor_high=cfg.signal_aug_time_warp_factor_high,
        gaussian_noise_sigma=cfg.signal_aug_gaussian_noise_sigma,
        fourier_surrogate_p=cfg.signal_aug_fourier_surrogate_p,
        mixup_alpha=cfg.signal_aug_mixup_alpha,
    )
    signal_aug_active = any([
        signal_aug_cfg.time_shift_max_frac > 0,
        signal_aug_cfg.channel_dropout_p > 0,
        signal_aug_cfg.freq_mask_p > 0,
        signal_aug_cfg.time_warp_p > 0,
        signal_aug_cfg.gaussian_noise_sigma > 0,
        signal_aug_cfg.fourier_surrogate_p > 0,
    ])
    signal_aug_gen = (torch.Generator(device="cuda").manual_seed(cfg.seed)
                      if signal_aug_active else None)

    # ---- model ----
    model = EEG2CTC(cfg, vocab).to("cuda")
    target_sr = model.encoder.spec.native_sr

    def make_loader(ds, *, shuffle: bool, train: bool):
        # Only the train loader applies text augmentation; dev/eval always
        # uses the original references so metrics stay comparable.
        ploookup = paraphrase_lookup if train else None
        prob = cfg.text_aug_prob if train else 0.0
        rng = text_aug_rng if train else None
        return DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=lambda b: _collate(
                b, target_sr=target_sr,
                paraphrase_lookup=ploookup, text_aug_prob=prob,
                text_aug_rng=rng,
            ),
            pin_memory=True, persistent_workers=cfg.num_workers > 0,
        )

    train_dl = make_loader(train_ds, shuffle=True, train=True)
    dev_bank = _dev_sample_bank(make_loader(dev_ds, shuffle=False, train=False), k=16)

    # ---- optimizer ----
    head_params = model.head_trainable_parameters()
    enc_params = model.encoder_trainable_parameters()

    head_opt = torch.optim.AdamW(
        [p for p in head_params if p.requires_grad],
        lr=cfg.head_lr, weight_decay=cfg.weight_decay,
    )
    enc_opt = torch.optim.AdamW(
        [p for p in enc_params if p.requires_grad],
        lr=cfg.encoder_lr, weight_decay=cfg.weight_decay,
    ) if enc_params else None

    # ---- W&B ----
    wb = _wandb_init(cfg, n_train=len(train_ds), n_dev=len(dev_ds))

    log = open(log_path, "a")
    samples_log = open(samples_path, "a")
    stats_log = open(stats_path, "a")

    log.write(json.dumps({
        "event": "init",
        "cell_id": cfg.cell_id,
        "n_train": len(train_ds),
        "n_dev": len(dev_ds),
        "vocab": cfg.vocab,
        "vocab_size": vocab.size,
        "encoder": cfg.encoder,
        "encoder_finetune": cfg.encoder_finetune,
        "encoder_feature_dim": model.encoder.spec.feature_dim,
        "encoder_native_sr": model.encoder.spec.native_sr,
        "trainable_head_params": sum(p.numel() for p in head_params),
        "trainable_encoder_params": sum(p.numel() for p in enc_params),
        "variant": cfg.variant,
        "cr_ctc_kl_weight": cfg.cr_ctc_kl_weight if cfg.variant == "crctc" else 0.0,
        "intermediate_ctc_weight": cfg.intermediate_ctc_weight if cfg.variant == "intctc" else 0.0,
        "aed_weight": cfg.aed_weight if cfg.variant == "ctcaed" else 0.0,
        "label_prior_weight": cfg.label_prior_weight,
    }) + "\n"); log.flush()

    # ---- label prior buffer ----
    label_prior = torch.zeros(vocab.size, device="cuda") - math.log(vocab.size)
    label_prior_initialized = False

    # ---- training loop ----
    model.set_encoder_trainable(False)  # warmup-freeze starts active
    encoder_unfrozen = False
    it = iter(_inf(train_dl))
    t0 = time.time()
    sample_every = max(1, int(cfg.sample_every_frac * cfg.total_steps))
    save_every = max(1, int(cfg.save_every_frac * cfg.total_steps))

    autocast_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32

    for step in range(1, cfg.total_steps + 1):
        # Encoder warmup-freeze toggle: flip once when we cross the threshold.
        if (not encoder_unfrozen
                and step > cfg.encoder_warmup_freeze_steps
                and cfg.encoder_finetune != "frozen"):
            model.set_encoder_trainable(True)
            encoder_unfrozen = True
            log.write(json.dumps({
                "event": "encoder_unfrozen", "step": step,
                "n_encoder_params": sum(p.numel() for p in enc_params),
            }) + "\n"); log.flush()

        # LR schedule (manual; both opts share warmup + total).
        lr_mult = _linear_warmup_cosine_decay(
            step, warmup=cfg.warmup_steps, total=cfg.total_steps)
        for pg in head_opt.param_groups:
            pg["lr"] = cfg.head_lr * lr_mult
        if enc_opt is not None:
            for pg in enc_opt.param_groups:
                pg["lr"] = cfg.encoder_lr * lr_mult

        head_opt.zero_grad(set_to_none=True)
        if enc_opt is not None:
            enc_opt.zero_grad(set_to_none=True)

        loss_acc = {"ctc": 0.0, "cr_ctc": 0.0, "intctc": 0.0, "aed": 0.0, "total": 0.0}

        for _ in range(cfg.grad_accum):
            batch = next(it)
            eeg = batch["eeg"].to("cuda", non_blocking=True)
            sr = batch["sr"]
            channels = batch["channels"]
            texts = batch["text"]
            targets, target_lengths = vocab.encode_batch(texts, device="cuda")

            # Per-step GPU-side signal augmentation (independent of dataset's
            # row-deterministic SpecAugment). Generator advances on each call,
            # so the two CR-CTC views see different augmentations as well.
            if signal_aug_active:
                eeg = augment.apply(eeg, sr, signal_aug_cfg,
                                    generator=signal_aug_gen)

            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(autocast_dtype != torch.float32)):
                # CR-CTC: two SpecAugmented forward passes. Augmentation is
                # already baked into ``eeg`` by the dataset (single-shot).
                # For CR-CTC we want two DIFFERENT augmentations of the same
                # underlying clip — easiest way: run encoder once, run head
                # twice on two views of features (we approximate "augmenting
                # the input" by augmenting the post-encoder features with
                # dropout already inside the head transformer).
                #
                # The full CR-CTC paper applies augmentation pre-encoder; we
                # approximate by running the head twice with stochastic
                # dropout enabled (model.train() ensures dropout is on).
                # This is a valid view-pair when head dropout > 0.
                aed_inputs = None
                if cfg.variant == "ctcaed":
                    # Build BOS-prefixed target IDs so the AED branch can
                    # teacher-force on shifted targets.
                    aed_inputs = _aed_targets(texts, vocab, max_len=cfg.aed_max_target_len,
                                              device="cuda")

                feats = model.encoder_features(eeg, sr, channels)

                # Optional feature-space mixup. We mix the encoder features
                # over the batch dim and combine the matching CTC losses
                # weighted by lam (standard mixup recipe).
                mixup_perm: torch.Tensor | None = None
                mixup_lam: torch.Tensor | None = None
                if signal_aug_cfg.mixup_alpha > 0:
                    feats, mixup_perm, mixup_lam = augment.feature_mixup(
                        feats, alpha=signal_aug_cfg.mixup_alpha,
                        generator=signal_aug_gen,
                    )

                out_a = model.head_forward(feats, aed_target_ids=aed_inputs)

                # Update running label prior from this batch's mean log-probs.
                with torch.no_grad():
                    batch_log_probs = F.log_softmax(out_a.logits.detach().float(), dim=-1)
                    batch_prior = batch_log_probs.mean(dim=(0, 1))
                    if not label_prior_initialized:
                        label_prior = batch_prior
                        label_prior_initialized = True
                    else:
                        label_prior = (cfg.label_prior_ema * label_prior
                                       + (1.0 - cfg.label_prior_ema) * batch_prior)

                if mixup_perm is not None and mixup_lam is not None:
                    # CTC mixup: weight the two per-sample CTC losses by lam,
                    # 1-lam. The CTC head saw `feats = lam*A + (1-lam)*A[perm]`,
                    # so the natural targets are A's targets (weight lam) and
                    # A[perm]'s targets (weight 1-lam).
                    targets_perm, target_lengths_perm = vocab.encode_batch(
                        [texts[int(j.item())] for j in mixup_perm], device="cuda")
                    ctc_a = _ctc_loss_with_prior(
                        out_a.logits, targets, target_lengths,
                        label_prior=label_prior if cfg.label_prior_weight > 0 else None,
                        label_prior_weight=cfg.label_prior_weight,
                    )
                    ctc_b = _ctc_loss_with_prior(
                        out_a.logits, targets_perm, target_lengths_perm,
                        label_prior=label_prior if cfg.label_prior_weight > 0 else None,
                        label_prior_weight=cfg.label_prior_weight,
                    )
                    ctc = mixup_lam * ctc_a + (1.0 - mixup_lam) * ctc_b
                else:
                    ctc = _ctc_loss_with_prior(
                        out_a.logits, targets, target_lengths,
                        label_prior=label_prior if cfg.label_prior_weight > 0 else None,
                        label_prior_weight=cfg.label_prior_weight,
                    )
                loss = ctc
                loss_acc["ctc"] += float(ctc.detach())

                if cfg.variant == "intctc" and out_a.intermediate_logits:
                    int_loss = sum(
                        _ctc_loss_with_prior(li, targets, target_lengths)
                        for li in out_a.intermediate_logits
                    ) / len(out_a.intermediate_logits)
                    loss = loss + cfg.intermediate_ctc_weight * int_loss
                    loss_acc["intctc"] += float(int_loss.detach())

                if cfg.variant == "ctcaed" and out_a.aed_logits is not None:
                    aed_loss = _aed_cross_entropy(out_a.aed_logits, aed_inputs)
                    loss = (1.0 - cfg.aed_weight) * loss + cfg.aed_weight * aed_loss
                    loss_acc["aed"] += float(aed_loss.detach())

                if cfg.variant == "crctc":
                    out_b = model.head_forward(feats, aed_target_ids=None)
                    cr = _cr_ctc_kl(out_a.logits, out_b.logits,
                                    temperature=cfg.cr_ctc_temperature)
                    loss = loss + cfg.cr_ctc_kl_weight * cr
                    loss_acc["cr_ctc"] += float(cr.detach())

                loss_acc["total"] += float(loss.detach())

            (loss / cfg.grad_accum).backward()

        # Clip + step.
        all_params: list[torch.nn.Parameter] = list(head_params) + list(enc_params)
        gnorm = float(torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip))
        head_opt.step()
        if enc_opt is not None:
            enc_opt.step()

        if step % cfg.log_every == 0 or step == 1:
            payload = {
                "step": step,
                "ctc_loss": loss_acc["ctc"] / cfg.grad_accum,
                "cr_ctc_kl": loss_acc["cr_ctc"] / cfg.grad_accum,
                "intermediate_ctc": loss_acc["intctc"] / cfg.grad_accum,
                "aed_loss": loss_acc["aed"] / cfg.grad_accum,
                "total_loss": loss_acc["total"] / cfg.grad_accum,
                "grad_norm": gnorm,
                "head_lr": head_opt.param_groups[0]["lr"],
                "encoder_lr": (enc_opt.param_groups[0]["lr"] if enc_opt is not None else 0.0),
                "encoder_active": model._encoder_active,
                "elapsed": round(time.time() - t0, 1),
            }
            log.write(json.dumps(payload) + "\n"); log.flush()
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in payload.items() if k != "step"} | {"step": step})

        if step % sample_every == 0 or step == cfg.total_steps:
            _log_dev_samples(model, dev_bank, vocab, samples_log, wb, step=step)
            _log_feature_stats(model, dev_bank, stats_log, wb, step=step)
            model.train()

        if step % save_every == 0 and step != cfg.total_steps:
            _save_checkpoint(model, cfg, run_dir / f"model_step{step}.pt", log)

    # Final save.
    final = run_dir / "model.pt"
    _save_checkpoint(model, cfg, final, log)
    log.write(json.dumps({"event": "done", "ckpt": str(final)}) + "\n")
    for f in (log, samples_log, stats_log):
        f.close()

    return final


# ============================================================================
# Helpers
# ============================================================================


def _aed_targets(texts: list[str], vocab, *, max_len: int, device: str) -> torch.LongTensor:
    """Build BOS-prefixed, EOS-terminated, right-padded target IDs for AED.

    BOS reuses BLANK_ID, EOS reuses BLANK_ID — we don't have explicit BOS / EOS
    in the CTC vocab. Cross-entropy ignores positions whose target == 0 via
    ``ignore_index=0`` (set in :func:`_aed_cross_entropy`).
    """
    out = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        ids = vocab.encode(t)[: max_len - 2]
        out[i, 0] = 0  # BOS
        for j, idv in enumerate(ids):
            out[i, j + 1] = idv
        eos_pos = min(len(ids) + 1, max_len - 1)
        out[i, eos_pos] = 0  # EOS sentinel
    return out


def _aed_cross_entropy(logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
    """Causal cross-entropy: predict targets[:, 1:] from logits[:, :-1]."""
    pred = logits[:, :-1, :].contiguous()
    tgt = targets[:, 1:].contiguous()
    return F.cross_entropy(
        pred.reshape(-1, pred.size(-1)).float(),
        tgt.reshape(-1),
        ignore_index=0,
    )


def _save_checkpoint(model: EEG2CTC, cfg: CTCConfig, path: Path, log) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": asdict(cfg),
    }, path)
    log.write(json.dumps({
        "event": "save",
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
    }) + "\n"); log.flush()


def _dev_sample_bank(dev_dl: DataLoader, *, k: int) -> list[dict]:
    bank: list[dict] = []
    n = 0
    for batch in dev_dl:
        bank.append(batch)
        n += batch["eeg"].size(0)
        if n >= k:
            break
    return bank


@torch.no_grad()
def _log_dev_samples(model: EEG2CTC, banks: list[dict], vocab,
                     samples_log, wb, *, step: int):
    model.eval()
    rows: list[dict] = []
    for b in banks:
        eeg = b["eeg"].to("cuda", non_blocking=True)
        feats = model.encoder.encode(eeg, b["sr"], b["channels"])
        out = model.head_forward(feats)
        log_probs = F.log_softmax(out.logits.float(), dim=-1)
        ids_per_row = chars_mod.ctc_greedy_decode(log_probs)
        hyps = [vocab.decode(ids) for ids in ids_per_row]
        for ref, hyp, sid, dset in zip(b["text"], hyps, b["subject_ids"], b["datasets"]):
            rows.append({"step": step, "subject": sid, "dataset": dset,
                         "ref": ref, "hyp": hyp})
    for r in rows:
        samples_log.write(json.dumps(r) + "\n")
    samples_log.flush()
    if wb is not None and rows:
        import wandb
        tbl = wandb.Table(columns=["step", "subject", "dataset", "ref", "hyp"],
                          data=[[r["step"], r["subject"], r["dataset"], r["ref"], r["hyp"]]
                                for r in rows])
        wb.log({f"samples/step{step}": tbl})


@torch.no_grad()
def _log_feature_stats(model: EEG2CTC, banks: list[dict], stats_log, wb, *, step: int):
    feats_all: list[torch.Tensor] = []
    for b in banks:
        eeg = b["eeg"].to("cuda", non_blocking=True)
        try:
            f = model.encoder.encode(eeg, b["sr"], b["channels"])
            feats_all.append(f.flatten().float().cpu())
        except Exception as e:
            stats_log.write(json.dumps({"step": step,
                                        "encode_error": f"{type(e).__name__}: {e}"}) + "\n")
            return
    if not feats_all:
        return
    flat = torch.cat(feats_all)
    payload = {
        "step": step,
        "feat_mean": float(flat.mean()),
        "feat_std": float(flat.std()),
        "feat_abs_max": float(flat.abs().max()),
        "feat_nonzero_frac": float((flat != 0).float().mean()),
    }
    stats_log.write(json.dumps(payload) + "\n"); stats_log.flush()
    if wb is not None:
        wb.log({f"feat/{k}": v for k, v in payload.items() if k != "step"} | {"step": step})


def _wandb_init(cfg: CTCConfig, *, n_train: int, n_dev: int):
    if "WANDB_API_KEY" not in os.environ:
        print("[train] WANDB_API_KEY not set; W&B logging disabled.", flush=True)
        return None
    import wandb
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "exp02-eeg-ctc"),
        name=cfg.cell_id,
        group=f"{cfg.encoder}_{cfg.vocab}_{cfg.variant}",
        tags=[cfg.encoder, cfg.vocab, cfg.variant, cfg.input, f"fold{cfg.fold}"],
        config={**asdict(cfg), "n_train": n_train, "n_dev": n_dev},
        dir=str(storage.WANDB_DIR),
        resume="allow",
    )


def _inf(dl: Iterable):
    while True:
        for b in dl:
            yield b
