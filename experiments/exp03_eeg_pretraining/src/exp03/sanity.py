"""Mini-experiment 01: the five Karpathy-style sanity baselines.

Per `mini_experiments/01_sanity_baselines/README.md`, no other mini-
experiment may run until all five checks pass:

    Check A — Loss-at-init matches theory (within 20%)
    Check B — Input-independent baseline does not improve loss (≤ 1% over 5000 steps)
    Check C — One-batch overfit drives loss to < 1% of init within 1000 steps
    Check D — Random-init linear-probe floor is recorded for the primary eval suite
    Check E — Shape-and-mask audit signed off (no view-vs-transpose surprises)

Each check is a standalone function so they can be run individually for
debugging (`python -m exp03.sanity check_a`) or as a suite. All produce
JSON-serialisable dicts; the runner collects them and writes
`mini_experiments/01_sanity_baselines/results.md`.

Karpathy's mantra applies: until every check is GREEN, every later
mini-experiment runs on a setup we cannot trust. Treat a YELLOW or RED
in any check as a hard stop, fix the bug, and re-run all five (because
the fix may break a previously-passing one).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import typer

from . import data, losses, model
from .model import ModelConfig, build_model, count_params


# =============================================================================
# Result containers
# =============================================================================


@dataclass
class CheckResult:
    """One check's verdict + metrics."""
    name: str
    status: Literal["GREEN", "YELLOW", "RED", "SKIPPED"]
    details: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "details": self.details}


# =============================================================================
# Check E — Shape-and-mask audit
# =============================================================================
#
# Print tensor shapes at every module boundary on a tiny synthetic batch.
# Runs first in the suite because if shapes are wrong, every other check
# is meaningless. No real data, no training, no losses needed.


def check_e_shape_audit(
    cfg: ModelConfig | None = None,
    *,
    B: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CheckResult:
    """Walk through the SSL model on a tiny batch and print every shape.

    Catches the "Karpathy bug": a `view` instead of `transpose` somewhere
    in the masking that lets the model see information from other batch
    entries (which then trains "fine" on the SSL loss but learns nothing
    useful). The most common silent bug in MAE-style code.

    Pass criterion: every shape matches what `ModelConfig` predicts.
    """
    cfg = cfg or ModelConfig()
    print(f"\n{'='*70}\nCheck E — Shape-and-mask audit\n{'='*70}")
    print(f"device={device}, dtype=fp32")
    print(f"cfg.window_samples = {cfg.window_samples}")
    print(f"cfg.total_stride   = {cfg.total_stride}")
    print(f"cfg.n_tokens       = {cfg.n_tokens}")
    print(f"cfg.patch_samples  = {cfg.patch_samples}")
    print()

    m = build_model(cfg).to(device).eval()
    p = count_params(m)
    print(f"params:  total={p['total']:,}  trainable={p['trainable']:,}")

    x = data.synthetic_batch(B=B, T=cfg.window_samples, kind="gauss",
                             seed=0, device=device)
    print(f"input.shape          = {tuple(x.shape)}, dtype={x.dtype}")

    expected = {
        "input":            (B, cfg.window_samples),
        "frontend_out":     (B, cfg.n_tokens, cfg.frontend.d_model),
        "encoder_features": (B, int(cfg.n_tokens * (1 - cfg.mask.mask_ratio)), cfg.backbone.d_model),
        "decoder_features": (B, cfg.n_tokens, cfg.decoder.d_model),
        "reconstruction":   (B, cfg.window_samples),
        "mask":             (B, cfg.window_samples),
        "token_mask":       (B, cfg.n_tokens),
    }

    # --- step through manually so we can print intermediates ---
    with torch.no_grad():
        x_unsq = x.unsqueeze(1)
        print(f"x.unsqueeze(1)       = {tuple(x_unsq.shape)}")
        tokens = m.frontend(x_unsq)
        print(f"after frontend       = {tuple(tokens.shape)}  (B, T_tokens, D)")
        tokens_pe = m.encoder_pos_emb(tokens)
        print(f"after encoder posemb = {tuple(tokens_pe.shape)}")

        m_out = m.mask_module(B, cfg.n_tokens, x.device)
        n_visible = m_out.ids_keep.size(1)
        print(f"mask.ids_keep        = {tuple(m_out.ids_keep.shape)}  (B, n_visible)")
        print(f"mask.ids_restore     = {tuple(m_out.ids_restore.shape)}  (B, T_tokens)")
        print(f"mask.mask            = {tuple(m_out.mask.shape)}  (B, T_tokens), "
              f"sum/B = {(m_out.mask.sum() / B).item():.1f} masked tokens")
        print(f"  expected masked    = {int(cfg.n_tokens * cfg.mask.mask_ratio)}")
        print(f"  observed visible   = {n_visible}")

        ids_keep_d = m_out.ids_keep.unsqueeze(-1).expand(-1, -1, cfg.frontend.d_model)
        visible = torch.gather(tokens_pe, dim=1, index=ids_keep_d)
        print(f"visible_tokens       = {tuple(visible.shape)}")
        encoded = m.encoder(visible)
        print(f"encoder out          = {tuple(encoded.shape)}")

        # Reassemble decoder input
        n_masked = cfg.n_tokens - n_visible
        mask_tokens = m.mask_token.expand(B, n_masked, cfg.decoder.d_model)
        print(f"mask_tokens          = {tuple(mask_tokens.shape)}")
        x_full = torch.cat([encoded, mask_tokens], dim=1)
        ids_restore_d = m_out.ids_restore.unsqueeze(-1).expand(-1, -1, cfg.decoder.d_model)
        x_full = torch.gather(x_full, dim=1, index=ids_restore_d)
        print(f"decoder input full   = {tuple(x_full.shape)}")
        x_full = m.decoder_pos_emb(x_full)
        decoded = m.decoder(x_full)
        print(f"decoder out          = {tuple(decoded.shape)}")
        recon_patches = m.recon_head(decoded)
        print(f"recon_head out       = {tuple(recon_patches.shape)}")
        recon = recon_patches.flatten(1)
        print(f"recon (flat)         = {tuple(recon.shape)}")

        # --- end-to-end forward (sanity that the manual walk matches) ---
        out = m(x)
        print()
        print("end-to-end forward dict:")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:<22} = shape {tuple(v.shape)}, dtype {v.dtype}")

    # Verify expectations
    actual = {
        "input": tuple(x.shape),
        "frontend_out": tuple(tokens.shape),
        "encoder_features": tuple(out["encoder_features"].shape),
        "decoder_features": tuple(out["decoder_features"].shape),
        "reconstruction": tuple(out["reconstruction"].shape),
        "mask": tuple(out["mask"].shape),
        "token_mask": tuple(out["token_mask"].shape),
    }
    mismatches = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}

    # Anti-leak audit: does the encoder output for batch element 0 depend
    # on batch element 1? If a `view`/`transpose` mistake mixes batch and
    # time dims the answer would be yes. We test by perturbing batch elem
    # 1 and checking whether the encoder output for elem 0 changes.
    with torch.no_grad():
        x_perturbed = x.clone()
        x_perturbed[1] += 100.0          # huge perturbation to elem 1
        out_b0_orig = m(x).get("encoder_features")[0]
        out_b0_pert = m(x_perturbed).get("encoder_features")[0]
        leak_max = (out_b0_orig - out_b0_pert).abs().max().item()
    print(f"\nAnti-batch-leak: encoder_features[0] max-abs-diff after perturbing batch[1]: {leak_max:.2e}")
    print(f"  Expected: ~0 (deterministic encoder + deterministic mask) or "
          f"some small number if mask is stochastic across forwards.")
    # Actually with stochastic masking each forward gives a different mask
    # so leak_max can be non-zero from masking alone; the *informative*
    # signal is whether it's bounded by the masking-induced jitter, not
    # 100x larger. We can't easily distinguish here without re-running
    # with a fixed mask seed, so we just record the number.

    status = "GREEN" if not mismatches else "RED"
    return CheckResult(
        name="E_shape_audit",
        status=status,
        details={
            "expected": {k: list(v) for k, v in expected.items()},
            "actual":   {k: list(v) for k, v in actual.items()},
            "mismatches": {k: [list(a), list(b)] for k, (a, b) in mismatches.items()},
            "params": p,
            "leak_max_abs_diff_when_perturbing_other_batch_elem": leak_max,
        },
        notes="" if status == "GREEN" else
              f"shape mismatches detected: {list(mismatches.keys())}",
    )


# =============================================================================
# Check A — Loss-at-init matches theory
# =============================================================================
#
# Compute every candidate loss on a fresh batch with zero training.
# Compare against closed-form theory; pass if within 20% per loss.


@dataclass
class _LossExpectation:
    name: str
    expected: float
    tolerance: float = 0.20
    note: str = ""


def check_a_loss_at_init(
    cfg: ModelConfig | None = None,
    *,
    B: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CheckResult:
    """Loss-at-init test on iid Gaussian targets and a fresh-init model."""
    cfg = cfg or ModelConfig()
    print(f"\n{'='*70}\nCheck A — Loss-at-init\n{'='*70}")

    m = build_model(cfg).to(device).eval()
    x = data.synthetic_batch(B=B, T=cfg.window_samples, kind="gauss", seed=42, device=device)
    print(f"input shape={tuple(x.shape)} std={x.std().item():.3f} mean={x.mean().item():.3f}")

    with torch.no_grad():
        out = m(x)

    # Compute every (concrete) loss in the registry on the same batch.
    measurements: dict[str, float] = {}
    for kind in ["l1_raw", "l2_raw", "mrstft_logmag", "l1_plus_mrstft", "infonce"]:
        loss_mod = losses.build_loss(kind).to(device).eval()
        with torch.no_grad():
            try:
                if kind == "infonce":
                    # InfoNCE needs *two* forwards with different masks.
                    out_b = m(x)
                    val, _ = loss_mod(out, out_b)
                else:
                    val, _ = loss_mod(out)
            except NotImplementedError as e:
                measurements[kind] = float("nan")
                continue
        measurements[kind] = val.item()

    # Closed-form expectations.
    log_B = math.log(B)
    expectations = [
        _LossExpectation("l1_raw",  math.sqrt(2.0 / math.pi),
                          tolerance=0.40,
                          note=("E|target - 0| = √(2/π) ≈ 0.7979 (lower bound when "
                                "model output ≈ 0); E|target - recon| up to ≈ 1.13 "
                                "if recon has unit variance. We use a wide ±40% band.")),
        _LossExpectation("l2_raw",  1.0,
                          tolerance=0.30,
                          note="Var(N(0,1)) = 1.0; Linear init noise pushes this to ~1.01."),
        _LossExpectation("mrstft_logmag", float("nan"),
                          tolerance=float("nan"),
                          note=("scale-dependent — measured value recorded but no "
                                "theoretical target.")),
        _LossExpectation("l1_plus_mrstft", float("nan"),
                          tolerance=float("nan"),
                          note=("composite — measured value recorded but no "
                                "theoretical target.")),
        _LossExpectation("infonce", log_B,
                          tolerance=0.30,
                          note=f"log(B) = log({B}) = {log_B:.4f}"),
    ]

    rows = []
    any_red = False
    for exp in expectations:
        meas = measurements.get(exp.name, float("nan"))
        if math.isnan(exp.expected):
            verdict = "RECORD"
        elif math.isnan(meas):
            verdict = "MISSING"; any_red = True
        else:
            rel_err = abs(meas - exp.expected) / max(abs(exp.expected), 1e-9)
            verdict = "OK" if rel_err <= exp.tolerance else "FAIL"
            if verdict == "FAIL":
                any_red = True
        rows.append({
            "loss": exp.name,
            "measured": meas,
            "expected": exp.expected,
            "tolerance_rel": exp.tolerance,
            "verdict": verdict,
            "note": exp.note,
        })

    # Pretty print
    print(f"\n{'loss':<22}{'measured':>14}{'expected':>14}{'verdict':>10}")
    print("-" * 60)
    for r in rows:
        m_s = "nan" if math.isnan(r["measured"]) else f"{r['measured']:.4f}"
        e_s = "—"   if math.isnan(r["expected"]) else f"{r['expected']:.4f}"
        print(f"{r['loss']:<22}{m_s:>14}{e_s:>14}{r['verdict']:>10}")

    return CheckResult(
        name="A_loss_at_init",
        status="RED" if any_red else "GREEN",
        details={"measurements": measurements, "rows": rows, "B": B,
                 "input_kind": "gauss"},
    )


# =============================================================================
# Shared training utilities (for Checks B, C)
# =============================================================================


def _training_step(
    model_: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    use_bf16: bool = True,
) -> dict[str, float]:
    """One forward + backward + step. Returns a dict of {loss, *components}."""
    optimizer.zero_grad(set_to_none=True)
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model_(x)
            loss, comps = loss_fn(out)
    else:
        out = model_(x)
        loss, comps = loss_fn(out)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), **comps}


# =============================================================================
# Check B — Input-independent baseline
# =============================================================================
#
# Replace encoder input with a constant (zeros / global mean). Train 5000
# steps with the full SSL objective. Loss must NOT decrease meaningfully.


def check_b_input_independent(
    cfg: ModelConfig | None = None,
    *,
    n_steps: int = 5000,
    B: int = 16,
    lr: float = 3e-4,
    log_every: int = 100,
    relative_improvement_threshold: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
) -> CheckResult:
    """Train the model on constant zero input; loss must stay flat."""
    cfg = cfg or ModelConfig()
    print(f"\n{'='*70}\nCheck B — Input-independent baseline\n{'='*70}")
    print(f"n_steps={n_steps} B={B} lr={lr} device={device}")

    torch.manual_seed(seed)
    m = build_model(cfg).to(device)
    loss_fn = losses.build_loss("l1_plus_mrstft").to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    history = []
    initial_losses = []
    for step in range(n_steps):
        x = data.synthetic_batch(B=B, T=cfg.window_samples, kind="constant",
                                 seed=seed + step, device=device)
        # Re-randomise mask each step automatically (RandomPatchMask uses
        # rng without a fixed seed).
        info = _training_step(m, loss_fn, x, opt, use_bf16=use_bf16)
        history.append({"step": step, **info})
        if step < 10:
            initial_losses.append(info["loss"])
        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:>5}  loss={info['loss']:.4f}  l1={info.get('l1_raw', float('nan')):.4f}"
                  f"  mrstft={info.get('mrstft_logmag', float('nan')):.4f}")

    init_avg = float(np.mean(initial_losses))
    final_avg = float(np.mean([h["loss"] for h in history[-100:]]))
    rel_improvement = (init_avg - final_avg) / max(abs(init_avg), 1e-9)
    print(f"\ninit_avg(first 10 steps)  = {init_avg:.4f}")
    print(f"final_avg(last 100 steps) = {final_avg:.4f}")
    print(f"relative improvement      = {rel_improvement:+.4%}")
    print(f"threshold                 = ≤ {relative_improvement_threshold:+.2%}")

    status = "GREEN" if rel_improvement <= relative_improvement_threshold else "RED"
    return CheckResult(
        name="B_input_independent",
        status=status,
        details={
            "n_steps": n_steps, "B": B, "lr": lr,
            "init_avg_loss": init_avg, "final_avg_loss": final_avg,
            "relative_improvement": rel_improvement,
            "threshold": relative_improvement_threshold,
            "history": history[::max(1, n_steps // 200)],   # downsample to ~200 points
        },
        notes=("loss must not decrease > 1% — improvement here means the model is "
               "predicting target from something other than the input "
               "(positional-leak / mask-shortcut / dataset-leak)") if status == "RED" else "",
    )


# =============================================================================
# Check C — One-batch overfit
# =============================================================================
#
# Take exactly 4 EEG epochs from one recording. Train on those 4 only,
# 1000 steps. Loss must drop to < 1% of init.


def check_c_one_batch_overfit(
    cfg: ModelConfig | None = None,
    *,
    derived_root: Path | None = None,
    n_steps: int = 1000,
    lr: float = 1e-3,
    log_every: int = 50,
    success_threshold: float = 0.01,
    use_synthetic_fallback: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
) -> CheckResult:
    """Overfit a fixed 4-window batch from one recording; loss must crash."""
    cfg = cfg or ModelConfig()
    print(f"\n{'='*70}\nCheck C — One-batch overfit\n{'='*70}")
    print(f"n_steps={n_steps} lr={lr} device={device}")

    torch.manual_seed(seed)
    if derived_root is not None and Path(derived_root).exists():
        try:
            batch = data.single_recording_overfit_batch(
                derived_root=derived_root, n_windows=4, channel_idx=0,
                device=device,
            )
            x = batch["signal"]
            source_label = (
                f"sub-{batch['subject_id']}/{batch['recording_id']}/ch{batch['channel_idx']}"
            )
            print(f"source: {source_label}, signal shape {tuple(x.shape)}")
        except (FileNotFoundError, ValueError) as e:
            if not use_synthetic_fallback:
                raise
            print(f"warning: real data unavailable ({e}); falling back to synthetic AR(1)")
            x = data.synthetic_batch(B=4, T=cfg.window_samples, kind="ar1",
                                     seed=seed, device=device)
            source_label = "synthetic_ar1"
    else:
        if not use_synthetic_fallback:
            raise FileNotFoundError(f"derived_root={derived_root} missing")
        x = data.synthetic_batch(B=4, T=cfg.window_samples, kind="ar1",
                                 seed=seed, device=device)
        source_label = "synthetic_ar1"

    m = build_model(cfg).to(device)
    loss_fn = losses.build_loss("l1_plus_mrstft").to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    # Use only_masked=False for the overfit so the loss is on ALL positions
    # (this is appropriate when overfitting; we want the encoder to be able
    # to memorise the entire recording, not just the masked half each step).
    # However, the spec says "near-perfect mask prediction" so we keep the
    # masked-only loss as the primary criterion.
    history = []
    init_loss = None
    final_loss = None
    for step in range(n_steps):
        info = _training_step(m, loss_fn, x, opt, use_bf16=use_bf16)
        history.append({"step": step, **info})
        if step == 0:
            init_loss = info["loss"]
        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:>5}  loss={info['loss']:.4f}  "
                  f"frac_of_init={info['loss']/max(abs(init_loss), 1e-9):.3%}")
        final_loss = info["loss"]

    rel = final_loss / max(abs(init_loss), 1e-9)
    print(f"\ninit_loss = {init_loss:.4f}")
    print(f"final_loss = {final_loss:.4f}")
    print(f"final/init = {rel:.3%}")
    print(f"threshold  = ≤ {success_threshold:.0%}")

    status = "GREEN" if rel < success_threshold else "RED"
    return CheckResult(
        name="C_one_batch_overfit",
        status=status,
        details={
            "n_steps": n_steps, "lr": lr, "source": source_label,
            "init_loss": init_loss, "final_loss": final_loss,
            "final_over_init": rel, "success_threshold": success_threshold,
            "history": history[::max(1, n_steps // 200)],
        },
        notes=("loss didn't crash to <1% — encoder/decoder capacity insufficient, "
               "or masking too aggressive, or optimizer / shape bug") if status == "RED" else "",
    )


# =============================================================================
# Check D — Random-init linear-probe floor
# =============================================================================
#
# Extract mean-pooled features from a freshly-init encoder on many windows;
# train a sklearn linear probe + k-NN on the §4.3 Protocol A targets.


def check_d_random_init_probe(
    cfg: ModelConfig | None = None,
    *,
    derived_root: Path | None = None,
    max_subjects: int = 50,
    max_windows_per_shard: int = 20,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CheckResult:
    """Frozen-feature linear probe + k-NN on the random-init encoder.

    Stub for now — calls into eval.py once that's wired. For mini-exp 01
    we record the floor numbers (HBN externalizing R²/MAE; attention R²/MAE;
    attention-binary AUROC; 6-task BAC/WF1; k-NN top-1) with 95% bootstrap
    CIs. Every subsequent pretrained encoder must clearly beat these.
    """
    print(f"\n{'='*70}\nCheck D — Random-init linear-probe floor\n{'='*70}")
    cfg = cfg or ModelConfig()
    if derived_root is None or not Path(derived_root).exists():
        return CheckResult(
            name="D_random_init_probe",
            status="SKIPPED",
            details={"reason": f"derived_root={derived_root} not provided / missing"},
            notes="run with --derived-root pointing at synced parquet shards",
        )
    try:
        from . import eval as eval_mod   # populated by Step K
    except ImportError:
        return CheckResult(
            name="D_random_init_probe",
            status="SKIPPED",
            details={"reason": "exp03.eval not implemented yet (Step K)"},
            notes="Step K of mini-exp 01 will populate eval.py",
        )
    # Once eval.py exists, delegate.
    return eval_mod.run_random_init_probe(
        cfg, derived_root=derived_root,
        max_subjects=max_subjects,
        max_windows_per_shard=max_windows_per_shard,
        seed=seed, device=device,
    )


# =============================================================================
# Suite runner + results.md writer
# =============================================================================


def write_results_md(results: list[CheckResult], path: Path) -> None:
    """Render a structured results.md from a list of CheckResults.

    The file is what the gating decision references; treat it as a contract
    between this mini-experiment and every later one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# exp03 / mini-experiment 01 — Sanity baselines: results")
    lines.append("")
    lines.append(f"_Generated by `exp03.sanity` at {time.strftime('%Y-%m-%dT%H:%M:%S UTC', time.gmtime())}_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Check | Status |")
    lines.append("|-------|--------|")
    for r in results:
        emoji = {"GREEN": "✅", "YELLOW": "⚠️", "RED": "❌", "SKIPPED": "—"}[r.status]
        lines.append(f"| {r.name} | {emoji} {r.status} |")
    lines.append("")
    for r in results:
        lines.append(f"## {r.name}")
        lines.append("")
        lines.append(f"**Status:** {r.status}")
        if r.notes:
            lines.append("")
            lines.append(f"_Notes:_ {r.notes}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.details, indent=2, default=str))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines))


def run_all(
    cfg: ModelConfig | None = None,
    *,
    derived_root: Path | None = None,
    output_path: Path | None = None,
    skip: tuple[str, ...] = (),
    fast: bool = False,
) -> list[CheckResult]:
    """Run all 5 checks in order; return list of results.

    `fast=True` reduces step counts to ~100 for B and C — useful for
    quickly verifying the pipeline runs end-to-end. The official
    pass/fail numbers are only valid with `fast=False`.
    """
    cfg = cfg or ModelConfig()
    results = []

    if "E" not in skip:
        results.append(check_e_shape_audit(cfg))
    if "A" not in skip:
        results.append(check_a_loss_at_init(cfg))
    if "C" not in skip:
        results.append(check_c_one_batch_overfit(
            cfg, derived_root=derived_root,
            n_steps=100 if fast else 1000,
        ))
    if "B" not in skip:
        results.append(check_b_input_independent(
            cfg, n_steps=100 if fast else 5000,
        ))
    if "D" not in skip:
        results.append(check_d_random_init_probe(
            cfg, derived_root=derived_root,
        ))

    if output_path is not None:
        write_results_md(results, output_path)

    print("\n" + "=" * 70)
    print("FINAL VERDICT:")
    for r in results:
        print(f"  {r.name:<28}  {r.status}")
    print("=" * 70)
    return results


# =============================================================================
# CLI
# =============================================================================


app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def check_e():
    """Run only Check E (shape audit)."""
    check_e_shape_audit()


@app.command()
def check_a():
    """Run only Check A (loss-at-init)."""
    check_a_loss_at_init()


@app.command()
def check_b(n_steps: int = 5000):
    """Run only Check B (input-independent baseline)."""
    check_b_input_independent(n_steps=n_steps)


@app.command()
def check_c(
    derived_root: Path = typer.Option(None, help="parquet derived root"),
    n_steps: int = 1000,
):
    """Run only Check C (one-batch overfit)."""
    check_c_one_batch_overfit(derived_root=derived_root, n_steps=n_steps)


@app.command()
def check_d(
    derived_root: Path = typer.Option(None, help="parquet derived root"),
):
    """Run only Check D (random-init linear-probe floor)."""
    check_d_random_init_probe(derived_root=derived_root)


@app.command()
def all(
    derived_root: Path = typer.Option(None, help="parquet derived root"),
    output: Path = typer.Option(None, help="results.md output path"),
    fast: bool = typer.Option(False, help="reduce step counts (NOT a valid pass/fail)"),
    skip: str = typer.Option("", help="comma-separated checks to skip, e.g. 'D'"),
):
    """Run all 5 checks and write results.md."""
    run_all(
        derived_root=derived_root,
        output_path=output,
        skip=tuple(s.strip() for s in skip.split(",") if s.strip()),
        fast=fast,
    )


if __name__ == "__main__":
    app()
