# exp03 / mini-experiment 11 — Continuous bottleneck vs FSQ

> **Status:** planned
>
> **Cross-reference:** [`brain/experiments/tokenization/05-quantization.md` §5](../../../../../brain/experiments/tokenization/05-quantization.md#5-finite-scalar-quantization-fsq--the-open-opportunity),
> [Mentzer et al. ICLR 2024](https://arxiv.org/abs/2309.15505),
> [UNITE ICML 2026](https://arxiv.org/abs/2603.22283)
>
> **Compute budget:** 18 H100-hours (3 bottleneck variants × 2 control
> columns × 3 seeds = 18 cells × 60 min average — FSQ joint training takes
> longer due to warmup schedules)
>
> **Gates:** none downstream

## Question

Can a discrete bottleneck — specifically Finite Scalar Quantization (FSQ),
the only quantization scheme structurally compatible with single-stage
joint training — be inserted into the chosen MAE-style pretraining pipeline
without instability, and does the resulting representation outperform the
continuous baseline?

## Why it matters

Every published EEG model with a discrete bottleneck (LaBraM, NeuroLM,
BioCodec, BrainRVQ, NeuroRVQ, TFM-Tokenizer) trains in two stages:
tokenizer first with reconstruction loss, then frozen for the masked
prediction stage. The user's scratch-only constraint rules this out. The
question is whether *single-stage joint* training of a discrete bottleneck
+ masked prediction is feasible at all, and whether it gives any practical
benefit over the continuous baseline.

The bottleneck research subagent's verdict was sharp on this:

- **Vanilla VQ-VAE** in joint training fails because of the bootstrapping
  problem (masked prediction needs meaningful tokens to predict; meaningful
  tokens require a trained encoder; trained encoder requires gradient
  signal from masked prediction). Compounded by codebook collapse.
- **RVQ** has the same issues compounded across multiple codebook layers,
  plus STE gradient noise per layer.
- **LFQ** (MAGVIT-v2) is two-stage in every published application; the
  binary bottleneck doesn't preserve the distance structure that EEG
  rhythms (which differ in amplitude smoothly) need.
- **FSQ** is the one quantization scheme that is structurally clean for
  joint training: no learned codebook (so nothing to collapse), no
  commitment loss (so no competing objective), no EMA tracking, and
  100 % codebook utilization by construction. The Cell Patterns 2025
  protein study showed FSQ outperforms VQ at codebook sizes > 2¹⁰. UNITE
  (ICML 2026) proves the principle of joint tokenization + generative
  objective from scratch.

**No EEG paper has used FSQ.** This experiment is the joint-training-clean,
discrete-bottleneck-clean alternative to continuous MAE. The honest
question is whether the gain (better cross-subject transfer per the
discrete-tokens-help-transfer pattern) outweighs the cost (extra
implementation complexity, slightly more delicate training).

## Variants

| Code | Variant | Bottleneck | Joint training? |
| ---- | ------- | ---------- | ---------------- |
| Q0 | Continuous (the exp04 default) | none | ✓ |
| Q1 | FSQ at modest size | FSQ with `levels=[5, 5, 5, 5, 5, 5]` → 15,625 effective codes | ✓ (single-stage) |
| Q2 | FSQ at larger size | FSQ with `levels=[8, 6, 5, 5, 5, 5]` → 36,000 effective codes | ✓ (single-stage) |

We deliberately exclude vanilla VQ-VAE, RVQ, and LFQ because the bottleneck
agent's analysis showed they are not single-stage compatible without
significant post-hoc engineering (BEATs-style alternation, which the user
explicitly does not want). FSQ is the cleanest test of "does discrete help
in single-stage scratch SSL".

## Controls

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| Q0 continuous                  | ✓          | ✓                  |
| Q1 FSQ 15K codes               | ✓          | ✓                  |
| Q2 FSQ 36K codes               | ✓          | ✓                  |

## Held constant

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: exp04 winner (likely S1 MAE-denoised). The
  reconstruction loss now applies to the *quantised* output, with
  stop-gradient on the quantised target before forming the
  masked-prediction target.
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Target: exp08 winner.
- Mask: exp10 winner.
- Optimiser: AdamW, LR carried forward.
- Pretraining duration: 8 epochs (with FSQ's λ-warmup detailed below).

The critical implementation detail — the **λ(t) warmup schedule**:

```
total_loss = L_recon(x, decoder(quantize(encoder(x_masked))))
           + λ(t) * L_masked_pred(stop_gradient(quantize(encoder(x_full))),
                                  predictor_output)
λ(t) = 0.0 for t < 0.15 * total_steps
     = linear ramp from 0.0 to 0.5 over t ∈ [0.15, 0.30] * total_steps
     = 0.5 thereafter
```

The warmup is essential: at initialization the FSQ codes are essentially
random, so masked-prediction loss provides no useful gradient signal
(predicting random codes is just predicting noise). Reconstruction loss
has to stabilise the quantised representation first, then the
masked-prediction objective comes online once the codes are meaningful.
This is the UNITE paper's structural pattern for joint training.

## Decision rule

Same as prior experiments:

- Strict win = ≥ 1 pp HBN 6-task BAC (per §4.3 Protocol A.2), non-overlapping CIs, noise-twin flat.
- Weak win = ≥ 0.5 pp with paired permutation p < 0.05.
- Tie = TOST equivalence within ε = 1 pp.

Two FSQ-specific criteria:

- **Codebook utilization**: log fraction of effective codes used per 1000
  steps. FSQ guarantees 100 % structural utilization, but if the encoder
  collapses to a small region of FSQ-space, effective utilization can
  still be low. Should reach > 80 % of codes used by end of training.
- **Cross-subject transfer test**: in addition to HBN 6-task BAC on the
  standard split, run HBN 6-task with leave-one-subject-out on a 5-subject
  subset. Discrete bottlenecks are theoretically expected to help cross-
  subject transfer (they enforce subject-invariant compression). If FSQ
  doesn't
  help cross-subject transfer, the entire discrete-bottleneck argument
  loses force.

## Pre-registered predictions

| Variant | HBN 6-task BAC (within-split) | HBN 6-task LOSO | Codebook utilization |
| ------- | ------------------------ | ---------- | --------------------- |
| Q0 continuous | reference | reference | n/a |
| Q1 FSQ 15K | tied or weak loss vs Q0 (~-0.5 pp); weak win on LOSO (~+0.5 pp) | ~85 % | |
| Q2 FSQ 36K | tied vs Q0; strict win on LOSO (~+1–2 pp) | ~70 % | |

The honest expected outcome: **continuous wins on within-split metrics,
FSQ wins on cross-subject transfer**. This is the well-documented pattern
from the EEG-FM literature ([`brain/experiments/tokenization/09-tokenization-free-alternatives.md` §2](../../../../../brain/experiments/tokenization/09-tokenization-free-alternatives.md#2-why-continuous-tokenizers-dominate-the-leaderboard)):
*continuous tokenizers tend to win on within-dataset accuracy; discrete
tokenizers tend to win on cross-domain transfer*. exp11 is the
single-stage-compatible test of whether this pattern survives the
constraint.

If FSQ wins on neither, the conclusion is clear: stay continuous.

If FSQ wins on LOSO but ties on within-split, the conclusion depends on
the headline goal: cross-subject is the relevant deployment scenario for
clinical EEG, so a marginal LOSO win could justify FSQ even at a tied
within-split number.

## Implementation pointers

- FSQ: 5-line implementation per [Mentzer et al.](https://arxiv.org/abs/2309.15505)
  ```python
  def fsq(z, levels=[5, 5, 5, 5, 5, 5]):
      L = torch.tensor(levels, device=z.device, dtype=z.dtype)
      z_bounded = (L - 1) * torch.tanh(z) / 2  # bound to [-(L-1)/2, (L-1)/2]
      z_quantized = z_bounded + (z_bounded.round() - z_bounded).detach()  # STE
      return z_quantized, z_bounded.round().long()  # continuous + integer index
  ```
  Project encoder output from `D=256` to `len(levels)=6` before FSQ; project
  the quantized output back to D for the decoder.
- Stop-gradient on the quantizer target for masked-prediction: `target = quantize(encoder(x_full)).detach()`.
- The masked-prediction loss is cross-entropy over the discrete code
  indices: convert each FSQ output to a flat integer in `[0, prod(levels))`,
  predict via a softmax head of size `prod(levels)`.
- Codebook utilization: count unique integer indices over a 10K-batch
  sliding window; `n_unique / prod(levels)`.

## Output

`mini_experiments/11_bottleneck_continuous_vs_fsq/results.md` containing:

1. 3 × 2 (× 3 seed) results table on HBN 6-task within-split.
2. Same on HBN 6-task LOSO 5-subject.
3. Codebook utilization trajectory per FSQ variant.
4. Reconstruction loss trajectory per variant (does FSQ slow down the
   reconstruction objective?).
5. Wall-clock cost per training step per variant.
6. The chosen bottleneck with decision-rule justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| FSQ joint training is unstable despite the warmup schedule (loss spikes when λ(t) ramps up) | Smooth the ramp; use longer warmup (0.30 → 0.50 of training); reduce target λ from 0.5 to 0.3. |
| FSQ codes collapse to a small subset because the encoder finds a local minimum | Add a small entropy bonus on the FSQ usage distribution (à la VQ-GAN's diversity loss); this works in FSQ because there's no codebook to compete with. |
| The 6-dim FSQ space is too low-dimensional and bottlenecks too aggressively | Increase to `levels=[5,5,5,5,5,5,5,5]` (390K codes). Or test 8-dim variants. |
| LOSO with only 5 subjects is too noisy to detect a real LOSO improvement | Run a larger LOSO (10–15 subjects) for the FSQ winner only, after the within-split sweep narrows to a single FSQ variant. |
| FSQ wins on LOSO but the gain disappears on the headline run with full pretraining | This is a known scaling risk; carry FSQ forward as an experimental arm but keep continuous as the safe default. |

## What gets carried forward

The single chosen bottleneck (continuous or one of the FSQ variants). If
FSQ wins, all the implementation details (levels, projection dimension,
λ warmup schedule, codebook utilization monitor) are frozen. If continuous
wins, this experiment is reported as a negative result and the headline
run uses continuous.
