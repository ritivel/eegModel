# exp03 / mini-experiment 15 — Loss-weight sensitivity and curriculum schedule

> **Status:** planned
>
> **Cross-reference:** [`brain/cortico-ssl-hypothesis.typ` §7, §9.1](../../../../../brain/cortico-ssl-hypothesis.typ),
> [`methodology.md` §4](../../methodology.md#4-hyperparameter-transfer-when-small-scale-tuning-is-trustworthy)
>
> **Compute budget:** 24 H100-hours (Phase A 8 weight-sweep cells +
> Phase B 4 curriculum cells, each × 2 control columns × 3 seeds)
>
> **Gates:** none downstream — exp15 is the final pre-headline-run
> sensitivity check; its output is the loss-weight vector and the
> curriculum schedule for the headline run

## Question

Are the loss weights specified in the cortico-ssl-hypothesis
($\lambda_\text{Bar}, \lambda_\text{IS}, \lambda_\text{STFT},
\lambda_\text{bisp}, \lambda_\text{VIC}, \lambda_\text{FSQ},
\lambda_\text{adv}) = (1.0, 0.3, 1.0, 0.1, 0.05, 1.0, 0.1)$ robust enough
that small perturbations do not change the qualitative outcome, and does
the proposed three-stage curriculum (Barron+MR-STFT only → ramp FSQ +
IS + bispectral → full loss with adversarial) materially improve over
training the full loss from step 0?

## Why it matters

The cortico-ssl-hypothesis explicitly flags weights as "the part of the
recipe most likely to need tuning". Multi-objective SSL is brittle in
two known ways:

1. **Weight imbalance**: a single dominant head (e.g.
   $\lambda_\text{STFT} = 1.0$ when MR-STFT magnitudes are much larger
   than other heads) can effectively reduce the others to noise. The
   hypothesis weights are a starting point chosen on theory grounds, not
   from an actual sweep on EEG.
2. **Curriculum dependency**: starting all losses at step 0 is the simple
   default but produces three known failure modes —
   - The FSQ codes are random at step 0; a strong $\lambda_\text{FSQ-CE}$
     forces the encoder to predict noise, destabilising training.
   - The bispectral loss requires a meaningful spectrum to compute against,
     which an untrained network does not produce; large
     $\lambda_\text{bisp}$ from step 0 contributes high-variance
     gradients.
   - The adversarial head needs the encoder to encode dataset identity
     before it can suppress it; ramping it from 0 (per Ganin et al. DANN)
     is standard.

The hypothesis §9.1 specifies a three-stage curriculum:

```
Steps 0–5k:    Barron + MR-STFT only.
Steps 5k–15k:  Ramp FSQ-CE + IS + bispectral with 0.1 × final weight.
Steps 15k–end: Full loss, full weights, AAMP at 50 %.
```

This is reasonable but never validated empirically. exp15 splits the
question into two phases: weight sensitivity (Phase A) and curriculum
necessity (Phase B).

## Variants

### Phase A — weight sensitivity sweep

We perturb each loss weight by ± 2× while holding the others at the
hypothesis defaults, plus one cell where all loss weights are normalised
by their gradient magnitudes (the "GradNorm" trick of
[Chen et al. ICML 2018](https://arxiv.org/abs/1711.02257)).

| Code | $\lambda_\text{Bar}$ | $\lambda_\text{IS}$ | $\lambda_\text{STFT}$ | $\lambda_\text{bisp}$ | $\lambda_\text{VIC}$ | $\lambda_\text{FSQ}$ | $\lambda_\text{adv}$ | Notes |
| ---- | -------------------- | ------------------- | --------------------- | --------------------- | -------------------- | -------------------- | -------------------- | ----- |
| L0 (baseline) | 1.0 | 0.3 | 1.0 | 0.1 | 0.05 | 1.0 | 0.1 | The hypothesis recipe |
| L1 (–Barron) | 0.5 | 0.3 | 1.0 | 0.1 | 0.05 | 1.0 | 0.1 | half-weight Barron |
| L2 (+Barron) | 2.0 | 0.3 | 1.0 | 0.1 | 0.05 | 1.0 | 0.1 | double-weight Barron |
| L3 (–STFT)   | 1.0 | 0.3 | 0.5 | 0.1 | 0.05 | 1.0 | 0.1 | half-weight MR-STFT |
| L4 (+STFT)   | 1.0 | 0.3 | 2.0 | 0.1 | 0.05 | 1.0 | 0.1 | double-weight MR-STFT |
| L5 (+bisp)   | 1.0 | 0.3 | 1.0 | 0.5 | 0.05 | 1.0 | 0.1 | 5× bispectral |
| L6 (+VIC)    | 1.0 | 0.3 | 1.0 | 0.1 | 0.20 | 1.0 | 0.1 | 4× VICReg auxiliary |
| L7 (GradNorm)| auto-normalised across heads per [Chen et al. ICML 2018](https://arxiv.org/abs/1711.02257) | — | — | — | — | — | — | adaptive |

We do not exhaustively grid over 7 weights — that would be 2⁷ = 128
cells. The sweep is one-at-a-time around the baseline, plus an adaptive
control. If the baseline turns out to be on a sharp ridge (multiple
single-weight perturbations strict-lose), we add a 2-axis cell at the
end to localise the ridge.

### Phase B — curriculum schedule

Phase B compares the hypothesis curriculum vs three alternatives.

| Code | Curriculum | Schedule |
| ---- | ---------- | -------- |
| Cu0 (no curriculum) | full loss from step 0 | All 7 losses at hypothesis weights from step 0. |
| Cu1 (hypothesis curriculum) | the §9.1 three-stage | Steps 0–5 k: Barron + MR-STFT only. 5–15 k: ramp FSQ + IS + bispectral with 0.1× final. 15 k–end: full. Adversary GRL ramps over first 10 % of total steps. |
| Cu2 (FSQ-only warmup) | only FSQ ramps, others fixed | All non-FSQ losses at full weight from step 0. FSQ ramps via the soft-rounding schedule of exp11 (5 % of training). |
| Cu3 (Cu1 with longer warmup) | Cu1 stretched | Steps 0–10 k: Barron + MR-STFT. 10 k–30 k: ramp. 30 k–end: full. |

## Controls (the §3 matrix)

|        | EEG signal | matched-noise twin |
| ------ | ---------- | ------------------ |
| L0..L7 | ✓ | ✓ (only on L0 + L7 to keep budget; the others use the L0 noise twin as a shared baseline) |
| Cu0..Cu3 | ✓ | ✓ (only on Cu0 + Cu1) |

The reduced noise-twin coverage is a budget concession; the hypothesis
weights are the central case and require a noise twin, the
adaptive-weight cell L7 also does (because GradNorm could in principle
"discover" that fitting noise is the easiest objective), and the rest
inherit Cu0 / L0's noise-twin baseline.

## Held constant

Everything from the exp02–exp14 winners. Specifically:

- Frontend, backbone, SSL framework, multi-rate strategy, reconstruction
  loss, phase loss, target signal, multi-condition input, mask, bottleneck,
  quick-wins stack: all set to their respective experiment winners.
- Window length: exp14 winner (likely C2 16 s or C3 30 s).
- Adversarial head: exp13 winner (the choice of head architecture and
  $\lambda_\text{adv}$, but only the *initial* weight; this experiment
  also re-explores the weight as part of the L0..L7 sweep).
- Pretraining duration: 8 epochs of the 100 h HBN-EEG subset (per [`mini_experiments.md` §4.1](../../mini_experiments.md#41-pretraining-corpus)).
- Optimiser: AdamW, LR carried forward.

The new ingredients are the loss-weight perturbations (Phase A) and the
curriculum schedule (Phase B).

## Decision rule

**Phase A — weight sensitivity**:

For each weight perturbation, we are *not* looking for a strict win — we
expect the hypothesis recipe to be near-optimal. We are looking for
**non-degradation**:

- *Pass*: HBN 6-task BAC within 0.5 pp of the L0 baseline (TOST equivalence
  $\varepsilon = 0.5$ pp). The baseline is robust along this axis.
- *Adopt*: ≥ 0.5 pp HBN 6-task BAC improvement with non-overlapping CIs. The
  perturbation is preferable to the hypothesis recipe.
- *Reject*: ≥ 0.5 pp HBN 6-task BAC degradation with $p < 0.05$. The
  hypothesis recipe is brittle along this axis; the headline run sticks
  to the unperturbed weight or, if multiple perturbations reject, the
  L7 GradNorm variant is adopted.

If L7 (GradNorm) strict-wins over L0 by ≥ 1 pp, the headline run uses
adaptive weighting and the manual weights become a documented starting
point only.

**Phase B — curriculum**:

- *Pass for Cu1 (hypothesis curriculum)*: ≥ 0.5 pp HBN 6-task BAC over
  Cu0 (no curriculum) with $p < 0.05$, *or* end-of-training FSQ
  utilisation ≥ 80 % when Cu0's is < 50 % (the structural failure
  mode the curriculum is designed to prevent).
- *Choose Cu2 (FSQ-only)*: if Cu2 ≥ Cu1 on HBN 6-task BAC, prefer Cu2 because
  it is simpler.
- *Choose Cu3 (longer warmup)*: only if both Cu1 and Cu2 fail to
  stabilise FSQ utilisation.

A failure of all curricula (Cu1, Cu2, Cu3 all tied with Cu0) means
single-stage joint training works without any curriculum at our scale,
and the headline run drops the curriculum entirely. That would be a
clean, useful negative result.

## Pre-registered predictions

| Variant | Predicted HBN 6-task BAC | Predicted FSQ utilisation | Predicted gradient stability |
| ------- | ------------------- | -------------------------- | ----------------------------- |
| L0 hypothesis | reference | ~85 % | stable |
| L1 –Barron | tied or weak loss (~–0.3 pp); P1 attack weakens | unchanged | slightly higher gradient variance |
| L2 +Barron | tied (saturated effect) | unchanged | stable |
| L3 –STFT | weak loss (~–0.5 pp); the spectral anchor weakens | unchanged | stable |
| L4 +STFT | weak loss (~–0.5 pp); STFT dominates other heads | unchanged | spike in batch gradient norm |
| L5 +bisp | tied; bispectral gradient is high-variance, more noise | unchanged | spikes during ramp |
| L6 +VIC | weak win on cross-subject LOSO; tied on within-split | unchanged | stable |
| L7 GradNorm | weak win, ~+0.5 pp | unchanged | stable |
| Cu0 no curriculum | reference | 50–70 % (FSQ struggles to converge from step 0) | spikes in first 5 k steps |
| Cu1 hypothesis curriculum | weak win, ~+0.5 pp; FSQ utilisation up | > 85 % | stable throughout |
| Cu2 FSQ-only warmup | tied or weak win over Cu0 | ~85 % | stable |
| Cu3 longer warmup | tied with Cu1 | > 85 % | stable; possibly slightly slower convergence |

The honest expected outcome: **the hypothesis recipe is robust along
single-weight perturbations (Phase A all pass), Cu1 or Cu2 strict-wins
over Cu0 mostly via the FSQ-utilisation argument, and L7 GradNorm gives
a small additional bump that is probably not worth the implementation
complexity**. The decision is whether to adopt GradNorm and Cu2 (the
two "small wins") or stick with the manual weights + Cu1.

## Implementation pointers

- Loss-weight sweep: trivial — multiply the existing loss heads by the
  variant-specific weights. Re-tune the global LR per cell because
  changing total loss magnitude changes the effective LR; sweep
  `{0.5×, 1×, 2×}` of the L0-best LR.
- GradNorm: reference implementation
  [`brianlan/pytorch-grad-norm`](https://github.com/brianlan/pytorch-grad-norm).
  Treat each loss head as a "task" and use the shared encoder's last
  layer as the shared parameter for gradient-norm balancing.
  Initial weights = the hypothesis L0 values.
- Curriculum implementation:
  ```python
  def curriculum_weights(step, total_steps):
      if step < 0.05 * total_steps:        # 0–5%
          return dict(barron=1.0, stft=1.0, others=0.0)
      elif step < 0.15 * total_steps:      # 5–15%
          alpha = (step - 0.05 * total_steps) / (0.10 * total_steps)
          return dict(barron=1.0, stft=1.0,
                      is_=alpha * 0.3, fsq=alpha * 1.0, bisp=alpha * 0.1,
                      vic=alpha * 0.05, adv=alpha * 0.1)
      else:
          return dict(barron=1.0, stft=1.0, is_=0.3, fsq=1.0,
                      bisp=0.1, vic=0.05, adv=0.1)
  ```
- Logging: per checkpoint, log each per-head loss value *and* its raw
  gradient norm before scaling. This is essential for diagnosing whether
  a perturbation rejected because of representation quality or because
  of gradient-scale issues.

## Output

`mini_experiments/15_loss_weight_curriculum/results.md` containing:

1. Phase A: 8 × 2 (× 3 seed) results table with HBN 6-task BAC, HBN
   ADHD-binary AUROC (and TUEV BAC, TUAB AUROC when TUH access lands),
   per-head loss, per-head gradient norm at end of training.
2. Phase B: 4 × 2 (× 3 seed) results table; FSQ utilisation trajectory
   per curriculum.
3. Sensitivity heatmap: HBN 6-task BAC as a function of each weight
   perturbation (visualises ridge or basin behaviour).
4. The chosen loss-weight vector and the chosen curriculum schedule.
5. Compute estimate for the headline run at the chosen configuration.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| Single-weight perturbations are too small to differentiate variants | Increase perturbation to ± 4× for the three weights with the largest gradient magnitudes (likely $\lambda_\text{STFT}, \lambda_\text{Bar}, \lambda_\text{FSQ}$). |
| GradNorm is unstable in a 7-task setup | Use the simpler "uncertainty weighting" variant of [Kendall et al. CVPR 2018](https://arxiv.org/abs/1705.07115) as a fallback — it has fewer hyperparameters and is more robust. |
| Cu0 (no curriculum) crashes due to FSQ blow-up at step 0 | This *is* a result; record the failure mode and the step at which divergence occurs, then continue with Cu1/Cu2/Cu3 only. |
| All curricula tie with Cu0 because our scale is too small to expose the FSQ-warmup necessity | Re-run only the FSQ-relevant cells at the 500 h subset; the curriculum benefit is expected to grow with corpus size. |
| Phase A and Phase B interact (the hypothesis weights might be optimal under Cu1 but not Cu0) | Phase B only uses the L0 weights. Phase A only uses Cu1. We do not run a full $A \times B$ grid — that's a documented limitation. |

## What gets carried forward

The full headline-run loss specification: a single weight vector
$(\lambda_\text{Bar}, \lambda_\text{IS}, \lambda_\text{STFT},
\lambda_\text{bisp}, \lambda_\text{VIC}, \lambda_\text{FSQ},
\lambda_\text{adv})$ — either the hypothesis values, an adaptive
weighting scheme, or a perturbed manual vector — *and* the curriculum
schedule expressing how those weights are introduced over training. The
full configuration table in §11 of the eventual headline-run plan
inherits this directly.
