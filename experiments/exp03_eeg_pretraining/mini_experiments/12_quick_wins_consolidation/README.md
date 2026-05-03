# exp03 / mini-experiment 12 — Quick wins consolidation

> **Status:** planned
>
> **Cross-reference:** [`methodology.md` §3 ablation matrix](../../methodology.md#3-how-to-design-one-ablation-the-matrix-shape),
> all prior mini-experiments
>
> **Compute budget:** 12 H100-hours (4 cells × 2 control columns × 3 seeds
> for each of two stacking blocks)
>
> **Gates:** none downstream — exp12 is the final consolidation that produces
> the headline-run configuration

## Question

Three modifications were proposed across the conversation as "free wins"
that should be applied uniformly across any architecture: Snake activations
in the convolutional frontend (BigVGAN-style periodic inductive bias),
BlurPool anti-aliasing before any strided operation, and VICReg as an
auxiliary regulariser on the encoder's pooled output. Do they survive a
strict ablation when stacked together with all the other choices made in
exp02–exp11, and does the resulting "everything-on" configuration
strictly improve over the "exp02–exp11 winners only" configuration?

## Why it matters

Every prior mini-experiment isolated one design axis. The legitimate
worry is that gains from individual experiments do *not* compose — what
helps in isolation can hurt in combination because the gradient signal,
loss balance, or regularisation regime changes. The Smol Training
Playbook's explicit warning ([`methodology.md` §1](../../methodology.md#1-the-mental-model-pretraining-is-capability-engineering-not-luck)):
"the real value of ablation experiments lies … in providing confidence for
future debugging".

exp12 is also the first cell where we pay full attention to the **two**
"free win" candidates that were never directly tested in the other
experiments:

- ~~**Snake activations**~~ **DROPPED 2026-05-03 from the deep-research
  refresh.** The Signal-Processing subagent's finding: the BigVGAN
  Snake-activation paper's published periodic-extrapolation property
  *does not hold in practice* — the 2022 follow-up study and subsequent
  empirical analyses show that Snake networks default to near-linear
  behaviour during training, with the periodic structure rarely
  materialising. The recommendation, when periodic inductive bias is
  desired, is to use **SIREN-style sine activations** or the **DONN
  oscillatory RNN architecture** instead. For exp12, Snake is removed
  entirely from the candidate "free wins" list; the alternative
  (SIREN / DONN) is too architecturally invasive for a quick-wins
  ablation and is deferred to a hypothetical future "periodic-bias
  architecture" mini-experiment.
- **BlurPool**: replacing strided conv with conv + low-pass + subsample
  per [Zhang ICML 2019](https://arxiv.org/abs/1904.11486). Anti-aliasing
  the encoder's downsampling. **2026-05-03 caveat from the Signal-
  Processing subagent**: BlurPool's binomial-5 low-pass kernel can
  attenuate genuine high-γ neural content (gamma > 30 Hz, which is
  cognitively meaningful and where inner-speech / motor-imagery content
  lives). For images this doesn't matter (no analogous "gamma" content
  to lose); for EEG the cost-benefit of anti-aliasing vs high-γ
  attenuation must be checked. The W2 cell now includes a
  high-γ-attenuation diagnostic (see Decision rule below).
- **VICReg auxiliary**: variance + invariance + covariance regulariser on
  the encoder's mean-pooled output across two augmented views. Tested as
  a *standalone* SSL framework in exp04 S2 (where it was likely a weak
  loser) but never as an *auxiliary* loss on top of the chosen MAE
  framework. The covariance term in particular is theoretically expected
  to suppress subject-fingerprint capacity in the representation.

## Variants

The experiment runs in two phases.

### Phase A — isolated quick wins (independent of each other)

| Code | Variant | Modification |
| ---- | ------- | ------------ |
| W0 | exp02–exp11 winners (the "current best" baseline at this point in the sequence) | none added |
| ~~W1~~ | ~~W0 + Snake activations in the conv frontend~~ | **DROPPED 2026-05-03** — see "Why it matters" |
| W2 | W0 + BlurPool before every stride in the conv frontend | replace `Conv1d(stride=s)` with `Conv1d(stride=1) → BlurPool(stride=s)` |
| W3 | W0 + VICReg auxiliary loss with weight 0.1 | sample two augmented views per batch, compute VICReg on encoder pooled output, add to total loss |

Each isolated change must strict-win individually (ε ≥ 0.5 pp HBN 6-task
BAC with non-overlapping CIs) to be considered for stacking.

### Phase B — stacked combination (only includes the Phase A winners)

If both remaining Phase A variants strict-win:

| Code | Variant | Modification |
| ---- | ------- | ------------ |
| W4 | W0 + BlurPool + VICReg auxiliary | both together |

The honest expected outcome (revised 2026-05-03): one or both of W2 / W3
strict-win; W4 marginally improves over the best-individual but with
diminishing returns; the cost-benefit of stacking is the deciding
factor.

## Controls

For Phase A:

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| W0 baseline (exp02–exp11)      | ✓          | ✓                  |
| W2 + BlurPool                  | ✓          | ✓                  |
| W3 + VICReg auxiliary          | ✓          | ✓                  |

For Phase B (run only if both remaining Phase A cells strict-win):

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| W4 stacked (BlurPool + VICReg) | ✓          | ✓                  |

The matched-noise twin for W3 (VICReg auxiliary) is informative: a
collapse-prevention regulariser that "improves" on Gaussian noise is
suspicious, since there's nothing to prevent collapse to in the noise.

## Held constant

Everything that was previously decided. Specifically:

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: exp04 winner.
- Multi-rate strategy: exp05 winner.
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Target: exp08 winner.
- Multi-condition input: exp09 winner.
- Mask: exp10 winner.
- Bottleneck: exp11 winner.

The W4 cell is therefore the *full proposed headline-run configuration*,
modulo final hyperparameter tuning. If W4 underperforms W0, the headline
run uses W0 (the simpler config). If W4 wins, the headline run uses W4.

## Decision rule

Same as prior experiments:

- Strict win = ≥ 0.5 pp HBN 6-task BAC for individual quick wins (lower
  bar because these are supposed to be small effects), ≥ 1 pp for the W4
  stacked combination.
- Weak win = ≥ 0.25 pp with paired permutation p < 0.05.
- Tie = TOST equivalence within ε = 0.5 pp.
- For W4 only: must strict-win against the *best* of W0 / W2 / W3,
  not just against W0. This protects against false additivity claims.

Anti-shortcut criterion: VICReg auxiliary is specifically expected to
*decrease* the source-dataset probe accuracy. Log this; if the source
probe accuracy goes *up* under W3, the auxiliary loss is being
counterproductive and W3 is disqualified regardless of HBN 6-task gains.

**High-γ-content preservation (W2 / BlurPool only, added 2026-05-03)**:
the BlurPool low-pass kernel attenuates content above its cutoff. For
EEG, this includes the high-γ band (30–100 Hz) where cognitively
meaningful neural content lives (motor imagery beta-rebound, gamma-band
attention modulation). W2 must show ≥ 90 % retention of in-band spectral
power in the 30–80 Hz band on a held-out 1000-window sample measured via
Welch periodogram. If retention drops below 90 %, W2 is disqualified
despite any HBN 6-task BAC gain — the gain is then known to be from
removing aliased content rather than from preserving the signal we care
about, and the cost (high-γ loss) outweighs the benefit (anti-aliasing)
for our headline run target.

## Pre-registered predictions

| Variant | Prediction HBN 6-task BAC | Source-probe trajectory | High-γ retention |
| ------- | ------------------- | ------------------------ | ---------------- |
| W0 baseline | reference | already trending down per exp02–exp11 monitoring | reference |
| W2 + BlurPool | tied or weak win, ~+0.5 pp; effect mostly visible in multi-rate eval | unchanged | **risk** of < 90 % retention; must check |
| W3 + VICReg aux | weak win, ~+0.5 pp; significant improvement on cross-subject LOSO | source probe drops faster than W0 | unchanged |
| W4 stacked | strict win, ~+0.5–0.8 pp over W0; weak win over best individual | source probe drops fastest | inherits W2's risk |

If Phase A reveals that one or more "free wins" actually *hurt* (a real
possibility for VICReg auxiliary if its loss weight is wrong), they are
dropped from W4 and the experiment ends with the largest-gain individual
win as the chosen configuration.

## Implementation pointers

- ~~Snake activations~~: **DROPPED** — see "Why it matters".
- BlurPool: 1-D variant with binomial kernel `[1, 4, 6, 4, 1] / 16` (size 5).
  Reference: [`adobe/antialiased-cnns`](https://github.com/adobe/antialiased-cnns).
  After applying, run a Welch periodogram on a held-out 1000-window
  sample to verify ≥ 90 % retention of 30–80 Hz spectral power
  (high-γ-attenuation diagnostic, added 2026-05-03).
- VICReg auxiliary: SelfEEG implementation as the reference. Use the
  pooled output of the chosen backbone (mean over time) as the embedding;
  augmentations are temporal crop + amplitude scaling (already used in
  exp04 S2 and exp09); coefficients γ_var = 1.0, γ_inv = 1.0,
  γ_cov = 0.04.

## Output

`mini_experiments/12_quick_wins_consolidation/results.md` containing:

1. Phase A: 4 × 2 (× 3 seed) results table.
2. Phase B (if reached): 1 × 2 (× 3 seed) results for W4.
3. Source-probe trajectories for all variants.
4. Final headline-run configuration table — every choice from exp02–exp12
   listed in one place, with a one-sentence justification per choice.
5. Compute estimate for the headline run at this configuration.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| The exp02–exp11 winners change while exp12 is being designed (sequencing dependency) | exp12 is intentionally last; if any earlier experiment is rerun and changes its winner, re-run exp12 with the updated W0 baseline. |
| VICReg auxiliary loss weight (0.1) is wrong for our setup | Sweep γ_aux ∈ {0.01, 0.1, 1.0} for W3 only; report the best. |
| Snake α collapses to a single value across all channels | Same fix as exp02: add the standard repulsive initialisation pattern. |
| W4 stacked improves on W0 but the improvement is below noise (CI overlap) | Report it as "does not strict-win"; carry forward W0 as the safe default for the headline run. The net cost-benefit of W4's complexity must be positive. |
| Different Phase A winners are mutually destructive (e.g. BlurPool's low-pass interferes with Snake's periodic activation) | Add an additional cell in Phase B testing pairwise combinations W1+W2, W1+W3, W2+W3 before the full stack. Reports which combinations are additive. |

## What gets carried forward

The full headline-run configuration. After exp12 produces its
`results.md`, the next folder (`headline_run/` or equivalent) inherits the
exact configuration as a single config file. Phase 3 of the methodology
(intermediate-scale validation per [`methodology.md` §2 Phase 3](../../methodology.md#phase-3--intermediate-scale-validation-1-3-days-1-node))
is the next step after this experiment, scaled up by ~10× in data and
model size, holding the configuration fixed.
