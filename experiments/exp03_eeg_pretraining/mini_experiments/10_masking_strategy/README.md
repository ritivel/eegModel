# exp03 / mini-experiment 10 — Masking strategy

> **Status:** planned
>
> **Cross-reference:** [EEG2Rep SSP](https://arxiv.org/html/2402.17772v2),
> [NeurIPT AAMP](https://arxiv.org/abs/2510.16548), [I-JEPA multi-block masking](https://arxiv.org/abs/2301.08243)
>
> **Compute budget:** 8 H100-hours (4 mask variants × 2 control columns × 3
> seeds = 24 cells × 20 min average — masking is cheap to vary)
>
> **Gates:** none downstream

## Question

Within the chosen MAE-style SSL framework, which masking strategy produces
the best representation: random patch mask (the vanilla baseline), block
masking, semantic-subsequence-preserving (SSP) from EEG2Rep, multi-block
masking from I-JEPA, or amplitude-aware masked pretraining (AAMP) from
NeurIPT?

## Why it matters

In MAE-style training, the masking strategy chooses which parts of the
input the model is forced to predict from context. For natural images,
random patch masking at 75 % works because images have high spatial
redundancy; nearby pixels are highly predictable from each other. EEG has
the analogous redundancy in time, but with two specific structures that
random masking gets wrong:

- **Easy interpolation regions**: long stretches of stable rhythmic
  activity (alpha rhythm during eyes-closed rest) can be reconstructed by
  trivially interpolating between unmasked patches. The model gets a
  beautiful loss decrease without learning anything about the actual
  signal-defining events. EEG2Rep's SSP masking
  ([arXiv:2402.17772](https://arxiv.org/html/2402.17772v2)) directly
  addresses this by deciding which segments to *preserve* rather than
  which to mask.
- **High-amplitude artifact regions**: random masking masks artifact
  windows as often as signal windows, but the model's reconstruction loss
  is then dominated by trying to predict the artifact (which is by
  definition unpredictable). NeurIPT's AAMP
  ([arXiv:2510.16548](https://arxiv.org/abs/2510.16548)) preferentially
  masks high-amplitude regions, forcing the model to predict them from
  surrounding context — which means the model learns to characterise the
  *neural state in which artifacts happen* rather than the artifacts
  themselves.

I-JEPA's multi-block masking is a third pattern: rather than scattered
patches, mask 4 large contiguous blocks. This is what gave I-JEPA its
~20 pp linear-probe gain on ImageNet and is included as a control.

## Variants

| Code | Variant | Masking algorithm | Mask ratio |
| ---- | ------- | ----------------- | ---------- |
| K0 | Random patch mask (vanilla MAE) | uniform sample of 50 % of patches | 50 % |
| K1 | Block masking | mask one or two contiguous blocks totalling 50 % of patches | 50 % |
| K2 | SSP (semantic subsequence preserving) | choose 2 contiguous unmasked subsequences spanning 50 % total, mask the rest | 50 % |
| K3 | Multi-block masking (I-JEPA-style) | mask 4 large contiguous blocks totalling 60 %, with 10 % unmasked context blocks | 60 % |
| K4 | AAMP (amplitude-aware) | sample a percentile ξ ∈ [50 %, 90 %] per batch; mask the patches whose mean absolute amplitude is closest to that percentile | 50 % |

The mask ratio variation between K3 and others is intentional: I-JEPA's
gain came partly from the higher mask ratio (60 % is its sweet spot per
the published paper), and we want to give that variant a fair chance.
EEG2Rep's SSP also performs best at 50 % masking per their published
ablation.

## Controls

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| K0 random                      | ✓          | ✓                  |
| K1 block                       | ✓          | ✓                  |
| K2 SSP                         | ✓          | ✓                  |
| K3 multi-block                 | ✓          | ✓                  |
| K4 AAMP                        | ✓          | ✓                  |

For matched-noise twin: AAMP applied to Gaussian noise will preferentially
mask high-amplitude noise samples, but those have no special meaning in
Gaussian noise (every sample's amplitude is just a draw from N(0,1)). So
AAMP on noise should perform identically to random mask on noise, which
is the right sanity check — if AAMP on noise improves over K0 on noise,
the AAMP gain is incidental to mask placement statistics, not to
amplitude-aware semantics.

## Held constant

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: exp04 winner.
- Bottleneck: continuous.
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Target: exp08 winner.
- Optimiser: AdamW, LR carried forward.
- Pretraining duration: 8 epochs.

## Decision rule

Same as prior experiments:

- Strict win = ≥ 1 pp TUEV BAC, non-overlapping CIs, noise-twin flat.
- Weak win = ≥ 0.5 pp with paired permutation p < 0.05.
- Tie = TOST equivalence within ε = 0.5 pp.

One masking-specific criterion:

- **Reconstruction loss must remain non-trivial**: a masking strategy that
  is too easy (low mask ratio, mask only obvious noise) will drive
  reconstruction loss to zero. The model then learns nothing. The minimum
  acceptable end-of-training reconstruction loss across variants must be
  > 0.1 (in normalised units after z-score). If a variant's loss is
  lower than this, it's disqualified — it solved a trivial task.

## Pre-registered predictions

| Variant | Prediction TUEV BAC |
| ------- | ------------------- |
| K0 random | reference |
| K1 block | tied; block forces some non-trivial prediction but mask is too coarse |
| K2 SSP | weak win, ~+0.5 pp; preserves semantic context |
| K3 multi-block | strict win, ~+1–2 pp; the I-JEPA insight applies cross-modally |
| K4 AAMP | strict win, ~+1 pp; amplitude-awareness directly attacks the dominant pathology |

The honest expected outcome: **K4 AAMP wins on a low-SNR-sensitive eval
(TUAR), K3 multi-block wins on the standard TUEV BAC**. They might also
combine cleanly: AAMP-style amplitude scoring within the multi-block
selection. We allow one post-hoc combined cell K5 = AAMP-restricted
multi-block to be added if K3 and K4 both strict-win independently.

## Implementation pointers

- Random / block / multi-block: trivial. PyTorch boolean tensor of shape
  `(batch, num_patches)` indicating which positions to mask.
- SSP: per the EEG2Rep paper Algorithm 1. Choose `M` random starting
  points, "grow" each into a preserved subsequence by symmetric expansion
  until the union of preserved subsequences reaches the target unmasked
  fraction. Standard reference at
  [`navidfoumani/EEG2Rep`](https://github.com/Navidfoumani/EEG2Rep).
- AAMP: compute per-patch RMS amplitude. Sample percentile ξ ∈ [0.5, 0.9]
  per batch. Mask patches whose RMS is closest to that percentile (within
  ±5 % of the targeted percentile). Cost: one extra `torch.quantile` per
  batch. Reference at the NeurIPT paper.
- All: ensure mask is recomputed per training step (i.e. don't cache).

## Output

`mini_experiments/10_masking_strategy/results.md` containing:

1. 5 × 2 (× 3 seed) results table.
2. End-of-training reconstruction loss per variant.
3. The chosen masking variant with decision-rule justification.
4. If K5 (AAMP + multi-block) is added, its result.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| K4 AAMP changes the effective mask ratio per batch (because high-amplitude patches cluster in some recordings) | Add a normalisation step: after selecting amplitude-targeted patches, randomly subsample to exactly the target mask fraction. |
| K3 multi-block's higher mask ratio (60 %) gives a different effective training signal that's hard to compare iso-anything | This is the I-JEPA paper's choice; we keep 60 % to match. The decision rule already accounts for this by comparing on downstream metrics, not on loss. |
| K2 SSP preserves the wrong subsequences (e.g. picks the high-amplitude artifact regions to preserve) | Add an amplitude prior to the SSP starting-point selection — bias starting points toward moderate-amplitude regions. (This is essentially "AAMP-aware SSP", an organic post-hoc combination.) |
| All masking strategies tie because the dataset is too small | Re-run with the full TUEG (~5 % of full corpus) instead of 100h. |

## What gets carried forward

The single chosen masking strategy. The mask hyperparameters (ratio,
block count, AAMP percentile range) are frozen for every later
experiment.
