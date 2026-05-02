# exp03 / mini-experiment 01 — Sanity baselines

> **Status:** planned
>
> **Cross-reference:** `[methodology.md` §2 Phase 1](../../methodology.md#phase-1--sanity-baselines-½-day-1-gpu)
>
> **Compute budget:** ≤ 4 H100-hours (everything runs on a single GPU, most
> tests in seconds to minutes)
>
> **Gates:** every other mini-experiment

## Question

Does the trainer + eval + monitoring pipeline behave correctly on inputs
where the answer is mathematically known, before spending compute on real
ablations?

## Why it matters

Karpathy's first rule: *"a 'fast and furious' approach to training neural
networks does not work and only leads to suffering."* Pretraining at scale
fails silently. A miscalibrated trainer or eval pipeline produces beautifully
declining loss curves on top of a fundamentally broken setup, and no later
experiment can be trusted until the floor is verified.

The Phase-1 sanity tests in `[methodology.md` §2.1](../../methodology.md#phase-1--sanity-baselines-½-day-1-gpu)
are cheap (seconds to minutes per check, all on one GPU) and catch a
specific catalogue of bugs that tend to be invisible in headline metrics:
data-leakage shortcuts (the model "predicts" the source dataset and inflates
its loss decrease), loss-at-init mismatches (suggests wrong target dim or
wrong normalisation), one-batch overfit failures (suggests the encoder /
decoder cannot represent the target at all), random-init-linear-probe
floors (sets the ablation floor — anything within 1 % of this is broken),
and shape bugs in the masking pipeline (the model accidentally sees the
"hidden" patches via a `view` vs `transpose` mistake — Karpathy's specific
example).

Until every check below passes, no other mini-experiment runs.

## The five checks

### Check A — Loss-at-init matches theory

For each loss being considered (L1 on raw signal, L2 on raw signal, MR-STFT
log-magnitude, Itakura-Saito on periodogram, masked cross-entropy on FSQ
codes), train for zero steps and measure the loss on a fresh batch. The
expected values are derivable in closed form:


| Loss              | Expected at init                    | Note                                                |
| ----------------- | ----------------------------------- | --------------------------------------------------- |
| L2 raw            | ≈ Var(target) ≈ 1.0 (after z-score) | Anything else means normalisation broken            |
| L1 raw            | ≈ √(2/π) ≈ 0.80                     | Folded normal mean for unit-variance Gaussian       |
| MR-STFT log-mag   | ≈ scale-dependent, ≈ 1–3            | Compute on a held-out batch first as reference      |
| FSQ masked CE     | ≈ log(36000) ≈ 10.49                | log of vocabulary; any random init should give this |
| InfoNCE / NT-Xent | ≈ log(B) for batch size B           | Standard contrastive baseline                       |


Any cell that disagrees with theory by > 20 % indicates a normalisation,
target-shape, or vocab-size bug.

### Check B — Input-independent baseline

Replace the encoder's input with all zeros (or with a per-channel constant
equal to the global mean). Train for 5000 steps with the full SSL objective.
The loss must **not** decrease meaningfully. If it does, the model is
predicting the target from something other than the input — a positional
embedding leak, a target-dataset leak via batch composition, or a
normalisation that bakes the answer into the input statistics.

This is Karpathy's classic trap and the failure mode that broke the SmolLM3
1-trillion-token restart.

### Check C — One-batch overfit

Take 4 EEG epochs (one per channel from one recording). Train the full
pipeline on those 4 examples only, with any masking ratio. The SSL loss must
drive to within 1 % of the loss-at-init *floor* (i.e. essentially zero
reconstruction error / near-perfect mask prediction) within 1000 steps.

If it doesn't:

- Encoder capacity is too small for the target;
- Decoder capacity is too small for the target;
- The masking strategy hides the answer too aggressively (e.g. 95 % mask is
unrecoverable for 4 examples);
- A `view` vs `transpose` bug is mixing batch and time dimensions, so the
model never sees consistent inputs (Karpathy's example again);
- The optimizer is broken (LR way too small, weight decay clamping, etc.).

### Check D — Random-init linear probe floor

Take a freshly-initialised encoder (all the way through, no pretraining at
all). Run the standard eval suite per `[../../mini_experiments.md` §4.3
Protocol A](../../mini_experiments.md#43-evaluation-suite-for-every-experiment-unless-overridden):
HBN ADHD-vs-no-diagnosis binary AUROC, HBN 6-task classification balanced
accuracy + weighted F1, k-NN top-1 on a 10k HBN subset (and TUAB AUROC +
TUEV BAC/WF1 once TUH NEDC access lands, as Protocol A.4 secondaries).

The result is the **ablation floor**. Any future pretrained encoder whose
linear-probe metrics fall within 1 % of this floor is broken — either the
SSL signal didn't transfer, or the eval extraction is broken (wrong feature
layer, wrong pooling, wrong train-eval split). The floor must be recorded
once and referred to forever.

The [EEG-FM-Bench paper](https://arxiv.org/html/2508.17742v1) finding that
"many EEG-FMs barely beat random under linear probing" is the warning sign:
this number must be a real number (not a hand-wave) for every later
comparison to mean anything.

### Check E — Shape-and-mask audit

For one tiny batch (B=2, C=1, T=1024), print at every module boundary in the
pipeline:

```
input.shape = (2, 1, 1024), input.dtype = float32
after frontend conv: (2, 256, 128)
after positional embedding: (2, 128, 256)
mask.shape = (2, 128), mask.sum(-1) = [64, 64]   # 50 % mask, balanced
encoder input (visible only): (2, 64, 256)
encoder output: (2, 64, 256)
decoder input (visible + masked tokens): (2, 128, 256)
decoder output: (2, 128, 256)  -> reshape -> (2, 1, 1024) raw
target.shape = (2, 1, 1024), loss masked positions only
```

Every shape on every line must match what was intended. The Karpathy bug
this catches: one `view` instead of `transpose` somewhere in the masking
that lets the model see information from other batch entries (which then
trains "fine" on the SSL loss but learns nothing useful). This is the
single most common silent bug in MAE-style code.

## Pass criteria

All of the following must be true. No partial-credit:

- A: every loss within 20 % of theoretical value at init
- B: input-independent baseline does not improve loss by more than 1 %
relative to its own init value
- C: one-batch overfit hits < 1 % of init loss within 1000 steps
- D: random-init linear probe number recorded for the primary suite (HBN
ADHD-binary AUROC, HBN 6-task BAC + WF1, k-NN top-1) with 95 % CI; secondary
TUAB AUROC + TUEV BAC + WF1 added when TUH NEDC access lands
- E: shape table reviewed and signed off

If any check fails, fix the bug and re-run *all* checks (because the fix may
break a previously-passing check).

## Output

A single file `mini_experiments/01_sanity_baselines/results.md` containing:

1. The loss-at-init table from Check A with measured-vs-theoretical values.
2. The input-independent baseline curve over 5000 steps (showing it stays
  flat).
3. The one-batch overfit curve (showing < 1 % within 1000 steps).
4. The random-init linear probe table for the primary suite (HBN ADHD-binary
  AUROC, HBN 6-task BAC + WF1, k-NN top-1) with means and 95 % bootstrap CIs.
  Secondary row for TUAB / TUEV (left blank with a note "pending TUH NEDC
  access; see mini_experiments.md §4.3 Protocol A.4") that gets filled in
  later without re-running the rest of the experiment.
5. The shape audit table as printed.
6. The git SHA and config hash that was tested.

This file becomes the canonical reference for every subsequent
mini-experiment's "is my eval delta vs the floor real" question.

## What gets carried forward

- The trustworthy training + eval pipeline.
- The random-init linear-probe floor numbers.
- The fixed feature-extraction layer (last encoder layer, mean-pooled over
time, with no projection head) used for downstream linear probe in every
later experiment.
- A small library of "known-good" sanity checks that re-run on every
pipeline change (added to CI).

## Risks and mitigations


| Risk                                                                                          | Mitigation                                                                              |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Loss-at-init off by 20–50 % because batch normalisation hasn't run yet                        | Run 1 forward pass of warmup before measuring loss-at-init                              |
| Random-init linear probe gives suspiciously high numbers on the binary task                   | Binary clinical labels (HBN ADHD-binary, or TUAB normal/abnormal) are easy; the floor can be 60–70 % AUROC. Cross-check on the 6-class task (HBN 6-task, or TUEV) where the random floor is much lower. |
| Input-independent baseline accidentally trains because positional embeddings carry the signal | Use the same positional embedding the real model uses, but zero out all token content   |
| One-batch overfit succeeds via batch-leak (model predicts other batch entries)                | Already detected by Check E. If E passes, C cannot fail this way.                       |


