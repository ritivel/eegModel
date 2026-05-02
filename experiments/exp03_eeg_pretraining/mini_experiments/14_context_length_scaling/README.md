# exp03 / mini-experiment 14 — Window length / context-length scaling

> **Status:** planned
>
> **Cross-reference:** [`brain/cortico-ssl-hypothesis.typ` §11 H1, H3](../../../../../brain/cortico-ssl-hypothesis.typ),
> [`methodology.md` §4](../../methodology.md#4-hyperparameter-transfer-when-small-scale-tuning-is-trustworthy),
> [EEGM2 (arXiv:2502.17873)](https://arxiv.org/abs/2502.17873)
>
> **Compute budget:** 24 H100-hours (4 window lengths × 2 backbones × 2 control
> columns × 3 seeds = 48 cells with strongly varying per-cell cost — the 30 s
> cells dominate the budget)
>
> **Gates:** none downstream — exp14 reports a configuration choice for the
> headline run

## Question

Does the long-context capability that justifies the choice of bidirectional
Mamba-2 over a Transformer (the O(N) vs O(N²) argument) actually translate
into better representations on downstream tasks, or is the optimal pretraining
window the standard 4-second window after all?

Concretely: at fixed total tokens, does pretraining at 4 s / 8 s / 16 s /
30 s windows on HBN-EEG produce different downstream behaviour, and does
the expected scaling pattern of EEGM2 ([Hong et al. 2025](https://arxiv.org/abs/2502.17873))
— performance peaks at ~ 30 s before plateauing on TUAB — replicate under
our
recipe?

## Why it matters

Three claims in the cortico-ssl-hypothesis depend critically on long-context
capability:

1. *H1 / H3.* The headline-run bar is HBN ADHD-binary AUROC ≥ 0.85
   (plus TUAB AUROC ≥ 0.89 as the literature-comparable secondary when
   TUH access lands) and rate invariance up to 2 kHz; at 2 kHz a 30-second
   window is **60 000
   samples**. A Transformer with FlashAttention-2 is intractable at this
   scale on H100 (memory grows quadratically in sequence length). The
   entire argument for picking Mamba-2 over a Transformer rests on the
   long-context regime being (a) feasible only for the SSM family and
   (b) actually beneficial.
2. *Phase-coupling biomarkers (P3, P7).* Theta–gamma cross-frequency
   coupling, the canonical signature of working memory, manifests over
   1–4-second cycles of theta. A 4-second window contains 1–2 theta
   cycles — enough to estimate phase but barely. An ERP-locked working
   memory effect across multiple trials needs 10–30 s of context to be
   identifiable in resting-state-like windows.
3. *Sleep / clinical EEG.* Sleep stages (NREM-1 vs NREM-2 vs REM) are
   *defined* on 30-second epochs (the AASM standard). A 4-second window
   forces the model to either re-aggregate at fine-tune time or to give
   up on sleep-staging benchmarks. EEGM2's 30-second peak on TUAB
   ([Hong et al. 2025](https://arxiv.org/abs/2502.17873)) is consistent
   with both arguments. (We will replicate this finding primarily on
   HBN ADHD-binary, with TUAB as a literature-comparable secondary when
   TUH access lands.)

The claim is plausible on theory grounds but **never directly tested in our
recipe**. exp02–exp13 all use the standard 4-second window. exp14 settles
the long-context question with a controlled iso-token comparison.

The empirical question is *not* "is longer better" (the answer is usually
"yes up to a point") but "*how much* better, and does the optimal point
sit before or after the compute cliff that retires the Transformer
backbone alternative".

## Variants

The variants differ in pretraining window length. Each variant is run with
two backbones (the exp03 winner — likely Mamba-2 — and a Transformer
control) so we can read off the long-context benefit *and* the
backbone-specific feasibility cliff.

| Code | Window length | # samples @ 250 Hz | # samples @ 2 kHz (eval only) | Backbones tested |
| ---- | ------------- | ------------------ | ------------------------------ | ----------------- |
| C0 | 4 s (the default) | 1 000 | 8 000 | exp03 winner + Transformer |
| C1 | 8 s | 2 000 | 16 000 | exp03 winner + Transformer |
| C2 | 16 s | 4 000 | 32 000 | exp03 winner + Transformer |
| C3 | 30 s | 7 500 | 60 000 | exp03 winner + (Transformer disqualified at 60k via OOM) |

The 30-second × 2-kHz cell is the cortico-ssl-hypothesis's headline
operating point. We evaluate all variants at this rate even when
pretrained on a shorter window.

## Controls (the §3 matrix)

|        | EEG signal | matched-noise twin |
| ------ | ---------- | ------------------ |
| C0 | ✓ | ✓ |
| C1 | ✓ | ✓ |
| C2 | ✓ | ✓ |
| C3 | ✓ | ✓ |

**Iso-token, not iso-step.** Each variant is trained for the *same number
of total samples seen* (≈ 35 M tokens), so longer-window cells take fewer
optimiser steps with proportionally larger batches. This is the right
match for representation-quality comparisons; matching steps would unfairly
favour shorter windows by giving them more gradient updates.

The matched-noise twin doubles as a check that long-context capacity
isn't being credited where it shouldn't: if a backbone improves on EEG and
*also* improves on Gaussian noise of equal length, what it's modelling is
positional correlations in the noise, not EEG content.

## Held constant

- Pretraining data: 100 h HBN-EEG subset (per [`mini_experiments.md` §4.1](../../mini_experiments.md#41-pretraining-corpus)), single rate 250 Hz (mixed-rate is
  exp05's question — a clean scaling sweep needs one rate held constant).
- Frontend: exp02 winner.
- SSL framework: exp04 winner.
- Bottleneck: continuous (FSQ adds a confounding factor — its codebook
  utilisation depends on token count, so we hold this fixed at continuous
  for the scaling sweep; if FSQ won exp11, run a follow-up sweep at the
  exp14 winner's window length).
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Target: exp08 winner.
- Multi-condition input: exp09 winner.
- Mask: exp10 winner.
- Optimiser: AdamW, LR retuned per cell at proxy scale (one 3-LR sweep
  per cell, since LR does not transfer across batch size in our recipe).
- Decoder width: held constant (the mask-prediction decoder is rate-agnostic).

## Decision rule

Two metrics matter:

1. **Downstream representation quality**: HBN 6-task BAC + HBN
   ADHD-binary AUROC (primary; per §4.3) at the *standard 4-second eval
   window* (so the eval suite is identical across variants — only
   pretraining window varies). When TUH access lands, also report TUEV
   BAC + TUAB AUROC as the literature-comparable secondary.
   - Strict win = ≥ 1 pp HBN 6-task BAC over C0, non-overlapping CIs,
     noise-twin flat.
   - Weak win = ≥ 0.5 pp with paired permutation $p < 0.05$.
2. **Long-context-only metric**: sleep-stage classification on
   Sleep-EDF (5-class, 30-second epochs). A variant trained at 30 s is
   structurally suited; one trained at 4 s must aggregate at fine-tune
   time. Strict win for C3 over C0: ≥ 5 pp accuracy.

Two scaling-specific criteria:

- **Throughput sanity** (Transformer disqualification):
  measure seconds/training-step on a single H100 at each window length.
  A variant whose throughput falls below 0.5 train-tokens/s/H100
  (i.e. < 1 / 4 of C0's throughput) is disqualified from the headline
  run *even if* its representation quality strict-wins, because the
  full 60 k-hour pretrain would not fit budget.
- **NaN / loss-spike incidence**: long-context training has known
  instability in SSMs (the segsum primitive in Mamba-2 in particular).
  Each cell must complete all 3 seeds without NaN losses; one NaN
  disqualifies the cell unless the bug is specifically pinpointed and
  fixed.

## Pre-registered predictions

| Variant | Backbone | Predicted HBN 6-task BAC | Predicted Sleep-EDF acc | Throughput |
| ------- | -------- | ------------------ | ------------------------ | ---------- |
| C0 4 s | Mamba-2 | reference | floor on Sleep-EDF | reference |
| C0 4 s | Transformer | tied | floor | reference |
| C1 8 s | Mamba-2 | weak win, ~+0.5 pp | weak win, ~+2 pp | ~0.7× |
| C1 8 s | Transformer | tied | weak win | ~0.6× |
| C2 16 s | Mamba-2 | strict win, ~+1–1.5 pp | strict win, ~+5 pp | ~0.4× |
| C2 16 s | Transformer | tied | tied | ~0.15× (already painful) |
| C3 30 s | Mamba-2 | strict win, ~+1.5–2 pp | strict win, ~+8 pp | ~0.2× |
| C3 30 s | Transformer | n/a (OOM) | n/a | < 0.05× |

The honest expected outcome (based on EEGM2 §5):
**C3 / Mamba-2 wins on the absolute metric and Sleep-EDF; C2 / Mamba-2 is
the best cost-quality compromise; the Transformer hits a cliff between C1
and C2**. The decision is whether to pay the 5× wall-clock cost of
30-second pretraining for a 1.5–2 pp gain. If the headline run targets
clinical / sleep deployment, yes. If it targets BCI accuracy at standard
windows, C0 / C1 may be sufficient.

## Implementation pointers

- Window-length sweep: in the dataloader, change only the `window_seconds`
  config; downstream framing (mask ratio, hop length) is recomputed
  proportionally.
- Iso-token training: scale total optimiser steps as
  `steps = total_tokens / (batch * window_samples)`. Re-tune LR per cell
  via a 3-LR sweep `{0.5×, 1×, 2×}` of the C0-best LR.
- Mamba-2 long-context: use the official `mamba_ssm` Triton kernel; do
  not reimplement segsum. Set `chunk_size=2048` for sequences > 16 k
  samples (per the
  [Mamba-2 reference issue 256](https://github.com/state-spaces/mamba/issues/256)).
- Transformer long-context: use FlashAttention-2 via
  `torch.nn.functional.scaled_dot_product_attention`. Expect OOM at C3
  with batch ≥ 4 on a single H100; this is the cliff the experiment is
  designed to surface.
- Sleep-EDF: standard 5-class fine-tune on 30-second epochs from
  Sleep-EDFx ([physionet 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)),
  4-fold cross-validated.

## Output

`mini_experiments/14_context_length_scaling/results.md` containing:

1. 4 × 2 × 3-seed results table on HBN 6-task BAC, HBN ADHD-binary AUROC (and TUEV BAC, TUAB AUROC when TUH access lands).
2. Same on Sleep-EDF accuracy + macro-F1.
3. Throughput table (samples/s/H100) per cell — annotated with the
   "feasibility cliff" line for the Transformer.
4. Per-window reconstruction-loss curves (longer windows have lower
   per-token loss because they have more context; this is expected
   and not a sign of better representation).
5. Phase-locking-value reconstruction quality at each window length —
   the long-context argument predicts longer windows preserve theta-
   phase better.
6. Wall-clock cost extrapolated to the 60 k-hour headline run.
7. The chosen pretraining window length with decision-rule justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| Mamba-2 segsum NaN at 60 k samples destroys the C3 cell | Use the official Triton kernel; verify on 100-step toy run before launching the full sweep; fallback to chunked-state mode if needed. |
| 30-second pretraining is too slow to complete in 24 H100-hours | Run C3 at iso-step instead of iso-token; report the result with the caveat. |
| Sleep-EDF eval doesn't differentiate variants because HBN-pretrained encoders need substantial fine-tuning to transfer to sleep | Add the HBN resting-state-eyes-closed task (well-defined sleep-adjacent state) for closer-to-pretrain transfer; add TUSL when TUH access lands. |
| Iso-token comparison is unfair to short windows because batch-size effects on convergence are nonlinear | Run a follow-up with iso-step matching; report both. |
| The 100 h corpus is too small to differentiate long-context cells (each window appears once) | Repeat with the 500 h subset; expected to widen the gap because longer windows benefit more from data scale. |
| All cells tie because the standard eval suite (HBN 6-task / HBN ADHD-binary) is dominated by short-range features | Add the [HBN-EEG ChallengeC1 cross-task](https://hbn-challenge-2025.github.io/) RT decoding, which involves ~10 s prestimulus context. |

## What gets carried forward

The single chosen pretraining window length. If the answer is C0 (4 s),
the multi-rate scaling claim from §11 H3 is downgraded to *"the model
can be applied at any rate, but pretraining at 4 s remains optimal"*.
If C3 (30 s) wins, the headline run uses 30-second windows, the
Transformer-vs-SSM argument is settled empirically (not just on
asymptotic-complexity grounds), and the compute budget for the headline
run is recomputed accordingly.
