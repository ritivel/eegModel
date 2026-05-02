# exp03 / mini-experiment 04 — SSL framework ablation

> **Status:** planned
>
> **Cross-reference:** [`methodology.md` §6.4](../../methodology.md#64-pretraining-objectives--what-works-and-what-doesnt),
> [`brain/eeg-research.md` §6.4](../../../../../brain/eeg-research.md#64-pretraining-objectives--what-works-and-what-doesnt),
> [DeeperBrain (Wang et al., arXiv 2601.06134)](https://arxiv.org/abs/2601.06134)
> §III-G + Table IV; the dedicated NSP ablation is [exp16](../16_nsp_auxiliary_head/).
>
> **Compute budget:** 30 H100-hours (5 framework variants + 1 added MER+NSP
> variant × 2 control columns × 3 seeds = 30 cells × 60 min average —
> frameworks differ substantially in per-step cost; diffusion is most expensive)
>
> **Gates:** experiments 06, 07, 08, 09, 10, 11, 12, 16

## Question

Among the SSL framework topologies that are compatible with pure scratch
training (no teacher network, no EMA momentum encoder, no two-stage
tokenizer) — masked autoencoder with raw target, masked autoencoder with
denoised target, siamese decorrelation (VICReg), time-frequency consistency
with shared encoder (TF-C), and score-matching diffusion (EEGDM-style) —
which one produces the best low-SNR EEG representation in a single
optimisation?

## Why it matters

The cross-modal research subagent's most important finding: most published
"SSL" frameworks for EEG sneakily depend on a teacher network or EMA
momentum encoder, and lose their published numbers when you strip that out.
LaBraM, CBraMod, EEG-X, EEGPT, EEG2Rep, MAEEG, BENDR all fall in this
category to some degree. The set of frameworks that are *honestly*
single-encoder, scratch, no-momentum is small, and within that set the
choice strongly determines:

- whether the model learns to reconstruct *noise* (raw-MAE failure mode) or
  *signal* (denoised-MAE);
- whether collapse is prevented architecturally (VICReg's variance term) or
  only by the reconstruction objective (raw-MAE);
- whether the model has a structural mechanism for handling noise (score
  matching's denoising objective is *literally* about distinguishing signal
  from noise at every noise level);
- whether phase information is preserved by design (TF-C with complex STFT
  branch) or only via the loss (MAE with sin/cos circular loss).

Each framework attacks low-SNR via a different mechanism. exp04 asks which
mechanism actually wins on the standard frozen-probing eval suite from
[`mini_experiments.md` §4.3](../../mini_experiments.md#43-evaluation-suite-for-every-experiment-unless-overridden)
— primary HBN 6-task BAC + HBN ADHD-binary AUROC + k-NN, with TUAB/TUEV as
the literature-comparable secondary once TUH access lands.

## Variants

Each variant uses the chosen frontend (default until exp02 lands: F0
vanilla strided conv) and chosen backbone (default until exp03 lands:
bidirectional Mamba-2). The framework determines the topology around the
backbone and the loss.

| Code | Variant | Topology summary | Loss | No teacher / EMA |
| ---- | ------- | ---------------- | ---- | ---------------- |
| S0 | MAE-raw (baseline) | Single encoder + lightweight decoder; predict raw signal at masked positions | L1 + 0.3 × MR-STFT | ✓ |
| S1 | MAE-denoised | Same as S0 but the prediction target is an offline-cleaned version of the input (bandpass 0.5–40 Hz applied to target only, raw input to encoder) | L1 + 0.3 × MR-STFT, target = bandpass(x) | ✓ |
| S2 | VICReg siamese | Single encoder, two augmented views (temporal crop + amplitude scaling), VICReg loss on pooled outputs | invariance + variance + covariance | ✓ |
| S3 | TF-C with shared encoder | One shared encoder applied separately to (a) raw signal, (b) complex STFT, contrastive loss across the two views + cross-domain consistency | NT-Xent (time) + NT-Xent (freq) + L2 cross-domain | ✓ |
| S4 | EEGDM-style score matching | Single encoder; trained as the conditioning network of a small DiT denoiser; both jointly optimised; encoder retained for downstream | DDPM ε-prediction loss | ✓ |
| S5 | DeeperBrain-style MER + NSP | S1 (MAE-denoised) plus a small **prediction head** that, from the encoder representation of a 50%-masked input, reads off a 9–19D vector of macroscopic dynamical statistics of the *full* window | Smooth L1 on raw + Smooth L1 on NSP target (5-band power, CFC, sample entropy, optionally PLV) | ✓ |

S4 is technically two networks (encoder + DiT) but trained jointly in one
optimisation with no teacher / EMA, so it qualifies as scratch single-stage
SSL. The DiT is discarded after pretraining; only the encoder transfers.

S5 was added on the basis of [DeeperBrain (arXiv 2601.06134, Dec 2025)](https://arxiv.org/abs/2601.06134),
which introduces NSP as a complementary objective to masked reconstruction
and reports dramatic frozen-probing gains (Table IV) — e.g. FACED
9-class emotion at 50.96 % balanced accuracy under frozen probing
versus 25.84 % for CBraMod and 16.13 % for LaBraM. The detailed NSP
ablation, including a sweep over $\lambda_{\rm NSP}$ and whether to use
a linear vs MLP head, lives in
[exp16](../16_nsp_auxiliary_head/) — exp04 includes S5 only as a
single representative cell so the framework comparison sees the
DeeperBrain recipe at iso-effort. If S5 strict-wins exp04, exp16 is
the follow-on that finds the right NSP configuration; if S5 ties or
loses, exp16 still runs as a "is NSP additive on top of the chosen
framework" question.

## Controls

|                         | EEG signal | matched-noise twin |
| ----------------------- | ---------- | ------------------ |
| S0 MAE-raw              | ✓          | ✓                  |
| S1 MAE-denoised         | ✓          | ✓                  |
| S2 VICReg               | ✓          | ✓                  |
| S3 TF-C shared encoder  | ✓          | ✓                  |
| S4 EEGDM diffusion      | ✓          | ✓                  |
| S5 MER + NSP            | ✓          | ✓                  |

The matched-noise twin for VICReg specifically: replace the EEG with
Gaussian noise but keep the same augmentation pipeline. If VICReg learns
something on Gaussian noise (the augmentations alone are giving it
representation structure), the EEG result is partly hallucinated.

## Held constant

- Pretraining data: 100h HBN-EEG subset (per [`mini_experiments.md` §4.1](../../mini_experiments.md#41-pretraining-corpus)), 250 Hz uniform.
- Frontend: F0 vanilla strided conv (or exp02 winner).
- Backbone: bidirectional Mamba-2, 6 layers (or exp03 winner).
- Bottleneck: continuous (no quantization).
- Optimiser: AdamW, LR swept {1e-4, 3e-4, 1e-3} per framework.
- Compute budget per cell: 4 hours of H100 (matched FLOPs is harder for S4
  diffusion because each step does a noise sampling — S4 gets the same
  wall-clock budget instead of matched FLOPs).
- Eval suite: standard.

For the augmentation pipeline used by S2 VICReg and S3 TF-C, both use the
**same** augmentations to make the comparison clean: temporal crop with
±100 ms jitter, amplitude scaling × ~ U(0.7, 1.3), Gaussian noise added at
SNR ∈ [10, 20] dB, channel dropout (zero out 10 % of windows). These are
the augmentations the SelfEEG library implements for both methods.

## Decision rule

Same as exp02 / exp03:

- Strict win = ≥ 2 pp HBN 6-task BAC (per §4.3 Protocol A.2), non-overlapping CIs, noise-twin flat.
- Weak win = ≥ 1 pp with paired permutation p < 0.05.
- Tie = TOST equivalence within ε = 1 pp.
- Loss = ≥ 1 pp below baseline with p < 0.05.

Two framework-specific criteria:

- **Stability**: each framework must complete all 5 seeds without divergence.
  S2 VICReg specifically: variance term must keep encoder feature std
  > 0.5 throughout training (a known failure mode).
- **Source-dataset probe must trend down**: the auxiliary "predict source
  dataset from encoder features" probe accuracy at end-of-training must be
  *lower* than at start-of-training. This is the anti-shortcut sanity from
  [`methodology.md` §6.4](../../methodology.md#64-pretraining-objectives--what-works-and-what-doesnt).
  A framework that wins HBN 6-task BAC but increases source-probe accuracy
  (now "predict recording site" + "predict subject ID" per §4.3) is learning
  site/subject fingerprint, not neural content; it's disqualified.

## Pre-registered predictions

| Variant | Prediction | Reasoning |
| ------- | ---------- | --------- |
| S0 MAE-raw | The floor. Loss curve looks great, downstream is mediocre. | The classic raw-MAE failure on noisy signals — model reconstructs noise. |
| S1 MAE-denoised | Strict win over S0, ~+2–3 pp on HBN 6-task BAC | The single most direct attack on low SNR. EEG-X reports this; we replicate it without the EMA branch. |
| S2 VICReg | Weak win on subject-fingerprinting (subject-ID probe trends down faster than S0/S1) but tied or weak loss on HBN 6-task BAC | VICReg attacks subject confound but not the raw-noise reconstruction issue. |
| S3 TF-C | Best on phase-locking-value reconstruction; strict win on HBN 6-task BAC against S0; tied with S1 | The complex STFT branch carries phase explicitly and is naturally noise-robust. |
| S4 EEGDM diffusion | Strict win on HBN 6-task BAC, ~+3–4 pp; expensive but most principled for low SNR | Score matching at multiple noise levels structurally distinguishes signal from noise. EEGDM beats LaBraM/CBraMod with 19× fewer params per [`brain/eeg-research.md` §7.3](../../../../../brain/eeg-research.md#73-the-diffusion-wave). |
| S5 MER + NSP | Tied or weak win over S1 on end-to-end fine-tuning; **strict win on frozen-probing**, ~+5–10 pp on emotion / motor imagery (the DeeperBrain pattern) | NSP forces the encoder representation itself to encode dynamical order parameters, making linear-probe readout viable. |

The expected outcome: **S1 MAE-denoised wins on practical metrics under
fine-tuning, S4 diffusion wins on absolute peak metric but at much higher
per-step cost, S3 TF-C wins on phase-specific metrics, S5 wins
specifically under frozen probing**. The chosen framework depends on (a)
how compute-constrained the headline run is and (b) whether
frozen-probing performance is the goal — DeeperBrain argues that it
should be, because it is the truer test of "universal" representation
quality.

## Implementation pointers

- S0 MAE: standard implementation. Encoder processes only visible patches
  (asymmetric, like image MAE), decoder is a 2-layer Mamba block reading
  visible + mask tokens, predicts raw signal at masked positions.
- S1 denoised target: precompute the bandpass-filtered version of the
  entire pretraining corpus once, store as a side-stream. Encoder still
  sees raw input; loss is computed against the precomputed clean target.
  The bandpass cutoff (0.5–40 Hz) is the default — exp08 will explore
  alternative target signals.
- S2 VICReg: use the SelfEEG library's `BarlowTwins` / `VICReg` modules as
  reference; replace their default encoder with our chosen backbone. Use
  the variance term coefficient γ = 1.0 (collapse-prevention default).
- S3 TF-C with shared encoder: per the SSL-framework subagent's note, the
  original TF-C uses two separate encoders. We share weights between them
  (apply the *same* encoder to raw signal and to complex STFT, after a
  small input projection per modality). This makes it fit the
  "single-encoder" constraint.
- S4 EEGDM: encoder is the chosen backbone. Denoiser is a small DiT
  (Diffusion Transformer) at the same width, 6 layers. Train with the
  ε-prediction loss at noise levels σ ∈ {0.001, 0.01, 0.1, 1.0}. After
  pretraining, use the encoder's pooled output as the representation.
  Reference: [arXiv:2508.20705](https://arxiv.org/abs/2508.20705) for
  EEGDM, [arXiv:2508.14086](https://arxiv.org/abs/2508.14086) for the
  S4-based variant.
- S5 MER + NSP: S1 plus a small linear head reading 9D NSP target per
  patch (5 spectral-power bands, 3 CFC values, sample entropy) from
  the encoder representation. Loss is `0.5 * smooth_l1(x_pred, x_clean)
  + 0.5 * smooth_l1(nsp_head(H), Y_NS_target)`. Pre-compute Y_NS once
  per pretraining corpus alongside the cleaned target. Reference:
  [DeeperBrain (arXiv:2601.06134)](https://arxiv.org/abs/2601.06134)
  §III-G; the dedicated NSP-only ablation, including head-architecture
  and weight sweeps, is in [exp16](../16_nsp_auxiliary_head/).

## Output

`mini_experiments/04_ssl_framework_ablation/results.md` containing:

1. 5×2 (×3 seed) results table.
2. Source-dataset probe trajectory per framework (must trend down).
3. Encoder feature health per framework.
4. Phase-locking-value reconstruction quality per framework.
5. Wall-clock cost per training epoch per framework (because S4 will be
   meaningfully more expensive).
6. The chosen framework with decision-rule justification.
7. If no framework strict-wins, the chosen pair (S1 + auxiliary loss from
   another framework) for the next experiment.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| S2 VICReg collapses (variance → 0 or covariance → identity) | Logging with a hard-stop trigger; restart with γ_var = 5.0 if it happens. |
| S4 EEGDM diffusion is too slow to fit budget | Run with 100 noise steps instead of 1000 during pretraining; this is standard. |
| S1 denoised target wins because the eval is dominated by the same bandpass artifact | Re-evaluate S1 on Sleep-EDF too, where bandpass is well-justified anyway. |
| All frameworks tie because 100h is too small to differentiate | Scale to 300h of HBN-EEG (still well within the 3000 h available) and re-run the top 2 frameworks; this is cheap because preprocessing is cached in S3. |
| The chosen frontend (F0 default) handicaps spectral frameworks (S3) | If exp02 has chosen a spectral frontend before exp04 launches, swap it in. Note that this couples the experiments. |

## What gets carried forward

The single chosen SSL framework. Its decoder architecture (for S0 / S1) or
its DiT architecture (for S4) or its augmentation pipeline (for S2 / S3) is
frozen for every later experiment.

If S1 (denoised target) wins, exp08 (which target signal) becomes critical:
it picks the actual denoising recipe (bandpass vs ICA vs PCA vs wavelet vs
the EEG-X recipe). If S4 wins, exp08 is moot (diffusion does the
denoising structurally) and exp09 (multi-condition input) becomes more
interesting (does adding artificial noise to the diffusion input help?).
