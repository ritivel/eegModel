# exp03 / mini-experiment 02 — Frontend ablation

> **Status:** planned
>
> **Cross-reference:** [`methodology.md` §2 Phase 2](../../methodology.md#phase-2--ablation-engine-1-3-days-1-8-gpus),
> [`methodology.md` §3 matrix shape](../../methodology.md#3-how-to-design-one-ablation-the-matrix-shape),
> [`brain/experiments/tokenization/04-frequency-domain.md`](../../../../../brain/experiments/tokenization/04-frequency-domain.md)
>
> **Compute budget:** 18 H100-hours (5 frontend variants × 2 control columns ×
> 3 seeds = 30 cells × 35 min average)
>
> **Gates:** experiments 05, 06, 07, 08, 09, 10, 11, 12

## Question

Which front-end maps the raw single-channel EEG signal to the encoder's
token sequence in a way that (a) preserves phase, (b) handles arbitrary
sampling rates without information loss, (c) is provably stable to noise,
and (d) does not waste capacity learning what classical signal processing
already knows?

## Why it matters

The frontend is the first layer the gradient signal sees. A bad one sets a
ceiling on every later layer's representational quality, because no
downstream layer can recover information that the frontend has already
discarded. Three EEG-specific failure modes make the choice load-bearing:

1. **Phase**. A standard strided-conv stack with GeLU activations is not
   analytic — it discards instantaneous phase. EEG carries cognitive
   information in cross-frequency phase coupling (theta-phase ↔
   gamma-amplitude), so a phase-blind frontend forces the rest of the
   network to either rediscover phase from amplitude (lossy) or to give up
   on phase-coded content entirely.
2. **Aliasing**. Strided-conv at stride 2 with no anti-aliasing folds
   everything above Nyquist of the output rate back into the passband
   (Zhang ICML 2019, [BlurPool](https://arxiv.org/abs/1904.11486)).
   For multi-rate EEG this is catastrophic: 200 Hz inputs alias into the
   exact same feature space as 2000 Hz inputs.
3. **Inductive bias**. Classical signal processing has 80 years of
   filter-design theory specifically about narrow-band signals. SincNet
   reduces an entire bandpass filter to two scalar parameters per filter;
   wavelet scattering proves Lipschitz-stability to deformations
   ([Mallat 2012](https://doi.org/10.1002/cpa.21413)). A learned conv stack
   can in principle discover all of this from data, but in low-data low-SNR
   EEG it usually doesn't, and even when it does it is less interpretable
   and less robust to distribution shift.

## Variants

Each variant is a function `R^T → R^{T' × D}` where `T` is the raw
sample count, `T'` is the token count, and `D = 256` is fixed (the
backbone's expected input width).

| Code | Variant | Parameters | Phase preserving | Multi-rate native | Noise-robust theorem |
| ---- | ------- | ---------- | ---------------- | ----------------- | -------------------- |
| F0 | Vanilla strided conv (3-layer, kernels (7,7,7), strides (2,2,2), GeLU, no BlurPool) | ~50k | ✗ | ✗ | none |
| F1 | F0 + Snake activations + BlurPool before every stride | ~50k | partial | ✗ | shift-invariant via BlurPool |
| F2 | SincNet (80 filters, Hz-parameterised cutoffs) → linear projection to D | ~160 + small projection | partial (real sinc) | ✓ (cutoffs in Hz) | 1-Lipschitz (bandpass is bounded) |
| F3 | Frozen Kymatio Scattering1D (J=6, Q=8) → linear projection to D | 0 + small projection | partial (modulus) | ✓ (wavelets are scale-natural) | provably 1-Lipschitz to deformations |
| F4 | Complex Gabor filterbank (40 filters, log-spaced 0.5–80 Hz) → real+imag concat → linear projection | ~80 + small projection | ✓ (complex-valued) | ✓ (Hz-spaced) | bandpass + small spectrum |

All variants are followed by an identical positional embedding and
identical Mamba-2 backbone (the default per
[`mini_experiments.md` §4.2](../../mini_experiments.md#42-default-architecture-for-any-axis-not-under-test)).

## Controls (the §3 matrix)

|                  | EEG signal | matched-noise twin |
| ---------------- | ---------- | ------------------ |
| F0 (baseline)    | ✓          | ✓                  |
| F1               | ✓          | ✓                  |
| F2               | ✓          | ✓                  |
| F3               | ✓          | ✓                  |
| F4               | ✓          | ✓                  |

Each cell trained for the same number of optimizer steps with the same
budget. The matched-noise twin is essential: a frontend that improves the
EEG metrics but *also* improves the noise metrics is just adding model
capacity, not extracting EEG-specific structure. Real wins must improve EEG
metrics *and* leave noise metrics flat.

## Held constant

- Pretraining data: the 100h TUEG subset per
  [`mini_experiments.md` §4.1](../../mini_experiments.md#41-pretraining-corpus).
- Sampling rate: uniform 250 Hz (multi-rate is exp05's question).
- Backbone: bidirectional Mamba-2, 6 layers, d=256, state N=64.
- Bottleneck: continuous (no quantization).
- SSL framework: vanilla MAE, 50 % random masking, asymmetric encoder /
  decoder.
- Reconstruction loss: L1 on raw signal + 0.3 × MR-STFT log-magnitude.
- No phase loss (exp07 will add it after the frontend is chosen).
- Optimiser: AdamW, lr swept over {1e-4, 3e-4, 1e-3} per variant, best LR
  per variant carried forward to the headline cell. Per
  [`methodology.md` §4](../../methodology.md#4-hyperparameter-transfer-when-small-scale-tuning-is-trustworthy)
  the LR is the only HP that does not transfer between data recipes; we
  re-tune it per frontend.
- Training duration: 8 epochs over the 100h corpus = ≈ 35M training tokens.
- Eval suite: TUAB AUROC + TUEV balanced accuracy + weighted F1 + k-NN
  top-1, plus the §4.3 label-free monitors.

## Decision rule

For each variant V relative to the F0 baseline:

- **Strict win**: V's mean TUEV BAC exceeds F0's mean by ≥ 2 percentage
  points with non-overlapping 95 % bootstrap CIs *and* the matched-noise
  twin shows no improvement (paired sign-flip permutation test
  p > 0.10 on the noise side).
- **Weak win**: ≥ 1 pp improvement on TUEV BAC with overlapping CIs but
  paired permutation test p < 0.05.
- **Tie**: TOST equivalence within ε = 1 pp on TUEV BAC.
- **Loss**: V's mean TUEV BAC is below F0's by ≥ 1 pp with p < 0.05.

Phase-preservation requirement (a hard constraint per the conversation):
even if V wins on TUEV BAC, if a follow-on phase-sensitive eval (a
phase-locking-value reconstruction test on a held-out segment) shows V is
worse than F0 by > 10 %, V is disqualified from carrying forward unless we
add a phase loss in exp07 that recovers it.

## Pre-registered predictions

| Variant | Prediction | Reasoning |
| ------- | ---------- | --------- |
| F0 vanilla | The floor. TUAB ~75 %, TUEV BAC ~40 %. | LaBraM-Base level on equivalent compute. |
| F1 (+ Snake + BlurPool) | Weak win on both, ~+1 pp on TUEV BAC | Snake adds a periodic prior matching EEG oscillations; BlurPool prevents aliasing. Both should be free wins. |
| F2 (SincNet) | Strict win on TUEV BAC, ~+2–3 pp; tied on TUAB | Bandpass priors should help on the multi-class TUEV more than the binary TUAB. |
| F3 (frozen scattering) | Strict win on noise robustness (the matched-noise twin shows no improvement), ~+1–2 pp on TUEV BAC | Lipschitz stability bounds the gradient contribution of artifacts. |
| F4 (complex Gabor) | Best on the phase-locking-value eval; competitive but not top on TUEV BAC | Phase preservation comes at the cost of doubling the channel count (real+imag). |

If F2 and F3 both strict-win independently, exp02 carries forward
**F2 + F3 hybrid** (SincNet → frozen scattering on top → linear projection)
and re-evaluates as a sixth cell. This is the only post-hoc cell allowed.

## Implementation pointers

- SincNet: [`speechbrain/SincNet`](https://github.com/speechbrain/SincNet),
  port the `SincConv1d` layer, replace the speech-rate cutoffs with EEG-rate
  cutoffs (0.5–80 Hz), expose the cutoffs in Hz.
- Kymatio scattering: [`kymatio.torch.Scattering1D(J=6, Q=8, T=window_len)`](https://www.kymat.io).
  GPU-native, fully differentiable. Set `oversampling=0` for cheapest
  inference.
- Snake activations:
  [`alibaba/snake`](https://github.com/EdwardDixon/snake) reference,
  per-channel learnable α.
- BlurPool: [`adobe/antialiased-cnns`](https://github.com/adobe/antialiased-cnns)
  has a 1-D Binomial-5 kernel; trivial to port.
- Complex Gabor: implement directly — `g(t) = exp(i ω₀ t) · exp(-t²/2σ²)`
  with log-spaced ω₀ and matched σ for constant-Q. Take real + imaginary as
  separate channels.
- Mamba-2 backbone: [`state-spaces/mamba`](https://github.com/state-spaces/mamba),
  use the official Triton kernel; bidirectional via two parallel Mamba
  blocks summed (FEMBA-style).

## Output

`mini_experiments/02_frontend_ablation/results.md` containing:

1. The 5×2 (×3 seed) results table with all metrics and CIs.
2. Phase-locking-value reconstruction quality per variant.
3. Encoder feature health per variant (std stable, rank > 0.5, source-probe
   trends down).
4. The chosen frontend with explicit decision-rule justification.
5. Failure analysis if no variant strict-wins (in which case the default F0
   carries forward to exp03 / exp04 and we revisit after).

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| All variants tie because 100h is too little data to differentiate | Add a 50h Sleep-EDF subset and re-run. Sleep stages have strong frequency-band signatures; should differentiate spectral frontends. |
| F3 Kymatio scattering output dimension is too high (~384) and dominates | Project to D=256 explicitly; the projection is part of the variant. |
| F2 SincNet cutoffs collapse to identical values | Add the standard SincNet repulsive regulariser on cutoffs to keep them spread. |
| Random-init linear probe (from exp01) is unexpectedly high — variants barely move it | This is the floor problem flagged in [`methodology.md` §1](../../methodology.md#1-the-mental-model-pretraining-is-capability-engineering-not-luck). Switch to a harder downstream task (TUEV instead of TUAB-binary, or BCIC-IV-2a). |

## What gets carried forward

The single chosen frontend variant. Its hyperparameters (LR, optionally
filter cutoffs / number of filters / scattering J,Q) are frozen for every
later experiment. If F2+F3 hybrid wins, both layers carry forward.
