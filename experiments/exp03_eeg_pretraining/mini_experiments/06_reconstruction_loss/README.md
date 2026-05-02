# exp03 / mini-experiment 06 — Reconstruction loss

> **Status:** planned
>
> **Cross-reference:** [`brain/experiments/tokenization/04-frequency-domain.md` §7](../../../../../brain/experiments/tokenization/04-frequency-domain.md#7-the-reconstruction-loss-design),
> [`methodology.md` §6.4](../../methodology.md#64-pretraining-objectives--what-works-and-what-doesnt)
>
> **Compute budget:** 12 H100-hours (5 loss variants × 2 control columns × 3
> seeds = 30 cells × 24 min average — pretraining is identical across
> variants, only the loss differs)
>
> **Gates:** experiment 07 (depends on the chosen reconstruction loss)

## Question

Within the chosen SSL framework (default: MAE-style with reconstruction at
masked positions), which time-domain reconstruction loss minimises the
contribution of high-amplitude EEG artifacts to the gradient while
preserving fidelity to the underlying neural signal: L2 (MSE), L1, Huber,
Barron with learned α, or Itakura-Saito on the periodogram?

## Why it matters

EEG amplitudes are heavy-tailed. A single eye blink (~100 µV) dominates
the L2 loss against a P300 (~5 µV) by a factor of 400 even though the P300
is the cognitively relevant signal. The choice of loss function is the
single cheapest, most direct attack on this pathology, and the literature
gives strong evidence that the standard L2 is the wrong default:

- **Huber** ([Huber 1964](https://projecteuclid.org/euclid.aoms/1177703732))
  bounds the gradient contribution of large residuals to a constant. Used
  by BioCodec ([arXiv:2510.09095](https://arxiv.org/abs/2510.09095)) and
  by DeeperBrain ([arXiv:2601.06134](https://arxiv.org/abs/2601.06134),
  Smooth L1 with $\beta = 1.0$ as the masked-reconstruction loss) on
  biosignals specifically. Drop-in replacement, single hyperparameter
  δ. Two independent biosignal foundation-model results converging on
  Huber over L2 is the strongest external evidence we have that this
  variant is at least competitive.
- **Barron's adaptive robust loss**
  ([CVPR 2019, arXiv:1701.03077](https://arxiv.org/abs/1701.03077))
  generalises L2, L1, Huber, Cauchy, Tukey under one parameterisation
  with a *learnable* α per channel. The model discovers per-channel which
  channels need heavy-tail robustness and which do not. Strongly principled
  for EEG where artifact distribution differs across channels and subjects.
- **Itakura-Saito** divergence
  ([Févotte et al., Neural Computation 2009](https://hal.archives-ouvertes.fr/hal-00257494))
  on the periodogram is *scale-invariant* (`d_IS(λx, λy) = d_IS(x, y)`).
  For Gaussian processes it equals the spectral relative-entropy rate — it
  is *exactly* the information-theoretic distance between two stationary
  Gaussian processes. No EEG SSL paper uses it. Five lines of code.
- **Multi-resolution STFT log-magnitude** is the speech-codec orthodoxy
  (BigVGAN, EnCodec, DAC); the log term gives equal weight to small and
  large frequency components.

The honest expected ordering is "L2 ≪ L1 < Huber < Barron < IS-on-spectrum",
but no public ablation exists for EEG SSL. exp06 establishes it.

## Variants

The variants apply only to the *time-domain reconstruction* term. Every
cell additionally includes the same MR-STFT log-magnitude term as an
auxiliary anchor (weight 0.3) to keep the comparison about the
time-domain loss only.

| Code | Variant | Time-domain loss | Spectral aux | Special notes |
| ---- | ------- | ---------------- | ------------ | ------------- |
| L0 | L2 (MSE) on raw signal | `(x - x̂)²` | MR-STFT 0.3× | Default; the failure baseline |
| L1 | L1 on raw signal | `|x - x̂|` | MR-STFT 0.3× | Cheaper, slightly more robust than L2 |
| L2 | Huber on raw signal | piecewise quadratic / linear at δ=1.0 (post-z-score) | MR-STFT 0.3× | BioCodec recipe |
| L3 | Barron with learned α per channel | Barron's general parameterisation | MR-STFT 0.3× | α and c learnable; expected ~α=0–1 (Cauchy-like) for noisy channels |
| L4 | Itakura-Saito on periodogram + L1 on raw (50/50) | `d_IS(|FFT|², |FFT̂|²)` plus L1 anchor | MR-STFT 0.3× | The novel cell — IS divergence has never been used as primary EEG SSL loss |

The variants are deliberately not exhaustive. We omit:

- Tukey biweight (vanishing gradients on outliers; documented to fail in
  gradient-based training per the SSL-framework subagent).
- Cauchy-only (subsumed by Barron at α=0).
- log-cosh (subsumed by Huber, no theoretical advantage).
- Pure MR-STFT-only (no time-domain term — fails at high frequencies
  where STFT bins are noise-dominated).
- VICReg / MEC as reconstruction losses (these are siamese / contrastive,
  not reconstruction; the framework experiment exp04 already tested them).

## Controls

|                              | EEG signal | matched-noise twin |
| ---------------------------- | ---------- | ------------------ |
| L0 L2 (MSE)                  | ✓          | ✓                  |
| L1 L1                        | ✓          | ✓                  |
| L2 Huber                     | ✓          | ✓                  |
| L3 Barron learned α          | ✓          | ✓                  |
| L4 IS-on-periodogram + L1    | ✓          | ✓                  |

Noise-twin sanity: a robust loss should *not* improve performance on
matched-noise input — there is no signal there to extract. If a loss
variant improves both EEG and noise metrics, it is just adding model
capacity in some incidental way (e.g. better gradient scaling).

## Held constant

- Frontend: exp02 winner (or F0 default).
- Backbone: exp03 winner (default Mamba-2).
- SSL framework: exp04 winner (default S1 MAE-denoised — note that
  the *target* is the bandpass-filtered signal, not raw, even in this
  reconstruction-loss experiment, because exp04 will likely have settled
  S1).
- Bottleneck: continuous.
- Mask: 50 % random.
- Optimiser: AdamW, LR swept {1e-4, 3e-4, 1e-3} per loss variant. The LR
  for IS divergence may be substantially different (the loss has different
  scale) — handled by the per-variant sweep.
- Pretraining duration: 8 epochs.

## Decision rule

Same as exp02 / exp03 / exp04:

- Strict win = ≥ 2 pp TUEV BAC, non-overlapping CIs, noise-twin flat.
- Weak win = ≥ 1 pp with paired permutation p < 0.05.
- Tie = TOST equivalence within ε = 1 pp.

Two loss-specific criteria:

- **Gradient stability**: monitor the per-batch gradient norm. If a variant
  produces gradient norms with std > 10 × the L2 baseline's std at any
  point during training, it is unstable and disqualified — even if its
  end-of-training metrics look good. (Robust losses can fail this if their
  hyperparameter is wrong: e.g. Barron with α stuck at -∞ effectively
  zeros gradients.)
- **Per-channel α inspection (Barron only)**: log the learned α per channel
  every 5 % of training. The expected pattern: α ≈ 2 (L2-like) for clean
  central channels, α ≈ 0 (Cauchy-like) for frontal channels prone to
  EOG. If α collapses to a single value across all channels, the
  per-channel parameterisation is not paying off.

## Pre-registered predictions

| Variant | Prediction on TUEV BAC | Prediction on noise-twin |
| ------- | ---------------------- | ------------------------ |
| L0 L2 | floor (~+0.0) | flat |
| L1 L1 | weak win, ~+0.5 pp | flat |
| L2 Huber | weak/strict win, ~+1–2 pp | flat |
| L3 Barron | strict win, ~+2–3 pp | flat |
| L4 IS+L1 | strict win, ~+2–4 pp; potentially the best | flat |

If L3 and L4 both strict-win, the carried-forward loss is L3 + L4 hybrid:
Barron on raw + IS divergence on periodogram + MR-STFT auxiliary. (We
deliberately allow one post-hoc combined cell, like in exp02.) This three-
term loss attacks both the time-domain heavy-tail problem and the
amplitude-scale invariance problem simultaneously.

## Implementation pointers

- L1, Huber: native PyTorch (`F.l1_loss`, `F.smooth_l1_loss`).
- Barron: reference implementation
  [`jonbarron/robust_loss_pytorch`](https://github.com/jonbarron/robust_loss_pytorch).
  Use the AdaptiveLossFunction with `num_dims=C` (per channel α and c).
  After z-score, the data scale is ~1; initialise scale = 1.0.
- Itakura-Saito: trivial 5-line implementation:
  ```python
  def itakura_saito(s_pred, s_true, eps=1e-8):
      r = (s_pred + eps) / (s_true + eps)
      return (r - torch.log(r) - 1).mean()
  ```
  Apply `F.softplus` to the network's frequency-domain output to enforce
  positivity; compute `s_pred = |torch.fft.rfft(x_pred)|² + eps`.
- MR-STFT: reference [BioCodec recipe](https://arxiv.org/abs/2510.09095) §2.2 — windows
  `[64, 128, 256, 512, 1024]` for 4-second windows at 250 Hz, hop = window/4,
  log-magnitude L1.

## Output

`mini_experiments/06_reconstruction_loss/results.md` containing:

1. 5 × 2 (× 3 seed) results table.
2. Per-variant gradient-norm trajectory plot.
3. For Barron: per-channel α distribution plot at end of training.
4. For IS: spectral fidelity (PSNR on the held-out spectrum reconstruction)
   in addition to TUEV BAC.
5. Encoder feature health per variant.
6. The chosen reconstruction loss with decision-rule justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| Barron's learnable α is unstable for the first few epochs (high-variance gradients) | Warmup: freeze α at 1.0 (Huber-like) for the first 1 epoch, then unfreeze. |
| IS divergence requires positive periodogram values; numerical issues at noisy frequencies | Add eps = 1e-8 inside both numerator and denominator; use double precision for the FFT. |
| The MR-STFT auxiliary anchor dominates the time-domain loss and washes out the variant-level differences | Reduce the auxiliary weight from 0.3 → 0.1 if the variants tie; document the change. |
| All variants tie because the MAE-denoised framework already mitigates the heavy-tail problem (the target is bandpass-filtered = no extreme outliers) | Re-run with raw target (S0) instead of denoised (S1), and report both. The S0 result tells us the true loss-function effect. |

## What gets carried forward

The single chosen reconstruction loss (or hybrid). The hyperparameters
(δ for Huber, α-init for Barron, eps for IS) are frozen for every later
experiment.
