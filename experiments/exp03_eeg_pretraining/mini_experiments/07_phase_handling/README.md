# exp03 / mini-experiment 07 — Phase handling

> **Status:** planned
>
> **Cross-reference:** [`brain/experiments/tokenization/04-frequency-domain.md` §5](../../../../../brain/experiments/tokenization/04-frequency-domain.md#5-phase-the-third-dimension),
> [`methodology.md` §6.1](../../methodology.md#61-encoder-feature-health)
>
> **Compute budget:** 18 H100-hours (4 phase variants × 2 control columns × 3
> seeds, with a phase-locking-value evaluation that adds ~10 min per cell)
>
> **Gates:** none downstream (output is a configuration choice for the
> headline run)

## Question

Does explicitly modelling phase in the SSL loss — via complex STFT
prediction, sin/cos circular phase loss, bispectral consistency, or any
combination — produce representations that better preserve cross-frequency
phase coupling than magnitude-only spectral losses, and does that
improvement translate into better downstream task performance?

## Why it matters

Phase is the part of EEG that magnitude-only losses systematically discard.
The cognitive significance of phase is well-documented in neuroscience:
phase-amplitude coupling between θ and γ bands tracks working memory
encoding, ERP phase-locking is the basis of P300 / N400 detection, and
cross-frequency phase-amplitude coupling is the canonical signature of
attentional and executive control. A pretraining loss that does not
preserve phase forces every downstream task that depends on phase to either
re-learn it (lossy, slow) or to rely on amplitude proxies (lossy, biased).

The trouble with phase is that it's a circular variable on `[0, 2π)`, and
standard MSE is undefined on a circle. Three published solutions:

- **Complex STFT prediction**: predict both the real and imaginary parts of
  the STFT separately, with L1 / L2 on each. Phase is captured implicitly
  via the relationship between Re and Im. Used by Schrödinger Bridge SE
  ([arXiv:2407.16074](https://arxiv.org/abs/2407.16074)) for speech.
- **Sin/cos circular loss**: predict `sin(φ)` and `cos(φ)` separately
  (each lives in `[-1, 1]`); MSE is well-defined on each. Used by
  NeuroRVQ ([arXiv:2510.13068](https://arxiv.org/abs/2510.13068)) — the
  most explicit phase-aware EEG loss in the published literature.
- **Bispectral consistency**: bicoherence captures phase coupling between
  frequency triples `(f₁, f₂, f₁+f₂)`. Voytek et al. NeuroImage 2018 prove
  that *all* standard PAC measures are bicoherence estimators. Bispectral
  consistency between predicted and target signal forces preservation of
  cross-frequency phase relationships. **No EEG SSL paper uses bispectrum
  matching as the primary cross-frequency-coupling loss**, though
  DeeperBrain ([arXiv:2601.06134](https://arxiv.org/abs/2601.06134))
  introduces a related but structurally different "predict the
  cross-frequency coupling value from the encoder representation"
  objective — see the framing note below.

The variants differ in implementation cost and in *what kind* of phase they
capture: complex STFT and sin/cos capture local instantaneous phase;
bispectral captures longer-range cross-frequency phase coupling. Both
matter; the question is whether either (or both) survives an honest
ablation.

### A framing note on "match the statistic" vs "predict the statistic"

The bispectral consistency variant P3 below computes the bicoherence on
*both* the predicted reconstruction and the target signal and minimises
$L_2$ between the two — i.e. the network is asked to make its decoded
waveform have the same cross-frequency-coupling structure as the target.
DeeperBrain instead attaches a small linear head to the encoder
representation that *predicts* the cross-frequency coupling value
directly, with no decoder involved
([arXiv:2601.06134](https://arxiv.org/abs/2601.06134) §III-G):

```
match the statistic (our P3):     match[ CFC(decode(encode(x))), CFC(x) ]
predict the statistic (DeeperBrain): match[ linear(encode(x)), CFC(x) ]
```

The two are not equivalent. The *predict* framing forces the encoder
representation itself to *encode* the statistic, which is a much
stronger constraint than asking the decoder pathway to *exhibit* it.
This is the central mechanism behind DeeperBrain's frozen-probing
advantage. The dedicated experiment for this framing — with all four
DeeperBrain statistics (relative spectral power, PLV, CFC, sample
entropy), not just CFC — is [exp16](../16_nsp_auxiliary_head/). exp07
keeps the *match* framing because it is what most spectral-loss
literature in audio uses (Yamamoto MR-STFT, NeuroRVQ); we are testing
those specific designs here. If exp16's NSP variant strict-wins exp16,
the headline run *also* runs the predict-the-statistic head; the two
are complementary, not competing.

## Variants

The variants are *additions* to the chosen reconstruction loss from exp06
(default L1 on raw + 0.3 × MR-STFT log-magnitude). They are not
replacements.

| Code | Variant | Phase term added | Computational cost over baseline |
| ---- | ------- | ---------------- | --------------------------------- |
| P0 | None (magnitude-only baseline) | — | reference |
| P1 | Complex STFT prediction | `+ 0.3 × L1(Re(STFT(x_pred)) - Re(STFT(x))) + 0.3 × L1(Im(STFT(x_pred)) - Im(STFT(x)))` over the same windows as MR-STFT | minimal (both Re and Im are byproducts of the existing FFT) |
| P2 | Sin/cos circular phase | `+ 0.2 × L1(sin(φ_pred) - sin(φ)) + 0.2 × L1(cos(φ_pred) - cos(φ))` where `φ = angle(STFT(x))`, restricted to bins with sufficient magnitude (top 50 % by power) | small (one extra `angle` computation) |
| P3 | Bispectral consistency on `[θ × γ]` quadrant only | `+ 0.1 × L2(b²_pred(f₁, f₂) - b²(f₁, f₂))` with `(f₁, f₂) ∈ [4, 12] Hz × [30, 80] Hz`; b² computed via Welch-style block averaging within batch | moderate (FFT pairwise is more expensive — only run on a 64-batch subset per step) |
| P4 | All three (P1 + P2 + P3) | sum of the above, weights 0.2 + 0.1 + 0.05 | ≈ P3 cost |

## Controls

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| P0 (magnitude only)            | ✓          | ✓                  |
| P1 complex STFT                | ✓          | ✓                  |
| P2 sin/cos circular            | ✓          | ✓                  |
| P3 bispectral on [θ × γ]       | ✓          | ✓                  |
| P4 all combined                | ✓          | ✓                  |

The matched-noise twin specifically catches a phase-loss failure mode:
Gaussian noise has uniformly random phase, so a model trained with phase
loss on noise will *not* learn anything (the loss is high entropy in the
phase variable). If the noise-twin metrics improve under phase loss, the
loss is leaking some non-phase signal — likely a magnitude correlation
between Re and Im that the model exploits.

## Held constant

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: exp04 winner (likely S1 MAE-denoised).
- Bottleneck: continuous.
- Mask: 50 % random.
- Reconstruction loss: exp06 winner (default L1 + MR-STFT).
- Optimiser: AdamW, LR fixed (carried forward from exp06's winning LR).
- Pretraining duration: 8 epochs.

The eval suite gets one critical addition for this experiment: a
**phase-locking-value (PLV) reconstruction test**. On a held-out 1000-sample
batch, compute PLV between the predicted signal and target signal at each
EEG band (δ, θ, α, β, γ). PLV close to 1.0 means phase is faithfully
reconstructed; PLV near 0.0 means phase is lost. This is the direct
measurement of what each phase variant is supposed to be improving.

## Decision rule

Two parallel decision rules, because the experiment has two outputs:

**Output A — chosen phase loss** (the standard win/loss as in prior
experiments, on HBN 6-task BAC per §4.3 Protocol A.2). Strict win = ≥ 1 pp
HBN 6-task BAC, non-overlapping CIs, noise-twin flat.

**Output B — phase fidelity** (whether the chosen variant actually
preserves phase, separate from downstream metrics). Strict pass: PLV at θ
and α bands ≥ 0.7 (the threshold is loose because EEG PLV is hard); strict
fail: PLV < 0.4. A variant can win on HBN 6-task BAC but fail PLV — in
that case it's recorded as "good representation but not phase-aware".

The dual rule lets us distinguish "phase loss helps downstream because
phase is inherently useful" from "phase loss helps downstream as a
regulariser, but the model is still phase-blind". The first is what we
want; the second is interesting but means the headline run still doesn't
have phase-coded representations.

## Pre-registered predictions

| Variant | HBN 6-task BAC | PLV (θ, α) |
| ------- | -------------- | ---------- |
| P0 magnitude only | baseline | low (0.2–0.4) — magnitude doesn't constrain phase |
| P1 complex STFT | weak win, ~+0.5 pp | moderate (0.5–0.7) — phase reconstructed implicitly |
| P2 sin/cos | weak win, ~+0.5–1 pp | high (0.7–0.85) — direct supervision |
| P3 bispectral | weak win, ~+1 pp on cross-freq tasks; tied on HBN 6-task | unrelated to PLV; bispectral measures cross-freq coupling |
| P4 combined | strict win, ~+1.5–2 pp | high (0.7–0.85) |

The honest expected outcome: **P2 sin/cos wins on PLV, P3 bispectral wins
on cross-frequency coupling tasks (which our standard eval suite does NOT
include), P4 combined gives marginal additional improvement at the cost of
implementation complexity**. The decision is whether phase coding is a
goal in itself (P2 mandatory) or just a means to better downstream metrics
(P0 baseline plus exp06's reconstruction loss might suffice).

## Implementation pointers

- Complex STFT: `torch.stft(x, n_fft, hop_length, return_complex=True)`;
  `Re = X.real`, `Im = X.imag`. Apply L1 to each.
- sin/cos circular: `phase = torch.angle(X)`; `loss_sin = (sin(φ_pred) - sin(φ)).abs().mean()`;
  `loss_cos = ...`. Mask out bins where magnitude is small to avoid
  fitting noise: `mask = X.abs() > 0.1 * X.abs().max()`.
- Bispectral: implement via FFT-based outer product:
  ```python
  X = torch.fft.rfft(x)
  # b(f1, f2) = E[X(f1) X(f2) conj(X(f1+f2))]
  # restrict to f1 ∈ [4, 12] Hz, f2 ∈ [30, 80] Hz
  b = torch.einsum('bi,bj->bij', X[..., f1_idx], X[..., f2_idx]) * \
      torch.conj(X[..., f1_idx + f2_idx_offset])
  bicoh = (b.abs() ** 2).mean(0) / (denominator + eps)
  ```
  Use Welch-style block averaging within a batch (split batch into 4
  sub-batches, compute bicoherence on each, average). Cost: roughly the
  same as one extra MR-STFT.

## Output

`mini_experiments/07_phase_handling/results.md` containing:

1. 5 × 2 (× 3 seed) results table.
2. PLV per band per variant (on held-out segments).
3. Bicoherence visualisation per variant — heatmap of `b²(f₁, f₂)` on the
   `[θ × γ]` quadrant, showing whether the predicted bicoherence matches
   the target.
4. Wall-clock cost per variant.
5. The chosen phase variant(s) with decision-rule justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| Phase loss makes the optimisation unstable (the loss surface is multimodal because of the angle wrapping) | Use sin/cos parameterisation (P2) which avoids wrapping. Avoid raw `angle` MSE. |
| Bispectral consistency is too noisy as a per-batch loss | Accumulate over 4 batches before computing the loss; this is roughly equivalent to Welch averaging. |
| Phase loss helps PLV but not HBN 6-task BAC — the eval doesn't reward phase | Add a phase-sensitive eval task: e.g. ERP latency estimation accuracy on a held-out evoked-response dataset (HBN's contrast-change-detection task is naturally evoked, so it's a candidate). |
| Restricting bispectral to [θ × γ] misses other meaningful coupling | First run with the full 0–80 × 0–80 quadrant; if it works restrict to physiologically motivated subregions. |

## What gets carried forward

The single chosen phase variant (or P0 if none strict-wins). The variant's
loss-weight contribution is frozen for every later experiment. The PLV
result is reported in the headline-run plan as a calibrated expectation.
