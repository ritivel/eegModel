# exp03 / mini-experiment 09 — Multi-condition input (WavLM-style)

> **Status:** planned
>
> **Cross-reference:** [`brain/asr-research.md` §6.2](../../../../../brain/asr-research.md#62-the-fixes--joint-training-dual-stream-fusion-se-aware-asr),
> WavLM ([Chen et al. JSTSP 2022](https://ieeexplore.ieee.org/document/9814838))
>
> **Compute budget:** 12 H100-hours (4 noise-injection variants × 2 control
> columns × 3 seeds = 24 cells × 30 min average)
>
> **Gates:** none downstream

## Question

Does WavLM-style multi-condition pretraining — at training time, with some
probability, mix synthetic EOG / EMG / line-noise into the encoder's input
while the reconstruction target stays clean — produce a more noise-robust
EEG encoder than training on the original signal alone?

## Why it matters

WavLM ([Chen et al. JSTSP 2022](https://ieeexplore.ieee.org/document/9814838))
took the HuBERT recipe and added one trick: at training time, with prob 0.5,
mix in additive noise from MUSAN / FreeSound at SNR ∈ [5, 20] dB, but ask
the model to predict the *clean* HuBERT cluster targets. The result was a
single encoder that was both better on clean speech *and* significantly
more noise-robust — 23 % relative N-WER improvement over HuBERT under
noise. Crucially, this was a strictly single-stage training change — no
extra teacher, no separate denoising network.

For EEG, the analogous setup is even cleaner because we have ground truth
about the target signal (we choose what to clean it to in exp08). The
question: does this trick transfer? It has never been tried on EEG.

The mechanism the trick relies on: the encoder cannot memorise the noise
because the noise is independently sampled per training step. The only
stable strategy is to learn features that are invariant to the additive
noise distribution, which by definition makes the encoder noise-robust.
This is a structural argument that should apply identically to EEG, but
the empirical question is whether the gain is real and how it interacts
with the other noise-attacking machinery (the denoised target from exp08,
the AAMP masking from exp10, the robust loss from exp06).

## Variants

The variants differ in noise type and injection probability. All use the
exp08 winner as the target signal (denoised) and the exp04 winner as the
SSL framework.

| Code | Variant | Noise type | Injection probability | SNR range (dB) |
| ---- | ------- | ---------- | --------------------- | -------------- |
| N0 | No noise injection (baseline) | — | 0.0 | n/a |
| N1 | Synthetic EOG only | bandpass 0.5–4 Hz Gaussian, amplitude scaled to mimic eye blinks | 0.3 | [5, 15] |
| N2 | Synthetic EMG only | bandpass 30–100 Hz Gaussian, amplitude scaled to mimic muscle bursts | 0.3 | [5, 15] |
| N3 | Mixed (EOG + EMG + 50/60 Hz line + Gaussian) | concatenation of the above plus broadband Gaussian | 0.3 | [0, 15] |
| N4 | N3 with higher mixing rate | same as N3 | 0.5 (matches WavLM) | [0, 10] (more aggressive) |

## Controls

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| N0 baseline (no injection)     | ✓          | ✓                  |
| N1 EOG injection               | ✓          | ✓                  |
| N2 EMG injection               | ✓          | ✓                  |
| N3 mixed @ 0.3                 | ✓          | ✓                  |
| N4 mixed @ 0.5 (WavLM)         | ✓          | ✓                  |

The matched-noise twin here has a special interpretation: the input is
already pure Gaussian noise. Adding more noise to noise should change
nothing structurally; the noise-twin metric should be unchanged across
variants. If a variant improves the noise-twin metric, it's exploiting
some artefact of the augmentation pipeline (e.g. statistical structure in
the EOG generator) — disqualified.

## Held constant

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: exp04 winner (likely S1 MAE-denoised).
- Bottleneck: continuous.
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Mask: 50 % random.
- Target: exp08 winner (denoised target — this is the critical pairing
  with multi-condition input; see Risks).
- Optimiser: AdamW, LR carried forward.
- Pretraining duration: 8 epochs.

The key training-time hyperparameter is the noise mixing routine. For each
batch, with probability `p_inject`, sample noise from the variant's noise
distribution at a randomly chosen SNR within the variant's range, add it
to the input, and pass it through the encoder. The reconstruction target
remains the original clean version.

## Decision rule

Same as prior experiments:

- Strict win = ≥ 1 pp TUEV BAC, non-overlapping CIs, noise-twin flat.
- Weak win = ≥ 0.5 pp with paired permutation p < 0.05.

Two multi-condition-specific criteria:

- **Robustness on artifact-heavy held-out set**: evaluate each variant on
  TUAR (TUH Artifact dataset) — a held-out subset where artifact
  prevalence is high. A multi-condition-pretrained encoder should
  improve more on TUAR than on TUEV. If it doesn't, the noise injection
  is generic regularisation rather than artifact-specific robustness.
- **Reconstruction loss on noise-injected vs clean input**: the variant
  must show *less* reconstruction-loss degradation when the input is
  noise-injected at test time vs the clean baseline (because the encoder
  has been trained to be invariant). If the degradation is the same as
  N0, the noise injection didn't teach robustness, just regularisation.

## Pre-registered predictions

| Variant | Prediction TUEV BAC | Prediction TUAR |
| ------- | ------------------- | --------------- |
| N0 baseline | reference | reference |
| N1 EOG only | tied or weak win, ~+0.5 pp on TUEV; weak win, ~+1 pp on TUAR (TUAR has lots of EOG) | |
| N2 EMG only | tied; weak win on TUAR for EMG-heavy windows |  |
| N3 mixed @ 0.3 | strict win, ~+1 pp TUEV; strict win, ~+2 pp TUAR | |
| N4 mixed @ 0.5 | tied or weak win on TUEV (might be over-noised); strict win on TUAR, ~+3 pp |  |

The honest expected outcome: **N3 (mixed @ 0.3) wins on TUEV; N4 (mixed
@ 0.5) wins on TUAR**. The choice depends on whether artifact-heavy
recordings are the priority (clinical use case) or general representation
quality is (BCI use case). Both can be reported.

## Implementation pointers

- EOG synthesis: 1-D Brownian motion (cumulative sum of Gaussian noise),
  bandpass-filtered to 0.5–4 Hz, scaled to peak ≈ 50 µV. Roughly mimics
  blink shape.
- EMG synthesis: white Gaussian noise, bandpass-filtered to 30–100 Hz,
  amplitude-modulated by a slowly varying envelope (10 Hz LP-filtered
  uniform [0,1]) to mimic burst-like EMG.
- Line noise: pure 50 or 60 Hz sinusoid, randomly shifted in phase per
  injection.
- Mixing routine:
  ```python
  if np.random.random() < p_inject:
      noise = sample_noise(variant)
      sigma_signal = clean_input.std()
      sigma_noise = noise.std()
      target_snr_db = np.random.uniform(*snr_range)
      scale = sigma_signal / (sigma_noise * 10**(target_snr_db / 20))
      noisy_input = clean_input + scale * noise
  else:
      noisy_input = clean_input
  ```
- Cache real EOG / EMG segments from the EEG denoising literature
  (`EEGdenoiseNet` provides clean and noisy reference segments) as a
  more realistic alternative to synthetic noise. Re-run N3 with these
  realistic noise sources as a sanity-check follow-up if N3 wins with
  synthetic.

## Output

`mini_experiments/09_multicondition_input/results.md` containing:

1. 5 × 2 (× 3 seed) results table on TUEV.
2. Same table on TUAR.
3. Reconstruction-loss-on-noisy-input table (showing whether the model
   learned to be robust).
4. Optional: results with realistic recorded EOG / EMG (from
   `EEGdenoiseNet`) instead of synthetic, for the N3 / N4 winners.
5. The chosen multi-condition strategy with decision-rule justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| The denoised target from exp08 *already* gives the model a clean signal to predict; multi-condition input is redundant | Run N3 also with raw target (T0) for comparison. The two together quantify which trick contributes more. |
| Synthetic noise doesn't match real artifact distribution; results don't transfer to real data | Always re-run the winner with `EEGdenoiseNet` real-noise references before declaring victory. |
| The model learns to "subtract the noise" trivially because it knows the noise was just added (the encoder sees both the original and the noisy version implicitly via gradient) | This is fine; the result is still a noise-invariant representation. The eval metric is what matters. |
| Higher injection rates over-noise the input and break learning | The standard fix from WavLM: don't go above p=0.5; if losses diverge, drop p. |

## What gets carried forward

The single chosen multi-condition recipe (which type of noise, what
probability, what SNR range). The recipe is added to the canonical training
loop, alongside whatever masking strategy exp10 selects.
