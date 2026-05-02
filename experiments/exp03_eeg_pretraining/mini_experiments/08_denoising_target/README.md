# exp03 / mini-experiment 08 — Denoised target

> **Status:** planned
>
> **Cross-reference:** [`brain/eeg-research.md` §7](../../../../../brain/eeg-research.md#7-eeg-denoising--the-three-camps),
> [`brain/eeg-research.md` §10.1](../../../../../brain/eeg-research.md#101-low-snr-strategies)
>
> **Compute budget:** 18 H100-hours (5 target variants × 2 control columns ×
> 3 seeds; the offline denoising itself is a one-time preprocessing pass and
> is not counted in the H100-hours)
>
> **Gates:** experiment 09 depends on the chosen target signal

## Question

If the SSL framework is MAE with a *denoised* target (the exp04 S1
recipe), which offline denoising method produces the cleanest target while
preserving the neural signal we actually want the encoder to learn:
bandpass filtering, ICA component rejection, PCA projection, wavelet
denoising, or the EEG-X recipe (IC-U-Net deep cleaning)?

## Why it matters

The denoised-target trick is the single most direct attack on EEG's low
SNR — instead of asking the model to reconstruct a noisy signal (and
therefore to learn the noise structure as a representation), ask it to
reconstruct a cleaned version. The cleaned target serves as a "what we
actually want the model to predict" supervisor without requiring labels.

The EEG-X paper ([arXiv:2511.08861](https://arxiv.org/abs/2511.08861)) is
the strongest published evidence for this approach in EEG. But EEG-X uses
IC-U-Net for cleaning, which is a learned model that itself was trained
elsewhere — adding a dependency to the pipeline. Cheaper alternatives
include classical signal processing methods that have decades of clinical
EEG validation:

- **Bandpass filtering**: 0.5–40 Hz captures the low-γ band and below.
  Discards everything above. Cheap, deterministic, no per-recording
  configuration.
- **ICA component rejection**: classical clinical pipeline. Decompose into
  components, automatically classify (ICLabel) and reject EOG / EMG / heart
  / line noise components, reconstruct. The most validated cleaning method
  in clinical practice, but requires multi-channel input — which conflicts
  with our iid-channel framing. Workaround: do ICA on the original
  multi-channel recording before iid expansion, then iid-expand the
  cleaned channels.
- **PCA projection**: keep only the top-K principal components per
  recording. Cheap; does not require multi-channel (can be done over time
  windows of a single channel). Used by EEGDM to lift to a higher-SNR
  latent space.
- **Wavelet denoising** (universal threshold or SURE-shrink): standard
  signal-processing denoiser. Per-channel, no learned model. Often used in
  clinical EEG for blink artifact removal.
- **IC-U-Net cleaning**: the EEG-X recipe. Most aggressive, but adds a
  pretrained dependency.

Each method differs in how aggressively it removes noise versus signal,
and the right answer for SSL pretraining is *not* necessarily the same as
the right answer for clinical visualisation: an aggressive denoiser may
remove subtle phase content that was actually useful for downstream BCI
tasks. exp08 measures this tradeoff directly.

## Variants

All variants share the same encoder + raw input; only the target signal
differs. The encoder still sees the noisy raw EEG; the loss compares its
reconstruction to the denoised target.

| Code | Variant | Cleaning method | Aggressiveness | Per-channel feasible |
| ---- | ------- | --------------- | -------------- | --------------------- |
| T0 | Raw target (the failure baseline from exp04 S0) | none | reference | yes |
| T1 | Bandpass 0.5–40 Hz | FIR bandpass with Kaiser window β=8.6 | low — preserves all in-band content including artifacts within band | yes |
| T2 | ICA + ICLabel cleaning + reconstruct | classical clinical pipeline | high — removes EOG / EMG / heart / line per component | requires multi-channel; do ICA on multi-channel original then iid-expand |
| T3 | PCA top-8 projection per 4-second window | retain top-8 PCs by variance, project back | moderate — keeps the dominant temporal patterns, drops low-variance noise | yes (PCA over time windows within a single channel) |
| T4 | Wavelet denoising (Daubechies-4, SURE-shrink, soft threshold) | per-channel wavelet | moderate — removes high-frequency noise spikes | yes |
| T5 | IC-U-Net cleaning (the EEG-X recipe) | pretrained IC-U-Net from [SCCN](https://github.com/sccn/IC-U-Net) | high — neural denoiser trained for EEG | requires multi-channel input to IC-U-Net |

T0 (raw target) is the explicit failure baseline — it's the same as exp04
S0 and is included so the magnitude of the gain over T0 can be measured.

## Controls

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| T0 raw target                  | ✓          | ✓                  |
| T1 bandpass 0.5–40             | ✓          | ✓                  |
| T2 ICA + ICLabel               | ✓          | ✓                  |
| T3 PCA top-8                   | ✓          | ✓                  |
| T4 wavelet denoising           | ✓          | ✓                  |
| T5 IC-U-Net cleaning           | ✓          | ✓                  |

The matched-noise twin sanity here is unusual: you have to define what
"denoising Gaussian noise" means. The convention: apply the same denoising
pipeline to the noise input as to the EEG. For T1 bandpass that just
restricts the noise spectrum; for T2 ICA the cleaning will be a no-op
because Gaussian noise has no structure to identify; for T5 IC-U-Net the
output is mostly zeros (the network classifies all components as noise).
The metric of interest is whether the model's downstream performance on
noise-twin remains flat — a denoising target should not give the model
any handle on Gaussian noise.

## Held constant

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: MAE with denoised target (S1; the exp04 winner if S1 won;
  if S4 EEGDM diffusion won exp04, this experiment is moot and skipped).
- Bottleneck: continuous.
- Mask: 50 % random.
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Optimiser: AdamW, LR carried forward from exp06.
- Pretraining duration: 8 epochs.

## Decision rule

Same as exp02 / exp03 / exp04 / exp06 / exp07:

- Strict win = ≥ 2 pp TUEV BAC over T0 raw, non-overlapping CIs,
  noise-twin flat.
- Weak win = ≥ 1 pp with paired permutation p < 0.05.
- Tie = TOST equivalence within ε = 1 pp.

Two target-specific criteria:

- **Signal-preservation sanity**: each cleaned target must retain
  > 80 % of the spectral power in the 1–30 Hz band of the original. A
  too-aggressive cleaner (e.g. wavelet thresholding with too-high
  threshold) will pass this with bandpass-style behaviour but smooth out
  important transient content. Spot-check on 5 randomly selected windows
  per variant.
- **Reconstruction floor**: each variant must achieve a noticeable drop in
  reconstruction loss vs T0 — if not, the cleaning isn't doing anything
  the model couldn't already do. (A signal that the bandpass filter
  removed should also be unrecoverable to the model from raw input.)

## Pre-registered predictions

| Variant | Prediction |
| ------- | ---------- |
| T0 raw target | floor — the failure case from exp04 S0 |
| T1 bandpass 0.5–40 | weak win, ~+1 pp; cheap, but high-γ content discarded |
| T2 ICA + ICLabel | strict win, ~+2–3 pp on TUEV BAC; substantial preprocessing cost upfront |
| T3 PCA top-8 | weak win, ~+0.5–1 pp; the cheapest single-channel-friendly method |
| T4 wavelet denoising | weak win, ~+1 pp; comparable to bandpass |
| T5 IC-U-Net cleaning | strict win, ~+2–4 pp on TUEV BAC; but adds external dependency on a pretrained model (which the user said no two-stage — so this is a soft violation) |

The honest expected outcome: **T2 ICA cleaning wins on absolute metric, T1
bandpass wins on practicality** (no per-recording configuration, no
multi-channel dependency, no extra dependency). T5 IC-U-Net is the
"borrowed pre-trained model" path that gives marginal further improvement
but adds infrastructure cost.

The most important secondary result: **whether the gain over T0 is small
enough that exp04 S0 (raw MAE + better loss + AAMP masking + multi-condition
input) approaches the same number without any denoising at all.** If yes,
the entire denoised-target pathway is replaceable by other improvements,
which simplifies the headline run.

## Implementation pointers

- Bandpass: `scipy.signal.firwin(N=251, cutoff=[0.5, 40], pass_zero=False, fs=250)` then
  `scipy.signal.filtfilt` for zero-phase. Apply once during pretraining-data
  preprocessing; cache the result.
- ICA + ICLabel: use MNE-Python's `mne.preprocessing.ICA(n_components=20, method='fastica')`
  on the original multi-channel recording. Then `mne_icalabel.label_components`
  for automatic classification. Reject components labelled as "muscle artifact",
  "eye blink", "eye movement", "heart beat", "line noise", or "channel noise".
  Reconstruct via `ica.apply(raw, exclude=...)`. Then iid-expand the cleaned
  channels.
- PCA top-8: `sklearn.decomposition.PCA(n_components=8)` on each (channel, 4s
  window) tensor reshaped to (windows, time-samples). Per-window cheap.
- Wavelet denoising: `pywt.threshold(coeffs, value=threshold, mode='soft')` after
  `pywt.wavedec(x, 'db4', level=5)` then `pywt.waverec`. Threshold via
  SURE-shrink rule.
- IC-U-Net: pretrained weights from
  [`sccn/IC-U-Net`](https://github.com/sccn/IC-U-Net) repo. Apply per
  recording on multi-channel data, then iid-expand.

All cleaning is done **once**, offline, and the cleaned signals stored as
side-streams in the dataset. This adds a few hours of CPU time upfront
(not counted in the H100-hours) and pays for itself many times over.

## Output

`mini_experiments/08_denoising_target/results.md` containing:

1. 6 × 2 (× 3 seed) results table.
2. Spectral power retention per variant (the signal-preservation sanity).
3. Reconstruction loss floor per variant.
4. The chosen target signal with decision-rule justification.
5. If T0 baseline is competitive (within 1 pp of any cleaned variant),
   recommend dropping the denoised-target pathway entirely and replacing
   with the multi-condition-input strategy from exp09.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| ICA fails on the iid-channel single-channel framing because ICA needs multi-channel | Do ICA on the original multi-channel recording, *then* iid-expand the cleaned channels. The cleaning depends on the spatial structure but the resulting per-channel signal is single-channel for SSL. |
| Different cleaning methods change the target distribution scale; the loss is no longer comparable across variants | Z-score the cleaned target separately per variant before computing the loss; report results in normalised metric space. |
| T5 IC-U-Net violates the "no pretrained dependency" constraint | Document this explicitly. T5's win is interesting; whether it's adopted depends on whether the user accepts the dependency as preprocessing. |
| All cleaned variants tie because the eval is dominated by features that survive any reasonable cleaning (e.g. low-frequency power) | Add a high-γ-sensitive eval — e.g. inner-speech decoding on Chisco subset, where the gain from preserving high-γ matters. |

## What gets carried forward

The single chosen target signal (or T0 if no clean variant strict-wins).
The cleaning recipe is then frozen, codified as an explicit preprocessing
stage, and added to the canonical preprocessing pipeline (V2 → V3, with
the cleaning step appended).
