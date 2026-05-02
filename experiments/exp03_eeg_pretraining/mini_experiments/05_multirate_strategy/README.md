# exp03 / mini-experiment 05 — Multi-rate strategy

> **Status:** planned
>
> **Cross-reference:** [`brain/experiments/tokenization/01-preprocessing.md`](../../../../../brain/experiments/tokenization/01-preprocessing.md) §2,
> [`methodology.md` §8.2](../../methodology.md#82-eeg-specific-gotchas-synthesised-from-exp01exp02--literature)
>
> **Compute budget:** 18 H100-hours (4 strategies × 2 control columns × 3
> seeds × 4 sample-rate test sets)
>
> **Gates:** none downstream — exp05 is one of the experiments most likely to
> have a definitive winner that just becomes a configuration choice

## Question

Of the four published methods to handle multiple EEG sampling rates without
losing information — uniform resample to a single rate, rate-conditioned Δ
in a state-space model, rate-specific CNN branches (MSR-HuBERT style),
SincNet's Hz-parameterised filters — which one minimises performance
degradation when the encoder must operate on rates it did not see during
pretraining?

## Why it matters

The pretraining corpus spans an order of magnitude in sampling rate
(100 Hz Sleep-EDF, 250 Hz TUEG, 500 Hz HBN-EEG / ZuCo, 1000 Hz THINGS-EEG2,
2000 Hz some clinical recordings — see
[`brain/experiments/tokenization/01-preprocessing.md` §2.1](../../../../../brain/experiments/tokenization/01-preprocessing.md#21-whats-at-stake)).
Every published EEG-FM solves this by uniform resampling to 200 / 250 / 256
Hz, throwing away high-γ and HFO content above 100 Hz Nyquist. The strict
information-theoretic reading is that this discards the band where lexical
and inner-speech content is known to live, which directly limits downstream
EEG-to-text and high-γ tasks.

The four alternatives are real but each has unproven scaling for EEG:

- **Mamba-Δ**: the SSM trick where setting `Δ = 1/fs` makes the same
  continuous A,B matrices work at any rate. Mathematically clean, zero
  extra parameters, but no published EEG-FM uses it; we'd be the first.
- **MSR-HuBERT branches**: separate CNN branches per rate, all converging
  to a shared frame rate before the backbone. Production-proven for speech
  (16 / 22.05 / 24 / 48 kHz) but never tried for EEG.
- **SincNet Hz-parameterised**: cutoff frequencies are stored in Hz, so
  the same SincNet layer applied at any rate gives the same passband.
  Multi-rate handling is essentially automatic.
- **Frozen scattering**: wavelets are scale-natural; higher fs gives more
  available octaves but the existing octave coefficients are unchanged.

## Variants

| Code | Variant | How multi-rate is handled | Information loss above 100 Hz Nyquist? |
| ---- | ------- | ------------------------- | ---------------------------------------- |
| M0 | Uniform resample to 250 Hz (the field default) | All inputs `scipy.signal.resample_poly` to 250 Hz before frontend | Yes: anything above 125 Hz Nyquist is gone |
| M1 | Mamba-Δ rate scaling | Frontend operates at native rate; Mamba's Δ parameter is set to `1/fs` per-batch | No (in theory) |
| M2 | MSR-HuBERT rate-specific CNN branches | One CNN branch per `fs ∈ {200, 256, 500, 1000, 2000}`; shared LayerNorm; shared backbone | No |
| M3 | SincNet Hz-parameterised + frozen scattering | SincNet cutoffs in Hz; Kymatio scattering with adaptive J,Q per rate | No |

Each variant is evaluated on the same downstream tasks but with input
sampling rate varied at evaluation time, including rates not seen at
pretraining.

## Controls

The matrix is bigger than the prior experiments because the test dimension
is "input rate at eval", not just signal vs noise:

|                  | EEG @ 250 Hz | EEG @ 500 Hz | EEG @ 1000 Hz | EEG @ 2000 Hz | matched-noise twin (250 Hz) |
| ---------------- | ------------ | ------------ | ------------- | ------------- | ---------------------------- |
| M0 resample      | ✓            | ✓ (resampled to 250) | ✓ (resampled to 250) | ✓ (resampled to 250) | ✓ |
| M1 Mamba-Δ       | ✓            | ✓ (native)   | ✓ (native)    | ✓ (native)    | ✓ |
| M2 branches      | ✓            | ✓ (native, branch routed) | ✓ (native, branch routed) | ✓ (native, branch routed) | ✓ |
| M3 SincNet+scattering | ✓       | ✓ (native)   | ✓ (native)    | ✓ (native)    | ✓ |

Pretraining is on the *mixed-rate* corpus (50h TUEG @ 250 Hz + 30h ZuCo @
500 Hz + 20h THINGS-EEG2 @ 1000 Hz) for a total of 100h, so the model has
seen 250 / 500 / 1000 Hz during pretraining. **2000 Hz is the held-out
test rate**: only seen at evaluation. M1 / M2 / M3 should retain
performance at 2000 Hz; M0 will resample everything down anyway.

## Held constant

- Frontend: M3 uses its own special frontend by definition; M0 / M1 / M2
  use the exp02 winner (or F0 vanilla until exp02 lands).
- Backbone: bidirectional Mamba-2 (the exp03 default; rate-conditioned Δ
  in M1 modifies it minimally).
- SSL framework: exp04 winner (default S1 MAE-denoised).
- Loss: L1 + 0.3 × MR-STFT log-magnitude.
- Pretraining duration: 8 epochs over the 100h mixed-rate corpus.
- Eval downstream task: TUEV (originally 250 Hz, but we will *upsample*
  it to 500 / 1000 / 2000 Hz to test out-of-pretrain rates) + an actual
  high-rate dataset (THINGS-EEG2 at 1000 Hz native, used for image-class
  retrieval) for the 1000 Hz native check.

The 2000 Hz held-out test rate is constructed by upsampling TUEV via
`scipy.signal.resample_poly` to 2000 Hz; this is information-preserving
(all frequency content stays the same; just more samples per second). The
question is whether M1 / M2 / M3 process the upsampled signal as well as
the native one.

## Decision rule

Two metrics matter:

1. **In-pretraining-rate performance**: TUEV BAC at 250 Hz. Each variant
   must be within 1 pp of the best. A variant that wins on multi-rate but
   tanks at the standard rate isn't useful.
2. **Out-of-pretraining-rate performance**: TUEV BAC at 2000 Hz (the
   held-out rate). The strict win condition: a variant maintains ≥ 95 %
   of its 250 Hz performance at 2000 Hz, while M0 (the resample baseline)
   does too because it just resamples down — so the comparison must
   *also* include the high-γ-relevant evaluation (THINGS-EEG2 at 1000 Hz
   native, which has actual content above 125 Hz).

For THINGS-EEG2:
- Strict win for M1 / M2 / M3 over M0: ≥ 5 pp top-1 retrieval
  improvement, non-overlapping CIs, noise-twin flat.
- Weak win: ≥ 2 pp with paired permutation p < 0.05.

The matched-noise twin at 250 Hz is the standard sanity. We don't run
noise-twin at higher rates because Gaussian noise sampled at 2000 Hz vs
upsampled from 250 Hz shouldn't differ for a well-behaved encoder.

## Pre-registered predictions

| Variant | TUEV @ 250 Hz | TUEV @ 2000 Hz | THINGS-EEG2 @ 1000 Hz top-1 retrieval |
| ------- | ------------- | -------------- | ------------------------------------------ |
| M0 resample | baseline | identical to 250 (resampled) | floor — high-γ content discarded |
| M1 Mamba-Δ | tied or weak loss vs M0 | ≥ 95 % of 250 Hz; passes | weak win over M0 |
| M2 branches | tied | passes | strict win over M0 |
| M3 SincNet+scattering | tied | passes; potentially the best because both layers are explicitly rate-agnostic | strict win over M0 |

The honest expected outcome: **M3 wins on the high-γ evaluation, M0 wins
on the standard rate by a hair** (because the CNN frontend is calibrated
for 250 Hz). The decision becomes whether high-γ content is important for
your downstream tasks (yes if EEG-to-text or inner-speech is on the
roadmap).

## Implementation pointers

- M0 resample: standard `scipy.signal.resample_poly`, with explicit FIR
  anti-alias filter (Kaiser window, β=8.6). Document the filter, not just
  the rate.
- M1 Mamba-Δ: in `state-spaces/mamba`, the `Mamba2` block exposes `Δ` as
  the input-dependent parameter computed via `softplus(linear(x))`. Override
  this to add a `+ rate_embedding(log(fs))` term, where `rate_embedding`
  is a small learned MLP from log(fs) to the Δ dimensionality.
- M2 branches: implement 5 separate `Conv1d`-based stems (one per rate),
  each with its own strides chosen so the output frame rate is constant
  (target 5 Hz frame rate; at 250 Hz that's stride 50, at 2000 Hz stride
  400). After each stem, a shared `LayerNorm` normalises into the same
  feature space before the shared backbone. Reference:
  [arXiv:2603.23048](https://arxiv.org/abs/2603.23048) Table 1.
- M3 SincNet + scattering: SincNet with cutoffs in Hz (not normalised
  frequency); Kymatio with `J = ceil(log2(fs / target_freq_min))` per
  input rate (so all rates resolve down to the same minimum frequency).

## Output

`mini_experiments/05_multirate_strategy/results.md` containing:

1. 4 × 5 × 3-seed results table (4 variants × 4 eval rates + noise-twin,
   3 seeds).
2. THINGS-EEG2 top-1 retrieval accuracy per variant.
3. Wall-clock cost per training step per variant (M2 has more parameters,
   M3 has frozen Kymatio overhead).
4. Encoder feature stats per input rate per variant (does the same encoder
   layer behave differently at different rates? Should not, for a
   genuinely rate-invariant variant.).
5. The chosen multi-rate strategy with decision-rule justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| Mamba-Δ rate conditioning destabilises training (unfamiliar regime for the SSM) | Warm-start with M0 (uniform 250 Hz) for 1 epoch, then introduce mixed rates over the next 2 epochs. |
| MSR-HuBERT branches don't generalise to held-out 2000 Hz (no branch defined for it) | Use the closest existing branch (1000 Hz) and resample 2000 Hz to 1000 Hz; this is a graceful failure but a known limitation. |
| M3 SincNet learns trivial cutoffs (all filters collapse to identical band) | Add the standard repulsive regulariser on cutoff distances. |
| M0 wins at 250 Hz by a wide enough margin that the high-γ argument doesn't outweigh it | Then exp05's answer is "stay at 250 Hz, but document that high-γ tasks will need a separate encoder". This is also a valid result. |

## What gets carried forward

The single chosen multi-rate strategy. If the answer is M0 (uniform
resample), the resampling rate (250 Hz) and filter family are frozen for
every later experiment. If the answer is M2 (branches), the branch
definitions for the supported pretraining rates are frozen. The 2000 Hz
test cell is informative regardless of which variant wins.
