# exp03 / mini-experiment 03 — Backbone ablation

> **Status:** planned
>
> **Cross-reference:** [`methodology.md` §2 Phase 2](../../methodology.md#phase-2--ablation-engine-1-3-days-1-8-gpus),
> [`brain/eeg-research.md` §6.1](../../../../../brain/eeg-research.md#61-the-2023-2026-eeg-fm-model-zoo)
>
> **Compute budget:** 24 H100-hours (4 backbone variants × 2 control columns
> × 3 seeds = 24 cells × 60 min average — backbones differ in throughput so
> the budget per cell varies)
>
> **Gates:** experiments 05, 06, 07, 08, 09, 10, 11, 12

## Question

At iso-FLOP and iso-data, which backbone family — Transformer, bidirectional
Mamba-2, LRU with complex eigenvalues, or a hybrid SSM × local-attention —
produces the best low-SNR EEG representation while handling sequences of up
to 60k samples (a 30-second window at 2 kHz)?

## Why it matters

The backbone determines four things that downstream layers cannot recover:

1. **Effective context length.** A pure Transformer's quadratic attention
   becomes intractable above ~10k tokens even on H100. Mamba-2 / LRU /
   Hyena are O(N) or O(N log N); they handle 60k without blinking. For
   single-channel EEG at 2 kHz, the difference is "you can model a
   30-second window" vs "you can't".
2. **Phase tracking.** A real-valued recurrence (Mamba) can preserve phase
   only if the input encoding and the loss include phase information; the
   recurrence itself cannot store oscillation phase as a single scalar.
   LRU's complex eigenvalues `λ = exp(-exp(ν) + iθ)` literally encode
   phase rotation in state space — a structural rather than incidental
   property.
3. **Selective gating of noise.** Mamba's input-dependent Δ, B, C gives
   the model a learned mechanism to gate out artifact regions in the state.
   S4 / S4D / LRU do not have this: their dynamics are input-independent.
   For low-SNR signals this is the difference between "the artifact gets
   smoothed into the state" and "the artifact gets actively suppressed".
4. **Training stability.** LRU's exponential parameterisation guarantees
   `|λ| < 1` unconditionally. Mamba-2 has known numerical issues in the
   `segsum` primitive that can produce NaNs in FP32 if the official Triton
   kernel is not used. Pure Transformers train predictably but can suffer
   loss spikes at scale. The choice here also chooses what training-time
   pathologies you will spend the next month debugging.

The question is which combination of these four properties is empirically
best on EEG, given that a learned phase loss (exp07) and a denoised target
(exp08) can compensate for some of the structural weaknesses. Different
papers in 2025–2026 picked different points on this spectrum (EEGM2 chose
Mamba-2; FEMBA chose bidirectional Mamba-1; EEGMamba added MoE; no
published paper uses LRU on EEG); a controlled iso-FLOP comparison does not
yet exist in the literature.

## Variants

Each backbone takes the (B, T', D=256) output of the chosen frontend
(default: F0 vanilla strided-conv until exp02 settles) and produces a
contextualised (B, T', D=256) sequence.

| Code | Variant | Layers | Params | Complexity | Library |
| ---- | ------- | ------ | ------ | ---------- | ------- |
| B0 | Bidirectional vanilla Transformer with RoPE, FlashAttention-2 | 6 | ≈ 8 M | O(N²) | `torch.nn.functional.scaled_dot_product_attention` |
| B1 | Bidirectional Mamba-2 (state N=64) | 6 | ≈ 8 M | O(N) | `state-spaces/mamba` (official Triton kernel mandatory) |
| B2 | Bidirectional LRU (complex eigenvalues, θ initialised to EEG bands δ/θ/α/β/γ) | 6 | ≈ 8 M | O(N) via parallel scan | `[lru-pytorch](https://github.com/Gothos/LRU-pytorch)` or hand-port |
| B3 | Hybrid: 5 × Mamba-2 + 1 × local-window attention (window=512), Zamba-2 ratio 5:1 | 6 | ≈ 8 M | O(N) + O(N · w) | composed from B0 + B1 |

LRU's θ initialisation: 64 state dimensions split as
`[δ:14, θ:14, α:12, β:12, γ:12]` with frequencies log-spaced inside each
band, replicating the BrainWave-Scattering Net trick of seeding wavelets at
physiologically meaningful scales.

## Controls

|                  | EEG signal | matched-noise twin |
| ---------------- | ---------- | ------------------ |
| B0 Transformer   | ✓          | ✓                  |
| B1 Mamba-2       | ✓          | ✓                  |
| B2 LRU           | ✓          | ✓                  |
| B3 Hybrid        | ✓          | ✓                  |

The matched-noise twin doubles as a check that long-range modelling capacity
isn't being credited where it shouldn't: if a backbone improves on EEG and
also improves on Gaussian noise, what it's modelling is positional
correlations, not EEG content.

## Held constant

- Pretraining data: 100h TUEG subset, single rate 250 Hz.
- Frontend: F0 vanilla strided conv (the §4.2 default; if exp02 has settled
  before exp03 launches, swap in the exp02 winner).
- Bottleneck: continuous.
- SSL framework: MAE, 50 % random mask, asymmetric encoder / decoder.
- Loss: L1 on raw signal + 0.3 × MR-STFT log-magnitude.
- Optimiser: AdamW + ScaledAdam variant per Zipformer ([Yao et al. ICLR
  2024](https://arxiv.org/abs/2310.11230)) — empirically more stable for
  long-sequence SSL. LR swept {1e-4, 3e-4, 1e-3} per backbone.
- Window length: 4 seconds at 250 Hz = 1000 samples (small enough that B0
  Transformer is feasible iso-FLOP; large enough to see meaningful
  long-range structure).
- Training duration: matched to *FLOPs not steps* — each backbone gets
  identical FLOPs per training run, computed via FLOP-counter (B0 needs
  fewer steps because each step is more expensive).
- Eval suite: standard per [`mini_experiments.md` §4.3](../../mini_experiments.md#43-evaluation-suite-for-every-experiment-unless-overridden).

## Decision rule

Same as exp02:

- Strict win = ≥ 2 pp TUEV BAC, non-overlapping CIs, noise-twin flat.
- Weak win = ≥ 1 pp with paired permutation p < 0.05.
- Tie = TOST within ε = 1 pp.
- Loss = ≥ 1 pp below baseline with p < 0.05.

Two additional backbone-specific criteria:

- **Long-sequence sanity**: each variant must run a 60k-sample sequence on
  one H100 in < 60 seconds per training step at the chosen layer count and
  width. If it can't, it's disqualified from the headline run regardless of
  small-window performance, because the headline run uses 30-second windows
  at 2 kHz.
- **Stability sanity**: each variant must complete the full pretraining run
  with no NaN losses across all 5 seeds. A single NaN seed disqualifies the
  variant unless the bug is specifically pinpointed and fixed (e.g. Mamba-2
  segsum FP32 issue → switch to FP16 with the official kernel).

## Pre-registered predictions

| Variant | Prediction | Reasoning |
| ------- | ---------- | --------- |
| B0 Transformer | Best on TUEV BAC at 1k-sample windows; fails the long-sequence sanity at 60k | Standard Transformer is competitive at short seq; quadratic cost kills it long. |
| B1 Mamba-2 | Strict win on long-sequence sanity; tied or weak win on TUEV BAC at 1k | EEGM2 result (better at long seq); selectivity helps low-SNR. |
| B2 LRU | Best on phase-locking-value eval (the structural advantage); weak loss on TUEV BAC | Phase tracking comes at the cost of no input selectivity for noise gating. |
| B3 Hybrid | Marginal weak win over B1 on TUEV BAC; same long-sequence performance as B1 | Local attention adds short-range precision; cost is implementation complexity. |

The honest expected outcome: **B1 Mamba-2 wins**, with B2 LRU being
explicitly preserved as an alternative if exp07 (phase handling) reveals
that explicit phase loss can't compensate for B1's real-valued state.

## Implementation pointers

- B0 Transformer: standard 6-layer, 4-head, d=256, RoPE position embeddings,
  FlashAttention-2 via `torch.nn.functional.scaled_dot_product_attention`.
  Bidirectional = no causal mask.
- B1 Mamba-2: the `mamba_ssm.modules.mamba2.Mamba2` block from
  `state-spaces/mamba`, wrapped bidirectionally as in
  [FEMBA](https://arxiv.org/abs/2603.26716) (forward Mamba + backward
  Mamba on reversed input, summed). **Use the official Triton kernel.** Do
  NOT reimplement segsum.
- B2 LRU: parallel scan implementation per [Orvieto et al. ICML
  2023](https://arxiv.org/abs/2303.06349) §3. Bidirectional = forward +
  reversed. The complex parallel scan needs care; reference
  implementations exist in the LRU repo and in `s5-pytorch`. Initialise θ
  to log-spaced EEG bands as described above.
- B3 Hybrid: stack 5 × B1 blocks then 1 × B0 block with local windowed
  attention (window=512). Zamba-2 paper has the exact recipe.
- ScaledAdam: implementation in `[lhotse-speech/icefall](https://github.com/lhotse-speech/icefall)` —
  port the optimizer file. Marginal but free win.

## Output

`mini_experiments/03_backbone_ablation/results.md` containing:

1. 4×2 (×3 seed) results table with all metrics and CIs.
2. Long-sequence throughput table (seconds per training step at
   {1k, 5k, 16k, 60k} samples per H100).
3. Stability table (NaN incidence, loss-spike count, divergence count
   across 5 seeds × 2 LR sweeps each).
4. Phase-locking-value reconstruction quality per backbone.
5. Encoder feature health per backbone.
6. The chosen backbone with decision-rule justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| Mamba-2 segsum NaN at FP32 destroys the run | Use the official Triton kernel; verify on a 100-step toy run before launching the full ablation. |
| LRU parallel scan has buggy backward pass for complex tensors | Compare gradient against finite-difference on a small example; if buggy, fall back to S5 (which has matured implementations). |
| Transformer wins at 1k-sample windows but cannot scale to 60k — do we still credit the win? | No. The headline run uses 30-second windows at 2 kHz. A backbone that wins only at toy scale loses outright. |
| Hybrid B3 is too compute-expensive to fit in budget | Drop to 7:1 Mamba:attention ratio per Jamba; this is the last variant to run, so cost overrun affects nothing else. |
| All variants tie on TUEV BAC (the test is too easy) | Switch the headline metric to TUEV WF1 or to the Sleep-EDF macro-F1, which is harder. |

## What gets carried forward

The single chosen backbone. The hybrid B3 result also informs whether the
headline run should add a single global-attention layer at the top — even
if B1 wins narrowly over B3, if B3's global-attention layer specifically
helps phase coupling, that signal goes into exp07's design.
