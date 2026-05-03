# exp03 / mini-experiment 19 — Decoder design

> **Status:** planned (added 2026-05-03 from the deep-research design refresh)
>
> **Cross-reference:** [`mini_experiments.md` §4.2 default architecture](../../mini_experiments.md#42-default-architecture-for-any-axis-not-under-test),
> [exp17 generative paradigm](../17_generative_paradigm/),
> [exp18 reconstruction target](../18_reconstruction_target/),
> [MAE (He et al. 2022, arXiv 2111.06377) §4.5 + Table 1a](https://arxiv.org/abs/2111.06377),
> [VideoMAE (Tong et al., NeurIPS 2022, arXiv 2203.12602)](https://arxiv.org/abs/2203.12602),
> [bioFAME (Liu et al. 2024, arXiv 2309.05927)](https://arxiv.org/abs/2309.05927),
> [SAMBA (Hong et al. 2025, arXiv 2511.18571)](https://arxiv.org/abs/2511.18571)
>
> **Compute budget:** 12 H100-hours (4 depths × 3 types = 12 cells × 2
> control columns × 5 seeds = 120 cells × 6 min average — decoders are
> small and cheap; the pretraining cost dominates not the decoder forward)
>
> **Gates:** none downstream — exp19 reports a decoder configuration that
> feeds into exp12 quick-wins consolidation

## Question

What decoder **depth** × **type** combination, applied on top of the
chosen frontend + backbone + generative-paradigm + reconstruction-target
stack from exp02 / 03 / 17 / 18, gives the best frozen-probe representation
quality on the §4.3 eval suite?

## Why it matters

Vision SSL has a clear and decisive answer for *its* setting: MAE (He
et al. 2022) Table 1a shows that for ImageNet linear-probe quality,
decoder depth 1 → 8 blocks moves the metric from 65.5 % → 73.5 %, while
fine-tuning quality stays flat at 84.8–84.9 %. The mechanistic
explanation is that a deeper decoder "absorbs" reconstruction
specialisation, leaving the encoder representations more abstract and
linearly-readable. **For frozen-probing-as-primary-eval, this is an
8 pp gap** — larger than the gap from any other architectural choice in
the MAE paper.

But the finding does *not* transfer cleanly to other modalities:

- **VideoMAE ([Tong et al. 2022](https://arxiv.org/abs/2203.12602))**: a
  1-block decoder *degrades* even fine-tuning for video — the higher
  redundancy and temporal-block masking makes the decoder's job harder,
  so it needs more capacity. The MAE 1≈8 finding *inverts* in the
  high-redundancy data regime.
- **bioFAME (biosignals, arXiv 2309.05927)**: encoder depth ablation in
  biosignals shows shallower (3-4 layers) > deeper (5-6) — the opposite
  of what scaling laws predict. The mechanistic explanation: low-SNR data
  doesn't reward representational depth the way high-SNR data does, and
  a deeper network is more prone to overfitting the noise.
- **SAMBA ([Hong et al. 2025](https://arxiv.org/abs/2511.18571))**: uses
  a U-Net-style decoder that mirrors the encoder structure with Mamba-2
  blocks and parameter-free linear interpolation upsampling — a
  fundamentally different decoder topology that "avoids checkerboard
  artifacts and better preserves the continuity of EEG signals". The
  best EEG-specific decoder design as of late 2025.
- **MAR / I-JEPA / latent-target paradigms**: have no traditional decoder
  at all — just a small predictor head (3-layer MLP) onto the latent or
  diffusion-loss space. exp17 / exp18 may pick a paradigm where the
  decoder is essentially absent.

The right answer for our recipe — bidirectional Mamba-2 backbone,
single-channel iid EEG, frozen-probing primary metric — is **unknown**.
The literature provides four plausible defaults that all disagree.

## Variants

The variants are a 4 × 3 matrix (depth × type), holding everything else
fixed at the exp02 / 03 / 17 / 18 winners.

### Depth axis (4 levels)

| Code | Decoder depth | Layers |
| ---- | ------------- | ------ |
| D1 | 1 | minimum reasonable: a single block of decoder |
| D2 | 2 | the §4.2 default |
| D4 | 4 | matches the encoder's bottom-half capacity |
| D8 | 8 | the MAE 2022 default; matches the encoder full depth |

### Type axis (3 options)

| Code | Decoder type | Description |
| ---- | ------------ | ----------- |
| TY-MA | Bidirectional Mamba-2 | the §4.2 default (matches encoder family); width d=256 |
| TY-TR | Bidirectional Transformer with RoPE + FlashAttention-2 | width d=256, 4 heads; the MAE 2022 default architecture |
| TY-UN | U-Net (SAMBA-style) | mirror-symmetric to the encoder, with parameter-free linear interpolation upsampling and Mamba-2 blocks at each refinement stage; depth column = number of refinement stages |

The **U-Net option** is included because the SAMBA paper specifically
identifies it as the best decoder topology for *EEG signal continuity*,
and it never appears in any of the other 16 mini-experiments. Its depth
column is interpreted differently (number of refinement stages, not
number of stacked blocks) but the parameter count is comparable to the
Mamba-2 / Transformer counterparts at matched depth.

### Cells (the 4 × 3 matrix)

|       | TY-MA Mamba-2 | TY-TR Transformer | TY-UN U-Net |
| ----- | ------------- | ----------------- | ----------- |
| **D1** | D1-MA (12 cells × 5 seeds) | D1-TR | D1-UN |
| **D2** | D2-MA (= §4.2 default) | D2-TR | D2-UN |
| **D4** | D4-MA | D4-TR | D4-UN |
| **D8** | D8-MA | D8-TR | D8-UN |

To keep the budget bounded, we run the **full 4 × 3 grid at 1 seed
each** as a fast initial sweep (12 cells × 1 seed × ~5 minutes = ~1 hour),
then run **5 seeds × matched-noise twin** for only the top-3 cells from
the initial sweep. This is a standard "screening then confirmation"
protocol from [`methodology.md` §3](../../methodology.md#3-how-to-design-one-ablation-the-matrix-shape).

## Controls (the §3 matrix, applied to confirmation cells only)

|              | EEG signal | matched-noise twin |
| ------------ | ---------- | ------------------ |
| Top-3 cells  | ✓          | ✓                  |

Matched-noise twin for U-Net specifically: the parameter-free
upsampling means the U-Net's output is *literally* a convolution of the
input — applied to Gaussian noise, the output is just smoothed Gaussian
noise. If the U-Net's EEG win partially generalises to noise, the gain
is from the upsampling structure, not from EEG-specific decoding.

## Held constant

- Pretraining data: 100h HBN-EEG subset.
- Frontend: exp02 winner.
- Backbone: exp03 winner (likely bidirectional Mamba-2, 6 layers,
  d=256, state N=64).
- Generative paradigm: exp17 winner. **If exp17 chose G1 AR (no
  decoder)** then this experiment runs only D1 × TY-{MA, TR} as a sanity
  check on the next-token-prediction *head* (which is conceptually the
  decoder for AR), with the U-Net column dropped.
- Reconstruction target: exp18 winner.
- Masking strategy + ratio: exp10 winner.
- Bottleneck: continuous (exp11 explores quantization separately).
- Loss family: dictated by exp18 winner.
- Optimiser: AdamW, LR re-tuned per cell (3-LR sweep) since width / depth
  changes affect the optimal LR. The 12-cell screening uses a single LR
  per cell (the exp18-winner's LR); the 3-cell confirmation does the
  full sweep.
- Pretraining duration per cell: 1 hour H100 wall-clock for screening,
  2 hours per seed for confirmation.
- Eval suite per [§4.3](../../mini_experiments.md#43-evaluation-suite-for-every-experiment-unless-overridden).

## Decision rule

Two-stage:

**Stage 1 (screening, 12 cells × 1 seed).** Rank cells by HBN 6-task BAC
on the 1-seed result. Promote the top 3 to confirmation.

**Stage 2 (confirmation, 3 cells × 5 seeds × matched-noise twin).** Apply
the standard exp03-family decision rule:

- **Strict win**: ≥ 1 pp HBN 6-task BAC over D2-MA (the §4.2 default),
  non-overlapping CIs, noise-twin flat.
- **Weak win**: ≥ 0.5 pp with paired permutation p < 0.05.
- **Tie**: TOST equivalence within ε = 0.5 pp. **In a tie, the smallest
  decoder is kept** — the burden of proof is on adding capacity.
- **Loss**: ≥ 0.5 pp below D2-MA with p < 0.05.

Two decoder-specific criteria:

- **Reconstruction-loss floor**: the chosen decoder must achieve
  ≥ 30 % reconstruction-loss reduction over D1-MA on training. A
  decoder that doesn't reduce loss meaningfully is just dead capacity.
- **Decoder-vs-encoder capacity ratio**: the chosen decoder must be
  ≤ 50 % of the encoder's parameter count. A decoder larger than half
  the encoder is structurally wrong for the asymmetric MAE design (per
  MAE 2022 §3.1: "the decoder is only used during pre-training to perform
  the image reconstruction task; only the encoder is used to produce
  image representations for recognition").

## Pre-registered predictions

| Cell | Prediction HBN 6-task BAC vs D2-MA | Reasoning |
| ---- | ----------------------------------- | --------- |
| D1-MA | weak loss, ~−1 pp | Too shallow; consistent with MAE 1-vs-8 finding for linear probe |
| D2-MA | reference (the §4.2 default) | — |
| D4-MA | **weak win, ~+0.5 pp** | The MAE pattern (deeper decoder helps linear probe) probably extends to D4 before plateauing |
| D8-MA | tied or weak loss | At D8, the decoder is the same depth as the encoder; the asymmetric advantage is gone |
| D1-TR | weak loss | Transformer at depth 1 has no global context; underperforms Mamba-2 at the same depth |
| D2-TR | tied vs D2-MA | Wash; the choice of decoder type matters less than depth |
| D4-TR | tied vs D4-MA | Same |
| D8-TR | tied vs D8-MA | Same |
| D1-UN | weak win on continuity-sensitive metric, tied on HBN 6-task | U-Net's upsampling structure preserves signal continuity but doesn't help the multi-class discrimination directly |
| D2-UN | weak win on PLV, tied on HBN 6-task | Same |
| D4-UN | **strict win on PLV reconstruction, weak win on HBN 6-task** | The SAMBA-style U-Net at moderate depth combines continuity preservation with enough capacity for representation specialisation |
| D8-UN | tied; over-parameterised | — |

The honest expected outcome: **D4-MA wins the headline metric; D4-UN
wins the phase / continuity metric; the gap between them is small enough
(~0.5 pp) that operational simplicity (Mamba-2 matches the encoder
family) makes D4-MA the recommended default.** The contrarian outcome —
which would be a surprise — is that all 12 cells are within noise of
each other, in which case the decoder choice doesn't matter and we keep
the §4.2 default D2-MA.

## Implementation pointers

- **TY-MA Mamba-2 decoder**: stack of `mamba_ssm.modules.mamba2.Mamba2`
  blocks at d=256, state N=64 (matches encoder). Bidirectional via
  forward + reversed sum. Wrap with input projection from masked-position
  embeddings + position embeddings.
- **TY-TR Transformer decoder**: stack of standard Transformer blocks at
  d=256, 4 heads, RoPE position embedding, FlashAttention-2 via
  `torch.nn.functional.scaled_dot_product_attention`. Bidirectional = no
  causal mask.
- **TY-UN U-Net decoder**: mirror-symmetric to encoder. Each refinement
  stage = (parameter-free linear interpolation upsampling by 2×) +
  (Mamba-2 block at increasing channel width). Skip connections from
  the corresponding encoder stage. Reference:
  [SAMBA §III](https://arxiv.org/abs/2511.18571).
- **Depth column for U-Net**: number of refinement stages, not stacked
  Mamba blocks. D1 = 1 upsampling + 1 Mamba block; D8 = 8 upsampling
  stages (which exceeds the encoder's 6 stages — extra stages are no-ops
  with identity skip). Practical limit is D6 = matched to encoder; we
  test D4 and D8 anyway to surface this limit empirically.
- **Screening protocol**: train all 12 cells for 30 minutes each (≈
  6 hours total wall-clock on a single H100), pick top 3 by HBN 6-task
  BAC. Then run those 3 with the full protocol (5 seeds × matched-noise
  twin × 2 hours per seed = 60 hours wall-clock total, fits in the
  12-H100-hour budget at 5× parallelism on the 8×A100 box).

## Output

`mini_experiments/19_decoder_design/results.md` containing:

1. The 12-cell screening table (1 seed each) with HBN 6-task BAC + total
   parameters per cell.
2. The 3 × 2 (× 5 seed) confirmation table on HBN CBCL externalizing R²,
   HBN 6-task BAC, attention regression R², k-NN top-1, PLV
   reconstruction quality, predict-site probe.
3. Decoder-vs-encoder capacity ratio per chosen cell.
4. Reconstruction-loss curves per cell.
5. The chosen decoder configuration with explicit decision-rule
   justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| The exp17 winner is G1 AR (no decoder), making this experiment partially moot | Reduce to a 4 × 2 grid (depth × {MA, TR}) testing the AR next-token-prediction head sizes. The U-Net column is dropped because it doesn't make sense for AR. Total compute drops from 12 to 6 H100-hours. |
| The exp18 winner is TR2 latent (small predictor head, not a true decoder) | Same reduction: test the predictor head's depth × type only. |
| The 12-cell screening is too noisy at 1 seed to identify the right top-3 | Increase screening seeds to 2 (24 cells × 30 minutes = 12 hours wall-clock); slightly over budget but more reliable. |
| All cells tie at the screening stage (the eval is dominated by encoder, not decoder) | This is itself an interesting result: if it happens, document that decoder choice is a low-importance axis for our recipe and keep D2-MA as the default. Compute saved by skipping confirmation. |
| U-Net decoder's parameter-free upsampling produces aliasing artifacts that hurt high-γ representation | Add an anti-aliasing low-pass filter (Kaiser β=8.6) before each upsampling step; this is the SAMBA recipe in §III. |

## What gets carried forward

The chosen decoder depth × type. Its hyperparameters (width, head count,
upsampling kernel) are frozen for every later experiment. The U-Net
result, even if not chosen, informs whether exp14 (context length) should
use a U-Net decoder for the long-context cells where the SAMBA paper
specifically reports U-Net's continuity-preservation advantage.
