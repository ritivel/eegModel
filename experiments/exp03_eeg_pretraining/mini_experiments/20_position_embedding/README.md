# exp03 / mini-experiment 20 — Position embedding

> **Status:** planned (added 2026-05-03 from the deep-research design refresh)
>
> **Cross-reference:** [`mini_experiments.md` §4.2 default architecture](../../mini_experiments.md#42-default-architecture-for-any-axis-not-under-test),
> [exp03 backbone ablation](../03_backbone_ablation/),
> [REVE (El Ouahidi et al., NeurIPS 2025, arXiv 2510.21585)](https://arxiv.org/abs/2510.21585)
> §3.2 + Table 19 (their 4D Fourier positional encoding ablation),
> [RoPE (Su et al. 2021, arXiv 2104.09864)](https://arxiv.org/abs/2104.09864),
> [NoPE (Kazemnejad et al., NeurIPS 2023, arXiv 2305.19466)](https://arxiv.org/abs/2305.19466)
>
> **Compute budget:** 6 H100-hours (5 position-embedding variants × 2 control
> columns × 3 seeds = 30 cells × 12 min average — position embedding
> changes are cheap to swap)
>
> **Gates:** none downstream — exp20 reports a configuration that feeds into
> exp12 quick-wins consolidation

## Question

For the chosen frontend + backbone (the exp02 + exp03 winners — likely
F0/F2 frontend feeding bidirectional Mamba-2), which positional encoding
scheme — **sinusoidal absolute, learned absolute, RoPE, NoPE (none),
or REVE-style 4D Fourier** — gives the best frozen-probe representation
on the §4.3 eval suite?

## Why it matters

Position embedding is the most "obviously settled" architectural choice
in the §4.2 default that **isn't actually settled**.

Three reasons it deserves a dedicated mini-experiment:

1. **Mamba-2 is *technically* position-implicit.** The selective scan
   mechanism encodes order through the recurrence itself, not through an
   explicit positional vector. Pure SSMs (S4, S4D, LRU) are usually run
   *without* any positional embedding. But every published EEG-Mamba
   paper (FEMBA, BioMamba, EEGMamba) adds a learned positional embedding
   anyway. EEGMamba is the lone exception. **Whether the embedding helps
   or hurts is empirically unsettled for biosignals.**
2. **REVE (NeurIPS 2025) has the strongest published EEG-specific
   evidence for any positional scheme.** Their 4D Fourier positional
   encoding (sinusoidal projection of the 3D electrode coordinates +
   the timestep, with a learned linear adaptation layer) outperforms
   both fixed-learnable and pure-MLP-based positional encodings in
   their Table 19 ablation. Their pretraining is the largest EEG-FM to
   date (60,000 hours, 25,000 subjects), so their conclusions carry
   weight. We've never tested this.
3. **RoPE is the dominant choice in modern LLMs (LLaMA, Mistral, Qwen,
   GPT-NeoX, Falcon) and offers good length extrapolation up to 2× via
   NTK scaling and 4–32× via YaRN.** Our exp14 (context-length scaling)
   tests pretrained-window 4 s → 30 s. RoPE's length extrapolation
   property may make a measurable difference for Mamba-2's long-context
   regime — an interaction that has never been tested for biosignal SSL.

## Variants

| Code | Variant | Mechanism | Where applied | Length-extrapolating? |
| ---- | ------- | --------- | ------------- | ---------------------- |
| P0 | **None (NoPE)** | no positional information added | n/a | n/a |
| P1 | **Sinusoidal absolute** (the §4.2 default proposal) | fixed sinusoidal vectors `PE_t = [sin(t/10000^{2i/d}), cos(...)]` added to frontend output | input to backbone | poor; degrades sharply beyond training length |
| P2 | **Learned absolute** | one learned `(T_max, d)` embedding table | input to backbone | none — no embeddings beyond `T_max` |
| P3 | **RoPE** | rotary position embedding applied via complex rotation of Q and K vectors at each Mamba-2 block; reference: [Su et al. 2021](https://arxiv.org/abs/2104.09864) | inside backbone (per-block) | good with NTK scaling; excellent with YaRN |
| P4 | **REVE-style 4D Fourier** | per-token Fourier features of `(electrode_x, y, z, time)` — for our single-channel iid setup, the spatial coordinates are constant per recording, so this reduces to 1D Fourier of time + a 3-coordinate spatial embedding for the iid-expanded channel; a small MLP adaptation layer combines them; reference: [REVE §3.2](https://arxiv.org/abs/2510.21585) | input to backbone | excellent (Fourier basis is natively position-extrapolating) |

**Why both NoPE and "remove the positional embedding entirely" deserve a
cell**. The standard rationale for adding a positional embedding to
Mamba is "belt and braces". But the standard rationale for removing it
(NoPE in Transformers) is "the network learns positional information
implicitly through the causal mask and through the input representation".
For Mamba, the recurrence itself encodes order; an additional positional
embedding may be either redundant or actively harmful (it can hurt
length extrapolation by tying representations to specific positions).
EEGMamba runs without positional embedding and reports SOTA on six EEG
tasks; the ablation has not been published.

**Why "absolute vs relative" doesn't appear as a separate axis here**.
The only relative-position scheme that has open-source production-quality
implementations for non-causal models is RoPE (which we test as P3).
ALiBi is causal-only and doesn't fit bidirectional Mamba-2.

## Controls (the §3 matrix)

|                               | EEG signal | matched-noise twin |
| ----------------------------- | ---------- | ------------------ |
| P0 None (NoPE)                | ✓          | ✓                  |
| P1 Sinusoidal absolute        | ✓          | ✓                  |
| P2 Learned absolute           | ✓          | ✓                  |
| P3 RoPE                       | ✓          | ✓                  |
| P4 REVE 4D Fourier            | ✓          | ✓                  |

Matched-noise twin for P3 specifically: RoPE's rotation is applied to
the Q, K vectors regardless of input content. On Gaussian noise, the
rotation produces noisy-but-rotated noise; the model should not learn
anything useful from it. If P3 wins on EEG and the noise-twin variant
also improves, the gain is from the rotation-as-regularisation effect
(it acts like a structured dropout on attention), not from the
positional information.

## Held constant

- Pretraining data: 100h HBN-EEG subset.
- Frontend: exp02 winner.
- Backbone: exp03 winner. **For pure SSM backbones (B1 Mamba-2, B2 LRU)**:
  positional embedding is added at the frontend output (input to
  backbone) for P1, P2, P4; injected per-block for P3. **For Transformer
  backbones (B0)**: same scheme; RoPE replaces the §4.2 RoPE that's
  already in the B0 implementation. **For hybrid (B3)**: applied
  identically to both Mamba and Transformer blocks.
- Generative paradigm: exp17 winner.
- Reconstruction target: exp18 winner.
- Decoder: exp19 winner (or §4.2 default if exp19 hasn't settled).
- Masking strategy: exp10 winner.
- Bottleneck: continuous.
- Optimiser: AdamW, LR retuned per cell (3-LR sweep) since position
  embedding can interact with the LR via the gradient flow into the
  backbone.
- Pretraining duration: 1 hour H100 wall-clock per seed.
- Eval suite per [§4.3](../../mini_experiments.md#43-evaluation-suite-for-every-experiment-unless-overridden),
  plus a **length-extrapolation check** (see Decision rule).

## Decision rule

For each variant V relative to P1 sinusoidal (the §4.2 default):

- **Strict win**: ≥ 1 pp HBN 6-task BAC over P1, non-overlapping CIs,
  noise-twin flat.
- **Weak win**: ≥ 0.5 pp with paired permutation p < 0.05.
- **Tie**: TOST equivalence within ε = 0.5 pp. **In a tie, the simplest
  variant wins**: P0 NoPE > P1 sinusoidal > P2 learned > P3 RoPE > P4
  Fourier-4D (in order of operational simplicity / parameter count).
- **Loss**: ≥ 0.5 pp below P1 with p < 0.05.

One position-embedding-specific criterion:

- **Length-extrapolation sanity**: each variant is evaluated at *both*
  the pretraining window (4 seconds at 500 Hz = 2000 samples) and a
  *2× extrapolated* window (8 seconds = 4000 samples) on the HBN
  6-task task. The 2× window is constructed by concatenating two
  consecutive pretraining windows from the same recording. P0 NoPE
  and P4 Fourier-4D should hold up; P2 learned absolute is expected to
  fail (no embeddings exist beyond the trained `T_max`); P1 sinusoidal
  and P3 RoPE should degrade gracefully. The chosen variant must lose
  ≤ 5 pp HBN 6-task BAC at the 2× window. exp14 (context-length
  scaling) tests this more thoroughly; here it's a sanity check.

## Pre-registered predictions

| Variant | Prediction HBN 6-task BAC at 4 s window | At 2× extrapolated 8 s window | Reasoning |
| ------- | ---------------------------------------- | ----------------------------- | --------- |
| P0 NoPE | **strict win, ~+1–2 pp over P1** | unchanged from 4 s | Mamba-2 already encodes order through its recurrence; adding sinusoidal absolute may interfere with the SSM's intrinsic positional handling. EEGMamba (the only published Mamba EEG-FM without positional embedding) reports SOTA. |
| P1 Sinusoidal | reference | weak loss, ~−2 pp at 8 s | The default; degrades smoothly past training length per [Position Encoding Comparison](https://mbrenndoerfer.com/writing/position-encoding-comparison-transformers). |
| P2 Learned absolute | tied or weak loss | **catastrophic loss, ~−10 pp at 8 s** | Learned embeddings have no representation for tokens beyond `T_max`. Useless for length extrapolation. |
| P3 RoPE | tied | **strict win at 8 s, ~+3 pp over P1** | RoPE's hallmark: graceful length extrapolation up to 2× without retuning. Within-training-length performance is comparable to sinusoidal. |
| P4 REVE 4D Fourier | **strict win, ~+1–2 pp** | weak win at 8 s | REVE's Table 19 ablation reports +2 pp average over fixed/learned/MLP positional schemes on 10 EEG benchmarks; the mechanism (Fourier features of explicit coordinates) is natively length-extrapolating. |

The honest expected outcome: **P0 NoPE and P4 Fourier-4D are essentially
tied at 4 s, with P4 having a slight edge at 8 s. P3 RoPE is the right
choice if exp14 conclusively shows that long-context generalisation is
needed; otherwise P0 NoPE is the right choice for operational
simplicity.** The P1 sinusoidal default that I proposed in the §4.2
table is *probably* the third-best option behind P0 and P4 — the
proposal was a placeholder, and this experiment is the empirical test.

## Implementation pointers

- **P0 NoPE**: do nothing. Skip the positional embedding addition step.
- **P1 Sinusoidal absolute**: standard implementation;
  `PE[t, 2i] = sin(t/10000^{2i/d})`, `PE[t, 2i+1] = cos(...)`. Add to
  frontend output before backbone input.
- **P2 Learned absolute**: `nn.Embedding(T_max, d)` with `T_max = 250`
  (the number of tokens at 4 s × 500 Hz / 8 = 250). Initialise with
  truncated normal (std = 0.02). Add to frontend output.
- **P3 RoPE**: for Mamba-2 backbone, RoPE is applied to the Q-like and
  K-like projections inside each block. Reference implementation in the
  `state-spaces/mamba` repo; the rotary frequencies are
  `θ_i = 10000^{-2i/d}` for i = 0, ..., d/2-1. NTK-aware scaling factor
  for length extrapolation: `θ_i' = θ_i / (T_eval/T_train)^{1/(d/2-1)}`.
  Reference: [Su et al. 2021](https://arxiv.org/abs/2104.09864),
  [NTK scaling](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/).
- **P4 REVE 4D Fourier**: per the REVE paper §3.2, generate
  `n_freq * 4` sinusoidal features from `(electrode_x, electrode_y,
  electrode_z, time_t)` — for our iid single-channel setup, the
  electrode coordinates are constant per (recording, channel) pair and
  can be precomputed. The time coordinate varies per token. Stack
  sin/cos to get a `2 * n_freq * 4` vector per token; project through a
  small linear layer + GeLU + LayerNorm to match `d`. Add to frontend
  output. `n_freq = 4` per the REVE paper. Reference:
  [REVE §3.2 + Algorithm 1](https://arxiv.org/abs/2510.21585).

For the iid-channel single-channel setup, the electrode coordinates for
P4 are looked up from the EGI HydroCel-128 standard layout (publicly
available; cached as a static lookup table in `src/exp03/electrodes.py`
once this experiment runs).

## Output

`mini_experiments/20_position_embedding/results.md` containing:

1. The 5 × 2 (× 3 seed) results table on HBN CBCL externalizing R²,
   HBN 6-task BAC, attention regression R², k-NN top-1, predict-site
   probe at the **4 s pretraining window**.
2. The same metrics at the **2× extrapolated 8 s window** — the
   length-extrapolation table.
3. Encoder feature health per variant.
4. The chosen positional embedding scheme with explicit decision-rule
   justification, plus a recommendation for whether RoPE/Fourier-4D
   should be used for the long-context cells in exp14.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| P0 NoPE wins by enough to invalidate the §4.2 default but the choice interacts with the chosen masking strategy (random masking shuffles patches; sinusoidal embedding makes order recoverable, NoPE does not) | Run a single-seed sanity cell P0-with-block-masking; if it differs from P0-with-random-masking, document the interaction and recommend P1 sinusoidal as the safer default. |
| P3 RoPE implementation for Mamba-2 has subtle bugs (the RoPE-Mamba combination is recent and not well-tested) | Verify against the official Mamba-2 repo's reference implementation; if it doesn't exist, fall back to a hand-port from the Vim (Vision Mamba) repo and validate via finite-difference gradients. |
| P4 REVE Fourier-4D's electrode coordinate lookup fails for HBN's HydroCel-128 (the coordinate table doesn't exist in MNE) | Use the standard EGI HydroCel-128 coordinates from the EEGLAB documentation; manually transcribe into a Python dict; cache. |
| All 5 variants tie on HBN 6-task BAC (the eval is too easy at this scale) | Switch the headline metric to HBN attention regression R² (Protocol A.1b); the continuous-label gradient is more discriminating. |
| The 8 s extrapolation eval is contaminated by the fact that the encoder's batch normalisation statistics were computed at 4 s — different window length means different statistics | Use LayerNorm (the §4.2 default already does); BatchNorm should be avoided. If exp02's chosen frontend uses BatchNorm, swap to LayerNorm for this experiment. |

## What gets carried forward

The single chosen positional embedding scheme. If P0 NoPE wins, the
configuration is "no positional embedding"; this is the simplest and
fastest choice. If P4 Fourier-4D wins, the electrode-coordinate lookup
table becomes a permanent fixture of the codebase and propagates to
every later experiment.

The length-extrapolation result also feeds directly into exp14 (context-
length scaling): a positional scheme that handles 2× extrapolation
gracefully removes one of the open questions about whether 30-second
windows can be evaluated on 4-second pretraining.
