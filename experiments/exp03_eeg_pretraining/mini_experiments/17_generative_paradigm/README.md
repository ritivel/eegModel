# exp03 / mini-experiment 17 — Generative paradigm for the Mamba backbone

> **Status:** planned (added 2026-05-03 from the deep-research design refresh)
>
> **Cross-reference:** [`mini_experiments.md` §4.2 default architecture](../../mini_experiments.md#42-default-architecture-for-any-axis-not-under-test),
> [`methodology.md` §2 Phase 2](../../methodology.md#phase-2--ablation-engine-1-3-days-1-8-gpus),
> [MAP (Liu & Yi, CVPR 2025, arXiv 2410.00871)](https://arxiv.org/abs/2410.00871),
> [Autoregressive Pretraining with Mamba in Vision (arXiv 2406.07537)](https://arxiv.org/abs/2406.07537),
> [MAR (Li et al., NeurIPS 2024, arXiv 2406.11838)](https://arxiv.org/abs/2406.11838)
>
> **Compute budget:** 10 H100-hours (3 paradigm variants × 2 control columns
> × 5 seeds = 30 cells × 20 min average — paradigms differ in per-step cost;
> MAR with diffusion head is most expensive)
>
> **Gates:** experiment 04 (must re-anchor framework comparison if MAE
> loses), 18 (reconstruction target), 19 (decoder design)

## Question

Given that the §4.2 default backbone is **bidirectional Mamba-2**, does the
§4.2 default SSL objective — **vanilla MAE with random patch masking and
pixel-space reconstruction at masked positions** — actually win against
the two paradigms that recent literature reports are better matches for
the Mamba/SSM backbone family: **scan-aligned causal autoregression (AR)**
and **bidirectional masked autoregression with a diffusion-loss head
(MAR)**?

## Why it matters

This experiment is the most consequential addition from the 2026-05-03
design refresh because it directly challenges the **single anchor point**
that every other mini-experiment in this folder has been pinned to.

Three independent 2024–2025 results, from three different research groups,
all point to the same conclusion: **MAE pretraining is structurally
mispaired with Mamba-class backbones**.

1. **MAP ([Liu & Yi, CVPR 2025, arXiv 2410.00871](https://arxiv.org/abs/2410.00871))**
   directly ablates MAE vs AR vs the MAP-hybrid (per-row local MAE +
   global AR across rows) on a hybrid Mamba-Transformer vision backbone
   and reports: *"MAE pretraining is better suited for Transformers,
   while AR is more compatible with Mamba. MAP, on the other hand, is
   more suited for the Mamba-Transformer backbone."* The MAP-hybrid
   beats both pure MAE and pure AR for the hybrid backbone; pure AR beats
   pure MAE for pure Mamba. The reported gap (Table 2 in the paper):
   AR for Mamba gets +1.4 pp over scratch supervised; MAE for Mamba gets
   +0.2 pp. **Vanilla MAE for Mamba is essentially worthless.**
2. **AR-Mamba ([arXiv 2406.07537](https://arxiv.org/abs/2406.07537))**:
   pure causal autoregressive pretraining of Mamba (Vim) achieves 83.2 %
   ImageNet top-1, +2.0 % over supervised — without any MAE objective at
   all. The mechanism: AR matches Mamba's intrinsically left-to-right
   scan structure; bidirectional MAE forces the model to do something it
   wasn't designed for.
3. **MAR ([Li et al., NeurIPS 2024, arXiv 2406.11838](https://arxiv.org/abs/2406.11838))**
   — "Autoregressive image generation without vector quantization" — uses
   bidirectional masked attention with random masking order plus a
   per-token diffusion-loss head, and beats both MAE and traditional AR
   on both generation FID and linear-probe representation quality. **MAR
   has never been applied to biosignals**; the absence is a genuine
   research gap rather than a deliberate omission.

These are vision-domain results, but the underlying mechanism (scan-order
alignment of pretraining objective with the SSM scan direction) is
architectural, not modality-specific. EEG is a 1D signal where Mamba's
linear-time scan is the entire selling point of the backbone choice; the
mechanism transfers directly.

The downstream consequence: **every other ablation in this folder
(exp02 frontend, exp04 framework, exp05 multirate, exp06–11 the
MAE-specific axes) is anchored on a baseline that the literature predicts
is the wrong baseline for Mamba.** If MAE wins exp17, every prior
experiment design stays valid. If AR or MAR wins, exp04's framework
comparison must be re-run with the new generative paradigm as the
backbone, and exp10's mask-strategy ablation becomes partially moot
(AR has no masking; MAR uses a different masking schedule than MAE).

This is therefore a **gate experiment** — exp17 must complete and report
its winner before exp04, exp18, exp19, and the headline run are designed.

## Variants

Each variant uses the chosen frontend (default: F0 vanilla strided conv
until exp02 lands) and the §4.2 default bidirectional Mamba-2 backbone
(6 layers, d=256, state N=64). The variant determines the
masking/ordering/loss topology around the backbone.

| Code | Variant | Topology summary | Per-step extra cost vs G0 |
| ---- | ------- | ---------------- | -------------------------- |
| G0 | **MAE-bidirectional** (the §4.2 baseline) | Random 50 % patch mask; encoder processes only visible patches; lightweight Mamba-2 decoder reads visible + learned mask tokens; predict raw signal at masked positions; loss = L1 + 0.3·MR-STFT log-mag on masked positions only | reference |
| G1 | **AR-causal-aligned** | No masking. Encoder is *unidirectional* (forward Mamba-2 only — the BiMamba backward stream is removed for this cell, since AR aligns with one scan direction). At each token position, predict the *next* token's representation (or the next raw sample window, see implementation pointers) using a small causal head. Loss = L1 + 0.3·MR-STFT log-mag on next-token prediction; computed per token | ~0.7× (no decoder, but per-token loss) |
| G2 | **MAR-bidirectional-masked-AR with diffusion head** | Random 50 % patch mask. Encoder is bidirectional (the §4.2 default BiMamba). At each masked position, the encoder's contextual representation is fed into a small (3-layer MLP, hidden 1024) **diffusion-loss head** that predicts the raw signal at that position via the EDM-style noise-conditioning loss (sample σ from log-normal, add σ·ε to the target, predict ε; reference: [MAR §3.2 + Appendix A](https://arxiv.org/abs/2406.11838)). 50 noise steps during training; no decoder transformer | ~1.4× (diffusion head sampling) |

**Why G2's masking ratio is the same as G0** (50 %): isolating the
*paradigm* from the *masking ratio* is the entire point. exp10
(rewritten as mask × ratio matrix on 2026-05-03) sweeps mask ratio after
exp17 picks the paradigm.

**Why no MAP-hybrid as a fourth cell.** The MAP-hybrid (per-row local MAE
+ global AR across rows) is conceptually appealing but its 2D-row
formulation does not transfer cleanly to single-channel 1D EEG: there is
no "row" dimension, only time. A 1D translation of MAP (e.g. local-window
MAE inside each ~250 ms block + AR across blocks) is essentially G2 with
a structured masking pattern; we treat it as an exp10 variant rather than
a separate G3 to keep this experiment focused.

## Controls (the §3 matrix)

|                              | EEG signal | matched-noise twin |
| ---------------------------- | ---------- | ------------------ |
| G0 MAE-bidirectional         | ✓          | ✓                  |
| G1 AR-causal-aligned         | ✓          | ✓                  |
| G2 MAR + diffusion head      | ✓          | ✓                  |

The matched-noise twin for G2 specifically: a diffusion-loss head trained
on Gaussian noise will achieve a non-trivial loss at every noise level
(the optimal ε-prediction is just zero — the noise *is* the noise).
What matters is whether the encoder *representation* (mean-pooled output)
develops downstream-useful structure. If it does on Gaussian noise, the
gain is from the augmentation pipeline, not the EEG content.

## Held constant

- Pretraining data: 100h HBN-EEG subset (per [`mini_experiments.md` §4.1](../../mini_experiments.md#41-pretraining-corpus)),
  500 Hz native (the minimum-offline preprocessing path).
- Frontend: F0 vanilla strided conv (the §4.2 default; if exp02 has settled
  before exp17 launches, swap in the exp02 winner).
- Backbone: bidirectional Mamba-2 for G0/G2; **unidirectional forward
  Mamba-2 for G1** (the only difference, mandated by AR's scan-alignment
  requirement). Width and layer count match across all three (6 layers,
  d=256, state N=64).
- Bottleneck: continuous (no quantization). exp11 explores quantization
  separately.
- Reconstruction loss: L1 + 0.3·MR-STFT log-magnitude on masked positions
  for G0; same per-token for G1; diffusion ε-prediction for G2 (different
  loss family by definition — what is held constant is the underlying
  raw-signal target, not the loss function).
- Optimiser: AdamW, β=(0.9, 0.95), wd=0.05, cosine schedule with 5 %
  warmup. LR swept {1e-4, 3e-4, 1e-3} per cell — per
  [`methodology.md` §4](../../methodology.md#4-hyperparameter-transfer-when-small-scale-tuning-is-trustworthy),
  LR is the one HP that doesn't transfer between data recipes; we re-tune
  it per generative paradigm because the loss function changes.
- Compute budget per cell: 1.5 hours of H100 wall-clock (matched, not
  matched-FLOP — G2 has the diffusion sampling overhead inside the budget).
- Eval suite per [`mini_experiments.md` §4.3](../../mini_experiments.md#43-evaluation-suite-for-every-experiment-unless-overridden):
  primary = HBN CBCL externalizing R² (A.1a) + HBN 6-task BAC (A.2) +
  k-NN top-1 (A.3); secondary = TUAB AUROC + TUEV BAC/WF1 (when TUH NEDC
  access lands); plus the §4.3 label-free monitors (encoder feature
  std/absmax/rank, predict-site probe, predict-subject k-NN).

## Decision rule

For each variant V relative to the G0 MAE baseline:

- **Strict win**: V's mean HBN 6-task BAC exceeds G0's mean by ≥ 2 percentage
  points with non-overlapping 95 % bootstrap CIs *and* the matched-noise
  twin shows no improvement (paired sign-flip permutation test
  p > 0.10 on the noise side) *and* the predict-site probe accuracy at
  end-of-training is no higher than G0's (we don't want a paradigm that
  wins HBN by learning site fingerprint).
- **Weak win**: ≥ 1 pp improvement on HBN 6-task BAC with overlapping CIs
  but paired permutation test p < 0.05.
- **Tie**: TOST equivalence within ε = 1 pp on HBN 6-task BAC. **In a tie,
  the pre-existing default (G0 MAE) is kept** — the burden of proof is on
  the new paradigm, not on the established baseline.
- **Loss**: V's mean HBN 6-task BAC is below G0's by ≥ 1 pp with p < 0.05.

Three paradigm-specific criteria:

- **Stability**: each variant must complete all 5 seeds without divergence.
  G2 specifically: the diffusion head's loss must not collapse to zero
  (which would indicate the encoder is producing constant outputs that the
  diffusion head trivially denoises).
- **Throughput sanity**: each variant must achieve ≥ 0.5× G0's tokens-per-
  second-per-H100 throughput. A paradigm that wins by 1 pp at 5× the
  compute cost is not the right choice for a Phase-4 headline run.
- **Encoder-feature-rank floor**: end-of-training encoder feature
  covariance rank must remain > 0.5 × feature_dim per
  [`methodology.md` §6.1](../../methodology.md#61-encoder-feature-health).
  This catches dimensional collapse — a pathology that vanilla MAE is
  immune to (the per-position L1 loss prevents collapse) but that AR and
  MAR can in principle suffer.

## Pre-registered predictions

| Variant | Prediction HBN 6-task BAC | Reasoning |
| ------- | ------------------------- | --------- |
| G0 MAE-bidirectional | reference (≈ 35–40 % expected, given exp01's random-init linear-probe floor pattern) | The default, anchored by §4.2; consistent with EEG-FM literature numbers at this scale. |
| G1 AR-causal-aligned | **strict win, ~+2–4 pp**; possible **weak loss** on phase-locking-value (PLV) eval | MAP and AR-Mamba both report decisive AR > MAE for Mamba in vision; the scan-alignment mechanism transfers cleanly to 1D. The PLV caveat: causal AR doesn't see future context, so phase reconstruction is degraded. |
| G2 MAR + diffusion head | **strict win, ~+3–5 pp**; **best PLV reconstruction** of all three | MAR combines bidirectionality (PLV-friendly) with the AR-style per-token prediction structure that matches the Mamba scan, plus the diffusion head provides a strong continuous-target loss without VQ overhead. The most expensive cell but expected to be the absolute winner. |

The honest expected outcome: **G2 MAR wins on the headline metric;
G1 AR is the cost-effective second-best and the right choice if compute
matters; G0 MAE is the default the rest of the field uses and probably
the right backstop if both G1 and G2 have stability issues.** The
anti-prediction: the MAP/AR-Mamba mechanism does *not* transfer, G0 wins,
and the rest of the spec stays anchored as written. We assign this
anti-prediction ~25 % probability — the mechanism is well-attested but
biosignal SSL has a long history of vision-derived intuitions failing to
transfer.

## Implementation pointers

- **G0 MAE-bidirectional**: identical to the exp04 S0 baseline. Encoder
  processes only visible patches (asymmetric, like image MAE); decoder is
  a 2-layer Mamba-2 block reading visible + learned mask tokens; predicts
  raw signal at masked positions. Reuse the same code path as exp04 S0.
- **G1 AR-causal-aligned**: replace the bidirectional Mamba-2 block (which
  is `forward + reversed` summed) with the **forward stream only** in the
  encoder. No mask token, no decoder. After the encoder, a small linear
  head predicts the next token's raw-signal patch from each position's
  contextual representation. Loss is L1 + 0.3·MR-STFT log-mag computed
  per position over the predicted-next-token vs ground-truth-next-token
  pair. Notes: (a) the *first* token has no context to predict from and
  is excluded from the loss; (b) the *last* token's prediction is also
  excluded since there's no ground truth next token in the window;
  (c) the §4.2 default 50 % mask ratio is irrelevant here because AR
  doesn't mask. The compute saving from removing the decoder partly
  offsets the cost of per-token loss vs MAE's per-masked-token loss.
- **G2 MAR + diffusion head**: same encoder as G0 (bidirectional Mamba-2,
  full backbone). Same 50 % random patch mask. **Replace the decoder
  Mamba-2 block** with a small (3-layer MLP, hidden 1024, GeLU) diffusion-
  loss head that takes the encoder's representation at masked positions
  (no learned mask token — the MAR design uses the encoder's contextual
  representation directly) plus a noise level σ embedding, and predicts
  the noise ε added to the ground-truth raw signal at that position.
  Sample σ from log-normal `LogNormal(P_mean=-1.2, P_std=1.2)` per
  position per training step (the EDM defaults). Use 50 noise steps for
  inference (only matters if we ever want to *generate* — for the
  representation-learning use case, only the loss matters). Reference:
  [MAR §3.2 + Appendix A](https://arxiv.org/abs/2406.11838),
  [EDM (Karras et al. 2022)](https://arxiv.org/abs/2206.00364).
- **Mamba-2 backbone**: [`state-spaces/mamba`](https://github.com/state-spaces/mamba)
  via the official Triton kernel. Bidirectional = forward Mamba +
  backward Mamba on reversed input, summed (FEMBA-style). Unidirectional
  for G1 = forward only.
- **Iso-data, not iso-step**: each cell trained on the same number of
  total tokens-seen (≈ 35M after iid expansion). G0 and G2 see the same
  number of mask predictions per token; G1 sees one prediction per token.
  Cell-specific batch size is tuned so wall-clock per cell is matched.

## Output

`mini_experiments/17_generative_paradigm/results.md` containing:

1. The 3 × 2 (× 5 seed) results table on HBN CBCL externalizing R²,
   HBN 6-task BAC, HBN attention regression R², k-NN top-1, predict-site
   probe accuracy at end-of-training, and (when TUH lands) TUAB AUROC +
   TUEV BAC/WF1.
2. PLV reconstruction quality per variant — the structural test of phase
   preservation; G1 (causal) is expected to underperform.
3. Encoder feature health per variant (std stable, rank > 0.5×dim,
   absmax/std bounded).
4. Wall-clock cost per training step per variant.
5. Throughput sanity table (tokens/s/H100) per variant.
6. The chosen generative paradigm with explicit decision-rule
   justification — and, if a non-G0 paradigm wins, an explicit list of
   downstream experiments that need re-anchoring (most importantly
   exp04, but also exp10's masking-strategy variants that don't apply
   to AR).

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| MAR's diffusion-loss head requires per-step noise sampling, blowing up wall-clock | Use only 50 noise levels (vs MAR paper's default 1000); EDM showed this is sufficient for representation learning. |
| G1 AR-causal-aligned underperforms G0 because exp03's chosen backbone (which may be bidirectional Mamba-2) inherently expects bidirectional context — and G1 forces unidirectional | This is part of the test: if scan-alignment matters, removing the backward stream should *help*, not hurt. If G1 loses to G0, it's evidence that the bidirectional context is doing real work and the MAP transfer hypothesis fails. |
| Stability issues in G2 (diffusion head loss collapse) | Add gradient clipping at norm 1.0; warm-start the diffusion head with 500 steps of pure-MSE training before introducing the noise-conditioning loss; if collapse persists, reduce the number of noise levels to 25. |
| All three cells tie on HBN 6-task BAC (eval too easy at this scale) | Add HBN CBCL externalizing R² as the headline metric (it's a continuous-label regression with much higher discrimination power than the 6-class classification). Per the 2026-05-03 §4.3 update, regression on CBCL is now A.1a (the primary metric). |
| The §4.2 default frontend (F0 vanilla conv) handicaps G2 (which needs richer encoder features for the diffusion head to be effective) | Acknowledged as a coupling between exp02 and exp17. If exp02 has chosen a richer frontend (F2 SincNet or F4 Gabor) before exp17 launches, swap it in. Otherwise, run exp17 once with F0 (this README) and once with the exp02 winner if F0 isn't it. |
| MAP-hybrid is the actual best paradigm but isn't tested here (we deferred it as an exp10 variant) | If both G1 and G2 strict-win, add a post-hoc G3 = "MAR + structured-block masking" cell that approximates the MAP recipe in 1D. Compute budget: ~3 H100-hours. |

## What gets carried forward

The single chosen generative paradigm becomes the **anchor** for every
downstream experiment. Specifically:

- **exp04 (SSL framework)** is re-anchored: instead of "denoised-MAE vs
  VICReg vs TF-C vs EEGDM vs MER+NSP" being measured against MAE-baseline
  S0, the framework comparison is measured against the **exp17 winner**
  as the new S0 baseline. If G2 MAR wins, then S0 becomes "MAR-baseline";
  S1 becomes "MAR + denoised target (the MAR-of-clean-signal)"; S2 VICReg
  becomes "joint-embedding head added to MAR encoder"; etc.
- **exp10 (mask × ratio matrix)** drops the masking variants that don't
  apply to G1 (AR has no masking); for G2, the masking strategies are
  re-tested under the diffusion-head setup.
- **exp18 (reconstruction target)** assumes a paradigm that *has* a
  reconstruction target; if G1 wins (no decoder, no reconstruction
  target — just next-token prediction), then exp18's target ablation
  becomes "next-token prediction target = raw vs latent vs codec-RVQ vs
  HuBERT-cluster" — same conceptual axis, different mechanics.
- **exp19 (decoder design)** is paradigm-dependent: G0 has a decoder, G1
  has none, G2 has a small MLP head (which is the "decoder" in MAR
  terminology). exp19 sweeps the decoder appropriate to the chosen
  paradigm.
- **exp14 (context length)** is paradigm-dependent for stability — long-
  context AR is known to be more stable than long-context MAE in the
  Mamba family (the official MAP code reports this), so the C3 30-second
  cell may be feasible for AR/MAR even if it has stability issues for
  MAE.
