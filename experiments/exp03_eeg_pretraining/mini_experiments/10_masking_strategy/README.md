# exp03 / mini-experiment 10 — Masking strategy × ratio

> **Status:** planned (rewritten 2026-05-03 — see scope-expansion note below)
>
> **Cross-reference:** [EEG2Rep SSP](https://arxiv.org/html/2402.17772v2),
> [NeurIPT AAMP](https://arxiv.org/abs/2510.16548),
> [I-JEPA multi-block masking](https://arxiv.org/abs/2301.08243),
> [wav2vec 2.0 span masking (Baevski 2020, arXiv 2006.11477)](https://arxiv.org/abs/2006.11477),
> [VideoMAE 90 % masking (Tong 2022, arXiv 2203.12602)](https://arxiv.org/abs/2203.12602),
> [SAMBA TSR masking (Hong 2025, arXiv 2511.18571)](https://arxiv.org/abs/2511.18571)
>
> **Compute budget:** 12 H100-hours (16-cell strategy × ratio screening +
> top-3 confirmation × 5 seeds × matched-noise twin)
>
> **Gates:** none downstream

## Question

Within the chosen generative paradigm (the exp17 winner — for MAE-family
paradigms only; AR has no masking and is exempt from this experiment),
**which combination of masking strategy and masking ratio** produces the
best representation? The cross-modal evidence on both axes is contested:

- **Strategy axis**: random patch (the vanilla baseline), block masking,
  semantic-subsequence-preserving (SSP) from EEG2Rep, multi-block from
  I-JEPA, amplitude-aware (AAMP) from NeurIPT, **span masking from
  wav2vec 2.0** (added 2026-05-03), and **TSR (Temporal-Semantic
  Random) from SAMBA** (added 2026-05-03).
- **Ratio axis**: 50 % is the §4.2 default (and what CBraMod, CoMET,
  SAMBA, EEG2Rep use); 60 % is what FEMBA / TimeMAE use; 75 % is what
  vision MAE and ST-EEGFormer (NeurIPS 2025 EEG Challenge winner) use;
  85–90 % is what VideoMAE uses for high-redundancy video and what the
  Vision-MAE deep-research subagent predicted optimal for 500 Hz EEG.

> **2026-05-03 scope expansion.** The original exp10 tested 5 masking
> strategies, all at the 50 % default ratio — implicitly assuming the
> ratio is settled. The deep-research refresh found that **the EEG-FM
> literature is genuinely split on mask ratio**, with 50 / 60 / 75 / 85
> % all defended by different SOTA papers (CBraMod, FEMBA, ST-EEGFormer,
> MTSMAE respectively). Worse, the Vision-MAE subagent's analysis of
> VideoMAE shows that **mask ratio scales with temporal redundancy** —
> images need 75 %, video 90–95 %, EEG at 500 Hz with 16 ms tokens is
> *more* redundant than video → predicted optimal is 85–90 %, not the
> 50 % default. exp10 is therefore rewritten as a 2-axis (strategy ×
> ratio) ablation matrix to test both questions in a single experiment,
> with screening + confirmation discipline to keep the cell count
> bounded.

## Why it matters

In MAE-style training, the masking strategy chooses which parts of the
input the model is forced to predict from context. For natural images,
random patch masking at 75 % works because images have high spatial
redundancy; nearby pixels are highly predictable from each other. EEG has
the analogous redundancy in time, but with two specific structures that
random masking gets wrong:

- **Easy interpolation regions**: long stretches of stable rhythmic
  activity (alpha rhythm during eyes-closed rest) can be reconstructed by
  trivially interpolating between unmasked patches. The model gets a
  beautiful loss decrease without learning anything about the actual
  signal-defining events. EEG2Rep's SSP masking
  ([arXiv:2402.17772](https://arxiv.org/html/2402.17772v2)) directly
  addresses this by deciding which segments to *preserve* rather than
  which to mask.
- **High-amplitude artifact regions**: random masking masks artifact
  windows as often as signal windows, but the model's reconstruction loss
  is then dominated by trying to predict the artifact (which is by
  definition unpredictable). NeurIPT's AAMP
  ([arXiv:2510.16548](https://arxiv.org/abs/2510.16548)) preferentially
  masks high-amplitude regions, forcing the model to predict them from
  surrounding context — which means the model learns to characterise the
  *neural state in which artifacts happen* rather than the artifacts
  themselves.

I-JEPA's multi-block masking is a third pattern: rather than scattered
patches, mask 4 large contiguous blocks. This is what gave I-JEPA its
~20 pp linear-probe gain on ImageNet and is included as a control.

## Variants — strategy × ratio matrix

### Strategy axis (4 strategies + 1 dropped)

| Code | Strategy | Masking algorithm |
| ---- | -------- | ----------------- |
| KS-RND | Random patch (vanilla MAE) | uniform sample of `r` % of patches independently |
| KS-SPAN | Span masking (wav2vec 2.0 style, added 2026-05-03) | sample mask-start positions with probability `p = 0.065` per token; each chosen start grows into a span of `M = 10` consecutive masked tokens (≈ 160 ms at 16 ms/token); spans can overlap. The effective mask ratio is determined by `p × M` (we tune `p` per cell to hit the target ratio `r`) |
| KS-MBL | Multi-block (I-JEPA style) | mask 4 large contiguous blocks totalling `r` % of tokens; the unmasked tokens form 1 large context block |
| KS-AAMP | Amplitude-aware (NeurIPT) | sample a percentile ξ ∈ [50 %, 90 %] per batch; mask the patches whose mean absolute amplitude is closest to that percentile, until `r` % is reached |
| ~~KS-SSP~~ | ~~SSP (EEG2Rep)~~ | **dropped 2026-05-03**: SSP is partially redundant with KS-MBL (both preserve large contiguous unmasked blocks); to keep the matrix bounded we drop SSP. If KS-MBL strict-wins, a follow-up ablation can re-introduce SSP as a variant of multi-block. |

### Ratio axis (4 ratios)

| Code | Mask ratio | Rationale |
| ---- | ---------- | --------- |
| KR-50 | 50 % | the §4.2 default; CBraMod / CoMET / SAMBA / EEG2Rep camp |
| KR-65 | 65 % | FEMBA (60 %) ≈ TimeMAE (60 %); modal mid-point |
| KR-75 | 75 % | vision MAE / ST-EEGFormer NeurIPS 2025 EEG Challenge winner camp |
| KR-85 | 85 % | VideoMAE / MTSMAE camp; the Vision-MAE subagent's EEG-redundancy prediction |

### Cells (the 4 × 4 = 16-cell matrix)

|         | KR-50 (50 %) | KR-65 (65 %) | KR-75 (75 %) | KR-85 (85 %) |
| ------- | ------------ | ------------ | ------------ | ------------ |
| **KS-RND** | KS-RND × KR-50 (= §4.2 default) | KS-RND × KR-65 | KS-RND × KR-75 | KS-RND × KR-85 |
| **KS-SPAN** | KS-SPAN × KR-50 | KS-SPAN × KR-65 | KS-SPAN × KR-75 | KS-SPAN × KR-85 |
| **KS-MBL** | KS-MBL × KR-50 | KS-MBL × KR-65 | KS-MBL × KR-75 | KS-MBL × KR-85 |
| **KS-AAMP** | KS-AAMP × KR-50 | KS-AAMP × KR-65 | KS-AAMP × KR-75 | KS-AAMP × KR-85 |

To keep the budget bounded we use the **screening + confirmation
protocol** (per [`methodology.md` §3](../../methodology.md#3-how-to-design-one-ablation-the-matrix-shape)):

- **Stage 1 (screening)**: all 16 cells × 1 seed × ~30 minutes
  = 8 H100-hours wall-clock. Rank by HBN 6-task BAC.
- **Stage 2 (confirmation)**: top-3 cells from screening × matched-noise
  twin × 5 seeds × ~30 minutes = ~5 H100-hours. Apply the standard
  decision rule.

Total compute: ~13 H100-hours (rounded down to 12 in the budget header).

The pre-2026-05-03 cells K0/K1/K2/K3/K4 map to: K0 = KS-RND × KR-50;
K1 = (KS-MBL or block) × KR-50 (we now exclude pure block masking
because it strictly underperforms KS-MBL per I-JEPA's published
ablation); K2 = KS-MBL × KR-50 (closest mapping); K3 = KS-MBL × KR-65;
K4 = KS-AAMP × KR-50.

## Controls

|                                | EEG signal | matched-noise twin |
| ------------------------------ | ---------- | ------------------ |
| Stage-1 screening: all 16 cells | ✓ (1 seed) | — (no twin in screening) |
| Stage-2 confirmation: top-3 cells | ✓ (5 seeds) | ✓ (5 seeds)       |

For matched-noise twin: AAMP applied to Gaussian noise will preferentially
mask high-amplitude noise samples, but those have no special meaning in
Gaussian noise (every sample's amplitude is just a draw from N(0,1)). So
AAMP on noise should perform identically to random mask on noise, which
is the right sanity check — if AAMP on noise improves over KS-RND × KR-50
on noise, the AAMP gain is incidental to mask placement statistics, not to
amplitude-aware semantics.

## Held constant

- Frontend: exp02 winner.
- Backbone: exp03 winner.
- SSL framework: exp04 winner.
- Bottleneck: continuous.
- Reconstruction loss: exp06 winner.
- Phase loss: exp07 winner.
- Target: exp08 winner.
- Optimiser: AdamW, LR carried forward.
- Pretraining duration: 8 epochs.

## Decision rule

Same as prior experiments:

- Strict win = ≥ 1 pp HBN 6-task BAC (per §4.3 Protocol A.2), non-overlapping CIs, noise-twin flat.
- Weak win = ≥ 0.5 pp with paired permutation p < 0.05.
- Tie = TOST equivalence within ε = 0.5 pp.

One masking-specific criterion:

- **Reconstruction loss must remain non-trivial**: a masking strategy that
  is too easy (low mask ratio, mask only obvious noise) will drive
  reconstruction loss to zero. The model then learns nothing. The minimum
  acceptable end-of-training reconstruction loss across variants must be
  > 0.1 (in normalised units after z-score). If a variant's loss is
  lower than this, it's disqualified — it solved a trivial task.

## Pre-registered predictions (revised 2026-05-03)

The 16-cell prediction grid (HBN 6-task BAC, expressed as Δ pp over
KS-RND × KR-50 the §4.2 default):

|         | KR-50 | KR-65 | KR-75 | KR-85 |
| ------- | ----- | ----- | ----- | ----- |
| **KS-RND**  | reference | weak win, +0.5 | weak win, +1.0 | tied (random masking degrades at extreme ratios because too few unmasked tokens to predict from) |
| **KS-SPAN** | weak win, +1 (the wav2vec finding: span > random for correlated 1D signals) | strict win, +1.5 | **strict win, +2** (predicted Phase-2 winner) | weak win, +1 |
| **KS-MBL**  | strict win, +1.5 (the I-JEPA finding) | strict win, +2 | strict win, +2.5 | weak win, +1.5 (multi-block at 85 % is a very hard task) |
| **KS-AAMP** | strict win, +1 (low-SNR-sensitive eval will see +2) | weak win, +1.5 | weak win, +1.5 | tied (AAMP at 85 % preferentially masks high-amplitude content; remaining low-amplitude context isn't enough to predict from) |

The honest expected outcome: **KS-MBL × KR-75 wins by ~2.5 pp HBN 6-task
BAC over the §4.2 default**, with KS-SPAN × KR-75 as the close
runner-up; KS-AAMP wins on low-SNR-sensitive evals
(HBN-Artifact-Synth from exp09, or TUAR when TUH lands). The
contrarian outcome would be KR-85 winning across all strategies (the
high-redundancy prediction), in which case the Vision-MAE subagent's
extrapolation from VideoMAE (90 % for video) transferred faithfully and
we should consider an even higher mask ratio (90 %) in a follow-up.

## Implementation pointers

- KS-RND: trivial. PyTorch boolean tensor of shape
  `(batch, num_patches)` with `~r %` `True` entries.
- KS-SPAN (added 2026-05-03): per [wav2vec 2.0 §3.1](https://arxiv.org/abs/2006.11477).
  At each token position, sample `Bernoulli(p)` to start a span; each
  start grows into `M = 10` consecutive masked tokens. Spans can overlap
  (overlapping spans count as one mask). Tune `p` per cell to hit the
  target ratio: `p ≈ r / M` (e.g. `p = 0.05` for 50 % at M=10; `p = 0.085`
  for 85 %).
- KS-MBL: pick 4 starting positions with at least `r/4` separation; each
  grows into a contiguous block of `r/4` tokens; the unmasked context
  spans the remaining ~25 % of the window. Per [I-JEPA §3.2](https://arxiv.org/abs/2301.08243).
- KS-AAMP: compute per-patch RMS amplitude. Sample percentile ξ ∈ [0.5, 0.9]
  per batch. Mask patches whose RMS is closest to that percentile (within
  ±5 % of the targeted percentile), until `r %` is masked. Cost: one extra
  `torch.quantile` per batch. Reference at the NeurIPT paper.
- All: ensure mask is recomputed per training step (i.e. don't cache).

## Output

`mini_experiments/10_masking_strategy/results.md` containing:

1. 5 × 2 (× 3 seed) results table.
2. End-of-training reconstruction loss per variant.
3. The chosen masking variant with decision-rule justification.
4. If K5 (AAMP + multi-block) is added, its result.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| K4 AAMP changes the effective mask ratio per batch (because high-amplitude patches cluster in some recordings) | Add a normalisation step: after selecting amplitude-targeted patches, randomly subsample to exactly the target mask fraction. |
| K3 multi-block's higher mask ratio (60 %) gives a different effective training signal that's hard to compare iso-anything | This is the I-JEPA paper's choice; we keep 60 % to match. The decision rule already accounts for this by comparing on downstream metrics, not on loss. |
| K2 SSP preserves the wrong subsequences (e.g. picks the high-amplitude artifact regions to preserve) | Add an amplitude prior to the SSP starting-point selection — bias starting points toward moderate-amplitude regions. (This is essentially "AAMP-aware SSP", an organic post-hoc combination.) |
| All masking strategies tie because the dataset is too small | Re-run with 300h of HBN-EEG instead of 100h (still well within the ~3000h available; preprocessing already cached in S3). |

## What gets carried forward

The single chosen masking strategy. The mask hyperparameters (ratio,
block count, AAMP percentile range) are frozen for every later
experiment.
