# exp03 / mini-experiment 18 — Reconstruction target

> **Status:** planned (added 2026-05-03 from the deep-research design refresh)
>
> **Cross-reference:** [`mini_experiments.md` §4.2 default architecture](../../mini_experiments.md#42-default-architecture-for-any-axis-not-under-test),
> [exp17 generative paradigm](../17_generative_paradigm/),
> [exp08 denoising target](../08_denoising_target/) (orthogonal: that ablates the *signal source* of the target; this ablates the *representation space* the target lives in),
> [I-JEPA (Assran et al., CVPR 2023, arXiv 2301.08243)](https://arxiv.org/abs/2301.08243),
> [EEG2Rep (Foumani et al. 2024, arXiv 2402.17772)](https://arxiv.org/abs/2402.17772),
> [MAGE (Li et al., CVPR 2023, arXiv 2211.09117)](https://arxiv.org/abs/2211.09117),
> [HuBERT (Hsu et al. 2021, arXiv 2106.07447)](https://arxiv.org/abs/2106.07447),
> [BioCodec (2025, github.com/usc-sail/BioCodec)](https://github.com/usc-sail/BioCodec)
>
> **Compute budget:** 14 H100-hours (6 target variants × 2 control columns
> × 5 seeds = 60 cells × 14 min average — most variants share encoder
> compute; the EMA-target and codec-RVQ cells are slightly more expensive)
>
> **Gates:** experiment 19 (decoder design depends on what's being decoded);
> experiment 06 (reconstruction loss; the right loss family depends on the
> target's representation space)

## Question

Holding the §4.2 architecture and the exp17 generative-paradigm winner
fixed, **what does the model predict at masked positions** (or at next-
token positions, if exp17 chose AR)? Six candidate target representation
spaces, each with substantial published evidence of being the right
choice in a different domain:

1. raw signal (the §4.2 default; what every EEG-FM does);
2. per-token-normalised raw signal (what MAE He et al. 2022 actually used,
   not raw);
3. latent representations from an EMA-target encoder (the I-JEPA / V-JEPA
   / EEG2Rep "predict in representation space" recipe);
4. discrete codec-RVQ tokens from BioCodec (pretrained on TUH-EEG, open
   source, never used as an SSL target);
5. HuBERT-style iterative-k-means cluster IDs (the mechanism that made
   HuBERT outperform wav2vec 2.0 in speech);
6. raw signal with a sparsity regularisation on the encoder output (the
   classical Olshausen & Field 1996 prior, never tested as an SSL
   reconstruction target for biosignals).

## Why it matters

Across speech (HuBERT > wav2vec 2.0 because of cluster targets), vision
(I-JEPA > MAE because of latent targets; MAGE > MAE by 6.7 pp linear-probe
because of semantic tokens), and EEG specifically (EEG2Rep > MAEEG by ~5 %
because of latent-space prediction; LaBraM > BENDR because of codebook
prediction targets), **the target representation space is the single
biggest lever for downstream representation quality**. Bigger than the
encoder architecture (which is the question of exp02/03), bigger than the
masking strategy (exp10), bigger than the loss function (exp06).

The reason is mechanical: the encoder's job during pretraining is to
produce features from which the *target* can be recovered. If the target
lives in a noise-dominated raw-signal space, the encoder is rewarded for
modelling noise. If the target lives in a denoised / quantised / latent
space, the encoder is rewarded for modelling structure. EEG's low SNR
makes this distinction load-bearing — much more so than for vision
(where pixels are essentially noise-free) or speech (where the noise floor
is well-characterised and the spectral content is well-known).

This experiment is **orthogonal to exp08 (denoised target)**:

- exp08 asks "**what signal** do we treat as the ground truth" — raw EEG
  vs bandpass-filtered vs ICA-cleaned vs wavelet-denoised vs IC-U-Net-
  cleaned. All variants live in raw-signal space; what changes is which
  preprocessing chain produces the target.
- exp18 (this) asks "**what representation space** does the target live
  in" — raw signal (any cleanliness) vs latent representations vs
  codec-RVQ tokens vs k-means cluster IDs. The signal source is held
  fixed; what changes is how the signal is encoded into the prediction
  target.

The two interact (one could test "ICA-cleaned signal in latent space" as
a 5×6 = 30-cell matrix), but at this scale we hold one fixed (raw signal
from exp08's chosen recipe) and sweep the other.

## Variants

The encoder, generative paradigm (exp17 winner), masking strategy, and
input signal are all fixed across variants. Only the prediction target
differs.

| Code | Variant | Target representation space | Loss family | Cost vs T0 |
| ---- | ------- | --------------------------- | ----------- | ---------- |
| TR0 | **Raw signal** (the §4.2 default) | raw 16ms-token at masked positions | L1 + 0.3·MR-STFT log-mag | reference |
| TR1 | **Per-token-normalised raw** (the actual MAE He 2022 recipe) | per-token mean-and-std-normalised raw at masked positions; mean+std are *predicted alongside* by a small auxiliary head so the normalisation is learned, not pre-applied | L1 on normalised raw + L1 on (mean, std) | ≈ 1.0× |
| TR2 | **Latent (EMA-target encoder)** | output of a momentum-updated copy of the encoder applied to the *unmasked full input* at the masked positions; no decoder, only a small predictor head; loss in latent space | smooth-L1 on latent representations | ≈ 1.5× (extra forward pass through EMA encoder) |
| TR3 | **BioCodec RVQ tokens** | discrete RVQ codebook IDs from the BioCodec encoder applied to the raw input at the masked positions; cross-entropy loss over the 1024-entry codebook | masked cross-entropy over codec vocab | ≈ 1.1× (codec forward is one-time precomputable) |
| TR4 | **HuBERT-style iterative k-means** | k-means cluster ID (k = 500) at the masked positions; targets refreshed every 100k steps via re-clustering on the latest encoder's intermediate-layer features | masked cross-entropy over k cluster IDs | ≈ 1.05× (re-clustering pass is amortised) |
| TR5 | **Raw signal + sparsity reg** | raw signal at masked positions (same as TR0) plus an L1 sparsity penalty on the encoder's output activations | L1 + 0.3·MR-STFT + 0.01·\|\|h\|\|₁ | ≈ 1.0× |

**Why TR1 deserves its own cell** (instead of being a hyperparameter of
TR0). MAE (He 2022) Table 1d shows that *per-patch normalised* pixels are
+1.6 pp linear probe over raw pixels — a non-trivial gap. EEG amplitudes
have heavy-tailed inter-subject variation (artifacts, electrode contact)
that is much worse than image pixels' intensity variation; the
normalisation correction may help substantially more for EEG than for
images. TR1 tests this directly.

**Why TR2 doesn't violate the "no teacher / no EMA" constraint** in
[`mini_experiments.md` §1](../../mini_experiments.md#1-why-mini-experiments-at-all):
the constraint is against *separate teacher networks* (DINOv2-style EMA
teachers that have their own training objective). TR2's EMA-target is the
*same encoder*, just at a slightly delayed parameter snapshot. There is
only one set of trainable parameters; the EMA is a regularisation device,
not a separate model. EEG2Rep (the EEG-specific replication of I-JEPA)
uses exactly this construction without violating their stated "single-
encoder" claim.

**Why TR3 (BioCodec) is feasible despite the "no two-stage tokenizer
training" constraint**: BioCodec is **already pretrained on TUH-EEG and
open-sourced** at [`usc-sail/BioCodec`](https://github.com/usc-sail/BioCodec).
We are not training a new codec; we are using an existing public model as
a fixed feature extractor (the same way exp03 F3 uses Kymatio scattering
as a frozen frontend). No two-stage training happens in our pipeline.

**Why TR4 (HuBERT iterative clustering) is the speech-side analogue
worth testing**. wav2vec 2.0 (continuous targets, contrastive) was
decisively beaten by HuBERT (discrete cluster targets, masked CE) in
2021. The mechanism — iterative refinement of cluster targets via
re-clustering on the encoder's own intermediate features — has never been
applied to EEG. The TFM-Tokenizer paper (ICLR 2026) is the closest
analogue but its "tokenizer" is itself a learned model with its own
training objective; pure k-means clustering on existing encoder features
is one level simpler and avoids the two-stage training prohibition.

**Why TR5 (sparsity-regularised) is novel**. Olshausen & Field (1996)
showed that sparse-coding objectives produce V1-like simple-cell
receptive fields from natural images — the foundational result of
unsupervised representation learning. No EEG SSL paper has tested whether
a sparsity prior on the encoder output produces analogous "EEG simple
cells" (oscillation-tuned narrow-band detectors). Cheap to add as a
regularisation term; could be a free win if it works.

## Controls (the §3 matrix)

|                              | EEG signal | matched-noise twin |
| ---------------------------- | ---------- | ------------------ |
| TR0 raw signal               | ✓          | ✓                  |
| TR1 normalised raw           | ✓          | ✓                  |
| TR2 latent (EMA-target)      | ✓          | ✓                  |
| TR3 BioCodec RVQ             | ✓          | ✓                  |
| TR4 HuBERT k-means cluster   | ✓          | ✓                  |
| TR5 raw + sparsity reg       | ✓          | ✓                  |

The matched-noise twin for TR3 is informative in a specific way: BioCodec's
RVQ codebook was trained on EEG, so applying it to Gaussian noise will
mostly hit a small number of "noise-like" codes. If the EEG variant
strict-wins TR0 but the noise variant *also* improves over TR0-noise, the
gain is from the codec's noise-modelling capacity, not from EEG-specific
structure.

The matched-noise twin for TR4 (k-means clusters): on Gaussian noise,
k-means produces clusters that are essentially random partitions of
N(0, 1) space — the cluster IDs carry no meaningful information. The
encoder should fail to learn anything useful from predicting these.
A passing TR4 must show this.

## Held constant

- Pretraining data: 100h HBN-EEG subset, 500 Hz native, minimum-offline
  preprocessing.
- Frontend: exp02 winner.
- Backbone: exp03 winner.
- **Generative paradigm: exp17 winner** (this is the load-bearing
  dependency — if exp17 chose G1 AR, then "masked positions" becomes
  "next-token positions" and TR2's "EMA-target encoder applied to full
  input" becomes "EMA-target encoder applied to past tokens").
- Masking strategy: exp10 winner (or random 50 % default if exp10 hasn't
  settled).
- Bottleneck: continuous (exp11 explores quantization separately, but
  TR3's discrete-codec target is *not* a bottleneck quantization — it's a
  prediction-target quantization, which is a different mechanism).
- Reconstruction loss family: dictated by the target — L1 + MR-STFT for
  raw-space targets (TR0, TR1, TR5); smooth-L1 for latent (TR2); cross-
  entropy for discrete (TR3, TR4). exp06 ablates loss-within-family;
  exp18 ablates the family.
- Decoder: small (default 2-layer Mamba-2; exp19 sweeps decoder depth ×
  type after exp18 settles which target the decoder is producing).
- Optimiser: AdamW, LR swept {1e-4, 3e-4, 1e-3} per cell.
- Compute budget per cell: 1.5 hours of H100 wall-clock per seed.
- Eval suite per [§4.3](../../mini_experiments.md#43-evaluation-suite-for-every-experiment-unless-overridden).

## Decision rule

For each variant V relative to the TR0 raw baseline:

- **Strict win**: ≥ 2 pp HBN 6-task BAC over TR0, non-overlapping CIs,
  noise-twin flat, and (for TR2 specifically) no representation collapse
  (encoder feature rank > 0.5 × feature_dim).
- **Weak win**: ≥ 1 pp with paired permutation p < 0.05.
- **Tie**: TOST equivalence within ε = 1 pp. **In a tie, the simplest
  variant (TR0 raw) is kept** — operational simplicity is a tiebreaker.
- **Loss**: ≥ 1 pp below TR0 with p < 0.05.

Three target-specific criteria:

- **TR2 collapse check**: the EMA-target latent space can collapse to a
  single point (the trivial solution: encoder always outputs the same
  latent, EMA target always matches, loss → 0). Encoder feature
  covariance rank must remain > 0.5 × feature_dim. If it collapses, TR2
  is disqualified.
- **TR3 codebook coverage**: BioCodec's RVQ codebook has 1024 entries
  per layer × 4 layers = 4096 codes total. The model's prediction
  distribution must cover at least 25 % of the codebook by end-of-
  training (per the §4.4 codebook-collapse anti-shortcut from
  [`methodology.md` §6.1](../../methodology.md#61-encoder-feature-health)).
  If usage is below 25 %, the model has collapsed to a small subset of
  codes — the loss looks fine but the representation is brittle.
- **TR4 cluster stability**: the k-means re-clustering pass refreshes
  targets every 100k steps. Each re-clustering must produce clusters
  whose centroids have ≥ 0.7 cosine similarity to the previous
  iteration's centroids on average — otherwise the targets are drifting
  too fast and the model is chasing a moving goal. If similarity drops
  below 0.7, halt re-clustering and freeze the targets.

## Pre-registered predictions

| Variant | Prediction HBN 6-task BAC | Reasoning |
| ------- | ------------------------- | --------- |
| TR0 raw signal | reference | The §4.2 baseline; what every EEG-FM does. |
| TR1 normalised raw | **strict win, ~+1–2 pp** | MAE 2022 Table 1d shows +1.6 pp on ImageNet linear probe; EEG's heavy-tailed amplitudes make the per-token normalisation correction even more impactful. Cheapest possible improvement. |
| TR2 latent (EMA-target) | **strict win, ~+3–5 pp** | I-JEPA jumps 7.9 pp on ImageNet linear probe; EEG2Rep replicates this for EEG with a similar gap; the mechanism is well-attested. The matched-noise twin is the disambiguator: if TR2 also wins on noise, the gain is from the EMA regularisation, not from latent prediction *per se*. |
| TR3 BioCodec RVQ | **strict win, ~+2–4 pp** | LaBraM > BENDR by ~5 pp via codebook prediction; the same mechanism (discrete targets with codebook coverage as an implicit anti-collapse) should transfer. The risk is that BioCodec was pretrained on TUH (adult clinical) while we evaluate on HBN (pediatric); the cross-distribution gap may eat some of the gain. |
| TR4 HuBERT k-means | **weak win, ~+1–2 pp** | Should beat TR0 (HuBERT's mechanism is well-attested) but lose to TR3 and TR2 (the cluster targets are stale and noisy compared to a learned codec or latent representation). The honest expected role of TR4 is as a "minimum-viable discrete target" baseline. |
| TR5 raw + sparsity reg | **tied, possibly weak win ~+0.5 pp** on phase-sensitive evaluation | Sparsity prior may produce more interpretable encoder features (the original Olshausen finding) but is unlikely to move the headline metric significantly without a proper analysis of receptive-field structure. Worth running because it's nearly free. |

The honest expected ranking: **TR2 latent ≈ TR3 BioCodec > TR1 normalised
> TR4 HuBERT-cluster > TR0 raw > TR5 sparsity-regularised** on the
headline metric. The choice between TR2 and TR3 likely comes down to
operational considerations: TR2 has no external dependency (the EMA
target is computed inline), TR3 depends on BioCodec staying available
and behaving consistently across distribution shift.

## Implementation pointers

- **TR0 raw signal**: identical to the current default; no change.
- **TR1 normalised raw**: per-token compute mean-and-std on the unmasked
  ground truth at the masked position; subtract, divide; predict. The
  loss is L1 on the normalised raw. Add an auxiliary head that predicts
  (mean, std) directly so the normalisation is recoverable at inference.
  Cost: ~5 % extra parameters in the auxiliary head.
- **TR2 latent (EMA-target)**: maintain `encoder_target = EMA(encoder)`
  with momentum 0.996 (the standard I-JEPA value). At each step, run the
  encoder on the masked input (visible only); run the EMA-target encoder
  on the unmasked full input (no gradient); take a small predictor head
  (3-layer MLP, hidden 1024) from the encoder output at masked positions
  and predict the corresponding EMA-target output. Loss = smooth-L1 in
  latent space. Reference: [I-JEPA §3](https://arxiv.org/abs/2301.08243),
  [EEG2Rep §3](https://arxiv.org/abs/2402.17772).
- **TR3 BioCodec RVQ**: clone [`usc-sail/BioCodec`](https://github.com/usc-sail/BioCodec),
  load the pretrained checkpoint. Precompute the RVQ codebook IDs for
  every parquet shard (one-time, ~30 minutes per release on a single GPU,
  cache to disk). At each training step, look up the codebook IDs at the
  masked positions; the prediction head is a small linear classifier
  outputting (4 RVQ layers × 1024 codes) softmax distributions; loss is
  cross-entropy summed across the 4 layers.
- **TR4 HuBERT iterative k-means**: at training step 0, initialise k-means
  cluster IDs from the encoder's MFCC-equivalent features (per-token
  mel-spectrogram); train for 100k steps with these as targets; then
  re-cluster on the encoder's *intermediate-layer-3* features (the
  HuBERT prescription); train for another 100k steps with these new
  targets; iterate. Use `sklearn.cluster.MiniBatchKMeans(n_clusters=500,
  batch_size=10000)` for the clustering pass. Reference:
  [HuBERT Appendix B](https://arxiv.org/abs/2106.07447).
- **TR5 raw + sparsity reg**: identical to TR0 plus an L1 norm of the
  encoder's output added to the loss with weight 0.01. Sweep weight
  {0.001, 0.01, 0.1} as part of the LR sweep. Reference:
  [Olshausen & Field (1996, Nature)](https://doi.org/10.1038/381607a0).
- **Decoder shared across cells**: 2-layer bidirectional Mamba-2, d=256
  (the §4.2 default; exp19 sweeps decoder design after exp18 picks the
  target).

## Output

`mini_experiments/18_reconstruction_target/results.md` containing:

1. The 6 × 2 (× 5 seed) results table on HBN CBCL externalizing R²,
   HBN 6-task BAC, attention regression R², k-NN top-1, predict-site
   probe, and (when TUH lands) TUAB AUROC + TUEV BAC/WF1.
2. Codebook coverage trajectory (TR3 only) — fraction of codes used per
   100k steps.
3. K-means cluster stability trajectory (TR4 only) — cosine similarity
   of consecutive cluster-centroid sets.
4. EMA-target latent rank (TR2 only) — must remain > 0.5 × feature_dim.
5. Sparsity-induced feature analysis (TR5 only) — fraction of zero-or-
   near-zero encoder activations, plus a visualisation of the
   "EEG-equivalent simple cells" if any emerge.
6. Wall-clock cost per training step per variant.
7. The chosen reconstruction target with explicit decision-rule
   justification.

## Risks

| Risk | Mitigation |
| ---- | ---------- |
| TR2 EMA-target collapses to a constant (all encoder outputs identical, EMA target identical, loss → 0) | Add the I-JEPA stability protocol: a variance-preserving regulariser on the encoder output (push variance toward 1.0 per dim); momentum 0.996 → 0.999 schedule across training; gradient clipping at norm 1.0. If collapse persists, halve the predictor head's hidden width to bottleneck the trivial-prediction path. |
| TR3 BioCodec is unavailable or its checkpoint format is incompatible with our infrastructure | Falls back to a Soundstream/Encodec-style codec trained on HBN-EEG specifically. This adds a one-time codec-training cost (~6 H100-hours via [`facebookresearch/encodec`](https://github.com/facebookresearch/encodec) on our preprocessed parquet) but removes the external dependency. Document as TR3-fallback. |
| TR4 HuBERT-style iterative clustering produces unstable targets that the model can't catch up to | Reduce re-clustering frequency from every 100k to every 250k steps; if still unstable, fix the targets after iteration 1 and don't refresh further. (This is the HuBERT-Base recipe; the paper used 2 iterations not infinite.) |
| TR5 sparsity weight sweep makes the experiment expensive | Limit to 3 weights {0.001, 0.01, 0.1} × 1 seed for the sweep; pick the best, then run 5 seeds at that weight. |
| All variants tie because the eval is dominated by features that survive any reasonable target choice | Switch the headline metric to HBN attention regression R² (Protocol A.1b) — continuous-label regression is more discriminating than 6-class classification. |
| TR3 (BioCodec) wins because of leakage: BioCodec was trained on TUH, our secondary eval is TUEV (also TUH) → unfair advantage on TUEV | Report TR3's TUEV result with a footnote acknowledging the train-eval data overlap. The primary metric is HBN-derived (TUEV-disjoint), so the leakage doesn't affect the winner-picker. |

## What gets carried forward

The single chosen reconstruction target. Its specific configuration
(EMA momentum for TR2, codebook for TR3, k for TR4, sparsity weight for
TR5) is frozen for every later experiment. exp19 (decoder design) then
sweeps decoders appropriate for the chosen target — a TR2 latent winner
needs no decoder (only a small predictor head), a TR3 codec winner needs
a small classifier head, a TR0/TR1/TR5 raw winner needs a true
reconstruction decoder.
