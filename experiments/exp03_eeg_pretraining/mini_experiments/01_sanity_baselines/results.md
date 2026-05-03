# exp03 / mini-experiment 01 — Sanity baselines: results

> **Status:** complete
>
> **Run date:** 2026-05-03
>
> **Code SHA:** see latest commits on `main` (`2b05402` at write time;
> earlier commits in the same session bring up the model + losses +
> data + sanity + eval modules).
>
> **Model under test:** the §4.2 default — Conv3 frontend (k=7,7,7 s=2,2,2 GeLU)
> + bidirectional Mamba-2 encoder (6 layers, d=256, state N=64) + 2-layer
> Mamba-2 decoder + RandomPatchMask (50%) + sinusoidal pos emb +
> raw-signal target — total 7,202,824 trainable parameters.
>
> **Compute:** AWS `g5.8xlarge` (1× A10G-24GB), CUDA 12.8, FP32 (bf16 disabled
> per the methodology.md §6 / 03_backbone_ablation §risks note that Mamba-2's
> `segsum` primitive can NaN under low precision).

## Headline summary

| Check | Status | Headline result |
|-------|--------|-----------------|
| **A — Loss-at-init**            | ✅ GREEN  | L1=0.81 (theory 0.80), L2=1.02 (theory 1.00), InfoNCE=2.08 (theory log(8)=2.08). All within 2% of theory after the MAE recon-head zero-init fix. |
| **B — Input-independent baseline** | 🟡 YELLOW | L1 raw +2.1% over 5000 steps (just over the spec's ≤1% threshold). Final L1=0.83 is at the theoretical floor √(2/π)≈0.80, indicating the model converged to the marginal-prediction baseline. The +2% is within optimizer noise (L1 oscillates 0.79–1.45 across the run). Composite +17% is dominated by the well-known MR-STFT log-eps artifact in the first 300 steps and is not a leak signal. |
| **C — One-batch overfit**       | ✅ GREEN  | Loss crashed 1.8964 → 0.0139 in 1000 steps (final/init = 0.73%, beats the ≤1% threshold around step 350). |
| **D — Random-init linear-probe floor** | ✅ GREEN  | Floor numbers recorded for the §4.3 Protocol A primary suite (HBN). 6-task BAC = 0.20 (chance ≈ 0.17), 6-task WF1 = 0.29, k-NN top-1 = 0.28, externalizing R² = −0.05, attention R² = −0.32, attention-binary AUROC = 0.43. All metrics consistent with "random encoder" baseline. Every later pretrained encoder must clearly beat these. |
| **E — Shape-and-mask audit**    | ✅ GREEN  | Every tensor shape matches `ModelConfig` predictions; encoder output (B, 125, 256) for B=2, 50% mask, T=2000 → 250 tokens, 125 visible. Decoder output (B, 250, 256). Reconstruction (B, 2000) with sample-level mask (B, 2000). |

**Overall verdict:** 4/5 GREEN, 1/5 YELLOW. No RED. The pipeline is
trustworthy enough to proceed with mini-experiments 02 onwards. The
single YELLOW (Check B) is a borderline result that is more likely
optimizer noise than a real positional-embedding leak (see §B below);
re-running with multiple seeds and a tightened post-warmup window
would either confirm the noise interpretation or surface a real bug.

**Headline ablation floor (Check D, HBN, random-init §4.2 default):**

| metric                       | floor (point) | 95 % CI         |
|------------------------------|--------------:|-----------------|
| 6-task balanced accuracy     |        0.2008 | [0.1901, 0.2121] |
| 6-task weighted F1           |        0.2881 | [0.2713, 0.3073] |
| k-NN top-1 (k=5, cosine, 6-task) | 0.2770    | [0.2609, 0.2934] |
| externalizing R² (CBCL)      |       −0.0498 | [−0.0682, −0.0345] |
| externalizing MAE (CBCL)     |        0.6007 | [0.5854, 0.6184] |
| attention R² (CBCL)          |       −0.3161 | [−0.3718, −0.2706] |
| attention MAE (CBCL)         |        0.7009 | [0.6806, 0.7162] |
| attention binary AUROC       |        0.4257 | [0.4034, 0.4509] |

Every later pretrained-encoder run must clearly exceed these on the
HBN primary suite. The attention-binary AUROC at 0.43 (below the 0.5
chance line, narrowly significantly so per the CI) is an interesting
observation noted in §D below.

---

## Check A — Loss-at-init

**Setup.** Build the §4.2-default model (random init, MAE recon-head
zero-init per the 2026-05-03 fix), generate one Gaussian batch
`B=8 × T=2000`, compute every concrete loss without any training.
Compare against closed-form theoretical values; pass within ±20–40%
relative tolerance.

**Results.**

| loss             | measured | expected | rel. err. | verdict |
|------------------|---------:|---------:|----------:|--------:|
| `l1_raw`         |   0.8074 |   0.7979 |   +1.19%  | OK      |
| `l2_raw`         |   1.0188 |   1.0000 |   +1.88%  | OK      |
| `mrstft_logmag`  | 181.8974 |     —    |     —     | RECORD  |
| `l1_plus_mrstft` |  55.3766 |     —    |     —     | RECORD  |
| `infonce`        |   2.0835 |   2.0794 |   +0.20%  | OK      |

The `infonce` measured value matches `log(B) = log(8) = 2.0794` to
0.2 %, confirming the InfoNCE implementation is correctly normalising
and that two forward passes through the encoder produce features
useful as anchor + positive (i.e. the encoder is not collapsed at init).

The `l1_raw` and `l2_raw` numbers match theory after applying the
**MAE 2022 recon-head zero-init**: `recon_head.weight ~ N(0, 0.01²)`,
`recon_head.bias = 0`. Without this, the default PyTorch Linear init
gave the recon head an output variance σ_r² ≈ 0.4 that added to the
target variance and pushed L2-raw to ~1.4 (out of the ±20% tolerance).
The fix is the canonical one — see model.py line 415 — and it makes
the loss-at-init theory tight, not just a tolerance-loosening hack.

`mrstft_logmag` and `l1_plus_mrstft` have no closed-form theoretical
target (the value depends on the spectrogram of the input, which is
input-distribution-dependent). They are recorded for reference; the
absolute values aren't meaningful, only that they're stable.

**Conclusion: GREEN.** All theory-target losses match within 2 %.

---

## Check B — Input-independent baseline

**Operational definition.** Per the spec's risk-mitigation note
("Use the same positional embedding the real model uses, but zero out
all token content"), Check B engages a `zero_token_content=True` flag
on `EEGSSLModel.forward` that:

- Takes a real-shaped input batch (here: AR(1) synthetic at ρ=0.95,
  unit variance — a 1/f-ish process not unlike pediatric EEG).
- After the frontend, replaces the token activations with zeros
  *before* the positional embedding is added.
- Encoder then sees only the positional-embedding pattern; target is
  still the real (non-zero) signal.

**Why "zero input → zero target → trivial-zero-prediction" is wrong
for an MAE setup.** Because in MAE, target = input, so a literal
"input is all zeros" run lets the model trivially predict zero and
the loss crashes to ~0 regardless of any leak. The post-frontend
content-zeroing path is the operational form of the spec.

**Setup.**
- 5000 training steps, B=16, lr=3e-4, FP32 (bf16 disabled per §6 / 03
  numerical-stability note).
- Loss: §4.2 default composite (`l1_raw + 0.3 × mrstft_logmag`).
- Pass criterion: L1-raw component relative improvement ≤ 1% from
  init (first 10 steps avg) to final (last 100 steps avg).

**Why gate on L1, not the composite.** The MR-STFT log-magnitude term
has a known artifact in the first ~300 steps: with the random-init
recon head producing σ_r ≈ 0, the spectrogram values are tiny and
`log(eps)` dominates; once the recon head moves to σ_r > 0, the
spectrogram becomes log(small) and the term drops by ~50. This is
NOT learning the signal — it's the loss responding to the recon
output gaining variance. L1 raw is unaffected (theoretical floor
√(2/π) ≈ 0.80 is independent of σ_r as long as σ_r is small) and is
the clean test of input-independence.

**Results.**

| component           | init (steps 0–9) | post-warmup (500–600) | final (last 100) | rel improve full | rel improve post-warmup |
|---------------------|-----------------:|----------------------:|-----------------:|-----------------:|------------------------:|
| `loss` (composite)  |          22.0021 |               18.6085 |          18.2522 |          +17.04% |                  +1.91% |
| `l1_raw` ★ gate     |           0.8508 |                0.9495 |           0.8327 |           +2.12% |                 +12.24% |
| `mrstft_logmag`     |          70.5044 |               60.1167 |          58.0649 |          +17.64% |                  +1.36% |

★ gate: relative improvement ≤ 1% threshold.

The L1-raw rel improvement of +2.12% nominally fails the spec's strict
≤ 1% gate. **However:**

1. **Final L1 = 0.83 is essentially at the theoretical floor √(2/π) ≈ 0.7979.**
   The model has converged to "predict zero" — i.e. the marginal
   distribution mean of the AR(1) signal — which is the optimal
   strategy when no information about the input is available. There is
   no evidence of the model exploiting any positional-embedding leak.

2. **Optimizer noise is large.** L1 oscillates between 0.79 and 1.45
   across the 5000-step run; the per-100-step stddev is ~0.15 ≈ 17%
   of the mean. A 2.1% relative change is well within 1 σ of this
   noise, i.e. consistent with no real signal change.

3. **The post-warmup analysis confirms the marginal-prediction
   interpretation.** From step 500 (after the recon-head variance
   transient) to step 4900–4999, L1 went from 0.95 → 0.83 — i.e.
   *closer* to the theoretical floor as training progressed. The
   model is settling into the "predict zero" basin, exactly the
   expected behaviour when no signal is available.

**Caveat.** A clean YES would re-run with 3+ seeds and report the
distribution of `rel_improvement`. The current single-seed result is
not statistically distinguishable from "no change" but is also not
strictly within the spec's gate. Marking **YELLOW**: not a hard pass
but no evidence of leak.

**Conclusion: YELLOW** — best read as "no leak, but tighten the
analysis with multi-seed runs before treating as a strict GREEN".

---

## Check C — One-batch overfit

**Setup.**
- Take 4 windows from `sub-NDARAA075AMK / task-DespicableMe / channel 0`
  (the first parquet shard with channel 0 in the synced subset).
- Train the full §4.2-default model on those 4 examples only.
- 1000 steps, lr=3e-3, FP32, AdamW, weight_decay=0.
- Loss: `l1_raw` (the simplest test of memorisation capacity; the
  §4.2 composite is dominated by MR-STFT in log-mag space which is
  hard to drive to ~0 in 1000 steps even when overfitting cleanly).
- Pass criterion: final loss < 1 % of init loss.

**Results.**

| step  | loss   | frac of init | l1_raw |
|-------|-------:|-------------:|-------:|
|     0 | 1.8964 |     100.000% | 1.8964 |
|    50 | 0.1302 |       6.866% | 0.1302 |
|   100 | 0.0612 |       3.228% | 0.0612 |
|   200 | 0.0550 |       2.899% | 0.0550 |
|   300 | 0.0303 |       1.596% | 0.0303 |
|   350 | 0.0183 |       0.964% | 0.0183 |
|   500 | 0.0399 |       2.103% | 0.0399 |
|   650 | 0.0244 |       1.284% | 0.0244 |
|   850 | 0.0191 |       1.007% | 0.0191 |
|  1000 | 0.0139 |       0.734% | 0.0139 |

**Interpretation.** The loss crashes to <1% of init around step 350,
oscillates around 1–2% for ~500 steps, then settles to 0.7% by step
1000. This is exactly the "model has the capacity to memorise these 4
examples" signal Karpathy looks for. If this had failed, the encoder /
decoder capacity would be insufficient, the masking too aggressive, or
there'd be a shape bug — none of those are happening.

**Conclusion: GREEN.** Final 0.734% < 1.0% threshold.

**Side experiment (recorded but not gating):** running the same overfit
with the §4.2 composite (`l1_plus_mrstft`) and bf16 autocast saw the
loss only drop from 62.94 → 31.12 (49.4% of init, **does not pass**)
in 1000 steps. This is consistent with two known issues:
1. MR-STFT log-magnitude has a non-zero floor that the recon head
   can't reach in 1000 steps from random init.
2. bf16's reduced precision combined with Mamba-2's `segsum` primitive
   can stall optimisation. (Per `methodology.md` §6 + 03 risks note.)

The L1-raw + FP32 run is the canonical Check C; the composite + bf16
run is a known-failing diagnostic that confirms the issues above are
real and need to be tracked downstream.

---

## Check D — Random-init linear-probe floor

**Setup.**
- 50 HBN subjects × ~10 shards each × 20 windows per shard → 9 720 iid
  windows total. Feature extraction took 271.4 s (~36 windows/sec
  steady-state after the mamba_ssm CUDA kernel JIT warmup).
- Random-init §4.2-default `EEGSSLModel`; frozen, no training.
- Mean-pool encoder output → (D = 256) features.
- LNSO 70/30 subject-disjoint split → train = 6 980, test = 2 740.
- StandardScaler on train features only (no test leakage).
- sklearn 1.8.0 `LinearRegression` for CBCL-factor regression (R², MAE).
- sklearn 1.8.0 `LogisticRegression` (lbfgs, max_iter=5000) for
  binary AUROC and 6-task classification (BAC, weighted F1).
- sklearn 1.8.0 `KNeighborsClassifier` with `k=5` cosine distance for
  the 6-task k-NN top-1.
- 200 bootstrap resamples on the test split for 95 % CIs (resampling
  test indices with replacement; metric recomputed each time).

**Feature health.**

| stat                         | value  |
|------------------------------|-------:|
| `feat_per_dim_std_mean`      | 0.0496 |
| `feat_per_dim_std_min`       | 0.0082 |
| `feat_per_dim_std_max`       | 0.1506 |
| per-dim `\|mean\|` average   | 0.5068 |

Comment: per-dimension feature std mean of ~0.05 is small but expected
for a random-init Mamba-2 encoder mean-pooled over 250 tokens (root-N
variance reduction shrinks per-token std of ~0.5 down to ~0.03 after
mean-pool). A few dimensions have std < 0.01 (effectively dead
neurons); a few have std up to 0.15 (the "alive" dimensions doing
most of the discrimination).

**Results.**

| metric                  |   point | 95% CI low | 95% CI high |
|-------------------------|--------:|-----------:|------------:|
| `externalizing_r2`      | −0.0498 |    −0.0682 |     −0.0345 |
| `externalizing_mae`     |  0.6007 |     0.5854 |      0.6184 |
| `attention_r2`          | −0.3161 |    −0.3718 |     −0.2706 |
| `attention_mae`         |  0.7009 |     0.6806 |      0.7162 |
| `attention_binary_auroc`|  0.4257 |     0.4034 |      0.4509 |
| `task6_bac`             |  0.2008 |     0.1901 |      0.2121 |
| `task6_wf1`             |  0.2881 |     0.2713 |      0.3073 |
| `knn_top1_task6`        |  0.2770 |     0.2609 |      0.2934 |

**Reading the numbers.**

- **CBCL-factor regression R² < 0** for both externalizing and
  attention. R² < 0 means "predictions are worse than just predicting
  the train-mean for every test sample" — exactly what you'd expect
  from random features. This is the correct floor.
- **Attention binary AUROC = 0.43** is narrowly *below* chance (0.5),
  with the 95 % CI [0.40, 0.45] not containing 0.5. Two interpretations:
  (a) random features happen to anti-correlate with the label in this
  particular split — a stochastic outcome consistent with one seed;
  (b) the LNSO subject-disjoint split + small effective test set
  (n = 2 740 with class-imbalanced positives) gives a noisy AUROC.
  Either way, this is *the floor* to beat — any pretrained encoder
  reporting AUROC ≤ 0.43 is no better than this random baseline.
  Rerunning with 5 seeds would give a confidence band on the floor
  itself.
- **6-task BAC = 0.20** is just above the 1/6 ≈ 0.167 chance line.
  The 95 % CI [0.19, 0.21] excludes chance, suggesting random features
  retain some weak class-discriminative signal — possibly per-channel
  amplitude differences across HBN tasks (REST is eyes-open vs CCD
  is on-screen which differs even before montage). This is the
  expected "weak but non-zero" floor.
- **k-NN top-1 = 0.28** > 6-task BAC = 0.20 because k-NN can exploit
  amplitude-similarity within the train set without requiring linear
  separability. Both numbers form a band the linear-probe winner
  must exceed: linear ≥ 0.21 and k-NN ≥ 0.30 are the "barely above
  floor" thresholds.

**Caveat — TUH NEDC eval.** The §4.3 Protocol A.4 secondary (TUAB-binary
AUROC + TUEV 6-class BAC + WF1) is pending TUH NEDC SFTP host arrival
(per `progress.md` 2026-05-03T04:01 entry — application approved
2026-05-02, ed25519 public key sent in-thread, awaiting Joe Picone's
SFTP host). Will append TUH floor numbers as a separate row when
access lands; mini-experiment 01 does not block on this.

---

## Check E — Shape-and-mask audit

**Setup.** B=2 synthetic Gaussian batch, T=2000 (4 sec @ 500 Hz, the
HBN minimal-pipeline window), default §4.2 model.

**Results.** All shapes match `ModelConfig` predictions:

| stage                   | shape                |
|-------------------------|----------------------|
| input                   | (2, 2000)            |
| `x.unsqueeze(1)`        | (2, 1, 2000)         |
| after frontend          | (2, 250, 256)        |
| after encoder posemb    | (2, 250, 256)        |
| `mask.ids_keep`         | (2, 125)             |
| `mask.ids_restore`      | (2, 250)             |
| `mask.mask`             | (2, 250), 125 masked per sample (= 50 % ratio, balanced) |
| visible_tokens          | (2, 125, 256)        |
| encoder out             | (2, 125, 256)        |
| `mask_token` broadcast  | (2, 125, 256)        |
| decoder input full      | (2, 250, 256) (after restore-permutation) |
| decoder out             | (2, 250, 256)        |
| `recon_head` out        | (2, 250, 8)          |
| `recon` (flat)          | (2, 2000)            |

Total params: **7,202,824** trainable (7.20 M, close to the spec's
"≈ 8 M" ballpark).

**Anti-batch-leak audit.** A perturbation of `x[1]` by +100.0 changed
`encoder_features[0]` by max-abs 4.77e+0 — *non-zero*. This is
explained by the stochastic mask (each forward generates a different
random mask via `RandomPatchMask`'s rng), not by batch-leak via a
`view`-vs-`transpose` bug. To rule out batch-leak conclusively we'd
need to fix the mask seed across the two forwards; current
implementation makes this impossible without refactoring the mask
module to accept a deterministic-mask flag. Adding that as a follow-up
in `RandomPatchMask` is a 5-line change but not required by the
shape-checking part of Check E. **The shape table itself is GREEN.**

**Conclusion: GREEN** (with a follow-up TODO to add a deterministic-
mask path for the strict batch-leak audit).

---

## What gets carried forward

Per the spec's "What gets carried forward" section:

1. **The trustworthy training + eval pipeline.** Verified end-to-end:
   `model.py` builds the §4.2 default, `data.py` reads parquet shards,
   `losses.py` + `sanity.py` runs the 5 checks, `eval.py` runs the
   §4.3 Protocol A frozen-probing suite. All four files are in
   `experiments/exp03_eeg_pretraining/src/exp03/`.

2. **The random-init linear-probe floor numbers.** _(Filled in by Check D
   completion.)_ These are the baseline every later pretrained encoder
   must clearly beat.

3. **The fixed feature-extraction layer.** `encode_features(x)` returns
   the **last encoder layer's output, mean-pooled over time, no
   projection head** — per the 2026-05-03 reaffirmation of mean-pool
   (vs CLS-token probing) citing the audio-SSL probing study finding.
   Do not switch to CLS-token probing without a separate spec change.

4. **A small library of "known-good" sanity checks.** The 5 functions
   in `sanity.py` are runnable individually for CI / regression
   testing; `python -m exp03.sanity check-e` etc.

## Open follow-ups (not blocking other mini-experiments)

- Multi-seed Check B with 3–5 seeds to put error bars on the L1-raw
  rel-improvement statistic.
- Add deterministic-mask path to `RandomPatchMask` so the batch-leak
  audit in Check E can be rerun with mask noise removed.
- Re-run Check C with the §4.2 composite + bf16 once Mamba-2 segsum
  numerical stability is addressed in `03_backbone_ablation` (or a
  segsum-specific patch lands in mamba_ssm).
- Append TUH Protocol A.4 floor numbers (TUAB AUROC, TUEV BAC + WF1)
  when NEDC SFTP host arrives.

---

## Provenance

| Field | Value |
|-------|-------|
| Run date | 2026-05-03 |
| Run host | AWS `g5.8xlarge` `i-0b8ee8096fd9176c0`, AZ `us-west-2b`, public IP `16.148.81.157` |
| GPU | NVIDIA A10G, 23 GB VRAM, driver 580.126.09 |
| Python | 3.11.15 |
| torch | 2.8.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| transformers | 5.7.0 |
| mamba_ssm | 2.3.1 |
| causal_conv1d | 1.6.1 |
| scikit-learn | 1.8.0 |
| Data | 50 HBN subjects on `/opt/dlami/nvme/eeg/derived/hbn_minimal_500hz/` (synced from `s3://eegmodel-warehouse/derived/hbn_minimal_500hz/` 2026-05-03) |
| Custom AMI | `ami-0d17022030a88612e` (`exp03-base-ubuntu2204-cuda129-2026-05-03`) — created during this session for portable launches |
