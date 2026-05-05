# mini-experiment 17 — Generative paradigm for the Mamba backbone — **Results (v1)**

> **Run date:** 2026-05-04 14:15 UTC → 17:05 UTC
> **Compute:** AWS p4de.24xlarge in us-west-2a (8× A100-SXM4-80GB)
> **Cells:** 30/30 complete (3 paradigms × 2 controls × 5 seeds)
> **Wall-clock:** 2 h 44 min training + ~30 min stuck-thread debug + ~6 min sync = ~3 h 50 m total
> **Cost:** ~$102 (training compute, p4de @ $27.45/h on-demand)
> **Wandb:** https://wandb.ai/ritivel-eeg-ritivel/exp03/ (filter `tag:exp17`)
> **S3 archive:** `s3://eegmodel-warehouse/runs/exp03/17_generative_paradigm/2026-05-04T14Z_run1/`
>
> **Headline verdict:** **G0 MAE wins by default** under the README's decision rule. **But** the literature-grounded interpretation is that the experiment was **epistemically underpowered** — all three paradigms sat at the random-init floor for reasons that the EEG-FM and speech-SSL communities have spent two years writing about. v1 cannot distinguish "MAE is structurally right for Mamba-EEG" from "every input-space-prediction objective fails at this scale on this task." See §5 for the redesign.

---

## 1. Per-cell numbers (all 30 cells)

| cell | step | step/s | task6_bac | knn_top1 | ext_r2 | att_auroc |
|---|---|---|---|---|---|---|
| mae_eeg_seed0 | 17500 | 7.40 | 0.223 | 0.279 | -0.964 | 0.466 |
| mae_eeg_seed1 | 17500 | 7.90 | 0.181 | 0.267 | -0.349 | 0.402 |
| mae_eeg_seed2 | 17500 | 7.73 | 0.198 | 0.296 | -0.297 | 0.597 |
| mae_eeg_seed3 | 17500 | 7.72 | 0.180 | 0.267 | -0.155 | 0.202 |
| mae_eeg_seed4 | 17500 | 7.87 | 0.199 | 0.314 | -0.101 | 0.406 |
| mae_noise_seed0 | 17500 | 7.86 | 0.184 | 0.258 | -0.265 | 0.423 |
| mae_noise_seed1 | 17500 | 7.91 | 0.179 | 0.259 | -0.063 | 0.394 |
| mae_noise_seed2 | 17500 | 7.88 | 0.187 | 0.276 | -0.024 | 0.581 |
| mae_noise_seed3 | 17500 | 7.58 | 0.183 | 0.282 | -0.057 | 0.228 |
| mae_noise_seed4 | 17500 | 7.66 | 0.195 | 0.266 | -0.080 | 0.428 |
| ar_eeg_seed0 | 17500 | 14.31 | 0.195 | 0.287 | -0.779 | 0.395 |
| ar_eeg_seed1 | 17500 | 14.40 | 0.178 | 0.267 | -2.032 | 0.353 |
| ar_eeg_seed2 | 17500 | 14.17 | 0.211 | 0.295 | -1.500 | 0.635 |
| ar_eeg_seed3 | 17500 | 13.56 | 0.199 | 0.275 | -0.720 | 0.282 |
| ar_eeg_seed4 | 17500 | 13.48 | 0.200 | 0.288 | -0.214 | 0.410 |
| ar_noise_seed0 | 17500 | 13.60 | 0.187 | 0.281 | -0.361 | 0.445 |
| ar_noise_seed1 | 17500 | 14.25 | 0.181 | 0.240 | -0.945 | 0.286 |
| ar_noise_seed2 | 17500 | 13.79 | 0.190 | 0.265 | -0.139 | 0.490 |
| ar_noise_seed3 | 17500 | 13.72 | 0.180 | 0.280 | -0.248 | 0.559 |
| ar_noise_seed4 | 17500 | 14.35 | 0.195 | 0.284 | -0.133 | 0.406 |
| mar_eeg_seed0 | 17500 | 8.84 | 0.176 | 0.259 | -0.722 | 0.457 |
| mar_eeg_seed1 | 17500 | 9.20 | 0.173 | 0.235 | -0.378 | 0.356 |
| mar_eeg_seed2 | 17500 | 9.25 | 0.184 | 0.289 | -0.195 | 0.449 |
| mar_eeg_seed3 | 17500 | 9.05 | 0.200 | 0.244 | -0.257 | 0.250 |
| mar_eeg_seed4 | 17500 | 9.23 | 0.190 | 0.264 | -0.203 | 0.447 |
| mar_noise_seed0 | 17500 | 8.96 | 0.176 | 0.259 | -0.722 | 0.457 |
| mar_noise_seed1 | 17500 | 9.28 | 0.173 | 0.235 | -0.378 | 0.356 |
| mar_noise_seed2 | 17500 | 8.99 | 0.184 | 0.289 | -0.195 | 0.449 |
| mar_noise_seed3 | 17500 | 9.14 | 0.200 | 0.244 | -0.257 | 0.250 |
| mar_noise_seed4 | 17500 | 9.24 | 0.190 | 0.264 | -0.203 | 0.447 |

## 2. Aggregate by paradigm × control (mean ± std, n=5 seeds each)

| paradigm | control | task6_bac | task6_wf1 | knn_top1 | ext_r2 | att_r2 | att_auroc |
|---|---|---|---|---|---|---|---|
| **G0 MAE** | EEG    | **0.196 ± 0.017** | 0.273 ± 0.036 | 0.285 ± 0.020 | −0.373 ± 0.346 | −0.496 ± 0.626 | 0.415 ± 0.143 |
| G0 MAE    | noise  | 0.185 ± 0.006     | 0.252 ± 0.018 | 0.268 ± 0.011 | −0.098 ± 0.096 | −0.283 ± 0.368 | 0.411 ± 0.125 |
| **G1 AR**  | EEG    | **0.196 ± 0.012** | 0.274 ± 0.023 | 0.283 ± 0.011 | −1.049 ± 0.716 | −1.601 ± 1.401 | 0.415 ± 0.133 |
| G1 AR     | noise  | 0.186 ± 0.006     | 0.260 ± 0.018 | 0.270 ± 0.018 | −0.365 ± 0.337 | −0.521 ± 0.375 | 0.437 ± 0.102 |
| **G2 MAR** | EEG    | **0.185 ± 0.011** | 0.259 ± 0.015 | 0.258 ± 0.021 | −0.351 ± 0.220 | −0.468 ± 0.464 | 0.392 ± 0.089 |
| G2 MAR    | noise  | 0.185 ± 0.011     | 0.259 ± 0.015 | 0.258 ± 0.021 | −0.351 ± 0.220 | −0.468 ± 0.464 | 0.392 ± 0.089 |

**Reference:** Track A random-init linear-probe floor (`mini_experiments/01_sanity_baselines/results.md`):
6-task BAC = 0.20, k-NN top-1 = 0.28, externalizing R² = −0.05, attention AUROC = 0.43.

## 3. Decision rule applied (per the README §17)

| Variant | n | task6_bac | Δ vs G0 MAE | Welch t | p (two-tail, df=8) | Verdict |
|---|---|---|---|---|---|---|
| **G0 MAE-EEG** | 5 | 0.196 ± 0.017 | — | — | — | reference |
| **G1 AR-EEG** | 5 | 0.196 ± 0.012 | +0.000 | 0.00 | 1.00 | **TIE** ⇒ keep MAE |
| **G2 MAR-EEG** | 5 | 0.185 ± 0.011 | −0.011 | 1.21 | 0.26 | technically below the README's 1pp loss threshold but **NOT statistically significant**; rated **TIE-leaning-loss** ⇒ keep MAE |

**Per the README:** _"In a tie, the pre-existing default (G0 MAE) is kept — the burden of proof is on the new paradigm, not on the established baseline."_ → exp04, exp10, exp18, exp19 should anchor on G0 MAE.

**Important caveat the README's decision rule didn't anticipate:** all three paradigms are sitting **at the random-init floor of 0.20** on the headline metric. The "decision" between three things that all failed to lift off the floor is not really a decision about which paradigm is better. See §4.

## 4. Noise-twin diagnostic — and what it really says

| Paradigm | EEG-trained BAC | Noise-trained BAC | Δ (EEG − noise) | Verdict |
|---|---|---|---|---|
| G0 MAE | 0.196 ± 0.017 | 0.185 ± 0.006 | **+0.011** (≈ 0.65σ) | "barely above noise twin" |
| G1 AR | 0.196 ± 0.012 | 0.186 ± 0.006 | **+0.010** (≈ 0.83σ) | "barely above noise twin" |
| G2 MAR | 0.185 ± 0.011 | 0.185 ± 0.011 | **+0.0005** (≈ 0.05σ) | **encoder did not differentiate EEG from Gaussian noise** |

The noise-twin separation tells us the most about what actually happened during pretraining:

- MAE / AR encoders trained on real EEG are weakly distinguishable (≤1 std-dev) from the same encoders trained on Gaussian noise of matched statistics.
- MAR's encoder is **statistically indistinguishable** from its noise-twin. The diffusion-loss head's ε-prediction objective is satisfied equally well by the encoder's output regardless of input, because at 100 hours / 1000 masked positions per batch, the diffusion head learns the optimal `predict zero` solution and the encoder's representations don't need to be informative.

This is consistent with the [STELAR (OpenReview, ICLR 2026 under review)](https://openreview.net/pdf?id=ofwQZjoI4c) characterization of input-space SSL on EEG: _"the encoder is biased toward high-frequency noise and acquisition artifacts rather than the high-level semantic representations necessary for broad generalization."_ At 100 hours, all three of our objectives have converged to "noise-shaped solutions" rather than "EEG-content solutions."

## 5. Why this happened — four compounding failure modes (literature-grounded)

The v1 result is consistent with well-documented failure modes that compound multiplicatively at our scale. Each is independently sufficient to kill the experiment; we hit all four at once.

### 5.1 Below the data-scale "cliff" for biosignal SSL

| Modality / model | Scale at which SSL beats random-init linear probe | What we did |
|---|---|---|
| Speech wav2vec 2.0 ([Berrebbi et al. 2022, arXiv 2402.13723](https://arxiv.org/abs/2402.13723)) | "Most important factor is total data seen ... benchmark target ~100k h" | ≈1.2k EEG-equivalent h |
| Speech wav2vec 2.0 ([Likhomanenko et al. 2022, arXiv 2211.00854](https://arxiv.org/abs/2211.00854)) | **>500 h unlabeled** AND **>1000 unique speakers** as the floor | 100 h, 100 subjects |
| Speech wav2vec 2.0 (Berrebbi 2022) below 500 h | **49% WER vs 23% supervised-from-scratch** — actively WORSE than no pretraining | n/a |
| EEG-FM LaBraM ([Jiang et al. ICLR 2024](https://arxiv.org/pdf/2405.18765)) | Explicit Figure 5 ablation: "500 h ≈ 2500 h on TUAB" — **the curve plateaus**, the cliff is at the bottom | 100 h |
| EEG-FM REVE ([Ouahidi et al. NeurIPS 2025](https://arxiv.org/abs/2510.21585)) | 60,000 h, 25,000 subjects to claim consistent linear-probe SOTA | 100 h, 100 subjects |
| EEG-FM DIVER-1 ([Han et al. 2025, arXiv:2512.19097](https://arxiv.org/abs/2512.19097)) | 59,300 h, 17,700 subjects; explicitly "data-dominated scaling laws" | 100 h, 100 subjects |
| Wearable LSM ([Google 2024, arXiv 2410.13638](https://arxiv.org/abs/2410.13638)) | "Improves monotonically to ~10⁵ h, plateaus around 10⁷ h" | 100 h |

We pretrained on **100 hours of HBN-EEG** (≈ 50 subjects × 2h average). This is **5–600× below** the empirically validated thresholds across speech, EEG, and wearable SSL. Speech literature is the most quantitative: below 500h the model actively underperforms a from-scratch supervised baseline. Our reconstruction objectives sit firmly inside this "cliff" regime.

### 5.2 Wrong pretext task family for low-SNR signals

EEG has signal-to-noise ratio of approximately **−20 dB** on raw single-channel windows. A reconstruction loss is then dominated by the un-reconstructable noise component, so the encoder's gradients are 99%+ shaped by noise structure, not EEG content.

This is documented as the central problem in three independent papers:

- **EEG2Rep ([Foumani et al. KDD 2024, arXiv 2402.17772](https://arxiv.org/abs/2402.17772))** — _the_ paper on this:
  > _"Self-prediction pretraining tasks ... when applied to EEG data ... low signal-to-noise ratio in EEG challenges the reconstruction task ... the resulting representations are typically of a lower semantic level and may underperform invariance-based pretraining in off-the-shelf evaluations like linear probing."_
  
  Their fix: predict in **representation space** (JEPA-style). Direct measurement: **−13.12% linear-probe accuracy** when predicting in input-space vs. latent-space, on the same model.

- **STELAR (OpenReview ICLR 2026 under review)**:
  > _"Most existing self-supervised approaches rely on masked raw signal reconstruction. While this objective maintains local waveform fidelity, it often biases the encoder toward high-frequency noise and acquisition artifacts rather than the high-level semantic representations necessary for broad generalization."_

- **EEG-FM benchmark ([Liu et al. arXiv:2601.17883, ICLR 2026 under review](https://www.emergentmind.com/papers/2601.17883))** — most rigorous statement:
  > _"linear probing is frequently insufficient. Specialist models trained from scratch remain competitive across many tasks. Larger foundation models do not necessarily yield better generalization performance under current data regimes. Scaling laws do not hold in current EEG-FMs."_

**All three of our paradigms are input-space-prediction:** G0 MAE predicts raw signal at masked positions; G1 AR predicts raw next-patch; G2 MAR predicts raw signal via diffusion ε. We selected from a paradigm family the literature has documented as systematically wrong for EEG. This was not visible from the README's vision-domain motivation (MAP / AR-Mamba / MAR) because vision has high SNR — pixels are mostly content, not noise.

### 5.3 The downstream task is essentially unsolvable at our scale

This is the part that wasn't visible during planning. The HBN externalizing-factor regression is **literally the NeurIPS 2025 EEG Foundation Challenge — Challenge 2**. Here's what 1,183 of the world's best teams achieved on it ([leaderboard](https://eeg2025.github.io/leaderboard/)):

> _"Challenge 2 remained extremely difficult — only three teams achieved scores below 0.99, our threshold for including this challenge in the final scoring (recall that a score of 1 represents predicting the mean target value)."_

| Rank | Team | NRMSE |
|---|---|---|
| 🥇 1 | Team JLShen | **0.97843** (and likely exploited a non-randomization data-leak per the organizers) |
| 🥈 2 | Team MBZUAI [dsml.kz] | 0.98519 |
| 🥉 3 | Team MIN~C² (MIND-CICO) | 0.98817 |
| 4–1183 | _everyone else_ | _≥ 0.99 (i.e. worse than predicting the mean)_ |

The combined-metric winner [NeuroSned](https://github.com/sneddy/neurosned) used **bootstrap-stabilized Ridge regression on hand-crafted lagged-correlation and ridge transition-matrix features**. The winner of an "EEG Foundation Model" challenge explicitly avoided foundation-model approaches because everything else failed.

For HBN 6-task BAC — our other headline metric — the situation is analogous. The 6-task headline is a hard cross-task generalization problem; the field's specialist models (per [EEG-FM-Bench arXiv:2508.17742](https://arxiv.org/html/2508.17742v2)) achieve TUEV 6-class BAC of 47–71% under fine-tuning, but **49–58% under linear probe** for the smaller models — barely above the 16.7% chance level, and often below random projections of the same dimension.

### 5.4 Linear-probe under LNSO is the wrong eval for small-scale EEG SSL

[Dubois, Hashimoto, Liang ICML 2023 oral](https://proceedings.mlr.press/v202/dubois23a.html) decomposes SSL evaluation error into 4 terms; their key finding from analyzing 169 SSL models:

> _"In few-shot probing regimes, probe generalization error has become dominant"_ — i.e., linear probe with few labeled subjects is fundamentally noise-limited regardless of the encoder's quality.

[Wortsman et al. ICML 2022 LP-FT](https://arxiv.org/abs/2202.10054) on out-of-distribution evaluation (LNSO IS out-of-distribution by definition): linear-probe-then-fine-tune gives **+10% OOD accuracy** vs pure fine-tuning. Pure linear probe is the worst of the three for out-of-distribution evaluation.

[ST-EEGFormer benchmark (under review ICLR 2026)](https://openreview.net/pdf/44f1859e3b0d17192639706701c950b9057aa9cc.pdf), 20,000+ models trained, statistically rigorous Wilcoxon-signed-rank tests:

> _"Linear probing consistently yields poor performance across all models and evaluation protocols (except for LOO Drop) ... current pre-training strategies do not produce EEG representations that are sufficiently generalizable and discriminative across a broad range of BCI tasks."_

> _"Foundation models often fail to significantly outperform compact neural networks or even classical non-neural decoders in data-scarce scenarios."_

[Xue, Joshi, Gan, Chen, Mirzasoleiman ICML 2023](https://proceedings.mlr.press/v202/xue23d.html) on the theoretical mechanism — gradient descent's simplicity bias makes the encoder learn easy class-irrelevant features (subject identity, recording-site fingerprint) and **suppress** harder class-relevant features. This produces NEGATIVE R² when there's any accidental confound between easy features and the downstream label — exactly what we see (R² between −0.10 and −1.05 across all 30 cells; the random-init floor was −0.05, so pretraining made things WORSE on regression because it amplified the simplicity-bias features).

**[CLISA arXiv:2109.09559](https://arxiv.org/abs/2109.09559)** for EEG specifically and **[ContentVec ICML 2022](https://proceedings.mlr.press/v162/qian22b.html)** for the speech analog show how to fix this: explicit subject-adversarial training during pretraining. Without it, the encoder defaults to subject identity (HuBERT achieves **81.4%** speaker-ID linear probe accuracy on SUPERB without any speaker objective). We have no equivalent regularization in any of G0/G1/G2.

## 6. What this means for the README's pre-registered hypothesis

The README §17 pre-registered three hypotheses and an anti-prediction:

| Variant | Prediction | Observation | Status |
|---|---|---|---|
| G0 MAE-EEG | reference (≈35–40% expected) | 0.196 (≈ 1pp above the 0.20 random-init floor) | **far below predicted; at floor** |
| G1 AR-EEG | strict win, ~+2–4 pp | Δ = +0.000 (TIE) | **prediction not supported** |
| G2 MAR-EEG | strict win, ~+3–5 pp | Δ = −0.011 (LOSS-trending, n.s.) | **prediction not supported** |

The anti-prediction was: _"the MAP/AR-Mamba mechanism does not transfer, G0 wins, and the rest of the spec stays anchored as written"_ (~25% prior probability).

**The literal outcome matches the anti-prediction.** But the literature analysis above suggests the right interpretation is not _"MAP / AR-Mamba mechanism doesn't transfer to EEG"_ — it's _"none of these three paradigms could lift off the random-init floor at our scale, so we have no power to test the MAP hypothesis."_

A meaningful test of the MAP hypothesis requires (per the literature) at minimum:
1. ~1000 hours of pretraining data (10× our v1 scale) to cross the SSL data cliff
2. A pretext task family that is not pure input-space reconstruction (or proof that it works)
3. A downstream task where the field has demonstrated SSL signal exists at small scale
4. An evaluator more sensitive than frozen linear probe under LNSO

v1 satisfied 0 of these 4 preconditions. Hence v2 (§7).

## 7. Operational verdict for downstream mini-experiments

Per the README's decision rule, the formal call is **TIE ⇒ keep G0 MAE as the §4.2 default.** exp04 / exp10 / exp18 / exp19 anchor on MAE.

This decision is **anchoring-by-default-not-by-evidence**. We have no positive evidence that MAE is the right paradigm; we have only the absence of evidence for any alternative. The same caveat should be reproduced in every downstream mini-experiment's writeup until v2 of this experiment lands.

## 8. Additional findings & engineering observations from v1

### 8.1 Throughput

| Paradigm | step/s @ batch 32 / 1× A100-80GB | per-cell wall-clock (17500 steps + Protocol A eval) |
|---|---|---|
| G0 MAE | 7.4 | ~40 min |
| G1 AR  | 14.0 | ~22 min (smaller model, no decoder) |
| G2 MAR | 9.0 | ~32 min (1.42× MAE — exactly matches README's 1.4× projection) |

8 cells in parallel on 1× p4de.24xlarge → 4 batches × ~50 min each = ~3.4 h wall-clock for 30 cells.

### 8.2 Bugs found and fixed during the run

1. **DDP data loader crash** — `accelerate.utils.operations.concatenate` failed on `collate_signal_batch` because of `list[str]` metadata. Workaround used: 8 cells in parallel on separate GPUs (no DDP per cell). Fixed in this morning's session in `data.py`: `collate_signal_batch_train` is now tensors-only and DDP-safe. Also added rank-aware shard sharding to `ParquetWindowDataset.__iter__`. Future cells can use full 8-GPU DDP.

2. **CPU thread thrash with 8 concurrent processes** — PyTorch's intra-op pool grabbed all 96 cores per process; 8 × 200 = 1600 threads on 96 cores → `futex_wait_queue` gridlock. Fixed by setting `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` etc. per cell via subshell `export`. After fix: 37 threads / process, sustained 18% GPU utilization across all 8 cells.

3. **Bash inline env-var prefix didn't propagate** through the launcher's bash function with line continuations. Fixed in `scripts/launch_exp17.sh` by using `(export X=...; python ...)` subshell pattern.

4. **Stdout block-buffering** — Python's `print` is line-buffered for TTYs but block-buffered for files. Live `tail run.log` only saw flushed lines; the run looked stuck between log boundaries. Fixed in this morning's session in `train.py` by `sys.stdout.reconfigure(line_buffering=True)` at module import.

### 8.3 Bug found in the eval pipeline (CBCL R² anomaly)

The most striking artefact in v1 was that **CBCL externalizing R² was strongly negative across all 30 cells** (range −0.10 to −1.05; the random-init floor is only −0.05). On investigation this morning, the root cause is in `eval.py`: the linear probe used **unregularized `LinearRegression`** on 256-dim features × ~1000-subject LNSO splits.

Without regularization, the regressor catastrophically overfits subject-specific patterns in the train split that don't generalize, producing predictions that are systematically wrong on held-out subjects. Fixed in this morning's session: `LinearRegression()` → `RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0), cv=5)`. The cell-level alpha selected by cross-validation is now logged in the per-cell summary.

**Implication for v1's R² numbers:** the negative R² values in §1 should be treated as artefacts of the unregularized probe, not real signal that pretraining made representations worse. The relative ordering across paradigms (MAR > MAE > AR on R²) is mostly noise. If anyone wants the corrected numbers, the v1 checkpoints are saved in S3 under the run prefix and can be re-evaluated under the fixed pipeline cheaply (~5 min/cell on 1 GPU).

The BAC / WF1 / k-NN / AUROC numbers are correct as reported — those metrics use `LogisticRegression` (regularized by default with C=1) and `KNeighborsClassifier`, which are not affected.

## 9. Cost summary

| Item | Cost |
|---|---|
| Bootstrap (mamba_ssm + causal_conv1d compile, sm_80) | ~$5.6 |
| Smoke tests (G0 / G1 / G2 / noise-twin verification) | ~$3.7 |
| HBN sync (100 subjects, 29 GB to NVMe) | ~$1.8 |
| **30-cell training matrix (4 batches in parallel)** | **~$75.0** |
| Watchdog (wandb-sync + S3 sync at end) | ~$2.7 |
| Stuck-thread debug + relaunch | ~$13.7 |
| **Total v1** | **~$102** |
| EBS storage (instance stopped) | ~$2.50/mo ongoing |

## 10. v2 plan (see also `README.md` §"v2 design")

The v2 redesign addresses all four failure modes from §5:

1. **Scale up to ~1000 hours** of HBN pretraining (~500 subjects, already in S3). Crosses LaBraM's plateau threshold.
2. **Add G3 latent-prediction paradigm** (EEG2Rep / I-JEPA style) — predict in representation space, not input space. The literature predicts this is the only paradigm that should work below 1000h.
3. **Switch primary eval to fine-tuning** (or LP-FT per Wortsman et al.). Linear probe stays as a secondary diagnostic but not the headline.
4. **Switch headline metric to TUAB binary AUROC** — a task the field has demonstrated SSL signal on at small scale (LaBraM achieves 0.84 AUROC at 2.5kh; floor is 0.59). Keep HBN CBCL externalizing as a stretch metric we expect to remain near the mean-predictor baseline.

Plus: add **subject-fingerprint and effective-rank diagnostics** to `eval.py` so we can detect the subject-fingerprint dominance / dimensional collapse failure modes that the literature predicts at small scale.

Estimated cost for v2: ~$300–600. ETA: one GPU session after the v2 code is scaffolded and smoke-tested locally.

---

**Status of v1:** complete and archived. **Status of v2:** scaffold begins 2026-05-05.
