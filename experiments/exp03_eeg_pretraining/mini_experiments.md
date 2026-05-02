# exp03 mini-experiments — master list

> **Purpose.** Resolve the open architectural and loss-function decisions for the
> iid-channel single-channel EEG self-supervised pretraining recipe by running a
> sequence of small, scoped, fast ablations *before* committing to a headline
> pretraining run.
>
> Read alongside `[methodology.md](./methodology.md)`. The methodology document
> is the *why* (Phase 0–4 framework, eval discipline, monitoring); this file and
> its child folders are the *what to actually run*.
>
> Last refreshed: **2026-05-02** — switched the pretraining corpus from TUEG
> to **HBN-EEG** (open access, on AWS public storage, larger iid yield from a
> 128-ch montage), and the primary frozen-probing eval from TUAB+TUEV to
> **HBN ADHD-vs-no-diagnosis binary AUROC + HBN 6-task classification BAC/WF1
> + k-NN on a 10k HBN subset**. TUAB+TUEV remain a *secondary* eval (§4.3
> Protocol A.4) for direct literature comparability, conditional on pending
> TUH NEDC access — they are reported alongside the primary HBN metrics but
> are **not** used for the §4.4 winner-picker decision rule. See §4.1 for
> the rationale (open access, no NEDC wait, 2.5× more iid examples per
> recording-hour, modern uniform montage).
>
> Last refreshed: **2026-05-01** — added experiments 13 (adversarial dataset probe),
> 14 (context-length scaling), 15 (loss-weight sensitivity + curriculum) on the
> basis of [`brain/cortico-ssl-hypothesis.typ`](../../../brain/cortico-ssl-hypothesis.typ);
> added experiment 16 (Neurodynamics Statistics Prediction auxiliary head),
> updated experiment 04 (added an MER+NSP framework variant), and surfaced
> frozen-probing as the primary evaluation protocol throughout, on the basis
> of [DeeperBrain (Wang et al., arXiv 2601.06134)](https://arxiv.org/abs/2601.06134).

---

## 1. Why mini-experiments at all

The previous chats settled the high-level framing: iid-channel pretraining
(each `(channel, recording)` pair is one training example), pure scratch SSL
(no teacher network, no EMA momentum encoder, no two-stage tokenizer training),
single-channel input, multi-rate corpus (200 / 256 / 500 / 1000 / 2000 Hz),
phase information must survive, low SNR is the dominant pathology, and H100
compute is available.

What the framing does *not* settle:

- which **front-end** maps the raw single-channel signal to the encoder's
token sequence (a vanilla strided conv, a SincNet bandpass bank, frozen
Kymatio scattering, complex Gabor, an S4-as-frontend, a complex CQT, …);
- which **backbone** processes the token sequence (a Transformer, a
bidirectional Mamba-2, an LRU with complex eigenvalues, a hybrid SSM ×
attention, a CNN-only U-Net);
- which **SSL framework** trains the encoder without a teacher (raw-target
MAE, denoised-target MAE, VICReg-style siamese, time-frequency-consistency,
EEGDM-style score matching);
- which **reconstruction loss** is used (L2, Huber, Barron with learned α,
Itakura-Saito on the periodogram, multi-resolution STFT, bispectral
consistency, sin/cos circular phase loss, or any combination);
- which **masking strategy** is used (random, semantic-subsequence-preserving,
amplitude-aware AAMP);
- whether the **bottleneck** is continuous, FSQ, RVQ, or LFQ;
- whether **multi-rate handling** is done by uniform resampling, by
rate-conditioned Δ in an SSM, by MSR-HuBERT-style rate-specific CNN
branches, by SincNet's Hz-parameterised cutoffs, or by frozen scattering;
- whether **multi-condition pretraining** (synthetic noise injected into the
input while the target stays clean) actually helps in EEG the way it helped
WavLM in speech;
- whether the **denoised-target trick** (predict an offline-cleaned signal
rather than the raw signal) gives the SNR robustness the EEG-X paper
reports, in a single-stage scratch setup;
- which **easy-win modifications** (Snake activations, BlurPool, VICReg as an
auxiliary regulariser) are worth paying the implementation cost for.

Each open question above maps to one mini-experiment in the table below. The
goal is to settle them with cheap, fast, statistically honest comparisons in
the §2 ablation discipline of `[methodology.md](./methodology.md)`, instead of
arguing about them on first principles or picking based on whichever recent
paper looks shiniest.

---

## 2. The list

Every row links to a folder containing a detailed README. Each experiment is
designed to fit in **≤ 24 H100-hours** (one node-day or less) for the full
ablation matrix including the matched-noise twin and five seeds per cell.
"Sequence" is the dependency: an experiment cannot start until its
predecessors are complete (because its winners are inputs).


| ID  | Folder                                                                                    | Question                                                                                                                                                             | Sequence                   | Compute (H100-hours) |
| --- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------------------- |
| 01  | `[01_sanity_baselines/](./mini_experiments/01_sanity_baselines/)`                         | Does the trainer + eval pipeline behave correctly on toy inputs? (Phase-1 sanity per `[methodology.md` §2](./methodology.md#phase-1--sanity-baselines-½-day-1-gpu).) | gate for everything        | ≤ 4                  |
| 02  | `[02_frontend_ablation/](./mini_experiments/02_frontend_ablation/)`                       | Which front-end is the right map from raw 1-D signal to token sequence?                                                                                              | parallel with 03, 04       | 18                   |
| 03  | `[03_backbone_ablation/](./mini_experiments/03_backbone_ablation/)`                       | Which backbone (Mamba-2 / LRU / Transformer / hybrid) processes the token sequence best at iso-FLOP?                                                                 | parallel with 02, 04       | 24                   |
| 04  | `[04_ssl_framework_ablation/](./mini_experiments/04_ssl_framework_ablation/)`             | Which scratch SSL framework (denoised-MAE / VICReg / TF-C / EEGDM-diffusion) gives the best low-SNR representation?                                                  | parallel with 02, 03       | 24                   |
| 05  | `[05_multirate_strategy/](./mini_experiments/05_multirate_strategy/)`                     | How should multiple sampling rates be handled — resample, rate-conditioned Δ, MSR-HuBERT branches, or scattering?                                                    | needs 02 + 03 winners      | 18                   |
| 06  | `[06_reconstruction_loss/](./mini_experiments/06_reconstruction_loss/)`                   | Which time-domain reconstruction loss best handles EEG's heavy-tailed amplitude distribution?                                                                        | needs 04 winner            | 12                   |
| 07  | `[07_phase_handling/](./mini_experiments/07_phase_handling/)`                             | Does an explicit phase-aware loss (sin/cos circular, complex STFT, bispectral) materially improve representations over magnitude-only?                               | needs 04 + 06 winners      | 18                   |
| 08  | `[08_denoising_target/](./mini_experiments/08_denoising_target/)`                         | Which target signal — raw, bandpass, ICA-cleaned, PCA-projected, wavelet-denoised — gives the best representation under low SNR?                                     | needs 04 winner            | 18                   |
| 09  | `[09_multicondition_input/](./mini_experiments/09_multicondition_input/)`                 | Does WavLM-style multi-condition pretraining (clean target + noise-injected input) work for scratch EEG SSL?                                                         | needs 04 + 08 winners      | 12                   |
| 10  | `[10_masking_strategy/](./mini_experiments/10_masking_strategy/)`                         | Within MAE-style training, does AAMP / SSP beat random masking on low-SNR EEG?                                                                                       | needs 04 winner            | 8                    |
| 11  | `[11_bottleneck_continuous_vs_fsq/](./mini_experiments/11_bottleneck_continuous_vs_fsq/)` | Can FSQ be trained jointly with the SSL objective in a single stage, and does it beat the continuous baseline?                                                       | needs 04 winner            | 18                   |
| 12  | `[12_quick_wins_consolidation/](./mini_experiments/12_quick_wins_consolidation/)`         | Do the proposed "free wins" (Snake activations, BlurPool anti-aliasing, VICReg auxiliary) survive a strict ablation?                                                 | needs 02 + 03 + 04 winners | 12                   |
| 13  | `[13_adversarial_dataset_probe/](./mini_experiments/13_adversarial_dataset_probe/)`       | Does a gradient-reversal "predict source dataset" probe actually reduce subject/rig fingerprinting (P6, H6) without hurting downstream performance?                  | needs 12 winner            | 12                   |
| 14  | `[14_context_length_scaling/](./mini_experiments/14_context_length_scaling/)`             | Does the long-context capability that justifies Mamba-2 over Transformer translate into better representations? At which window length (4 s / 8 s / 16 s / 30 s)?    | needs 02 + 03 winners      | 24                   |
| 15  | `[15_loss_weight_curriculum/](./mini_experiments/15_loss_weight_curriculum/)`             | Are the hypothesis loss weights robust to ± 2× perturbation, and does the proposed three-stage curriculum materially improve over training the full loss from step 0? | needs 06 + 07 + 11 + 13 winners | 24              |
| 16  | `[16_nsp_auxiliary_head/](./mini_experiments/16_nsp_auxiliary_head/)`                     | Does adding a DeeperBrain-style Neurodynamics Statistics Prediction (NSP) auxiliary head — predicting spectral power, CFC, sample entropy directly from the encoder representation — materially improve frozen-probing performance beyond the existing reconstruction-based losses? | needs 12 winner            | 12                   |


**Total compute budget**: ≈ 264 H100-hours (12 added by experiment 16 plus
6 added by the new MER+NSP variant in experiment 04), comfortably under
~1.5 weeks on a single 8×H100 node. This is the ablation tier of the
methodology's compute heuristic (`[methodology.md` §1](./methodology.md#1-the-mental-model-pretraining-is-capability-engineering-not-luck)):
spend roughly the same amount of compute on derisking as on the headline run.

---

## 3. The dependency graph

```
                          01 sanity baselines
                          (gate for everything)
                                  │
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                    ▼
        02 frontend         03 backbone        04 SSL framework
              │                   │                    │
              └───────┬───────────┘                    │
                      ▼                                ▼
                  05 multirate              ┌──────────┼──────────┬──────────┐
                                            ▼          ▼          ▼          ▼
                                      06 recon    08 denoised  10 mask  11 bottleneck
                                         loss       target     strategy   FSQ vs cont
                                            │          │
                                            ▼          ▼
                                      07 phase    09 multi-cond
                                       handling     input
                                            │
                                            ▼
                                       12 quick wins
                                        (final stack)
                                            │
              ┌──────────────────┬──────────┴──────────────┬───────────────────────┐
              ▼                  ▼                         ▼                       ▼
   14 context-length      13 adversarial         16 NSP auxiliary head     15 loss weights + curriculum
        scaling           dataset probe          (DeeperBrain-style)        (depends on 06+07+11+13)
   (parallel with         (depends on 12)        (depends on 12)
   02+03 winners only)
```

The graph is an *advisory* sequence, not a hard dependency. Pairs of
experiments that don't strictly depend on each other can and should run in
parallel on a multi-GPU node. The honest hard gates are:

- **01 must finish before any other experiment starts.** A miscalibrated
trainer or eval suite contaminates every later result.
- **02, 03, 04 can run in parallel** by holding the other two axes at a
reasonable default (vanilla strided-conv frontend, Mamba-2 backbone,
raw-target MAE framework).
- **05–11 depend on the 02–04 winners** because they extend the chosen
configuration in one direction at a time.
- **12 is the final consolidation of the core stack** — it stacks the individual wins from
every prior experiment and checks that the combination still works.
- **13–16 are post-12 hardening experiments**. 13 (adversarial probe)
asks whether the cheapest defence against P6 from the cortico-ssl-hypothesis
actually pays off; 14 (context-length) asks whether the long-context
argument for Mamba-2 over Transformer is empirically supported; 15
(loss weights + curriculum) asks whether the hypothesis recipe
weights and the three-stage curriculum schedule are robust;
16 (NSP auxiliary head, added on the basis of DeeperBrain) asks whether
a "predict the dynamical statistic from the encoder representation"
auxiliary head materially improves frozen-probing performance. 14 can
run in parallel with 02+03+04+05 if a Mamba-2 baseline is already
trusted; 13, 15, and 16 must wait for 12.

---

## 4. Cross-cutting protocol (applies to every mini-experiment)

The following hold across all twelve experiments unless an individual
experiment's README explicitly overrides them.

### 4.1 Pretraining corpus

Use the same fixed pretraining subset for every comparison: **100 hours of
HBN-EEG drawn from at least 200 subjects**, single-channel iid-expanded (so
a 128-channel 4-second epoch becomes 128 independent training examples),
**at the native 500 Hz sampling rate**, with **minimum-offline** preprocessing:

```
NaN sanitation  →  per-channel z-score  →  ±5σ clip  →
4-second non-overlapping windowing  →  iid-channel expansion  →
float16 parquet shards
```

That is **all** the offline preprocessing. Notch filtering, bandpass
filtering, and resampling are **deliberately not done offline** — they are
hypotheses tested by:

- **exp02 (frontend ablation)**: F2 SincNet learns Hz-parameterised bandpass
  cutoffs end-to-end, F3 frozen scattering provides a fixed wavelet basis,
  F4 complex Gabor provides a Hz-spaced complex bandpass. If we offline-
  bandpass at 0.5–100 Hz before the frontend sees the signal, we strictly
  reduce what these learnable / Hz-parameterised filters can express. The
  comparison would be biased toward F0 (vanilla strided conv, no learned
  filtering) since F0 alone benefits from offline filtering it cannot learn
  itself. Hence we feed *raw* signal to all five frontends in the headline
  ablation cell.
- **exp05 (multi-rate strategy)**: the entire question is "how should
  multiple device sample rates be handled without losing high-γ content
  above 100 Hz Nyquist". Resampling everything to 250 Hz before the
  frontend sees it predetermines this experiment. We keep HBN at 500 Hz
  native and bring in Sleep-EDF (100 Hz) and THINGS-EEG2 (1000 Hz) at
  *their* native rates for exp05; the held-out 2000 Hz test is constructed
  by information-preserving upsampling.
- **exp14 (context-length scaling)**: a 30-second window at 2 kHz is
  60,000 samples — exactly the regime where Mamba-2's O(N) attention
  beats Transformer's O(N²). Resampling to 250 Hz reduces 60,000 →
  7,500 samples and makes the long-context argument moot.

A **literature-comparability cell** is run exactly once, in exp02: F0
vanilla strided conv is additionally trained on a *preprocessed* input
stream (60 Hz notch + 0.5–100 Hz Butterworth bandpass + 500 → 250 Hz
polyphase resample applied per recording before iid expansion). This cell
is the apples-to-apples cell against BENDR / LaBraM-Base / CBraMod / REVE,
which were all measured on preprocessed input. It is reported in the
exp02 results table but is **not** used for the §4.4 winner-picker
decision rule — it exists purely so we can claim a comparable number for
the literature.

The 100-hour HBN-EEG subset is sampled subject-disjoint from the train
split; LNSO (leave-N-subjects-out) eval splits are constructed from the
remainder. Source: AWS public bucket `s3://fcp-indi/data/Projects/HBN/EEG/`
(BIDS-iEEG layout, EEGLAB `.set/.fdt`); fall-back to OpenNeuro `ds005516`.

HBN-EEG ([Healthy Brain Network EEG, Shirazi et al. 2024 bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2))
is a 3,000+-subject pediatric/young-adult corpus (ages 5–22), 128-channel EGI
HydroCel @ 500 Hz native, openly distributed on AWS public storage
(`s3://fcp-indi/data/Projects/HBN/EEG/`) and OpenNeuro (`ds005516`) — no
credentialed-access wait. 100 hours × 128 channels at 4-second non-overlapping
windows gives roughly **11.5 million training examples** after iid expansion
(invariant to sample rate; each example is a 2,000-sample float16 vector at
the native 500 Hz, ~50 GB total at parquet float16) — enough to be in the
regime where SSL signal overcomes initialization noise, small enough that one
full pretraining pass is 4–8 hours on H100 at this sequence length.

**Why HBN-EEG over TUEG.** TUEG (Temple University EEG Corpus) is the *de
facto* pretraining corpus in the EEG-FM literature (BENDR, LaBraM, CBraMod,
REVE all use it). We deliberately switch to HBN-EEG for these
mini-experiments because:

1. **Open access, no application wait.** HBN is on AWS public storage and
   OpenNeuro under CC0; we can sync it into our warehouse bucket today and
   bootstrap mini-experiment 01 immediately. TUEG is gated via TUH NEDC
   (~1–2 business days for credentials).
2. **2.5× more iid examples per recording-hour.** HBN's 128-channel montage
   versus TUEG's typical 23-channel montage means each hour of recording
   yields ~5.6× more (subject, channel) iid pairs. After matching for the
   100-hour budget, we get ~11.5M iid examples vs TUEG's ~4.5M — better
   coverage of the iid-expansion axis we actually scale on.
3. **Modern uniform montage.** HydroCel-128 is a single sensor topology
   collected with a single device class at four CMI sites. TUEG is a
   multi-decade, multi-device clinical corpus with ~7 distinct montages and
   a wide range of sampling rates that have to be reconciled in
   preprocessing — the source of much of the §4.4 "predict source
   dataset" anti-shortcut headache.

**Tradeoff acknowledged.** HBN is pediatric/young-adult and clinically
enriched (~50 % at least one DSM-V diagnosis: ADHD, anxiety, learning
disorders), so this corpus teaches features that are *low-noise scalp EEG of
adolescents*, not adult clinical EEG. A model pretrained on HBN evaluated on
TUAB (adult clinical) is a **cross-distribution** test of representation
quality, which is a *stronger* probe of "did SSL learn universal features"
than in-distribution eval (per the same DeeperBrain argument that motivates
frozen-probing over fine-tuning, §4.3 below). For literature comparability,
TUAB + TUEV are kept as a *secondary* eval (Protocol A.4) and reported
alongside the primary HBN metrics once TUH NEDC access lands.

### 4.2 Default architecture (for any axis not under test)

When an experiment varies one axis, the others are pinned to:


| Axis                | Default                                                                            | Why this default                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Frontend            | 3-layer 1-D conv with kernels (7, 7, 7), strides (2, 2, 2), GeLU, no anti-aliasing | The dumbest reasonable baseline. Lets the other axes' improvements be measured on top of a known floor.     |
| Backbone            | Bidirectional Mamba-2, 6 layers, d=256, state N=64                                 | Best published O(N) backbone for long 1-D biosignals (EEGM2, FEMBA), H100-optimised, clear winner of exp03. |
| Bottleneck          | Continuous (no quantization)                                                       | Lowest implementation risk, most published EEG-FM use this, removes a confound.                             |
| SSL framework       | MAE with 50% random mask, decoder reconstructs raw signal                          | Standard, widely understood, no teacher, no EMA.                                                            |
| Reconstruction loss | L1 on raw signal + 0.3 × multi-resolution STFT log-magnitude                       | Reasonable starting point per [BioCodec](https://arxiv.org/abs/2510.09095).                                 |
| Phase               | Magnitude-only spectral loss (no explicit phase term)                              | Default. Whether to add phase is exp07.                                                                     |
| Mask                | Random patch mask, 50% ratio                                                       | Vanilla MAE default.                                                                                        |
| Multi-rate          | Native 500 Hz HBN, no offline resampling                                            | The minimum-offline §4.1 spec — resampling is a hypothesis (exp05), not a default. Multi-rate aux corpora (Sleep-EDF 100 Hz, THINGS-EEG2 1000 Hz) only enter in exp05. |


This pinning is critical: when exp02 tests 5 frontend variants, the **only**
difference between them is the frontend itself — backbone, framework, loss,
masking, etc. are all identical. Otherwise the result is uninterpretable.

### 4.3 Evaluation suite (for every experiment unless overridden)

We treat **frozen-probing** as the primary evaluation protocol and
end-to-end fine-tuning as a secondary one. The reason, surfaced
explicitly by [DeeperBrain (Wang et al., arXiv 2601.06134)](https://arxiv.org/abs/2601.06134)
Table IV, is that several recent EEG foundation models (LaBraM,
CBraMod, CSBrain, REVE) match each other to within ~2 pp under
fine-tuning but diverge by 10–35 pp under frozen probing — sometimes
collapsing to chance (e.g. CSBrain on SEED-V at 20.0 % balanced
accuracy = chance for 5 classes; CBraMod on ISRUC at 37.27 %; REVE
on PhysioNet-MI at 25.01 % = chance for 4 classes). Fine-tuning hides
representational quality because the encoder gets to re-optimise its
parameters for each task; the frozen-probing score is the more honest
test of "did the SSL produce useful features".

Run the same eval suite after every pretraining cell, in two protocols:

**Protocol A — frozen-probing (primary).** The encoder is frozen
end-to-end; only a single-layer linear head (or k-NN) is trained on top
of the mean-pooled encoder output. This is the canonical "linear probe"
of the image-SSL literature and is what we report as the headline
metric for every comparison.

1. **Linear probe on HBN ADHD-vs-no-diagnosis binary** (primary). Frozen
   encoder, single-layer linear head, train on HBN train split (subject-
   disjoint LNSO), evaluate on HBN held-out subjects. Metric: AUROC.
   ADHD is the most common HBN clinical label and is well-balanced
   (~40 % positive); the binary task replaces TUAB's normal-vs-abnormal
   role in the eval suite.
2. **Linear probe on HBN 6-task classification** (primary) — which of the
   six cognitive tasks the subject is performing: resting state (eyes
   open + closed), sequence learning, symbol search, surround suppression,
   contrast change detection, video watching. Same protocol. Metric:
   balanced accuracy and weighted F1. Perfect 6-class symmetry with the
   role TUEV played in the original spec.
3. **k-NN on a 10k-sample HBN subset** (k = 5, cosine distance on encoder
   pooled output). Metric: top-1 accuracy. Cheap, architecture-independent,
   detects representational collapse without training a probe at all.
4. **(Secondary, when TUH NEDC access is approved) Linear probe on TUAB
   binary AUROC + TUEV 6-class BAC/WF1** — same protocol as 1 and 2, on
   the canonical EEG-FM literature benchmark. Reported for direct
   comparability with LaBraM, CBraMod, BIOT, REVE; **not** used for the
   §4.4 winner-picker decision rule. Until TUH access lands, this row is
   left blank in results tables. Note that pretrain-on-HBN →
   eval-on-TUAB/TUEV is a cross-distribution test (pediatric → adult
   clinical), which is a *more* honest probe of universal representation
   quality than same-distribution eval — see DeeperBrain reasoning above.

**Protocol B — end-to-end fine-tuning (secondary).** The full encoder
is unfrozen; the same head is added; everything is jointly optimised on
the downstream task. We report this for completeness because it is the
field standard, but a variant that wins on B and loses on A is
*disqualified* — it has not learned a universal representation, only
a transferable initialisation.

The decision rules in §4.4 use the *Protocol A delta* (frozen-probing
delta over the matched control) as the primary input. Protocol B is a
sanity check, not a winner-picker. A variant must (i) not hurt
fine-tuning by more than 0.5 pp and (ii) win in frozen probing by the
margin its experiment specifies.

**Label-free monitors**, logged every checkpoint, are unchanged:

- Encoder feature std (should be stable)
- Encoder feature absmax / std ratio (should be bounded)
- Encoder covariance rank (should stay > 0.5 × feature dim — see
  [methodology.md §6.1](./methodology.md#61-encoder-feature-health))
- "Predict recording site" linear probe on encoder features (HBN was
  collected at four CMI sites: RU, CBIC, CUNY, SI) — should **decrease**
  over training, otherwise the model is learning site-rig fingerprint
  rather than neural content. Plus a "predict subject ID" k-NN on a
  held-out batch as a stronger anti-shortcut signal (an encoder that
  identifies subjects too well has likely learned a per-subject
  fingerprint instead of a transferable representation).

Total eval time per cell: ≈ 15 minutes on one H100 (now that frozen
probing and fine-tuning both run). Still trivial overhead relative to
the pretraining cost.

### 4.4 Matched-noise twin (mandatory)

Every experiment runs each cell *twice*: once on the actual EEG, once on
matched Gaussian noise of identical mean and variance per epoch. The noise
twin must have a 95% bootstrap CI on every downstream metric that does not
overlap the EEG cell, otherwise the result is *not credible* — the model is
hallucinating from architectural priors, not learning EEG structure. This is
the [Jo et al. 2024](https://arxiv.org/abs/2405.06459) protocol enforced at
every ablation level, not just at the headline run.

### 4.5 Statistical protocol

For every comparison between two cells:

- 5 seeds per cell × 5 folds where applicable (cross-subject splits).
- Report mean ± std on every metric.
- Decision rule for "X beats Y": one-sided paired sign-flip permutation test
at α = 0.05 on the seed-paired metric differences, with a pre-registered
effect size threshold (specified per experiment).
- Decision rule for "X is equivalent to Y": TOST equivalence test with
pre-registered margin ε (specified per experiment).

This is the same protocol used in
`[../exp02_eeg_ctc/](../exp02_eeg_ctc/)` matched-pair §4.3 tests, applied
uniformly here.

### 4.6 Storage and reproducibility

Every run writes to `mini_experiments/NN_name/runs/<run_id>/`:

- `config.yaml` — fully resolved config (no defaults left implicit)
- `metrics.jsonl` — every metric per epoch
- `eval.json` — final downstream metrics
- `feature_health.jsonl` — std / absmax / rank / source-probe per checkpoint
- `seed.txt`, `git_sha.txt`

Each experiment's `README.md` ends with a `results` table that gets filled in
as runs complete. No run is "done" until that table is updated.

---

## 5. What this list deliberately does not cover

To stay honest about scope:

- **No cross-channel modelling experiments.** The iid-channel decision is
upstream and being separately tested in
`[brain/experiments/pretraining-experiment/](../../../../brain/experiments/pretraining-experiment/)`.
- **No teacher-student, no EMA momentum encoder, no two-stage tokenizer
pretraining, no distillation.** Ruled out by the scratch SSL constraint.
Means SALT, BYOL, DINO, data2vec, EEG2Rep, EEGPT, LaBraM, NeuroLM,
CBraMod-style, MTDP are all out of scope as full systems (their loss-level
or architectural ideas may still be borrowed inside one of the cells).
- **No headline 60k-hour pretraining run.** That is Phase 4 from
`[methodology.md` §2](./methodology.md#phase-4--headline-run-the-actual-pretraining)
and only happens after these mini-experiments have settled the
configuration.
- **No EEG-to-text fine-tuning evaluation.** The eval here is restricted to
the HBN-derived linear probes + k-NN (and TUAB/TUEV as a secondary check
once TUH access lands), which are cheap and run in <10 minutes per cell.
Whether the resulting encoder is good enough to feed into the ZuCo /
Brennan / Chisco fine-tunes is decided once we have a winner.
- **No hyperparameter optimisation beyond LR sweeps.** Per
`[methodology.md` §4](./methodology.md#4-hyperparameter-transfer-when-small-scale-tuning-is-trustworthy),
every recipe gets a 3-LR × 3-seed sweep at proxy scale; no exotic
hyperparameter search. Width / depth scaling is reserved for Phase 3.

---

## 6. Cross-references

- `[methodology.md](./methodology.md)` — the playbook these mini-experiments
instantiate.
- `[../exp01_eeg_to_text/](../exp01_eeg_to_text/)` — preprocessing
conventions (notch, bandpass, resample, normalise) and the V2 audit
template.
- `[../exp02_eeg_ctc/](../exp02_eeg_ctc/)` — matched-pair statistical test
template, "predict source dataset" anti-shortcut probe, run-storage
layout (`runs/<id>/stats.jsonl`).
- `[brain/experiments/pretraining-experiment/](../../../../brain/experiments/pretraining-experiment/)`
— the channel-aggregation experiment that decides the iid-channel
framing upstream of this folder.
- `[brain/eeg-research.md](../../../../brain/eeg-research.md)` §6, §7,
§10 — the literature behind the variants tested here.
- `[brain/asr-research.md](../../../../brain/asr-research.md)` §3, §4, §5,
§6 — the speech-side cousins that motivate many variants (multi-condition
pretraining, MR-STFT loss, MSR-HuBERT branches, anti-aliased CNNs).

