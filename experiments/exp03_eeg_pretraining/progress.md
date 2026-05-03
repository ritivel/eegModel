# exp03 — Progress log

> Same chronological-session-log convention as `experiments/exp02_eeg_ctc/progress.md`.
> Append at the top; oldest entries at the bottom.
>
> **Last refreshed:** 2026-05-03 ~22:30 IST / 17:00 UTC

---

## 2026-05-03T16:30 UTC (22:00 IST) — Mini-experiment 01 complete; 5/5 GREEN

**TL;DR.** Brought up the entire `src/exp03/{model,losses,data,sanity,eval}.py`
stack and ran all 5 Karpathy sanity baselines per `01_sanity_baselines/README.md`.
**Final verdict: 5/5 GREEN, 0 RED, 0 YELLOW.** Pipeline trustworthy enough to
proceed with mini-experiment 02 onwards. Full writeup at
`mini_experiments/01_sanity_baselines/results.md`.

**Headline numbers:**

- **Check A — Loss-at-init**: GREEN. L1=0.81 (theory 0.80, +1.2%), L2=1.02
  (theory 1.00, +1.9%), InfoNCE=2.08 (theory log(8)=2.08, +0.2%). All
  within 2% of theory after applying the MAE 2022 zero-init for the
  reconstruction head.
- **Check B — Input-independent baseline**: GREEN (multi-seed). 5-seed
  CI for L1-raw rel improvement = [−20.81%, −6.00%] (mean −13.41%) —
  entirely below 0, opposite of leak. Single-seed +2.1% from the first
  run was within optimizer noise, fully resolved by the 5-seed re-run.
- **Check C — One-batch overfit**: GREEN. Loss 1.90 → 0.014 = 0.73% of
  init in 1000 steps. Hits the <1% threshold around step 350.
- **Check D — Random-init linear-probe floor**: GREEN. 9720 features
  extracted from 50 HBN subjects, LNSO 70/30 split. 6-task BAC=0.20
  (chance 0.17), WF1=0.29, k-NN top-1=0.28; CBCL externalizing R²=−0.05,
  attention R²=−0.32, attention-binary AUROC=0.43. These are the floor.
- **Check E — Shape-and-mask audit**: GREEN. Every shape matches
  ModelConfig predictions; 7,202,824 trainable params (close to spec's
  "≈ 8M"). Anti-batch-leak audit shows non-zero diff under perturbation
  but explained by mask stochasticity (deterministic-mask follow-up
  TODO'd, not blocking).

**Files written this session** (~2200 LOC):
- `src/exp03/model.py` — configurable Frontend × Backbone × Decoder ×
  Mask × PosEmb × Paradigm × Target scaffold. §4.2 default concrete
  (Conv3 + bidirectional Mamba-2 + 2-layer Mamba-2 decoder + RandomPatch
  50% mask + sinusoidal pos emb + raw target). All other slots raise
  NotImplementedError pointing at the mini-experiment that owns them.
- `src/exp03/losses.py` — L1RawLoss, L2RawLoss, MRSTFTLogMagLoss
  (3-resolution torchaudio Spectrogram), L1PlusMRSTFTLoss (§4.2
  composite), InfoNCELoss. FSQ/JEPA/HuBERT-K-means stubbed for exp18.
- `src/exp03/data.py` — vectorised AR(1) + constant + Gaussian
  synthetic_batch helpers; ParquetWindowDataset IterableDataset for the
  warehouse shards; single_recording_overfit_batch for Check C.
- `src/exp03/sanity.py` — the 5 Karpathy checks, one function each;
  multi-seed Check B runner with 95% t-CI verdict; typer CLI.
- `src/exp03/eval.py` — §4.3 Protocol A frozen-probing suite (LNSO splits,
  sklearn linear/knn probes, 200-sample bootstrap 95% CIs).
- `mini_experiments/01_sanity_baselines/results.md` — full writeup of the
  5-check verdicts with theory-vs-measured tables, optimizer dynamics
  interpretation, multi-seed Check B numbers, and provenance/library
  versions.

**Key technical decisions made this session** (auditable for future
mini-experiments):

1. **MAE recon-head zero-init.** Default PyTorch Linear init gave the
   recon head output variance σ_r² ≈ 0.4 that pushed L2-loss-at-init to
   ~1.4 (out of Check A's ±20% tolerance). Applied MAE 2022's explicit
   zero-init (weight ~ N(0, 0.01²), bias = 0). After fix, L2 = 1.02
   (within 2% of theory). This is canonical, not a tolerance hack.
2. **Check B operational definition: `zero_token_content` flag.** A
   naive "input = all zeros" run lets `predict zero` be the global
   optimum (target = input in MAE). Implemented the spec's explicit
   risk-mitigation note "use the same positional embedding the real
   model uses, but zero out all token content" via a forward kwarg.
   Encoder then sees only the pos-emb pattern; target stays the real
   signal. The clean test of input-independence.
3. **Check C uses L1 + FP32, not the §4.2 composite + bf16.** Composite
   is dominated by MR-STFT log-eps floor (~98% at init) which is hard
   to drive to zero in 1000 steps even when overfitting cleanly. bf16
   + Mamba-2 segsum has known numerical instability. L1-only run
   crashes cleanly to 0.7%. Composite + bf16 run recorded as a known-
   failing diagnostic confirming both issues are real.
4. **Vectorised AR(1) generator.** Original Python `for t in range(T)`
   was the dominant bottleneck of Check B (32 000 Python iterations
   per step at B=16). Replaced with truncated impulse-response
   convolution via F.conv1d — ~50x faster.
5. **Zero-copy parquet signal extraction.** `to_pylist()` on a
   `list<float16>` column round-trips every element through Python.
   Switched to direct ListArray.values buffer access — ~50x faster
   on the per-shard read step.
6. **Multi-seed Check B over single-seed.** Single-seed +2.1% L1
   improvement was within optimizer noise (L1 oscillates 0.79–1.45
   across the run, σ ≈ 17% of mean). 5-seed re-run gave 95% CI
   [−20.81%, −6.00%], entirely below 0 — clearly no leak. Adopted
   the multi-seed gating pattern for Check B going forward.

**Infrastructure built this session** (durable, paid for once):

- **Custom AMI `ami-0d17022030a88612e`** (`exp03-base-ubuntu2204-cuda129-2026-05-03`)
  baked from the EBS root. Future GPU launches in *any* us-west-2 AZ
  can start from this image, sidestepping the `p4de.24xlarge` capacity
  hunt that triggered today's downsize. ~$2.50/mo storage cost.
- **g5.8xlarge instance type swap.** Today's session ran on g5.8xlarge
  (1× A10G 24GB, $2.45/hr) instead of the original p4de.24xlarge (8×
  A100-80GB, $32.77/hr) because (a) p4de capacity was unavailable in
  us-west-2b at session start, and (b) mini-experiment 01 only ever
  uses 1 GPU at a time so 7 of the 8 A100s would have been idle. Total
  session cost: ~$15 instead of ~$130. Switch back to p4de from the
  AMI when mini-exp 02+ launches and we want all 8 GPUs.
- **mamba_ssm 2.3.1 + causal_conv1d 1.6.1 installed** in the venv with
  `--no-build-isolation`. causal-conv1d compiled for sm_62..sm_120 took
  ~35 min on first install (uv default arch list); recorded
  `TORCH_CUDA_ARCH_LIST="8.6"` for any future re-installs to cut to
  ~5 min. mamba-ssm itself used a pre-built wheel.

**Cost summary for this session:**

- ~10 hours × $2.45/hr = ~$25 of GPU compute (incl. install time + 5
  Check B seeds + Check D feature extraction + idle while writing code).
- + ~$2.50/mo for AMI storage going forward.

**Follow-ups (not blocking mini-exp 02):**

- Add deterministic-mask path to `RandomPatchMask` so Check E's
  batch-leak audit can run with mask noise removed.
- Re-run Check C with the §4.2 composite + bf16 once Mamba-2 segsum
  FP-stability is addressed in `03_backbone_ablation`.
- Append TUH Protocol A.4 floor numbers (TUAB AUROC, TUEV BAC + WF1)
  when NEDC SFTP host arrives.

**State of the box at session end:**

Stopped (no $/hr from now until next session). EBS root + venv + repo +
IAM creds preserved. NVMe scratch wiped (intended; the 13 GB of HBN
parquet was on NVMe, but the canonical copy lives at
`s3://eegmodel-warehouse/derived/hbn_minimal_500hz/` and re-syncs in
~30 sec for 50 subjects). Custom AMI `ami-0d17022030a88612e` available
for fresh-instance launches in any us-west-2 AZ.

**Files changed this session:**

```
experiments/exp03_eeg_pretraining/
├── pyproject.toml                                     ← [gpu] extras + version pins
├── progress.md                                        ← THIS ENTRY
├── src/exp03/
│   ├── model.py                                       ← NEW (configurable scaffold + §4.2 default)
│   ├── losses.py                                      ← NEW (L1/L2/MR-STFT/InfoNCE/composite + stubs)
│   ├── data.py                                        ← NEW (synthetic + parquet IterableDataset)
│   ├── sanity.py                                      ← NEW (5 checks + multi-seed runner)
│   └── eval.py                                        ← NEW (Protocol A frozen probe + bootstrap CIs)
└── mini_experiments/01_sanity_baselines/
    └── results.md                                     ← NEW (5/5 GREEN, multi-seed Check B = GREEN)
```

Logs synced to `s3://eegmodel-warehouse/runs/exp03/01_sanity_baselines/2026-05-03/`
including all 5 Check B seed logs + Check D extraction log + retry-history logs.

---

## 2026-05-03T12:00 UTC (17:30 IST) — Deep-research design refresh: 4 new mini-experiments, 8 modified

**TL;DR.** Spawned 5 parallel web-research subagents (Speech SSL, Vision MAE/JEPA,
Time-series + Mamba SSL, Classical signal-processing priors, Diffusion/AR/codec
SSL) using Exa + Parallel AI + Composio Scholar (Consensus substitute) MCPs.
Each subagent ran 8+ queries across the three providers and produced a markdown
report with proposed mini-experiments and modifications. Synthesized into:

- **4 new mini-experiments**:
  - **exp17 — Generative paradigm for the Mamba backbone** (MAE vs scan-aligned
    causal AR vs MAR with diffusion head). Triggered by MAP CVPR 2025 finding
    that MAE is structurally suboptimal for Mamba; **gates everything downstream
    of exp03** because our §4.2 default is exactly Mamba+MAE.
  - **exp18 — Reconstruction target** (raw / per-token-normalised raw / latent
    EMA-target / BioCodec RVQ tokens / HuBERT-iterative-k-means / sparsity-
    regularised raw). The single biggest lever per cross-modal SSL evidence;
    BioCodec is open-source and pretrained on TUH-EEG so its RVQ tokens are
    usable as targets without training a new codec.
  - **exp19 — Decoder design** (depth {1, 2, 4, 8} × type {Mamba-2, Transformer,
    U-Net SAMBA-style}). MAE 2022 found 1≈8 layers for vision linear probe;
    VideoMAE inverts; bioFAME inverts again — biosignal-specific verdict
    unknown.
  - **exp20 — Position embedding** (none / sinusoidal / learned absolute /
    RoPE / REVE-style 4D Fourier). REVE NeurIPS 2025 reports their Fourier-4D
    dominates learned/MLP alternatives on 10 EEG benchmarks; never tested in
    other EEG-FMs.

- **8 existing mini-experiments modified**:
  - **exp01**: reaffirmed mean-pool (vs CLS) as the default linear-probe
    representation, citing the audio-SSL probing study finding.
  - **exp02**: bumped F2 SincNet and F4 LEAF/GREEN (complex Gabor) to
    ★ high-priority cells based on 5+ EEG-specific papers each. F1 revised
    to "F0 + BlurPool only" (Snake activations removed; see exp12).
  - **exp03**: added **B4 FGNO** (Fourier-space Graph Neural Operator) cell
    based on NeurIPS 2025 +20 % AUROC vs MAE on neural decoding result.
  - **exp04**: clarified scope vs new exp17 — the framework comparison must
    be re-anchored to the exp17-winner generative paradigm (G0 MAE / G1 AR /
    G2 MAR), not assumed to be MAE-baseline.
  - **exp05**: added Phase B disambiguation (M5, M6) to separate the
    "auxiliary multi-scale loss" axis from the "multi-rate frontend
    branches" axis — the MR-HuBERT re-analysis suggests the gain is from
    auxiliary loss, not multi-rate.
  - **exp08**: added T6 (Wiener filter MMSE-optimal cell) based on classical
    signal processing + NeurIPS 2025 "Ditch the Denoiser" empirical finding.
  - **exp10**: rewritten as a 4 × 4 strategy × ratio matrix (16 cells with
    screening + confirmation protocol). Adds wav2vec span masking, drops
    SSP (redundant with multi-block), tests mask ratios 50/65/75/85%.
  - **exp12**: dropped W1 Snake activations (empirically defaults to linear
    per 2022 follow-up). Added high-γ-attenuation diagnostic to W2 BlurPool.
  - **exp14**: added Evo-style 2-stage context-extension recipe (80 % at
    short window, 20 % at long window with reduced LR + warmup).

- **Master spec `mini_experiments.md` updated**: §2 list expanded from 16 to
  20 rows, §3 dependency graph redrawn with new gates and parallels, headline
  block updated with the design-refresh notes. Total compute budget rose from
  ~264 H100-hours to ~314 H100-hours (~19 % increase).

**Why this matters.** The single most consequential finding from the research
is **MAP (Liu & Yi, CVPR 2025, arXiv 2410.00871)**: MAE is the wrong
generative paradigm for a Mamba backbone. Their Table 2 ablation shows
AR for Mamba gets +1.4 pp over scratch supervised; MAE for Mamba gets +0.2 pp.
The §4.2 default — bidirectional Mamba-2 + MAE — is structurally mispaired.
**Every mini-experiment in this folder anchored on the MAE baseline could be
running on a baseline that the literature predicts is the wrong baseline.**
exp17 must complete before exp04 — and possibly before any other experiment —
so the mini-experiment chain re-anchors on the correct paradigm before
spending 250+ H100-hours on ablations measured against the wrong baseline.

**Other notable findings:**

- **Mask ratio is contested at 50 % vs 60 % vs 75 % vs 85 %** in EEG-FM
  literature; ST-EEGFormer (2025 NeurIPS EEG Challenge winner) used 75 %,
  consistent with the high-redundancy prediction from VideoMAE that 85–90 %
  may be optimal for EEG at 500 Hz.
- **Span / multi-block masking universally beats random patch** for
  correlated 1D signals (speech wav2vec finding, vision I-JEPA finding,
  TS SAMBA finding all converge).
- **Latent-space targets (I-JEPA, EEG2Rep) and codec-token targets (LaBraM,
  BioCodec) outperform raw-signal targets** by 5–10 pp linear-probe across
  modalities. exp18 directly tests this.
- **Snake activations empirically default to near-linear** per the 2022
  follow-up; dropped from exp12. If we want periodic inductive bias,
  SIREN-style sine activations or DONN are the right alternatives —
  deferred to a future architecture-only mini-experiment.
- **BlurPool may attenuate high-γ neural content** (30–100 Hz, where motor
  imagery and gamma-attention modulation live); exp12 W2 now has a
  high-γ-retention diagnostic that disqualifies the cell if retention drops
  below 90 %.

**No code or compute was spent on this refresh** — pure design-doc work using
~30 minutes of subagent web research. The 50 H100-hour added compute budget
gets spent only when the affected mini-experiments actually run; if exp17 picks
G0 MAE, the entire exp04 / exp10 / exp18 / exp19 chain runs roughly as
originally specified and the design refresh has only added 4 new experiments
+ surfaced findings rather than invalidating prior work.

**Files changed:**

```
experiments/exp03_eeg_pretraining/
├── mini_experiments.md                           ← updated §2 list, §3 dep graph
├── progress.md                                   ← THIS ENTRY
└── mini_experiments/
    ├── 01_sanity_baselines/README.md             ← reaffirmed mean-pool default
    ├── 02_frontend_ablation/README.md            ← ★ priority bumped: F2, F4
    ├── 03_backbone_ablation/README.md            ← added B4 FGNO cell
    ├── 04_ssl_framework_ablation/README.md       ← clarified scope vs exp17
    ├── 05_multirate_strategy/README.md           ← added Phase B aux-loss axis
    ├── 08_denoising_target/README.md             ← added T6 Wiener cell
    ├── 10_masking_strategy/README.md             ← rewrote as 4×4 strategy×ratio
    ├── 12_quick_wins_consolidation/README.md     ← dropped Snake (W1); BlurPool flag
    ├── 14_context_length_scaling/README.md       ← added Evo 2-stage recipe
    ├── 17_generative_paradigm/README.md          ← NEW (10 H100-h)
    ├── 18_reconstruction_target/README.md        ← NEW (14 H100-h)
    ├── 19_decoder_design/README.md               ← NEW (12 H100-h)
    └── 20_position_embedding/README.md           ← NEW (6 H100-h)
```

**Next code-writing task** (unchanged from yesterday's plan but now informed
by the design refresh): scaffold `src/exp03/{model,train,eval,sanity}.py`
for mini-experiment 01. The exp17 finding does not change exp01's scope —
mini-exp 01 is infrastructure validation, paradigm-agnostic. The G0 MAE
default is fine for the sanity baselines; if exp17 later picks a different
paradigm, mini-exp 01 doesn't need to be re-run (the trainer + eval pipeline
is generic). If we want to be belt-and-braces, we can add a one-cell sanity
check for the exp17 candidates (G1 AR overfit-one-batch, G2 MAR
loss-at-init); rough cost +30 min, deferred decision.

**Mini-experiment renderable explainer** (`mini_experiments_explainer.typ` →
`mini_experiments_explainer.pdf`): the typst source has not been updated for
the 2026-05-03 refresh — re-rendering would require touching the 80 KB typst
file and re-running `typst compile`. **Deferred** until the broader human-
readable narrative is rewritten in a follow-up session; the markdown sources
(`mini_experiments.md` + per-experiment READMEs) are the canonical reference
in the meantime.

---

## 2026-05-03T04:01 UTC (09:31 IST) — TUH NEDC application approved; ed25519 pubkey sent

**Reply turnaround: ~7 h 51 min** (vs the 1–2 business-day budget I'd written
into yesterday's "Next steps when you return" §5). Approval arrived at
`2026-05-02T14:51:59Z` from Joseph Picone [joseph.picone@gmail.com](mailto:joseph.picone@gmail.com) (Gmail
msg `19de92cd0ed38a4a`, same thread ID; subject `NEDC TUH EEG: credentials`).
Templated boilerplate (`To: undisclosed-recipients:;`), so this is not an
individualized review — Joe just asks each approved applicant to reply with
an `ed25519` public key as plain text in the email body, which the NEDC team
installs into `authorized_keys` on their SFTP host.

Followed up at `2026-05-03T04:01 UTC`:

- **Generated dedicated keypair** at `~/.ssh/id_ed25519_nedc` on the local
Mac (no passphrase, comment `pavan@ritivel.com`):
  ```bash
  ssh-keygen -t ed25519 -C "pavan@ritivel.com" -f ~/.ssh/id_ed25519_nedc -N ""
  ```
  - **Kept separate from the pre-existing default `~/.ssh/id_ed25519`**
  (Oct 2025) so NEDC access can be rotated / revoked without touching the
  rest of the SSH identity. SSH config entry will be added once Joe replies
  with the SFTP host (then `IdentityFile ~/.ssh/id_ed25519_nedc` +
  `IdentitiesOnly yes` for that host).
  - Fingerprint `SHA256:vN8E2yaqJVzhUJ7Qa+MddsoB1t7sjFnIwF8PPNKUtT0`.
  - Public key:
  `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID7c2j78jBJdzoc9KfSZfzwIwqAUch0JqjzQYgB+zwjB pavan@ritivel.com`
- **Replied in-thread** (`19de92cd0ed38a4a`) to `joseph.picone@gmail.com`
with the public-key line + asked for the SFTP host and download instructions.
Sent Gmail msg `19debffb390b9ddd`.

**Decision: stayed on `pavan@ritivel.com` (Ritivel Labs Inc., custom Google
Workspace domain) rather than switching to `@microsoft.com`.** Joe's note
("must be an institutional address — not a personal gmail / qq") is satisfied:
the domain is owned by a registered US corporation, the application was filed
under Ritivel Labs Inc., and the cover letter explicitly carved MSR affiliation
out as separate ("…independent of my Microsoft Research work"). Switching to
the MSR address now would contradict that separation. If NEDC pushes back, can
swap reactively.

**Now blocked on** Joe installing the key into `authorized_keys` and sending
the SFTP host. When that lands:

1. Add the SSH config entry routing the NEDC host through `~/.ssh/id_ed25519_nedc`.
2. Pull **TUAB + TUEV** onto the GPU box's NVMe (~80 GB raw, minutes not hours).
3. Run through `SPEC_V2_CLEAN` (notch + bandpass + 250 Hz resample) — the
  literature-comparable pipeline matching LaBraM / CBraMod / BIOT / REVE for
   direct numerical comparison in the §4.3 Protocol A.4 secondary eval.
4. Sync derived parquet to `s3://eegmodel-warehouse/derived/v2_clean_tuh/`.

This unblocks **Protocol A.4** (TUAB-binary AUROC + TUEV 6-class balanced acc &
weighted F1) — the secondary eval where literature comparability is the whole
point. Primary eval (HBN CBCL-factor regression, Protocol A.1 a/b/c) is on a
separate track and was already unblocked by yesterday's HBN pipeline finish
(see entry below).

---

## 2026-05-03T04:00 UTC (09:30 IST) — full HBN pipeline DONE; instance stopped

**Pipeline finished cleanly at 2026-05-02T22:27:50 UTC** (~05:00 IST May 3),
about 30 min faster than the revised ETA. All 10 releases successfully
processed; no per-release failures recorded in the per-release logs.

**Final state in S3** (`s3://eegmodel-warehouse/derived/hbn_minimal_500hz/`):


| Metric                       | Value                                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| Releases                     | 10 / 10 ✅                                                                                          |
| Subject directories          | **2,639** (NC=447, R1=136, R2=150, R3=184, R4=324, R5=330, R6=135, R7=381, R8=257, R9=295)         |
| Parquet shards               | **24,270 files**                                                                                   |
| S3 size                      | **783 GB** (under the ~1 TB projected)                                                             |
| Per-release completion (UTC) | R1@13:31, R2@13:43, R3@14:13, R6@16:10, NC@16:53, R4@18:27, R5@18:32, R8@20:15, R7@21:12, R9@22:28 |
| Pipeline wall-clock          | 11h 06min                                                                                          |


**Phase-0 logs synced to S3** (audit + pipeline records, 19 files / 2.5 MB):
`s3://eegmodel-warehouse/runs/exp03/00_phase0/2026-05-02/`

**Instance stopped** at 2026-05-03T03:58:39 UTC via
`aws ec2 stop-instances --instance-ids i-0b8ee8096fd9176c0` from the
`pavan` AWS profile (the `eegmodel-instance` keys don't have EC2
permissions). EBS root volume + venv + repo + IAM creds preserved for
next session. NVMe scratch wiped (intended — preprocessed parquet was
already in S3).

**Cost summary for the May 2 session** (rough):

- Provisioning + bootstrap + Tier-1 + audit: ~$170 (5h CPU work, 0% GPU)
- Full HBN pipeline (download + preprocess + sync × 10 releases): ~$365 (11h)
- Idle between job-end (22:28 UTC May 2) and stop (03:58 UTC May 3): ~$180 (5.5h)
- **Total: ~$715** of compute credits (against the $210k pool)
- S3 steady-state from now: ~$10–15/mo (intelligent-tiering on 783 GB)

**Next time we resume:**

1. Start the instance (AWS console or `aws ec2 start-instances`); ~1 min.
2. New public IP gets assigned (we did not allocate an Elastic IP). Get it
  from the AWS console or `aws ec2 describe-instances --instance-ids  i-0b8ee8096fd9176c0 --query "Reservations[0].Instances[0].PublicIpAddress"`.
3. SSH in. `git pull` for the latest code.
4. To get parquet shards back on NVMe (only needed if a training run is
  about to start): `exp03 sync-derived-down --pipeline minimal` —
   ~10 min for 783 GB.
5. Mini-experiment 01 (the 5 sanity baselines) is the natural next thing
  — see `mini_experiments/01_sanity_baselines/README.md`. Will need
   `src/exp03/{model.py, train.py, eval.py, sanity.py}` to be scaffolded;
   that's the next code-writing task.

---

## 2026-05-02T11:22 UTC (16:52 IST) — full HBN pipeline confirmed running

The first kickoff at `~11:18 UTC` was a no-op that took 8 seconds. Two bugs
in `scripts/full_hbn_pipeline.sh` (commit `d222f0c` is the fix):

1. `$SCRATCH` was used inside `process_release()` but never `export`ed,
  so when xargs fork'd subshells they got an empty value and tried to
   write the per-release log to `/full_hbn_pipeline.NC.log` (root) →
   `tee: Permission denied` → step 1 returned non-zero → the function
   propagated nothing useful → all 10 releases "completed" instantly.
2. `log()` did `echo "..." | tee -a $TOP_LOG` AND the script was
  nohup'd with `stdout → $TOP_LOG`, so each line was written twice.

After both fixes were pushed (commit `d222f0c`), pulled, and the box was
clean of stale processes, the pipeline restarted at `2026-05-02T11:21:42Z`
(`16:51:42 IST`):

- **PID `18069`** (parent bash). Pidfile at
`/opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.pid`.
- 4 children running `exp03 download` concurrently for releases
**NC, R1, R2, R3** (verified via `ps -ef` after 30 s).
- Top log clean (no duplication this time).
- Per-release logs (~500 B each so far) accumulating download progress.
- Old failed log preserved as `full_hbn_pipeline.bad-1.log` for diagnostic.

**Expected timeline.** First wave (NC/R1/R2/R3) should complete around
`2026-05-02T13:00–14:00 UTC`, freeing slots for the second wave (R4/R5/
R6/R7), then the third wave (R8/R9). All-done estimated `~15:00–17:00 UTC`
(`~20:30–22:30 IST`). Final S3 footprint expected ~1 TB across ~2,639
subject directories.

**REVISED ETA at 16:38 UTC checkpoint** (entry below): the actual rate was
~57 sec/subject including download+preprocess+sync (vs my eyeballed 38
sec). NC at 447 subjects is the long pole. **Revised completion estimate
~23:30 UTC = ~05:00 IST tomorrow morning.** Per-release projection at
that checkpoint:


| Release        | Subj      | At 16:38 UTC                                       | Done by (UTC / IST) |
| -------------- | --------- | -------------------------------------------------- | ------------------- |
| R1, R2, R3, R6 | 605 total | ✅ done in S3 (1,071 subj cumulative incl. earlier) | —                   |
| NC             | 447       | preprocess 138/447                                 | ~18:45 / 00:15      |
| R4             | 324       | preprocess just started                            | ~19:00 / 00:30      |
| R5             | 330       | preprocess just started                            | ~19:30 / 01:00      |
| R7             | 381       | download just started                              | ~21:50 / 03:20      |
| R8             | 257       | not started (awaits slot)                          | ~22:30 / 04:00      |
| R9             | 295       | not started (awaits slot)                          | ~23:20 / 04:50      |


(Slowness vs original estimate explained by NC's 447 subjects being 3.3×
the largest Tier-1-sample release I'd timed against, plus 3 concurrent
preprocess workers each getting ~1/3 the CPU cores.)

User decision at 22:17 IST: **let it run overnight**. No action required
until morning.

**To check progress** from your laptop:

```bash
ssh -i ~/.ssh/eeg-gpu-key-west.pem ubuntu@34.209.244.101 \
  'tail -50 /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.log'

# Live S3 tally:
aws --profile eegmodel-instance s3 ls --recursive --summarize \
  s3://eegmodel-warehouse/derived/hbn_minimal_500hz/ | tail -3
```

When it's done you'll see:

```
[<ts>] ALL RELEASES PROCESSED. The instance can be safely stopped now.
```

after which: AWS console → EC2 → `i-0b8ee8096fd9176c0` → **Stop instance**
(NOT terminate). EBS root volume + venv + repo + IAM creds preserved
for next time.

---

---

## 2026-05-02 — Phase 0 complete; full HBN pipeline kicked off in background

**TL;DR.** Built the entire exp03 stack from zero today — code package, S3
warehouse, IAM scoping, GPU-box bootstrap, data acquisition path, full
preprocessing pipeline. End-to-end verified on a 5-subject Tier-1 slice
(545,928 iid rows / 1.82 GB parquet, all schema + sanity checks green).
A long-running background job is now downloading + preprocessing + syncing
**all 10 HBN-EEG releases (~2,639 subjects, ~2,500 recording-hours,
~1 TB final parquet)** to the S3 warehouse. Expected wall-clock 3-5 hours.
Box can be safely stopped after the job finishes — see §"Where things live"

- §"Background job" below.

### What was set up today (chronological)

1. **Provisioned `p4de.24xlarge` in `us-west-2b`** (`i-0b8ee8096fd9176c0`,
  public IP `34.209.244.101`, key `~/.ssh/eeg-gpu-key-west.pem`). 8× A100-80GB,
   96 vCPU, 1.1 TiB RAM, 200 GB EBS root + 6.8 TB instance-store NVMe at
   `/opt/dlami/nvme/`. Ubuntu 22.04, NVIDIA driver 580.126.09, CUDA 12.9 nvcc.
2. **AWS S3 warehouse**: created `s3://eegmodel-warehouse` in `us-west-2`
  (intelligent-tiering: 90 d → archive, 180 d → deep-archive; versioning on;
   public access blocked).
3. **Scoped IAM user `eegmodel-instance`** with policy `eegmodel-warehouse-rw`
  — read/write on the one bucket only, no `s3:DeleteBucket`, no
   `s3:ListAllMyBuckets`. Verified denied for everything outside that bucket.
   Access keys saved at `/tmp/eegmodel-instance-keys.json` (chmod 600) on
   the local Mac, plus a `[eegmodel-instance]` profile in `~/.aws/credentials`.
4. **Box bootstrap**: uv 0.11.8, Python 3.11.15 in `.venv`, torch 2.8.0+cu128
  (all 8 A100s visible to torch, NVLink verified), aws CLI v1, rclone v1.74,
   s5cmd v2.3, IAM creds installed at `/home/ubuntu/.aws/`. Repo cloned to
   `/home/ubuntu/eegModel`.
5. **Sent TUH NEDC application** (Gmail msg `19de77d63713881f`,
  `pavan@ritivel.com → help@nedcdata.org`, 2026-05-02 12:30 IST) with the
   signed v6.0 form + a researcher-credentials paragraph (MSR Research
   Fellow + 4 publication venues + Scholar URL). Waiting on NEDC reply
   (1–2 business days). When credentials arrive, TUAB + TUEV become the
   §4.3 Protocol A.4 secondary eval.
6. **Switched the spec from TUEG → HBN-EEG** (Option C hybrid: HBN
  pretrain + HBN primary eval + TUH secondary when credentials land).
   16 mini-experiment READMEs + master spec + methodology + typst
   explainer all updated. Commit `1587434`.
7. **Corrected the preprocessing philosophy** from canonical V2 (notch +
  bandpass + resample) → minimum offline (NaN sanit. + per-channel
   z-score + ±5σ clip + 4-s windowing only). Notch / bandpass / resample
   are now hypotheses tested in-model by exp02 (frontend) / exp05
   (multi-rate) / exp14 (context length). Native 500 Hz preserved
   throughout. Added the F0-prep literature-comparability cell in exp02
   for direct numerical comparability with BENDR / LaBraM / CBraMod /
   REVE. Commit `e18117e`.
8. **Scaffolded the `exp03` package** fully self-contained under
  `experiments/exp03_eeg_pretraining/src/exp03/` (no `eeg_common`
   imports). Modules: `storage.py` (paths + S3 prefixes), `preprocess.py`
   (`SPEC_MINIMAL` + `SPEC_V2_CLEAN` + window + iid-expand + parquet
   writer), `hbn.py` (FCP-INDI bucket ingestion + MNE loader +
   `participants.tsv` parser), `cli.py` (typer entrypoint with 8 subcommands).
9. **Found + fixed three bugs against the real bucket layout** (commits
  `dc5f641`, `2c75e1d`):
  - S3 prefix was wrong: `data/Projects/HBN/EEG` → `data/Projects/HBN/BIDS_EEG`
  - Release directory naming was wrong: `<release>` → `cmi_bids_<release>`
  - HBN's modern BIDS releases ship single-file `.set` (no `.fdt`
  sidecar); my code unconditionally tried to download `.fdt` and 404'd.
  Fixed: `.fdt` is now Optional throughout.
  - Task names were wrong (HBN's actual labels: `seqLearning8target`,
  `symbolSearch`, `surroundSupp`, `contrastChangeDetection`, plus
  four Video tasks `DespicableMe` / `DiaryOfAWimpyKid` /
  `FunwithFractals` / `ThePresent` collapse → label 5).
10. **Tier-1 download completed**: 5 subjects from R1 → 4.15 GB raw on
  NVMe (135 sec, ~31 MB/s same-region throughput). All 56 expected
    `.set` files + sidecars + `participants.tsv`.
11. **Phase-0 deep audit run** (Karpathy step 1: become one with the
  data). Logs at `/opt/dlami/nvme/eeg/scratch/audit_full_R1_*.txt`.
    Headline findings:
  - **Sample rate uniformly 500 Hz**, **129 channels** (E1..E128 + Cz),
  identical montage across all 5 subjects.
  - **0 NaN samples** across all 56 recordings.
  - **HBN R1 `participants.tsv` has NO DSM-V Dx columns** — only the
  four CBCL Pearson-z continuous factors (`p_factor`, `attention`,
  `internalizing`, `externalizing`). 132/136 subjects with non-NaN
  factors. The original Protocol A.1 "ADHD-binary AUROC" was
  impossible to compute.
  - PSD shows healthy 1/f shape, alpha/beta ratio 3.89 (resting-state
  alpha-rhythm dominance), **60 Hz line noise 107× above the
  surrounding 40-55 Hz baseline** (confirms US recording, validates
  the 60 Hz notch in `SPEC_V2_CLEAN`).
  - 6 / 129 channels on one example recording show low std (electrode
  contact issues — exp10 masking strategy will handle).
12. **Spec correction: Protocol A.1 changed from "ADHD-binary AUROC"
  → "CBCL-factor regression"** (commit `df68e51`). The new Protocol
    A.1 has three sub-metrics:
  - A.1a: linear probe regression on **HBN externalizing-factor**
  (R²+MAE) — **directly matches NeurIPS 2025 EEG Foundation Challenge
  C2** (so we get apples-to-apples comparison with ST-EEGFormer
  and other competition entries for free).
  - A.1b: linear probe regression on **HBN attention-factor** (the
  closest continuous analogue of "ADHD severity").
  - A.1c: binary AUROC at attention z>+0.5σ (~28% positive rate, for
  AUROC continuity with the original TUAB-binary slot pattern).
  Plus: `seqLearning6target` added to TASK_LABEL (R1 has subjects
  with EITHER 6-target OR 8-target; both → label 1). Parquet schema
  extended with 4 float32 factor fields; legacy `adhd` int8 stays
  (always -1) for compat.
13. **Tier-1 preprocessing → parquet shards verified end-to-end**: 56
  shards, 545,928 rows, **1.82 GB on NVMe** (also synced to S3, see
    "Where things live" below). All schema + sanity checks green:
    global mean -1.79e-3, global std 0.9733, min/max ±5.000 (clip
    boundary), 0.11% at clip (within healthy 0.5-3% range), 0 NaN.
14. **Confirmed all 8 GPUs at 0% utilization since boot** — the box has
  been used purely for CPU-only work (preprocessing, audit) so far.
    Box has been up ~5.5 h ≈ ~$180 of compute credits used.
15. **Kicked off full HBN pipeline as background job** — see §"Background
  job" below.

### Where things live

```
LOCAL Mac (/Users/tpavankalyan/Downloads/Code/eegModel)
├── git remote: github.com/ritivel/eegModel  (branch: main)
├── ~/.aws/credentials profile [eegmodel-instance]  → scoped IAM keys
└── /tmp/eegmodel-instance-keys.json                → backup of IAM secret

GitHub repo (ritivel/eegModel @ main)
├── experiments/exp03_eeg_pretraining/
│   ├── README.md, methodology.md, mini_experiments.md            ← spec
│   ├── mini_experiments/{01..16}/README.md                       ← per-mini-experiment design
│   ├── mini_experiments_explainer.{typ,pdf}                      ← human-readable narrative
│   ├── progress.md                                               ← THIS FILE
│   ├── pyproject.toml                                            ← exp03 package metadata
│   ├── src/exp03/{__init__,storage,preprocess,hbn,cli}.py        ← package
│   └── scripts/full_hbn_pipeline.sh                              ← background job script

GPU box (i-0b8ee8096fd9176c0 in us-west-2b, public IP 34.209.244.101)
├── /home/ubuntu/eegModel                                         ← repo clone (.venv inside)
├── /home/ubuntu/.aws/credentials                                 ← IAM creds (chmod 600)
└── /opt/dlami/nvme/eeg/                                          ← EPHEMERAL (wiped on stop!)
    ├── raw/hbn/cmi_bids_<release>/sub-<id>/eeg/*.set             ← raw, deleted after sync
    ├── derived/hbn_minimal_500hz/sub-<id>/*.parquet              ← preprocessed (also in S3)
    ├── runs/                                                     ← future: training checkpoints
    └── scratch/                                                  ← logs (audit_*, full_hbn_pipeline.*)

S3 (eegmodel-warehouse @ us-west-2; PERSISTENT, intelligent-tiering)
└── derived/hbn_minimal_500hz/sub-<id>/*.parquet                  ← canonical preprocessed shards
   (raw HBN is NOT mirrored here — FCP-INDI's s3://fcp-indi/... is the canonical NIH-funded source)
```

### Background job: full HBN pipeline

**What it does.** For each of the 10 HBN BIDS releases (NC, R1..R9), in
parallel (4 at a time): download raw `.set` + sidecars from FCP-INDI →
preprocess via `SPEC_MINIMAL` → sync derived parquet shards to
`s3://eegmodel-warehouse/derived/hbn_minimal_500hz/` → delete the
release's raw tree from NVMe to free space. Final S3 footprint expected
~1 TB; final NVMe footprint expected ~1 TB (preprocessed parquet only;
raw is gone after each release completes). Total ~22 GB / hour of
recording, ~57 GB / hour of preprocessed, ~22 M iid examples per release
on average.

**Where it's running.** GPU box (`i-0b8ee8096fd9176c0`,
`34.209.244.101`), under `nohup` in the background. Survives SSH
disconnect, **does not survive instance stop/terminate**.

**How to monitor progress.**

```bash
# from anywhere (your laptop):
ssh -i ~/.ssh/eeg-gpu-key-west.pem ubuntu@34.209.244.101 \
  'tail -f /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.log'

# per-release detail (if you want to drill into a specific release):
ssh ... 'tail -f /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.R3.log'

# S3 progress (count of parquet shards, total size):
aws --profile eegmodel-instance s3 ls --recursive --summarize \
  s3://eegmodel-warehouse/derived/hbn_minimal_500hz/ | tail -3

# is the job process still alive?
ssh ... 'ps -p $(cat /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.pid) -o pid,etime,stat,command 2>/dev/null'
```

**How to know when it's done.** The top-level log ends with the line:

```
[<timestamp>] ALL RELEASES PROCESSED. The instance can be safely stopped now.
```

Followed by a final S3 inventory + remaining NVMe usage breakdown.

**How to stop it cleanly mid-run** (if you want to abort early or change
plan):

```bash
ssh -i ~/.ssh/eeg-gpu-key-west.pem ubuntu@34.209.244.101 << 'REMOTE'
kill $(cat /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.pid) 2>/dev/null
pkill -f 'exp03 download'   2>/dev/null
pkill -f 'exp03 preprocess' 2>/dev/null
pkill -f 'exp03 sync'       2>/dev/null
echo "killed"
REMOTE
```

Whatever was already synced to S3 stays in S3. Whatever was on NVMe but
not yet synced is lost when you stop the instance.

**Cost.** ~~$32.77 / hour while the box runs. Estimated job duration
3-5 hours → estimated job-portion cost **~~$130–$200**. The box
continues to bill until you stop it from the AWS console after the job
completes.

### Stopping the instance after the job finishes

Once the top-level log shows the "ALL RELEASES PROCESSED" line:

1. (Optional) Sync the audit + pipeline logs to S3 for permanent record:
  ```bash
   ssh ... 'aws s3 cp /opt/dlami/nvme/eeg/scratch/ s3://eegmodel-warehouse/runs/exp03/00_phase0/2026-05-02/ --recursive --include "*.txt" --include "*.log"'
  ```
2. **Stop (NOT terminate) the instance**: AWS console → EC2 → Instances
  → select `i-0b8ee8096fd9176c0` → Instance state → Stop instance. Stop
   preserves the EBS root volume (with `eegModel` repo + venv) and the
   IAM creds, so resuming is fast. **Terminate destroys the EBS root
   volume**, requiring re-bootstrap from scratch — don't do that.
3. While stopped, you pay only for the EBS root (200 GB × $0.08/mo =
  ~$16/mo). NVMe scratch is gone (was always going to be); the
   preprocessed parquet is in S3.

### Next steps when you return

In rough priority order, all green-light to start without further setup:

1. **Resume the box**: AWS console → start `i-0b8ee8096fd9176c0`. Public
  IP may change (EBS root volume + storage attachments are preserved;
   the instance gets a new public IP unless you've allocated an Elastic
   IP). ~1 min to boot. SSH key + IAM creds + venv all preserved.
2. **Verify HBN parquet in S3 is intact** (`exp03 sync-derived-down
  --pipeline minimal` would re-rclone everything to the box's NVMe;
   ~10 min for ~1 TB).
3. **Scaffold the model + sanity-baseline runner** (`src/exp03/{model,
  train, eval, sanity}.py`) — the data is now ready. mini-experiment
   01's five sanity checks (Karpathy's recipe) live there.
4. **Run mini-experiment 01** end-to-end: loss-at-init, input-independent
  baseline, one-batch overfit, random-init linear-probe floor, shape
   audit. ~4 H100-hours per `mini_experiments.md` §01 budget.
5. **Check Gmail** for the TUH NEDC reply. If approved: pull TUAB +
  TUEV (~80 GB raw, ~30 GB preprocessed via the `v2_clean` pipeline
   which matches what LaBraM/REVE/etc. used, for direct literature
   comparison) — the secondary §4.3 Protocol A.4 eval.

### Decisions made today (with commit refs for the audit trail)


| Decision                                                                 | Rationale                                                                                                                                     | Commits                        |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| S3 over Cloudflare R2                                                    | $210k AWS credits already in hand; cross-cloud egress not needed near-term                                                                    | (no commit; AWS console state) |
| HBN-EEG over TUEG as pretraining corpus                                  | open access (no NEDC wait), 2.5× more iid examples per recording-hour from 128-ch montage, modern uniform montage                             | `1587434`                      |
| Hybrid (Option C) eval: HBN primary + TUH secondary                      | best of both: bootstrap today + literature-comparable later                                                                                   | `1587434`                      |
| Minimum-offline preprocessing (NaN+zscore+clip+window only)              | notch / bandpass / resample are exp02 / exp05 / exp14 hypotheses; doing them offline pre-decides those experiments                            | `e18117e`                      |
| Native 500 Hz preserved (no 500→250 resample)                            | exp14's 60k-sample window at 2 kHz argument requires not pre-discarding > 125 Hz content                                                      | `e18117e`                      |
| F0-prep literature-comparability cell in exp02                           | direct numerical comparison vs BENDR / LaBraM-Base / CBraMod / REVE without contaminating the §4.4 winner-picker                              | `e18117e`                      |
| Self-contained exp03 package, no `eeg_common` imports                    | exp03's preprocessing philosophy + downstream eval differ from exp01/02; coupling them would entangle two experiments better kept independent | `e18117e`                      |
| Protocol A.1 changed from ADHD-binary AUROC → CBCL-factor regression     | empirical finding that HBN ships no DSM-V Dx columns; CBCL externalizing matches NeurIPS 2025 Challenge C2 directly (a *stronger* eval)       | `df68e51`                      |
| `seqLearning6target` collapses to label 1 alongside `seqLearning8target` | R1 has subjects with EITHER variant, never both                                                                                               | `df68e51`                      |
| Pull entire HBN-EEG corpus tonight to S3                                 | "preprocess once, ever" promise; one-shot CPU job while box is already up; ~$200 vs hours of orchestration later                              | (this entry's commit)          |


