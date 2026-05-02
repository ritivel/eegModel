# exp03 — Progress log

> Same chronological-session-log convention as `experiments/exp02_eeg_ctc/progress.md`.
> Append at the top; oldest entries at the bottom.
>
> **Last refreshed:** 2026-05-02 ~17:00 IST / 11:30 UTC

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
+ §"Background job" below.

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

**Cost.** ~$32.77 / hour while the box runs. Estimated job duration
3-5 hours → estimated job-portion cost **~$130–$200**. The box
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

| Decision | Rationale | Commits |
|---|---|---|
| S3 over Cloudflare R2 | $210k AWS credits already in hand; cross-cloud egress not needed near-term | (no commit; AWS console state) |
| HBN-EEG over TUEG as pretraining corpus | open access (no NEDC wait), 2.5× more iid examples per recording-hour from 128-ch montage, modern uniform montage | `1587434` |
| Hybrid (Option C) eval: HBN primary + TUH secondary | best of both: bootstrap today + literature-comparable later | `1587434` |
| Minimum-offline preprocessing (NaN+zscore+clip+window only) | notch / bandpass / resample are exp02 / exp05 / exp14 hypotheses; doing them offline pre-decides those experiments | `e18117e` |
| Native 500 Hz preserved (no 500→250 resample) | exp14's 60k-sample window at 2 kHz argument requires not pre-discarding > 125 Hz content | `e18117e` |
| F0-prep literature-comparability cell in exp02 | direct numerical comparison vs BENDR / LaBraM-Base / CBraMod / REVE without contaminating the §4.4 winner-picker | `e18117e` |
| Self-contained exp03 package, no `eeg_common` imports | exp03's preprocessing philosophy + downstream eval differ from exp01/02; coupling them would entangle two experiments better kept independent | `e18117e` |
| Protocol A.1 changed from ADHD-binary AUROC → CBCL-factor regression | empirical finding that HBN ships no DSM-V Dx columns; CBCL externalizing matches NeurIPS 2025 Challenge C2 directly (a *stronger* eval) | `df68e51` |
| `seqLearning6target` collapses to label 1 alongside `seqLearning8target` | R1 has subjects with EITHER variant, never both | `df68e51` |
| Pull entire HBN-EEG corpus tonight to S3 | "preprocess once, ever" promise; one-shot CPU job while box is already up; ~$200 vs hours of orchestration later | (this entry's commit) |
