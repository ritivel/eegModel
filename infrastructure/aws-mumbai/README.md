# AWS Mumbai capacity-block week (2026-W19)

A self-contained runbook for the **8× H100 80GB capacity block** reserved in
`ap-south-1` (Mumbai) from **2026-05-05 17:00 IST → 2026-05-12 17:00 IST**.

## Why this exists

`scripts/track_a_run_on_gpu_box.sh` was written for one-off Lambda / `g5.8xlarge`
boxes. The capacity-block week is different in three ways that matter:

1. **Region is `ap-south-1`, warehouse is `us-west-2`.** Pulling 800 GB of
   derived parquet across regions during the GPU's billable hour costs ~$60
   in idle H100 time. We pre-warm a Mumbai mirror instead.
2. **The GPU is a *reservation*, not on-demand.** You launch *into* a
   `CapacityReservationId`; if you forget that flag the instance fails with
   "InsufficientInstanceCapacity" even though you've paid.
3. **The reservation has a hard end-time.** At `2026-05-12T11:30:00Z` AWS
   force-stops the instance. Anything still on local NVMe is gone.
   `teardown.sh` runs ~30 min before that.

## The lifecycle

```
T-9h ── purchase-capacity-block.sh   ── irreversible $5,285.95 charge
T-1h ── prewarm-data.sh               ── 800 GB warehouse → Mumbai mirror (run earlier; ~30-60 min)
T+0  ── launch-instance.sh            ── boot p5.48xlarge into the reservation
T+5m ── ssh + bootstrap.sh            ── repo clone + uv install + sync-derived-down (in-region!)
T+15m── exp03 train ...               ── you're back to your normal workflow
…
T+167h ── teardown.sh                 ── sync runs/ to warehouse, save logs, terminate
T+168h ── reservation force-stops     ── if teardown didn't run, NVMe is gone
```

## Files in this directory

| File | When to run | Cost impact |
|---|---|---|
| `.env`              | source on local Mac (created from `.env.example`)   | — |
| `purchase-capacity-block.sh` | once, ~9 h before T+0           | **$5,285.95 non-refundable** |
| `prewarm-data.sh`   | once, hours before T+0                              | ~$16 transfer + ~$5/wk Mumbai S3 |
| `launch-instance.sh`| at T+0 (or any time in the 168 h window)            | already paid |
| `bootstrap.sh`      | runs on the GPU box once at first SSH               | — |
| `monitor.sh`        | local — shows instance state, h remaining, $/h burn | — |
| `teardown.sh`       | at T+167h (set a calendar reminder)                 | — |

## What's where (storage map for the week)

```
local Mac ──────────────────────────────────────────────────────────┐
  ~/.ssh/eeg-mumbai-2026w19.pem    SSH key, ed25519                  │
  /Users/.../eegModel              your working repo, git main       │
  /tmp/eeg-mumbai-sync.log         pre-warm progress                 │
                                                                     │
                                          ┌──────────────────────────┘
                                          │ ssh -A
                                          ▼
GPU box (p5.48xlarge, ap-south-1a) ────────────────────────────────┐
  /opt/dlami/nvme/eeg/                                              │
    raw/hbn/                       not here — pull from FCP-INDI    │
    raw/{tuab,tuev}/               not here — rsync from NEDC if    │
                                   needed (already warehoused-      │
                                   derived covers Protocol A.4)     │
    derived/                       <─ pulled from Mumbai mirror     │
      hbn_minimal_500hz/                                             │
      hbn_v2_clean_250hz/                                            │
      tuab_v2_clean_250hz/                                           │
      tuev_v2_clean_250hz/                                           │
    runs/<exp>/<id>/               this week's outputs              │
    models/hf_cache/               HF model downloads                │
                                                                     │
                                          ┌──────────────────────────┘
                                          │  rclone, in-region, fast
                                          ▼
S3 mirror (ap-south-1) ────────────────────────────────────────────┐
  s3://eeg-mumbai-139156132535/                                     │
    derived/                       refreshed by prewarm-data.sh     │
    runs/exp03/                    push here from teardown.sh       │
                                                                     │
                                          ┌──────────────────────────┘
                                          │  one-shot replication at end
                                          ▼
S3 warehouse (us-west-2, persistent) ──────────────────────────────┐
  s3://eegmodel-warehouse/                                          │
    derived/                       canonical preprocessed shards    │
    runs/exp03/                    canonical run history            │
    models/hf_cache/               cached HF weights                │
─────────────────────────────────────────────────────────────────── ┘
```

The Mumbai bucket is **a temporary cache for this week**. After
`teardown.sh` syncs runs/ back to `eegmodel-warehouse`, the Mumbai bucket
can be emptied and deleted to stop the ~$5/week storage charge. We keep it
through the week in case a debug session needs to relaunch.

## "Why didn't you use FSx Lustre / SkyPilot / a baked AMI?"

Considered, deferred:

- **FSx Lustre** — adds an AZ-pinned filesystem, $40+ for the week, requires
  VPC plumbing. The `p5.48xlarge` already ships with **30 TB of local NVMe**
  (`/opt/dlami/nvme/`), which is bigger than your entire derived corpus and
  faster than Lustre Scratch_2 for sequential parquet reads. Use NVMe.
- **SkyPilot** — its strength is multi-cloud auto-failover. You bought a
  single-region reservation; falling over to GCP is moot. Revisit if you
  start running multi-week jobs across providers.
- **Baked AMI / ECR Docker image** — `bootstrap.sh` is currently <10 min on
  the DLAMI. Bake one if it grows past 20 min or you start launching daily.

## Going from this week to the next

When you grab the next reservation, the only files that change are
`.env` (new offering ID, new dates, new SG/keypair name, new bucket) and
`launch-instance.sh`'s subnet ID if the AZ shifts. Everything else is reusable.
