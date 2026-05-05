# AWS Mumbai capacity-block week (2026-W19)

Runbook for the **8× H100 80GB** AWS Capacity Block reserved in `ap-south-1` from
**2026-05-05 17:00 IST → 2026-05-12 17:00 IST**.

> **The shell scripts in this directory are deprecated.** They worked, but
> we rebuilt the lifecycle as a proper Python package: see
> [`packages/eeg_ops`](../../packages/eeg_ops/). The scripts remain for
> reference and as a fallback if SkyPilot ever has an outage.

## What we use now

| Concern | Tool | Why |
|---|---|---|
| Buy / monitor capacity blocks | `eeg-ops capacity find/buy/status` | Boto3-direct, with state.toml memo and PURCHASE-typed gate. |
| Launch / SSH / re-attach the cluster | **SkyPilot** ([`eeg.sky.yaml`](./eeg.sky.yaml), invoked via `eeg-ops cluster up`) | Native AWS Capacity Block support since v0.7. `--retry-until-up` waits for the reservation to flip from `payment-pending` → `active` automatically; no need to babysit. Falls back across regions if the reservation expires. |
| Pre-warm in-region S3 mirror | `eeg-ops data prewarm --region ap-south-1` | Idempotent server-side `aws s3 sync` from `eegmodel-warehouse` (us-west-2) to `eeg-mumbai-139156132535` (ap-south-1) so the GPU box pulls in-region. |
| Box → S3 credentials | IAM instance profile, created via `eeg-ops iam create --region ap-south-1` | Box has S3 RW + CloudWatch via instance role; no `~/.aws/credentials` copy. |
| Cost backstop | CloudWatch billing alarm via `eeg-ops alarm create --daily-budget-usd 800` | Alerts if estimated charges blow past the budget. |
| Checkpoint durability across NVMe wipe | [`s3torchconnector`](https://github.com/awslabs/s3-connector-for-pytorch) via `eeg_ops.checkpoint` + `--s3-ckpt-bucket` flag on `exp03 train` | Each ckpt written direct-to-S3 with the AWS CRT (~40 % faster than EBS+sync); on resume, training transparently restores Accelerate state from S3. EndDate force-stop is now recoverable. |
| Run-history sync at end of week | `eeg-ops cluster down --sync-runs` (calls `eeg-ops checkpoint sync-runs` on the box) | NVMe → warehouse before terminate. |

## Lifecycle, end-to-end

```bash
# Local Mac, once per rental
eeg-ops capacity find  --region ap-south-1 --gpus H100:8 --duration-hours 168
eeg-ops capacity buy   cb-xxx --region ap-south-1 --expected-fee 5285.95
eeg-ops iam create     --region ap-south-1
eeg-ops alarm create   --region ap-south-1 --daily-budget-usd 800
eeg-ops data prewarm   --region ap-south-1                # ~30 min for 800 GB

# Local Mac, at reservation start time (or any time during the window)
eeg-ops cluster up    --yaml infrastructure/aws-mumbai/eeg.sky.yaml \
                      --name eeg-mumbai-2026w19

# After cluster is up, run training jobs
eeg-ops cluster exec 'exp03 train --paradigm mar --steps 17500 \
                       --wandb-run-name exp17-g2-seed0 \
                       --s3-ckpt-bucket eegmodel-warehouse \
                       --s3-ckpt-prefix runs/exp03/exp17-g2-seed0'

# Set a calendar reminder for T+167h
eeg-ops cluster down  --name eeg-mumbai-2026w19 --sync-runs
```

## What the SkyPilot task YAML does

[`eeg.sky.yaml`](./eeg.sky.yaml) describes the full cluster:

- **`resources`** — `p5.48xlarge` in `ap-south-1`, the right DLAMI, the IAM
  instance profile.
- **`workdir: ../..`** — your local repo gets rsync'd to
  `/home/ubuntu/sky_workdir` on the box. Edit on Mac, `sky exec` to apply.
- **`file_mounts`** — `/mnt/s3-mumbai` is the regional cache bucket exposed
  via Mountpoint for S3 (POSIX read-through cache; alternative to copying
  parquet to NVMe).
- **`envs`** — `WANDB_API_KEY` and `HF_TOKEN` are forwarded from your local
  shell at launch time.
- **`setup`** — runs once per cluster lifetime. Installs uv, the venv,
  editable installs of all four packages (`eeg_common`, `eeg_ops`,
  `exp01-03`), `mamba-ssm`, `s3torchconnector[dcp]`, `wandb`, `accelerate`.
  Then `rclone copy`s the in-region S3 mirror to NVMe at full bandwidth.
- **`run`** — keeps the cluster up so you can `sky exec` into it for
  interactive runs, or push managed jobs.

## Storage diagram (final, as deployed)

```
local Mac
  ~/.config/eeg-ops/state.toml              ← reservation IDs, cluster name
  ~/.ssh/eeg-mumbai-2026w19.pem             ← (legacy fallback if you bypass sky)
  ~/.sky/config.yaml                        ← aws.specific_reservations: [cr-…]
  /Users/.../eegModel                       ← the repo

       ↓ sky launch (workdir rsync) + IAM instance profile

GPU box (p5.48xlarge in ap-south-1a)
  /home/ubuntu/sky_workdir/                 ← editable repo
  /home/ubuntu/sky_workdir/.venv/           ← editable installs
  /opt/dlami/nvme/eeg/                      ← 30 TB local NVMe
    derived/<pipeline>/                     ← rcloned from Mumbai mirror
    runs/<exp>/<id>/                        ← live training outputs
  /mnt/s3-mumbai/                           ← Mountpoint of S3 (alt POSIX path)

       ↓ s3torchconnector during training (every ckpt_every steps)
       ↓ rclone at end of week (cluster down --sync-runs)

S3 (us-west-2 — durable warehouse)
  s3://eegmodel-warehouse/
    derived/<pipeline>/                     ← canonical preprocessed shards
    runs/exp03/<run_id>/                    ← all checkpoints for this run
      ckpt_step{N}.pt                       ← single-file mirror via S3CheckpointSink
      accelerate/                           ← resume-state synced via accelerate hook
    models/hf_cache/

S3 (ap-south-1 — this-week regional cache)
  s3://eeg-mumbai-139156132535/
    derived/<pipeline>/                     ← refreshed by `eeg-ops data prewarm`
                                              (one shot per rental)
```

## Reproducing the week's outcomes after the cluster terminates

Everything that matters is in `s3://eegmodel-warehouse/runs/exp03/`. If a
future cluster has the same `s3-ckpt-bucket`/`s3-ckpt-prefix`,
`exp03 train` will resume mid-step from S3. To re-launch on a different
provider, change `infra:` in `eeg.sky.yaml` (e.g. `lambda` or `gcp`),
re-run `eeg-ops cluster up`, and SkyPilot brings you up there with the
same code and same data.

## Files in this directory

| File | Purpose | Status |
|---|---|---|
| `README.md` | this file | live |
| `eeg.sky.yaml` | SkyPilot task spec | live (canonical entrypoint) |
| `.env`, `.env.example` | per-region IDs (legacy; superseded by `state.toml`) | reference |
| `purchase-capacity-block.sh` | shell version of `eeg-ops capacity buy` | deprecated |
| `prewarm-data.sh` | shell version of `eeg-ops data prewarm` | deprecated |
| `launch-instance.sh` | shell version of `eeg-ops cluster up` | deprecated |
| `bootstrap.sh` | shell version of the SkyPilot YAML's `setup` block | deprecated |
| `monitor.sh` | shell version of `eeg-ops capacity status` | deprecated |
| `teardown.sh` | shell version of `eeg-ops cluster down --sync-runs` | deprecated |

The shell scripts will be deleted once the SkyPilot path has been exercised
end-to-end on a real run.
