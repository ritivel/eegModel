# AWS Mumbai capacity-block week (2026-W19)

Runbook for the **8× H100 80GB** AWS Capacity Block reserved in `ap-south-1` from
**2026-05-05 17:00 IST → 2026-05-12 17:00 IST**.

The full lifecycle is driven by the [`eeg-ops`](../../packages/eeg_ops/) Python
CLI; this directory keeps the SkyPilot task YAML and this runbook. There are no
shell scripts anymore — everything lives in `packages/eeg_ops` with proper
typer commands, type hints, and tests.

## Architecture

| Concern | Tool | Notes |
|---|---|---|
| Capacity block lifecycle | `eeg-ops capacity find/buy/status` | Boto3-direct, with `state.toml` memo. Hard-fails if the upfront fee drifts. |
| Cluster launch / SSH / terminate | `eeg-ops cluster up/status/exec/down` | Defaults to `--via boto3` (deterministic, ~3 min). `--via skypilot` available for non-restricted networks. |
| Bootstrap on the GPU box | cloud-init user-data emitted by `eeg_ops.launcher` | Mirrors the SkyPilot YAML's `setup` block: uv venv → editable installs → `s3torchconnector[dcp]` → rclone of derived shards from the in-region mirror. |
| In-region S3 mirror | `eeg-ops data prewarm` | Idempotent server-side `aws s3 sync` from `eegmodel-warehouse` (us-west-2) to `eeg-mumbai-139156132535` (ap-south-1). |
| Box ↔ S3 auth | IAM instance profile via `eeg-ops iam create` | Box assumes role; no `~/.aws/credentials` copy. |
| Spend backstop | CloudWatch alarm via `eeg-ops alarm create --daily-budget-usd 800` | AWS billing metrics in us-east-1. |
| Checkpoint durability | `s3torchconnector` (single-file *or* DCP) via `eeg_ops.checkpoint` | `--use-dcp` for multi-rank FSDP/DDP runs. Resume from S3 is automatic when `--s3-ckpt-resume`. |
| Operations Hub log | Notion via `eeg_ops.notion` | Every CLI action and every training event lands in the Events database under the [Operations Hub](https://www.notion.so/357939fbcda4813fab4bc5fe0d84ea2c). |

## Lifecycle, end-to-end

```bash
# Local Mac, once per rental
eeg-ops capacity find  --region ap-south-1 --duration-hours 168
eeg-ops capacity buy   cb-xxx --region ap-south-1 --expected-fee 5285.95
eeg-ops iam create     --region ap-south-1
eeg-ops alarm create   --region ap-south-1 --daily-budget-usd 800
eeg-ops data prewarm   --region ap-south-1                     # ~30 min for 800 GB

# Local Mac, at reservation start time
export WANDB_API_KEY=...
export HF_TOKEN=...
export NOTION_API_KEY=...      # optional but recommended
eeg-ops cluster up

# Run training; uses the existing exp03 CLI
eeg-ops cluster exec 'exp03 train --paradigm mar --steps 17500 \
    --wandb-run-name exp17-g2-seed0 \
    --s3-ckpt-bucket eegmodel-warehouse \
    --s3-ckpt-prefix runs/exp03/exp17-g2-seed0 \
    --use-dcp \
    --notion-experiment-id 357939fb-cda4-81e8-9639-f3635c218199'

# End of week
eeg-ops cluster down --sync-runs
```

## What `eeg.sky.yaml` is for

[`eeg.sky.yaml`](./eeg.sky.yaml) is the SkyPilot task spec used when `eeg-ops
cluster up --via skypilot` runs. The default `--via boto3` path doesn't need
it; the YAML stays for two reasons:

1. **Multi-cloud failover later.** The same YAML can target `lambda`, `gcp`,
   or `aws/<other-region>` by changing `infra:`. SkyPilot's `--retry-until-up`
   waits for the reservation to flip from `payment-pending` to `active`.
2. **Self-documenting bootstrap.** The `setup:` block is the canonical
   description of what a GPU box needs to land in our work environment;
   `eeg_ops.launcher` mirrors it as cloud-init user-data.

## Storage diagram (as deployed)

```
local Mac
  ~/.config/eeg-ops/state.toml              ← reservation IDs, cluster name, Notion sessions
  ~/.ssh/eeg-mumbai-2026w19.pem
  ~/.sky/config.yaml                        ← aws.specific_reservations + remote_identity

       ↓ ec2:RunInstances + IAM instance profile

GPU box (p5.48xlarge in ap-south-1a)
  /home/ubuntu/eegModel/                    ← repo + venv (editable installs)
  /opt/dlami/nvme/eeg/                      ← 30 TB local NVMe
    derived/<pipeline>/                     ← rcloned from Mumbai mirror
    runs/<exp>/<id>/                        ← live training outputs
                                              (mirrored to S3 every ckpt_every steps)

       ↓ s3torchconnector during training (single-file or DCP per --use-dcp)
       ↓ rclone copy at end of week (eeg-ops cluster down --sync-runs)

S3 (us-west-2 — durable warehouse)
  s3://eegmodel-warehouse/
    derived/<pipeline>/                     ← canonical preprocessed shards
    runs/exp03/<run_id>/                    ← every checkpoint + accelerate state
    models/hf_cache/

S3 (ap-south-1 — this-week regional cache)
  s3://eeg-mumbai-139156132535/
    derived/<pipeline>/                     ← refreshed by `eeg-ops data prewarm`
```
