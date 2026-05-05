# eeg-ops

A small, opinionated CLI for the cluster-lifecycle around the eegModel
experiments. It wraps three things:

1. **AWS EC2 Capacity Block reservation** — discover, purchase, monitor.
2. **SkyPilot** — launch the actual cluster against a reservation and stream
   in-region S3 mirrors of the preprocessed dataset.
3. **PyTorch checkpointing to S3** — direct-to-S3 distributed checkpoints via
   [`s3torchconnector`](https://github.com/awslabs/s3-connector-for-pytorch),
   so that a force-stop at the reservation's `EndDate` is recoverable.

It deliberately *delegates* to SkyPilot for the hard parts (auto-failover,
multi-cloud, retry-until-up). It exists to give us a single curated entrypoint
that knows about our specific region/bucket/reservation conventions, plus a
small set of AWS chores that SkyPilot itself doesn't handle (capacity block
purchase, IAM instance profile creation, CloudWatch alarms, cross-region
warehouse mirroring).

## Commands

```text
$ eeg-ops --help

Usage: eeg-ops [OPTIONS] COMMAND [ARGS]...

Commands:
  capacity   AWS Capacity Block lifecycle (find, buy, list, status).
  cluster    Cluster lifecycle: up / status / down (wraps SkyPilot).
  data       Pre-warm a regional S3 mirror of the preprocessed shards.
  iam        Create/inspect the IAM instance profile the box uses for S3.
  alarm      CloudWatch alarms for spend / reservation expiry.
  checkpoint S3 checkpoint utilities (sync runs/, validate, restore).
  config     Show resolved config (region, reservation, buckets, key paths).
```

Each subcommand has its own `--help` with the full flag set.

## Typical week

```bash
# 1) Find an offering, buy it, save IDs to ~/.config/eeg-ops/state.toml
eeg-ops capacity find --gpus H100:8 --duration-hours 168 --region ap-south-1
eeg-ops capacity buy  cb-06acabbeb025ff1d3 --yes

# 2) One-time IAM + alarms (idempotent on re-run)
eeg-ops iam create
eeg-ops alarm create --daily-budget-usd 800

# 3) Pre-warm Mumbai mirror from us-west-2 warehouse
eeg-ops data prewarm --region ap-south-1

# 4) Launch the cluster (SkyPilot under the hood, waits for reservation active)
eeg-ops cluster up --retry-until-up
eeg-ops cluster status

# 5) Train (uses your existing `exp03 train` CLI, run via `sky exec`)
eeg-ops cluster exec 'exp03 train --paradigm mar --steps 17500 ...'

# 6) End of week
eeg-ops cluster down --sync-runs   # pushes runs/ to warehouse, then sky down
```

## State and config

State (offering IDs, reservation IDs, public IPs) is kept in
`~/.config/eeg-ops/state.toml` so commands compose without re-passing flags.
Buckets and AMI IDs default to per-region values that are encoded in the
package; override with env vars or `--region <other>` if you ever expand
beyond `ap-south-1`.
