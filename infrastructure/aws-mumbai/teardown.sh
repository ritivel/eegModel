#!/usr/bin/env bash
# teardown.sh — run at T+167h, before the reservation force-stops.
#
# Pushes runs/ (this week's training outputs) from the GPU box's NVMe back
# to the warehouse, terminates the instance, and offers to empty the
# Mumbai bucket so you stop paying ~$5/wk for transient storage.
#
# Idempotent: re-running is safe; instance terminate is no-op once gone.
#
# Usage:
#   source .env
#   bash teardown.sh                  # interactive (recommended)
#   bash teardown.sh --yes-all        # unattended (cron-friendly)

set -uo pipefail
cd "$(dirname "$0")"

YES=0; [[ "${1:-}" == "--yes-all" ]] && YES=1

confirm() {
  [[ $YES -eq 1 ]] && return 0
  read -rp "$1 [y/N] " r; [[ "$r" =~ ^[Yy]$ ]]
}

INSTANCE_ID="$(cat ~/.ssh/eeg-mumbai-instance-id 2>/dev/null || true)"
PUB="$(cat ~/.ssh/eeg-mumbai-host 2>/dev/null || true)"

echo "==========================================="
echo "  Teardown — eeg-mumbai-2026w19"
echo "  Instance: ${INSTANCE_ID:-<unknown>}  IP: ${PUB:-<unknown>}"
echo "==========================================="

# ---- step 1: push runs/ to the warehouse via the GPU box -----------
# We do this from the GPU side so the upload is from-Mumbai-to-Oregon
# at S3-server-side speeds (still cross-region but ~10-50× faster than
# downloading-then-uploading from your Mac).
if [[ -n "$PUB" ]] && confirm "[1/4] sync runs/ from GPU box to s3://$S3_WAREHOUSE_BUCKET/runs/exp03/?"; then
  ssh -i "$EC2_KEY_PATH" -o StrictHostKeyChecking=accept-new ubuntu@"$PUB" "
    set -uo pipefail
    source ~/eegModel/.venv/bin/activate 2>/dev/null || true
    DATA_ROOT='${REMOTE_DATA_ROOT}'
    if [[ -d \$DATA_ROOT/runs/exp03 ]]; then
      echo '  pushing \$DATA_ROOT/runs/exp03/ to warehouse...'
      rclone copy \$DATA_ROOT/runs/exp03/ s3w:runs/exp03/ \
        --transfers 32 --checkers 32 --progress
    fi
    if [[ -d \$DATA_ROOT/runs/01_sanity_baselines ]] || \
       [[ -d \$DATA_ROOT/runs/exp17 ]]; then
      echo '  pushing top-level runs/ subdirs...'
      rclone copy \$DATA_ROOT/runs/ s3w:runs/exp03/ \
        --transfers 32 --checkers 32 --progress
    fi
  "
fi

# ---- step 2: pull bootstrap.log + nvidia-smi history ---------------
if [[ -n "$PUB" ]] && confirm "[2/4] pull bootstrap.log + dmesg + last nvidia-smi to local archive?"; then
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  ARCHIVE="$HOME/eeg-mumbai-2026w19-$STAMP"
  mkdir -p "$ARCHIVE"
  scp -i "$EC2_KEY_PATH" \
    ubuntu@"$PUB":/var/log/eeg-bootstrap.log "$ARCHIVE/" 2>/dev/null || true
  ssh -i "$EC2_KEY_PATH" ubuntu@"$PUB" \
    "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv > /tmp/nvsmi.csv && cat /tmp/nvsmi.csv" \
    > "$ARCHIVE/nvidia-smi.csv" 2>/dev/null || true
  ssh -i "$EC2_KEY_PATH" ubuntu@"$PUB" \
    "dmesg | tail -200" > "$ARCHIVE/dmesg.tail" 2>/dev/null || true
  echo "  archived to $ARCHIVE"
fi

# ---- step 3: terminate instance ------------------------------------
if [[ -n "$INSTANCE_ID" ]] && confirm "[3/4] terminate $INSTANCE_ID? (NVMe is ephemeral — ensure step 1 completed!)"; then
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" \
    --output json | python3 -m json.tool
  rm -f ~/.ssh/eeg-mumbai-host ~/.ssh/eeg-mumbai-instance-id
fi

# ---- step 4: optional bucket cleanup -------------------------------
if confirm "[4/4] empty the Mumbai cache bucket s3://$S3_MUMBAI_BUCKET/? (saves ~\$5/wk; you can re-warm next week)"; then
  aws s3 rm "s3://$S3_MUMBAI_BUCKET" --recursive --region "$S3_MUMBAI_REGION"
  if confirm "    ... and delete the bucket itself?"; then
    aws s3api delete-bucket --bucket "$S3_MUMBAI_BUCKET" --region "$S3_MUMBAI_REGION"
  fi
fi

echo
echo "==========================================="
echo "  Teardown complete."
echo "  Reservation auto-expires at $CB_END_DATE"
echo "==========================================="
