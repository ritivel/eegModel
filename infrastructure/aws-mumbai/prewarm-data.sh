#!/usr/bin/env bash
# prewarm-data.sh — replicate s3://eegmodel-warehouse/derived/ from us-west-2
# into s3://eeg-mumbai-139156132535/derived/ (ap-south-1) so the GPU box
# pulls in-region instead of cross-region (saves ~2-4 h of H100 idle).
#
# Cross-region transfer cost is paid either way (~$16 for 800 GB). Doing
# it now means the GPU box's bootstrap takes 15 min instead of 2-4 h.
#
# This is idempotent — `aws s3 sync` skips objects already at destination
# with the same size + ETag. Safe to re-run.
#
# Usage:
#   source .env
#   bash prewarm-data.sh                              # all derived pipelines
#   bash prewarm-data.sh hbn_minimal_500hz            # one pipeline
#   bash prewarm-data.sh hbn_minimal_500hz tuab_v2_clean_250hz   # subset

set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${S3_WAREHOUSE_BUCKET:-}" ]] || [[ -z "${S3_MUMBAI_BUCKET:-}" ]]; then
  echo "ERROR: source .env first"; exit 1
fi

PIPELINES=("$@")
if [[ ${#PIPELINES[@]} -eq 0 ]]; then
  PIPELINES=(hbn_minimal_500hz hbn_v2_clean_250hz tuab_v2_clean_250hz tuev_v2_clean_250hz)
fi

LOG_DIR="/tmp"
LOG="$LOG_DIR/eeg-mumbai-sync.log"

echo "==========================================="
echo "  Pre-warm Mumbai S3 mirror"
echo "==========================================="
echo "  src:       s3://${S3_WAREHOUSE_BUCKET}/derived/   ($S3_WAREHOUSE_REGION)"
echo "  dst:       s3://${S3_MUMBAI_BUCKET}/derived/      ($S3_MUMBAI_REGION)"
echo "  pipelines: ${PIPELINES[*]}"
echo "  log:       $LOG"
echo "==========================================="

# Ensure dst bucket exists (idempotent — fails harmlessly if already there).
if ! aws s3api head-bucket --bucket "$S3_MUMBAI_BUCKET" --region "$S3_MUMBAI_REGION" 2>/dev/null; then
  echo "[setup] creating $S3_MUMBAI_BUCKET in $S3_MUMBAI_REGION..."
  aws s3api create-bucket --bucket "$S3_MUMBAI_BUCKET" --region "$S3_MUMBAI_REGION" \
    --create-bucket-configuration "LocationConstraint=$S3_MUMBAI_REGION"
  aws s3api put-bucket-versioning --bucket "$S3_MUMBAI_BUCKET" --versioning-configuration Status=Enabled
fi

T0=$(date +%s)
for pl in "${PIPELINES[@]}"; do
  echo "[sync] $pl ..."
  aws s3 sync "s3://${S3_WAREHOUSE_BUCKET}/derived/${pl}/" \
              "s3://${S3_MUMBAI_BUCKET}/derived/${pl}/" \
              --source-region "$S3_WAREHOUSE_REGION" \
              --region "$S3_MUMBAI_REGION" \
              --no-progress 2>&1 | tee -a "$LOG"
done

T1=$(date +%s)
echo
echo "[done] $((T1 - T0)) s elapsed"

echo "[verify] dst byte count ..."
aws s3 ls "s3://${S3_MUMBAI_BUCKET}/derived/" --recursive --summarize \
  --region "$S3_MUMBAI_REGION" | tail -3
