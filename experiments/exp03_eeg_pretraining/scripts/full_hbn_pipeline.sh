#!/usr/bin/env bash
# scripts/full_hbn_pipeline.sh
# ------------------------------------------------------------------
# Full HBN-EEG pipeline: download + minimal-preprocess + sync-to-S3 + delete-raw,
# across all 10 BIDS releases (NC, R1..R9) totalling ~2,639 subjects, ~2,500
# recording-hours, ~2.2 TB raw / ~1 TB preprocessed parquet.
#
# Run from the GPU box (or any box with the exp03 venv installed):
#
#   nohup bash experiments/exp03_eeg_pretraining/scripts/full_hbn_pipeline.sh \
#     > /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.log 2>&1 &
#   echo $! > /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.pid
#
# Then disconnect SSH; the job survives the disconnect via nohup.
#
# Per-release pipeline (parallelised 4-at-a-time via xargs):
#   1. download raw .set + BIDS sidecars + participants.tsv to NVMe scratch
#   2. apply the SPEC_MINIMAL pipeline (NaN sanit. + per-channel z-score +
#      ±5σ clip + 4-s windowing + iid-channel expansion + float16 parquet)
#   3. sync derived/hbn_minimal_500hz/ → S3 warehouse (idempotent, rclone copy)
#   4. delete the per-release raw tree (`cmi_bids_<rel>/`) to free NVMe
#      (parquet shards stay on NVMe AND in S3 for the next step's training run)
#
# Per-release, per-step logs land at
#   /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.<release>.log
# Top-level log at
#   /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.log
#
# Resumability: rerunning the script is *almost* idempotent — `exp03 download`
# skips already-present .set files, `exp03 preprocess` skips already-present
# parquet shards, `exp03 sync-derived-up` (rclone copy) skips matching files in
# S3. The only step that is destructive on rerun is step 4 (delete raw); if the
# script aborts mid-preprocess and we rerun, step 4 of an earlier release may
# have already deleted that release's raw, in which case `exp03 download` will
# re-pull. Cheap to recover from.
#
# Total estimated wall-clock: 3-5 hours on a p4de.24xlarge (96 vCPUs, 100 Gbps
# network). Per-release: ~30 min download + ~30 min preprocess + ~5 min sync.
#
# To check progress: tail -f /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.log
# To check S3 progress: aws s3 ls --recursive --summarize s3://eegmodel-warehouse/derived/hbn_minimal_500hz/ | tail -3
# To kill: kill $(cat /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.pid) ; pkill -f exp03

# NOT -e: we want per-release failures to be logged but not abort the whole run.
set -uo pipefail

# -------------------------------------------------------------------
# setup
# -------------------------------------------------------------------
SCRATCH=/opt/dlami/nvme/eeg/scratch
mkdir -p "$SCRATCH"
TOP_LOG="$SCRATCH/full_hbn_pipeline.log"

# Activate venv so `exp03` is on PATH
source /home/ubuntu/eegModel/.venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ts()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$TOP_LOG"; }

# -------------------------------------------------------------------
# per-release function (one HBN release: NC, R1, ..., R9)
# -------------------------------------------------------------------
process_release() {
  local rel="$1"
  local rel_log="$SCRATCH/full_hbn_pipeline.${rel}.log"
  local raw_dir="/opt/dlami/nvme/eeg/raw/hbn/cmi_bids_${rel}"

  echo "[$(ts)] [${rel}] === START ===" | tee -a "$rel_log" "$TOP_LOG"

  echo "[$(ts)] [${rel}] step 1: download" | tee -a "$rel_log" "$TOP_LOG"
  if ! exp03 download "$rel" --max-subjects 1000 >>"$rel_log" 2>&1; then
    echo "[$(ts)] [${rel}] DOWNLOAD FAILED — see $rel_log" | tee -a "$TOP_LOG"
    return 1
  fi

  echo "[$(ts)] [${rel}] step 2: preprocess (minimal)" | tee -a "$rel_log" "$TOP_LOG"
  if ! exp03 preprocess "$rel" --pipeline minimal >>"$rel_log" 2>&1; then
    echo "[$(ts)] [${rel}] PREPROCESS FAILED — see $rel_log" | tee -a "$TOP_LOG"
    return 1
  fi

  echo "[$(ts)] [${rel}] step 3: sync derived/ → S3" | tee -a "$rel_log" "$TOP_LOG"
  if ! exp03 sync-derived-up --pipeline minimal >>"$rel_log" 2>&1; then
    echo "[$(ts)] [${rel}] SYNC FAILED — see $rel_log" | tee -a "$TOP_LOG"
    return 1
  fi

  echo "[$(ts)] [${rel}] step 4: delete raw tree to free NVMe" | tee -a "$rel_log" "$TOP_LOG"
  rm -rf "$raw_dir"

  # Snapshot space + a quick S3 row-count for this release
  local s3_count
  s3_count="$(aws s3 ls --recursive "s3://eegmodel-warehouse/derived/hbn_minimal_500hz/" 2>/dev/null | wc -l)"
  echo "[$(ts)] [${rel}] === DONE === (S3 now has ${s3_count} parquet shards total)" \
    | tee -a "$rel_log" "$TOP_LOG"
}
export -f process_release ts

# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
log "==============================================================="
log "full HBN pipeline starting on $(hostname) (PID $$)"
log "releases: NC R1 R2 R3 R4 R5 R6 R7 R8 R9  (10 total)"
log "parallelism: 4 releases concurrently (xargs -P 4)"
log "expected wall-clock: 3-5 hours"
log "top log:  $TOP_LOG"
log "per-release logs: $SCRATCH/full_hbn_pipeline.<release>.log"
log "S3 destination: s3://eegmodel-warehouse/derived/hbn_minimal_500hz/"
log "==============================================================="

echo "NC R1 R2 R3 R4 R5 R6 R7 R8 R9" | tr ' ' '\n' \
  | xargs -n 1 -P 4 -I REL bash -c 'process_release "$@"' _ REL

# -------------------------------------------------------------------
# final verification
# -------------------------------------------------------------------
log ""
log "==============================================================="
log "FINAL VERIFICATION"
log "==============================================================="

log "S3 inventory under derived/hbn_minimal_500hz/:"
aws s3 ls --recursive --summarize s3://eegmodel-warehouse/derived/hbn_minimal_500hz/ \
  | tail -3 | tee -a "$TOP_LOG"

log ""
log "remaining NVMe usage:"
du -sh /opt/dlami/nvme/eeg/* 2>/dev/null | tee -a "$TOP_LOG"

log ""
log "==============================================================="
log "ALL RELEASES PROCESSED. The instance can be safely stopped now."
log "==============================================================="
