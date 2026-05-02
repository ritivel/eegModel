#!/usr/bin/env bash
# scripts/full_hbn_pipeline.sh
# ------------------------------------------------------------------
# Full HBN-EEG pipeline: download + minimal-preprocess + sync-to-S3 + delete-raw,
# across all 10 BIDS releases (NC, R1..R9) totalling ~2,639 subjects, ~2,500
# recording-hours, ~2.2 TB raw / ~1 TB preprocessed parquet.
#
# Run from the GPU box (or any box with the exp03 venv installed):
#
#   SCRIPT=/home/ubuntu/eegModel/experiments/exp03_eeg_pretraining/scripts/full_hbn_pipeline.sh
#   LOG=/opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.log
#   PIDFILE=/opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.pid
#   nohup bash "$SCRIPT" > "$LOG" 2>&1 < /dev/null &
#   echo $! > "$PIDFILE"
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
# Per-release detailed logs:  /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.<release>.log
# Top-level summary log:      /opt/dlami/nvme/eeg/scratch/full_hbn_pipeline.log
#
# Total estimated wall-clock: 3-5 hours on a p4de.24xlarge.
#
# Resumability: each `exp03 download` skips already-present .set, each
# `exp03 preprocess` skips already-present parquet, `sync-derived-up`
# (rclone copy) skips matching files in S3. Step 4 (delete raw) is the
# only destructive step; if we re-run after a partial completion, deleted
# raw will be re-downloaded — cheap to recover.

# NOT -e: per-release failures should be logged but not abort the whole run.
set -uo pipefail

# -------------------------------------------------------------------
# setup — these get exported to per-release subshells via `export`
# -------------------------------------------------------------------
export SCRATCH=/opt/dlami/nvme/eeg/scratch
mkdir -p "$SCRATCH"

# Activate the exp03 venv so `exp03` is on PATH inside subshells
source /home/ubuntu/eegModel/.venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ts()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }   # stdout only — nohup redirects stdout to the top log
export -f ts log

# -------------------------------------------------------------------
# per-release function — fully self-contained, called via xargs
# -------------------------------------------------------------------
process_release() {
  local rel="$1"
  local rel_log="$SCRATCH/full_hbn_pipeline.${rel}.log"
  local raw_dir="/opt/dlami/nvme/eeg/raw/hbn/cmi_bids_${rel}"

  # Each per-release status message goes to BOTH the per-release log
  # and stdout (which the parent script redirects to the top log).
  echo "[$(ts)] [${rel}] === START ===" | tee -a "$rel_log"

  echo "[$(ts)] [${rel}] step 1: download" | tee -a "$rel_log"
  if ! exp03 download "$rel" --max-subjects 1000 >>"$rel_log" 2>&1; then
    echo "[$(ts)] [${rel}] DOWNLOAD FAILED — see $rel_log"
    return 1
  fi

  echo "[$(ts)] [${rel}] step 2: preprocess (minimal)" | tee -a "$rel_log"
  if ! exp03 preprocess "$rel" --pipeline minimal >>"$rel_log" 2>&1; then
    echo "[$(ts)] [${rel}] PREPROCESS FAILED — see $rel_log"
    return 1
  fi

  echo "[$(ts)] [${rel}] step 3: sync derived/ → S3" | tee -a "$rel_log"
  if ! exp03 sync-derived-up --pipeline minimal >>"$rel_log" 2>&1; then
    echo "[$(ts)] [${rel}] SYNC FAILED — see $rel_log"
    return 1
  fi

  echo "[$(ts)] [${rel}] step 4: delete raw to free NVMe" | tee -a "$rel_log"
  rm -rf "$raw_dir"

  local n_subj_in_s3
  n_subj_in_s3="$(aws s3 ls 's3://eegmodel-warehouse/derived/hbn_minimal_500hz/' 2>/dev/null | grep -c 'PRE sub-' || echo 0)"
  echo "[$(ts)] [${rel}] === DONE === (S3 now has $n_subj_in_s3 subject directories)" \
    | tee -a "$rel_log"
}
export -f process_release

# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------
log "==============================================================="
log "full HBN pipeline starting on $(hostname) (PID $$)"
log "releases: NC R1 R2 R3 R4 R5 R6 R7 R8 R9  (10 total)"
log "parallelism: 4 releases concurrently (xargs -P 4)"
log "expected wall-clock: 3-5 hours"
log "top log:  $SCRATCH/full_hbn_pipeline.log"
log "per-release logs: $SCRATCH/full_hbn_pipeline.<release>.log"
log "S3 destination: s3://eegmodel-warehouse/derived/hbn_minimal_500hz/"
log "==============================================================="

# `printf` instead of `echo | tr` so each release is on its own line cleanly.
# `xargs -I REL` substitutes REL with each input line; -P 4 runs 4 concurrently.
# (No `-n 1` — `-I` already implies one-arg-per-invocation.)
printf 'NC\nR1\nR2\nR3\nR4\nR5\nR6\nR7\nR8\nR9\n' \
  | xargs -P 4 -I REL bash -c 'process_release "$@"' _ REL

# -------------------------------------------------------------------
# final verification
# -------------------------------------------------------------------
log ""
log "==============================================================="
log "FINAL VERIFICATION"
log "==============================================================="

log "S3 inventory under derived/hbn_minimal_500hz/:"
aws s3 ls --recursive --summarize s3://eegmodel-warehouse/derived/hbn_minimal_500hz/ \
  | tail -3

log ""
log "remaining NVMe usage:"
du -sh /opt/dlami/nvme/eeg/* 2>/dev/null

log ""
log "==============================================================="
log "ALL RELEASES PROCESSED. The instance can be safely stopped now."
log "==============================================================="
