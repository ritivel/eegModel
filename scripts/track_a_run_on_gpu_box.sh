#!/usr/bin/env bash
# scripts/track_a_run_on_gpu_box.sh
# ------------------------------------------------------------------
# End-to-end runner for Track A of mini-experiment 01's last open
# follow-up: append the §4.3 Protocol A.4 (TUH literature-comparable)
# floor row to mini_experiments/01_sanity_baselines/results.md.
#
# Pipeline (single-pass, ~1.5 h on g5.8xlarge):
#   1. rsync TUAB v3.0.1 + TUEV v2.0.1 from NEDC SFTP → NVMe
#   2. apply SPEC_V2_CLEAN (60 Hz notch + 0.5–100 Hz bandpass + 250 Hz
#      polyphase resample + per-channel z-score + 4-s windowing + iid
#      expansion) → parquet shards under derived/{tuab,tuev}_v2_clean_250hz/
#   3. sync derived shards to s3://eegmodel-warehouse/
#   4. extract frozen features from a random-init §4.2-default model
#      (encoded by `EEGSSLModel.encode_features`) on both corpora
#   5. run Protocol A.4 floor probes:
#         - TUAB:  binary normal/abnormal AUROC + 95% CI
#         - TUEV:  6-class BAC + WF1 + k-NN top-1 + 95% CIs
#   6. write a JSON summary to runs/01_sanity_baselines/<ts>/protocol_a4_floor.json
#   7. (optional) sync the JSON to S3 for permanent record
#
# Usage:
#   # On the GPU box, with the eegfm venv active and SSH agent
#   # forwarding the NEDC key (ssh -A from your Mac):
#   bash experiments/scripts/track_a_run_on_gpu_box.sh
#
# Env-var knobs (sane defaults; override only if you know why):
#   EEG_DATA_ROOT   — default /opt/dlami/nvme/eeg
#   TRACK_A_TUH_MAX_SUBJECTS — cap subjects per TUH corpus during feature
#                              extraction.  Default 100 (good CIs at low cost).
#   TRACK_A_TUH_MAX_PER_SPLIT — cap *recordings per split* during preprocess.
#                                Default empty (= ingest everything).  Set to
#                                a small int (e.g. 50) for smoke-testing.
#   TRACK_A_SKIP_RSYNC=1 — skip rsync (assumes raw/{tuab,tuev}/ is populated)
#   TRACK_A_SKIP_PREPROCESS=1 — skip preprocess (assumes derived/ is populated)
#   TRACK_A_SKIP_SYNC=1 — skip the S3 sync at the end
#
# Resumability: every step is idempotent. rsync re-uses local files; preprocess
# skips already-written parquet; sync-derived-up is rclone copy (skips matching
# objects). Re-running after a partial completion only re-does what's missing.
#
# Cost expectation: ~1.5 h on g5.8xlarge ≈ $3.70 of compute (or g6.8xlarge if
# g5 capacity is tight; same NVMe + similar $/h).

set -uo pipefail

# -------------------------------------------------------------------
# setup
# -------------------------------------------------------------------
: "${EEG_DATA_ROOT:=/opt/dlami/nvme/eeg}"
export EEG_DATA_ROOT

SCRATCH="${EEG_DATA_ROOT}/scratch"
mkdir -p "$SCRATCH"

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/eegModel}"
RESULTS_MD="${REPO_ROOT}/experiments/01_sanity_baselines/results.md"

# Activate the eegfm venv so `eegfm` is on PATH
if [[ -d "${REPO_ROOT}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# Run-ID rooted at the timestamp; everything we write lands under this.
RUN_TS="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RUN_DIR="${EEG_DATA_ROOT}/runs/01_sanity_baselines/${RUN_TS}_track_a"
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/track_a.log"

ts()  { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "==============================================================="
log "Track A: Protocol A.4 (TUH) floor on $(hostname)"
log "PID $$"
log "EEG_DATA_ROOT=$EEG_DATA_ROOT"
log "RUN_DIR=$RUN_DIR"
log "==============================================================="

# Sanity-check the SSH chain BEFORE we kick off the long jobs.
if [[ "${TRACK_A_SKIP_RSYNC:-0}" != "1" ]]; then
  log "checking nedc-tuh SSH access (TEST file pull)..."
  if ! rsync -aL nedc-tuh:data/tuh_eeg/TEST "$RUN_DIR/TEST.smoke" >>"$LOG" 2>&1; then
    log "ERROR: nedc-tuh access not working. Did you forward the SSH agent?"
    log "       From your Mac:  ssh -A ubuntu@<gpu-box-ip>"
    log "       Then re-run this script."
    exit 1
  fi
  log "  TEST file pulled OK ($(stat -c %s "$RUN_DIR/TEST.smoke" 2>/dev/null || stat -f %z "$RUN_DIR/TEST.smoke") bytes)"
fi

# -------------------------------------------------------------------
# step 1: rsync TUAB v3.0.1 + TUEV v2.0.1 from NEDC
# -------------------------------------------------------------------
if [[ "${TRACK_A_SKIP_RSYNC:-0}" == "1" ]]; then
  log "step 1: SKIPPED (TRACK_A_SKIP_RSYNC=1)"
else
  log "step 1: rsync TUAB v3.0.1"
  if ! eegfm tuh-rsync tuab >>"$LOG" 2>&1; then
    log "ERROR: TUAB rsync failed; see $LOG"
    exit 1
  fi
  log "step 1: rsync TUEV v2.0.1"
  if ! eegfm tuh-rsync tuev >>"$LOG" 2>&1; then
    log "ERROR: TUEV rsync failed; see $LOG"
    exit 1
  fi
fi

# -------------------------------------------------------------------
# step 2: preprocess via SPEC_V2_CLEAN
# -------------------------------------------------------------------
if [[ "${TRACK_A_SKIP_PREPROCESS:-0}" == "1" ]]; then
  log "step 2: SKIPPED (TRACK_A_SKIP_PREPROCESS=1)"
else
  PRE_FLAGS=()
  if [[ -n "${TRACK_A_TUH_MAX_PER_SPLIT:-}" ]]; then
    PRE_FLAGS+=("--n" "${TRACK_A_TUH_MAX_PER_SPLIT}")
  fi
  log "step 2a: preprocess TUAB (v2_clean${TRACK_A_TUH_MAX_PER_SPLIT:+, n=${TRACK_A_TUH_MAX_PER_SPLIT}})"
  if ! eegfm tuh-preprocess tuab --pipeline v2_clean "${PRE_FLAGS[@]}" >>"$LOG" 2>&1; then
    log "ERROR: TUAB preprocess failed; see $LOG"
    exit 1
  fi
  log "step 2b: preprocess TUEV (v2_clean${TRACK_A_TUH_MAX_PER_SPLIT:+, n=${TRACK_A_TUH_MAX_PER_SPLIT}})"
  if ! eegfm tuh-preprocess tuev --pipeline v2_clean "${PRE_FLAGS[@]}" >>"$LOG" 2>&1; then
    log "ERROR: TUEV preprocess failed; see $LOG"
    exit 1
  fi
fi

# -------------------------------------------------------------------
# step 3: sync derived parquet to S3 (post-preprocess so we don't lose
#         the work if the box dies between probe runs)
# -------------------------------------------------------------------
if [[ "${TRACK_A_SKIP_SYNC:-0}" == "1" ]]; then
  log "step 3: SKIPPED (TRACK_A_SKIP_SYNC=1)"
else
  log "step 3: sync derived/{tuab,tuev}_v2_clean_250hz/ → S3"
  if ! eegfm sync-derived-up --pipeline tuh >>"$LOG" 2>&1; then
    log "WARN: TUH sync failed (continuing — parquet is still on NVMe)"
  fi
fi

# -------------------------------------------------------------------
# step 4 + 5: extract features + run Protocol A.4 (random-init floor)
# -------------------------------------------------------------------
TUH_MAX_SUBJECTS="${TRACK_A_TUH_MAX_SUBJECTS:-100}"

# We DON'T re-run the HBN floor; that's already in results.md and the
# Mamba-2 random-init produces the same numbers up to the per-seed band.
# What we want is just the A.4 row. But check_d's API requires a HBN
# derived_root (the primary; eval.py refuses to start without it). Pass
# the same HBN root the existing run used and accept that the HBN
# numbers will be re-printed; they're not what we'll commit to results.md.
HBN_DERIVED="${EEG_DATA_ROOT}/derived/hbn_minimal_500hz"
TUAB_DERIVED="${EEG_DATA_ROOT}/derived/tuab_v2_clean_250hz"
TUEV_DERIVED="${EEG_DATA_ROOT}/derived/tuev_v2_clean_250hz"

if [[ ! -d "$HBN_DERIVED" ]]; then
  log "step 4: HBN parquet not on NVMe; pulling from S3 (small subset for the probe call)..."
  if ! eegfm sync-derived-down --pipeline minimal >>"$LOG" 2>&1; then
    log "ERROR: HBN sync-derived-down failed; the random-init probe needs HBN to even start"
    exit 1
  fi
fi

OUT_JSON="${RUN_DIR}/check_d_with_a4.json"
log "step 4 + 5: random-init linear-probe floor with Protocol A.4"
log "  HBN max_subjects=50, TUH max_subjects=$TUH_MAX_SUBJECTS"
log "  output JSON: $OUT_JSON"

if ! python -m eegfm.sanity check-d \
      --derived-root "$HBN_DERIVED" \
      --max-subjects 50 \
      --max-windows-per-shard 20 \
      --derived-root-tuab "$TUAB_DERIVED" \
      --derived-root-tuev "$TUEV_DERIVED" \
      --tuh-max-subjects "$TUH_MAX_SUBJECTS" \
      --output-json "$OUT_JSON" >>"$LOG" 2>&1; then
  log "ERROR: check-d run failed; see $LOG"
  exit 1
fi

log "step 6: render the Protocol A.4 floor row from $OUT_JSON"

python - <<PYEOF | tee -a "$LOG"
import json, datetime, pathlib

j = json.loads(pathlib.Path("$OUT_JSON").read_text())
md = j["details"].get("metrics_a4", {})

def _fmt(d, key):
    if not isinstance(d, dict) or "point" not in d:
        return f"{d.get('reason') if isinstance(d, dict) else d}"
    return f"{d['point']:.4f} [{d.get('ci_low_95', float('nan')):.4f}, {d.get('ci_high_95', float('nan')):.4f}]"

print()
print("===========================================================")
print("Protocol A.4 floor (paste into 01_sanity_baselines/results.md)")
print("===========================================================")
print(f"_Filled in: {datetime.datetime.utcnow().isoformat()}Z_")
print(f"_Run JSON:  $OUT_JSON_")

print()
print("| metric | point | 95% CI |")
print("|--------|------:|-------:|")
tuab = md.get("tuab", {})
tuev = md.get("tuev", {})

if "tuab_binary_auroc" in tuab:
    d = tuab["tuab_binary_auroc"]
    print(f"| TUAB binary AUROC | {d['point']:.4f} | [{d.get('ci_low_95', float('nan')):.4f}, {d.get('ci_high_95', float('nan')):.4f}] |")
else:
    print(f"| TUAB binary AUROC | — | {tuab.get('reason', 'MISSING')} |")

for name, label in (("tuev_bac", "TUEV 6-class BAC"),
                    ("tuev_wf1", "TUEV 6-class WF1"),
                    ("tuev_knn_top1", "TUEV k-NN top-1 (k=5, cosine)")):
    d = tuev.get(name, {})
    if "point" in d:
        print(f"| {label} | {d['point']:.4f} | [{d.get('ci_low_95', float('nan')):.4f}, {d.get('ci_high_95', float('nan')):.4f}] |")
    else:
        print(f"| {label} | — | {d.get('reason', 'MISSING')} |")

print()
print("(also see ${OUT_JSON} for n_train, n_test, class distribution, split mode)")
PYEOF

log "==============================================================="
log "Track A complete. The floor row is printed above; copy it into:"
log "    $RESULTS_MD"
log "  (under the existing Check D 'Headline ablation floor' table)"
log "==============================================================="

# -------------------------------------------------------------------
# step 7: ship the run JSON to S3 (permanent record)
# -------------------------------------------------------------------
if [[ "${TRACK_A_SKIP_SYNC:-0}" != "1" ]]; then
  log "step 7: sync $RUN_DIR → s3://eegmodel-warehouse/runs/eegfm/01_sanity_baselines/${RUN_TS}_track_a/"
  aws s3 cp --recursive "$RUN_DIR" \
    "s3://eegmodel-warehouse/runs/eegfm/01_sanity_baselines/${RUN_TS}_track_a/" \
    >>"$LOG" 2>&1 || log "WARN: S3 cp failed (run JSON is still on NVMe at $RUN_DIR)"
fi

log "DONE."
