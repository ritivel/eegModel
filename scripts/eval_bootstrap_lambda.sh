#!/usr/bin/env bash
# scripts/eval_bootstrap_lambda.sh
# ------------------------------------------------------------------
# Set up a Lambda 1×A100 box for running `eegfm-eval` reproduction tests.
# This is the "first-time" bootstrap; idempotent on re-runs.
#
# Usage (from your Mac, with eeg-gpu-key-west.pem and an instance IP):
#
#     ssh -A -i ~/.ssh/eeg-gpu-key-west.pem ubuntu@<lambda-ip> \
#         'bash -s' < scripts/eval_bootstrap_lambda.sh
#
# Steps:
#   1. install uv, create the venv at /home/ubuntu/eegfm/.venv
#   2. pull this repo + install eegfm + eegfm_eval (with [gpu])
#   3. install mamba-ssm + causal-conv1d (build-from-source on this box's CUDA)
#   4. sync TUAB+TUEV preprocessed parquet from S3 → /home/ubuntu/data/derived/
#   5. download LaBraM-Base + CBraMod weights for reproduction tests
#   6. smoke-test the harness end-to-end with synthetic data

set -uo pipefail
DATA=${EEG_DATA_ROOT:-/home/ubuntu/data}
REPO=${REPO:-/home/ubuntu/eegModel}
mkdir -p "$DATA"/{derived,scratch,checkpoints,runs}

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

log "==============================================================="
log "Lambda eval-bootstrap on $(hostname)"
log "DATA=$DATA  REPO=$REPO"
log "==============================================================="

# --- 1. uv + venv -----------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- 2. clone or refresh repo ---------------------------------------
if [ ! -d "$REPO/.git" ]; then
  log "ERROR: repo not at $REPO. ssh -A and `git clone` first, or rsync from your Mac."
  exit 1
fi
cd "$REPO"
log "current commit: $(git rev-parse --short HEAD)  ($(git log -1 --pretty=%s))"

if [ ! -d "$REPO/.venv" ]; then
  log "creating venv"
  uv venv --python 3.11 .venv
fi
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

log "installing eegfm + eegfm_eval (GPU extras)"
uv pip install -e ".[gpu]"

# --- 3. mamba-ssm + causal-conv1d (build from source against the venv's torch) ----
if ! python -c "import mamba_ssm" 2>/dev/null; then
  log "installing causal-conv1d + mamba-ssm (build-from-source, ~10-15 min)"
  export TORCH_CUDA_ARCH_LIST="8.0"   # Lambda A100 = sm_80
  export MAMBA_FORCE_BUILD=TRUE
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  export MAX_JOBS=16
  uv pip install --no-build-isolation causal-conv1d==1.6.1 mamba-ssm==2.3.1
fi

# --- 4. sync TUH preprocessed parquet from S3 -----------------------
log "syncing derived/tuab_v2_clean_250hz/ from S3 (~2 GB)"
aws s3 sync s3://eegmodel-warehouse/derived/tuab_v2_clean_250hz/ \
            "$DATA/derived/tuab_v2_clean_250hz/" --no-progress

log "syncing derived/tuev_v2_clean_250hz/ from S3 (~1 GB)"
aws s3 sync s3://eegmodel-warehouse/derived/tuev_v2_clean_250hz/ \
            "$DATA/derived/tuev_v2_clean_250hz/" --no-progress

du -sh "$DATA/derived/"*

# --- 5. fetch pretrained weights for reproduction tests --------------
mkdir -p "$DATA/checkpoints"
if [ ! -f "$DATA/checkpoints/labram-base.pth" ]; then
  log "downloading LaBraM-Base from HuggingFace"
  # NOTE: exact URL TBD by research subagent — placeholder here.
  # Update to the real Hub URL once known. This is a ~150 MB file.
  : "${LABRAM_BASE_URL:?Set LABRAM_BASE_URL env or fill in once subagent returns}"
  curl -L -o "$DATA/checkpoints/labram-base.pth" "$LABRAM_BASE_URL"
fi

if [ ! -f "$DATA/checkpoints/cbramod-pretrained.pth" ]; then
  log "downloading CBraMod from HuggingFace"
  : "${CBRAMOD_URL:?Set CBRAMOD_URL env or fill in once subagent returns}"
  curl -L -o "$DATA/checkpoints/cbramod-pretrained.pth" "$CBRAMOD_URL"
fi

# --- 6. smoke-test the harness --------------------------------------
log "smoke-testing eegfm-eval (random-init encoder, synthetic task)"
EEG_DATA_ROOT="$DATA" eegfm-eval --random-init --task smoke --strategy lp --device cuda \
  --output-dir "$DATA/runs/smoke" 2>&1 | tail -20

log "==============================================================="
log "Bootstrap complete. Now run reproduction tests, e.g.:"
log "  eegfm-eval --reproduce labram_base --reproduce-checkpoint $DATA/checkpoints/labram-base.pth \\"
log "             --task tuab --strategy ft --derived-root $DATA/derived"
log "==============================================================="
