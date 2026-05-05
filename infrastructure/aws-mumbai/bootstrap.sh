#!/usr/bin/env bash
# bootstrap.sh — runs on the GPU box, ONCE, after first SSH.
#
# Goal: take a fresh DLAMI p5.48xlarge from boot to "exp03 train works
# against in-region preprocessed data" in ~10-15 minutes, idempotently.
#
# What it does (each step idempotent — re-run is safe):
#   1. /opt/dlami/nvme/eeg directory + ownership
#   2. uv (fast Python package manager)
#   3. rclone configured for the Mumbai bucket
#   4. exp03 venv + editable installs (eeg_common, exp01, exp02, exp03)
#   5. wandb login (if WANDB_API_KEY is set) + HF login (if HF_TOKEN)
#   6. sync derived shards from Mumbai mirror → NVMe
#   7. nvidia-smi sanity check + report
#
# This script reads /opt/eeg-config/env (written by user-data on first
# boot). If you launched manually without our user-data, set the same vars
# in your shell before running.
#
# Time budget on a p5.48xlarge with the Mumbai mirror prewarmed:
#   step 1-5:  ~3 min (uv install + venv + wheels)
#   step 6:    ~5 min (250-300 GB rclone @ ~500 MB/s in-region)
#   step 7:    instant
#
# Resumable: each step has a sentinel file under /opt/eeg-config/done/. Re-
# running skips completed steps unless you `rm -rf /opt/eeg-config/done/`.

set -uo pipefail

DONE_DIR="/opt/eeg-config/done"
sudo mkdir -p "$DONE_DIR"
sudo chown -R "$USER:$USER" /opt/eeg-config

if [[ -f /opt/eeg-config/env ]]; then
  # shellcheck disable=SC1091
  source /opt/eeg-config/env
fi

REPO_ROOT="${REPO_ROOT:-$HOME/eegModel}"
DATA_ROOT="${EXP03_DATA_ROOT:-/opt/dlami/nvme/eeg}"
BUCKET="${EXP03_S3_BUCKET:-eeg-mumbai-139156132535}"
BUCKET_REGION="${EXP03_S3_REGION:-ap-south-1}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[bootstrap $(ts)] $*"; }
done_or_run() {
  local name="$1"; shift
  if [[ -f "$DONE_DIR/$name" ]]; then
    log "step '$name': SKIP (already done)"
    return 0
  fi
  log "step '$name': running..."
  if "$@"; then
    touch "$DONE_DIR/$name"
    log "step '$name': OK"
  else
    log "step '$name': FAILED"
    return 1
  fi
}

# -------------------------------------------------------------------
# 1. NVMe data root
# -------------------------------------------------------------------
step_data_root() {
  sudo mkdir -p "$DATA_ROOT"
  sudo chown -R "$USER:$USER" "$DATA_ROOT"
  mkdir -p "$DATA_ROOT"/{raw/hbn,raw/tuab,raw/tuev,derived,runs,models/hf_cache,scratch}
}

# -------------------------------------------------------------------
# 2. uv (Python package manager — 10-100× faster than pip for our case)
# -------------------------------------------------------------------
step_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    grep -q 'HOME/.local/bin' "$HOME/.bashrc" || \
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  fi
  uv --version
}

# -------------------------------------------------------------------
# 3. rclone — configured for Mumbai mirror
# -------------------------------------------------------------------
step_rclone() {
  if ! command -v rclone >/dev/null 2>&1; then
    curl -fsSL https://rclone.org/install.sh | sudo bash
  fi
  mkdir -p "$HOME/.config/rclone"
  # The exp03 sync-derived-down expects an `s3:` rclone remote. We add two:
  #   `s3:`  — Mumbai mirror (this week's source of derived/)
  #   `s3w:` — us-west-2 warehouse (for end-of-week sync of runs/)
  cat > "$HOME/.config/rclone/rclone.conf" <<EOF
[s3]
type = s3
provider = AWS
env_auth = true
region = $BUCKET_REGION
bucket = $BUCKET

[s3w]
type = s3
provider = AWS
env_auth = true
region = ${S3_WAREHOUSE_REGION:-us-west-2}
bucket = ${S3_WAREHOUSE_BUCKET:-eegmodel-warehouse}
EOF
  rclone listremotes
}

# -------------------------------------------------------------------
# 4. clone repo + venv + editable installs
# -------------------------------------------------------------------
step_repo() {
  if [[ ! -d "$REPO_ROOT" ]]; then
    git clone "${REMOTE_REPO_URL:?REMOTE_REPO_URL not set in /opt/eeg-config/env}" "$REPO_ROOT"
  fi
  cd "$REPO_ROOT"
  git checkout "${REMOTE_REPO_BRANCH:-main}"
  git pull --ff-only || true
}

step_venv() {
  cd "$REPO_ROOT"
  if [[ ! -d .venv ]]; then
    uv venv .venv --python 3.11
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -e packages/eeg_common
  for exp in exp01_eeg_to_text exp02_eeg_ctc exp03_eeg_pretraining; do
    if [[ -f "experiments/$exp/pyproject.toml" ]]; then
      uv pip install -e "experiments/$exp"
    fi
  done
  # H100-targeted installs that aren't always in the experiment pyprojects:
  uv pip install -U mamba-ssm causal-conv1d "torch>=2.7" wandb accelerate || true
  python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}')"
}

# -------------------------------------------------------------------
# 5. login services (gated on env vars)
# -------------------------------------------------------------------
step_logins() {
  cd "$REPO_ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --relogin "$WANDB_API_KEY" || true
  else
    log "  WANDB_API_KEY not set — runs will log offline (you can sync later)"
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
  fi
}

# -------------------------------------------------------------------
# 6. sync derived shards from Mumbai mirror
# -------------------------------------------------------------------
step_sync_derived() {
  cd "$REPO_ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export EXP03_DATA_ROOT="$DATA_ROOT"
  # Use rclone directly so we get the in-region speed on the Mumbai bucket.
  for pl in hbn_minimal_500hz hbn_v2_clean_250hz tuab_v2_clean_250hz tuev_v2_clean_250hz; do
    if rclone lsd "s3:derived/$pl" >/dev/null 2>&1; then
      log "  rclone copy s3:derived/$pl → $DATA_ROOT/derived/$pl/"
      rclone copy "s3:derived/$pl" "$DATA_ROOT/derived/$pl/" \
        --transfers 64 --checkers 64 --s3-chunk-size 64M
    else
      log "  s3:derived/$pl not present in Mumbai mirror; skipping"
    fi
  done
  du -sh "$DATA_ROOT/derived"/* 2>/dev/null || true
}

# -------------------------------------------------------------------
# 7. GPU sanity
# -------------------------------------------------------------------
step_gpu_check() {
  nvidia-smi
  echo
  echo "GPU layout per torch:"
  cd "$REPO_ROOT" && source .venv/bin/activate && python - <<'EOF'
import torch
n = torch.cuda.device_count()
print(f"GPU count: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/2**30:.1f} GB  cc={p.major}.{p.minor}")
EOF
}

# -------------------------------------------------------------------
# Run them
# -------------------------------------------------------------------
log "==============================================="
log "  bootstrap on $(hostname)"
log "  REPO_ROOT=$REPO_ROOT"
log "  DATA_ROOT=$DATA_ROOT"
log "  BUCKET=s3://$BUCKET ($BUCKET_REGION)"
log "==============================================="

done_or_run "01_data_root"   step_data_root
done_or_run "02_uv"          step_uv
done_or_run "03_rclone"      step_rclone
done_or_run "04_repo"        step_repo
done_or_run "05_venv"        step_venv
done_or_run "06_logins"      step_logins
done_or_run "07_sync_derived" step_sync_derived
done_or_run "08_gpu_check"   step_gpu_check

log "==============================================="
log "  bootstrap complete."
log "  Next:  source $REPO_ROOT/.venv/bin/activate"
log "         exp03 paths   # sanity-check storage layout"
log "         exp03 train --paradigm mae --steps 100 --wandb-mode disabled --no-eval-at-end"
log "==============================================="
