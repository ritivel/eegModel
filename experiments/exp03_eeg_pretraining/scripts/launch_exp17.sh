#!/usr/bin/env bash
# Launcher for the exp17 30-cell matrix:
#   3 paradigms (G0 MAE / G1 AR / G2 MAR)
#   × 2 controls   (EEG signal / matched-noise twin)
#   × 5 seeds      (0..4)
#   = 30 cells
#
# Runs 8 cells in parallel on 8 GPUs (1 GPU per cell). Iso-data per the README §17:
# 17500 steps × batch 32 → ≈ 35M tokens-seen per cell. wandb is offline (sync later).
#
# Usage:  bash scripts/launch_exp17.sh [paradigm-filter]
#   e.g.  bash scripts/launch_exp17.sh         # all 30 cells
#         bash scripts/launch_exp17.sh mae,ar  # only the 20 MAE+AR cells

set -uo pipefail

REPO=${REPO:-/home/ubuntu/eegModel}
EXP03=$REPO/experiments/exp03_eeg_pretraining
DATA=${EXP03_DATA_ROOT:-/opt/dlami/nvme/eeg}/derived/hbn_minimal_500hz
RUNS=${EXP03_DATA_ROOT:-/opt/dlami/nvme/eeg}/runs/exp17
LOG=${EXP03_DATA_ROOT:-/opt/dlami/nvme/eeg}/scratch/exp17_launch.log

STEPS=${STEPS:-17500}
BATCH=${BATCH:-32}
DIFFUSION_BATCH_MUL=${DIFFUSION_BATCH_MUL:-1}

mkdir -p "$RUNS" "$(dirname "$LOG")"
source "$REPO/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"

PARADIGMS=${1:-mae,ar,mar}
IFS=',' read -ra ALLOWED_PARADIGMS <<<"$PARADIGMS"

# Build the cell list
CELLS=()
for paradigm in "${ALLOWED_PARADIGMS[@]}"; do
  for control in eeg noise; do
    for seed in 0 1 2 3 4; do
      CELLS+=("$paradigm,$control,$seed")
    done
  done
done

total=${#CELLS[@]}
echo "[$(date -u +%H:%M:%S)] launching $total cells across 8 GPUs ($STEPS steps × batch $BATCH each)" | tee -a "$LOG"

run_cell() {
  local gpu=$1
  local triple=$2
  IFS=',' read paradigm control seed <<<"$triple"
  local cell_id="${paradigm}_${control}_seed${seed}"
  local out="$RUNS/$cell_id"
  mkdir -p "$out"
  local twin_flag=""
  [ "$control" = "noise" ] && twin_flag="--noise-twin"

  local mar_extra=""
  if [ "$paradigm" = "mar" ]; then
    mar_extra="--diffusion-batch-mul $DIFFUSION_BATCH_MUL"
  fi

  echo "[$(date -u +%H:%M:%S)] start $cell_id on GPU $gpu (pid $$)" >> "$LOG"
  local t0=$(date +%s)
  # Cap CPU thread fan-out per process so 8 parallel Python processes don't
  # contend on 96 cores. Each process gets ~10 cores worth of OMP/MKL/BLAS.
  # Use `export` (in the subshell) rather than inline prefix to avoid
  # bash line-continuation quirks across function boundaries.
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export NUMEXPR_NUM_THREADS=8
    export VECLIB_MAXIMUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false
    export WANDB_MODE=offline
    python -m exp03.cli train \
      --paradigm "$paradigm" --steps "$STEPS" --batch-size "$BATCH" --seed "$seed" \
      --num-workers 0 \
      --data-root "$DATA" \
      --output-dir "$out" \
      --wandb-mode offline \
      --wandb-run-name "exp17-$cell_id" \
      --wandb-tags "exp17,$paradigm,$control,seed$seed" \
      --eval-at-end --eval-max-subjects 50 \
      --warmup-steps 100 --log-every 200 \
      $twin_flag $mar_extra \
      > "$out/run.log" 2>&1
  )
  local rc=$?
  local t1=$(date +%s)
  local dur=$((t1 - t0))
  if [ $rc -eq 0 ]; then
    echo "[$(date -u +%H:%M:%S)] OK   $cell_id on GPU $gpu (${dur}s)" >> "$LOG"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL $cell_id on GPU $gpu (rc=$rc, ${dur}s) — see $out/run.log" >> "$LOG"
  fi
}

i=0
batch_n=0
while [ $i -lt $total ]; do
  batch_n=$((batch_n + 1))
  pids=()
  for gpu in 0 1 2 3 4 5 6 7; do
    [ $i -lt $total ] || break
    triple="${CELLS[$i]}"
    run_cell "$gpu" "$triple" &
    pids+=($!)
    i=$((i+1))
  done
  echo "[$(date -u +%H:%M:%S)] batch $batch_n: ${#pids[@]} cells in flight" | tee -a "$LOG"
  wait "${pids[@]}"
  echo "[$(date -u +%H:%M:%S)] batch $batch_n done ($i/$total cells finished)" | tee -a "$LOG"
done

echo "[$(date -u +%H:%M:%S)] ALL $total CELLS COMPLETE" | tee -a "$LOG"
