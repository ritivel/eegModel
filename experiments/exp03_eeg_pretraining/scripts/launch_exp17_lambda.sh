#!/usr/bin/env bash
# Single-GPU sequential launcher for the MAR re-run (with diffusion_batch_mul=4
# per the canonical MAR paper recipe).
#
# Designed for Lambda 1× A100 boxes where there's only one GPU and we
# need the cells to run one after another. Cells are still iso-data per
# the README's §17 spec (17500 steps × batch 32 ≈ 35M tokens-seen).
#
# Usage:  bash scripts/launch_exp17_lambda.sh [paradigm-filter]
#   default: mar  (10 cells: 5 mar_eeg + 5 mar_noise)
#   override: bash scripts/launch_exp17_lambda.sh mae,ar  to run those instead

set -uo pipefail

REPO=${REPO:-/home/ubuntu/exp03}
DATA=${EXP03_DATA_ROOT:-/home/ubuntu/data}/hbn_minimal_500hz
RUNS=${EXP03_DATA_ROOT:-/home/ubuntu/data}/runs/exp17_mar_mul4
LOG=${EXP03_DATA_ROOT:-/home/ubuntu/data}/scratch/exp17_lambda.log

STEPS=${STEPS:-17500}
BATCH=${BATCH:-32}
DIFFUSION_BATCH_MUL=${DIFFUSION_BATCH_MUL:-4}

mkdir -p "$RUNS" "$(dirname "$LOG")"
source "$REPO/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"

PARADIGMS=${1:-mar}
IFS=',' read -ra ALLOWED_PARADIGMS <<<"$PARADIGMS"

# Build the cell list (all seeds, both controls, given paradigms).
CELLS=()
for paradigm in "${ALLOWED_PARADIGMS[@]}"; do
  for control in eeg noise; do
    for seed in 0 1 2 3 4; do
      CELLS+=("$paradigm,$control,$seed")
    done
  done
done

total=${#CELLS[@]}
echo "[$(date -u +%H:%M:%S)] launching $total cells SEQUENTIALLY on GPU 0 ($STEPS steps × batch $BATCH × diff_mul $DIFFUSION_BATCH_MUL)" | tee -a "$LOG"

run_cell() {
  local triple=$1
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

  echo "[$(date -u +%H:%M:%S)] start $cell_id" >> "$LOG"
  local t0=$(date +%s)

  (
    export CUDA_VISIBLE_DEVICES=0
    # Cap thread fan-out — even single-process, we don't want OMP grabbing
    # all 30 vCPUs and contending with our DataLoader.
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export NUMEXPR_NUM_THREADS=8
    export VECLIB_MAXIMUM_THREADS=8
    export TOKENIZERS_PARALLELISM=false
    export WANDB_MODE=online
    cd "$REPO"
    python -m exp03.cli train \
      --paradigm "$paradigm" --steps "$STEPS" --batch-size "$BATCH" --seed "$seed" \
      --num-workers 0 \
      --data-root "$DATA" \
      --output-dir "$out" \
      --wandb-mode online \
      --wandb-project exp03 \
      --wandb-run-name "exp17b-$cell_id" \
      --wandb-tags "exp17b,exp17_mar_mul4,$paradigm,$control,seed$seed" \
      --eval-at-end --eval-max-subjects 50 \
      --warmup-steps 100 --log-every 200 \
      $twin_flag $mar_extra \
      > "$out/run.log" 2>&1
  )
  local rc=$?
  local t1=$(date +%s)
  local dur=$((t1 - t0))
  if [ $rc -eq 0 ]; then
    echo "[$(date -u +%H:%M:%S)] OK   $cell_id (${dur}s)" >> "$LOG"
  else
    echo "[$(date -u +%H:%M:%S)] FAIL $cell_id (rc=$rc, ${dur}s) — see $out/run.log" >> "$LOG"
  fi
}

for i in $(seq 0 $((total - 1))); do
  triple="${CELLS[$i]}"
  run_cell "$triple"
done

echo "[$(date -u +%H:%M:%S)] ALL $total CELLS COMPLETE" | tee -a "$LOG"
