# Pilot progress — Apr 30, 2026

Live record of all training runs, where they are, and when they're expected to
finish. All times in **IST**.

## Compute

| Box | Host | GPUs | Notes |
|---|---|---|---|
| A | `ubuntu@192.222.53.60` | 8× H100 80GB | bulk pilot |
| B | `ubuntu@192.222.53.81` | 1× H100 80GB | first noise-train baseline |

SSH: `ssh -i ~/Downloads/modal_biosigtotext ubuntu@<host>`
W&B: <https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text>

## Pilot config (every cell below)

- 3-stage curriculum: alignment → frozen-LM SFT → LoRA SFT
- `--stage1-steps 300 --stage2-steps 1200 --stage3-steps 500` = 2,000 steps total
- Soft-prompt cells (linear / qformer): `--batch-size 8 --no-grad-checkpoint --num-workers 4`
- Vocab cells: `--batch-size 1 --num-workers 4` (gradient checkpointing on; needed for the trainable extended embedding table)
- Decoder: `google/gemma-4-E2B-it` for everything (bf16, sdpa attention)

## Currently running (9 cells in parallel)

Launched 13:39 IST (Box A) / 13:46 IST (Box B).

| Box | GPU | Cell | Started | Throughput | ETA finish | W&B run |
|---|---:|---|---|---|---|---|
| A | 0 | `reve.linear.eeg.0`         | 13:39 | ~0.8 s/step | **~14:14** | live |
| A | 1 | `reve.qformer.eeg.0`        | 13:39 | ~0.8 s/step | **~14:14** | live |
| A | 2 | `tfm.linear.eeg.0`          | 13:39 | ~0.8 s/step | **~14:14** | live |
| A | 3 | `tfm.qformer.eeg.0`         | 13:39 | ~0.8 s/step | **~14:14** | live |
| A | 4 | `reve.vocab.eeg.0`          | 13:39 | ~4   s/step | **~16:00** | live |
| A | 5 | `tfm.vocab.eeg.0`           | 13:39 | ~4   s/step | **~16:00** | live |
| A | 6 | `reve.linear.eeg.1`         | 13:39 | ~0.8 s/step | **~14:14** | live |
| A | 7 | `tfm.linear.eeg.1`          | 13:39 | ~0.8 s/step | **~14:14** | live |
| **B** | **0** | **`reve.linear.noise_train.0`** | **13:46** | **~0.8 s/step** | **~14:21** | live |

`(encoder).(bridge).(input).(fold)` — `input` is one of `eeg`, `noise_train`,
`noise_test`. `noise_train` replaces every EEG sample with a per-channel
Gaussian of matched mean/std (Jo et al. 2024 §4.3).

## Already complete (from previous, less-optimized runs)

These metrics were captured at lower step counts (50 or 250 steps) and lower
batch sizes — kept for reference; the current 2k-step pilot will overwrite the
`metrics.json` files for the same `cell_id`s.

| Cell | Steps | bs | BLEU-1 | BLEU-4 | ROUGE-1-F | Box |
|---|---:|---:|---:|---:|---:|---|
| reve.linear.eeg.0     | 250 | 1 | 0.015 | 0.002 | 0.007 | A |
| reve.qformer.eeg.0    | 250 | 1 | 0.001 | 0.000 | 0.000 | A |
| reve.vocab.eeg.0      | 250 | 1 | 0.000 | 0.000 | 0.000 | A |
| tfm.linear.eeg.0      | 250 | 1 | 0.052 | 0.009 | 0.000 | A |
| tfm.qformer.eeg.0     | 250 | 1 | 0.000 | 0.000 | 0.000 | A |
| tfm.vocab.eeg.0       | 250 | 1 | 0.000 | 0.000 | 0.000 | B |

## Wall-time summary

- **~14:14 IST** (≈ 30 min from 13:46): 6 of 8 Box-A cells + Box-B's noise cell ready.
  → 4 soft-prompt fold-0 cells + 2 fold-1 cells + 1 noise cell.
- **~14:21 IST** (≈ 35 min from 13:46): Box B's noise cell wraps eval.
  → first matched (eeg, noise_train) pair: `reve.linear.eeg.0` × `reve.linear.noise_train.0`.
- **~16:00 IST** (≈ 2h 15m from 13:46): both vocab cells on Box A finish.
  → full pilot complete.

## After everything finishes

```bash
# On Box A (or rsync the /home/ubuntu/data/exp01/eval/ dir to one host first):
.venv/bin/python -c "
from exp01 import eval as ev, storage
import json
def load(c): return json.load(open(storage.EVAL/c/'metrics.json'))
eeg   = load('reve_linear_eeg_fold0_dec-gemma4-e2b')
noise = load('reve_linear_noise_train_fold0_dec-gemma4-e2b')
print(ev.eeg_noise_gap(eeg, noise))
"
```

A positive gap with the bootstrap 95% CI strictly above 0 = the model is
actually using the EEG signal (not just the language prior). A non-positive gap
or a CI that crosses 0 = not using EEG (per Jo et al. 2024 §4.3).

## Per-cell artifacts (on the box that ran the cell)

```
$EXP01_DATA_ROOT/runs/<cell_id>/
  log.jsonl              per-step loss, grad_norm, lr, stage; events
  sample_gens.jsonl      periodic dev (ref, hyp) snapshots during training
  stats.jsonl            encoder feature stats (mean, std, abs_max) per stage
  model_stage1.pt        checkpoint after Stage 1 (alignment)
  model_stage2.pt        checkpoint after Stage 2 (frozen-LM SFT)
  model.pt               final checkpoint (after Stage 3 if LoRA enabled)
  run.log                stdout/stderr from the parallel orchestrator

$EXP01_DATA_ROOT/eval/<cell_id>/
  metrics.json           summary mean / 95% CI for all 6 metrics
  predictions.parquet    one row per test example with full metadata
```

`$EXP01_DATA_ROOT` = `/home/ubuntu/data/exp01` on both boxes.

## Quick monitoring commands

```bash
# Live GPU utilization (Box A)
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "watch -n2 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"

# Pilot orchestrator logs (Box A)
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "tail -f /tmp/pilot-soft.log /tmp/pilot-vocab.log /tmp/pilot-fold1.log"

# Live training loss for a single cell
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "tail -f /home/ubuntu/data/exp01/runs/reve_linear_eeg_fold0_dec-gemma4-e2b/log.jsonl"

# Pull a finished cell's metrics + first 3 predictions
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "cd ~/work/eegModel/experiments/exp01_eeg_to_text && \
   .venv/bin/python -c \"
import json, pyarrow.parquet as pq
m = json.load(open('/home/ubuntu/data/exp01/eval/reve_linear_eeg_fold0_dec-gemma4-e2b/metrics.json'))
print({k: v['mean'] for k, v in m['scores'].items()})
for r in pq.read_table('/home/ubuntu/data/exp01/eval/reve_linear_eeg_fold0_dec-gemma4-e2b/predictions.parquet').to_pylist()[:3]:
    print('REF:', repr(r['ref'][:120]))
    print('HYP:', repr(r['hyp'][:120]))\""
```

## Next-step suggestions (ordered)

1. **At ~14:14 IST**, fetch `metrics.json` + sample predictions for the 6
   fast cells. Confirm BLEU-1 has moved off zero on at least the linear
   cells (target: ≥ 0.05 at 2k steps).
2. **At ~14:21 IST**, compute the first EEG−noise gap from
   `reve.linear.eeg.0` vs `reve.linear.noise_train.0`. If positive with a
   non-overlapping CI → REVE+linear is using EEG. Encouraging signal.
3. **At ~16:00 IST**, full pilot done. Decide whether to:
   - Extend to 12k-step (overnight) runs of the surviving (encoder, bridge)
     pairs — gives publishable BLEU.
   - Add `noise_train` matches for `tfm.linear`, `tfm.vocab`, `reve.vocab`
     so we have full §4.3 coverage on Box B.
4. **Once confident in the pilot**, fan out to all 5 LNSO folds × 3 input
   conditions = the full §5 matrix. ~3 days on Box A, ~2.5 days on A+B.
