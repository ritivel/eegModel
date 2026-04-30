# Pilot progress — Apr 30, 2026

Live record of all training runs, where they are, and when they're expected to
finish. All times in **IST**.

> **Pivot at ~14:00 IST**: the pilot launched at 13:39 IST completed but hit
> three bugs (a noise-cell `NameError`, a LoRA target-modules regex that
> wrapped Gemma 4's vision tower instead of the language model so stage-3
> grad was structurally zero, and a collapsed bridge that produced
> identical hypotheses for different EEG inputs). All three are now fixed.
> The full 9-cell pilot was archived to `archive/buggy_run_2026-04-30T08-30/`
> and re-launched at 14:08 IST (Box A) / 14:00 IST (Box B) against the same
> step budget. Full root-cause analysis, code-fix snippets, and citations
> live in `[diagnostic_report.md](./diagnostic_report.md)`.

## Compute


| Box | Host                   | GPUs         | Notes                      |
| --- | ---------------------- | ------------ | -------------------------- |
| A   | `ubuntu@192.222.53.60` | 8× H100 80GB | bulk pilot                 |
| B   | `ubuntu@192.222.53.81` | 1× H100 80GB | first noise-train baseline |


SSH: `ssh -i ~/Downloads/modal_biosigtotext ubuntu@<host>`
W&B: [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text)

## Pilot config (every cell below)

- 3-stage curriculum: alignment → frozen-LM SFT → LoRA SFT
- `--stage1-steps 300 --stage2-steps 1200 --stage3-steps 500` = 2,000 steps total
- Soft-prompt cells (linear / qformer): `--batch-size 8 --no-grad-checkpoint --num-workers 4`
- Vocab cells: `--batch-size 1 --num-workers 4` (gradient checkpointing on; needed for the trainable extended embedding table)
- Decoder: `google/gemma-4-E2B-it` for everything (bf16, sdpa attention)
- **Stage-1 alignment loss** (new): CLIP-style InfoNCE between sentence-pooled
bridge output and sentence-pooled (frozen) text-token embeddings,
`weight=1.0`, `temperature=0.07`. Auto-disabled when `batch_size < 2`
(so off for vocab cells). Chance baseline = `log(batch_size)`.
- **RVQ commitment loss** (new): standard VQ-VAE commitment loss for
off-diagonal vocab cells (`reve.vocab`), `weight=1.0`, so the codebook
actually learns instead of staying at random init.

## Currently running (9 cells in parallel)

Launched 14:08 IST (Box A) / 14:00 IST (Box B). Snapshot at **15:03 IST**.


| Box   | GPU   | Cell                            | Started   | Stage @ snapshot      | Throughput       | ETA finish | W&B run                                                                                |
| ----- | ----- | ------------------------------- | --------- | --------------------- | ---------------- | ---------- | -------------------------------------------------------------------------------------- |
| A     | 0     | `reve.linear.eeg.0`             | 14:08     | stage 2, step 340     | ~1.15 s/step     | **~15:33** | `[g5luo0ae](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/g5luo0ae)`     |
| A     | 1     | `reve.qformer.eeg.0`            | 14:08     | stage 2, step 350     | ~1.15 s/step     | **~15:33** | `[u62ofgx5](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/u62ofgx5)`     |
| A     | 2     | `tfm.linear.eeg.0`              | 14:08     | stage 2, step 360     | ~1.15 s/step     | **~15:32** | `[me3f96bm](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/me3f96bm)`     |
| A     | 3     | `tfm.qformer.eeg.0`             | 14:08     | stage 2, step 330     | ~1.15 s/step     | **~15:34** | `[xxbzsx9u](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/xxbzsx9u)`     |
| A     | 4     | `reve.vocab.eeg.0`              | 14:08     | stage 2, step 100     | ~1.10 s/step     | **~15:50** | `[6tgxo9g8](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/6tgxo9g8)`     |
| A     | 5     | `tfm.vocab.eeg.0`               | 14:08     | stage 2, step 90      | ~1.20 s/step     | **~15:50** | `[8u7oxn9b](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/8u7oxn9b)`     |
| A     | 6     | `reve.linear.eeg.1`             | 14:10     | stage 2, step 270     | ~1.15 s/step     | **~15:38** | `[quvry6s2](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/quvry6s2)`     |
| A     | 7     | `tfm.linear.eeg.1`              | 14:10     | stage 2, step 280     | ~1.15 s/step     | **~15:38** | `[y088o2jd](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/y088o2jd)`     |
| **B** | **0** | `**reve.linear.noise_train.0`** | **14:00** | **stage 2, step 800** | **~1.20 s/step** | **~15:25** | `**[7ne592ia](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/7ne592ia)`** |


`(encoder).(bridge).(input).(fold)` — `input` is one of `eeg`, `noise_train`,
`noise_test`. `noise_train` replaces every EEG sample with a per-channel
Gaussian of matched mean/std (Jo et al. 2024 §4.3).

## Diagnostic series to watch

- `**stageN/loss`** — LM cross-entropy on the next-token target.
- `**stage1/align_loss`** — sentence-pooled CLIP InfoNCE. Chance = `log(batch_size)`
≈ **2.079 for bs=8**. Soft-prompt cells should drift below 2.0 on EEG and
stay at 2.08 on noise. (We already see `reve.linear.noise_train.0`
pinned at 2.08–2.11 throughout stage 1, which is the right diagnostic
behaviour for noise inputs.)
- `**stage1/commit_loss`** — VQ commitment loss for `reve.vocab` only.
Should decrease over stage 1 as the RVQ codebook moves toward the
encoder distribution. Pre-fix value was a *constant* (codebook never
updated). Already dropping: 39 → 35 → 28 by step 220.

## Pre-fix run, archived (do NOT use these for §4.3 conclusions)

`reve.linear.noise_train.0` failed before producing any metrics.
Everything else completed but with the bugs above; eval BLEU is therefore
misleading. Archived under `$EXP01_DATA_ROOT/archive/buggy_run_2026-04-30T08-30/`
on each box.


| Cell                      | Steps        | bs  | BLEU-1    | BLEU-4    | ROUGE-1-F | Box | Notes                       |
| ------------------------- | ------------ | --- | --------- | --------- | --------- | --- | --------------------------- |
| reve.linear.eeg.0         | 300/1200/500 | 8   | 0.107     | 0.004     | 0.116     | A   | bridge collapsed (LM prior) |
| reve.linear.eeg.1         | 300/1200/500 | 8   | 0.078     | 0.004     | 0.085     | A   | bridge collapsed            |
| reve.qformer.eeg.0        | 300/1200/500 | 8   | 0.089     | 0.004     | 0.100     | A   | bridge collapsed            |
| reve.vocab.eeg.0          | 300/1200/500 | 1   | **0.000** | **0.000** | **0.000** | A   | LoRA grad ≡ 0 + RVQ frozen  |
| tfm.linear.eeg.0          | 300/1200/500 | 8   | 0.115     | 0.004     | 0.007     | A   | bridge collapsed            |
| tfm.linear.eeg.1          | 300/1200/500 | 8   | 0.058     | 0.003     | 0.064     | A   | bridge collapsed            |
| tfm.qformer.eeg.0         | 300/1200/500 | 8   | 0.076     | 0.003     | 0.092     | A   | bridge collapsed            |
| tfm.vocab.eeg.0           | 300/1200/500 | 1   | **0.000** | **0.000** | **0.000** | A   | LoRA grad ≡ 0               |
| reve.linear.noise_train.0 | —            | —   | —         | —         | —         | B   | crashed (`NameError: np`)   |


## Wall-time summary (post-fix re-launch)

- **~15:25 IST** (≈ 85 min from 14:00 on Box B): Box B's noise cell wraps eval.
→ first matched (eeg, noise_train) pair: `reve.linear.eeg.0` × `reve.linear.noise_train.0`.
- **~15:33 IST** (≈ 85 min from 14:08): the four soft-prompt fold-0 cells +
two fold-1 cells finish stage 3 + eval on Box A.
- **~15:50 IST** (≈ 1h 42m from 14:08): both vocab cells on Box A finish.
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

> Note: pre-fix this test was useless because the bridge was collapsed AND
> the noise baseline never even ran. With the fix the test is meaningful for
> the first time.

## Per-cell artifacts (on the box that ran the cell)

```
$EXP01_DATA_ROOT/runs/<cell_id>/
  log.jsonl              per-step loss, align_loss, commit_loss, grad_norm, lr, stage; events
  sample_gens.jsonl      periodic dev (ref, hyp) snapshots during training
  stats.jsonl            encoder feature stats (mean, std, abs_max) per stage
  model_stage1.pt        checkpoint after Stage 1 (alignment)
  model_stage2.pt        checkpoint after Stage 2 (frozen-LM SFT)
  model.pt               final checkpoint (after Stage 3 if LoRA enabled)
  run.log                stdout/stderr from the parallel orchestrator

$EXP01_DATA_ROOT/eval/<cell_id>/
  metrics.json           summary mean / 95% CI for all 6 metrics
  predictions.parquet    one row per test example with full metadata

$EXP01_DATA_ROOT/archive/buggy_run_2026-04-30T08-30/
  runs/                  all pre-fix run dirs (8 cells on Box A, 3 on Box B)
  eval/                  pre-fix eval results
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

# Live training loss + new diagnostic series for a single cell
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

1. **At ~15:25 IST**, Box B's noise cell wraps. Compute the first EEG−noise
  gap from `reve.linear.eeg.0` (still mid-flight on Box A) once that cell
   also finishes (~15:33). If positive with a non-overlapping CI →
   REVE+linear is using EEG. Encouraging signal.
2. **At ~15:33 IST**, pull `metrics.json` + sample predictions for the 6
  soft-prompt cells. Confirm BLEU-1 has moved meaningfully above the
   buggy run (target: ≥ 0.10 for soft-prompt; previously the buggy
   "ceiling" of 0.115 came purely from collapsed-prefix LM-prior overlap,
   so a *real* improvement should look both higher *and* show that
   different EEG inputs produce different hypotheses).
3. **At ~15:50 IST**, vocab cells finish. Confirm BLEU-1 > 0 (was 0.000)
  and that `tfm.vocab.eeg.0` produces meaningful text instead of
   `'HeHeHeHe...'`.
4. **At ~16:00 IST**, full pilot done. Decide whether to:
  - Extend to 12k-step (overnight) runs of the surviving (encoder, bridge)
   pairs — gives publishable BLEU.
  - Add `noise_train` matches for `tfm.linear`, `tfm.vocab`, `reve.vocab`
  so we have full §4.3 coverage on Box B.
5. **Once confident in the pilot**, fan out to all 5 LNSO folds × 3 input
  conditions = the full §5 matrix. ~3 days on Box A, ~2.5 days on A+B.

