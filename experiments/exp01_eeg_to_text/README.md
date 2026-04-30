# Exp01 — EEG-to-Text Decoding

Fine-tune three EEG foundation models — REVE, DIVER-1, TFM-Tokenizer — into
open-vocabulary EEG-to-text decoders, reporting on ZuCo v1+v2 with the
**unique-sentence + leave-N-subjects-out** split (Yin et al. 2024) and the
**mandatory noise baseline** (Jo et al. 2024). Full design: `report/main.pdf`.

Runs on a single GPU box (H100 80 GB or A100 40/80 GB). One Python package +
one CLI; no external orchestrator.

## Layout

```
src/exp01/
  storage.py     Local paths under $EXP01_DATA_ROOT
  config.py      CellConfig dataclass + run-matrix helpers
  data.py        HF download, Yin unique-sentence split, LNSO folds, Jo noise twin
  encoders.py    REVE, DIVER-1, TFM-Tokenizer (uniform encode() interface)
  bridges.py     Linear+SoftPrompt, Q-Former, Vocab-extension, RVQHead
  decoder.py     Gemma 4 loader, freeze, LoRA, vocab extension
  model.py       EEG2Text(encoder, bridge, decoder)
  train.py       3-stage curriculum (align → frozen-LM SFT → LoRA SFT)
  eval.py        BLEU 1–4 / ROUGE-1-F / BERTScore-F1 + bootstrap CIs + perm test
  cli.py         exp01 ... entry point

report/          experimental design, methodology, references
pyproject.toml
```

## Setup

```bash
# 1. Clone + install (Python ≥3.11; uv recommended).
git clone https://github.com/ritivel/eegModel.git
cd eegModel/experiments/exp01_eeg_to_text
uv venv && source .venv/bin/activate
uv pip install -e .

# 2. Env vars.
export EXP01_DATA_ROOT=$HOME/data/exp01     # all artifacts land here (~150 GB)
export HF_TOKEN=hf_xxx                       # gated repos
export WANDB_API_KEY=...                     # optional; runs offline if unset
export WANDB_PROJECT=exp01-eeg-to-text       # optional override

# 3. The HF_TOKEN must have read access to (you've accepted the EUAs):
#   - tankalapavankalyan/exp01-eeg-to-text-sentences  (gated dataset, ~72 GB)
#   - brain-bzh/reve-base                              (gated REVE encoder)
#   - google/gemma-4-E2B-it                            (open)

# 4. DIVER-1 has no public HF mirror. Drop its weights at:
#   $EXP01_DATA_ROOT/diver1/pytorch_model.bin
# (skip this if you only want REVE + TFM cells)
```

## Run order

```bash
exp01 download-models       # ~5 min  — caches REVE + TFM + Gemma 4 in HF cache
exp01 download-data         # ~30 min — caches ~72 GB unified dataset
exp01 inspect-channels      # quick   — sanity-check channel naming per source
exp01 make-splits           # ~2 min  — computes the 5 LNSO folds
exp01 smoke                 # ~5 min  — end-to-end micro-cell per code path
exp01 pilot --encoders reve,tfm   # Phase 1 (fold 0, EEG, 6 cells)
exp01 full                  # Phase 3 (fan out across 5 folds × 3 inputs)
```

Single-cell helpers for debugging:

```bash
exp01 train reve.linear.eeg.0     # train one cell (encoder.bridge.input.fold)
exp01 eval  reve.linear.eeg.0     # evaluate one cell

# Override step counts / batch size / LoRA at the CLI:
exp01 train reve.linear.eeg.0 --stage1-steps 100 --stage2-steps 400 --stage3-steps 200 --batch-size 1
exp01 train tfm.vocab.eeg.0 --no-lora
```

## Multi-machine pilot

`exp01 pilot --parallel` round-robins one cell per visible GPU. To shard the
pilot across two machines:

```bash
# Box A (8 GPUs): 5 cells in parallel.
exp01 pilot --parallel \
    --cells reve.linear.eeg.0,reve.qformer.eeg.0,reve.vocab.eeg.0,tfm.linear.eeg.0,tfm.qformer.eeg.0 \
    --stage1-steps 100 --stage2-steps 400 --stage3-steps 200 --batch-size 1

# Box B (1 GPU): the remaining cell.
exp01 pilot --cells tfm.vocab.eeg.0 \
    --stage1-steps 100 --stage2-steps 400 --stage3-steps 200 --batch-size 1
```

Both boxes log to the same W&B project (`exp01-eeg-to-text`); cell IDs are
unique so the runs interleave naturally. `metrics.json` and
`predictions.parquet` for each cell live under each box's
`$EXP01_DATA_ROOT/eval/<cell_id>/` — collect them with `scp` or `rsync`.

## What lands where (per cell)

```
$EXP01_DATA_ROOT/runs/<cell_id>/
  log.jsonl              per-step loss, grad_norm, lr, stage; events
  sample_gens.jsonl      periodic dev (ref, hyp) snapshots during training
  stats.jsonl            encoder feature stats (mean, std, abs_max) per stage
  model_stage1.pt        checkpoint after Stage 1 (alignment)
  model_stage2.pt        checkpoint after Stage 2 (frozen-LM SFT)
  model.pt               final checkpoint (after Stage 3 if LoRA enabled)

$EXP01_DATA_ROOT/eval/<cell_id>/
  metrics.json           summary mean / 95% CI for all 6 metrics
  predictions.parquet    one row per test example with full metadata:
                         subject_id, dataset, ref, hyp,
                         bleu1..4, rouge1_f, bertscore_f1,
                         eeg_channels, eeg_time, sr
```

## W&B

Each cell becomes one W&B run named `<cell_id>`, grouped by `<encoder>_<bridge>`,
tagged with encoder / bridge / input / fold. Logged:

- per-step: `stage{1,2,3}/loss`, `grad_norm`, `lr`
- per-stage: `feat_mean`, `feat_std`, `feat_abs_max` (catches dead encoders / NaNs)
- per-stage: 16-example sample-generation tables (`samples/<stage>_step<N>`)
- eval: `eval/{bleu1..4, rouge1_f, bertscore_f1}` + CI bounds
- eval: 500-row predictions Table

If `WANDB_API_KEY` is unset, training runs without W&B logging (jsonl files
under `runs/<cell_id>/` still capture everything).

## Hyperparameters that often need tuning

- `CellConfig.batch_size` × `grad_accum` — peaks at ~50 GB on H100 80 GB with
  defaults (8 × 4); halve `batch_size` for 40 GB cards.
- `stage1_steps / stage2_steps / stage3_steps` — defaults are 2 k / 6 k / 4 k;
  the smoke target is 20 / 20 / 10.
- `qformer_queries` (default 32) — tradeoff between context length and
  bridge capacity for the Q-Former cells.
- `rvq_codebook` (default 8192) — only relevant for off-diagonal vocab cells
  (REVE × vocab and DIVER-1 × vocab).

## Diagnostic commands

```bash
exp01 inspect-channels      # channel names per source (sanity-check encoder normalisation)
ls $EXP01_DATA_ROOT/runs/   # cells that have at least one stage checkpoint
tail -f $EXP01_DATA_ROOT/runs/<cell_id>/log.jsonl  # live per-step training log
python -c "import pyarrow.parquet as pq; pq.read_table('$EXP01_DATA_ROOT/eval/<cell_id>/predictions.parquet').to_pandas().head()"
```
