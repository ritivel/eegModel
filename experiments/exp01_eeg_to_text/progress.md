# Pilot progress — Apr 30, 2026

Live record of all training runs, where they are, and when they're expected to
finish. All times in **IST**.

> **Document map** (so the right thing is easy to find):
>
> | doc | what it has |
> | --- | ----------- |
> | [`results.md`](./results.md) | **V1 pilot** results (Apr-30 morning, 9 cells, post-bug-fix). The negative §4.3 baseline. |
> | [`diagnostic_report.md`](./diagnostic_report.md) | Root-cause analysis of the four V1 bugs, with code-fix snippets and citations. |
> | [`next_experiments.md`](./next_experiments.md) | The full 24-h plan after the V1 pilot finished. Tracks A / B / C / D. |
> | [`results_track_a_v2.md`](./results_track_a_v2.md) | **Track A — V2 preprocessing pilot.** Live (this evening). 9 cells. |
> | [`results_track_b_ctc.md`](./results_track_b_ctc.md) | **Track B — CTC (ASR-style direct EEG → char).** Live (rolling fan-out as Track A frees GPUs). |

> **Day-2 pivot (~14:00 IST → 19:53 IST):** the morning V1 pilot completed
> with three bugs (later four) which were fixed and re-run; the post-fix V1
> matched-pair test gave an unambiguously negative §4.3 result (noise BLEU-1
> 0.136 > EEG 0.114, sign-flip *p* < 1e-4). RCA in `diagnostic_report.md`,
> results in `results.md`. The diagnosis pointed at preprocessing — REVE
> and TFM were trained on bandpass + notch + per-recording z-score data,
> we were feeding raw uV. **Track A** (V2 preprocessing) re-runs the same
> cells with the encoder-correct preprocessing pipeline; **Track B** (CTC)
> is the user-suggested architectural pivot that removes the LM from the
> loss entirely (the LM-prior trap was the dominant V1 failure mode).

---

## Compute

| Box | Host                   | GPUs         | Notes                      |
| --- | ---------------------- | ------------ | -------------------------- |
| A   | `ubuntu@192.222.53.60` | 8× H100 80GB | bulk pilot                 |
| B   | `ubuntu@192.222.53.81` | 1× H100 80GB | matched §4.3 noise twin    |

SSH: `ssh -i ~/Downloads/modal_biosigtotext ubuntu@<host>`
W&B: [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text)

---

## Track A — V2 preprocessing (running)

Launched **19:53 IST**. Step budget per cell: `300 (alignment) + 1200 (frozen-LM SFT) + 500 (LoRA SFT)` = 2 000 steps.

| Box | GPU | Cell                                  | Started | Stage @ snapshot 20:46 | Throughput   | ETA finish | W&B run                                                                                              |
| --- | --- | ------------------------------------- | ------- | ---------------------- | ------------ | ---------- | ---------------------------------------------------------------------------------------------------- |
| A   | 0   | `reve.linear.eeg.0` v2                | 19:53   | stage 2, step 1060     | ~1.9 s/step  | ~21:30     | [g???](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text)                                       |
| A   | 1   | `reve.qformer.eeg.0` v2               | 19:53   | stage 2, step 1040     | ~1.9 s/step  | ~21:35     | (best cell — align_loss 1.896 < chance 2.08, lowest LM loss)                                         |
| A   | 2   | `tfm.linear.eeg.0` v2                 | 19:53   | stage 2, step 1060     | ~1.9 s/step  | ~21:30     | (broken bridge — generates `'**'` / `'event-time-…'`)                                                |
| A   | 3   | `tfm.qformer.eeg.0` v2                | 19:53   | stage 2, step 1080     | ~1.9 s/step  | ~21:30     | (LM-prior English biographies)                                                                       |
| A   | 4   | `reve.vocab.eeg.0` v2                 | 19:53   | **stage 3 done — eval running** | (slow, bs=1) | ~21:00     | (1/16 unique — `'HeHeHe…'` collapse)                                                                |
| A   | 5   | `tfm.vocab.eeg.0` v2                  | 19:53   | **stage 3 done — eval running** | (slow, bs=1) | ~21:00     | (1/16 unique — `'HeHeHe…'` collapse)                                                                |
| A   | 6   | `reve.linear.eeg.1` v2 (fold-1)       | 19:53   | stage 2, step 1020     | ~1.9 s/step  | ~21:35     | (align_loss 1.989 < chance — confirms reve cells)                                                    |
| A   | 7   | `tfm.linear.eeg.1` v2 (fold-1)        | 19:53   | stage 2, step 1030     | ~1.9 s/step  | ~21:35     | (broken — Korean filler chars)                                                                       |
| **B** | **0** | `reve.linear.noise_train.0` v2  | 19:53   | **DONE — eval done — BLEU-1=0.126** | ~1.4 s/step | **20:11**  | (matched §4.3 baseline; floor that EEG cells must beat)                                              |

**See [`results_track_a_v2.md`](./results_track_a_v2.md) for the in-flight quantitative + qualitative findings.**

The big result: V2 unlocks a real but partial signal in stage 1 (REVE
cells go sub-chance on align_loss for the first time), but the LM-prior
trap still dominates greedy generation — soft-prompt cells produce the
same "Florida congressman" biographies as the noise twin. Vocab cells
are still fully collapsed (1/16 unique).

---

## Track B — CTC (rolling fan-out)

Launched **20:58 IST** (lead cell, Box B). Other cells fan out to Box A
GPUs as Track-A cells finish. Same step budget as Track A.

CTC removes Gemma from the loss entirely → no LM prior to escape. CER
on the noise twin should be ≈ 1.0 (the head has no language prior to
fall back on); CER on EEG cells, if non-trivial, is structurally
attributable to EEG content.

| Box | GPU | Cell                                | When | Status                            | W&B run                                                                                                  |
| --- | --- | ----------------------------------- | ---- | --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **B** | **0** | `reve.ctc.eeg.0` v2 (bs=16, ga=2) | 20:58 | **running** (lead cell + smoke)   | [e1qp3pze](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/e1qp3pze)                         |
| A   | 4   | `reve.ctc.noise_train.0` v2         | ~21:05 | queued (after vocab eval)        | (matched §4.3 pair for `reve.ctc.eeg.0`)                                                                 |
| A   | 5   | `tfm.ctc.eeg.0` v2                  | ~21:05 | queued (after vocab eval)        |                                                                                                          |
| A   | 0   | `tfm.ctc.noise_train.0` v2          | ~21:35 | queued (after `reve.linear.eeg.0`) | (matched §4.3 pair for `tfm.ctc.eeg.0`)                                                                |
| A   | 1   | `reve.ctc.eeg.1` v2                 | ~21:35 | queued (after `reve.qformer.eeg.0`) | (fold-1 robustness)                                                                                   |
| A   | 2   | `tfm.ctc.eeg.1` v2                  | ~21:35 | queued (after `tfm.linear.eeg.0`)  | (fold-1 robustness)                                                                                   |
| A   | 6   | `reve.ctc.eeg.0` v2 bs=32           | ~21:35 | queued (after `reve.linear.eeg.1`) | (large-batch ablation)                                                                                |
| A   | 7   | `tfm.ctc.eeg.0` v2 bs=32            | ~21:35 | queued (after `tfm.linear.eeg.1`)  | (large-batch ablation)                                                                                |

**See [`results_track_b_ctc.md`](./results_track_b_ctc.md) for the design,
diagnostic series, and pre-registered §4.3 decision rule.**

---

## Wall-time summary

| time IST | event                                                                |
| -------- | -------------------------------------------------------------------- |
| 19:53    | Track A launched (8 cells Box A + 1 cell Box B)                      |
| 20:11    | Box B noise twin **DONE** → BLEU-1=0.126 [0.119, 0.134] (§4.3 floor) |
| 20:58    | Track B lead cell `reve.ctc.eeg.0` launched on Box B                 |
| ~21:00   | `reve.vocab.eeg.0` + `tfm.vocab.eeg.0` evals finish → Box A GPUs 4,5 free → 2 more CTC cells launched |
| ~21:30   | Box A soft-prompt cells (GPUs 0-3, 6-7) finish → 5 more CTC cells launched |
| ~22:00   | Box B `reve.ctc.eeg.0` finishes → matched-pair §4.3 gap (CTC) computable |
| ~22:30   | All Track B cells finish → full §4.3 matrix on CTC cells              |
| ~23:00   | Final results writeup: `results_track_a_v2.md` + `results_track_b_ctc.md` updated with final numbers |

---

## Pre-fix V1 run, archived (do NOT use these for §4.3 conclusions)

Archived under `$EXP01_DATA_ROOT/archive/buggy_run_2026-04-30T08-30/` on
each box. See `results.md` and `diagnostic_report.md`.

---

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
  metrics.json           summary mean / 95% CI for all 8 metrics (BLEU 1-4, ROUGE-1-F,
                         BERTScore-F1, CER, WER)
  predictions.parquet    one row per test example with full metadata

$EXP01_DATA_ROOT/archive/buggy_run_2026-04-30T08-30/
  runs/                  all pre-fix run dirs
  eval/                  pre-fix eval results
```

`<cell_id>` is e.g. `reve_qformer_eeg_fold0_pp-v2_dec-gemma4-e2b` (Track A)
or `reve_ctc_eeg_fold0_pp-v2_dec-gemma4-e2b` (Track B).
`$EXP01_DATA_ROOT` = `/home/ubuntu/data/exp01` on both boxes.

---

## Quick monitoring commands

```bash
# Live training loss + new diagnostic series for a single cell
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "tail -f /home/ubuntu/data/exp01/runs/reve_qformer_eeg_fold0_pp-v2_dec-gemma4-e2b/log.jsonl"

# Live GPU utilization
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "watch -n2 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"

# Pull a finished cell's metrics + first 3 predictions
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "cd ~/work/eegModel/experiments/exp01_eeg_to_text && \
   .venv/bin/python -c \"
import json, pyarrow.parquet as pq
m = json.load(open('/home/ubuntu/data/exp01/eval/reve_qformer_eeg_fold0_pp-v2_dec-gemma4-e2b/metrics.json'))
print({k: v['mean'] for k, v in m['scores'].items()})
for r in pq.read_table('/home/ubuntu/data/exp01/eval/reve_qformer_eeg_fold0_pp-v2_dec-gemma4-e2b/predictions.parquet').to_pylist()[:3]:
    print('REF:', repr(r['ref'][:120]))
    print('HYP:', repr(r['hyp'][:120]))\""

# Compute matched §4.3 gap once both EEG and noise cells finish
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "cd ~/work/eegModel/experiments/exp01_eeg_to_text && \
   .venv/bin/python -c \"
from exp01 import eval as ev, storage; import json
def load(c): return json.load(open(storage.EVAL/c/'metrics.json'))
eeg   = load('reve_qformer_eeg_fold0_pp-v2_dec-gemma4-e2b')
noise = load('reve_linear_noise_train_fold0_pp-v2_dec-gemma4-e2b')
print(ev.eeg_noise_gap(eeg, noise))\""
```
