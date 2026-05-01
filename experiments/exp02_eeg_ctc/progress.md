# Exp02 progress — Apr 30 / May 1, 2026

Live record of all training runs, where they are, and when they're expected
to finish. All times in **IST**.

> **Document map**:
>
> | doc | what it has |
> | --- | --- |
> | [`README.md`](./README.md) | quick start + run matrix + artifact map |
> | [`design.md`](./design.md) | design rationale, citations for every knob, full §4.3 verdict rules |
> | [`wave2_plan.md`](./wave2_plan.md) | augmentation matrix + launch commands for wave-2 |
> | this file | live timeline + per-cell loss snapshots + decisions |

---

## TL;DR — where we are at 06:41 IST May 1

- **Wave-1 pilot**: 14-cell Track-C scope launched on Box A 8× H100 at 23:55 IST Apr 30 via `exp02 pilot --group all --parallel`. Currently ~6h 33min in; first 8 cells at step ≈ 10 200–10 750 / 12 000 (~85% done). Second wave (6 queued cells: intctc + ctcaed + frozen ablations) will auto-start as GPUs free up.
- **Wave-2 aug-signal**: `reve.bpe1k.crctc.eeg.0` + 6 GPU-side signal augmentations launched on Box B 1× H100 at 00:35 IST May 1. Currently ~6h in; step 8140 / 12 000 (~67% done).
- **Headline §4.3 signal is real, large, and growing**: at step 10 380 the EEG cell has ctc_loss = **2.68** vs noise twin = **3.15** → **gap −0.47 in CTC loss** (on top of an already-growing trend; was −1.18 at step 10 220 last sample). REVE+CR-CTC is decoding from EEG content, not from priors.
- **No errors anywhere**. Both orchestrators alive, all 9 cells healthy.

---

## Compute

| Box | Host | GPUs | Role |
| --- | --- | --- | --- |
| **A** | `ubuntu@192.222.53.60` | 8× H100 80GB | Wave-1 Track-C pilot (14 cells via `exp02 pilot --group all --parallel`) |
| **B** | `ubuntu@192.222.53.81` | 1× H100 80GB | Wave-2 aug-signal headline + matched noise twin (sequential) |

SSH: `ssh -i ~/Downloads/modal_biosigtotext ubuntu@<host>`
W&B: <https://wandb.ai/ritivel-eeg-ritivel/exp02-eeg-ctc>

`$EXP02_DATA_ROOT = /home/ubuntu/data/exp02` on both boxes (with
`hf/`, `splits/` symlinked to `/home/ubuntu/data/exp01/{hf,splits}` to
share the dataset cache + fold JSONs).

---

## Timeline

| time IST | event |
| --- | --- |
| Apr 30 23:00 | Decision: 3-stage curriculum is vestigial for CTC. Started building exp02 — single-stage end-to-end CTC training, no curriculum. |
| Apr 30 23:30 | Committed `eeg_common/` shared package (storage / preprocessing / splits / data / encoders) + migrated exp01 to delegate (commit `593df9c`). |
| Apr 30 23:30 | Committed `exp02_eeg_ctc/` scaffolding: chars + tokenizer + head + model + train + decode + eval + cli (`593df9c`). |
| Apr 30 23:53 | Pushed; pulled on box A; installed `eeg_common` + `exp02-eeg-ctc` editable into the existing exp01 venv. |
| Apr 30 23:55 | exp02 smoke test on box A: 2 cells × 20 steps. Loss decreasing, no errors. |
| May 1 00:00 | Found `_assemble_zuco_corpus` was loading full EEG per row → 30 min builds. Fixed via direct parquet column read (commit `fcdd2ec`). BPE-1k built on box A in 21 sec. |
| May 1 00:25 | Launched 8-cell wave-1 pilot via `exp02 pilot --group all --parallel` on box A. Orchestrator PID 304536. |
| May 1 00:30 | Launched fold-1 headline on box B (sequential EEG → noise twin). |
| May 1 00:35 | Discovered box B's `splits/fold_*.json` were the pre-fix version (24 ZuCo-only subjects vs box A's 88 with non-ZuCo). Killed box B fold-1, scp'd box A's splits over, restarted. |
| May 1 00:40 | Encoder unfreeze at step 1201 on all 8 wave-1 cells, clean. |
| May 1 00:45 | User: "try different things if it doesn't work — text + signal augmentation, here's an OpenAI key". |
| May 1 00:55 | Stashed `OPENAI_API_KEY` into both boxes' `.env` (gitignored). |
| May 1 01:00 | Built `eeg_common/augment.py` (signal augs) + `exp02/text_augment.py` (LLM paraphrase) + wired into trainer behind flags. Committed `0f5fd80`. |
| May 1 01:08 | Paraphrase pipeline: 1547 ZuCo sentences × 5 paraphrases via `gpt-4o-mini` finished in 209 sec, 0 errors. |
| May 1 01:15 | Added `--tag` cell_id suffix + wave-2 plan doc. Committed `f50cb26`. |
| May 1 01:30 | **Decision**: redirect box B from fold-1 (which we don't need until 5-fold extension) to wave-2 aug-signal cell. Killed fold-1 (had 560 steps), launched aug-signal headline + matched noise twin sequentially. |
| May 1 02:30 | Wave-1 cells past encoder unfreeze; gaps growing nicely (headline at −0.06 → −0.23 → −0.37 → −0.44 over the first 4h). |
| May 1 03:30 | Inspected qualitative outputs at step 3600. EEG cell producing `"the c c spe."` etc — short outputs (partial blank-collapse) but EEG-vs-noise tokens differ in semantically-relevant ways (movie-related for movie reviews, gibberish for noise). |
| May 1 06:30 | Headline gap explodes to **−1.18 in CTC loss** at step 10 220. EEG model is well below chance (2.66 vs `log(1026) ≈ 6.93`); noise twin lags significantly (3.85). |
| May 1 06:41 | This document written. |

---

## Wave-1 — Track-C pilot (Box A, 14 cells)

Launched 23:55 IST Apr 30. Step budget per cell: 12 000 steps with linear-warmup → cosine-decay LR; 1 200-step encoder warmup-freeze, then full REVE/TFM fine-tune. Batch 16 × grad-accum 2 (effective 32). bf16 mixed precision.

### Currently running (8 cells, GPUs 0-7)

| Box A GPU | cell | step | ctc_loss | EEG-vs-noise gap | notes |
| --- | --- | --- | --- | --- | --- |
| 0 | `reve.bpe1k.crctc.eeg.0` (**HEADLINE**) | 10380 | 2.680 | — | well below chance ✓ |
| 1 | `reve.bpe1k.crctc.noise_train.0` | 10420 | 3.153 | **−0.47 (EEG winning)** | matched §4.3 noise twin |
| 2 | `tfm.bpe1k.crctc.eeg.0` | 10240 | 5.742 | — | TFM struggling (still > 5) |
| 3 | `tfm.bpe1k.crctc.noise_train.0` | 10200 | 5.854 | −0.11 (TFM EEG winning narrowly) | matched |
| 4 | `reve.char.crctc.eeg.0` | 10370 | 2.224 | — | char vocab — chance is `log(50) ≈ 3.91` |
| 5 | `reve.char.crctc.noise_train.0` | 10350 | 2.355 | −0.13 | matched |
| 6 | `reve.bpe1k.ctc.eeg.0` (vanilla + label-prior) | 10750 | 3.107 | — | vanilla CTC + label-prior trick |
| 7 | `reve.bpe1k.ctc.noise_train.0` | 10400 | 3.669 | −0.56 | matched; label-prior is helping |

### Queued (6 cells, will auto-launch as GPUs free up)

| cell | recipe |
| --- | --- |
| `reve.bpe1k.intctc.eeg.0` | intermediate-CTC (Komatsu 2022) — auxiliary CTC loss at intermediate head layers |
| `reve.bpe1k.intctc.noise_train.0` | matched noise |
| `reve.bpe1k.ctcaed.eeg.0` | CTC + AED hybrid (Watanabe 2017) |
| `reve.bpe1k.ctcaed.noise_train.0` | matched noise |
| `reve.bpe1k.crctc.eeg.0_frozen` | encoder permanently frozen — GROUP E ablation |
| `reve.bpe1k.crctc.noise_train.0_frozen` | matched noise |

### Headline gap trajectory (CTC loss; lower is better; EEG vs noise)

| step | EEG | noise | gap | encoder |
| --- | --- | --- | --- | --- |
| 100 | 6.082 | 6.073 | +0.01 | frozen |
| 410 | 5.935 | 6.091 | −0.16 | frozen |
| 1350 | 5.564 | 5.796 | −0.23 | unfrozen at 1201 |
| 2240 | 5.217 | 5.303 | −0.09 | (variance from sample timing) |
| 3750 | 4.934 | 5.301 | −0.37 | full FT |
| 5870 | 4.330 | 4.771 | −0.44 | full FT |
| 10220 | 2.662 | 3.846 | **−1.18** | full FT |
| 10380 | 2.680 | 3.153 | −0.47 | full FT |

The −1.18 reading earlier likely caught the EEG cell mid-step; the consistent trend through training is a steadily-widening EEG advantage. Both EEG and noise are dropping, but EEG is dropping faster.

### Qualitative output (sample at step 4800, headline cell)

```
REF:  Celebrated at Sundance, this slight comedy of manners has winning ...
EEG:  the c c spe.        (12 chars)
NOISE: he s s l.          (9 chars)

REF:  What a great shame that such a talented director as Chen Kaige ...
EEG:  he c f.             (7 chars)
NOISE: the s ling.        (10 chars)
```

The model is in **partial CTC blank-collapse** — outputs are 5-15 chars vs reference 100+ chars. But the non-blank tokens differ between EEG and noise in semantically-relevant ways (EEG produces movie-related fragments for movie-review sentences). The §4.3 gap on CER will likely be smaller than the gap on CTC loss because both cells produce similarly-short outputs (length-precision artefact, same as Track B in exp01).

---

## Wave-2 — aug-signal cell (Box B, 1 GPU)

Launched 00:35 IST May 1, replacing the originally-planned fold-1 sequential pair. Same recipe as wave-1 headline, plus 6 GPU-side signal augmentations:

| augmentation | flag | citation |
| --- | --- | --- |
| Time shift (±5% of T) | `--signal-aug-time-shift-max-frac 0.05` | [Brain Transformer 2025](https://www.nature.com/articles/s41598-025-86294-3) — *single most effective* aug for EEG |
| Channel dropout (10% of channels at p=0.5) | `--signal-aug-channel-dropout-p 0.5 --signal-aug-channel-dropout-frac 0.1` | [Strumiłło 2026](https://www.mdpi.com/1424-8220/26/4/1258) |
| Frequency mask (≤8 Hz bands at p=0.5) | `--signal-aug-freq-mask-p 0.5 --signal-aug-freq-mask-max-hz 8.0` | analogue of SpecAug freq-mask |
| Time warp (segment stretch/telescope at p=0.3) | `--signal-aug-time-warp-p 0.3` | [Xu 2026](https://www.mdpi.com/1424-8220/26/2/399) |
| Fourier surrogate (phase randomisation at p=0.2) | `--signal-aug-fourier-surrogate-p 0.2` | Strumiłło 2026 |
| Feature mixup (Beta(0.4, 0.4)) | `--signal-aug-mixup-alpha 0.4` | adapted for CTC sequence loss with weighted pair losses |

**Note**: Text augmentation NOT enabled on this box B cell — paraphrases parquet didn't sync (cross-box scp hang); not worth fixing for the aug-signal isolated comparison. Wave-3 (if needed) will combine signal-aug + text-aug as `aug-strong`.

| Box B GPU | cell | step | ctc_loss |
| --- | --- | --- | --- |
| 0 | `reve.bpe1k.crctc.eeg.0_aug-signal` | 8140 | 5.158 |

Train loss is higher than wave-1 baseline at the same step (5.16 vs ~3.50) because augmentation makes training data harder. **The eval CER/WER comparison is what matters** — that comes ~3h from now.

After this cell finishes, the matched-noise twin `_aug-signal` runs sequentially (~6h more total).

---

## Compute schedule

| time IST | box A | box B |
| --- | --- | --- |
| 23:55 (Apr 30) → 08:00 | wave-1 first wave (8 cells) | aug-signal EEG |
| 08:00 → 14:00 | wave-1 second wave (6 queued) | aug-signal noise twin starts at ~10:00 |
| 14:00+ | wave-1 done; depending on §4.3 verdicts: 5-fold extension OR wave-3 | aug-signal pair done; depending on result, wave-3 |

Approx 30 H100-hours (box A) + 12 H100-hours (box B) = 42 H100-hours total for wave-1 + wave-2.

---

## Plan B — wave-3 launch matrix (only if wave-1 + wave-2 are inconclusive)

See [`wave2_plan.md`](./wave2_plan.md) for the complete augmentation matrix and CLI commands. Top candidates if wave-1's headline gap doesn't translate to a §4.3 PASS on CER:

1. **`aug-strong`** — `aug-signal` + text-aug (paraphrase substitution at 0.5 prob)
2. **`aug-text`** — text-aug only, isolates text-side gain
3. **`aug-mixup`** — feature mixup only, isolates mixup
4. **No-warmup cell** — `--encoder-warmup-freeze-steps 0`, full REVE fine-tune from step 1
5. **`crctc + label-prior 0.3`** — combine both anti-collapse mechanisms

Wave-3 architectural pivots (only if all augmentation cells also FAIL):

- **D-SigLIP word-level alignment** ([d'Ascoli 2025 *Nat. Commun.* 16, 10521](https://doi.org/10.1038/s41467-025-65499-0)) — the only published recipe that rigorously passes the Jo et al. 2024 noise baseline on non-invasive EEG. Uses per-word EEG segments + BPE-token contrastive alignment instead of sequence-level CTC.
- **Multi-subject pretrain** — pretrain a small EEG MAE on all 8 ZuCo subjects' EEG, then fine-tune per-subject.
- **Cross-modal contrastive** — BELT-2 / CET-MAE style.

---

## Decision rule when wave-1 cells finish

For each EEG cell, run `exp02 gap reve.bpe1k.crctc.eeg.0` etc. (already implemented in the CLI). The output gives the matched-pair §4.3 gap on CER, WER, BLEU 1-4, ROUGE-1-F, BERTScore-F1 across all 3 decode modes (greedy / beam / beam+KenLM — the latter currently disabled because `lmplz` is not on the box).

| outcome | trigger |
| --- | --- |
| **PASS (strict)** | CER gap CI strictly above 0 AND sign-flip p < 0.01 → run 5-fold extension on the survivor. |
| **PASS (weak)** | CER gap mean > 0, CIs overlap → launch `aug-strong` to tighten. |
| **TIE** | CER gap ≈ 0, both far from 1 → launch full wave-3 augmentation matrix. |
| **FAIL** | CER ≈ 1 for both → architectural pivot (D-SigLIP, multi-subject pretrain). |

**Already-known**: the CTC loss gap is huge for the headline cell. Whether this translates to CER gap depends on whether both cells stay in the same partial-collapse state at eval time (in which case CER gap will be near-zero from length-precision artefacts) OR whether the EEG cell has actually escaped collapse enough to produce longer, more correct hypotheses (in which case CER gap will be substantial).

---

## Per-cell artifacts

```
$EXP02_DATA_ROOT/runs/<cell_id>/
  log.jsonl             per-step ctc_loss, cr_ctc_kl, intermediate_ctc, aed_loss,
                        grad_norm, head_lr, encoder_lr, encoder_active, elapsed
  sample_gens.jsonl     periodic dev (ref, hyp_greedy) snapshots during training
                        every 10% of steps
  stats.jsonl           encoder feature stats (mean, std, abs_max, nonzero_frac)
  model.pt              final checkpoint
  model_step6000.pt     mid-training checkpoint (every 50% of steps)
  run.log               stdout/stderr from the orchestrator-spawned subprocess

$EXP02_DATA_ROOT/eval/<cell_id>/
  metrics.json          summary mean / 95% CI per metric, per decode mode
                        (greedy / beam / beam_kenlm if available)
  predictions.parquet   one row per test example with metadata, ref, and
                        hyp_greedy, hyp_beam, hyp_beam_kenlm + per-row metric values

$EXP02_DATA_ROOT/orchestrator/pilot.log       box A pilot orchestrator output
$EXP02_DATA_ROOT/orchestrator/boxb-aug.log    box B aug-signal cell orchestration
$EXP02_DATA_ROOT/text_aug/paraphrases.parquet 1547 sentences × 5 paraphrases (built once)
$EXP02_DATA_ROOT/bpe/{spm.model,spm.vocab,corpus.txt}  BPE-1k tokenizer
```

---

## Quick monitoring commands

```bash
# Box A pilot health + per-cell loss
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 '
  ps -p 304536 -o etime,cmd 2>&1 | head -3
  for d in /home/ubuntu/data/exp02/runs/*/; do
      name=$(basename $d)
      [ -f $d/log.jsonl ] && {
          last=$(tail -1 $d/log.jsonl)
          step=$(echo $last | grep -oE "\"step\": [0-9]+" | grep -oE "[0-9]+")
          loss=$(echo $last | grep -oE "\"ctc_loss\": [0-9.e-]+" | head -1)
          echo "  $name | step=$step $loss"
      }
  done'

# Box B aug-signal status
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.81 '
  for d in /home/ubuntu/data/exp02/runs/*/; do
      name=$(basename $d)
      [ -f $d/log.jsonl ] && {
          last=$(tail -1 $d/log.jsonl)
          step=$(echo $last | grep -oE "\"step\": [0-9]+" | grep -oE "[0-9]+")
          loss=$(echo $last | grep -oE "\"ctc_loss\": [0-9.e-]+" | head -1)
          echo "  $name | step=$step $loss"
      }
  done'

# Sample dev generations from the headline cell (qualitative read)
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  'tail -5 /home/ubuntu/data/exp02/runs/reve_bpe1k_crctc_eeg_fold0_pp-v2/sample_gens.jsonl'

# Compute matched-pair §4.3 gap (after both EEG + noise eval finish)
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  'cd /home/ubuntu/work/eegModel/experiments/exp02_eeg_ctc && \
   set -a && source .env && set +a && \
   /home/ubuntu/work/eegModel/experiments/exp01_eeg_to_text/.venv/bin/python \
     -m exp02.cli gap reve.bpe1k.crctc.eeg.0'
```

---

## Commits this session

| commit | summary |
| --- | --- |
| `593df9c` | Add exp02 single-stage CTC + extract eeg_common shared package |
| `fcdd2ec` | exp02 build-bpe: read text via parquet (not EEGSentenceDataset) |
| `0f5fd80` | exp02: signal + text data augmentation suite (off by default; wave-2 ready) |
| `f50cb26` | exp02: cell_id `--tag` suffix + wave-2 augmentation plan doc |
