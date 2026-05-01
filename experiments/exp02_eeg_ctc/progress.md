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
> | [`findings.md`](./findings.md) | **wave-1 audit + wave-3 plan (May 1, 08:00 IST)** |
> | this file | live timeline + per-cell loss snapshots + decisions |

---

## TL;DR — where we are at 09:00 IST May 1

- **Wave-1 pilot completed at step 12000** on Box A. Final losses are exactly what the audit predicted: REVE+CR-CTC headline ctc=2.91 (eeg) vs 3.68 (noise) → **gap −0.77 in CTC loss**, but the qualitative dev outputs are **30-character non-content gibberish** for both EEG and noise (`'lious about t war ged g im hull hle.'` vs `'the moved sc ling stiousovle.'`). CER ≈ 1 expected for both, §4.3 verdict will be TIE.
- **Wave-1 root causes** identified (see [`findings.md`](./findings.md) for the full audit): TFM encoder was permanently frozen (bug), no English priors in the head (CTC over a randomly-init transformer), only **1 237 unique training sentences** (sequence-CTC over a 1.2K-sentence vocabulary), substantial data quality issues (DERCo 95% truncated, EMMT 96% zero-padded across channels, NaN rows, multi-paragraph monsters).
- **Wave-3 launched at 08:30 IST May 1** with all 4 root causes addressed: cleaned data (drop EMMT/DERCo, drop garbage labels, drop > 12 s rows, drop NaN/zero), GPT-4o-mini paraphrase substitution at p=0.5 over the 1 547 ZuCo sentences (× 5 paraphrases each), 6 GPU-side signal augmentations (time shift / channel dropout / freq mask / time warp / fourier surrogate / mixup), and **DistilBERT bridge head** (`LMBridgeHead`) replacing the random-init 4-layer transformer with the 6-layer pretrained DistilBERT encoder.
- **All 9 GPUs running wave-3**: Box A has 8 cells (4 transformer-only controls + 4 lm-bridge cells across {clean, lm-clean, lm-aug, aug-clean}), Box B running 2 cells sequentially (`char-lm-aug` pair). ETA ~5h on Box A, ~10h on Box B.

---

## Compute

| Box | Host | GPUs | Role (current) | Role (wave-1) |
| --- | --- | --- | --- | --- |
| **A** | `ubuntu@192.222.53.60` | 8× H100 80GB | Wave-3 cells 0–7 (`pilot --group wave3 --parallel --cells :8`) | Wave-1 14-cell Track-C pilot (completed) |
| **B** | `ubuntu@192.222.53.81` | 1× H100 80GB | Wave-3 cells 8–9 sequential (`pilot --group wave3 --parallel --cells 8:`) | Wave-2 aug-signal headline (killed at step 9 970, wave-3 supersedes) |

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
| May 1 07:30 | **Audit triggered** by suspicion that wave-1 results don't match the qualitative outputs. Discovers (1) TFM encoder permanently frozen due to `@torch.no_grad()` + `.eval()` in `_tokenize_no_grad`; (2) only 1 237 unique training sentences for a 1024-vocab BPE-CTC task; (3) DERCo 95% truncated, EMMT 96% zero-padded, 8 garbage-text rows; (4) the random-init transformer head has no English priors. Full audit landed in [`findings.md`](./findings.md). |
| May 1 08:00 | **Wave-3 plan committed** (`94475dd`): TFMEncoder fix, `EEGSentenceDataset` quality filters (`drop_sources / min_text_chars / max_text_chars / max_seconds / drop_nan_rows / drop_zero_rows`), `LMBridgeHead` (DistilBERT 6-layer, 42.5 M params bridge), `wave3_cells()` matrix in `config.py`, `--group wave3` in CLI. TFM removed from `all_track_c_cells()`. |
| May 1 08:15 | **Wave-1 + wave-2 killed** on both boxes. ~30 min of training lost on each cell, but step-12000 logs / sample_gens / stats preserved for analysis; checkpoints at step 6000 (50% mark) survive. |
| May 1 08:20 | DistilBERT pre-downloaded into HF cache on both boxes; `paraphrases.parquet` (1 547 sentences × 5 paraphrases, 827 KB) scp'd from Box A → Box B. |
| May 1 08:30 | **Wave-3 launched** (`pilot --group wave3 --parallel`): 8 cells on Box A (`--cells :8`), 2 sequential cells on Box B (`--cells 8:`). All 4 lm-bridge cells crashed in the first 4 minutes due to a transformers-API mismatch (`Transformer.forward(x=...)` was renamed to `(hidden_states, ...)` in DistilBERT 4.40+). |
| May 1 08:45 | Patched (`edd62b9`): `LMBridgeHead._bridge_forward` now iterates over `self.transformer.layer` directly instead of calling the wrapper's `forward`, sidestepping the API churn. Verified with a (B=2, T=100, V=1026) smoke test on Box A GPU 2. |
| May 1 08:55 | Re-launched the 4 lm-bridge cells (Box A GPUs 2–5) and the 2 char-lm-aug cells (Box B). All 9 GPUs now active; 8 wave-3 cells running on Box A, 1 on Box B (1 queued). |
| May 1 09:00 | First reasonable losses observed: transformer-only `clean` cells at step 430, ctc≈5.94 (still warming up); lm-bridge cells at step 150, ctc≈6.21 (starting from random head + pretrained DistilBERT priors); char-lm-aug at step 140, ctc=3.53 (char vocab `log(50)≈3.91` baseline). All cells healthy, encoder still frozen (warmup-freeze active until step 1200). |
| May 1 12:10 | **Wave-3 mid-run audit (3 h 40 min in)**. Transformer-only cells healthy and converging: `clean.eeg` ctc=2.67 vs noise=3.43 (gap −0.76 at step 9.4k), matching the wave-1 baseline gap. **All 5 lm-bridge cells in complete blank-collapse**: dev outputs are empty strings, ctc stuck at 6.0+ (≈ `log(vocab) = 6.93`), gradient norms anomalously low (0.3–1.5). Diagnosis: `head_lr = 1e-3` was applied to **every** head parameter, including the 42.5 M pretrained DistilBERT layers. Standard BERT/DistilBERT fine-tune is 1e-5 to 5e-5; 1e-3 destroys the pretrained weights in the first ~100 steps. |
| May 1 12:30 | Patched (`4562b8f`): added `cfg.bridge_lr = 1e-5` and split the head optimizer into two param groups — `proj` (input projection + output head + AED, ~1.2 M params @ `head_lr = 1e-3`) and `bridge` (the pretrained DistilBERT layers, ~42.5 M params @ `bridge_lr = 1e-5`). LR scheduler now scales each group by its own `_base_lr` so warmup/cosine-decay applies correctly across all three rates (encoder, projection, bridge). |
| May 1 12:35 | Killed the 5 broken lm-bridge cells; transformer-only cells (`clean`, `aug-clean`) left running. Re-launched the 4 lm-bridge cells on Box A GPUs 2–5 and the 2 char-lm-aug cells on Box B with the bridge_lr fix. Confirmed via run.log: `head_opt: proj=1.18M @ lr=0.001; bridge=42.53M @ lr=1e-05`. |
| May 1 12:50 | First step-100 lm-bridge losses were ctc≈50–65 with gradient-norm ≈480 (clipped to 1.0). After 5 minutes (step 320): noise twins back to ctc≈6.1, EEG cells at ctc≈12 and falling. After 10 min (step 580): all 4 lm-bridge cells stable at ctc≈6.0–6.3 (log(1024)=6.93 chance). Char-lm-aug already at ctc=3.19 (below char chance of log(50)=3.91). Encoder still frozen until step 1200. |

---

## Wave-3 — post-audit launch (May 1 08:30 IST, 10 cells across 9 GPUs)

| Box | GPU | cell | features | hypothesis |
| --- | --- | --- | --- | --- |
| A | 0 | `reve.bpe1k.crctc.eeg.0_clean` | clean data only | "Does cleaning the data alone move CER?" |
| A | 1 | `reve.bpe1k.crctc.noise_train.0_clean` | matched twin | — |
| A | 2 | `reve.bpe1k.crctc.eeg.0_h-lm-bridge_lm-clean` | clean data + DistilBERT bridge | "Does the LM bridge alone move CER (no aug)?" |
| A | 3 | `reve.bpe1k.crctc.noise_train.0_h-lm-bridge_lm-clean` | matched twin | — |
| A | 4 | `reve.bpe1k.crctc.eeg.0_h-lm-bridge_lm-aug` | **HEADLINE**: clean + bridge + paraphrase + signal aug | "Full stack — does it pass strict §4.3?" |
| A | 5 | `reve.bpe1k.crctc.noise_train.0_h-lm-bridge_lm-aug` | matched twin | — |
| A | 6 | `reve.bpe1k.crctc.eeg.0_aug-clean` | clean + paraphrase + signal aug, **no LM bridge** | "Does aug alone move CER? (controls for the LM-bridge contribution)" |
| A | 7 | `reve.bpe1k.crctc.noise_train.0_aug-clean` | matched twin | — |
| B | 0 | `reve.char.crctc.eeg.0_h-lm-bridge_char-lm-aug` | char vocab + bridge + paraphrase + signal aug | "Does char vocab beat BPE-1k under the bridge? Per-character CTC may avoid the BPE length-precision artefact." |
| B | 0 (queued) | `reve.char.crctc.noise_train.0_h-lm-bridge_char-lm-aug` | matched twin (sequential after the eeg cell finishes) | — |

Each EEG cell has a matched §4.3 noise twin per Jo et al. 2024. Eval runs
greedy + beam (50) + (when KenLM is available) beam + KenLM 4-gram rescore.
After all cells finish, `exp02 gap <eeg_key>` reports the matched-pair
gap on CER / WER / BLEU 1-4 / ROUGE-1-F / BERTScore-F1.

**ETA**: Box A ~5 h (parallel, ~13:30 IST), Box B ~10 h (~18:30 IST for cell B-1).

---

## Wave-1 — Track-C pilot (Box A, 14 cells) — COMPLETED, results pessimistic

**Final state at step 12 000** (May 1 08:15 IST, just before kill-and-relaunch):

| cell | ctc_loss | gap vs noise | qualitative (sample step 10 800) |
| --- | --- | --- | --- |
| `reve.bpe1k.crctc.eeg.0` (HEADLINE) | 2.91 | **−0.77** | `'lious about t war ged g im hull hle.'` (38 chars vs ref 118) |
| `reve.bpe1k.crctc.noise_train.0` | 3.68 | — | `'the moved sc ling stiousovle.'` (29 chars) |
| `reve.bpe1k.ctc.eeg.0` | 3.16 | **−0.68** | empty string everywhere |
| `reve.bpe1k.ctc.noise_train.0` | 3.84 | — | empty string |
| `reve.char.crctc.eeg.0` | 2.27 | −0.10 | `'e a o o a i te ae a i ei a ee a i...'` (vowel soup) |
| `reve.char.crctc.noise_train.0` | 2.37 | — | similar vowel soup |
| `tfm.bpe1k.crctc.eeg.0` | 5.57 | −0.10 | empty / `.` (TFM-frozen bug; wave-1 cells invalidated) |
| `tfm.bpe1k.crctc.noise_train.0` | 5.67 | — | empty / `.` |

The CTC-loss gap is real and growing through training, but the qualitative
output makes clear that the §4.3 verdict will be **TIE on CER**: both EEG
and noise produce ~30-character non-content fragments. Wave-3 is the
intervention.

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
| `4d0b420` | exp02 progress.md: live timeline of the May 1 pilot |
| `94475dd` | **Wave-3**: TFM-frozen fix + data filters (drop_sources/min_text/max_text/max_seconds/drop_nan/drop_zero) + DistilBERT `LMBridgeHead` + `wave3_cells()` matrix |
| `c69739a` | exp02 cli: `--cells` slice flag for multi-box pilot orchestration |
| `edd62b9` | exp02 lm_bridge_head: iterate per-layer to be robust to transformers API churn |
| `c0c0564` | exp02 progress.md: wave-3 launch + post-audit timeline |
| `2b42182` | exp02 findings.md: full patch table + wave-3 launch timeline + decision rule |
| `4562b8f` | **Wave-3 mid-run fix**: split LM-bridge optimizer into bridge_lr (1e-5) + head_lr (1e-3) groups; the unified head_lr=1e-3 destroyed the pretrained DistilBERT weights and trapped the 5 lm-bridge cells in complete blank-collapse |
