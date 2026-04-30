# Track A — V2 preprocessing pilot

**Status:** in flight. Launched 19:53 IST, Apr-30. Snapshot at **20:46 IST**.
**Hypothesis:** Does giving REVE / TFM the bandpass + notch + per-recording
z-score + 15-σ clip pipeline they were *pretrained on* close the §4.3
EEG-vs-noise gap?
**Decoder budget:** stage1=300, stage2=1200, stage3=500 (same as the V1 pilot
on Apr-30 morning, so cell-to-cell deltas are clean).

> Companion docs:
>
> - [`results.md`](./results.md) — V1 pilot results (the negative §4.3 baseline this track tries to overturn).
> - [`next_experiments.md`](./next_experiments.md) — the full 24-h plan (this is Track A; Track B is CTC).
> - [`results_track_b_ctc.md`](./results_track_b_ctc.md) — the parallel CTC track.
> - W&B project: [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text) (filter `pp-v2_*`)

---

## TL;DR (mid-flight)

1. **V2 preprocessing is doing measurable work in stage 1.** Two REVE
   cells went **meaningfully sub-chance** on the InfoNCE alignment loss
   for the first time in this project (`reve.qformer.eeg.0` 1.896,
   `reve.linear.eeg.1` 1.989, vs chance log(8)=2.079). LM losses are
   30–60% lower at the same step than the V1 baseline.
2. **The matched noise twin finished and gives BLEU-1 = 0.126 [0.119, 0.134]**.
   This is the §4.3 floor that the EEG cells must clear to count as
   "decoding from EEG content".
3. **Qualitative samples show the LM-prior trap is still in full effect**
   on the soft-prompt cells. Even `reve.qformer.eeg.0` (lowest LM loss,
   sub-chance align) produces "He was a member of the United States Navy
   from 1941 to 1945…" for every dev sentence, with only 9/16 unique
   hypotheses. The noise baseline produces almost identical biographies
   ("He was a member of the Republican Party and served as the 45th
   Governor of Florida…", 18/257 unique on the test set).
4. **Vocab cells are still degenerate** — `'HeHeHeHeHeHe…'` repetition,
   1/16 unique, exactly the pre-fix V1 vocab pathology. The new vocab
   rows aren't escaping the issue with V2 preprocessing alone.
5. **TFM is much weaker than REVE** across the board. `tfm.linear` cells
   produce non-English output (`'**'`, `'event-time-event-time-…'`,
   Korean filler characters, repeated commas). May be a single-channel-vs-
   multi-channel architecture mismatch — needs Track-B-style follow-up.

The early read: **V2 preprocessing alone unlocks a real but partial
signal in REVE** (sub-chance alignment is a strict positive), but
the LM prior still dominates greedy generation. **Track B (CTC) is now
critical** — it removes the LM from the loop entirely, so any signal
is unambiguously from EEG.

---

## 1. Setup

- **Boxes:** Box A (8× H100 80 GB) + Box B (1× H100 80 GB).
- **Cells (9 total):**
  - GPU 0–3 (Box A, bs=8, no grad-ckpt): `reve.linear / reve.qformer / tfm.linear / tfm.qformer  ×  eeg  ×  fold0`
  - GPU 4–5 (Box A, bs=1, grad-ckpt on): `reve.vocab / tfm.vocab  ×  eeg  ×  fold0`
  - GPU 6–7 (Box A, bs=8, no grad-ckpt): `reve.linear / tfm.linear  ×  eeg  ×  fold1` (fold-1 robustness)
  - GPU 0  (Box B, bs=8, no grad-ckpt): `reve.linear  ×  noise_train  ×  fold0` (matched §4.3 baseline)
- **Preprocessing pipeline V2** — applied per row before noise-twin
  substitution, so the matched pair is fair:
  - REVE-target (`v2_reve`): bandpass 0.5–99.5 Hz + notch 50 Hz + polyphase
    resample to 200 Hz + per-recording z-score + 15-σ clip
    (matches REVE pretraining recipe, [arXiv 2510.21585 §3.1.1](https://arxiv.org/abs/2510.21585))
  - TFM-target (`v2_tfm`): bandpass 0.1–75 Hz + notch 50 Hz + polyphase
    resample to 200 Hz + per-recording z-score + 15-σ clip
    (matches LaBraM/BIOT/TFM pretraining recipe, [arXiv 2502.16060 §B.2](https://arxiv.org/abs/2502.16060))
- **Three secondary data fixes pinned to this launch:**
  - Subject pool now built from `ALL_SOURCES` (not just ZUCO_SOURCES) →
    DERCo / EMMT / `eeg_sem_relev` participants now appear in
    `train_subjects` instead of being silently dropped.
  - `num_words<1` filter relaxed for word-only sources (DERCo / EMMT /
    `eeg_sem_relev`) so their rows are kept.
  - Sentence-hash filter is no longer applied to non-ZuCo rows
    (the partition is over ZuCo unique sentences only).
  - Result: **18 391 train rows (was 15 979 in V1, +15%)**
    — `derco=110/110, emmt=841/841, eeg_sem_relev=1461/1461` all
    flow through now.
- **Three Track-A-launch crashes already fixed and in main:**
  - DERCo OOM cap: time-axis truncated at 12 s in the collator
    (DERCo "sentences" are 9-min Grimm fairy tales)
  - REVE pos_bank wrap: silent shrink-to-(matched, 3) when channel names
    don't match → wrapped to always return (C, 3) with sensible fallback
    positions for unknown channels.
  - bs=32 OOM at 80 GB → re-architected as 2 fold-1 cells at bs=8.

---

## 2. Quantitative — what we have so far

### 2.1 Matched §4.3 noise baseline (Box B, finished)

| metric | mean | 95% CI |
| ------ | ---- | ------ |
| **BLEU-1** | **0.1263** | [0.1190, 0.1339] |
| BLEU-2 | 0.0174 | [0.0155, 0.0197] |
| BLEU-3 | 0.0085 | [0.0072, 0.0099] |
| BLEU-4 | 0.0047 | [0.0038, 0.0059] |
| ROUGE-1-F | 0.1277 | [0.1193, 0.1361] |
| BERTScore-F1 | −0.0550 | [−0.0662, −0.0443] |
| CER (added) | tba | — |
| WER (added) | tba | — |

`n_test = 257` (ZuCo only, fold 0).

V2 noise BLEU-1 (0.126) is *slightly lower* than V1 noise BLEU-1
(0.136). Mechanically: the V2 z-scored noise has the same per-channel
statistics as the V2 z-scored EEG, so the bridge sees a more uniform
prefix that nudges Gemma into a more concentrated mode. Either way,
**the EEG cells need to clear ≈0.126 BLEU-1 with disjoint CIs to count
as decoding from EEG**.

### 2.2 Per-cell training trajectory (snapshot at 20:46 IST)

For each cell, latest stage / step / LM loss + (where applicable)
stage-1 align_loss vs the chance level `log(B)`:

| cell | stage | step | LM loss | stage1 align_loss end | dev div | verdict |
| ---- | ----- | ---- | ------- | --------------------- | ------- | ------- |
| `reve.linear.eeg.0` | 2 | 1060/1200 | 3.42 | 2.038 (≈chance 2.08) | 15/16 | weak alignment, diverse hyps |
| `reve.qformer.eeg.0` | 2 | 1040/1200 | **2.84** | **1.896 (sub-chance)** | 9/16 | **strongest cell** |
| `tfm.linear.eeg.0` | 2 | 1060/1200 | 5.15 | 2.099 (chance) | 14/16 | bridge produces non-English |
| `tfm.qformer.eeg.0` | 2 | 1080/1200 | 3.03 | 2.085 (chance) | 12/16 | LM-prior English |
| `reve.vocab.eeg.0` | 3 | 500/500 (done) | 3.15 | n/a (bs=1) | **1/16** | `'HeHe…'` collapse |
| `tfm.vocab.eeg.0` | 3 | 500/500 (done) | 3.40 | n/a (bs=1) | **1/16** | `'HeHe…'` collapse |
| `reve.linear.eeg.1` | 2 | 1020/1200 | 3.09 | **1.989 (sub-chance)** | 10/16 | fold-1 confirms reve |
| `tfm.linear.eeg.1` | 2 | 1030/1200 | 5.36 | 2.092 (chance) | 16/16 | high diversity but garbage |
| `reve.linear.noise_train.0` (B) | done | — | — | — | — | matched §4.3 baseline |

`dev div` is "unique hypothesis count / total" in the most recent
periodic dev-sample snapshot during training (16 dev sentences per
snapshot). 1/16 = mode collapse; 16/16 = full diversity (good or bad).

The single strongest qualitative signal is **stage-1 align_loss
crossing below chance for two REVE cells** (it never did in V1, even
once across all 9 cells). This is the contrastive objective registering
that bridge-pooled EEG features have non-trivial mutual information with
text-token embeddings. **It's a real positive — the bridge is starting
to encode sentence-discriminating information.** Whether that
information makes it to the generated text is a separate question
(see §3).

### 2.3 Stage-1 trajectory comparison: V1 vs V2 (same step budget)

| cell | V1 stage1 loss end | V2 stage1 loss end | V1 align_loss end | V2 align_loss end |
| ---- | ------------------- | ------------------- | ------------------ | ------------------ |
| `reve.linear.eeg.0` | ~10.3 | **4.67** | ~2.08 | 2.038 |
| `reve.qformer.eeg.0` | ~10.5 | **2.85** | ~2.08 | **1.896** |
| `tfm.linear.eeg.0` | ~9.8 | 7.70 | ~2.09 | 2.099 |
| `tfm.qformer.eeg.0` | ~10.1 | 3.65 | ~2.09 | 2.085 |
| `reve.vocab.eeg.0` | ~9.5 | 5.16 | n/a (bs=1) | n/a (bs=1) |
| `tfm.vocab.eeg.0` | ~8.7 | 7.69 | n/a (bs=1) | n/a (bs=1) |

V2 stage-1 LM losses are **2-4 points lower** than V1 at the same
step on the REVE cells, and align_loss for the strongest REVE cell
(`reve.qformer`) crosses below chance for the first time. The TFM cells
show much smaller V2 vs V1 gains (their bridges look broken — see §3.2).

---

## 3. Qualitative — what the model is actually generating

### 3.1 `reve.qformer.eeg.0` (best cell quantitatively) at stage 2 step 900

**9 / 16 unique hypotheses** in the most recent dev snapshot:

> **REF:** *"Presents a good case while failing to provide a reason for us to care…"*
> **HYP:** *"He was the first person to be elected to the U.S. House of Representatives from the distri…"*
>
> **REF:** *"Bread, My Sweet has so many flaws it would be easy for critics to shred it."*
> **HYP:** *"He was a member of the United States Navy from 1941 to 1945, serving as a lieutenant and l…"*
>
> **REF:** *"The film often achieves a mesmerizing poetry."*
> **HYP:** *"He was a member of the U.S. House of Representatives from 1953 to 1955. He was the first R…"*

For comparison, the **noise twin** on the same dev sentences:

> **REF:** *"At times, the movie looks genuinely pretty."*
> **HYP:** *"He was a member of the Republican Party and served as the 45th Governor of Florida from Ja…"*

The qualitative distinction between EEG and noise is essentially
"**Democrats vs Republicans**" — the same failure mode as V1 §4.2 in
the original results. The bridge is biasing Gemma slightly differently
between EEG and noise, but neither prefix is recovering the actual
reference.

### 3.2 `tfm.linear.eeg.0` and `tfm.linear.eeg.1` — broken bridge

> **REF:** *"Presents a good case while failing to provide a reason for us to care…"*
> **HYP:** *`'**'`*
>
> **REF:** *"Bread, My Sweet has so many flaws it would be easy for critics to shred it."*
> **HYP:** *`'event-time-event-time-event-time-event-time-event-time-…'`*
>
> **REF:** *"The film often achieves a mesmerizing poetry."*
> **HYP:** *`'    '`* (spaces)

`tfm.linear.eeg.1` (fold 1) is even worse — Korean filler characters,
repeated commas, nonsense:

> **REF:** *"Ford was born on a prosperous farm in Springwells Township…"*
> **HYP:** *`' version, and, I,  I,  I,  I,  I,  I,  I,  I,  I,  I,  I,  I,  I, …'`*
>
> **REF:** *"Timothy Bush, Sr. (c. 1728 - c. 1815) - soldier."*
> **HYP:** *`' ( 1800)의 (1800)의 (1800)의 (1800)의 (1800)의 (1800)의 …'`*

Plausible cause: TFM is a *single-channel* tokenizer (per its ICLR 2026
paper §3.1) — its STFT pretrained recipe expects single-channel input,
and our 105-channel ZuCo data flattens to `(B, 105 × 24, 64)` features
where each frame is a per-channel STFT chunk with no inter-channel
context. The Q-Former cell does better (English biographies, see §3.3)
because it cross-attends over all those channel-flattened frames before
emitting the soft-prompt; the linear cell just attention-pools and
projects, which keeps the per-channel artefacts visible to Gemma.

### 3.3 `tfm.qformer.eeg.0` at stage 2 step 900

12/16 unique. LM-prior English, but well-formed:

> **REF:** *"Presents a good case while failing to provide a reason for us to care…"*
> **HYP:** *"He was born in 1920 in the United States and died in 1992 in the United States. He was a p…"*
>
> **REF:** *"Bread, My Sweet has so many flaws it would be easy for critics to shred it."*
> **HYP:** *"He was born in 1924 in New York City and died in 1994 in New York City. He was a prominent…"*

Same biographical-Wikipedia mode as `reve.qformer`, just a slightly
different template ("born in YYYY in CITY and died in YYYY+50…").

### 3.4 Vocab cells — still degenerate

Both `reve.vocab.eeg.0` and `tfm.vocab.eeg.0` finished training (stage 3
step 500) and produce the **same `'HeHeHeHeHe…'` collapse** seen in the
buggy V1 run, with **1/16 unique hypotheses**. The new vocab rows + RVQ
codebook for `reve.vocab` (commit_loss 30.98 → 32.70 — codebook moving
slightly but not converging) and the TFM pre-discrete tokens for
`tfm.vocab` are evidently still mapping to a constant "He" prefix that
Gemma autoregressively repeats.

This is a **separate failure mode** from the soft-prompt LM-prior trap —
it's specifically the new vocab embedding rows collapsing. Likely fixes
(Track C / future): (a) initialise new vocab rows to the mean of
existing English-token embeddings rather than `randn × 0.02`,
(b) add per-position regularisation (entropy bonus on the vocab
distribution), (c) train the new rows for many more steps before
unfreezing the LoRA tail.

---

## 4. What the matched-pair §4.3 result will probably say

Based on stage-2 LM losses and qualitative samples, predicted final
BLEU-1 per cell (prediction made at 20:46 IST, before any cell finishes):

| cell | predicted BLEU-1 | confident gap > 0.126 (noise floor)? |
| ---- | ---------------- | ------------------------------------ |
| `reve.qformer.eeg.0` | 0.13–0.14 | borderline |
| `reve.linear.eeg.0` | 0.11–0.13 | unlikely |
| `reve.linear.eeg.1` | 0.12–0.13 | unlikely |
| `tfm.qformer.eeg.0` | 0.10–0.12 | no |
| `tfm.linear.eeg.0` | < 0.05 (gen broken) | no |
| `tfm.linear.eeg.1` | < 0.05 (gen broken) | no |
| `reve.vocab.eeg.0` | < 0.02 (HeHeHe…) | no |
| `tfm.vocab.eeg.0` | < 0.02 (HeHeHe…) | no |

**Expected outcome:** the strongest cell (`reve.qformer.eeg.0`) lands
within ~1 BLEU-1 point of the noise baseline, with overlapping CIs.
That would still leave the §4.3 protocol **failing** (the diagnosis is
"is EEG > noise with disjoint CIs?"), but a substantially closer race
than V1 (where noise won by 0.022 BLEU-1 with `p < 1e-4`).

This is consistent with the qualitative read: V2 unlocks a real but
small signal in stage 1 alignment, which gets diluted to within
LM-prior noise by the time stage 3 LoRA finishes.

The conclusive answer arrives at **~21:30–21:45 IST** when the 6
soft-prompt cells finish and per-cell metrics + per-source breakdowns
land in `$EXP01_DATA_ROOT/eval/<cell_id>/metrics.json`.

---

## 5. What this implies for the next move

1. **CTC (Track B) is now critical**, not optional. The LM-prior trap
   on the soft-prompt path is structural — the bridge can move stage-1
   align_loss below chance and still get smothered by Gemma's prior at
   generation. Removing the LM from the loss entirely
   (`F.ctc_loss(log_probs, char_targets)`) is the cleanest test of
   "does the encoder actually use EEG content?". See
   [`results_track_b_ctc.md`](./results_track_b_ctc.md).

2. **Vocab cells need a separate debugging pass** — they're regressing
   to the V1 pathology even with V2 preprocessing. Probably a future
   Track C.

3. **TFM is probably a dead end without architecture changes** — its
   single-channel design doesn't compose cleanly with multi-channel ZuCo
   EEG. Could be tried with a per-channel CTC head (Track B variant) or
   with the channel-major attention pooling that the Q-Former does, but
   not worth more soft-prompt LM cells.

4. **REVE × Q-Former is the strongest soft-prompt config** — `align_loss`
   sub-chance, `LM loss` lowest, dev hyp diversity 9/16 (vs 1/16 for
   vocab, 12-16/16 for the worse cells). After Track A wraps, this is
   the cell to extend with a 12k-step run + word-level CL (the BELT-2 /
   D-SigLIP recipe).

---

## 6. Reproducibility / artifacts

| artifact | path |
| -------- | ---- |
| per-cell `metrics.json` + `predictions.parquet` | `$EXP01_DATA_ROOT/eval/<cell_id>/` |
| per-step training logs | `$EXP01_DATA_ROOT/runs/<cell_id>/log.jsonl` |
| dev-sample generations during training | `$EXP01_DATA_ROOT/runs/<cell_id>/sample_gens.jsonl` |
| W&B project | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text) (filter `pp-v2_*`) |

`<cell_id>` is e.g. `reve_qformer_eeg_fold0_pp-v2_dec-gemma4-e2b`.

This file will be updated with the final §4.3 matched-pair gap and
per-source breakdowns once all Track-A cells finish (~21:30–21:45 IST).
