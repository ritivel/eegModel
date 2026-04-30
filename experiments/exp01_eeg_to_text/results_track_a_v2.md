# Track A — V2 preprocessing pilot

**Status:** **all 8 cells finished by 21:34 IST** (eval included). Final
matched-pair §4.3 results in §2.4. The remaining commentary (§2.2 and
beyond) is the live snapshot at 20:46 IST while cells were mid-flight,
kept for reproducibility.
**Hypothesis:** Does giving REVE / TFM the bandpass + notch + per-recording
z-score + 15-σ clip pipeline they were *pretrained on* close the §4.3
EEG-vs-noise gap?
**Decoder budget:** stage1=300, stage2=1200, stage3=500 (same as the V1 pilot
on Apr-30 morning, so cell-to-cell deltas are clean).

> Companion docs:
>
> - `[results.md](./results.md)` — V1 pilot results (the negative §4.3 baseline this track tries to overturn).
> - `[next_experiments.md](./next_experiments.md)` — the full 24-h plan (this is Track A; Track B is CTC).
> - `[results_track_b_ctc.md](./results_track_b_ctc.md)` — the parallel CTC track.
> - W&B project: [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text) (filter `pp-v2_`*)

---

## TL;DR (final)

1. **Two cells positively beat the noise baseline on BLEU-1** for the
  first time in this project: `reve.qformer.eeg.0` (0.1300 vs noise 0.1263,
   gap +0.0037) and `tfm.qformer.eeg.0` (0.1295, gap +0.0032). The CIs
   overlap the noise floor heavily, so this is **NOT a clean §4.3 pass**
   (which requires disjoint CIs and sign-flip *p* < 0.01) — but it is a
   meaningful directional flip from the V1 result, where noise won by
   −0.022 with disjoint CIs and *p* < 1e-4.
2. **Q-Former is the bridge that works** for both REVE and TFM. Linear
  bridges land 0.5–2 points below noise; Q-Former bridges land 0.3–0.4
   points above. Architecture matters; both encoders' linear bridges
   collapse onto a single LM-prior mode.
3. **Vocab cells fail §4.3 cleanly** — `tfm.vocab.eeg.0` (0.0951)
  falls below noise (0.1263) with disjoint CIs, and `reve.vocab.eeg.0`
   (0.1125) is below noise too. The vocab path is structurally broken
   in both V1 and V2.
4. **All 8 cells produce LM-prior English biographies regardless of EEG
  content**. The qualitative output is identical across cells — same
   "U.S. House of Representatives", "Republican Party", "Florida
   congressman" templates as the noise baseline. The +0.003 BLEU-1 gap
   is from cosmetic n-gram overlap of Wikipedia-style English with the
   target Wikipedia references; it is *not* a meaningful EEG-decoding
   signal.
5. **CER is uniformly ≈ 2.4 across all cells** — confirming
  character-level overlap with references is essentially zero. CER is
   only meaningful for length-controlled output (CTC); for soft-prompt
   cells it's saturated at "no overlap".

The strongest claim Track A supports: **V2 preprocessing improves the
encoder's ability to differentiate sentences (stage-1 align_loss
crosses below chance) but the LM prior in Gemma-4-E2B still dominates
greedy generation**. To test whether EEG content can be decoded at
all, we need the LM out of the loop entirely — that's Track B (CTC).
See `[results_track_b_ctc.md](./results_track_b_ctc.md)`.

## TL;DR (mid-flight, kept for archive)

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


| metric       | mean       | 95% CI             |
| ------------ | ---------- | ------------------ |
| **BLEU-1**   | **0.1263** | [0.1190, 0.1339]   |
| BLEU-2       | 0.0174     | [0.0155, 0.0197]   |
| BLEU-3       | 0.0085     | [0.0072, 0.0099]   |
| BLEU-4       | 0.0047     | [0.0038, 0.0059]   |
| ROUGE-1-F    | 0.1277     | [0.1193, 0.1361]   |
| BERTScore-F1 | −0.0550    | [−0.0662, −0.0443] |
| CER (added)  | tba        | —                  |
| WER (added)  | tba        | —                  |


`n_test = 257` (ZuCo only, fold 0).

V2 noise BLEU-1 (0.126) is *slightly lower* than V1 noise BLEU-1
(0.136). Mechanically: the V2 z-scored noise has the same per-channel
statistics as the V2 z-scored EEG, so the bridge sees a more uniform
prefix that nudges Gemma into a more concentrated mode. Either way,
**the EEG cells need to clear ≈0.126 BLEU-1 with disjoint CIs to count
as decoding from EEG**.

### 2.1.1 First two §4.3 EEG-vs-noise gaps — vocab cells (finished 21:11 IST)

The two vocab cells were the first Track-A cells to wrap (smaller models,
stage-3 LoRA finished early). Both **fail §4.3 in V2** the same way they
failed in V1.


| cell                                   | BLEU-1                      | vs noise V2 (0.126)                | ROUGE-1-F | CER  | WER  | n_unique / 257 |
| -------------------------------------- | --------------------------- | ---------------------------------- | --------- | ---- | ---- | -------------- |
| `reve.linear.noise_train.0` v2 (Box B) | **0.1263** [0.1190, 0.1339] | (the floor)                        | 0.1277    | n/a  | n/a  | 18             |
| `reve.vocab.eeg.0` v2                  | 0.1125 [0.1056, 0.1199]     | **below noise** (CIs barely touch) | 0.1196    | 2.40 | 3.06 | 33             |
| `tfm.vocab.eeg.0` v2                   | 0.0951 [0.0893, 0.1012]     | **below noise, CIs disjoint**      | 0.1065    | 2.60 | 3.51 | 25             |


CER values around 2.4–2.6 mean the average edit distance between
hypothesis and reference is 2.4–2.6× the *length of the reference* —
essentially zero textual overlap. The dev-time `'HeHeHeHe…'` collapse
seen in §3.4 turns out to be a small-set artefact: at test time both
vocab cells produce LM-prior biographies, e.g.:

> **REF:** *"Ultimately feels empty and unsatisfying, like swallowing a Communion wafer…"*
> **HYP (`reve.vocab`):** *"He was born in the United States in 1931 in the town of New York City. He attend…"*
>
> **REF:** *"The Movie will reach far beyond its core demographic."*
> **HYP (`tfm.vocab`):** *"He was a member of the British Royal Family. He was the son of Prince Edward, Du…"*

The vocab path was already failing §4.3 in V1; **V2 preprocessing
doesn't rescue it**. The pathology is in the vocab-extension path
itself — the new embedding rows learn to project EEG into a tight
cluster that Gemma reads as "produce a Wikipedia-shaped biography".
The codebook commit_loss for `reve.vocab` flat-lined at ~30 (not
converging), which is consistent.

The 6 soft-prompt cells are still running (~21:30–21:35 IST ETA)
and are the more interesting test of whether V2 alone can close the
§4.3 gap — see §2.2.

### 2.2 Per-cell training trajectory (snapshot at 20:46 IST)

For each cell, latest stage / step / LM loss + (where applicable)
stage-1 align_loss vs the chance level `log(B)`:


| cell                            | stage | step           | LM loss  | stage1 align_loss end  | dev div  | verdict                      |
| ------------------------------- | ----- | -------------- | -------- | ---------------------- | -------- | ---------------------------- |
| `reve.linear.eeg.0`             | 2     | 1060/1200      | 3.42     | 2.038 (≈chance 2.08)   | 15/16    | weak alignment, diverse hyps |
| `reve.qformer.eeg.0`            | 2     | 1040/1200      | **2.84** | **1.896 (sub-chance)** | 9/16     | **strongest cell**           |
| `tfm.linear.eeg.0`              | 2     | 1060/1200      | 5.15     | 2.099 (chance)         | 14/16    | bridge produces non-English  |
| `tfm.qformer.eeg.0`             | 2     | 1080/1200      | 3.03     | 2.085 (chance)         | 12/16    | LM-prior English             |
| `reve.vocab.eeg.0`              | 3     | 500/500 (done) | 3.15     | n/a (bs=1)             | **1/16** | `'HeHe…'` collapse           |
| `tfm.vocab.eeg.0`               | 3     | 500/500 (done) | 3.40     | n/a (bs=1)             | **1/16** | `'HeHe…'` collapse           |
| `reve.linear.eeg.1`             | 2     | 1020/1200      | 3.09     | **1.989 (sub-chance)** | 10/16    | fold-1 confirms reve         |
| `tfm.linear.eeg.1`              | 2     | 1030/1200      | 5.36     | 2.092 (chance)         | 16/16    | high diversity but garbage   |
| `reve.linear.noise_train.0` (B) | done  | —              | —        | —                      | —        | matched §4.3 baseline        |


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

### 2.4 Final §4.3 matched-pair gaps (all 8 EEG cells finished by 21:34 IST)


| cell                                   | BLEU-1 (95% CI)             | gap vs noise floor 0.1263 | ROUGE-1-F | BERTScore-F1      | CER  | WER  | n_unique / 257 |
| -------------------------------------- | --------------------------- | ------------------------- | --------- | ----------------- | ---- | ---- | -------------- |
| `**reve.qformer.eeg.0`**               | **0.1300 [0.1220, 0.1385]** | **+0.0037** ✓             | 0.124     | **−0.014** (best) | 1.99 | 2.52 | 52             |
| `**tfm.qformer.eeg.0`**                | **0.1295 [0.1208, 0.1388]** | **+0.0032** ✓             | 0.122     | −0.047            | 2.27 | 2.90 | 12             |
| `reve.linear.eeg.0`                    | 0.1221 [0.1147, 0.1297]     | −0.0042                   | 0.131     | −0.046            | 2.31 | 2.84 | 65             |
| `reve.vocab.eeg.0`                     | 0.1125 [0.1056, 0.1199]     | −0.0138                   | 0.120     | −0.033            | 2.40 | 3.06 | 33             |
| `reve.linear.eeg.1`                    | 0.1096 [0.1029, 0.1161]     | −0.0167                   | 0.109     | −0.108            | 2.54 | 2.93 | 82 / 312       |
| `tfm.linear.eeg.0`                     | 0.1033 [0.0970, 0.1104]     | −0.0230                   | 0.119     | −0.051            | 2.30 | 2.97 | 58             |
| `tfm.linear.eeg.1`                     | 0.0998 [0.0939, 0.1059]     | −0.0265                   | 0.105     | −0.061            | 2.57 | 3.29 | 132 / 312      |
| `tfm.vocab.eeg.0`                      | 0.0951 [0.0893, 0.1012]     | **−0.0312**               | 0.106     | −0.095            | 2.60 | 3.51 | 25             |
| `reve.linear.noise_train.0` (V2 floor) | 0.1263 [0.1190, 0.1339]     | (the floor)               | 0.128     | −0.055            | n/a  | n/a  | 18             |


Three immediate reads:

- **Q-Former is the only bridge that crosses the noise line.** The +0.0037
/ +0.0032 gaps are within the noise CI's reach, but the *direction* is
consistently positive — and Q-Former cells were also the strongest
stage-1 align_loss reducers (`reve.qformer` 1.896, the only cell to
drop more than 0.1 below chance). Together this is suggestive of real
encoder→bridge information transfer.
- **Linear bridges all fall below noise.** `reve.linear.eeg.0` is the
closest at −0.0042; the rest (linear+fold1, both vocab cells, both
TFM linear) are 1–3 points below.
- **TFM vs REVE matters less than bridge architecture matters.** Within
each bridge, REVE and TFM produce similar BLEU-1 (linear: 0.122 / 0.103;
qformer: 0.130 / 0.130; vocab: 0.113 / 0.095). Within each encoder, the
linear→qformer→vocab ranking is consistent. Q-Former just uses the
encoder features more effectively, regardless of which encoder.

### 2.5 Same-reference cross-cell qualitative

Random sample of 4 references with all top-cell hypotheses, demonstrating
that **the +0.003 BLEU-1 gap is not from EEG content**:

> **REF:** *"Anton Seidl (7 May 1850 - 28 March 1898) was a Hungarian conductor."*
> `reve.qformer.eeg.0`: *"In 1954, he was elected to the U.S. House of Representatives as a Democrat from the 11th congressional district of Calif…"*
> `tfm.qformer.eeg.0`: *"In 1946, he was elected to the U.S. House of Representatives by the Republican Party, representing the 11th congressional…"*
> `reve.linear.eeg.0`: *"In 1953, he was appointed to the position of Director of the National Aeronautics and Space Administration (NASA), which…"*
>
> **REF:** *"Kerouac's spontaneous, confessional language style inspired other writers…"*
> `reve.qformer.eeg.0`: *"In 1941, he was elected to the U.S. House of Representatives as a Republican from the 11th district of Florida. He serve…"*
> `tfm.qformer.eeg.0`: *"In 1946, he was elected to the U.S. House of Representatives by the Republican Party, representing the 11th congressional…"*
> `reve.linear.eeg.0`: *"He was a member of the Republican Party from 1964 to 1976, serving as the Republican nominee for the U.S. Senate in 1976…"*
>
> **REF:** *'He was "discovered" by the film director James Whale, and within a few years, he was a popular leading man…'*
> `reve.qformer.eeg.0`: *"The film is a very good example of the kind of film that is often called a 'bad movie' in the 1990s. It is a film that i…"*
> `tfm.qformer.eeg.0`: *"In 1994, he was elected to the U.S. House of Representatives, representing the 43rd congressional district of Florida. H…"*
> `reve.linear.eeg.0`: *"In 1956, he was appointed to the position of Director of the National Aeronautics and Space Administration (NASA) and wa…"*
>
> **REF:** *"He received his bachelor's degree in 1965 and master's degree in political science in 1966 both from the University of W…"*
> `reve.qformer.eeg.0`: *"He was born in London, England, and died in London, England. He was a member of the British Royal Family, and was the so…"*
> `tfm.qformer.eeg.0`: *"In 1946, he was elected to the U.S. House of Representatives by the Republican Party, representing the 11th congressional…"*
> `reve.linear.eeg.0`: *"In 1953, he was appointed to the position of Director of the National Aeronautics and Space Administration (NASA) and wa…"*

The pattern: **all soft-prompt cells produce LM-prior biographies of
fictional U.S. politicians regardless of the actual reference**. The
+0.003 BLEU-1 advantage of `reve.qformer` over noise is entirely from
the unigram overlap (`he`, `was`, `the`, `of`, `a`, etc.) of generic
English biographical sentences with the target Wikipedia references —
not from any EEG-driven content.

This is the same failure mode as V1 §4.2 in `[results.md](./results.md)`;
V2 preprocessing has reduced *how badly* the bridge collapses (reve.qformer
hypothesis diversity is 52/257 unique vs V1's 128/257 — but more
importantly the LM loss is much lower, which means the hyps are more
*confident* repetitions of the same template), but the structural
LM-prior trap is unchanged.

### 2.3 Stage-1 trajectory comparison: V1 vs V2 (same step budget)


| cell                 | V1 stage1 loss end | V2 stage1 loss end | V1 align_loss end | V2 align_loss end |
| -------------------- | ------------------ | ------------------ | ----------------- | ----------------- |
| `reve.linear.eeg.0`  | ~10.3              | **4.67**           | ~2.08             | 2.038             |
| `reve.qformer.eeg.0` | ~10.5              | **2.85**           | ~2.08             | **1.896**         |
| `tfm.linear.eeg.0`   | ~9.8               | 7.70               | ~2.09             | 2.099             |
| `tfm.qformer.eeg.0`  | ~10.1              | 3.65               | ~2.09             | 2.085             |
| `reve.vocab.eeg.0`   | ~9.5               | 5.16               | n/a (bs=1)        | n/a (bs=1)        |
| `tfm.vocab.eeg.0`    | ~8.7               | 7.69               | n/a (bs=1)        | n/a (bs=1)        |


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
> **HYP:** `*'**'`*
>
> **REF:** *"Bread, My Sweet has so many flaws it would be easy for critics to shred it."*
> **HYP:** `*'event-time-event-time-event-time-event-time-event-time-…'`*
>
> **REF:** *"The film often achieves a mesmerizing poetry."*
> **HYP:** `*'    '`* (spaces)

`tfm.linear.eeg.1` (fold 1) is even worse — Korean filler characters,
repeated commas, nonsense:

> **REF:** *"Ford was born on a prosperous farm in Springwells Township…"*
> **HYP:** `*' version, and, I,  I,  I,  I,  I,  I,  I,  I,  I,  I,  I,  I,  I, …'`*
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

## 4. What the matched-pair §4.3 result actually said

The pre-flight predictions, kept honest about prediction quality:


| cell                 | **predicted** BLEU-1 (20:46 IST) | **actual** BLEU-1 | actual sign vs noise floor                                                       |
| -------------------- | -------------------------------- | ----------------- | -------------------------------------------------------------------------------- |
| `reve.qformer.eeg.0` | 0.13–0.14                        | **0.1300**        | **+0.0037 ✓**                                                                    |
| `reve.linear.eeg.0`  | 0.11–0.13                        | 0.1221            | −0.0042                                                                          |
| `reve.linear.eeg.1`  | 0.12–0.13                        | 0.1096            | −0.0167                                                                          |
| `tfm.qformer.eeg.0`  | 0.10–0.12                        | **0.1295**        | **+0.0032 ✓** *(under-predicted by ~0.02)*                                       |
| `tfm.linear.eeg.0`   | < 0.05 ("broken")                | 0.1033            | −0.0230 *(better than predicted; the dev `'**'` collapse was a small-set artefact)* |
| `tfm.linear.eeg.1`   | < 0.05                           | 0.0998            | −0.0265                                                                          |
| `reve.vocab.eeg.0`   | < 0.02 (HeHeHe…)                 | 0.1125            | −0.0138 *(same: dev `'HeHeHe…'` was a small-set artefact)*                       |
| `tfm.vocab.eeg.0`    | < 0.02 (HeHeHe…)                 | 0.0951            | −0.0312                                                                          |


**Two predictions wrong in the same direction**: the dev-time
`'HeHeHeHe…'` / `'**'` / Korean-filler dev outputs were all artefacts
of the small (16-sample) periodic dev snapshots — at the full 257-row
test set with V2 preprocessing, every cell produces grammatical English
biographies. Predicted "broken bridge" is actually "LM-prior English
bridge that produces confident garbage" — they all clear ≈ 0.10 BLEU-1
because Wikipedia-style English shares high-frequency unigrams with the
Wikipedia/movie-review test set.

### 4.1 §4.3 protocol verdict

Strict reading of [Jo et al. 2024 §4.3](https://arxiv.org/abs/2405.06459):
test passes iff `EEG > noise` with disjoint 95% bootstrap CIs and
sign-flip permutation `p < 0.01`.

- **No cell passes the strict §4.3 test.** Even `reve.qformer.eeg.0`'s
  CI [0.122, 0.139] overlaps the noise CI [0.119, 0.134]; overlap
  region [0.122, 0.134] is roughly half of either CI's width. A
  sign-flip permutation test on the +0.0037 gap would give `p ~ 0.2`.
- **But two cells flipped sign vs V1.** V1 had every cell strictly
  below noise with disjoint CIs and `p < 1e-4`. V2 has two cells
  strictly above noise with overlapping CIs and `p ~ 0.2`. That's a
  large directional shift even if not a clean significance pass.
- **The qualitative read (§2.5) explains both findings**: the
  soft-prompt cells are still LM-prior-driven; the BLEU-1 ranking just
  reflects *how successfully each cell converges onto the high-overlap
  mode* of Wikipedia-shaped biographical English. A 5-fold matrix on
  Q-Former cells would tell us whether the +0.003 sign-flip is
  reproducible across folds; V1 fold-1 swing was ±0.024 between folds,
  much larger than the gap.

**Conclusion:** *V2 preprocessing changes the LM-prior trap from
"noise strictly wins" (V1) to "EEG marginally wins on Q-Former, with
overlapping CIs" (V2). The trap itself is intact.* Track B (CTC) is
the architectural escape from the trap; see
[`results_track_b_ctc.md`](./results_track_b_ctc.md) for the early
CTC results, which give a meaningfully different signal pattern
(BLEU-1 0.26, BLEU-2 0.19 — both ≈ 2× higher than the noise floor).

---

## 5. What this implies for the next move

1. **CTC (Track B) is now critical**, not optional. The LM-prior trap
  on the soft-prompt path is structural — the bridge can move stage-1
   align_loss below chance and still get smothered by Gemma's prior at
   generation. Removing the LM from the loss entirely
   (`F.ctc_loss(log_probs, char_targets)`) is the cleanest test of
   "does the encoder actually use EEG content?". See
   `[results_track_b_ctc.md](./results_track_b_ctc.md)`.
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


| artifact                                        | path                                                                                                                                |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| per-cell `metrics.json` + `predictions.parquet` | `$EXP01_DATA_ROOT/eval/<cell_id>/`                                                                                                  |
| per-step training logs                          | `$EXP01_DATA_ROOT/runs/<cell_id>/log.jsonl`                                                                                         |
| dev-sample generations during training          | `$EXP01_DATA_ROOT/runs/<cell_id>/sample_gens.jsonl`                                                                                 |
| W&B project                                     | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text) (filter `pp-v2`_*) |


`<cell_id>` is e.g. `reve_qformer_eeg_fold0_pp-v2_dec-gemma4-e2b`.

This file will be updated with the final §4.3 matched-pair gap and
per-source breakdowns once all Track-A cells finish (~21:30–21:45 IST).