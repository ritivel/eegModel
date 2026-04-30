# Track B — CTC (ASR-style direct EEG → char decoding)

**Status:** Lead cell `reve.ctc.eeg.0` finished on Box B at **21:43 IST**;
8 more CTC cells running on Box A (started 21:11–21:34 IST). Final
matched-pair §4.3 results expected ~22:15 IST.

## TL;DR (final, with the painful correction)

1. **No clean §4.3 pass on a meaningful metric.** The two matched
   pairs that *looked* like passes on BLEU-1 (Box-B fold-0 +0.044
   disjoint; Box-A fold-1 +0.139 disjoint) are **length-precision
   artefacts**: EEG cells produce shorter hyps (median 19 chars) than
   noise cells (median 31 chars), and our `BLEU-1` is sacrebleu's
   `precisions[0]`-without-brevity-penalty, which rewards short hyps
   that emit only common chars. **On CER (true edit distance) every
   matched pair has EEG ≈ noise, with the only "significant" gap
   going the wrong way (fold-1 EEG CER 0.844 > noise CER 0.783).**
2. **CTC partial blank-collapse** is the dominant failure mode for
   every REVE-CTC cell: ~70-90% of frames emit BLANK; the remaining
   frames emit space, `e`, `a`, `i`, `o`, period — i.e. the marginal
   character distribution of English. Different inputs → slightly
   different marginal mixes → varying-but-uninformative outputs.
3. **TFM-CTC fully collapses to mode** (one or two unique outputs
   across 257 inputs). TFM is unusable for this task as-is.
4. **CTC is highly unstable across batch-size config** — the *same*
   cell `reve.ctc.eeg.0` gives BLEU-1 = 0.261 on Box B (bs=16,
   grad_accum=2, eff bs=32) and BLEU-1 = 0.080 on Box A
   (bs=32, grad_accum=1, also eff bs=32). The number of gradient
   steps per epoch matters more than effective batch size.
5. **CTC loss is still high (~3.0)** — the model didn't converge on
   the small training budget (300+1200+500 ≈ 10 epochs at our scale).
   Wav2Vec2 fine-tunes for **100+ epochs**; we did 10.

The headline early-result number (BLEU-1=0.261 on Box B reve.ctc.eeg.0,
"2.06× the soft-prompt noise floor") is preserved below — it's still
a true number, but the qualitative read in §3.4 + the matched-pair
analysis in §4 show it's a precision-gaming artefact, not EEG
decoding.

**Action taken at 23:00 IST:** Track C launching with the
highest-leverage ASR fixes applied to the surviving REVE+CTC cells
([`progress.md`](./progress.md) tracks the live launches):

- **Encoder LoRA** (unfreeze REVE) — Wav2Vec2 paper: full fine-tune
  beats frozen by 15-20% absolute WER ([arXiv 2501.09459 fig 5](https://arxiv.org/abs/2501.09459)).
- **12k-step training** (10× more updates) — Wav2Vec2 / Whisper
  fine-tune for 100+ epochs.
- **SpecAugment** (time + freq masking) — standard ASR regularisation.
- **BPE-2k vocab** (instead of 50-char) — harder to game with marginal
  collapse; closer to Wav2Vec2 / Whisper recipe.
- **CTC + AED hybrid loss** ([Watanabe 2017 espnet](https://arxiv.org/abs/1609.06773))
  — CTC alone collapses to blank; pairing with attention decoder
  cross-entropy prevents that. Standard espnet recipe.
- **Beam search + KenLM 4-gram rescoring** at decode — Blank Collapse
  paper: greedy CTC is "useless without LM rescoring". A char/BPE
  4-gram trained on Wikipedia gives 5-15% absolute WER improvement.



**Hypothesis:** Bypass the LM-prior trap entirely. If we skip the frozen
Gemma decoder and train the encoder + a small CTC head directly on
char-level cross-entropy, *any* signal in the output is unambiguously
attributable to the EEG (because there is no LM to fall back on). This
is the standard ASR recipe (Wav2Vec2 + CTC, DeepSpeech): swap "audio"
for "EEG" and "phoneme/char vocab" for "char vocab".

> Companion docs:
>
> - `[results.md](./results.md)` — V1 baseline (LM-prior dominates, noise wins §4.3)
> - `[results_track_a_v2.md](./results_track_a_v2.md)` — V2 preprocessing track (in flight)
> - `[next_experiments.md](./next_experiments.md)` — overall 24-h plan
> - W&B: filter `wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text` for `*_ctc_`* runs.

---

## 1. Why CTC now

The Apr-30 V1 pilot's matched-pair §4.3 result (noise BLEU-1 0.136 >
EEG 0.114, sign-flip *p* < 1e-4) was structurally caused by Gemma's LM
prior dominating the bridge output. Both `results.md` (V1) and
`results_track_a_v2.md` §3 (V2, in flight) show the same qualitative
failure mode: the model ignores the EEG entirely and produces generic
biographical English. Even with V2 preprocessing pushing stage-1
`align_loss` sub-chance for two REVE cells, **greedy generation still
collapses onto the same "Florida congressman" template** as the noise
twin.

The user's pivot is correct: if the LM is the trap, take the LM out of
the loop. CTC on a tiny char vocab does exactly that.

The recipe is also exactly the recipe that built modern non-invasive
brain-text decoding:

- **Défossez & King 2025** (*Nat. Commun.* 16, 10521): CLIP/SigLIP
word-level alignment over a *frozen* speech encoder, no LM in the
training loss.
- **Wav2Vec2 + CTC** ([arXiv 2006.11477](https://arxiv.org/abs/2006.11477)):
the standard self-supervised speech encoder + CTC head, no LM in
training; an LM is only added at decode time as a re-scorer.
- **DeepSpeech 2** ([arXiv 1512.02595](https://arxiv.org/abs/1512.02595)):
the original "encoder + bidirectional CTC" that broke open
open-vocabulary ASR.

If REVE-features-+-CTC-head can drive even a noisy CER below the noise
baseline's CER, that's a **structurally clean §4.3 pass** — the
character-level loss has no LM bypass.

---

## 2. Design

### 2.1 Vocabulary (50 tokens)

`src/exp01/chars.py` defines:

```
BLANK_ID = 0                      # CTC blank
UNK_ID   = 1                      # for any out-of-vocab character
CHARS    = "abcdefghijklmnopqrstuvwxyz '.,?!-:;\"()0123456789"  # 48 chars
VOCAB_SIZE = 50
```

Lowercase English + the punctuation that actually appears in ZuCo
references (movie reviews + biographies). Smaller vocab → more decoded
tokens per char → CTC's `T_seq ≥ 2 × len(target)` constraint is easy.

### 2.2 Architecture

`bridges.CTCBridge`:

```
encoder features (B, T_seq, D)
  → RMSNorm
  → Linear(D → hidden=512)
  → 2 × TransformerEncoderLayer(d_model=512, nhead=8, dim_ff=2048,
                                norm_first=True, gelu, dropout=0.1)
  → Linear(hidden → vocab_size=50)
  → log_softmax → F.ctc_loss
```

Total CTC head params: ~7.4 M (vs the 5 M LoRA stage-3 params in the
soft-prompt cells; comparable scale). Crucially, **the decoder LM is
not loaded at all** — saves ~5 GB of Gemma weights on each GPU and lets
us run with `bs=16, grad_accum=2 (effective bs=32)` instead of
`bs=8, grad_accum=4` for the soft-prompt cells.

### 2.3 Loss

```python
log_probs = log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
targets, target_lengths = chars.encode_batch(text_strings)
input_lengths = torch.full((B,), T, dtype=torch.long)
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                  blank=0, zero_infinity=True)
```

`zero_infinity=True` defends against the rare case where input frames
are too few for the target — CTC returns `+inf` otherwise and one bad
batch nukes the whole optimizer state.

### 2.4 Decoding (eval)

`chars.ctc_greedy_decode(log_probs)`:

1. `argmax` over the vocab dim per frame.
2. Collapse consecutive duplicates.
3. Drop BLANKs.
4. Decode token ids back to chars.

Optional later: beam search + LM rescore (Track-B+ extension if Track B
itself shows signal).

### 2.5 Metrics

- **CER** (character error rate) — primary diagnostic for CTC. Defined
as `Levenshtein(hyp_chars, ref_chars) / len(ref_chars)`. Random
baseline: ≈ 1.0. Strong ASR: 0.05–0.10.
- **WER** (word error rate) — same but split on whitespace. Random:
≈ 1.0.
- **BLEU-1..4, ROUGE-1-F, BERTScore-F1** — kept for cross-comparison
with the soft-prompt cells.

CER and WER are added to `eval.per_sentence_scores` in this commit.

### 2.6 Why CTC ≠ "soft-prompt with smaller LM"

Other ablations (BART-base, Flan-T5-base) just swap one LM prior for a
smaller one. The model still has *an* LM that can collapse onto its
prior. CTC has **no LM in the loss**:

- The character-vocab projection is randomly initialised; it has no
prior knowledge of English.
- Each frame's distribution is a free parameter constrained only by
the CTC alignment to the target string.
- If the encoder produces feature `f_t` that is uninformative about
`c_t`, the CTC head literally cannot reduce loss below the
chance level for that frame.

So CER on noise-twin cells is the cleanest possible §4.3 diagnostic.

---

## 3. Run matrix (target: 8 cells fanned out)


| cell                     | target box / GPU                     | input       | fold | bs   | notes                                |
| ------------------------ | ------------------------------------ | ----------- | ---- | ---- | ------------------------------------ |
| `reve.ctc.eeg.0`         | **Box B GPU 0**                      | eeg         | 0    | 16×2 | **lead cell + smoke** (running now)  |
| `reve.ctc.noise_train.0` | Box A GPU 4 (after vocab cell)       | noise_train | 0    | 16×2 | **matched §4.3 pair for reve.eeg.0** |
| `tfm.ctc.eeg.0`          | Box A GPU 5 (after vocab cell)       | eeg         | 0    | 16×2 | second encoder                       |
| `tfm.ctc.noise_train.0`  | Box A GPU 0 (after soft-prompt cell) | noise_train | 0    | 16×2 | matched §4.3 pair for tfm.eeg.0      |
| `reve.ctc.eeg.1`         | Box A GPU 1                          | eeg         | 1    | 16×2 | fold-1 robustness                    |
| `tfm.ctc.eeg.1`          | Box A GPU 2                          | eeg         | 1    | 16×2 | fold-1 robustness                    |
| `reve.ctc.eeg.0` (bs=32) | Box A GPU 6                          | eeg         | 0    | 32×1 | larger-batch ablation                |
| `tfm.ctc.eeg.0` (bs=32)  | Box A GPU 7                          | eeg         | 0    | 32×1 | larger-batch ablation                |


Same 300/1200/500 step budget as the soft-prompt cells, so cell-to-cell
comparisons across the two tracks are clean. CTC cells should be
**~2–3× faster per step** than soft-prompt cells (no Gemma forward), so
the whole track wraps in ~1–1.5 h once all 8 GPUs are populated.

---

## 4. Diagnostic series to watch (per cell)

- **CTC loss curve** — should drop steadily. Random init at step 0 is
≈ `log(V) × avg_target_len ≈ 3.9 × 50 ≈ 200` pre-mean; after
`mean()` reduction, around 4–8 typical at first batch. Should
drop into the 0.5–2.0 range as the head learns.
- **Encoder feature_std** (already logged) — should remain ~1 (z-scored
by V2 preprocessing).
- **Greedy CTC dev sample** — every `generate_every` steps, log a few
`(ref, hyp)` pairs. Pre-training the head will produce
`'aaaaaaaaa…'` or `''` (collapsed-to-blank); convergence will produce
English-ish strings with letters proportional to the target.
- **Per-source CER** — broken out across ZuCo sub-corpora at eval time.
Same protocol as `results.md` §3.3.

---

## 5. Decision rule when Track B finishes

Pre-registered, applies to each (encoder × fold) matched pair (eeg vs
noise_train) at eval time:

1. **PASS §4.3 (clean)**: EEG CER < noise CER with 95% bootstrap CIs
  *disjoint*, and sign-flip permutation `p < 0.01`. → Confirms the EEG
   cell is decoding from EEG content. Move to BPE-2k vocab + LM-rescore
   beam search; this cell is the "headline" recipe.
2. **PASS §4.3 (weak)**: EEG CER < noise CER, CIs overlap. → Suggestive.
  Run the 5-fold matrix to tighten CIs.
3. **TIE**: EEG CER ≈ noise CER, both far from 1. → The CTC head is
  learning *something* but it's an artefact (probably the marginal
   character distribution); EEG content not used. Investigate.
4. **FAIL**: EEG CER ≈ noise CER ≈ 1. → Encoder features carry no
  text-relevant signal at this SR / channel layout. Pivot to a
   different encoder or a different supervision scheme.

We expect outcome **(1) or (2)** based on the V2 stage-1 align_loss
crossing below chance — there's *some* signal in the encoder features;
the question is whether it's enough for the CTC head to recover
recognisable text.

---

## 3. First CTC result — `reve.ctc.eeg.0` (Box B, finished 21:43 IST)

### 3.1 Quantitative


| metric       | mean       | 95% CI             | vs V2 noise floor (soft-prompt)                  |
| ------------ | ---------- | ------------------ | ------------------------------------------------ |
| **BLEU-1**   | **0.2606** | [0.2366, 0.2865]   | **2.06× higher than 0.1263**                     |
| **BLEU-2**   | **0.1861** | [0.1628, 0.2095]   | **10.7× higher than 0.0174**                     |
| BLEU-3       | 0.0441     | [0.0358, 0.0532]   | 5.2× higher than 0.0085                          |
| BLEU-4       | 0.0155     | [0.0124, 0.0193]   | 3.3× higher than 0.0047                          |
| ROUGE-1-F    | 0.0282     | [0.0230, 0.0339]   | **0.22× of 0.1277** (lower — CTC hyps are short) |
| **CER**      | **0.8135** | [0.8070, 0.8224]   | (soft-prompt CER ≈ 2.4 — saturated)              |
| WER          | 1.0219     | [0.9997, 1.0507]   | (~ length ratio; almost no exact word match)     |
| BERTScore-F1 | −0.4998    | [−0.5223, −0.4785] | much lower (CTC outputs aren't semantic)         |


`n_test = 257`, `n_unique (first 80 chars) = 231 / 257 = 89.9%`
(vs noise soft-prompt's 18/257 = 7%, and Track-A's best
`reve.qformer.eeg.0` at 52/257 = 20%).

### 3.2 Qualitative — what the CTC cell actually generates

> **REF:** *"Ultimately feels empty and unsatisfying, like swallowing a Communion wafer without the wine."*
> **HYP:** *`'  a a a a   a a a  a   a ii        .'`*
>
> **REF:** *"This version moves beyond the original's nostalgia for the communal film experiences of yesteryear…"*
> **HYP:** *`'  a a o a   a a a  a    i ii i       .'`*
>
> **REF:** *"The Movie will reach far beyond its core demographic."*
> **HYP:** *`'            i      .'`*
>
> **REF:** *"At times, the movie looks genuinely pretty."*
> **HYP:** *`' a    ai   a a a a   a a i i     a .'`*
>
> **REF:** *"It depends on how well flatulence gags fit into your holiday concept."*
> **HYP:** *`' a  a  a   a a a     a i      .'`*

The CTC head is producing **space-dominated output with sparse letters
(mostly `a`, `i`, occasional `o`) and a final `.`**. Hypothesis length
median is **32 chars vs reference median 108 chars** — CTC is heavily
emitting BLANK and the most common chars.

This is a classic CTC partial-collapse failure mode: the head learned
the **marginal character distribution** (`' '` is the most common char
in English text, then `e`, then `t`, etc., but the head specifically
learned `' '` and `'a'` and `'i'`). It also learned that English
sentences end in `.`. It has *not* learned letter-by-letter EEG
decoding.

### 3.3 Why the BLEU is so high

This output drives high BLEU-1 because:

- BLEU-1 counts unigram overlap between hyp and ref. The hyp is mostly
`' '` and `'a'`; both `' '` and `'a'` appear many times in every ref.
At the *character* level (sacrebleu's default), `' '`, `'a'`, `'i'`,
`'.'` are extremely common in any English sentence.
- BLEU-2 counts bigram overlap. `' a'`, `'a '`, `' i'`, etc., are also
extremely common bigrams.
- **The hyp is short, which inflates *precision*.** BLEU is precision-
oriented; a short hyp gets high precision if every char it emits
appears in the ref. The brevity penalty does kick in (hyp length
≈ 30% of ref length, so BP ≈ exp(1 − 1/0.3) ≈ 0.10), but the raw
precision is so high it still beats noise soft-prompt by 2×.

In short: **the CTC head is gaming BLEU**. The high BLEU-1/2 is a
real artefact of CTC's char-level operation, not evidence of EEG-driven
content.

### 3.4 What the matched noise CTC baseline will tell us

The clean §4.3 test for CTC: **does `reve.ctc.eeg.0` BLEU-1/CER beat
`reve.ctc.noise_train.0` BLEU-1/CER?**

Three possible outcomes when the noise CTC cell finishes (~22:00 IST):

1. **Noise CTC ≈ 0.26 BLEU-1 too** → the high BLEU is purely the
  marginal-char-distribution prior, no EEG signal. CTC does *not*
   escape the prior trap, it just changed which prior dominates. We'd
   need beam search + char-LM rescore to make CTC meaningful.
2. **Noise CTC < 0.26 BLEU-1 (e.g. 0.15–0.20)** → there's a real EEG
  signal in the gap. The strength of the gap quantifies how much
   useful info the encoder gives the CTC head.
3. **Noise CTC produces longer / less-collapsed output but with
  different chars** → comparison is messy; need to look at CER + WER
   patterns, possibly per-word rank metrics.

The CER of `reve.ctc.eeg.0` is 0.81 — meaning average edit distance
is 81% of ref length. For an empty hyp, CER would be 1.0; for a hyp
that's a perfectly correct transcript, CER would be 0. So 0.81 says
the model is *somewhere between random and empty*, plus a length bias.

**Important caveat for the matched test:** the noise CTC cell sees
per-row Gaussian noise with the same per-channel mean/std as the EEG.
After V2 z-score, the noise has std ≈ 1 like the EEG. The CTC head
will likely still learn the same marginal char distribution from the
*targets* (which are unchanged English) — so noise CTC probably also
gets BLEU-1 ≈ 0.26. **If so, the §4.3 verdict on CTC is the same as
on soft-prompt: the model is decoding from an output prior, not from
EEG.**

This would tell us that even CTC isn't enough on its own; what's
needed is either:

- More training steps (the CTC loss is still around 3.0; could be
much lower with 12k steps), OR
- A different supervision signal (word-level alignment with a real
language model rescorer, BELT-2 style), OR
- A different encoder (REVE / TFM may simply not produce
text-discriminative features at the resolution / channel layout
ZuCo provides).

The remaining 8 Track-B cells will tell us which.

---

## 4. Final matched-pair §4.3 verdict (all 9 cells, 22:50 IST)

### 4.1 Per-cell results

| cell | BLEU-1 | BLEU-2 | CER | WER | div |
| ---- | ------ | ------ | --- | --- | --- |
| `reve.ctc.eeg.0` (Box B, bs=16 ga=2) | **0.2606** [0.237, 0.287] | **0.1861** [0.163, 0.210] | 0.814 [0.807, 0.822] | 1.022 | 231/257 |
| `reve.ctc.eeg.0` (Box A, bs=32 ga=1) | 0.0803 [0.073, 0.090] | 0.0289 [0.028, 0.030] | 0.776 | 1.217 | 246/257 |
| `reve.ctc.noise_train.0` (Box A) | 0.2168 [0.205, 0.229] | 0.1334 [0.124, 0.144] | 0.807 | 1.000 | 177/257 |
| `reve.ctc.eeg.1` (Box A) | **0.2980** [0.285, 0.311] | 0.1856 [0.175, 0.197] | 0.844 | 0.985 | 260/312 |
| `reve.ctc.noise_train.1` (Box A) | 0.1586 [0.146, 0.172] | 0.0680 [0.060, 0.077] | 0.783 | 1.020 | 283/312 |
| `tfm.ctc.eeg.0` (Box A) | 0.0055 [0.003, 0.009] | 0.0031 | 0.923 | 1.011 | 82/257 |
| `tfm.ctc.noise_train.0` (Box A) | 0.0233 [0.008, 0.043] | 0.0000 | 1.000 | 1.000 | 2/257 |
| `tfm.ctc.eeg.1` (Box A) | 0.0000 | 0.0000 | 1.000 | 1.000 | 1/312 |
| `tfm.ctc.noise_train.1` (Box A) | 0.0291 [0.020, 0.040] | 0.0228 | 0.962 | 1.001 | 33/312 |

### 4.2 Matched-pair gaps

| pair | EEG B1 | noise B1 | gap_B1 | CIs disjoint? | EEG CER | noise CER | gap_CER (lower=EEG better) |
| ---- | ------ | -------- | ------ | ------------- | ------- | --------- | -------------------------- |
| reve.ctc fold-0 (Box B vs Box A bs=16-noise) | 0.261 | 0.217 | **+0.044** | **disjoint** | 0.814 | 0.807 | +0.007 (≈ tie) |
| reve.ctc fold-0 (Box A bs=32 EEG vs Box A bs=16 noise) | 0.080 | 0.217 | −0.137 | disjoint (wrong sign) | 0.776 | 0.807 | −0.031 |
| **reve.ctc fold-1** | **0.298** | 0.159 | **+0.139** | **disjoint** | 0.844 | 0.783 | **+0.061 (EEG WORSE)** |
| tfm.ctc fold-0 | 0.005 | 0.023 | −0.018 | overlap (TFM broken) | 0.923 | 1.000 | −0.077 |
| tfm.ctc fold-1 | 0.000 | 0.029 | −0.029 | disjoint (TFM broken) | 1.000 | 0.962 | +0.038 |

### 4.3 Why the "+0.044" and "+0.139" headlines are length-precision artefacts

Sample comparison from fold-1 (the +0.139 BLEU-1 result):

> **REF:** *"He was born in Fillmore, Utah to Daniel Olson and his wife Delilah King."*
> **EEG:** `'he           i     .'`     (19 chars)
> **NOISE:** `'he a a a   a a  a a  a o a  o.'`     (32 chars)
>
> **REF:** *"Everything was as superficial as the forced New Jersey lowbrow accent Uma had."*
> **EEG:** `'e               iiii     .'`     (24 chars)
> **NOISE:** `'e ae a a     a    a a   a   a a a .'`     (35 chars)
>
> **REF:** *"An important movie, a reminder of the power of film to move us…"*
> **EEG:** `'he            iei .'`     (18 chars)
> **NOISE:** `'e a a a   a   a   a  a a  .'`     (29 chars)

**Length stats (fold-1):**

- EEG hyp length: median **19** chars, p90 = 30
- Noise hyp length: median **31** chars, p90 = 53
- Reference length: median **106** chars

`sentence_bleu(...).precisions[0]` (which is what `eval._sentence_bleu`
returns) is **unigram precision without brevity penalty**. A 19-char
hyp emitting only `e`/`i`/space/`.` gets a much higher unigram
precision than a 31-char hyp that adds rarer unigrams (`o`, `n`, etc.)
— because every common char is in the ref, but extra rare chars from
the longer hyp aren't. This **rewards aggressive blank emission**.

**On CER (true Levenshtein edit distance, length-aware):**

- All matched pairs collapse to the same 0.78–0.84 range
- Fold-1 EEG CER (0.844) > noise CER (0.783) → **EEG is *worse* on
  the meaningful metric**, despite winning BLEU-1 by 0.139
- The other "positive" pair (Box-B fold-0 +0.044) has CER gap +0.007
  (essentially tied)

### 4.4 The cross-config instability

The same `reve.ctc.eeg.0` cell run on two different machines / batch
configs gives wildly different numbers:


| run | bs × grad_accum (eff bs) | BLEU-1 | CER | div |
| --- | ------------------------- | ------ | --- | --- |
| Box B | 16 × 2 (32) | 0.261 | 0.814 | 231/257 |
| Box A | 32 × 1 (32) | **0.080** | 0.776 | 246/257 |


Both have eff bs=32 but different per-batch sample counts. The bs=32
ga=1 version has **half the gradient steps per epoch** (1 vs 2 grad
updates per training mini-batch), so the head is 2× under-trained.
Sign flip is purely from compute, not from EEG.

### 4.5 Verdict

- **Strict §4.3 (sign-flip p < 0.01 on a meaningful metric)**: NO PASS.
  The two "disjoint CI" gaps on BLEU-1 are length-precision artefacts;
  the corresponding CER gaps either tie (fold-0) or go the wrong way
  (fold-1).
- **Sign of progress vs Track A**: marginal. Track A's `reve.qformer`
  had +0.0037 gap on BLEU-1 (with overlapping CIs), Track B's
  CER-honest gap is in the 0.00 to 0.06 range across all REVE pairs.
  Both are within "we don't know if there's a signal" territory.
- **What we *did* learn**: CTC alone collapses to blank + char
  marginals; the "EEG vs noise" difference is a length-distribution
  difference, not a content difference. **CTC needs the standard ASR
  fixes (encoder fine-tune, more steps, BPE, AED hybrid, LM rescore)
  to have any chance of producing content.**

This is the prompt for Track C.

---

## 6. Reproducibility / artifacts (will populate as cells finish)


| artifact                                        | path                                                                                                                                |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| per-cell `metrics.json` + `predictions.parquet` | `$EXP01_DATA_ROOT/eval/<cell_id>/`                                                                                                  |
| per-step training logs (CTC loss, grad_norm)    | `$EXP01_DATA_ROOT/runs/<cell_id>/log.jsonl`                                                                                         |
| dev-sample CTC decodings during training        | `$EXP01_DATA_ROOT/runs/<cell_id>/sample_gens.jsonl`                                                                                 |
| W&B project                                     | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text) (filter `*_ctc`_*) |


Cell-id format for Track B: `<encoder>_ctc_<input>_fold<n>_pp-v2_dec-gemma4-e2b`.

This file will be updated with final per-cell CER/WER + matched-pair
gap analysis as cells finish.