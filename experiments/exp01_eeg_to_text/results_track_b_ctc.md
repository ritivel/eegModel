# Track B — CTC (ASR-style direct EEG → char decoding)

**Status:** Lead cell `reve.ctc.eeg.0` finished on Box B at **21:43 IST**;
8 more CTC cells running on Box A (started 21:11–21:34 IST). Final
matched-pair §4.3 results expected ~22:15 IST.

**Headline (lead cell only, full matched-pair table after 22:15):**
The CTC cell's BLEU-1 = **0.2606 [0.237, 0.287]** is **2.06× higher**
than the V2 noise baseline (0.1263). BLEU-2 is **11× higher** (0.186
vs 0.017). But the qualitative output is mostly blanks + the most
common letters (`'  a a a a   a a a  a   a ii  .'`), which suggests
the CTC head learned the marginal char distribution rather than
EEG-driven content. **The matched noise-CTC baseline (still training,
~22:00 IST) will tell us whether this is an EEG signal or just a
char-prior**. See §3 for the partial result + interpretation.

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


| metric | mean | 95% CI | vs V2 noise floor (soft-prompt) |
| ------ | ---- | ------ | ------------------------------- |
| **BLEU-1** | **0.2606** | [0.2366, 0.2865] | **2.06× higher than 0.1263** |
| **BLEU-2** | **0.1861** | [0.1628, 0.2095] | **10.7× higher than 0.0174** |
| BLEU-3 | 0.0441 | [0.0358, 0.0532] | 5.2× higher than 0.0085 |
| BLEU-4 | 0.0155 | [0.0124, 0.0193] | 3.3× higher than 0.0047 |
| ROUGE-1-F | 0.0282 | [0.0230, 0.0339] | **0.22× of 0.1277** (lower — CTC hyps are short) |
| **CER** | **0.8135** | [0.8070, 0.8224] | (soft-prompt CER ≈ 2.4 — saturated) |
| WER | 1.0219 | [0.9997, 1.0507] | (~ length ratio; almost no exact word match) |
| BERTScore-F1 | −0.4998 | [−0.5223, −0.4785] | much lower (CTC outputs aren't semantic) |

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