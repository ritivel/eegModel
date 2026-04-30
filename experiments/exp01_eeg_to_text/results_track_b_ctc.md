# Track B — CTC (ASR-style direct EEG → char decoding)

**Status:** in flight. Lead cell `reve.ctc.eeg.0` launched on Box B at
**20:58 IST**, Apr-30. Other CTC cells will fan out to Box A GPUs as
Track-A cells finish.
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

## 6. Reproducibility / artifacts (will populate as cells finish)


| artifact                                        | path                                                                                                                                |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| per-cell `metrics.json` + `predictions.parquet` | `$EXP01_DATA_ROOT/eval/<cell_id>/`                                                                                                  |
| per-step training logs (CTC loss, grad_norm)    | `$EXP01_DATA_ROOT/runs/<cell_id>/log.jsonl`                                                                                         |
| dev-sample CTC decodings during training        | `$EXP01_DATA_ROOT/runs/<cell_id>/sample_gens.jsonl`                                                                                 |
| W&B project                                     | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text) (filter `*_ctc_`*) |


Cell-id format for Track B: `<encoder>_ctc_<input>_fold<n>_pp-v2_dec-gemma4-e2b`.

This file will be updated with final per-cell CER/WER + matched-pair
gap analysis as cells finish.