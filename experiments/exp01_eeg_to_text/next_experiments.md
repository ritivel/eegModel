# Next-step experiments — Apr 30 evening

The Apr-30 pilot finished at ~10:21 IST with an unambiguously **negative** §4.3 result
(noise BLEU-1 0.136 > EEG BLEU-1 0.114, CIs disjoint, sign-flip *p* < 1e-4 — see
`[results.md](./results.md)`). The next-step plan below is built on three pieces
of new information:

1. **Two of our pretrained encoders expect a preprocessing pipeline we are not
  running**. The §4.3 failure is consistent with the encoders producing
   essentially-uninformative features on raw, unfiltered, unnormalised EEG — and
   the bridge then having no signal to align against text.
2. **Three of eight datasets contribute zero rows because of two stacked filters**
  (subject-pool built only from ZuCo + an `num_words<1` filter that drops
   word-level-only sources).
3. **The literature consensus on what does pass the §4.3 noise test** is now
  sharp: Défossez & King 2025 (Nat. Comm., A100 40 GB) and Brain4FMs 2026 both
   converge on the same recipe — bandpass + notch + per-recording z-score + clip
  - word-level contrastive alignment.

---

## 1. What the encoders actually expect (verified from each paper / repo)

### 1.1 REVE (`brain-bzh/reve-base`, NeurIPS 2025, [arXiv 2510.21585](https://arxiv.org/abs/2510.21585))

> "Remaining signals were resampled to 200 Hz, **band-pass filtered (0.5–99.5 Hz)**,
> and converted to float32 […] To address amplitude variations across recordings,
> we applied **Z-score normalization with statistics computed across the recording
> sessions** to ensure robust statistics. After normalization, **values exceeding
> 15 standard deviations were clipped**, as in Défossez et al. (2023)."

What we currently do: only a `F.interpolate(mode="linear")` resample.
What we currently miss: bandpass, notch, per-recording z-score, 15-std clipping.

### 1.2 TFM-Tokenizer (`Jathurshan/TFM-Tokenizer`, ICLR 2026, [arXiv 2502.16060](https://arxiv.org/abs/2502.16060))

Pretrained with the BIOT/LaBraM recipe:

> "All EEG recordings are resampled to 200 Hz. […] we apply a bandpass filter
> (0.1–75 Hz) and a notch filter (50 Hz)."

STFT params (which we already match): `n_fft=200`, `hop=100`, Hann window,
magnitude-only, `center=False`, `onesided=True`.

What we currently miss: bandpass, notch.

### 1.3 Défossez & King 2025 (Nat. Comm., the only 2025 word-decoding paper that passes the §4.3 protocol)

> "the M/EEG recordings were **bandpass filtered between [0.1, 40] Hz and resampled
> to 50 Hz**, using built-in functions from MNE-Python, then **scaled using sklearn's
> RobustScaler and clamped in the range [−5, 5]**."

For the EEG branch with REVE/TFM we'll keep the encoder-native rate (200 Hz) and
RobustScaler-equivalent (per-recording z-score + 15-σ clip), as that's what
the encoders pretrained on. But the cutoff range 0.1–40 Hz is a defensible
fallback for the *word-level* contrastive branch (Sec. 3.3).

### 1.4 BELT-2 (BLEU-1 52.2% on ZuCo) and CET-MAE (BLEU-4 8.99 on ZuCo)

Both use **word-level / BPE-level** contrastive loss instead of sentence-pooled
contrastive. We are currently sentence-pooled. ZuCo's `word_eeg_segments` column
gives us per-word EEG already — we just have to use it.

---

## 2. Per-source data audit (sampled; full audit blocked on V2 pipeline)


| source               | sr (Hz) | #chan | std (µV) | abs_max | sentence_eeg present? | word_eeg_segments      | channel layout                 |
| -------------------- | ------- | ----- | -------- | ------- | --------------------- | ---------------------- | ------------------------------ |
| `zuco_v1_nr`         | 500     | 105   | 4.17     | 37      | yes                   | yes                    | `E001..E105` (EGI HydroCel)    |
| `zuco_v2_nr`         | 500     | 105   | 2.69     | 23      | yes                   | yes                    | `E001..E105`                   |
| `zuco_v2_tsr`        | 500     | 105   | 2.71     | 27      | yes                   | yes                    | `E001..E105`                   |
| `derco_preprocessed` | 1000    | 32    | —        | —       | **no** (None)         | yes (372–451 segs/row) | `Fp1, Fz, F3, F7, FT9` (10-20) |
| `emmt`               | 256     | 4     | —        | —       | **no** (None)         | yes (~9 segs/row)      | `RAW_TP9, RAW_AF7, …` (Muse)   |
| `eeg_sem_relev`      | 2858    | 32    | —        | —       | **no** (None)         | yes                    | `ch01..ch32` (anonymised)      |


Three quick reads:

1. ZuCo v1 std ≈ 4 µV vs ZuCo v2 std ≈ 2.7 µV — same hardware (105-ch EGI HydroCel),
  ~1.5× scale difference. After the per-recording z-score they collapse to std≈1.
2. **DERCo / EMMT / `eeg_sem_relev` have `sentence_eeg = None`**. Our `_row_to_array`
  falls back to concatenating `word_eeg_segments` along time — but our row filter
   `num_words<1` then drops those rows because for several DERCo/EMMT shards
   `num_words` is 0 even when 372 word segments exist. **Two stacked bugs**: the
   subject pool is ZuCo-only (so DERCo/EMMT subjects never appear in any fold),
   and the `num_words` filter drops the rest.
3. DERCo "sentences" are *full Grimm fairy tales* (372–451 words per row). We
  need to re-segment by `word_strings` punctuation before they're usable for a
   sentence-level objective.

---

## 3. The plan

The matrix is designed to **saturate 8× H100 80 GB on Box A + 1× H100 80 GB on
Box B + 1× A100 40 GB on Box C** for the next 24 h. Each track answers one
specific diagnostic question.

### Track A — Preprocessing (gates everything else; ~3–4 h)

The single hypothesis: **does the §4.3 gap close once REVE/TFM see the
preprocessing they were trained against?**

10 cells in parallel, fold 0, `300 / 1200 / 500` step budget:


| Cell                                | Box | GPU | Preprocessing  | Notes                                                       |
| ----------------------------------- | --- | --- | -------------- | ----------------------------------------------------------- |
| `reve.linear.eeg.0` v2              | A   | 0   | V2             | matched-pair eeg side                                       |
| `reve.qformer.eeg.0` v2             | A   | 1   | V2             | architecture sanity                                         |
| `reve.vocab.eeg.0` v2               | A   | 2   | V2             | RVQ + new embed-table                                       |
| `tfm.linear.eeg.0` v2               | A   | 3   | V2             | second encoder                                              |
| `tfm.qformer.eeg.0` v2              | A   | 4   | V2             | architecture sanity                                         |
| `tfm.vocab.eeg.0` v2                | A   | 5   | V2             | tfm native vocab path                                       |
| `reve.linear.eeg.0` v2 + bs=32      | A   | 6   | V2 + bs=32     | InfoNCE chance level log(32)≈3.47 — meaningful contrastive  |
| `tfm.linear.eeg.0` v2 + bs=32       | A   | 7   | V2 + bs=32     | same for tfm                                                |
| `reve.linear.noise_train.0` v2      | B   | 0   | V2             | **matched noise baseline for the §4.3 test**                |
| `reve.linear.eeg.0` v2 + smaller LM | C   | 0   | V2 + BART-base | LM-prior weaker → EEG signal more visible (bs=8 fits 40 GB) |


**V2 preprocessing pipeline** (applied per row in `_row_to_array`):

1. `np.nan_to_num(eeg)` (defensive — DERCo has zero-filled rows).
2. **Bandpass**: 4th-order Butterworth `filtfilt`. Cutoffs:
  - REVE-target: 0.5 → 99.5 Hz (matches REVE pretraining; only feasible when
   `sr ≥ 250` Hz; for `sr < 250` we drop the high cutoff to `0.4 × sr`)
  - TFM-target: 0.1 → 75 Hz (matches LaBraM/BIOT/TFM pretraining)
3. **Notch**: 50 Hz (EU recordings — ZuCo Switzerland is 50 Hz, DERCo also 50 Hz),
  `Q=30`, IIR `filtfilt`. Skip if `sr ≤ 110` (Nyquist-violation guard).
4. **Anti-aliased decimation**: `scipy.signal.resample_poly(up=200*g, down=sr*g)`
  where `g` is chosen so up/down are integers. This replaces our naive
   `F.interpolate(mode="linear")`.
5. **Per-recording z-score**: `(x - x.mean()) / (x.std() + 1e-6)`, computed
  across all `(channel, time)` samples of *the entire recording session* —
   matching REVE's "statistics computed across the recording sessions". For
   sentence-level rows this is the per-row mean/std (good enough proxy).
6. **15-σ clip**: `np.clip(x, -15, 15)`.

Three secondary data fixes pinned to V2:

- **Subject pool** built from `ALL_SOURCES`, not just ZUCO_SOURCES, so DERCo /
EMMT / `eeg_sem_relev` subjects can appear in `train_subjects`.
- **Drop the `num_words<1` filter** for sources whose `sentence_eeg` is `None`
(i.e. DERCo, EMMT, `eeg_sem_relev`) — keep them iff `word_eeg_segments` is
non-empty. The collator already concatenates the segments along time.
- **DERCo re-segmentation**: pre-process step that breaks `word_strings` /
`word_eeg_segments` into sentence-shaped subrows along punctuation
boundaries. Adds ~3× more DERCo rows.

**Diagnostic series to watch (per cell):**

- `align_loss` over stage 1: should drop *below* `log(B)` for EEG (vs flat at
`log(B)` for noise). At bs=8 chance is 2.08; at bs=32 chance is 3.47.
- Encoder feature `feat_std` post-z-score: should be O(1), not O(few µV).
- Generation diversity at the end of stage 3: ≥ 50% unique hypotheses across
test inputs (the pre-fix collapse was 7–28% unique).

**Decision rule after Track A finishes:**

- **Pass** (V2 EEG BLEU-1 > V2 noise BLEU-1 with disjoint 95% CIs and
sign-flip p < 0.01): launch Track B + the 5-fold extension on the surviving
cells.
- **Fail** (gap still ≤ 0): continue to Track B (word-level CL is the next-most
likely source of signal) but do *not* launch the 5-fold matrix; that's still
premature.

### Track B — Word-level contrastive supervision (~3–4 h)

Hypothesis: **per-word EEG ⊕ per-word BPE alignment is what makes BELT-2 / CET-MAE
work; sentence-pooled InfoNCE is too coarse to break the LM-prior trap on its own.**

The implementation:

- Pull `word_eeg_segments` + `word_strings` from each ZuCo row.
- For each batch of B sentences with N total words, encode each per-word EEG
segment through the (now-aligned) encoder + bridge into a (N, d_lm) matrix.
- Tokenize each word into BPE pieces; pool the BPE token embeddings (frozen)
into one (N, d_lm) target matrix.
- Apply **D-SigLIP** (Défossez & King 2025) instead of standard InfoNCE so
repeated words across the batch are not treated as in-batch negatives:
$$\mathcal{L}*{\text{D-SigLIP}} = \frac{1}{N}\sum*{i,j=1}^N \log\frac{1}{1 + e^{z_{ij}(-t \langle b_i, w_j\rangle + b)}}$$
where $z_{ij} = +1$ if word$_i$ == word$_j$ else $-1$, with learnable $t, b$.
- Mix with the existing sentence-level loss at `weight=0.5`, run for the same
300/1200/500 budget.

8 cells in parallel:


| Cell                                            | Box | GPU | Notes                               |
| ----------------------------------------------- | --- | --- | ----------------------------------- |
| `reve.linear.eeg.0` v2 + word-CL (D-SigLIP)     | A   | 0   | best Track-A baseline + word CL     |
| `reve.qformer.eeg.0` v2 + word-CL               | A   | 1   |                                     |
| `tfm.linear.eeg.0` v2 + word-CL                 | A   | 2   |                                     |
| `tfm.qformer.eeg.0` v2 + word-CL                | A   | 3   |                                     |
| `reve.linear.eeg.0` v2 + word-CL + bs=32        | A   | 4   | strongest contrastive headroom      |
| `tfm.linear.eeg.0` v2 + word-CL + bs=32         | A   | 5   |                                     |
| `reve.linear.eeg.0` v2 + word-CL + 12k-step ext | A   | 6   | same as Track-A best, longer budget |
| `reve.qformer.eeg.0` v2 + word-CL + 12k-step    | A   | 7   |                                     |
| `reve.linear.noise_train.0` v2 + word-CL        | B   | 0   | matched noise (word-CL chance)      |
| `reve.linear.eeg.0` v2 + word-CL + BART-base    | C   | 0   | smaller-LM ablation                 |


### Track C — Architecture ablations (overlap with Track B; ~6 h on remaining slots)

When a Track-A or Track-B cell finishes early, recycle its GPU into one of:

- **Smaller LM**: Gemma-1B-IT, Flan-T5-base, BART-base. The hypothesis is the
Gemma-4-E2B prior is so strong that a small bridge can't measurably move it.
- **LoRA on the encoder**: light fine-tune of REVE's last 2 attention blocks
(PEFT, r=8). The encoder's pretraining was MAE on 60k h of clinical EEG — it
may need to specialise for sentence reading.
- **Vocab + Q-Former + word-CL**: the vocab path produces *discrete* IDs that the
LM treats as tokens; combined with word-level alignment this is the BELT-2
recipe verbatim.

### Track D — Eval-side improvements (cheap, parallel; CPU-bound)

- **Constrained decoding**: at eval time, restrict the LM's output vocab to the
union of training-set + test-set words. This kills the "Florida congressman"
failure mode by construction. We expect BLEU-1 to either rise (signal really
was there but smothered by the prior) or stay flat (we're decoding from
noise even with the constraint, in which case Track A/B/C answer is the
truth).
- **Word-rank / closed-set top-k**: per-word rank metric (BELT-2's
closed-set evaluation). Forces a proper denominator.
- **Cross-fold averaging**: report 5-fold mean ± std for all surviving cells so
the within-fold gap (which we now know is ≈0.02 BLEU-1) is no longer the
same magnitude as the EEG-vs-noise gap.

---

## 4. Compute schedule (next 24 h)


| Slot        | Box A (8× H100)                                                                        | Box B (1× H100) | Box C (1× A100 40 GB) |
| ----------- | -------------------------------------------------------------------------------------- | --------------- | --------------------- |
| 18:30–22:00 | Track A · 8 cells                                                                      | Track A · noise | Track A · BART-base   |
| 22:00–02:00 | Track B · 8 cells                                                                      | Track B · noise | Track B · BART-base   |
| 02:00–06:00 | Track C · 8 cells                                                                      | Track C · noise | Track C · T5-base     |
| 06:00–11:00 | 5-fold extension on the best surviving (encoder, bridge) — fold 1..4 + the noise twin. |                 |                       |


Approx 12k–24k step-equivalents per day across all GPUs. Total ≈ 168 H100-hours

- 24 A100-hours.

---

## 5. Reproducibility / re-launch commands

```bash
# Track A: fan out 8 cells across Box A's 8 GPUs.
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.60 \
  "cd /home/ubuntu/work/eegModel/experiments/exp01_eeg_to_text && \
   set -a && source .env && set +a && \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup .venv/bin/python -m exp01.cli pilot \
     --parallel \
     --cells reve.linear.eeg.0,reve.qformer.eeg.0,reve.vocab.eeg.0,tfm.linear.eeg.0,tfm.qformer.eeg.0,tfm.vocab.eeg.0,reve.linear.eeg.0_bs32,tfm.linear.eeg.0_bs32 \
     --preprocess v2 \
     --stage1-steps 300 --stage2-steps 1200 --stage3-steps 500 \
     --batch-size 8 --no-grad-checkpoint --num-workers 4 \
     > /tmp/track-a.log 2>&1 & echo PID-A=\$!"

# Track A noise twin on Box B (H100):
ssh -i ~/Downloads/modal_biosigtotext ubuntu@192.222.53.81 \
  "cd /home/ubuntu/work/eegModel/experiments/exp01_eeg_to_text && \
   set -a && source .env && set +a && \
   CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m exp01.cli train \
     reve.linear.noise_train.0 \
     --preprocess v2 \
     --stage1-steps 300 --stage2-steps 1200 --stage3-steps 500 \
     --batch-size 8 --no-grad-checkpoint --num-workers 4 \
     > /tmp/track-a-noise.log 2>&1 & echo PID-noise=\$!"

# Track A smaller-LM on Box C (A100 40 GB):
ssh -i ~/Downloads/modal_biosigtotext ubuntu@<box-c> \
  "cd /home/ubuntu/work/eegModel/experiments/exp01_eeg_to_text && \
   set -a && source .env && set +a && \
   CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m exp01.cli train \
     reve.linear.eeg.0 \
     --preprocess v2 --decoder facebook/bart-base \
     --stage1-steps 300 --stage2-steps 1200 --stage3-steps 500 \
     --batch-size 16 --num-workers 4 \
     > /tmp/track-a-bart.log 2>&1 & echo PID-bart=\$!"
```

After Track A finishes (~3.5 h), the matched-pair gap is computed via:

```bash
.venv/bin/python -c "
from exp01 import eval as ev, storage; import json
def load(c): return json.load(open(storage.EVAL/c/'metrics.json'))
eeg   = load('reve_linear_eeg_fold0_v2_dec-gemma4-e2b')
noise = load('reve_linear_noise_train_fold0_v2_dec-gemma4-e2b')
print(ev.eeg_noise_gap(eeg, noise))
"
```

A positive `gap` with the bootstrap 95% CI strictly above 0 = V2 alone closes
the §4.3 gap. A non-positive gap = continue to Track B before any longer runs.

---

## 6. Citations

- **REVE preprocessing recipe**: El Ouahidi et al., *REVE: A Foundation Model
for EEG*, NeurIPS 2025. [arXiv 2510.21585](https://arxiv.org/abs/2510.21585) §3.1.1.
- **TFM-Tokenizer recipe**: Pradeepkumar et al., *Tokenizing Single-Channel EEG
with Time-Frequency Motif Learning*, ICLR 2026. [arXiv 2502.16060](https://arxiv.org/abs/2502.16060) §B.2.
- **D-SigLIP word-level alignment**: d'Ascoli et al., *Towards decoding individual
words from non-invasive brain recordings*, *Nat. Commun.* 16, 10521 (2025).
[doi:10.1038/s41467-025-65499-0](https://doi.org/10.1038/s41467-025-65499-0).
- **BPE-level alignment / multi-task EEG decoding**: Zhou et al., *BELT-2: Bootstrapping
EEG-to-Language representation alignment for multi-task brain decoding*,
[arXiv 2409.00121](https://arxiv.org/abs/2409.00121). 52.2% BLEU-1 on ZuCo.
- **Per-channel z-score consensus**: anonymous *Graph-Enhanced EEG-to-Text
Decoding*, ICLR 2026 submission, [openreview vEYRsHoWJ2](https://openreview.net/pdf?id=vEYRsHoWJ2) §3.1; Brain4FMs benchmark, [arXiv 2602.11558](https://arxiv.org/abs/2602.11558) §3.1.
- **§4.3 noise baseline protocol**: Jo, Lee, et al., *Are EEG-to-Text Models Working?*,
arXiv 2405.06459 / *Sci. Reports* 2025.

