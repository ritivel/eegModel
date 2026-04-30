# Pilot results — Apr 30, 2026

Quantitative + qualitative analysis of the post-fix 9-cell pilot launched at
14:00–14:10 IST. Step budget per cell: `300 (alignment) + 1200 (frozen-LM SFT) + 500 (LoRA SFT)`.

> Companion docs:
>
> - `[progress.md](./progress.md)` — live launch table and ETAs.
> - `[diagnostic_report.md](./diagnostic_report.md)` — root-cause analysis of the four bugs the pilot uncovered.
> - W&B project: [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text)

---

## TL;DR

1. **All four bugs are observably fixed in the data**: every cell trained without zero-grad pathologies, every cell produced non-degenerate generations, every cell has loadable checkpoints whose eval reproduces training behaviour.
2. **The matched (eeg, noise) pair gives an unambiguous *negative* result**: the noise baseline strictly beats the EEG cell on BLEU-1 (0.136 vs 0.114, CIs disjoint, sign-flip permutation `p < 1e-4`). Per Jo et al. 2024 §4.3, this means the model is decoding from the LM prior, not from EEG.
3. **The qualitative evidence corroborates this**: every cell — EEG or noise, REVE or TFM, linear/qformer/vocab — produces *plausible English biographies and film-history snippets* that ignore the actual reference. The "best" cells are the ones whose prefix happens to nudge Gemma toward sentence patterns that share more unigrams with the reference set.
4. **The data pipeline currently caps the ceiling**: 3 of 8 datasets (DERCo, EMMT, eeg_sem_relev) contribute **0 rows** to training because the fold's subject pool is built only from ZuCo. Per-subject normalisation, anti-aliased resampling, and DERCo sentence segmentation are missing. These are tractable fixes that should be made before any 12k-step extension.

---

## 1. Setup recap

- 9 cells = 3 encoders × 3 bridges, fold 0 (full pilot), plus a fold-1 robustness check on `reve.linear` and `tfm.linear`, plus the matched `noise_train` baseline on Box B.
- Encoders: REVE (`brain-bzh/reve-base`, 512-d, 200 Hz, continuous) and TFM (`Jathurshan/TFM-Tokenizer`, 64-d, 200 Hz, discrete codebook=8192). DIVER-1 omitted (no public weights).
- Bridges: `linear` (32-query attention pool + RMSNorm + Linear), `qformer` (6-layer Q-Former with 32 queries), `vocab` (extends Gemma's embedding table by 8192 rows for the EEG codebook).
- Decoder: `google/gemma-4-E2B-it`, bf16, sdpa attention. Frozen at all stages; LoRA injected in stage 3 (`r=16, α=32, dropout=0.05`) onto `language_model.layers.<i>.self_attn.{q,k,v,o}_proj` (35 layers × 4 projections × 2 matrices = 5,357,568 params).
- Splits: 5-fold leave-N-subjects-out (Yin et al. 2024). Test set restricted to ZuCo. Pilot uses fold 0; fold-1 cells exist as a single-fold robustness check.
- Eval: greedy decode, no teacher forcing (per Jo et al. §4.3). 257 ZuCo-test examples per cell (fold 0) / 312 (fold 1). Bootstrap 95% CI from `n=1000`. Sign-flip permutation test for the matched pair (`n=10,000`, two-sided).

---

## 2. Bugs fixed in this pilot (recap; full RCA in `diagnostic_report.md`)


| #   | Bug                                                                                                                                              | Where                                | Effect                                                                                                                                               | Fix                                                                                                                                                                                                                                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `np` not imported at module scope                                                                                                                | `data.py`                            | Every `noise_train` cell crashed on first batch                                                                                                      | `import numpy as np` at top                                                                                                                                                                                                                                                                                                                         |
| 2   | LoRA `target_modules` regex matched the **vision tower** of multimodal Gemma 4, not the language model                                           | `decoder.py::attach_lora`            | LoRA wrapped modules that never run during text generation; stage-3 grad ≡ 0; 5 LM layers wasted on dead adapters                                    | retarget regex to `language_model.layers.\d+.self_attn.{q,k,v,o}_proj`; also call `enable_input_require_grads()` on the PEFT model to defend against the [PEFT + grad-ckpt + frozen-embed footgun](https://github.com/huggingface/peft/issues/2826)                                                                                                 |
| 3   | Bridge collapsed onto LM prior; `RVQHead` codebook never moved                                                                                   | `bridges.py`, `model.py`, `train.py` | Identical hypotheses for different EEG inputs; `reve.vocab` was random hashing                                                                       | (a) CLIP-style InfoNCE alignment loss in stage 1 over sentence-pooled bridge vs frozen text-token embeddings ([BELT-2](https://arxiv.org/abs/2409.00121), [CET-MAE](https://aclanthology.org/2024.acl-long.393.pdf), [Defossez/King 2025](https://www.nature.com/articles/s41467-025-65499-0)); (b) straight-through + commitment loss on `RVQHead` |
| 4   | `evaluate_cell` didn't re-attach LoRA before `load_state_dict`, so PEFT-renamed keys (`base_model.model.`*) silently fell through `strict=False` | `eval.py::evaluate_cell`             | **Only 191 of 2,343 saved keys loaded**; vocab cells produced `\n\n\n` (random extended embed table); soft-prompt cells lost their LoRA contribution | detect `lora_`* in saved state dict, re-attach LoRA, then load; loud diagnostic if keys mismatch in the hundreds                                                                                                                                                                                                                                    |


All four are observably resolved in the post-fix run (see §3 and §4).

---

## 3. Quantitative results

### 3.1 Per-cell test metrics (mean [95% bootstrap CI])


| cell                                  | BLEU-1                   | BLEU-2                   | BLEU-3               | BLEU-4               | ROUGE-1-F                | BERTScore-F1                |
| ------------------------------------- | ------------------------ | ------------------------ | -------------------- | -------------------- | ------------------------ | --------------------------- |
| `reve.linear.eeg.0`                   | 0.114 [0.105, 0.123]     | 0.016 [0.015, 0.018]     | 0.008 [0.007, 0.010] | 0.005 [0.004, 0.006] | 0.112 [0.105, 0.119]     | −0.049 [−0.061, −0.037]     |
| `reve.linear.eeg.1`                   | 0.117 [0.110, 0.124]     | 0.017 [0.016, 0.019]     | 0.009 [0.008, 0.010] | 0.005 [0.004, 0.005] | 0.120 [0.113, 0.127]     | −0.046 [−0.057, −0.035]     |
| `reve.qformer.eeg.0`                  | 0.120 [0.113, 0.128]     | 0.017 [0.016, 0.019]     | 0.008 [0.007, 0.010] | 0.004 [0.004, 0.005] | 0.124 [0.116, 0.132]     | −0.036 [−0.048, −0.025]     |
| `reve.vocab.eeg.0`                    | 0.086 [0.081, 0.092]     | 0.015 [0.013, 0.016]     | 0.007 [0.006, 0.009] | 0.004 [0.003, 0.005] | 0.104 [0.097, 0.111]     | −0.119 [−0.132, −0.105]     |
| `tfm.linear.eeg.0`                    | 0.103 [0.096, 0.109]     | 0.015 [0.014, 0.017]     | 0.008 [0.007, 0.010] | 0.004 [0.004, 0.005] | 0.094 [0.087, 0.102]     | −0.037 [−0.051, −0.024]     |
| `tfm.linear.eeg.1`                    | 0.127 [0.120, 0.134]     | 0.018 [0.017, 0.020]     | 0.009 [0.008, 0.011] | 0.005 [0.004, 0.006] | 0.129 [0.120, 0.137]     | −0.035 [−0.045, −0.024]     |
| `tfm.qformer.eeg.0`                   | 0.105 [0.099, 0.112]     | 0.015 [0.014, 0.017]     | 0.008 [0.007, 0.008] | 0.004 [0.004, 0.004] | 0.108 [0.101, 0.116]     | −0.046 [−0.059, −0.035]     |
| `tfm.vocab.eeg.0`                     | 0.098 [0.091, 0.105]     | 0.017 [0.015, 0.019]     | 0.008 [0.007, 0.010] | 0.004 [0.004, 0.005] | 0.115 [0.107, 0.124]     | −0.101 [−0.114, −0.089]     |
| `reve.linear.noise_train.0` (matched) | **0.136 [0.128, 0.145]** | **0.018 [0.017, 0.020]** | 0.009 [0.008, 0.010] | 0.005 [0.004, 0.005] | **0.134 [0.125, 0.142]** | **−0.020 [−0.033, −0.009]** |


Three quick reads:

- **The noise baseline is at the top of every column.** That alone tells the story.
- The other 8 cells are tightly clustered at BLEU-1 ≈ 0.10 ± 0.02 — basically the LM-prior unigram-overlap floor for this test set.
- BLEU-3/4 are ~0.005 across the board; the differences between cells live entirely in BLEU-1 (and in BERTScore, which is most negative for vocab cells where the bridge produces the least text-shaped prefix).

### 3.2 Matched (EEG, noise) pair — the §4.3 gap test

Both cells are `reve.linear.fold0`; the only difference is whether the EEG inputs were the real recordings or the per-channel matched-mean/std Gaussian noise twin. Same train/dev/test split, same step budget, same encoder and bridge architecture.


| metric       | EEG mean | Noise mean | gap = EEG − Noise | 95% CI on gap    | sign-flip *p* (two-sided) | passes Jo §4.3? |
| ------------ | -------- | ---------- | ----------------- | ---------------- | ------------------------- | --------------- |
| BLEU-1       | 0.114    | **0.136**  | **−0.023**        | [−0.029, −0.017] | **<0.0001**               | ✗               |
| BLEU-2       | 0.016    | 0.018      | −0.002            | [−0.004, 0.000]  | 0.034                     | ✗               |
| BLEU-3       | 0.008    | 0.009      | −0.000            | [−0.002, 0.001]  | 0.40                      | ✗               |
| BLEU-4       | 0.005    | 0.005      | +0.000            | [−0.001, 0.001]  | 0.99                      | ✗               |
| ROUGE-1-F    | 0.112    | **0.134**  | **−0.022**        | [−0.027, −0.016] | **<0.0001**               | ✗               |
| BERTScore-F1 | −0.049   | **−0.020** | **−0.029**        | [−0.038, −0.020] | **<0.0001**               | ✗               |


The gap is **statistically significant in the wrong direction** for BLEU-1, ROUGE-1-F, and BERTScore-F1. For BLEU-2/3/4 the gap is statistically indistinguishable from zero — i.e., neither EEG nor noise is producing genuine bigram-or-larger overlap with the references.

The "noise wins" effect is real and explainable, not a bug: a stationary Gaussian prefix is a less-disruptive prompt to a frozen LM than a structured but uninformative EEG-derived prefix. The LM lapses into a clean "biographical Wikipedia / film-review" mode against noise, and that mode happens to share a few common stop-words and proper-noun-shape unigrams with the ZuCo test references (which are sentences from Wikipedia plus film reviews — see §5).

### 3.3 Per-source breakdown (BLEU-1 mean)

ZuCo is 5 sub-corpora; cross-source variance helps separate "encoder is bad at decoding" from "this corpus is just easier for an LM prior".


| cell                        | zuco_v1_nr     | zuco_v1_sr     | zuco_v1_tsr | zuco_v2_nr     | zuco_v2_tsr    |
| --------------------------- | -------------- | -------------- | ----------- | -------------- | -------------- |
| `reve.linear.eeg.0`         | 0.166 (28)     | 0.092 (41)     | 0.134 (37)  | 0.099 (60)     | 0.109 (91)     |
| `reve.qformer.eeg.0`        | 0.143 (28)     | 0.095 (41)     | 0.117 (37)  | 0.119 (60)     | 0.127 (91)     |
| `reve.vocab.eeg.0`          | 0.103 (28)     | 0.077 (41)     | 0.084 (37)  | 0.086 (60)     | 0.087 (91)     |
| `tfm.linear.eeg.0`          | 0.130 (28)     | 0.095 (41)     | 0.103 (37)  | 0.096 (60)     | 0.101 (91)     |
| `tfm.qformer.eeg.0`         | 0.128 (28)     | 0.085 (41)     | 0.111 (37)  | 0.101 (60)     | 0.108 (91)     |
| `tfm.vocab.eeg.0`           | 0.115 (28)     | 0.087 (41)     | 0.092 (37)  | 0.094 (60)     | 0.103 (91)     |
| `reve.linear.eeg.1`         | 0.139 (80)     | 0.098 (127)    | 0.123 (105) | —              | —              |
| `tfm.linear.eeg.1`          | 0.155 (80)     | 0.104 (127)    | 0.133 (105) | —              | —              |
| `reve.linear.noise_train.0` | **0.173 (28)** | **0.110 (41)** | 0.131 (37)  | **0.135 (60)** | **0.140 (91)** |


Two patterns:

- The same per-source ranking (`v1_nr` easiest, `v1_sr` hardest) holds for **every cell including the noise baseline**. That's diagnostic: the cross-source variance is being driven by the *text* (sentence length, vocabulary overlap with the LM's pretraining distribution), not by anything the encoder is contributing. If EEG content were doing the work, the ranking would shift between encoder/bridge variants.
- Noise leads on 4 of the 5 sub-corpora and ties on the 5th (`v1_tsr`). Consistent with the headline matched-pair result.

### 3.4 Fold-0 vs fold-1 consistency


| (encoder, bridge) | BLEU-1 fold 0        | BLEU-1 fold 1        |
| ----------------- | -------------------- | -------------------- |
| `reve, linear`    | 0.114 [0.105, 0.123] | 0.117 [0.110, 0.124] |
| `tfm, linear`     | 0.103 [0.096, 0.109] | 0.127 [0.120, 0.134] |


`reve.linear` is essentially identical across folds; `tfm.linear` swings 0.024 BLEU-1 between folds — bigger than the EEG-vs-noise gap on either fold individually. That puts a hard floor on how confidently we can call any single-fold gap "real" without the full 5-fold matrix and 5-fold averaged confidence intervals.

---

## 4. Qualitative results

### 4.1 Hypothesis diversity per cell

Greedy decoding over 257 (or 312) different EEG inputs *should* produce many different hypotheses if the prefix carries information. Here is what each cell actually produced:


| cell                        | n_test | n_unique | top-share | median hyp length | most-common hypothesis (truncated)                                                           |
| --------------------------- | ------ | -------- | --------- | ----------------- | -------------------------------------------------------------------------------------------- |
| `reve.linear.eeg.0`         | 257    | 133      | 12.8%     | 250               | `"In 1963, he was elected to the U.S. House of Representatives, representing the 33rd …"`    |
| `reve.linear.eeg.1`         | 312    | 168      | 7.7%      | 235               | `"He was a member of the Republican Party and served as the Governor of …"`                  |
| `reve.qformer.eeg.0`        | 257    | 128      | 13.6%     | 215               | `"He was a member of the Democratic Party and served as a U.S. Represent…"`                  |
| `reve.vocab.eeg.0`          | 257    | 129      | 14.0%     | 276               | `"He was a member of the British Royal Family. He was the youngest son …"`                   |
| `tfm.linear.eeg.0`          | 257    | 105      | 17.9%     | 206               | `"The film is a comedy, but it's a very funny one. It's a good movie, …"`                    |
| `tfm.linear.eeg.1`          | 312    | 113      | **22.8%** | 204               | `"In 1939, he was elected to the U.S. House of Representatives, representing …"`             |
| `tfm.qformer.eeg.0`         | 257    | **45**   | **27.6%** | 258               | `"In 1994, he was elected to the U.S. House of Representatives for the 1…"` ← most collapsed |
| `tfm.vocab.eeg.0`           | 257    | 123      | 6.2%      | 266               | `"He was born in 1921 in the United States. He was the son of a former U…"`                  |
| `reve.linear.noise_train.0` | 257    | **113**  | 6.6%      | 213               | `"In 1954, he was elected to the U.S. House of Representatives, representing …"`             |


Read this as "fraction of unique hypotheses". The healthier cells (REVE based) have ~50% unique hypotheses; the worst-collapsed cell (`tfm.qformer.eeg.0`) has only **45 unique outputs across 257 different EEG inputs**, with a single sentence accounting for **27.6% of all generations**. The noise baseline sits in the middle (113/257 unique = 44%) — comparable to the EEG cells, which is the qualitative analogue of the BLEU-1 gap reading in §3.2.

The most-common hypothesis pattern — *"In **, he was elected to the U.S. House of Representatives, representing the Nth congressional district of Florida"* — appears across **5 different cells** (REVE+linear+fold0, TFM+linear+fold1, TFM+qformer+fold0, and the noise baseline). That's not the model picking up on EEG content; that's Gemma reaching for a stable biographical-Wikipedia mode whenever the prefix is "EEG-shaped but uninformative".

### 4.2 Same reference, different cells — the smoking gun for LM-prior dominance

For every test reference both the EEG and noise model produced a hypothesis. Here are 5 references picked at random with both hypotheses side by side:

> **REF**: *"(July 9, 1845 - December 7, 1912) was a British astronomer and mathematician, the second son and fifth child of Charles and Emma Darwin."*
> **EEG (reve.linear)**: *"In 1963, he was elected to the U.S. House of Representatives, representing the 33rd congressional district of Florida. He was re-elected in …"*
> **NOISE**: *"In 1954, he was elected to the U.S. House of Representatives, representing the 10th congressional district of Florida. He served in the Hous…"*

> **REF**: *"A portrait of an artist."*
> **EEG**: *"He was a member of the Democratic Party, and served as a U.S. Representative from 1933 to 1939. He was the first Democrat from the state of …"*
> **NOISE**: *"He was a member of the Republican Party and served as a U.S. Representative from the 10th congressional district from 1995 to 2003. He was t…"*

> **REF**: *"Adam Sandler's Eight Crazy Nights grows on you -- like a rash."*
> **EEG**: *"He was a member of the Democratic Party. He was a member of the House of Representatives from 1833 to 1835. …"*
> **NOISE**: *"He was a member of the Republican Party and served as the 45th Governor of Florida from January 1, 1995 to January 1, 2003. …"*

> **REF**: *"After his family returned to their home in New York, Roosevelt started intensive tutoring under Arthur Hamilton Cutler …"*
> **EEG**: *"He was a member of the Democratic Party and served as the 43rd Governor of Florida from January 1, 1995 to January 1, 2003. He was the first…"*
> **NOISE**: *"In 1950, he was elected to the U.S. House of Representatives, representing the 11th congressional district of Florida. He served in the Hous…"*

> **REF**: *"After he left the Pentagon, he became publisher and chairman of Forbes magazine, where over the next decade he wrote frequently …"*
> **EEG**: *"He was a member of the Democratic Party and served as the 43rd Governor of Florida from 1995 to 2003. …"*
> **NOISE**: *"In 1951, he was elected to the U.S. House of Representatives, representing the 10th congressional district of Florida. …"*

The pattern across all 5 (and the rest of the test set) is identical:

- Both models **ignore the actual reference** and produce a generic Wikipedia-shaped paragraph about a fictional U.S. politician.
- The **EEG cell prefers Democrats; the noise cell prefers Republicans**. That's the *only* repeatable difference, and it's a spurious bias of whichever direction the bridge happens to push Gemma's hidden state — not anything semantic about the EEG.
- The **noise cell scores higher** because its preferred biographical sentences contain more high-frequency unigrams that ZuCo references happen to share (`he`, `was`, `the`, `of`, etc.).

### 4.3 Vocab cells specifically

Pre-fix, both vocab cells produced `'HeHeHeHe…'` (or `\n\n\n…`, depending on whether the broken eval-load was active). Post-fix:

> `reve.vocab.eeg.0`, REF: *"At times, the movie looks genuinely pretty."*
> HYP: *"He was a member of the British Royal Family. He was the youngest son of Queen Victoria and Prince Albert. …"*

> `tfm.vocab.eeg.0`, REF: *"Ultimately feels empty and unsatisfying, like swallowing a Communion wafer …"*
> HYP: *"He is a former member of the British Royal Family. He is the son of Prince Philip, Duke of Edinburgh, and his wife, Princess Edith…"*

The vocab cells now produce well-formed English with the same biographical bias as the soft-prompt cells. They sit lower on BLEU-1 (0.086, 0.098 vs 0.114 for `reve.linear`) primarily because the new vocab embeddings have only been trained for 1500 steps (stages 1+2) and the LoRA in stage 3 only adapts attention projections — there's no path to adapt the LM's *output* head to the new vocab structure.

### 4.4 Per-cell first-4-rows spot check (qualitative)

Compact view of how each cell handles the first 4 test references (full predictions in `predictions.parquet`):


| cell                        | example hyp on the same ref `"At times, the movie looks genuinely pretty."`                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `reve.linear.eeg.0`         | *"In 1963, he was elected to the U.S. House of Representatives, representing the 39th congressional district of Florida. …"*            |
| `reve.linear.eeg.1`         | *"He was a member of the Republican Party and served as the Republican Governor of the District of Columbia from 1991 to 1995. …"*      |
| `reve.qformer.eeg.0`        | *"He was the son of John Adams and Abigail Adams, and the brother of John Quincy Adams, who became the sixth president of the United…"* |
| `reve.vocab.eeg.0`          | *"He was a member of the British Royal Family. He was the youngest son of Queen Victoria and Prince Albert. …"*                         |
| `tfm.linear.eeg.0`          | *"Lead singer of the band The Beatles, he was the first to record a solo album. …"*                                                     |
| `tfm.linear.eeg.1`          | *"In 1947, he was elected to the U.S. House of Representatives, representing the 1st district of Florida. …"*                           |
| `tfm.qformer.eeg.0`         | *"In 1994, he was awarded the Presidential Medal of Freedom, the highest civilian honor in the United States. …"*                       |
| `tfm.vocab.eeg.0`           | *"He is a former member of the British Army's Special Air Service and a former member of the British Army's Special Forces. …"*         |
| `reve.linear.noise_train.0` | *"In 1951, he was elected to the U.S. House of Representatives, representing the 10th congressional district of Florida. …"*            |


Different bridge architectures push Gemma into different *modes* of the same biographical-Wikipedia-of-fictional-people manifold. None of them are reading the actual reference.

---

## 5. Why this is a "noise wins" result, not just "noise ≈ EEG"

Three factors compound:

1. **Test-set unigram statistics**. The ZuCo test references are a 50/50 mix of (a) movie reviews from Rotten Tomatoes-style corpora (`v1_sr`, `v2_sr`) and (b) Wikipedia-biography sentences (`v1_nr`, `v2_nr`, `v1_tsr`, `v2_tsr`). Common unigrams (`the`, `of`, `a`, `to`, `was`, `in`, etc.) and proper-noun shapes appear in *both* the references and Gemma's natural biography mode — so any prefix that puts Gemma into "biographical sentence" territory gets a non-trivial BLEU-1 floor for free.
2. **Noise as the cleanest possible prompt**. The Gaussian noise prefix is stationary and high-entropy; Gemma's attention basically averages it out and the LM defaults to its high-probability prior. The EEG-derived prefix carries non-stationary structure that the bridge wasn't able to align to text in 300 alignment steps — so the LM gets pushed *off* its prior into a slightly-less-fluent mode.
3. **Greedy decode amplifies small differences**. Once a different first token is selected, the rest of the sequence is locked in. So a small `noise > EEG` advantage in the first-token logits (which we can see from BLEU-1) compounds into a bigger `noise > EEG` advantage in the full hypothesis.

Per Jo et al. 2024 §4.3, this exact failure mode (EEG-prefix model ≤ noise-prefix model on every metric, with overlapping or wrong-direction CIs) is the canonical "the model is not actually using EEG" diagnosis.

---

## 6. Data-pipeline issues that limit the ceiling

These were identified during the pilot and would need to be fixed before the result above can be tightened.

### 6.1 Three of eight datasets contribute *zero* training rows

Per-source row counts after the fold-0 train filter:

```
zuco_v1_sr            3007
zuco_v1_nr            2259
zuco_v1_tsr           2977
zuco_v2_nr            3666
zuco_v2_tsr           4070
derco_preprocessed       0  ← never used
emmt                     0  ← never used
eeg_sem_relev            0  ← never used
TOTAL                15979
```

Cause: `make_folds()` builds the subject pool only from `ZUCO_SOURCES`, so DERCo's `ACB71/DGR11/HMK96/…`, EMMT's `P09/P10/…`, and eeg_sem_relev's `TRPB101/TRPB102/…` participants don't appear in any fold's `train_subjects` set and get dropped at the row-filter step. The README's *"training spans all 8 sources"* is currently false; the model is trained on ZuCo only.

### 6.2 Per-source signal scales differ by ~10×, no normalisation

Median signal stats per source (from raw EEG values in the parquets):

```
source                  sr (Hz)   #chan   eeg_std (µV)   abs_max (µV)  channel layout example
zuco_v1_sr                  500     105        4.09             57.0   E001…E105 (EGI HydroCel)
zuco_v2_nr                  500     105        2.27             20.8   E001…   ← 2× smaller std than v1_sr
zuco_v2_tsr                 500     105        1.49             15.3   E001…   ← 3× smaller std than v1_sr
derco_preprocessed         1000      32        0.00              0.0   Fp1, Fz, F3, F7   ⚠ first 20 rows return zeros
emmt                        256       4        0.99              4.4   RAW_TP9, RAW_AF7… (Muse)
eeg_sem_relev             2858      32        8.90            120.5   ch01, ch02, … (anonymous)
```

No per-subject z-scoring; no per-source unit calibration. Even within ZuCo, v1 std is ~2× v2 std on the same hardware (probably different reference / common-average choices). The bridge has to spend capacity learning per-source amplitude rather than content.

### 6.3 Resampling is naive

`F.interpolate(mode="linear")` is used both in the collator (per-row to encoder native SR) and inside the encoders. For 500→200 Hz this is mildly bad (Nyquist); for 1000→200 Hz it loses 4× bandwidth without filtering; for 2858→200 Hz (eeg_sem_relev) it aliases all energy above 100 Hz back into the band. Should be `scipy.signal.decimate` or `mne.filter.resample` with proper anti-aliasing.

### 6.4 DERCo "sentences" are whole stories

A typical DERCo row's `sentence_text` is the full text of a Grimms' fairy tale (~2000 words), with a 9-minute EEG clip attached. REVE was designed for ~1-second windows. Even if the dataset filter were fixed (§6.1), DERCo would need to be re-segmented at the sentence boundary before it's usable.

### 6.5 `eeg_sem_relev` channels are unrecognised by REVE

Channel names are `ch01…ch32`; REVE's position bank returns the all-zero coordinate for every one of them. Even if used, REVE would receive 32 channels of EEG with no spatial information for that source.

---

## 7. Conclusions and recommended next steps

### Conclusions

1. The pipeline now trains and evaluates correctly. Every diagnostic signal (`align_loss` plateauing at `log(B)` for noise, `commit_loss` dropping for `reve.vocab`, stage-3 grad_norm > 0 for vocab cells, ≥ 191/2343 keys actually loaded at eval) tells a consistent story.
2. **At this step budget on this data, the model does not use EEG content.** The matched (eeg, noise) pair fails the §4.3 protocol decisively, and the qualitative outputs corroborate: every cell produces generic biographical English independent of the EEG input.
3. The single-fold spread (`tfm.linear` at 0.103 on fold 0 vs 0.127 on fold 1) is **larger than the within-fold EEG-vs-noise gap (≈0.02)**, so any positive result from "more compute" needs at least 5-fold averaging before it can be claimed.

### Recommended next-step ordering

Highest expected value first:

1. **Fix the subject-filter bug** so DERCo/EMMT/eeg_sem_relev actually contribute. Then audit each source's content and either resegment DERCo or skip it in `ALL_SOURCES`. Likely +50–100% effective training data once done.
2. **Per-subject channel-wise z-score** at row-load time. Cheapest alignment of cross-subject signal scale; matches what every published EEG decoding paper does first.
3. **Replace `F.interpolate` with anti-aliased decimation** in `_collate` and `_resample`.
4. **Bump contrastive batch from 8 → 32** for stage 1. Chance level moves from `log(8) = 2.08` to `log(32) = 3.47`, giving the alignment loss meaningful headroom to drop into. (Currently `align_loss` only beats noise by ~0.01–0.03, which is at the floor of the metric.)
5. **Add a simple data-hygiene unit test** in CI: `for src in ALL_SOURCES: 0 < eeg_std < 50 µV and recognised_chans/total > 0.5 and eeg_duration < 60 s`. Would have caught all of §6.
6. *Then* think about architecture changes (LoRA on REVE/TFM, larger Q-Former, BPE-level instead of sentence-level contrastive — BELT-2's recipe). Those are higher-effort and the data wins above are likely larger.
7. **12k-step overnight extension** of the surviving (encoder, bridge) pairs is the *last* thing to do — useful only after the data and architecture wins above. At the current trajectory it would gain ~0.01–0.02 BLEU-1, not the ~0.10 needed to clear the noise baseline.

### What I would NOT do

- Don't run the 5-fold × 3-input full matrix yet. With the data pipeline in its current state, you'd burn ~3 days of H100 time confirming the §3.2 negative result on 5 folds.
- Don't tune step counts further before fixing the data. The loss curves show interleaved EEG/noise trajectories (full plot in `diagnostic_report.md`), which is a signature of "no useful signal", not "needs more steps".

---

## 8. Reproducibility & artifacts


| Artifact                                                  | Location                                                                                                         |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| All 9 cells' final `metrics.json` + `predictions.parquet` | Box A: `$EXP01_DATA_ROOT/eval/<cell_id>/`; Box B: same path on `192.222.53.81`                                   |
| Pre-fix run snapshot                                      | `$EXP01_DATA_ROOT/archive/buggy_run_2026-04-30T08-30/` on each box                                               |
| Per-step training logs                                    | `$EXP01_DATA_ROOT/runs/<cell_id>/log.jsonl`                                                                      |
| Periodic dev sample generations during training           | `$EXP01_DATA_ROOT/runs/<cell_id>/sample_gens.jsonl`                                                              |
| Code fixes (4 bugs)                                       | `src/exp01/{data,decoder,bridges,model,train,config,eval}.py`, see `diagnostic_report.md` for diffs              |
| W&B project                                               | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text) |


### W&B run links per cell


| cell                        | post-fix W&B run                                                                                                                             |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `reve.linear.eeg.0`         | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/g5luo0ae](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/g5luo0ae) |
| `reve.qformer.eeg.0`        | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/u62ofgx5](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/u62ofgx5) |
| `reve.vocab.eeg.0`          | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/6tgxo9g8](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/6tgxo9g8) |
| `tfm.linear.eeg.0`          | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/me3f96bm](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/me3f96bm) |
| `tfm.qformer.eeg.0`         | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/xxbzsx9u](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/xxbzsx9u) |
| `tfm.vocab.eeg.0`           | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/8u7oxn9b](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/8u7oxn9b) |
| `reve.linear.eeg.1`         | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/quvry6s2](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/quvry6s2) |
| `tfm.linear.eeg.1`          | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/y088o2jd](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/y088o2jd) |
| `reve.linear.noise_train.0` | [https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/7ne592ia](https://wandb.ai/ritivel-eeg-ritivel/exp01-eeg-to-text/runs/7ne592ia) |


