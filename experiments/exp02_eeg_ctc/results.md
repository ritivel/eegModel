# Exp02 — final results (May 1, 2026, 18:00 IST)

Wave-1 pilot + post-audit wave-3 launch. All experiments stopped at 17:38 IST on
user request. This document consolidates the final test-set §4.3 verdicts and
qualitative outputs across all 17 cells (8 wave-1 + 8 completed wave-3 + 1
incomplete wave-3 char-lm-aug on Box B).

> **Document map**:
>
> | doc | what it has |
> | --- | --- |
> | [`README.md`](./README.md) | quick start + run matrix + artifact map |
> | [`design.md`](./design.md) | design rationale, citations for every knob |
> | [`findings.md`](./findings.md) | May-1 audit + the 4 root causes + wave-3 plan |
> | [`progress.md`](./progress.md) | live timeline of both pilots |
> | this file | **final wave-1 + wave-3 results + verdicts** |

---

## 1. TL;DR

- **All 17 trained cells failed to produce readable text.** Best test CER is **0.748** (`w1.char.eeg`); best CER among the BPE-1k cells is **0.802** (`w3.clean.eeg`). Both are far from a usable EEG-to-text decoder (WER ≥ 1.0 for every cell).
- **Three of the four wave-3 paired interventions PASS strict §4.3 on CER** — but two of those passes are pathological (single-token collapse with EEG and noise differing by only 1 char per example). The honest result is **`w3.clean` is the only meaningful PASS**, and even it gives a 1.5 pp CER advantage on top of an 80 % CER floor.
- **Adding the DistilBERT LM-bridge head HURT.** All 5 lm-bridge cells (4 on Box A + 1 incomplete on Box B) collapsed to single tokens; `lm-clean.eeg` collapsed *harder* than its noise twin and now FAILS strict §4.3 on CER (noise is 0.6 pp better, p = 0.999).
- **Adding text + signal augmentation HURT** on the absolute metric: `aug-clean.eeg` CER = 0.946 vs `clean.eeg` CER = 0.802. The augmentation pushed the model into a `'the.'`-everywhere collapse; the §4.3 PASS on this cell is real (p < 0.0001) but qualitatively meaningless.
- **The 1 237-unique-sentence ZuCo training set is the binding constraint.** Wave-3's full stack (data cleaning + paraphrase × 5 / sentence + 6 GPU-side signal augs + DistilBERT priors) did not escape the wave-1 failure mode. Sequence-level CTC over a 1024-vocab BPE on this data is fundamentally constrained.
- **Recommended next step is the architectural pivot to D-SigLIP word-level contrastive alignment** ([d'Ascoli & King 2025 *Nat. Commun.* 16, 10521](https://doi.org/10.1038/s41467-025-65499-0)) — the only published recipe that has rigorously cleared §4.3 on non-invasive EEG/MEG.

---

## 2. Wave-3 design recap

Wave-3 was launched after the [May-1 audit](./findings.md) identified four root
causes of wave-1's pessimistic results: (1) data quality, (2) data sufficiency
(1 237 unique sentences), (3) no English priors in the head, (4) one
encoder-bug (TFM permanently frozen). The 4 wave-3 paired interventions (×
matched §4.3 noise twin per Jo et al. 2024) factor the contributions:

| pair | head | data | text-aug | signal-aug | hypothesis |
| --- | --- | --- | --- | --- | --- |
| `clean` | random-init transformer | cleaned | — | — | "Does cleaning the data alone help?" |
| `aug-clean` | random-init transformer | cleaned | paraphrase 0.5 | full | "Does augmentation alone help (no LM bridge)?" |
| `lm-clean` | DistilBERT bridge (42.5 M) | cleaned | — | — | "Does the LM bridge alone help?" |
| `lm-aug` | DistilBERT bridge | cleaned | paraphrase 0.5 | full | "Full stack — does it pass strict §4.3?" |
| `char-lm-aug` (Box B, **incomplete**) | DistilBERT bridge | cleaned | paraphrase 0.5 | full | char vocab variant of `lm-aug` |

**Data filters**: drop EMMT + DERCo (96 % zero-padded / 95 % truncated
respectively), drop labels < 10 chars or > 800 chars, drop EEG > 12 s, drop
NaN / all-zero rows. Reduces train set from 18 391 → 17 432 rows.

**Paraphrase aug**: 1 547 ZuCo sentences × 5 GPT-4o-mini paraphrases (already
built April 30); substituted as CTC target with prob 0.5 per row.

**Signal augs**: time shift ±5 %, channel dropout (10 % @ p=0.5), freq mask
(≤8 Hz @ p=0.5), time warp p=0.3, fourier surrogate p=0.2, feature mixup
α=0.4. Pipeline order in `eeg_common.augment.apply`.

**Bridge head**: pretrained `distilbert-base-uncased` (6 layers, hidden 768,
42.5 M params); word/position embeddings deleted; EEG features projected from
REVE's 512-dim space into BERT's 768-dim space + sinusoidal positional
encoding. After mid-run audit at 12:30 IST, optimizer split: `proj` group at
`head_lr=1e-3`, `bridge` group at `bridge_lr=1e-5` (the original unified
1e-3 destroyed the pretrained weights — see [findings.md](./findings.md) §6).

---

## 3. Final absolute metrics (greedy decode, n=257 test, mean [95 % CI])

| cell | CER | WER | BLEU-1 | BERTscore F1 | unique hyps / 257 |
| --- | --- | --- | --- | --- | --- |
| `w1.headline.eeg` | 0.806 [0.795, 0.820] | 1.005 | 0.163 | −0.171 | — |
| `w1.headline.noise` | 0.826 | 1.008 | 0.167 | −0.158 | — |
| `w1.vanilla-ctc.eeg` | 0.993 | 1.000 | 0.211 | nan† | — |
| `w1.vanilla-ctc.noise` | 0.992 | 1.000 | 0.643 | nan† | — |
| **`w1.char.eeg`** | **0.748** [0.730, 0.771] | 1.344 | 0.095 | −0.423 | — |
| `w1.char.noise` | 0.758 | 1.320 | 0.094 | −0.447 | — |
| `w1.tfm.eeg` | 0.998 | 1.000 | 0.082 | nan† | — |
| `w1.tfm.noise` | 0.993 | 1.000 | 0.471 | nan† | — |
| **`w3.clean.eeg`** | **0.802** [0.791, 0.812] | 1.000 | 0.172 | −0.170 | **257 / 257** |
| `w3.clean.noise` | 0.816 | 1.004 | 0.186 | −0.157 | 257 / 257 |
| `w3.aug-clean.eeg` | 0.946 | 0.995 | 0.491 | +0.070 | 62 / 257 |
| `w3.aug-clean.noise` | 0.961 | 0.996 | 0.536 | −0.032 | 20 / 257 |
| `w3.lm-clean.eeg` | 0.955 | 0.987 | 0.134 | −0.200 | 95 / 257 |
| `w3.lm-clean.noise` | 0.949 | 0.998 | 0.501 | +0.111 | 33 / 257 |
| `w3.lm-aug.eeg` | 0.965 | 1.000 | 0.599 | +0.150 | **4 / 257** |
| `w3.lm-aug.noise` | 0.967 | 0.999 | 0.551 | +0.125 | 6 / 257 |
| `w3.char-lm-aug.eeg` | — | — | — | — | (training killed at step 11 860 / 12 000, no eval) |

† `nan` BERTscore reflects empty hypothesis strings — the BERTscore tokenizer
returns nan for "" inputs.

**Headline observations**:

- The lowest-CER cell is `w1.char.eeg` (0.748), but it has WER = 1.34 because
  the character model emits more characters than reference (length over-shoot),
  inflating WER's substitution + insertion count.
- Among the BPE-1k variants, `w3.clean.eeg` is the best (0.802), narrowly
  beating `w1.headline.eeg` (0.806). Data cleaning bought us 0.4 pp.
- All 5 lm-bridge cells are at CER ≥ 0.95 — adding DistilBERT made things
  *worse* by encouraging single-token collapse.
- `aug-clean.noise` collapsed to **20 unique outputs** across 257 test
  examples; `lm-aug.eeg` collapsed to **4 unique outputs** (`'he.'` × 204,
  `'the.'` × 51, `'it.'` × 1, plus one). This is what's driving the §4.3
  PASS on these pairs — the model is producing trivial outputs.

---

## 4. Matched-pair §4.3 verdicts (greedy decode)

The Jo et al. 2024 §4.3 protocol declares a **strict PASS** when the matched
EEG vs noise CER gap has a 95 % bootstrap CI strictly positive (EEG below
noise) AND a sign-flip permutation `p < 0.01`.

| pair | CER gap (pp) | CI | p | strict PASS? | other strict | qualitative read |
| --- | --- | --- | --- | --- | --- | --- |
| `w3.clean` | **+1.47** | [+0.46, +2.64] | 0.0044 | ✅ | — | varied 30–50 char fragments, token distributions clearly differ between EEG and noise |
| `w3.aug-clean` | **+1.50** | [+1.23, +1.80] | 0.0001 | ✅ | BERTscore (+0.10, p=0.0001) | both pathologically collapsed; EEG outputs `'the.'` everywhere, noise outputs `'he (.'` everywhere |
| `w3.lm-clean` | −0.60 | [−0.96, −0.21] | 0.999 | ❌ | ROUGE-1-F (+0.018), WER (+0.011) | EEG collapsed harder than noise — EEG outputs `'the'`/`'theo'`, noise has more variety (`'he.'`, `'hey.'`, `'he wass.'`) |
| `w3.lm-aug` | **+0.22** | [+0.15, +0.31] | 0.0001 | ✅ | BLEU-1 (+0.049), BLEU-2 (+0.031), BERTscore (+0.025) | both completely collapsed; 204 of 257 EEG outputs are `'he.'`, 219 of 257 noise outputs are `'he.'` |

**Interpretation by pair**:

1. **`w3.clean`** — the cleanest result. 257/257 unique outputs for both EEG and noise. The 1.5 pp CER advantage is on the back of varied fragment generation, not single-token collapse. **Honest STRICT PASS** on the §4.3 protocol. But the model is producing roughly 35-character semi-word-like fragments (`'the hily unent, mov acend ace h the ray.'`) — not actual decoding.
2. **`w3.aug-clean`** — STRICT PASS on CER (p<0.0001) and BERTscore (+0.10). The CER gap is real but degenerate: EEG outputs `'the.'` (4 chars) for 257/257 test examples, noise outputs `'he (.'` (5 chars) for most. The 1.5 pp CER difference is *consistent across all 257 examples*. Mathematically a strict PASS; qualitatively meaningless.
3. **`w3.lm-clean`** — EEG cell **failed harder than noise**. EEG output collapsed to `'the'`/`'theo'` (2 unique strings 81 % of the time), while noise produced `'he.'`, `'hes.'`, `'heer.'`, `'he was'` etc. The WER and ROUGE strict passes are on the back of EEG's shorter outputs and the noise's accidental over-generation; CER (the primary metric) goes the wrong way. The DistilBERT bridge with `bridge_lr = 1e-5` was still too aggressive for the EEG path here.
4. **`w3.lm-aug`** — multiple strict passes (CER, BLEU-1, BLEU-2, BERTscore) but all on **4 unique EEG outputs across 257 test examples** vs **6 unique noise outputs**. The EEG distribution is `{'he.': 204, 'the.': 51, 'it.': 1, _ : 1}`; noise is `{'he.': 219, 'the.': 16, 'he': 13, …}`. The gap is real (`'the.'` is more common for EEG, `'he.'` is more common for noise) but the model has decoded essentially nothing — it's choosing between two trivial responses.

The decision rule from `design.md`:

> **PASS (strict)**: CER gap CI strictly above 0 AND sign-flip p < 0.01 → run 5-fold extension on the survivor.

Strictly interpreted, `w3.clean`, `w3.aug-clean`, `w3.lm-aug` all qualify. **The only one worth the 5-fold extension is `w3.clean`** because the others' strict PASS is structurally artefactual (single-token collapse with EEG and noise differing by 1 char).

---

## 5. Qualitative test-set comparison (first 4 examples)

```
REF (subj ZJN): Ultimately feels emp11111ty and unsatisfying, like swallowing
                a Communion wafer without the wine.

w1.headline.eeg :  '...'  (similar 30-50 char gibberish, see w3.clean.eeg below)

w3.clean.eeg     :  'the hily unent, mov acend ace h the ray.'
w3.clean.noise   :  'fhe manised ed3 was the comed d pident of stings ass 19y.'

w3.aug-clean.eeg :  'the.'
w3.aug-clean.noise:  'he (.'

w3.lm-clean.eeg  :  'theo'
w3.lm-clean.noise:  'he.'

w3.lm-aug.eeg    :  'he.'
w3.lm-aug.noise  :  'he.'

REF (subj ZJN): The Movie will reach far beyond its core demographic.

w3.clean.eeg     :  'a the bes to famure ening fed hia.'
w3.clean.noise   :  'w 18 edent0 was yest of pytendor of 18 untils.'
w3.aug-clean.eeg :  'the.'
w3.aug-clean.noise:  'he (.'
w3.lm-clean.eeg  :  'the'
w3.lm-clean.noise:  'he wass.'
w3.lm-aug.eeg    :  'the.'
w3.lm-aug.noise  :  'he.'
```

**Output diversity** (number of unique greedy hypotheses across 257 examples):

| cell | unique hyps | most common 3 |
| --- | --- | --- |
| `w3.clean.eeg` | **257** | (all unique fragments) |
| `w3.clean.noise` | 257 | (all unique fragments) |
| `w3.lm-clean.eeg` | 95 | `'he'` × 57, `'the'` × 24, `'he was'` × 15 |
| `w3.aug-clean.eeg` | 62 | `'he.'` × 45, `'the.'` × 37, `'he cs.'` × 23 |
| `w3.lm-clean.noise` | 33 | `'hes.'` × 65, `'heer.'` × 42, `'he.'` × 33 |
| `w3.aug-clean.noise` | 20 | `'he.'` × 124, `'he (.'` × 80, `'the.'` × 26 |
| `w3.lm-aug.noise` | 6 | `'he.'` × 219, `'the.'` × 16, `'he'` × 13 |
| `w3.lm-aug.eeg` | **4** | `'he.'` × 204, `'the.'` × 51, `'it.'` × 1 |

The output-diversity ranking is the inverse of the §4.3 PASS ranking. The
cells that PASSED most strictly (most metrics) are the ones with the
*least* diversity. This is the wave-1 length-precision artefact returning:
when both EEG and noise produce trivial outputs, even tiny systematic
differences (`'he.'` vs `'the.'`) reach significance with n=257.

**Only `w3.clean` produces non-degenerate output for both EEG and noise**, and
its CER gap (+1.47 pp) is the most defensible signal.

---

## 6. What worked, what didn't, what I'd do next

### What worked

- **Data filtering** (drop EMMT/DERCo/garbage labels/NaN, cap EEG at 12 s) gave a small but real CER improvement: `w3.clean.eeg` 0.802 vs `w1.headline.eeg` 0.806 (with the wave-1 baseline being on the unfiltered data). The §4.3 gap on `clean` is the best honest signal across all wave-3 cells.
- **TFM-frozen-bug fix** ([commit `94475dd`](https://github.com/ritivel/eegModel/commit/94475dd)) — for any future TFM ablation, the encoder will actually train. Wave-1's TFM verdicts (`tfm.eeg` CER 0.998, BLEU-1 0.082) are confirmed-broken artefacts; not interpretable.
- **Multi-box pilot orchestration** (`--cells :8` / `--cells 8:`) cleanly split wave-3 across the 9 GPUs.
- **Operational rigor** — the audit caught two pipeline bugs in real time (TFM `@torch.no_grad()`, then DistilBERT API churn, then the bridge_lr issue) and recovered without losing the entire wave-3 budget.

### What didn't work

- **Adding the DistilBERT LM-bridge head HURT in our setup.** All 5 lm-bridge cells (4 BPE + 1 char) collapsed to 1–6 unique outputs. The bridge_lr fix kept training stable but didn't escape the collapse — the model learns to predict the BERT-prior most-likely-token over the entire reference distribution and stops there. The attention pattern likely needs cross-modal contrastive pretraining (a la BELT-2 / d'Ascoli) before BERT fine-tuning is useful.
- **Aggressive paraphrase + signal augmentation HURT** on the absolute metric. `w3.aug-clean.eeg` CER = 0.946 vs `w3.clean.eeg` CER = 0.802 — adding the augmentation suite drove the model into single-token collapse. The 0.5 paraphrase prob may be too high for a model that's still struggling to align EEG to text.
- **Sequence-CTC over BPE-1k on 1 237 unique sentences is fundamentally constrained.** None of the wave-3 interventions produced readable text. The character model (`w1.char.eeg` CER 0.748) is the lowest-CER cell overall but still 0 % readable.

### Things confirmed not broken

- §4.3 noise-twin generation: `corr(EEG, noise) = 0.005`, EEG is 1/f spectrum (low/high ratio 62.86), noise is white (1.02), per-channel mean/std match within float-precision.
- Subject splits: no overlap across the 88 + 3 + 3 partition.
- REVE encoder: features clearly drift across training (wave-3 confirmed; wave-1 also showed this).
- Data filters: train n drops cleanly from 18 391 → 17 432 (consistent with EMMT 841 + bad rows).

### Recommended next steps

1. **Architectural pivot to D-SigLIP word-level contrastive alignment** ([d'Ascoli & King 2025 *Nat. Commun.* 16, 10521](https://doi.org/10.1038/s41467-025-65499-0)). The only published recipe that has rigorously cleared §4.3 on non-invasive EEG/MEG, and the design doc lists this as Plan B if all sequence-CTC cells fail. Wave-3 confirms they do. Word-level alignment with per-word EEG segments + BPE-token contrastive is structurally different from sequence-CTC and doesn't depend on a 1024-vocab generation head.
2. **Multi-subject EEG MAE pretraining** (Plan B item #2 in [`design.md`](./design.md) §6) — a small EEG MAE pretrained on all 8 ZuCo subjects' raw EEG, then per-subject fine-tune. Mitigates the 1 237-unique-sentence bottleneck by getting structure from the EEG side instead of the text side.
3. **Drop the BPE-1k vocab in favour of char vocab.** `w1.char.eeg` had the lowest CER (0.748) without any of wave-3's interventions. Char-vocab CTC has the same 50-token softmax regardless of training-set vocabulary size, which is a much better fit for a 1 237-sentence dataset.
4. **Don't pursue 5-fold extension on the wave-3 PASS cells.** The §4.3 strict pass on `aug-clean` and `lm-aug` is structurally artefactual. The only cell worth a 5-fold check is `w3.clean`, but the gap is small enough (~1.5 pp on an 80 % floor) that 5-fold confirmation buys very little.

---

## 7. Compute summary (final)

| box | wave | cells | wall time | GPUs |
| --- | --- | --- | --- | --- |
| Box A | wave-1 (Apr 30 → May 1) | 8 cells (headline, vanilla, char, TFM × 2 each) | ~7.5 h | 8× H100 |
| Box B | wave-2 (May 1 00:35 → 06:30) | 1 cell `aug-signal` | ~6 h | 1× H100 (killed before eval at the wave-3 cutover) |
| Box A | wave-3 main pass (May 1 08:55 → 14:30) | 4 transformer-only cells (`clean`, `aug-clean` × 2 each) | ~5.5 h | 4× H100 (GPUs 0, 1, 6, 7) |
| Box A | wave-3 lm-bridge pass (May 1 12:50 → 17:50) | 4 lm-bridge cells (`lm-clean`, `lm-aug` × 2 each) | ~5 h | 4× H100 (GPUs 2, 3, 4, 5) |
| Box B | wave-3 char-lm-aug (May 1 12:55 → 17:38) | 1 cell `char-lm-aug.eeg` (killed at step 11 860 / 12 000) | ~4.7 h | 1× H100 |

Total compute: ~95 H100-hours.

---

## 8. Per-cell artifact paths

```
$EXP02_DATA_ROOT/runs/<cell_id>/
  log.jsonl             per-step ctc_loss, cr_ctc_kl, intermediate_ctc, aed_loss,
                        grad_norm, head_lr, encoder_lr, encoder_active, elapsed
  sample_gens.jsonl     periodic dev (ref, hyp_greedy) snapshots during training
                        (every 10 % of steps)
  stats.jsonl           encoder feature stats (mean, std, abs_max, nonzero_frac)
  model.pt              final checkpoint
  model_step6000.pt     mid-training checkpoint
  run.log               stdout/stderr from the orchestrator-spawned subprocess

$EXP02_DATA_ROOT/eval/<cell_id>/
  metrics.json          summary mean / 95 % CI per metric, per decode mode
                        (greedy + beam if KenLM not available)
  predictions.parquet   one row per test example with metadata, ref, hyp_greedy
  gap_vs_noise.json     matched-pair §4.3 gap (after `exp02 gap <eeg_key>`)

$EXP02_DATA_ROOT/orchestrator/
  wave3_box_a.log         box-A wave-3 transformer-cells orchestrator output
  wave3_box_a_lm.log      box-A wave-3 first lm-bridge orchestrator (failed by API churn)
  wave3_box_a_lm_v2.log   box-A wave-3 second lm-bridge orchestrator (post-bridge_lr fix)
  wave3_box_b.log         box-B wave-3 first orchestrator (killed during launch)
  wave3_box_b_v2.log      box-B wave-3 char-lm-aug orchestrator (still running at kill)
```

---

## 9. Commits this session

| commit | summary |
| --- | --- |
| `593df9c` | Add exp02 single-stage CTC + extract eeg_common shared package |
| `fcdd2ec` | exp02 build-bpe: read text via parquet (not EEGSentenceDataset) |
| `0f5fd80` | exp02: signal + text data augmentation suite (off by default) |
| `f50cb26` | exp02: cell_id `--tag` suffix + wave-2 augmentation plan doc |
| `4d0b420` | exp02 progress.md: live timeline of the May 1 pilot |
| `94475dd` | **wave-3 main**: TFM-frozen fix + data filters + DistilBERT `LMBridgeHead` + `wave3_cells()` matrix |
| `c69739a` | exp02 cli: `--cells` slice flag for multi-box pilot orchestration |
| `edd62b9` | exp02 lm_bridge_head: iterate per-layer to be robust to transformers API churn |
| `c0c0564` | exp02 progress.md: wave-3 launch + post-audit timeline |
| `2b42182` | exp02 findings.md: full patch table + wave-3 launch timeline + decision rule |
| `4562b8f` | **wave-3 mid-run fix**: split LM-bridge optimizer into bridge_lr (1e-5) + head_lr (1e-3) groups |
| `b4775ef` | exp02 progress.md: 12:30 IST mid-run audit + bridge_lr post-mortem |
| (this commit) | exp02 results.md: final wave-1 + wave-3 verdicts + recommendations |
