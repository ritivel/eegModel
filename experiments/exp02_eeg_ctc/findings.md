# Exp02 — wave-1 / wave-2 findings (May 1 2026, 07:30 IST)

Comprehensive investigation triggered by the slow CER gap on the headline
cell at step ~11k / 12k. Bottom line: **wave-1 is heading for a §4.3 TIE
or weak PASS, not a strict PASS.** The CTC-loss gap (-0.7) is real but
the qualitative outputs of EEG and noise cells are both short, non-content
gibberish — the gap will largely vanish on CER. Three independent root
causes plus one outright pipeline bug. Wave-3 is launched immediately
to address all four.

---

## 1. Live status (as of investigation)

| box | cell | step | ctc_loss | gap vs noise | qualitative |
| --- | --- | --- | --- | --- | --- |
| A | `reve.bpe1k.crctc.eeg.0` (HEADLINE) | 11100 | 2.79 | **−0.65** | 30-char fragments, occasionally word-like |
| A | `reve.bpe1k.ctc.eeg.0` (vanilla + label-prior) | 11510 | 2.96 | **−0.71** | empty string for every dev example |
| A | `reve.char.crctc.eeg.0` | 11110 | 2.22 | −0.17 | "e a o ie an e" vowel soup |
| A | `tfm.bpe1k.crctc.eeg.0` | 10930 | 5.62 | −0.15 | empty / "." — see Bug 1 below |
| B | `reve.bpe1k.crctc.eeg.0_aug-signal` | 8780 | 5.23 | (twin not started) | "the s.", "he.", same length-collapse |

The headline gap trajectory is monotone-improving (−0.01 → −0.70 over
11k steps). The "−1.18" reading in `progress.md` was a single noisy
step. Gradients flow, encoder features are non-degenerate, and §4.3
noise twin is correctly white / uncorrelated with EEG. **Training is
working as designed; the design just isn't enough.**

---

## 2. Root causes

### 2.1 Pipeline bug — TFM encoder is permanently frozen

The dev-feature stats from `stats.jsonl` for both TFM cells are
**bit-exact identical for all 9 sample points**:

```
tfm.bpe1k.crctc.eeg:    step=1200..10800   mean=-0.002 std=0.125 abs_max=0.928
tfm.bpe1k.crctc.noise:  step=1200..10800   mean=-0.001 std=0.125 abs_max=0.893
```

vs REVE features which clearly drift over training (`mean=-0.077 → -0.033`,
`std=5.236 → 5.117`). The TFM encoder produces 100% identical features
regardless of training step.

Root cause is in `packages/eeg_common/src/eeg_common/encoders.py`:

```192:212:packages/eeg_common/src/eeg_common/encoders.py
    @torch.no_grad()
    def _tokenize_no_grad(self, eeg: torch.Tensor, sr: float):
        from einops import rearrange
        eeg = _resample(eeg, sr, self.spec.native_sr)
        ...

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        _, embs = self._tokenize_no_grad(eeg, sr)
        return embs
```

`encode()` calls a `@torch.no_grad()` method, so even when
`model.set_encoder_trainable(True)` flips `requires_grad=True` on every
TFM parameter, **no gradients ever reach the tokenizer**. The
`self.tokenizer.eval()` call in `__init__` keeps it permanently in eval
mode (no dropout / batchnorm updates). The TFM cells are effectively
running the GROUP-E `_frozen` ablation, not the GROUP-B encoder
ablation.

**Action**: drop the `@torch.no_grad()` and `.eval()` call. Until then,
**all TFM verdicts in the run matrix are invalid** and the matched-pair
gap is meaningless.

### 2.2 Data sufficiency — sequence-CTC over a 1.2K-sentence vocabulary

| metric | value | comment |
| --- | --- | --- |
| Train rows | 18,391 | OK by absolute count |
| **Unique train sentences (ZuCo)** | **1,237** | The actual cardinality of "what we're decoding to" |
| Sentence repetition factor | **6.57×** | Each sentence read by ~12 ZuCo subjects; minimal linguistic diversity |
| Dev rows | 240 from **3 subjects** | Tiny + dominated by ZMG |
| BPE vocab | 1,024 | Mismatched to 1.2K unique sentences |

A BPE-1024 sequence-CTC head with 1,237 unique training sentences is
fundamentally close to a 1,237-class classification disguised as text
generation. The model can game the loss by learning peaky/short-blank
predictions over a tiny target distribution and never learn real
generation. The closest published recipe that *does* clear the §4.3
bar on non-invasive EEG/MEG ([d'Ascoli & King 2025
*Nat. Commun.* 16, 10521](https://doi.org/10.1038/s41467-025-65499-0))
used **~175 hours / ~1.4 M words** with **per-word contrastive
alignment**, not sequence generation.

We have **~68 hours of usable EEG** trying to do the harder task, with
no English-language priors in the model — the head is being asked to
learn token transitions, word boundaries, syntax, and EEG-to-text
alignment all at once from 1,237 sentences.

### 2.3 Data quality — many real issues

| issue | impact |
| --- | --- |
| **DERCo source 95% truncated** | 110 rows × 541 s median, max 777 s. Trainer caps at 12 s → **19 hours of EEG silently chopped off**. Net DERCo contribution: 0.4 h |
| **eeg_sem_relev: 57% of rows >12 s** | Median 13 s, max 24 s. ~29% of signal lost |
| **Channel mismatch zero-padding**: 105 (ZuCo, 87%) / 32 (DERCo+eeg_sem_relev, 8.5%) / 4 (EMMT, 4.6%) | EMMT rows are 96% zero-padded across the channel axis when batched with ZuCo. P(batch contains non-ZuCo) = 89% |
| **REVE positional fallback** for unknown channel names (eeg_sem_relev's BioSemi naming) | Most rows from non-ZuCo sources get degenerate positional embeddings |
| **Sample-rate mismatch**: 500/256/1000/2859 Hz → forced to 200 Hz | The 2859→200 ratio is poor through linear-interp resample |
| **Tiny garbage labels**: 8 rows with text < 10 chars (`'the'`, `'a a her'`, `'oak the'`, etc.) | Model trains on garbage |
| **Multi-paragraph monster row** ("and a fox once lived together…", **3,622 chars**, 5 copies) | Hundreds of BPE tokens crammed into 12-second window |
| **2% NaN rows + 2% all-zero rows** | NaN rescued by `nan_to_num`, but noise twin generation runs `eeg.std()` *before* preprocessing → NaN noise downstream |

### 2.4 Model architecture — no English priors

The `CTCHead` is a 4-layer randomly-initialised TransformerEncoder. The
head has to learn English token transitions from scratch alongside the
EEG-to-text alignment. With 1.2K unique sentences this is hopeless.
Modern speech-CTC systems either start from a pretrained text encoder
(BERT/T5) or run a separate KenLM at decode time — we have neither in
the loop during training.

---

## 3. Wave-3 plan (launched immediately, all 9 GPUs)

### 3.1 Code-level fixes (commit before launch)

1. **TFMEncoder bug fix** — drop `@torch.no_grad()` from `_tokenize_no_grad`
   and remove `.eval()` from `__init__`. Move the `_tokenize_no_grad` rename
   to a `forward()` method that respects `self.training`.
2. **Data filtering** — add to `EEGSentenceDataset`:
   - `drop_sources: tuple[str, ...]` — exclude entire sources by name
   - `max_seconds: float | None` — drop rows whose duration exceeds the cap
     (rather than silently truncating in the collator)
   - `min_text_chars: int = 10` — drop rows with garbage labels
   - `max_text_chars: int = 800` — drop the multi-paragraph monsters
   - `drop_nan_rows: bool = True` — drop rows with any NaN value
   - `drop_zero_rows: bool = True` — drop rows whose EEG is all zeros
3. **LM-bridge head** — `LMBridgeHead` class that replaces `CTCHead`'s
   transformer with a pretrained `distilbert-base-uncased` (66M params,
   6 BERT layers, hidden=768). Word/position embeddings are bypassed;
   continuous EEG features are projected into BERT's hidden space. Full
   end-to-end fine-tune on top of REVE full fine-tune. Adds a new
   `head_type` config field.
4. **Drop TFM from default pilot** — remove `encoder_ablation_cells()`
   from `all_track_c_cells()` until the TFM bug-fix lands and we
   actually want a TFM ablation.

### 3.2 Wave-3 cell matrix (10 cells, 9 GPUs)

Box A is reused after wave-1 finishes (~30 min from now). Each EEG cell
has a matched §4.3 noise twin per Jo et al. 2024.

| GPU | cell tag | recipe | hypothesis tested |
| --- | --- | --- | --- |
| A.0 | `clean` | REVE + crctc + cleaned data | "Does cleaning the data alone help?" |
| A.1 | `clean` (noise) | matched twin | — |
| A.2 | `lm-clean` | REVE + DistilBERT bridge + cleaned data | "Does LM bridge alone help (no aug)?" |
| A.3 | `lm-clean` (noise) | matched twin | — |
| A.4 | `lm-aug` | REVE + DistilBERT bridge + cleaned + paraphrase + signal aug | **HEADLINE** — full stack |
| A.5 | `lm-aug` (noise) | matched twin | — |
| A.6 | `aug-clean` | REVE + crctc + cleaned + paraphrase + signal aug | "Does aug help WITHOUT LM bridge?" — controls for LM bridge contribution in cell A.4 |
| A.7 | `aug-clean` (noise) | matched twin | — |
| B.0 | `lm-aug` (char) → noise sequential | REVE + DistilBERT bridge + char vocab + paraphrase + signal aug | "Does char vocab beat BPE-1k under the LM bridge?" |

Wave-1 cells stay in `runs/` for evaluation; wave-3 cells live at new
suffixed paths so neither overwrites the other. After wave-3 finishes
we run `exp02 gap` on every pair to get the matched-pair §4.3 verdict
on CER, WER, BLEU, and per-decode-mode breakdowns.

### 3.3 What success looks like

Per Jo et al. §4.3 (and the doc's own decision rule):

- **Strict PASS**: CER 95% bootstrap CIs disjoint AND sign-flip
  permutation `p < 0.01`, with EEG below noise.
- **TIE/FAIL** (where wave-1 is heading): CER ≈ 1 for both cells, both
  far from useful.

The four interventions in wave-3 are designed so that *any one* of
{cleaning, paraphrase, LM bridge} could individually move CER, and the
combination should compound. If the headline (`lm-aug`) doesn't reach
strict PASS, the controls (`clean`, `lm-clean`, `aug-clean`) tell us
which intervention was responsible for whatever gap we did get.

If even `lm-aug` fails on CER, we pivot to D-SigLIP word-level
contrastive alignment ([d'Ascoli & King 2025](https://doi.org/10.1038/s41467-025-65499-0))
— the only published recipe that has rigorously cleared the §4.3 bar
on non-invasive EEG/MEG.

---

## 4. Things confirmed NOT broken

- §4.3 noise twin generation: corr(EEG, noise) = 0.005, EEG is 1/f
  spectrum (low/high ratio 62.86), noise is white (1.02), per-channel
  mean/std match within float-precision.
- Subject splits: no overlap between train/dev/test sets across the
  88 + 3 + 3 partition.
- Sentence splits: no overlap (sentence_filter set sizes 1237 / 155 / 155).
- REVE encoder *is* being trained: features drift over training.
- Training processes are alive and progressing on both boxes.
- Encoder feature stats are bounded and non-degenerate for REVE.

---

## 5. Per-issue patches landed (commits 94475dd, c69739a, edd62b9)

| # | issue | file | change | commit |
| --- | --- | --- | --- | --- |
| 1 | TFM frozen-encoder bug | `packages/eeg_common/src/eeg_common/encoders.py` | drop `@torch.no_grad()` from `_tokenize`, drop `self.tokenizer.eval()` from `__init__`; the discrete `tokenize()` path keeps `with torch.no_grad():` (codebook lookup is non-differentiable anyway) | `94475dd` |
| 2 | Garbage labels in train | `packages/eeg_common/src/eeg_common/data.py` | `min_text_chars`, `max_text_chars` filters in `EEGSentenceDataset` (default 10 / 800) | `94475dd` |
| 3 | Excessive truncation of long EEG | `packages/eeg_common/src/eeg_common/data.py` | `max_seconds` filter — drops 12 s+ rows at index time instead of silently truncating in the collator | `94475dd` |
| 4 | NaN / zero rows | `packages/eeg_common/src/eeg_common/data.py` | `drop_nan_rows`, `drop_zero_rows` filters; runtime fallback walks to next row if a degenerate one slips through | `94475dd` |
| 5 | Useless DERCo / EMMT | `packages/eeg_common/src/eeg_common/data.py` | `drop_sources` filter (default `"derco_preprocessed,emmt"` in wave-3) | `94475dd` |
| 6 | NaN-rich noise twin | `packages/eeg_common/src/eeg_common/data.py` | If the source row was NaN, replace the noise twin with `N(0,1)` instead of `NaN * std + mu` | `94475dd` |
| 7 | No English priors in CTC head | `experiments/exp02_eeg_ctc/src/exp02/lm_bridge_head.py` (new) | `LMBridgeHead`: pretrained `distilbert-base-uncased` (42.5 M params, 6 layers, hidden=768) replacing the random-init 4-layer transformer; word/position embeddings deleted, EEG features projected into BERT's hidden space, sinusoidal positions for unbounded sequence lengths | `94475dd` |
| 8 | TFM in default pilot | `experiments/exp02_eeg_ctc/src/exp02/config.py` | TFM removed from `all_track_c_cells()`; opt-in via `include_diver1=True` only | `94475dd` |
| 9 | LM-bridge wired through model + cli | `model.py`, `config.py`, `cli.py` | `head_type` config field with `transformer` / `lm_bridge` choice; full set of `--head-type / --head-lm-model-id / --head-lm-max-seq-len` flags; `_DIFF_ABLE_FIELDS` extended so the parallel orchestrator emits the right `--flag value` for each cell | `94475dd` |
| 10 | Wave-3 launch matrix | `experiments/exp02_eeg_ctc/src/exp02/config.py` | `wave3_cells()` (10 cells) factoring the four root causes; `--group wave3` in pilot; `_clean_data_kwargs / _paraphrase_kwargs / _signal_aug_kwargs` factory helpers | `94475dd` |
| 11 | Multi-box pilot orchestration | `experiments/exp02_eeg_ctc/src/exp02/cli.py` | `--cells <slice>` flag on `exp02 pilot` so Box A can run cells 0–7 while Box B runs cells 8–9 without spawning the same cell twice | `c69739a` |
| 12 | DistilBERT API churn | `experiments/exp02_eeg_ctc/src/exp02/lm_bridge_head.py` | First wave-3 launch crashed: `TypeError: Transformer.forward() missing 1 required positional argument: 'hidden_states'`. DistilBERT 4.40+ renamed the wrapper's `forward(x=...)` to `forward(hidden_states=...)`. `_bridge_forward` now iterates `self.transformer.layer` directly — robust to either signature | `edd62b9` |

## 6. Wave-3 launch (May 1 08:30 → 08:55 IST)

| time IST | event |
| --- | --- |
| 07:30 | Audit triggered after the user spotted that the loss curve looked too good given the `progress.md` qualitative samples. |
| 08:00 | `94475dd` committed (TFM-fix + data filters + LM-bridge head + wave-3 matrix). |
| 08:15 | Wave-1 + wave-2 killed on both boxes (`pkill -KILL -f "exp02.cli (train\|pilot)"`). Step-12 000 logs / sample_gens / stats preserved; `model_step6000.pt` checkpoint survives for evaluation. |
| 08:20 | DistilBERT pre-downloaded into HF cache on both boxes. `paraphrases.parquet` (827 KB) scp'd from Box A → Box B. |
| 08:25 | `c69739a` committed: `--cells :8` / `--cells 8:` slice flag on pilot. |
| 08:30 | Wave-3 launched on both boxes. **Within 4 minutes the 4 lm-bridge cells crashed** with `TypeError: Transformer.forward() missing 1 required positional argument: 'hidden_states'`. The 4 transformer-only cells (clean × 2, aug-clean × 2) kept running. |
| 08:45 | `edd62b9` committed: bypass the DistilBERT `Transformer.forward` wrapper, iterate `self.transformer.layer` directly. Smoke-tested locally on Box A GPU 2 with a synthetic (B=2, T=100) input: shape correct, gradients flow. |
| 08:55 | Re-launched the 4 lm-bridge cells (Box A GPUs 2–5) and the 2 char-lm-aug cells (Box B). All 9 GPUs now active. |
| 09:00 → ~13:30 | Wave-3 main pass running. ETA ~5 h on Box A (parallel), ~10 h on Box B (sequential). |

## 7. Wave-3 cell matrix (10 cells, 9 GPUs)

| GPU | cell | features | hypothesis |
| --- | --- | --- | --- |
| A.0 | `clean` (eeg) | data filters only | "Does cleaning the data alone help?" |
| A.1 | `clean` (noise) | matched twin | — |
| A.2 | `lm-clean` (eeg) | data filters + LM-bridge | "Does LM bridge alone help (no aug confound)?" |
| A.3 | `lm-clean` (noise) | matched twin | — |
| A.4 | `lm-aug` (eeg) | data filters + LM-bridge + paraphrase 0.5 + 6 signal augs | **HEADLINE** — full stack |
| A.5 | `lm-aug` (noise) | matched twin | — |
| A.6 | `aug-clean` (eeg) | data filters + paraphrase + signal aug, **no LM bridge** | "Does aug help WITHOUT bridge? Controls A.4." |
| A.7 | `aug-clean` (noise) | matched twin | — |
| B.0 | `char-lm-aug` (eeg) | char vocab + LM-bridge + paraphrase + signal aug | "Does char vocab beat BPE-1k under the bridge?" |
| B.0 (queued) | `char-lm-aug` (noise) | matched twin (sequential after eeg cell) | — |

After all cells finish, `exp02 gap <eeg_key>` reports the matched-pair
gap on CER / WER / BLEU 1-4 / ROUGE-1-F / BERTScore-F1 across
greedy / beam / beam + KenLM decode modes.

## 8. Decision rule for the wave-3 verdict

Per Jo et al. 2024 §4.3 (and the `design.md` decision rule):

- **Strict PASS**: CER 95% bootstrap CIs disjoint AND sign-flip
  permutation `p < 0.01`, with EEG below noise — declare success on
  this fold, run 5-fold extension on the survivor.
- **Weak PASS**: CER mean below noise, CIs overlap — strong evidence
  but not strict; report and consider running aug variants.
- **TIE/FAIL** (where wave-1 ended): CER ≈ 1 for both — the
  architectural changes weren't enough; pivot to **D-SigLIP word-level
  contrastive alignment** ([d'Ascoli & King 2025
  *Nat. Commun.* 16, 10521](https://doi.org/10.1038/s41467-025-65499-0)),
  which is the only published recipe that has rigorously cleared
  the §4.3 bar on non-invasive EEG/MEG.

The four interventions in wave-3 are designed so that the controls
isolate the contribution of each. Reading the matrix:

- `clean` vs wave-1 baseline → contribution of data cleaning alone.
- `lm-clean` vs `clean` → contribution of LM bridge alone.
- `aug-clean` vs `clean` → contribution of paraphrase + signal aug alone.
- `lm-aug` vs `lm-clean` → marginal contribution of aug *given* LM bridge.
- `lm-aug` vs `aug-clean` → marginal contribution of LM bridge *given* aug.
- `char-lm-aug` vs `lm-aug` → contribution of char vocab over BPE-1k under bridge.

Whichever combination wins, we know which of the four root causes was
load-bearing.
