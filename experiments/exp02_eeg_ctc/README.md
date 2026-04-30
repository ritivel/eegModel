# Exp02 — single-stage CTC EEG-to-text

Direct ASR-style decoding of text from EEG: pretrained EEG foundation model
(REVE / TFM-Tokenizer / DIVER-1) + CTC head + (optional) CR-CTC consistency
regularisation + beam search + KenLM 4-gram rescore at decode time. **No
language model in the training loop**, so by construction the model can't
collapse onto an LM prior — anything it produces above the §4.3 noise floor
is attributable to EEG content.

The architecture deliberately strips the 3-stage curriculum from
`[exp01](../exp01_eeg_to_text/)` (`alignment → frozen-LM SFT → LoRA SFT`),
which was designed for a soft-prompt + frozen-Gemma bridge that needed a
modality-alignment phase. CTC has no LM in the loop, so a single end-to-end
training loop (Wav2Vec2 / Willett 2023 / Card 2024 standard) is the right
shape. See `[design.md](./design.md)` for the full citation trail and the
ablation matrix.

## Quick start

```bash
# 1. Install (Python ≥ 3.11; uv recommended).
cd eegModel
uv venv && source .venv/bin/activate
uv pip install -e packages/eeg_common
uv pip install -e experiments/exp02_eeg_ctc

# 2. Env vars.
export EXP02_DATA_ROOT=$HOME/data/exp02         # all artifacts land here
export HF_TOKEN=hf_xxx                            # gated repos
export WANDB_API_KEY=...                          # optional; runs offline if unset
export WANDB_PROJECT=exp02-eeg-ctc                # optional override

# 3. One-shot setup (run on a box with the dataset cached).
exp02 download-models                             # REVE + TFM into HF cache
exp02 download-data                               # ~72 GB unified dataset
exp02 make-splits                                 # 5-fold LNSO + Yin sentence partition
exp02 build-bpe --vocab-size 1024                 # train sentencepiece BPE-1k on ZuCo train refs + Wikipedia
exp02 build-kenlm --order 4                       # build a KenLM 4-gram on the BPE token stream

# 4. Smoke test (~3 min on H100).
exp02 smoke

# 5. Run a single cell.
exp02 train reve.bpe1k.crctc.eeg.0
exp02 eval  reve.bpe1k.crctc.eeg.0
```

## Layout

```
src/exp02/
  storage.py            STORAGE — bound to $EXP02_DATA_ROOT
  config.py             CTCConfig dataclass + run-matrix helpers
  chars.py              char vocab + sentencepiece BPE-1k loader
  tokenizer_build.py    one-shot BPE training (ZuCo + Wikipedia text)
  head.py               CTCHead + AED head + intermediate-CTC + label-prior
  model.py              EEG2CTC(encoder, head)
  train.py              SINGLE-STAGE end-to-end CTC trainer
  decode.py             greedy / beam / beam + KenLM rescore
  kenlm_build.py        one-shot KenLM 4-gram build
  eval.py               CER / WER / BLEU + bootstrap CIs + matched-pair §4.3
  cli.py                exp02 entry point
```

## Run matrix

The full Track-C scope is ~16 distinct training cells + 10 fold-extension
cells = ~26 cells, fitting on 9× H100 in roughly 24 h.

```
Headline:                reve.bpe1k.crctc.eeg.0          (default recipe)
                         reve.bpe1k.crctc.noise_train.0  (matched §4.3 noise twin)

Encoder ablation:        tfm.bpe1k.crctc.eeg.0
                         diver1.bpe1k.crctc.eeg.0        (if weights present)

Vocab ablation:          reve.char.crctc.eeg.0           (vs default BPE-1k)

CTC variant ablation:    reve.bpe1k.ctc.eeg.0            (vanilla, no CR-CTC)
                         reve.bpe1k.intctc.eeg.0         (intermediate CTC)
                         reve.bpe1k.ctcaed.eeg.0         (CTC + attention decoder hybrid)

Decode ablation:         per-cell: greedy | beam | beam + KenLM (eval-time, free)

Encoder-finetune ablation: reve.bpe1k.crctc.eeg.0_frozen (frozen REVE, head only)

5-fold extension:        survivor cell × {EEG, noise} × folds 0..4 (~10 cells)
```

Each `eeg` cell has a matched `noise_train` twin per the Jo et al. (2024)
§4.3 protocol; pairs are evaluated together via
`exp02.eval.matched_pair_gap`.

## What lands where (per cell)

```
$EXP02_DATA_ROOT/runs/<cell_id>/
  log.jsonl                     per-step ctc_loss, cr_ctc_kl, intermediate_loss,
                                grad_norm, lr; events
  sample_gens.jsonl             periodic dev (ref, hyp) snapshots during training
  stats.jsonl                   encoder feature stats (mean, std, abs_max)
  model.pt                      final checkpoint

$EXP02_DATA_ROOT/eval/<cell_id>/
  metrics.json                  summary mean / 95% CI for CER, WER, BLEU 1-4,
                                ROUGE-1-F, BERTScore-F1, plus per-decode-mode
                                breakdown (greedy / beam / beam_kenlm)
  predictions.parquet           one row per test example with metadata, ref,
                                hyp_greedy, hyp_beam, hyp_beam_kenlm
```

Plus shared artifacts under `$EXP02_DATA_ROOT/`:

```
$EXP02_DATA_ROOT/
  hf/                           HF cache (REVE, TFM, dataset parquets)
  splits/                       fold JSONs (shared with exp01 if you set
                                  EXP02_DATA_ROOT == EXP01_DATA_ROOT)
  bpe/spm.model                 sentencepiece BPE-1k
  bpe/spm.vocab
  kenlm/4gram.arpa              built KenLM model (text)
  kenlm/4gram.binary            built KenLM model (binary, fast)
  wandb/                        local wandb run dirs
```

## See also

- `[design.md](./design.md)` — design rationale + citations
- `[../exp01_eeg_to_text/](../exp01_eeg_to_text/)` — the LM-bridge experiment
- `[../../packages/eeg_common/](../../packages/eeg_common/)` — shared building blocks

