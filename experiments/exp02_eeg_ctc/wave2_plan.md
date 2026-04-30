# Exp02 wave-2 plan — augmentation cells

After wave-1 completes (~12-24h), if any §4.3 verdict is **TIE** or
**weak PASS**, wave-2 launches augmentation-enabled variants of the
surviving recipe. The augmentation suite is committed and tested — wave-1
just doesn't enable it. All toggles default OFF; wave-2 enables specific
combos via CLI flags.

## What's available (commit `0f5fd80`)

### Signal augmentations (`eeg_common.augment`, GPU-side per-step)

| augmentation | flag | recommended | citation |
|---|---|---|---|
| Time shift | `--signal-aug-time-shift-max-frac 0.05` | strong | [Brain Transformer 2025](https://www.nature.com/articles/s41598-025-86294-3) — most effective single augmentation for EEG |
| Channel dropout | `--signal-aug-channel-dropout-p 0.5 --signal-aug-channel-dropout-frac 0.1` | strong | [Strumiłło 2026](https://www.mdpi.com/1424-8220/26/4/1258) §"Channels Dropout" |
| Frequency mask | `--signal-aug-freq-mask-p 0.5 --signal-aug-freq-mask-max-hz 8` | medium | FFT-band masking (analogue of SpecAug's freq-mask) |
| Time warp | `--signal-aug-time-warp-p 0.3` | medium | [Xu 2026](https://www.mdpi.com/1424-8220/26/2/399) — segment stretch/telescope |
| Fourier surrogate | `--signal-aug-fourier-surrogate-p 0.2` | medium | Strumiłło 2026 — phase randomisation, preserves amplitude spectrum |
| Gaussian noise | `--signal-aug-gaussian-noise-sigma 0.05` | mild | additive on z-scored input |
| Feature mixup | `--signal-aug-mixup-alpha 0.4` | strong | [Alwasiti & Yusoff](https://pmc.ncbi.nlm.nih.gov/articles/PMC12501431/) — EEG mixup for classification, adapted for CTC sequence loss with weighted pair losses |

Pipeline order (in `augment.apply`):
`time_shift → time_warp → freq_mask → fourier_surrogate → channel_dropout → gaussian_noise`.

Mixup is applied at the post-encoder feature level (see `feature_mixup`).

### Text augmentation (`exp02.text_augment`, async OpenAI paraphrase)

| flag | recommended | citation |
|---|---|---|
| `--text-aug-prob 0.5` | strong | [Huang 2023, arXiv 2305.16333](https://arxiv.org/abs/2305.16333) — neural text augmentation for ASR: 9-15% relative WER improvement |
| `--text-aug-paraphrase-path .../paraphrases.parquet` | (auto) | defaults to `$EXP02_DATA_ROOT/text_aug/paraphrases.parquet` |

Each ZuCo training sentence has 5 paraphrases pre-generated via `gpt-4o-mini`
(commit `0f5fd80`'s `exp02 build-paraphrases --n-per-sentence 5`). At
training time, with probability `text_aug_prob`, the trainer substitutes a
random paraphrase as the CTC target. **Dev / eval always use the original
references** so metrics stay comparable across cells.

## Wave-2 launch matrix (run after wave-1 verdicts)

| cell tag | recipe | hypothesis |
|---|---|---|
| `aug-strong` | headline + `time_shift 0.05` + `channel_dropout 0.5/0.1` + `freq_mask 0.5/8Hz` + `mixup 0.4` + `text_aug 0.5` | "all the levers" — biggest expected lift |
| `aug-signal` | headline + the 4 signal augs (no text aug) | isolates signal-side gain |
| `aug-text` | headline + text_aug 0.5 only | isolates text-side gain |
| `aug-mixup` | headline + mixup 0.4 only | isolates mixup |
| `aug-fourier` | headline + fourier_surrogate 0.2 only | isolates Fourier surrogate |

Each cell trains as `reve.bpe1k.crctc.eeg.0` with the appropriate `--tag`.
Their matched noise twins use the same flags + `noise_train` input.

## Launch (when ready)

The augmentation matrix is **not** part of `exp02 pilot --group all` so it
doesn't auto-run. Launch each cell explicitly:

```bash
# Single cell on a free GPU
CUDA_VISIBLE_DEVICES=0 exp02 train reve.bpe1k.crctc.eeg.0 \
    --tag aug-strong \
    --signal-aug-time-shift-max-frac 0.05 \
    --signal-aug-channel-dropout-p 0.5 \
    --signal-aug-channel-dropout-frac 0.1 \
    --signal-aug-freq-mask-p 0.5 \
    --signal-aug-freq-mask-max-hz 8.0 \
    --signal-aug-mixup-alpha 0.4 \
    --text-aug-prob 0.5

# Then its matched §4.3 noise twin
CUDA_VISIBLE_DEVICES=1 exp02 train reve.bpe1k.crctc.noise_train.0 \
    --tag aug-strong \
    --signal-aug-time-shift-max-frac 0.05 \
    --signal-aug-channel-dropout-p 0.5 \
    --signal-aug-channel-dropout-frac 0.1 \
    --signal-aug-freq-mask-p 0.5 \
    --signal-aug-freq-mask-max-hz 8.0 \
    --signal-aug-mixup-alpha 0.4 \
    --text-aug-prob 0.5
```

`--tag aug-strong` makes the run live at
`$EXP02_DATA_ROOT/runs/reve_bpe1k_crctc_eeg_fold0_pp-v2_aug-strong/`,
distinct from the wave-1 baseline at
`$EXP02_DATA_ROOT/runs/reve_bpe1k_crctc_eeg_fold0_pp-v2/`.

## Decision rule for triggering wave-2

After wave-1 finishes:

```bash
exp02 gap reve.bpe1k.crctc.eeg.0      # headline matched-pair gap
exp02 gap reve.bpe1k.ctc.eeg.0        # vanilla CTC + label prior
exp02 gap reve.char.crctc.eeg.0       # char vocab
# ... etc
```

- **Strict PASS** (CER gap CI disjoint AND p<0.01): no wave-2 needed; run
  the 5-fold extension instead (`fold_extension_cells(survivor)`).
- **Weak PASS** (gap > 0, CIs overlap): launch wave-2 `aug-strong` to try
  to tighten the CIs.
- **TIE** (both close to 1, no gap): launch the full wave-2 matrix to
  identify which augmentation moves the needle.
- **FAIL** (CER ≈ 1 for both): wave-2 won't help; pivot to a different
  encoder (DIVER-1 if weights become available, or train a new EEG MAE
  encoder).
