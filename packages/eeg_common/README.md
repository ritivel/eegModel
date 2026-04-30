# eeg-common

Shared building blocks for the `eegModel` experiments:

| module | what it does |
| --- | --- |
| `eeg_common.storage` | `Storage` dataclass — keyed-by-env-var filesystem paths (`hf/`, `splits/`, `runs/`, `eval/`, `wandb/`) |
| `eeg_common.preprocessing` | Per-row EEG pipelines — bandpass + notch + polyphase resample + per-recording z-score + clip; encoder-aware presets (`v2_reve`, `v2_tfm`, `v2_dk25`); SpecAugment |
| `eeg_common.splits` | Yin et al. (2024) unique-sentence + leave-N-subjects-out fold builder; persists to `<storage.splits>/fold_<n>.json` |
| `eeg_common.data` | HuggingFace dataset download + `EEGSentenceDataset` parquet streamer (8 sources: ZuCo v1+v2, DERCo, EMMT, eeg_sem_relev) + Jo et al. (2024) noise twin |
| `eeg_common.encoders` | Uniform encoder interface: `REVEEncoder`, `TFMEncoder`, `DIVER1Encoder`. Each exposes `encode(eeg, sr, channels) -> (B, T_seq, D)` and (where applicable) `tokenize` and `attach_lora` |

Both `experiments/exp01_eeg_to_text` (LM-bridge architecture) and `experiments/exp02_eeg_ctc` (single-stage CTC architecture) depend on this package.

The package itself is storage-root-agnostic: every function and class that touches the filesystem takes a `Storage` argument. Experiment packages bind their own `Storage` instance from their own env var (`EXP01_DATA_ROOT`, `EXP02_DATA_ROOT`).
