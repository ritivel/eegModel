# exp03 — Self-supervised EEG pretraining (iid-channel single-channel recipe)

> **Status:** scaffolding + Phase-0 data acquisition (mini-experiment 01 sanity baselines next).
>
> **What this is:** a from-scratch SSL pretraining recipe for EEG, where each
> `(subject, channel, 4-second window)` triple is one independent training
> example. Pretrains on **HBN-EEG** (Healthy Brain Network EEG, 3000+ pediatric
> /young-adult subjects, 128-channel HydroCel @ 500 Hz native, openly distributed
> on AWS public storage `s3://fcp-indi/data/Projects/HBN/EEG/`); evaluates by
> frozen-probing on **HBN ADHD-binary AUROC + HBN 6-task classification BAC/WF1
> + k-NN top-1**, with **TUAB + TUEV** as a literature-comparable secondary
> when TUH NEDC access lands.

## Read these first

- [`mini_experiments.md`](./mini_experiments.md) — master spec (§4.1 corpus,
  §4.3 eval suite, §4.4 matched-noise twin discipline, §4.5 statistical
  protocol). The corrected preprocessing philosophy ("minimum offline,
  maximum in-model") lives in §4.1.
- [`methodology.md`](./methodology.md) — the playbook (Phase 0–4 framework,
  monitors, EEG-specific gotchas).
- [`mini_experiments_explainer.pdf`](./mini_experiments_explainer.pdf) — the
  human-readable narrative version of the above two docs (rendered from
  `mini_experiments_explainer.typ` via `typst compile`).
- Per-experiment design docs:
  [`mini_experiments/01..16_*/README.md`](./mini_experiments/).

## Code layout (this folder is self-contained — no `eeg_common` imports)

```
experiments/exp03_eeg_pretraining/
├── pyproject.toml                  # exp03 package: name=exp03-eeg-pretraining
├── README.md                       # this file
├── methodology.md                  # the playbook
├── mini_experiments.md             # master spec
├── mini_experiments_explainer.pdf  # rendered explainer
├── mini_experiments_explainer.typ  # typst source
├── mini_experiments/               # 16 per-experiment design docs (planned + READMEs)
└── src/exp03/
    ├── __init__.py                 # package version, public API surface
    ├── storage.py                  # filesystem + S3 paths (rooted at $EXP03_DATA_ROOT)
    ├── preprocess.py               # SPEC_MINIMAL + SPEC_V2_CLEAN pipelines; window + iid expand;
    │                               # parquet writer with a fixed pyarrow schema
    ├── hbn.py                      # HBN-EEG ingestion: list FCP-INDI bucket,
    │                               # download .set/.fdt + sidecars + participants.tsv,
    │                               # MNE-based loader, ADHD-binary label derivation
    └── cli.py                      # `exp03 ...` entrypoints (typer)
```

**Why self-contained.** `packages/eeg_common` is shaped around fine-tuning
*frozen pretrained encoders* (REVE / TFM / DIVER-1) on the exp01 sentence
dataset, with a "canonical V2 preprocessing" philosophy that matches each
encoder's expected input. exp03 trains from scratch on HBN-EEG with a
deliberately *different* preprocessing philosophy ("minimum offline, maximum
in-model" per `mini_experiments.md` §4.1) and a different downstream eval
suite. Sharing modules between the two experiments would couple decisions
that are better kept independent.

## Install

```bash
# from repo root, inside the exp03 venv:
uv pip install -e experiments/exp03_eeg_pretraining
# verify:
exp03 --help
exp03 paths
```

## Quick start (Tier 1 — sanity-baselines slice, ~1 GB raw)

```bash
# 0. Set the data root (NVMe scratch on the GPU box)
export EXP03_DATA_ROOT=/opt/dlami/nvme/eeg

# 1. Discover what's available on FCP-INDI's public bucket
exp03 list-releases
exp03 list-subjects NC --max 5

# 2. Pull 5 subjects from release NC (~5 × 220 MB = ~1 GB raw)
exp03 download NC --max-subjects 5

# 3. Phase-0 data audit (Karpathy step 1: become one with the data)
exp03 audit NC

# 4. Apply the minimum-offline preprocessing → parquet shards
exp03 preprocess NC --pipeline minimal

# 5. (optional, only for exp02 F0-prep literature-comparability cell)
exp03 preprocess NC --pipeline v2_clean

# 6. Mirror the preprocessed shards to the warehouse so future GPU
#    boxes can pull instead of re-preprocessing
exp03 sync-derived-up --pipeline minimal
```

For the Tier-2 100-hour subset (~200 subjects, ~50 GB raw, ~25 GB
preprocessed parquet), bump `--max-subjects 200` and run overnight.

## Preprocessing philosophy in one paragraph

The offline pipeline is **deliberately minimal**: NaN sanitation +
per-channel z-score + ±5σ clip + 4-second non-overlapping windowing + iid-
channel expansion + float16 parquet shard write. **No notch, no bandpass,
no resampling.** Notch and bandpass are hypotheses tested by exp02 (the F2
SincNet variant learns Hz-parameterised cutoffs end-to-end, F3 frozen
scattering provides a fixed wavelet basis, F4 complex Gabor a Hz-spaced
complex bandpass). Resampling is the question of exp05 (multi-rate strategy)
and exp14 (context-length scaling at 2 kHz × 30 s = 60k samples). Doing
any of these offline pre-decides those experiments. A separate
`v2_clean` pipeline (60 Hz notch + 0.5–100 Hz Butterworth bandpass +
500 → 250 Hz polyphase resample) exists only for the exp02 F0-prep
literature-comparability cell; it is never used by the §4.4 winner-picker.

## Expected first results

After running the Tier-1 pipeline above, you should see:

```
$ ls /opt/dlami/nvme/eeg/derived/hbn_minimal_500hz/
sub-NDARxxxx/   sub-NDARyyyy/   ...
$ ls /opt/dlami/nvme/eeg/derived/hbn_minimal_500hz/sub-NDARxxxx/
task-RestingState.parquet            # ~25 MB
task-SequenceLearning.parquet
task-SymbolSearch.parquet
task-SurroundSuppression.parquet
task-ContrastChangeDetection.parquet
task-Video1.parquet
```

Each parquet shard has the schema documented in `preprocess.py`'s
docstring, with one row per `(channel, 4-sec window)` after iid expansion.
This is the input format every later mini-experiment reads.
