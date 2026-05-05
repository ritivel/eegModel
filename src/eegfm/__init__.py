"""eegfm — self-supervised EEG pretraining.

Public API:

    from eegfm.storage import Storage, from_env
    from eegfm import preprocess, hbn, tuh, paradigms, train

Submodules:

    storage     Filesystem + S3 paths for run artifacts.
    preprocess  Minimum-offline pipeline (NaN + z-score + clip + window) and
                the literature-comparability variant (notch + bandpass +
                resample). See BEST_PRACTICES.md and the per-experiment
                Notion page for the rationale.
    hbn         HBN-EEG ingestion: list / download from `s3://fcp-indi/...`,
                read EEGLAB .set/.fdt via MNE, parse BIDS-iEEG metadata.
    tuh         TUH-EEG ingestion (TUAB v3.0.1 + TUEV v2.0.1): walk the
                rsync'd local tree, read EDF via MNE, parse `.rec` event
                annotations, derive Protocol A.4 labels.
    diffusion   Minimal cosine-schedule + epsilon-prediction loss for the
                MAR (G2) diffusion head (port of LTH14/mar's diffusion utilities).
    paradigms   Generative-paradigm heads: MAE (G0), AR (G1), MAR (G2),
                latent-prediction (G3). G2's diffusion head ports MAR's
                SimpleMLPAdaLN.
    train       Generic SSL pretraining loop (HF accelerate + wandb), shared
                across paradigms.
    cli         Typer CLI; `eegfm --help` for entrypoints.

Deliberately self-contained: no imports from `packages/eeg_common`. eeg_common
is shaped around fine-tuning frozen pretrained encoders (REVE / TFM / DIVER-1)
for downstream tasks, whereas eegfm trains foundation models from scratch
with a different preprocessing philosophy and eval suite.
"""

from __future__ import annotations

__version__ = "0.1.0"
