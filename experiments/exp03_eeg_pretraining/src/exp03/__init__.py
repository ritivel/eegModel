"""exp03 — self-supervised EEG pretraining.

Public API:

    from exp03.storage import Storage, from_env
    from exp03 import preprocess, hbn, tuh, paradigms, train

Submodules:

    storage     Filesystem + S3 paths for the experiment's artifacts.
    preprocess  Minimum-offline pipeline (NaN + z-score + clip + window) and
                the F0-prep literature-comparability variant (notch + bandpass
                + resample) per `mini_experiments.md` §4.1.
    hbn         HBN-EEG ingestion: list / download from `s3://fcp-indi/...`,
                read EEGLAB .set/.fdt via MNE, parse BIDS-iEEG metadata.
    tuh         TUH-EEG ingestion (TUAB v3.0.1 + TUEV v2.0.1): walk the
                rsync'd local tree, read EDF via MNE, parse `.rec` event
                annotations, derive Protocol A.4 labels.
    diffusion   Minimal cosine-schedule + epsilon-prediction loss for the
                MAR (G2) diffusion head (port of LTH14/mar's diffusion utilities).
    paradigms   Three generative-paradigm heads for exp17: MAE / AR / MAR.
                G2's diffusion head ports MAR's SimpleMLPAdaLN.
    train       Generic SSL pretraining loop (HF accelerate + wandb), shared
                across paradigms.
    cli         Typer CLI; `exp03 --help` for entrypoints.

Deliberately self-contained: no imports from `packages/eeg_common`. eeg_common
is shaped around fine-tuning frozen pretrained encoders (REVE / TFM / DIVER-1)
on the exp01 sentence dataset, whereas exp03 trains from scratch on HBN-EEG
with a different preprocessing philosophy and a different downstream eval
suite. Sharing modules between them would couple two experiments that are
better kept independent.
"""

from __future__ import annotations

__version__ = "0.1.0"
