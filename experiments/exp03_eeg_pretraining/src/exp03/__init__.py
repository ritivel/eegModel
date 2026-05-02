"""exp03 — self-supervised EEG pretraining.

Public API:

    from exp03.storage import Storage, from_env
    from exp03 import preprocess, hbn

Submodules:

    storage     Filesystem + S3 paths for the experiment's artifacts.
    preprocess  Minimum-offline pipeline (NaN + z-score + clip + window) and
                the F0-prep literature-comparability variant (notch + bandpass
                + resample) per `mini_experiments.md` §4.1.
    hbn         HBN-EEG ingestion: list / download from `s3://fcp-indi/...`,
                read EEGLAB .set/.fdt via MNE, parse BIDS-iEEG metadata.
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
