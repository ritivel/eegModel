"""exp01 encoders — thin wrappers around :mod:`eeg_common.encoders` that
bind the exp01 storage so callers don't have to pass it explicitly.

Re-exports the encoder classes and the ``ensure_tfm_source`` / ``load_encoder``
helpers; existing call-sites keep working unchanged.
"""

from __future__ import annotations

from pathlib import Path

from eeg_common import encoders as _encoders
from eeg_common.encoders import (
    DIVER1Encoder as _DIVER1Encoder,
    EEGEncoder,
    EncoderSpec,
    REVEEncoder as _REVEEncoder,
    TFMEncoder as _TFMEncoder,
)

from . import storage


__all__ = [
    "DIVER1Encoder",
    "EEGEncoder",
    "EncoderSpec",
    "REVEEncoder",
    "TFMEncoder",
    "ensure_tfm_source",
    "load_encoder",
]


class REVEEncoder(_REVEEncoder):
    def __init__(self, model_id: str = "brain-bzh/reve-base"):
        super().__init__(storage.STORAGE, model_id=model_id)


class TFMEncoder(_TFMEncoder):
    def __init__(self):
        super().__init__(storage.STORAGE)


class DIVER1Encoder(_DIVER1Encoder):
    def __init__(self):
        super().__init__(storage.STORAGE)


def ensure_tfm_source() -> Path:
    return _encoders.ensure_tfm_source(storage.STORAGE)


def load_encoder(name: str) -> EEGEncoder:
    if name == "reve":
        return REVEEncoder()
    if name == "tfm":
        return TFMEncoder()
    if name == "diver1":
        return DIVER1Encoder()
    raise ValueError(f"unknown encoder: {name}")
