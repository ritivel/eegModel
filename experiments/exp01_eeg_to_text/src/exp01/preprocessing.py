"""exp01 preprocessing — re-exports from :mod:`eeg_common.preprocessing`.

All concrete pipeline definitions (``V2_REVE``, ``V2_TFM``, ``V2_DK25``,
``specaugment``, ``for_encoder``) live in the shared package so exp01 and
exp02 stay in sync. Call sites that did ``from . import preprocessing`` keep
working unchanged.
"""

from __future__ import annotations

from eeg_common.preprocessing import (
    PIPELINES,
    PreprocessSpec,
    V1_NOOP,
    V2_DK25,
    V2_REVE,
    V2_TFM,
    bandpass,
    common_average_reference,
    for_encoder,
    notch,
    resample_polyphase,
    specaugment,
    zscore_per_channel,
    zscore_per_recording,
)

__all__ = [
    "PIPELINES",
    "PreprocessSpec",
    "V1_NOOP",
    "V2_DK25",
    "V2_REVE",
    "V2_TFM",
    "bandpass",
    "common_average_reference",
    "for_encoder",
    "notch",
    "resample_polyphase",
    "specaugment",
    "zscore_per_channel",
    "zscore_per_recording",
]
