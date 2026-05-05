"""Encoder adapter — one minimal interface, several concrete implementations.

Every downstream task needs to be able to:

    feats = encoder.encode(x)            # (B, C, T) → (B, D),  no grad
    feats = encoder.forward(x)           # (B, C, T) → (B, D),  with grad (for FT)
    encoder.expected_channels   list[str] | None    # `None` ⇒ iid single-channel
    encoder.sample_rate         int                 # Hz
    encoder.window_samples      int                 # samples
    encoder.d_features          int                 # output feature dim
    encoder.parameters()        Iterable[Tensor]    # for FT optimiser

Concrete loaders (factory functions, not subclasses):

    load_encoder("eegfm", checkpoint=...)         our pretrained model
    load_encoder("random_init", arch="mamba2_d256_l6")
    load_encoder("labram_base", checkpoint=...)   for reproduction tests
    load_encoder("cbramod", checkpoint=...)       for reproduction tests

If a task needs a channel set the encoder doesn't have, the task is
responsible for re-mapping (most external models expect the TUH 23-channel
order; our model is iid-single-channel so it accepts anything).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------


@dataclass
class EncoderSpec:
    """Static metadata about what input shape the encoder expects."""
    name: str                                 # human label, e.g. "eegfm-mae-mamba2-d256-l6"
    d_features: int                           # output feature dim
    sample_rate: int                          # Hz
    window_samples: int                       # int(window_s * sample_rate)
    expected_channels: list[str] | None       # None = iid single-channel; the encoder accepts (B, T) or (B, 1, T)
    n_params: int                             # total parameter count
    pretraining: str                          # short tag (e.g. "mae", "mar", "random_init", "labram")


class Encoder(nn.Module):
    """Wraps any pretrained EEG model behind a uniform `(B, C, T) → (B, D)` API."""

    def __init__(self, model: nn.Module, spec: EncoderSpec):
        super().__init__()
        self.model = model
        self.spec = spec

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """No grad — use for linear probe feature extraction."""
        self.model.eval()
        return self._forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """With grad — use for fine-tuning."""
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        """Subclass override — bridge the `(B, C, T)` external API to the model's
        internal API. Default: assume iid single-channel input → flatten and
        let the model handle it."""
        raise NotImplementedError("override in factory")

    def parameters(self, recurse: bool = True) -> Iterable[Tensor]:
        return self.model.parameters(recurse=recurse)


# ---------------------------------------------------------------------------
# Factory: load_encoder
# ---------------------------------------------------------------------------


EncoderKind = Literal[
    "eegfm",          # our own checkpoints
    "random_init",    # untrained eegfm (for floor probes)
    "labram_base",    # for reproduction tests
    "cbramod",
    "eegpt",
    "biot",
    "reve_base",
]


def load_encoder(kind: EncoderKind, *, checkpoint: Path | str | None = None,
                 arch: str | None = None, device: str = "cpu") -> Encoder:
    """Factory — dispatches to one of `_load_<kind>`. Returns a ready Encoder."""
    if kind == "eegfm":
        return _load_eegfm(Path(checkpoint), device=device)
    if kind == "random_init":
        return _load_random_init(arch or "mamba2_d256_l6", device=device)
    if kind == "labram_base":
        return _load_labram_base(Path(checkpoint) if checkpoint else None, device=device)
    if kind == "cbramod":
        return _load_cbramod(Path(checkpoint) if checkpoint else None, device=device)
    if kind == "biot":
        return _load_biot(Path(checkpoint) if checkpoint else None, device=device)
    if kind == "eegpt":
        return _load_eegpt(Path(checkpoint) if checkpoint else None, device=device)
    raise NotImplementedError(f"encoder {kind!r} not yet wired (see adapter.py to add)")


# ---------------------------------------------------------------------------
# eegfm — our own checkpoint
# ---------------------------------------------------------------------------


def _load_eegfm(checkpoint: Path, *, device: str) -> Encoder:
    """Load our `EEGSSLModel` from a pretraining checkpoint."""
    from eegfm.model import build_model, ModelConfig
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = raw.get("config", {})
    # Reconstruct the model config from the saved TrainConfig dict.
    mcfg = ModelConfig()                  # defaults; override the few fields we care about
    if "backbone_layers" in cfg_dict:
        mcfg.backbone.n_layers = int(cfg_dict["backbone_layers"])
    if "backbone_d_model" in cfg_dict:
        mcfg.backbone.d_model = int(cfg_dict["backbone_d_model"])
        mcfg.frontend.d_model = int(cfg_dict["backbone_d_model"])
        mcfg.decoder.d_model = int(cfg_dict["backbone_d_model"])
    model = build_model(mcfg)
    state_dict = raw.get("state_dict", raw)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eegfm] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)

    spec = EncoderSpec(
        name=f"eegfm-{cfg_dict.get('paradigm', 'unknown')}-mamba2-d{mcfg.backbone.d_model}-l{mcfg.backbone.n_layers}",
        d_features=mcfg.backbone.d_model,
        sample_rate=500,                 # our PIPELINE_MINIMAL is 500 Hz
        window_samples=int(cfg_dict.get("window_samples", 2000)),
        expected_channels=None,          # iid single-channel
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining=str(cfg_dict.get("paradigm", "unknown")),
    )

    enc = Encoder(model, spec)

    def _forward(x: Tensor) -> Tensor:
        # eegfm model expects (B, T) for single-channel; flatten (B, C, T) → (B*C, T)
        if x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B * C, T)
            feats = enc.model.encode_features(x)              # (B*C, D)
            feats = feats.reshape(B, C, -1).mean(dim=1)       # iid pool over channels
        else:
            feats = enc.model.encode_features(x)              # (B, T) → (B, D)
        return feats

    enc._forward = _forward                                   # type: ignore[method-assign]
    return enc


# ---------------------------------------------------------------------------
# random_init — for floor probes
# ---------------------------------------------------------------------------


def _load_random_init(arch: str, *, device: str) -> Encoder:
    """Build a random-init eegfm model. `arch` like 'mamba2_d256_l6' / 'transformer_d256_l6'."""
    from eegfm.model import build_model, ModelConfig
    mcfg = ModelConfig()
    # Parse arch string crudely
    if "mamba2" in arch:
        mcfg.backbone.kind = "mamba2"
    elif "transformer" in arch:
        mcfg.backbone.kind = "transformer"
    if "_d" in arch:
        d = int(arch.split("_d")[1].split("_")[0])
        mcfg.frontend.d_model = mcfg.backbone.d_model = mcfg.decoder.d_model = d
    if "_l" in arch:
        mcfg.backbone.n_layers = int(arch.split("_l")[1])
    model = build_model(mcfg).to(device)
    spec = EncoderSpec(
        name=f"random_init-{arch}",
        d_features=mcfg.backbone.d_model,
        sample_rate=500, window_samples=2000,
        expected_channels=None,
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining="random_init",
    )
    enc = Encoder(model, spec)

    def _forward(x: Tensor) -> Tensor:
        if x.ndim == 3:
            B, C, T = x.shape
            feats = enc.model.encode_features(x.reshape(B * C, T))
            feats = feats.reshape(B, C, -1).mean(dim=1)
        else:
            feats = enc.model.encode_features(x)
        return feats

    enc._forward = _forward                                   # type: ignore[method-assign]
    return enc


# ---------------------------------------------------------------------------
# Third-party EEG-FMs — used to validate our harness via reproduction tests.
#
# We don't hand-port these models. `braindecode` (BSD, pip-install) already
# ships clean PyTorch ports of LaBraM / CBraMod / BIOT / EEGPT with
# `from_pretrained` factories pointing at the canonical HuggingFace mirrors.
# Using a maintained scientific-Python library here is cleaner and lower-risk
# than copy-paste vendoring; the `pip install braindecode` dep is the cost.
# ---------------------------------------------------------------------------


def _load_labram_base(checkpoint: Path | None, *, device: str) -> Encoder:
    """Load LaBraM-Base via braindecode (`braindecode/labram-pretrained` HF mirror).

    `checkpoint=None` ⇒ pull from HF; pass an explicit local `.pth` to override.
    """
    from braindecode.models import Labram                    # type: ignore[import-untyped]
    if checkpoint is None or str(checkpoint) in ("hf", "auto"):
        model = Labram.from_pretrained("braindecode/labram-pretrained")
    else:
        model = Labram()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[labram_base] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)
    spec = EncoderSpec(
        name="labram-base",
        d_features=200,
        sample_rate=200,
        window_samples=2000,                  # 10 s for TUAB; tasks override (TUEV is 1000)
        expected_channels=None,               # LaBraM accepts any channel set via electrode names
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining="labram_vqcodebook",
    )
    enc = Encoder(model, spec)
    enc._forward = lambda x: enc.model(x)                    # type: ignore[method-assign]
    return enc


def _load_cbramod(checkpoint: Path | None, *, device: str) -> Encoder:
    """Load CBraMod via braindecode (or HF `weighting666/CBraMod`)."""
    from braindecode.models import CBraMod as _CBraMod       # type: ignore[import-untyped]
    if checkpoint is None or str(checkpoint) in ("hf", "auto"):
        try:
            model = _CBraMod.from_pretrained("weighting666/CBraMod")
        except Exception:                                    # noqa: BLE001
            # Fallback: download the .pth manually then load
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download("weighting666/CBraMod",
                                         "pretrained_weights/pretrained_weights.pth")
            model = _CBraMod()
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    else:
        model = _CBraMod()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
    model.to(device)
    spec = EncoderSpec(
        name="cbramod",
        d_features=200,
        sample_rate=200,
        window_samples=800,                  # 4 s @ 200 Hz; tasks override
        expected_channels=None,
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining="cbramod_mae",
    )
    enc = Encoder(model, spec)
    enc._forward = lambda x: enc.model(x)                    # type: ignore[method-assign]
    return enc


def _load_biot(checkpoint: Path | None, *, device: str) -> Encoder:
    """Load BIOT (six-dataset 18-channel) via braindecode."""
    from braindecode.models import BIOT                      # type: ignore[import-untyped]
    if checkpoint is None or str(checkpoint) in ("hf", "auto"):
        model = BIOT.from_pretrained("braindecode/biot-pretrained-six-datasets-18chs")
    else:
        model = BIOT(n_chans=18, n_times=2000, sfreq=200, hop_length=100)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
    model.to(device)
    spec = EncoderSpec(
        name="biot",
        d_features=256,
        sample_rate=200,
        window_samples=2000,
        expected_channels=[
            "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
            "C3-A2", "C4-A1",
        ],
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining="biot_six_datasets",
    )
    enc = Encoder(model, spec)
    enc._forward = lambda x: enc.model(x)                    # type: ignore[method-assign]
    return enc


def _load_eegpt(checkpoint: Path | None, *, device: str) -> Encoder:
    """Load EEGPT-Large (10M, 4-s @ 256Hz, 58-ch vocabulary) via braindecode."""
    from braindecode.models import EEGPT                     # type: ignore[import-untyped]
    if checkpoint is None or str(checkpoint) in ("hf", "auto"):
        model = EEGPT.from_pretrained("braindecode/eegpt-pretrained")
    else:
        model = EEGPT()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
    model.to(device)
    spec = EncoderSpec(
        name="eegpt",
        d_features=512,
        sample_rate=256,
        window_samples=1024,                  # 4 s @ 256 Hz
        expected_channels=None,               # 58-channel vocab; pass channel names per-task
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining="eegpt_mcae",
    )
    enc = Encoder(model, spec)
    enc._forward = lambda x: enc.model(x)                    # type: ignore[method-assign]
    return enc
