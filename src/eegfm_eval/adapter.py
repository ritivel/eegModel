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
        return _load_labram_base(Path(checkpoint), device=device)
    if kind == "cbramod":
        return _load_cbramod(Path(checkpoint), device=device)
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
# labram_base — for reproduction tests
# ---------------------------------------------------------------------------


def _load_labram_base(checkpoint: Path, *, device: str) -> Encoder:
    """Load LaBraM-Base from `labram-base.pth` (HuggingFace).

    This is a thin port — we re-implement just enough of LaBraM's `labram_base_patch200_200`
    to load the official checkpoint and forward (B, C, T) → (B, D). See the
    upstream reference implementation at
    `https://github.com/935963004/LaBraM/blob/main/modeling_finetune.py`.

    NOTE: implementation lives in `_labram.py` to keep this dispatcher small.
    """
    from ._labram import LaBraMBase
    model = LaBraMBase()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[labram_base] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)

    spec = EncoderSpec(
        name="labram-base",
        d_features=200,                      # LaBraM-Base hidden dim
        sample_rate=200,
        window_samples=2000,                  # 10 s for TUAB; some tasks use 1000 (5 s for TUEV)
        expected_channels=None,               # LaBraM accepts any channel count
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining="labram_vqcodebook",
    )
    enc = Encoder(model, spec)

    def _forward(x: Tensor) -> Tensor:
        return enc.model(x)
    enc._forward = _forward                                   # type: ignore[method-assign]
    return enc


# ---------------------------------------------------------------------------
# cbramod — for reproduction tests
# ---------------------------------------------------------------------------


def _load_cbramod(checkpoint: Path, *, device: str) -> Encoder:
    """Load CBraMod from `pretrained_weights.pth` (HuggingFace).

    Same pattern as LaBraM. Implementation in `_cbramod.py`.
    """
    from ._cbramod import CBraMod
    model = CBraMod()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[cbramod] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)
    spec = EncoderSpec(
        name="cbramod",
        d_features=200,
        sample_rate=200,
        window_samples=800,                   # 4 s @ 200 Hz; tasks may need longer
        expected_channels=None,
        n_params=sum(p.numel() for p in model.parameters()),
        pretraining="cbramod_mae",
    )
    enc = Encoder(model, spec)

    def _forward(x: Tensor) -> Tensor:
        return enc.model(x)
    enc._forward = _forward                                   # type: ignore[method-assign]
    return enc
