"""EEG SSL model: configurable Frontend × Backbone × Paradigm scaffold.

This file defines the architecture for `exp03` self-supervised pretraining.
Per `mini_experiments.md` §4.2, **the only concrete combination implemented
right now** is the §4.2 default (Conv3 frontend → bidirectional Mamba-2
encoder → 2-layer Mamba-2 MAE decoder, with random 50% patch masking and
raw-signal targets). Every other slot is plumbed via the `ModelConfig`
schema but raises `NotImplementedError` with a pointer to the
mini-experiment that will fill it in. This is the explicit "configurable
scaffold from day 1" decision (2026-05-03 chat) so that exp02 (frontend
ablation), exp03 (backbone ablation), exp17 (generative paradigm), exp18
(reconstruction target), exp19 (decoder design), and exp20 (position
embedding) can each fill in *one slot* without touching the rest.

§4.2 default in one sentence:

    `(B, T_samples) raw EEG`
        ── Conv3Frontend (k=7,7,7 s=2,2,2 GeLU) ─→ `(B, T', d_model)`
        ── add SinusoidalPosEmb ─────────────────→ `(B, T', d_model)`
        ── RandomPatchMask (50%, asymmetric) ────→ visible_idx, masked_idx
        ── BidiMamba2 encoder (6 layers d=256) ──→ `(B, n_visible, d_model)`
        ── insert mask tokens + decoder pos emb ─→ `(B, T', d_model)`
        ── Mamba2 decoder (2 layers d=256) ──────→ `(B, T', d_model)`
        ── Linear → patch reconstruction ────────→ `(B, T', patch_samples)`
        ── reshape ──────────────────────────────→ `(B, T_samples)`
        ── L1 + 0.3·MR-STFT(log-mag) on masked positions only

`build_model(ModelConfig)` is the single public entrypoint. Everything
else in this file is implementation detail that gets composed via the
`*_REGISTRY` dicts. Adding a new frontend / backbone / paradigm = adding
one entry to the right registry; nothing else in this file changes.

Library notes:
    - Mamba-2: official upstream block from `mamba_ssm` (state-spaces/mamba)
      so we can compose forward + reverse-on-reversed-input bidirectionally
      (the FEMBA pattern). HF's `Mamba2Model` wraps this with language-model
      machinery (tokenizer, causal LM head) we'd just have to disable.
    - Transformer fallback: `torch.nn.functional.scaled_dot_product_attention`
      (FlashAttention-2 path on Ampere+) — pure stdlib, no extra deps,
      useful when mamba_ssm install is broken or for sm_<8 GPUs.
    - Positional embedding: hand-rolled (~10 lines per kind); HF has
      `RotaryEmbedding` etc. but their plumbing assumes a specific cache
      pattern that's awkward when mixing Mamba and attention layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# =============================================================================
# Configs
# =============================================================================
#
# All configs are frozen dataclasses so they can be hashed (for run-id
# generation) and JSON-serialised (for wandb / results.md provenance).
# The `kind` literal on each one is the dispatch key into the corresponding
# registry; the rest of the fields are kind-specific (e.g. `n_heads` only
# matters for `kind="transformer"`).
# =============================================================================


FrontendKind = Literal["conv3", "sincnet", "leaf", "complex_gabor", "scattering"]
BackboneKind = Literal["transformer", "mamba2", "lru", "hybrid_mamba_attn", "fgno"]
DecoderKind = Literal["mamba2", "transformer", "unet_samba"]
MaskKind = Literal["random_patch", "wav2vec_span", "multi_block", "tube"]
PosEmbKind = Literal["sinusoidal", "learned", "rope", "nope", "fourier_4d"]
ParadigmKind = Literal["mae", "ar", "mar", "jepa"]
TargetKind = Literal["raw", "raw_per_patch_norm", "latent_jepa", "fsq_codec", "kmeans_cluster"]


@dataclass(frozen=True)
class FrontendConfig:
    kind: FrontendKind = "conv3"
    d_model: int = 256
    in_channels: int = 1                       # iid single-channel
    kernel_sizes: tuple[int, ...] = (7, 7, 7)
    strides: tuple[int, ...] = (2, 2, 2)
    activation: Literal["gelu", "relu", "snake"] = "gelu"


@dataclass(frozen=True)
class BackboneConfig:
    kind: BackboneKind = "mamba2"
    n_layers: int = 6
    d_model: int = 256
    bidirectional: bool = True
    # mamba2 specific
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 256
    rmsnorm: bool = True
    # transformer specific
    n_heads: int = 4
    rope: bool = True
    dropout: float = 0.0


@dataclass(frozen=True)
class DecoderConfig:
    kind: DecoderKind = "mamba2"
    n_layers: int = 2                          # MAE 2022 finding: 1 ≈ 8 layers for vision linear probe
    d_model: int = 256
    bidirectional: bool = True
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 256
    rmsnorm: bool = True


@dataclass(frozen=True)
class MaskConfig:
    kind: MaskKind = "random_patch"
    mask_ratio: float = 0.50                   # MAE default; alt values explored in exp10
    span_length: int = 4                       # only used by wav2vec_span / multi_block


@dataclass(frozen=True)
class PosEmbConfig:
    kind: PosEmbKind = "sinusoidal"
    max_len: int = 2048


@dataclass(frozen=True)
class ParadigmConfig:
    kind: ParadigmKind = "mae"
    causal: bool = False                       # only meaningful for ar
    diffusion_head: bool = False               # only meaningful for mar


@dataclass(frozen=True)
class TargetConfig:
    kind: TargetKind = "raw"
    fsq_vocab_size: int = 36000                # only used by fsq_codec target


@dataclass(frozen=True)
class ModelConfig:
    """Top-level config; pass to `build_model(cfg)`."""

    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    pos_emb: PosEmbConfig = field(default_factory=PosEmbConfig)
    paradigm: ParadigmConfig = field(default_factory=ParadigmConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    # Window length in raw samples. 2000 = 4 s @ 500 Hz (HBN minimal pipeline).
    window_samples: int = 2000

    @property
    def total_stride(self) -> int:
        s = 1
        for stride in self.frontend.strides:
            s *= stride
        return s

    @property
    def n_tokens(self) -> int:
        return self.window_samples // self.total_stride

    @property
    def patch_samples(self) -> int:
        return self.total_stride

    def __post_init__(self):
        if self.window_samples % self.total_stride != 0:
            raise ValueError(
                f"window_samples ({self.window_samples}) must be divisible by "
                f"total frontend stride ({self.total_stride})"
            )
        if self.frontend.d_model != self.backbone.d_model:
            raise ValueError(
                f"frontend.d_model ({self.frontend.d_model}) must match "
                f"backbone.d_model ({self.backbone.d_model})"
            )
        if self.backbone.d_model != self.decoder.d_model:
            raise ValueError(
                f"backbone.d_model ({self.backbone.d_model}) must match "
                f"decoder.d_model ({self.decoder.d_model})"
            )


# =============================================================================
# Frontend (one variant concrete; others stubbed to a registry of NotImpl)
# =============================================================================


class Conv3Frontend(nn.Module):
    """3-layer 1-D conv front-end (the §4.2 default).

    Maps `(B, in_channels=1, T=window_samples)` → `(B, T', d_model)` where
    T' = window_samples / prod(strides). With the default (kernels 7/7/7,
    strides 2/2/2), the total stride is 8 → T'=2000/8=250 tokens for a
    4-second window at 500 Hz.

    Per `02_frontend_ablation/README.md`, this is "the dumbest reasonable
    baseline" — anti-aliasing and learned cutoffs are exp02 hypotheses.
    """

    def __init__(self, cfg: FrontendConfig):
        super().__init__()
        self.cfg = cfg
        layers = []
        in_ch = cfg.in_channels
        # Out-channel schedule: ramp from in_ch to d_model in equal steps so
        # the first layer doesn't have to do all the channel work.
        out_chs = []
        for i in range(len(cfg.kernel_sizes)):
            # Last layer hits exactly d_model.
            out_ch = cfg.d_model if i == len(cfg.kernel_sizes) - 1 else max(
                cfg.d_model // (2 ** (len(cfg.kernel_sizes) - 1 - i)), in_ch
            )
            out_chs.append(out_ch)

        for k, s, out_ch in zip(cfg.kernel_sizes, cfg.strides, out_chs):
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=k, stride=s,
                    padding=k // 2,
                    bias=True,
                )
            )
            if cfg.activation == "gelu":
                layers.append(nn.GELU())
            elif cfg.activation == "relu":
                layers.append(nn.ReLU())
            elif cfg.activation == "snake":
                raise NotImplementedError(
                    "Snake activation deferred — see "
                    "12_quick_wins_consolidation/README.md (W1 Snake was dropped"
                    " 2026-05-03 because it empirically defaults to near-linear)."
                )
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T_samples)  -> (B, T', d_model)
        h = self.net(x)                           # (B, d_model, T')
        return rearrange(h, "b d t -> b t d")


def _frontend_not_implemented(kind: str, mini_exp: str) -> Callable[..., nn.Module]:
    """Factory of placeholder constructors that raise on instantiation."""
    def _ctor(cfg: FrontendConfig):
        raise NotImplementedError(
            f"Frontend kind={kind!r} not implemented yet — see "
            f"experiments/exp03_eeg_pretraining/mini_experiments/{mini_exp}/README.md"
        )
    return _ctor


FRONTEND_REGISTRY: dict[str, Callable[[FrontendConfig], nn.Module]] = {
    "conv3":           lambda cfg: Conv3Frontend(cfg),
    "sincnet":         _frontend_not_implemented("sincnet",         "02_frontend_ablation"),
    "leaf":            _frontend_not_implemented("leaf",            "02_frontend_ablation"),
    "complex_gabor":   _frontend_not_implemented("complex_gabor",   "02_frontend_ablation"),
    "scattering":      _frontend_not_implemented("scattering",      "02_frontend_ablation"),
}


def build_frontend(cfg: FrontendConfig) -> nn.Module:
    return FRONTEND_REGISTRY[cfg.kind](cfg)


# =============================================================================
# Position embedding
# =============================================================================


class SinusoidalPosEmb(nn.Module):
    """Standard transformer sinusoidal position embedding.

    Buffer-only (no learnable params). Adds to a `(B, T, D)` input.
    """

    def __init__(self, cfg: PosEmbConfig, d_model: int):
        super().__init__()
        pe = torch.zeros(cfg.max_len, d_model)
        position = torch.arange(0, cfg.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[: x.size(1)]


class LearnedPosEmb(nn.Module):
    """Learnable absolute positional embedding."""

    def __init__(self, cfg: PosEmbConfig, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, cfg.max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NoPE(nn.Module):
    """No position embedding (identity). Mamba-2's scan is positional in itself."""

    def __init__(self, cfg: PosEmbConfig, d_model: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _pos_emb_not_implemented(kind: str, mini_exp: str) -> Callable[..., nn.Module]:
    def _ctor(cfg: PosEmbConfig, d_model: int):
        raise NotImplementedError(
            f"PosEmb kind={kind!r} not implemented yet — see "
            f"experiments/exp03_eeg_pretraining/mini_experiments/{mini_exp}/README.md"
        )
    return _ctor


POS_EMB_REGISTRY: dict[str, Callable[[PosEmbConfig, int], nn.Module]] = {
    "sinusoidal":  lambda cfg, d_model: SinusoidalPosEmb(cfg, d_model),
    "learned":     lambda cfg, d_model: LearnedPosEmb(cfg, d_model),
    "nope":        lambda cfg, d_model: NoPE(cfg, d_model),
    "rope":        _pos_emb_not_implemented("rope",        "20_position_embedding"),
    "fourier_4d":  _pos_emb_not_implemented("fourier_4d",  "20_position_embedding"),
}


def build_pos_emb(cfg: PosEmbConfig, d_model: int) -> nn.Module:
    return POS_EMB_REGISTRY[cfg.kind](cfg, d_model)


# =============================================================================
# Backbone: bidirectional Mamba-2 (concrete) + Transformer fallback
# =============================================================================


class _Mamba2Block(nn.Module):
    """Single Mamba-2 layer with pre-norm residual.

    Wraps the upstream `mamba_ssm.Mamba2` block in a residual:

        x → norm → Mamba2 → + → x

    The unidirectional version. The bidirectional wrapper below stacks
    forward + reverse copies and sums them.
    """

    def __init__(self, cfg: BackboneConfig | DecoderConfig):
        super().__init__()
        from mamba_ssm import Mamba2          # lazy import — only loaded when used

        self.norm = nn.RMSNorm(cfg.d_model) if cfg.rmsnorm else nn.LayerNorm(cfg.d_model)
        self.mixer = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            headdim=cfg.headdim,
            chunk_size=cfg.chunk_size,
            rmsnorm=cfg.rmsnorm,
            use_mem_eff_path=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.mixer(self.norm(x))


class BidiMamba2Stack(nn.Module):
    """`n_layers` of bidirectional Mamba-2 (FEMBA-style fwd + rev sum).

    For each layer, we compute Mamba2(x) + Mamba2(flip(x))[reverted] in
    parallel and sum the outputs. Equivalent to summing two unidirectional
    Mamba-2 layers, doubling the per-layer FLOPs but giving us bidirectional
    context — required for the masked-reconstruction objective where future
    context should help reconstruct masked positions.

    Uni-directional path is selected when `cfg.bidirectional=False`.
    """

    def __init__(self, cfg: BackboneConfig | DecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.bidirectional = cfg.bidirectional
        self.fwd_layers = nn.ModuleList(_Mamba2Block(cfg) for _ in range(cfg.n_layers))
        if self.bidirectional:
            self.rev_layers = nn.ModuleList(_Mamba2Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.RMSNorm(cfg.d_model) if cfg.rmsnorm else nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if not self.bidirectional:
            for layer in self.fwd_layers:
                x = layer(x)
            return self.final_norm(x)

        h_fwd = x
        h_rev = torch.flip(x, dims=[1])
        for f_layer, r_layer in zip(self.fwd_layers, self.rev_layers):
            h_fwd = f_layer(h_fwd)
            h_rev = r_layer(h_rev)
        h_rev = torch.flip(h_rev, dims=[1])
        return self.final_norm(h_fwd + h_rev)


class TransformerStack(nn.Module):
    """Bidirectional Transformer fallback — stdlib only, no mamba_ssm dep.

    Each layer is pre-norm + multihead self-attn (bidirectional) + pre-norm
    + GLU feed-forward, RMSNorm. Useful when mamba_ssm is unavailable
    (e.g., on a CPU/MPS dev box) or as the B0 cell in exp03 backbone
    ablation. Uses `F.scaled_dot_product_attention` so FlashAttention-2
    kicks in automatically on Ampere+.

    Note: the §4.2 default is Mamba-2; this is a fallback / future-exp03 cell,
    not the headline path. If you select kind="transformer" you must still
    set `n_heads`, `rope` etc. on `BackboneConfig` (their defaults are sane).
    """

    def __init__(self, cfg: BackboneConfig | DecoderConfig):
        super().__init__()
        if not isinstance(cfg, BackboneConfig):
            raise NotImplementedError("Transformer decoder via this path not yet wired")

        d_model = cfg.d_model
        n_heads = cfg.n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.cfg = cfg

        # Each "layer" = norm-attn-residual + norm-MLP-residual.
        # We hand-roll instead of nn.TransformerEncoderLayer because we want
        # bf16-friendly, dropout=0 default, and RMSNorm rather than LayerNorm.
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.RMSNorm(d_model) if cfg.rmsnorm else nn.LayerNorm(d_model),
                "qkv": nn.Linear(d_model, 3 * d_model, bias=False),
                "out_proj": nn.Linear(d_model, d_model, bias=False),
                "norm2": nn.RMSNorm(d_model) if cfg.rmsnorm else nn.LayerNorm(d_model),
                "mlp_in": nn.Linear(d_model, 4 * d_model, bias=False),
                "mlp_out": nn.Linear(4 * d_model, d_model, bias=False),
            }))
        self.final_norm = nn.RMSNorm(d_model) if cfg.rmsnorm else nn.LayerNorm(d_model)

    def _attn(self, x: torch.Tensor, layer: nn.ModuleDict) -> torch.Tensor:
        B, T, D = x.shape
        n_heads = self.cfg.n_heads
        head_dim = D // n_heads
        qkv = layer["qkv"](x)                        # (B, T, 3D)
        qkv = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each (B, H, T, head_dim)
        # Bidirectional (no causal mask).
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = rearrange(attn_out, "b h t d -> b t (h d)")
        return layer["out_proj"](attn_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        for layer in self.layers:
            x = x + self._attn(layer["norm1"](x), layer)
            h = layer["norm2"](x)
            h = layer["mlp_out"](F.gelu(layer["mlp_in"](h)))
            x = x + h
        return self.final_norm(x)


def _backbone_not_implemented(kind: str, mini_exp: str) -> Callable[..., nn.Module]:
    def _ctor(cfg):
        raise NotImplementedError(
            f"Backbone kind={kind!r} not implemented yet — see "
            f"experiments/exp03_eeg_pretraining/mini_experiments/{mini_exp}/README.md"
        )
    return _ctor


BACKBONE_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mamba2":             lambda cfg: BidiMamba2Stack(cfg),
    "transformer":        lambda cfg: TransformerStack(cfg),
    "lru":                _backbone_not_implemented("lru",                 "03_backbone_ablation"),
    "hybrid_mamba_attn":  _backbone_not_implemented("hybrid_mamba_attn",   "03_backbone_ablation"),
    "fgno":               _backbone_not_implemented("fgno",                "03_backbone_ablation"),
}


def build_backbone(cfg: BackboneConfig) -> nn.Module:
    return BACKBONE_REGISTRY[cfg.kind](cfg)


# =============================================================================
# Decoder (MAE asymmetric decoder; 1-2 layers usually enough)
# =============================================================================


class Mamba2Decoder(nn.Module):
    """Small Mamba-2 stack used as the MAE decoder.

    Identical structure to the encoder backbone but typically much shallower
    (2 layers vs 6) — MAE 2022 found 1≈8 layers for vision linear probe;
    biosignal-specific verdict is exp19's question. We make `n_layers`
    settable and default to 2 per `mini_experiments.md` §4.2.
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.stack = BidiMamba2Stack(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


def _decoder_not_implemented(kind: str, mini_exp: str) -> Callable[..., nn.Module]:
    def _ctor(cfg):
        raise NotImplementedError(
            f"Decoder kind={kind!r} not implemented yet — see "
            f"experiments/exp03_eeg_pretraining/mini_experiments/{mini_exp}/README.md"
        )
    return _ctor


DECODER_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mamba2":      lambda cfg: Mamba2Decoder(cfg),
    "transformer": _decoder_not_implemented("transformer", "19_decoder_design"),
    "unet_samba":  _decoder_not_implemented("unet_samba",  "19_decoder_design"),
}


def build_decoder(cfg: DecoderConfig) -> nn.Module:
    return DECODER_REGISTRY[cfg.kind](cfg)


# =============================================================================
# Masking
# =============================================================================
#
# The MAE pattern: emit two index tensors per batch — the visible tokens
# (which the encoder sees) and the masked tokens (which the encoder must
# reconstruct via the decoder's mask tokens). The encoder operates on
# (B, n_visible, D); the decoder reassembles (B, T, D) by inserting a
# learned mask-token vector at masked positions, applying the inverse
# permutation to restore the original token order, and then adding the
# decoder's positional embedding.


@dataclass
class MaskOutput:
    """Container returned by mask modules.

    Fields:
        ids_keep:    (B, n_visible) long indices into the original T tokens
        ids_restore: (B, T)        long indices to undo the random shuffle
        mask:        (B, T)        float32 mask, 1.0 at masked positions, 0.0 at visible
    """
    ids_keep: torch.Tensor
    ids_restore: torch.Tensor
    mask: torch.Tensor


class RandomPatchMask(nn.Module):
    """Standard MAE per-sample random shuffle + first-`n_keep` keep.

    Generates a different mask per batch element (per the MAE paper) so the
    encoder can't memorise positional priors. Shuffles by per-sample noise,
    keeps the first (1-mask_ratio) fraction.

    This implementation matches MAE 2022 §3.2 exactly.
    """

    def __init__(self, cfg: MaskConfig):
        super().__init__()
        self.cfg = cfg
        if not 0.0 < cfg.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {cfg.mask_ratio}")

    def forward(self, B: int, T: int, device: torch.device) -> MaskOutput:
        n_keep = int(T * (1.0 - self.cfg.mask_ratio))
        noise = torch.rand(B, T, device=device)               # uniform
        ids_shuffle = torch.argsort(noise, dim=1)             # ascending
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]                    # first n_keep are visible

        mask = torch.ones(B, T, device=device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)    # 0 at visible, 1 at masked
        return MaskOutput(ids_keep=ids_keep, ids_restore=ids_restore, mask=mask)


def _mask_not_implemented(kind: str, mini_exp: str) -> Callable[..., nn.Module]:
    def _ctor(cfg):
        raise NotImplementedError(
            f"Mask kind={kind!r} not implemented yet — see "
            f"experiments/exp03_eeg_pretraining/mini_experiments/{mini_exp}/README.md"
        )
    return _ctor


MASK_REGISTRY: dict[str, Callable[[MaskConfig], nn.Module]] = {
    "random_patch":  lambda cfg: RandomPatchMask(cfg),
    "wav2vec_span":  _mask_not_implemented("wav2vec_span", "10_masking_strategy"),
    "multi_block":   _mask_not_implemented("multi_block",  "10_masking_strategy"),
    "tube":          _mask_not_implemented("tube",         "10_masking_strategy"),
}


def build_mask(cfg: MaskConfig) -> nn.Module:
    return MASK_REGISTRY[cfg.kind](cfg)


# =============================================================================
# The full SSL model
# =============================================================================


class EEGSSLModel(nn.Module):
    """Composes Frontend → Encoder (with pos emb + masking) → Decoder → reconstruction head.

    The §4.2 default forward flow on a `(B, window_samples)` input:

        x: (B, T_samples)
            └── unsqueeze(1) → (B, 1, T)
            └── frontend → (B, T_tokens, D)
            └── + encoder_pos_emb
            └── mask_module(B, T_tokens) → ids_keep, ids_restore, mask
            └── gather_visible → (B, n_visible, D)
            └── encoder backbone → (B, n_visible, D)
            └── decoder_input_with_mask_tokens → (B, T_tokens, D)  (re-shuffled by ids_restore)
            └── + decoder_pos_emb
            └── decoder → (B, T_tokens, D)
            └── reconstruction_head: Linear D → patch_samples → (B, T_tokens, P)
            └── reshape → (B, T_samples_reconstructed) where P*T_tokens == T_samples

    The `forward` method returns the reconstructed signal *and* the mask, so
    the loss function can decide whether to penalise everything or only
    masked positions (MAE convention: only masked positions).

    For Check D (random-init linear probe), use `encode_features(x)` to
    get the mean-pooled encoder output directly without running the
    decoder — that's the canonical "frozen feature" per §4.3 Protocol A.

    This class deliberately does NOT compute the loss internally — losses
    live in `losses.py` and are wired up by the trainer / sanity checks.
    Keeping forward returning `(reconstruction, mask, encoder_features)`
    makes Check A (loss-at-init) trivial: just call several losses on the
    same tuple.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # --- frontend ------------------------------------------------------
        self.frontend = build_frontend(cfg.frontend)

        # --- positional embeddings (encoder + decoder, separate to allow
        # different schemes per exp20) -------------------------------------
        self.encoder_pos_emb = build_pos_emb(cfg.pos_emb, cfg.frontend.d_model)
        self.decoder_pos_emb = build_pos_emb(cfg.pos_emb, cfg.decoder.d_model)

        # --- masking + encoder backbone -----------------------------------
        self.mask_module = build_mask(cfg.mask)
        self.encoder = build_backbone(cfg.backbone)

        # --- mask token + decoder -----------------------------------------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder.d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # If encoder.d_model != decoder.d_model we'd need a projection here;
        # forced equal in ModelConfig.__post_init__.
        self.decoder = build_decoder(cfg.decoder)

        # --- reconstruction head: per-token Linear → patch_samples raw values
        # Zero-init the bias and use a small Normal for the weight, per MAE
        # 2022 §3 and the discussion in `01_sanity_baselines/README.md` Check A.
        # This makes the model output ≈ 0 at init, so the loss-at-init values
        # match the closed-form theory (L2 ≈ Var(target) = 1.0 for z-scored
        # input, L1 ≈ √(2/π) ≈ 0.7979). Without this, PyTorch's default
        # Linear init produces output variance σ_r² ≈ 0.4 that adds to the
        # target variance and confuses Check A.
        self.recon_head = nn.Linear(cfg.decoder.d_model, cfg.patch_samples, bias=True)
        nn.init.normal_(self.recon_head.weight, std=0.01)
        nn.init.zeros_(self.recon_head.bias)

        if cfg.target.kind != "raw":
            raise NotImplementedError(
                f"Reconstruction target {cfg.target.kind!r} not implemented; "
                f"see mini_experiments/18_reconstruction_target/README.md"
            )
        if cfg.paradigm.kind != "mae":
            raise NotImplementedError(
                f"Paradigm {cfg.paradigm.kind!r} not implemented; "
                f"see mini_experiments/17_generative_paradigm/README.md"
            )

    # ------------------------------------------------------------------
    # Encoder-only path: useful for frozen-probing eval (Check D).
    # ------------------------------------------------------------------
    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen-probe path: returns *mean-pooled* encoder output.

        x: (B, T_samples) raw EEG signal (single channel, iid)
        out: (B, d_model) mean-pooled features

        Per §4.3 Protocol A and the 2026-05-03 reaffirmation of mean-pool
        (vs CLS-token probing), this is the canonical feature for every
        downstream linear-probe / k-NN evaluation. Do not switch to CLS-
        token probing without a separate spec change.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, T_samples)
        tokens = self.frontend(x)        # (B, T_tokens, D)
        tokens = self.encoder_pos_emb(tokens)
        # NB: at eval time we run the encoder on *all* tokens (no masking)
        # — the linear probe is on the trained-on-pretext-but-frozen-now
        # encoder's representation of the full window.
        encoded = self.encoder(tokens)   # (B, T_tokens, D)
        return encoded.mean(dim=1)       # mean-pool over time

    # ------------------------------------------------------------------
    # Full SSL forward: reconstructs and returns mask.
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full MAE forward.

        Returns a dict with keys:
            "reconstruction":  (B, T_samples) raw reconstructed signal
            "target":          (B, T_samples) the target to compare against (raw input)
            "mask":             (B, T_samples) 1 at masked sample positions, 0 at visible
            "encoder_features": (B, n_visible, D)  encoder output before decoder
            "decoder_features": (B, T_tokens, D)   decoder output (pre-recon-head)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, _, T_samples = x.shape
        target = x.squeeze(1)             # (B, T_samples)

        # --- frontend ---
        tokens = self.frontend(x)         # (B, T_tokens, D)
        tokens = self.encoder_pos_emb(tokens)
        T_tokens = tokens.size(1)
        D = tokens.size(2)

        # --- mask + encode ---
        m = self.mask_module(B, T_tokens, x.device)
        n_visible = m.ids_keep.size(1)
        ids_keep_d = m.ids_keep.unsqueeze(-1).expand(-1, -1, D)
        visible_tokens = torch.gather(tokens, dim=1, index=ids_keep_d)   # (B, n_visible, D)
        encoded = self.encoder(visible_tokens)

        # --- decoder input: encoded + mask tokens, restored to original order
        n_masked = T_tokens - n_visible
        mask_tokens = self.mask_token.expand(B, n_masked, D)
        x_full = torch.cat([encoded, mask_tokens], dim=1)                # (B, T_tokens, D)
        ids_restore_d = m.ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_full = torch.gather(x_full, dim=1, index=ids_restore_d)        # restore original order

        x_full = self.decoder_pos_emb(x_full)
        decoded = self.decoder(x_full)                                    # (B, T_tokens, D)

        # --- per-token reconstruction → flatten to sample-level signal ---
        recon_patches = self.recon_head(decoded)                         # (B, T_tokens, patch_samples)
        recon = rearrange(recon_patches, "b t p -> b (t p)")             # (B, T_samples)

        # --- expand token-level mask to sample-level ---
        # Each token covers `patch_samples` samples; broadcast the mask
        # accordingly. This is the mask the loss should weight by.
        sample_mask = repeat(m.mask, "b t -> b (t p)", p=self.cfg.patch_samples)

        return {
            "reconstruction": recon,
            "target": target,
            "mask": sample_mask,
            "encoder_features": encoded,
            "decoder_features": decoded,
            "token_mask": m.mask,
            "ids_keep": m.ids_keep,
            "ids_restore": m.ids_restore,
        }


# =============================================================================
# Public entrypoint
# =============================================================================


def build_model(cfg: ModelConfig) -> EEGSSLModel:
    """Public model constructor. Use this everywhere.

    >>> from exp03.model import ModelConfig, build_model
    >>> model = build_model(ModelConfig())          # §4.2 default
    >>> # or override any axis:
    >>> from exp03.model import BackboneConfig
    >>> cfg = ModelConfig(backbone=BackboneConfig(kind="transformer", n_layers=4))
    >>> model = build_model(cfg)
    """
    return EEGSSLModel(cfg)


def count_params(model: nn.Module) -> dict[str, int]:
    """Helper for the shape audit / wandb logging."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
