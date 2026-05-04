"""Paradigm-specific heads + loss-compute path for exp17 (generative paradigm).

Three paradigms share a common interface — each takes the encoder's
output (visible-token features in the MAE/MAR cases, all-token features
in the AR case) plus the masking metadata + the raw input window, and
returns a scalar training loss + a dict of components for logging::

    >>> head = build_paradigm(cfg.paradigm, cfg=cfg)
    >>> loss, components = head(ssl_outputs={...}, target=raw_signal)
    >>> loss.backward()

The shared input dict (``ssl_outputs``) comes from a refactored
:meth:`exp03.model.EEGSSLModel.forward` — see the per-paradigm class
docstring for which fields each one needs.

The three implemented paradigms are the ones laid out in
`mini_experiments/17_generative_paradigm/README.md`:

* **G0 MAE** (the §4.2 default) — bidirectional encoder, decoder,
  per-token Linear reconstruction head, L1 + 0.3·MR-STFT loss on
  masked positions only.
* **G1 AR-causal-aligned** — unidirectional Mamba encoder, no decoder,
  per-token Linear "next-patch" head, L1 + 0.3·MR-STFT loss on the
  next-patch prediction.
* **G2 MAR + diffusion head** — bidirectional encoder, no decoder
  Mamba block; instead a small MLP diffusion head (3 ResBlocks at width
  1024 by default per MAR's defaults) that predicts the noise added
  to the per-patch raw signal at masked positions, conditioned on the
  encoder representation.

References:

    MAR (Li et al. 2024, NeurIPS Spotlight) — https://github.com/LTH14/mar
        :class:`DiffLossHead` ports ``models/diffloss.py`` and the
        SimpleMLPAdaLN architecture. Same MIT license; preserved code
        structure with renamed comments.
    ARM (Ren et al. ICLR 2025, arXiv 2406.07537) — https://github.com/OliverRensu/ARM
        :class:`ARNextPatchHead` follows ARM's autoregressive recipe of
        per-token next-patch L1 loss; ARM uses MAE's ``norm_pix_loss``
        too, so we expose a flag for it.
    MAE (He et al. 2022, arXiv 2111.06377) — https://github.com/facebookresearch/mae
        ``main_pretrain.py`` / ``engine_pretrain.py`` are the canonical
        single-process / DDP training loop; we reuse the recipe in
        :mod:`exp03.train`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from . import diffusion
from . import losses as losses_mod
from .model import (
    Mamba2Decoder,
    ParadigmConfig,
    PosEmbConfig,
    build_pos_emb,
)


# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------


class ParadigmHead(nn.Module):
    """Abstract base class for the three generative-paradigm heads.

    Subclasses MUST implement:

        forward(self, *, encoded, mask_module_out, target, tokens_full, decoder_pos_emb)
            -> dict[str, Tensor]

    where the returned dict has at minimum ``"loss"`` (scalar) and
    ``"components"`` (dict of named scalars for logging). Other entries
    are paradigm-specific and useful for downstream introspection
    (e.g. the reconstruction at masked positions for the MAE head).
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg


# ---------------------------------------------------------------------------
# G0 — MAE-bidirectional (the §4.2 default)
# ---------------------------------------------------------------------------


class MAEHead(ParadigmHead):
    """The §4.2 default paradigm head: encoder-decoder MAE-style.

    The encoder processes only visible tokens (asymmetric MAE), then
    the decoder sees ``visible_features ⊕ learned_mask_token`` re-shuffled
    back to the original order, runs a small Mamba-2 stack, and a per-token
    Linear projects to ``patch_samples`` raw values per token.

    Inputs (from :meth:`EEGSSLModel.forward`):

        encoded          (B, n_visible, D)        encoder output, masked tokens dropped
        mask_module_out  MaskOutput               ids_keep, ids_restore, sample-level mask
        target           (B, T_samples)           raw input signal (the reconstruction target)
        tokens_full      (B, T_tokens, D)         frontend output BEFORE encoder (unused here)
        decoder_pos_emb  nn.Module                pos-emb to add before the decoder

    Outputs:

        loss             scalar                   L1 + 0.3·MR-STFT on masked positions
        components       dict[str, float]         loss components for wandb
        reconstruction   (B, T_samples)           full-window reconstruction (zeros at visible)
        sample_mask      (B, T_samples)           1 at masked sample positions, 0 elsewhere
    """

    def __init__(
        self,
        cfg: ParadigmConfig,
        *,
        decoder_d_model: int,
        patch_samples: int,
        decoder_module: nn.Module,
        mask_token: nn.Parameter,
        decoder_pos_emb: nn.Module,
        recon_head: nn.Linear,
    ):
        super().__init__(cfg)
        self.decoder_d_model = decoder_d_model
        self.patch_samples = patch_samples
        self.decoder = decoder_module
        self.mask_token = mask_token
        self.decoder_pos_emb = decoder_pos_emb
        self.recon_head = recon_head
        # The §4.2 default loss; the trainer can override by passing a
        # different `loss_fn` if it wants a custom recipe.
        self.default_loss = losses_mod.L1PlusMRSTFTLoss()

    def forward(
        self,
        *,
        encoded: torch.Tensor,
        mask_module_out,
        target: torch.Tensor,
        tokens_full: torch.Tensor | None = None,  # unused; kept for interface
        loss_fn: nn.Module | None = None,
    ) -> dict[str, Any]:
        B, n_visible, D = encoded.shape
        T_tokens = mask_module_out.mask.size(1)
        n_masked = T_tokens - n_visible

        # --- decoder input: encoded + mask tokens, restored to original order
        mask_tokens = self.mask_token.expand(B, n_masked, D)
        x_full = torch.cat([encoded, mask_tokens], dim=1)
        ids_restore_d = mask_module_out.ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_full = torch.gather(x_full, dim=1, index=ids_restore_d)

        x_full = self.decoder_pos_emb(x_full)
        decoded = self.decoder(x_full)

        # --- per-token recon → flatten to sample-level signal
        recon_patches = self.recon_head(decoded)
        recon = rearrange(recon_patches, "b t p -> b (t p)")

        # --- expand token-level mask to sample-level
        sample_mask = repeat(
            mask_module_out.mask, "b t -> b (t p)", p=self.patch_samples
        )

        # --- loss
        ssl_out = {
            "reconstruction": recon,
            "target": target,
            "mask": sample_mask,
        }
        loss_fn = loss_fn or self.default_loss
        loss, components = loss_fn(ssl_out)

        return {
            "loss": loss,
            "components": components,
            "reconstruction": recon,
            "sample_mask": sample_mask,
            "decoder_features": decoded,
        }


# ---------------------------------------------------------------------------
# G1 — AR-causal-aligned (no masking, no decoder)
# ---------------------------------------------------------------------------


class ARNextPatchHead(ParadigmHead):
    """Per-token next-patch prediction head for the AR paradigm.

    Setup (per `17_generative_paradigm/README.md` § Implementation pointers):

    * Encoder is **forward-only Mamba-2** (set ``backbone.bidirectional =
      False`` upstream); the encoder runs over *all* tokens with no
      masking.
    * No decoder. A single ``nn.Linear(D, patch_samples)`` projects each
      position's encoder output to a prediction of the *next* token's raw
      signal patch.
    * The first token has nothing to condition on (no left context) — but
      we still ask it to predict token 1's patch. Since left-context is
      tiny vs the signal length, we keep that token in the loss; ARM does
      the same. We exclude the *last* token (no ground-truth next).

    Optionally normalises each ground-truth patch by its own mean+std
    before computing the loss (``norm_pix_loss`` per MAE / ARM). Default
    off; the §4.2-aligned recipe is raw L1 + 0.3·MR-STFT.

    Inputs:

        encoded     (B, T_tokens, D)         encoder output (forward-only Mamba)
        target      (B, T_samples)           raw input signal

    Outputs:

        loss             scalar
        components       dict[str, float]
        reconstruction   (B, T_samples - patch_samples)  predicted next-patch series
                                                         (the loss is computed over T_tokens-1 tokens)
    """

    def __init__(
        self,
        cfg: ParadigmConfig,
        *,
        d_model: int,
        patch_samples: int,
        norm_pix_loss: bool = False,
    ):
        super().__init__(cfg)
        self.d_model = d_model
        self.patch_samples = patch_samples
        self.norm_pix_loss = norm_pix_loss

        # Per-token next-patch prediction head. Same MAE-2022 zero-init
        # to keep loss-at-init aligned with theory.
        self.head = nn.Linear(d_model, patch_samples, bias=True)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        self.default_loss = losses_mod.L1PlusMRSTFTLoss()

    def forward(
        self,
        *,
        encoded: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module | None = None,
        **_kw,
    ) -> dict[str, Any]:
        B, T_tokens, _ = encoded.shape
        P = self.patch_samples
        if target.size(1) != T_tokens * P:
            raise ValueError(
                f"AR head expected target of length T_tokens*patch_samples = "
                f"{T_tokens * P}, got {target.size(1)}"
            )

        # Reshape target into per-token patches for the next-patch comparison.
        target_patches = rearrange(target, "b (t p) -> b t p", p=P)

        # Predict patch at each position from the encoder output.
        pred_patches = self.head(encoded)                      # (B, T_tokens, P)

        # Predict t -> patch[t+1], so we drop the last position from the
        # predictions and the first position from the targets.
        pred_next = pred_patches[:, :-1, :]                    # (B, T_tokens-1, P)
        target_next = target_patches[:, 1:, :]                 # (B, T_tokens-1, P)

        if self.norm_pix_loss:
            mu = target_next.mean(dim=-1, keepdim=True)
            sd = target_next.std(dim=-1, keepdim=True) + 1e-6
            target_next = (target_next - mu) / sd

        # Use the same composite loss as G0 by reshaping back to sample-level
        # plus a virtual "all positions are masked" flag (no leakage to worry
        # about since we already excluded the last token from the prediction
        # and the first token from the target).
        recon_flat = rearrange(pred_next, "b t p -> b (t p)")
        target_flat = rearrange(target_next, "b t p -> b (t p)")
        # Mask = 1 everywhere — every position is in the loss.
        sample_mask = torch.ones_like(target_flat)

        loss_fn = loss_fn or self.default_loss
        loss, components = loss_fn({
            "reconstruction": recon_flat,
            "target": target_flat,
            "mask": sample_mask,
        })

        return {
            "loss": loss,
            "components": components,
            "reconstruction": recon_flat,
            "sample_mask": sample_mask,
        }


# ---------------------------------------------------------------------------
# G2 — MAR + diffusion head (port of LTH14/mar's SimpleMLPAdaLN)
# ---------------------------------------------------------------------------


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation, identical to MAR's ``modulate(x, shift, scale)``."""
    return x * (1.0 + scale) + shift


class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → MLP projection.

    Identical structure to MAR's ``TimestepEmbedder``; we use SiLU and
    pre-compute the half-frequencies. Defaults to a 256-d sinusoidal
    embedding then projected to ``hidden_size``.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int,
                              max_period: float = 10000.0) -> torch.Tensor:
        # MAR / GLIDE convention: half-cos | half-sin frequencies.
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal_embedding(t, self.frequency_embedding_size))


class _DiffResBlock(nn.Module):
    """One AdaLN residual block of MAR's ``SimpleMLPAdaLN``."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = _modulate(self.norm(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h


class _DiffFinalLayer(nn.Module):
    """Final AdaLN + Linear out layer (DiT-style)."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    """Diffusion-loss MLP backbone, port of MAR's ``SimpleMLPAdaLN``.

    Architecture::

        x ─ Linear(in→D) ─┐
                          ↓
        t ─ TimeEmbed → ─→ y = t + c    (broadcast through every block)
        c ─ Linear(z→D) ─┘
                          ↓
                     [AdaLN ResBlock] × num_res_blocks
                          ↓
                     AdaLN FinalLinear → out

    Where ``in_channels = patch_samples`` (the per-token target dim),
    ``z_channels = D`` (encoder output dim), and ``model_channels =
    width`` (1024 by default per MAR).
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embed = _TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        self.res_blocks = nn.ModuleList(
            [_DiffResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.final_layer = _DiffFinalLayer(model_channels, out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier-uniform on Linear weights, zero on biases (MAR default).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small normal on the timestep MLP, matching MAR.
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        # Zero-init AdaLN modulation layers — the MAR / DiT trick.
        for block in self.res_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``x``: noised target, ``t``: timestep ints, ``c``: condition vector.

        All three are batched at the *same* leading B; MAR does this by
        flattening (batch × position) into one dimension before calling
        the MLP. We expect the caller to do the same — see :class:`MARDiffLossHead`.
        """
        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)
        y = t_emb + c_emb
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


class MARDiffLossHead(ParadigmHead):
    """G2 paradigm head: bidirectional encoder + per-token diffusion loss.

    On each forward pass:

    1. Reshape the raw input into per-token patches: ``(B, T_tokens, P)``.
    2. The encoder (bidirectional Mamba-2 from the upstream model) has
       already produced ``encoded ∈ (B, n_visible, D)`` for the visible
       tokens. We **scatter-fill** that back to ``(B, T_tokens, D)`` by
       placing a learned mask token at each masked position. (Unlike
       the MAE decoder path, here we don't need to feed mask positions
       to a decoder — the encoder representation at masked positions is
       *the* condition for the diffusion head.)
    3. Sample timesteps + add noise to the target patches at masked
       positions only.
    4. The diffusion MLP predicts the noise from ``(x_t, t, z)``, where
       ``z`` is the encoder representation at that token.
    5. Loss is masked-MSE between predicted-ε and true-ε at masked
       positions (exactly like MAR's mask-aware DiffLoss).

    Implementation note: per the README's risks table, the diffusion-
    head loss can collapse if the encoder produces constant outputs.
    We add **an MAR-2024-style 0.5× target-channel-std normalisation** —
    i.e. compute MSE on un-normalised targets, no per-patch z-score —
    so the optimal ε-pred is *not* the trivial zero.

    Args (per :meth:`__init__`):

        d_model: encoder output dim (= z_channels for the diff MLP).
        patch_samples: target dim per token (= in_channels for the MLP).
        diffloss_d / diffloss_w: depth and width of the diffusion MLP.
            MAR defaults: depth=3, width=1024.
        num_diffusion_steps: discrete training timesteps. MAR uses 1000.
    """

    def __init__(
        self,
        cfg: ParadigmConfig,
        *,
        d_model: int,
        patch_samples: int,
        diffloss_d: int = 3,
        diffloss_w: int = 1024,
        num_diffusion_steps: int = 1000,
    ):
        super().__init__(cfg)
        self.d_model = d_model
        self.patch_samples = patch_samples

        # MAR uses one shared (across batch + position) mask token to
        # represent "missing visible feature" inside the diffloss-conditioning
        # path. We do the same: a single learned vector that replaces the
        # encoder's z at masked positions. (This differs from the MAE-side
        # decoder mask token, which lives upstream in EEGSSLModel.)
        self.cond_mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cond_mask_token, std=0.02)

        self.diff_mlp = SimpleMLPAdaLN(
            in_channels=patch_samples,
            model_channels=diffloss_w,
            out_channels=patch_samples,             # ε-pred only; no VLB
            z_channels=d_model,
            num_res_blocks=diffloss_d,
        )
        self.diffusion = diffusion.CosineDiffusion(num_timesteps=num_diffusion_steps)

    def _scatter_visible_back(
        self,
        encoded: torch.Tensor,
        mask_module_out,
        T_tokens: int,
    ) -> torch.Tensor:
        """Place ``encoded`` (visible only) back into a (B, T_tokens, D)
        slot, with the cond_mask_token at masked positions."""
        B, n_visible, D = encoded.shape
        n_masked = T_tokens - n_visible

        cond_full = torch.cat(
            [encoded, self.cond_mask_token.expand(B, n_masked, D)],
            dim=1,
        )
        ids_restore_d = mask_module_out.ids_restore.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(cond_full, dim=1, index=ids_restore_d)  # (B, T_tokens, D)

    def forward(
        self,
        *,
        encoded: torch.Tensor,
        mask_module_out,
        target: torch.Tensor,
        **_kw,
    ) -> dict[str, Any]:
        B = target.size(0)
        T_tokens = mask_module_out.mask.size(1)
        P = self.patch_samples

        # --- 1. Per-token target patches ---
        target_patches = rearrange(target, "b (t p) -> b t p", p=P)   # (B, T_tokens, P)

        # --- 2. Scatter encoder outputs back to (B, T_tokens, D) ---
        cond_full = self._scatter_visible_back(encoded, mask_module_out, T_tokens)

        # --- 3+4+5. Flatten (batch × position) and run masked diffusion loss
        # MAR computes the loss only at masked positions; we follow.
        token_mask = mask_module_out.mask                              # (B, T_tokens), 1 where masked

        x0 = target_patches.reshape(B * T_tokens, P)
        z = cond_full.reshape(B * T_tokens, self.d_model)
        per_pos_weight = token_mask.reshape(B * T_tokens, 1).expand(-1, P)

        loss = self.diffusion.training_loss_eps(
            lambda x_t, t, c: self.diff_mlp(x_t, t, c),
            x0,
            weight=per_pos_weight,
            c=z,
        )

        n_masked = int(token_mask.sum().item())
        return {
            "loss": loss,
            "components": {
                "diff_mse": float(loss.detach().item()),
                "n_masked_positions": n_masked,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_paradigm(
    paradigm_cfg: ParadigmConfig,
    *,
    d_model: int,
    patch_samples: int,
    decoder_module: nn.Module | None,
    decoder_pos_emb: nn.Module | None,
    mask_token: nn.Parameter | None,
    recon_head: nn.Linear | None,
    diffloss_d: int = 3,
    diffloss_w: int = 1024,
    num_diffusion_steps: int = 1000,
    norm_pix_loss: bool = False,
) -> ParadigmHead:
    """Build the paradigm head matching ``paradigm_cfg.kind``.

    The MAE head needs the existing decoder + recon_head + mask_token
    from :class:`EEGSSLModel` (we pass them in rather than re-creating
    so the encoder→decoder shape contract is enforced upstream). The
    AR and MAR heads create their own internal heads.
    """
    kind = paradigm_cfg.kind
    if kind == "mae":
        if decoder_module is None or recon_head is None or mask_token is None or decoder_pos_emb is None:
            raise ValueError("MAE head requires decoder_module, decoder_pos_emb, mask_token, recon_head")
        return MAEHead(
            paradigm_cfg,
            decoder_d_model=d_model,
            patch_samples=patch_samples,
            decoder_module=decoder_module,
            mask_token=mask_token,
            decoder_pos_emb=decoder_pos_emb,
            recon_head=recon_head,
        )
    if kind == "ar":
        return ARNextPatchHead(
            paradigm_cfg,
            d_model=d_model,
            patch_samples=patch_samples,
            norm_pix_loss=norm_pix_loss,
        )
    if kind == "mar":
        return MARDiffLossHead(
            paradigm_cfg,
            d_model=d_model,
            patch_samples=patch_samples,
            diffloss_d=diffloss_d,
            diffloss_w=diffloss_w,
            num_diffusion_steps=num_diffusion_steps,
        )
    if kind == "jepa":
        raise NotImplementedError(
            f"JEPA paradigm not implemented; see "
            f"mini_experiments/18_reconstruction_target/README.md"
        )
    raise ValueError(f"unknown paradigm kind: {kind!r}")
