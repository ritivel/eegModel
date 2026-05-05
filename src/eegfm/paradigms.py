"""Paradigm-specific heads + loss-compute path for exp17 (generative paradigm).

Three paradigms share a common interface — each takes the encoder's
output (visible-token features in the MAE/MAR cases, all-token features
in the AR case) plus the masking metadata + the raw input window, and
returns a scalar training loss + a dict of components for logging::

    >>> head = build_paradigm(cfg.paradigm, cfg=cfg)
    >>> loss, components = head(ssl_outputs={...}, target=raw_signal)
    >>> loss.backward()

The shared input dict (``ssl_outputs``) comes from a refactored
:meth:`eegfm.model.EEGSSLModel.forward` — see the per-paradigm class
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
        :mod:`eegfm.train`.
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
# G3 — Latent-prediction (EEG2Rep / I-JEPA style) — added 2026-05-05 for
# mini-experiment 17 v2. See `17_generative_paradigm/README.md` "v2 design".
# ---------------------------------------------------------------------------


class LatentJEPAHead(ParadigmHead):
    """JEPA-style latent-space prediction head for the G3 paradigm.

    The motivation (per `17_generative_paradigm/results.md` §5.2) is that
    on EEG, raw-signal-reconstruction objectives (G0/G1/G2) are dominated
    by ~−20 dB acoustic noise — the encoder learns to predict noise
    structure, not EEG content. The fix that EEG2Rep / I-JEPA / ECG-JEPA
    converged on independently is to **predict the representation of
    masked patches**, where the target representation is computed by an
    EMA-updated copy of the online encoder ("target encoder"). The
    target encoder has already filtered out un-reconstructable noise
    because it learns under the same SSL pressure.

    Setup (per `17_generative_paradigm/README.md` v2 G3 row):

    * **Online encoder** = the existing ``EEGSSLModel`` encoder (frontend
      + bidirectional Mamba-2). Sees only visible patches under random
      50% masking. Its output at *visible* positions is fed to the
      predictor along with learned mask tokens at masked positions.
    * **Target encoder** = an EMA copy of the online encoder, momentum
      0.996 (BYOL default). Sees the FULL un-masked input and produces
      ``(B, T_tokens, D)`` features. Its parameters are detached from
      the autograd graph; updated via a momentum step after every
      optimizer step (the train loop is responsible for calling
      :meth:`update_target_encoder`).
    * **Predictor** = a small MLP (default 2-layer, hidden 1024) that
      maps ``(B, T_tokens, D) -> (B, T_tokens, D)``. Predicts the
      target encoder's representation at every position (loss is
      computed only on masked positions).
    * **Loss** = smooth-L1 (Huber) between the predictor's output and
      the target encoder's output, averaged over masked positions.
      Smooth-L1 (rather than L2) per the I-JEPA recipe — more robust
      to outlier patches that the target encoder happens to encode
      with high norm.

    Inputs (from a refactored :meth:`EEGSSLModel.forward` that
    additionally computes ``target_full_features``):

        encoded               (B, n_visible, D)   online encoder output, masked tokens dropped
        mask_module_out       MaskOutput          ids_keep, ids_restore, sample-level mask
        target_full_features  (B, T_tokens, D)    target encoder output on the FULL un-masked input
        target                (B, T_samples)      raw input signal — UNUSED here; kept for interface

    Outputs:

        loss                  scalar              smooth-L1 on masked positions
        components            dict[str, float]    loss components for wandb
        predicted_latent      (B, T_tokens, D)    predictor output at all positions

    References:

    * EEG2Rep (Foumani et al. KDD 2024, https://arxiv.org/abs/2402.17772):
      ``+13.12% linear-probe accuracy`` over input-space prediction on
      EEG. The semantic-subsequence-preserving (SSP) masking strategy
      is from this paper; we use the standard random-patch masking from
      our existing :mod:`eegfm.model.RandomPatchMask` for v2 to keep the
      ablation clean (G0/G2/G3 all use the same masking).
    * I-JEPA (Assran et al. CVPR 2023, https://arxiv.org/abs/2301.08243):
      The original representation-space prediction paper, on images.
    * ECG-JEPA (Kim 2024, https://arxiv.org/abs/2410.04339):
      The closest biosignal precedent — JEPA on 12-lead ECG, SOTA on
      PTB-XL; the architecture template we follow most closely.
    * BYOL (Grill et al. NeurIPS 2020, https://arxiv.org/abs/2006.07733):
      EMA target-encoder mechanism; momentum 0.996 is the BYOL default.

    Wire-up notes (for :class:`EEGSSLModel` integration — done in
    model.py for v2):

    1. ``EEGSSLModel.__init__`` builds a target encoder as
       ``copy.deepcopy(self.encoder)`` plus a deepcopy of the frontend
       and ``encoder_pos_emb``, all with ``requires_grad=False``. They
       are passed into ``LatentJEPAHead(target_encoder=...)``.
    2. ``EEGSSLModel.forward`` for the JEPA paradigm runs the FULL input
       (no masking) through the target encoder under ``torch.no_grad()``,
       producing ``target_full_features``. It then runs the visible
       patches through the online encoder and dispatches to this head.
    3. The training loop calls ``model.paradigm.update_target_encoder(
       model.encoder)`` after each ``optimizer.step()``. (We expose
       ``update_target_encoder_from(model)`` as a convenience that
       handles the frontend + encoder + pos-emb update.)
    """

    def __init__(
        self,
        cfg: ParadigmConfig,
        *,
        d_model: int,
        target_encoder: nn.Module,
        target_frontend: nn.Module | None = None,
        target_pos_emb: nn.Module | None = None,
        ema_momentum: float = 0.996,
        predictor_hidden: int = 1024,
        predictor_layers: int = 2,
    ):
        super().__init__(cfg)
        self.d_model = d_model
        self.ema_momentum = ema_momentum

        # The target encoder + frontend + pos-emb are deep copies of the
        # online versions, with grad disabled. Updated via momentum from
        # the online encoder after each training step.
        self.target_encoder = target_encoder
        self.target_frontend = target_frontend
        self.target_pos_emb = target_pos_emb
        for module in (self.target_encoder, self.target_frontend, self.target_pos_emb):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

        # Predictor: visible-features (with mask tokens at masked positions
        # in original order) → predicted latent at every position.
        layers: list[nn.Module] = []
        in_dim = d_model
        for _ in range(max(predictor_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, predictor_hidden))
            layers.append(nn.GELU())
            in_dim = predictor_hidden
        layers.append(nn.Linear(in_dim, d_model))
        self.predictor = nn.Sequential(*layers)

        # Predictor's mask token: a learned vector that fills the masked
        # positions in the input to the predictor (NOT the same as the
        # MAE decoder's mask_token, which goes into the decoder).
        self.mask_token_predictor = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token_predictor, std=0.02)

    def forward(
        self,
        *,
        encoded: torch.Tensor,                          # (B, n_visible, D)
        mask_module_out,
        target_full_features: torch.Tensor,             # (B, T_tokens, D), no_grad
        target: torch.Tensor | None = None,             # raw signal, UNUSED
        loss_fn: nn.Module | None = None,
    ) -> dict[str, Any]:
        del target, loss_fn  # not used by this head

        B, n_visible, D = encoded.shape
        T_tokens = mask_module_out.mask.size(1)
        n_masked = T_tokens - n_visible

        # Build predictor input: visible features (online encoder) +
        # learned mask tokens at masked positions, restored to original
        # token order using ids_restore.
        mask_tokens = self.mask_token_predictor.expand(B, n_masked, D)
        x_full = torch.cat([encoded, mask_tokens], dim=1)
        ids_restore_d = mask_module_out.ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_full = torch.gather(x_full, dim=1, index=ids_restore_d)

        # Predictor pass — predicts target encoder's features at every
        # position. We compute loss only on masked positions.
        predicted = self.predictor(x_full)              # (B, T_tokens, D)

        # Smooth-L1 (Huber) loss per (sample, token, dim), then average
        # over the embedding dim and the masked positions only. Per the
        # I-JEPA recipe.
        per_pos_per_dim = F.smooth_l1_loss(
            predicted, target_full_features.detach(), reduction="none"
        )                                               # (B, T_tokens, D)
        per_pos = per_pos_per_dim.mean(dim=-1)          # (B, T_tokens)

        mask = mask_module_out.mask.float()             # (B, T_tokens), 1 at masked
        n_masked_total = mask.sum().clamp(min=1.0)
        loss = (per_pos * mask).sum() / n_masked_total

        return {
            "loss": loss,
            "components": {
                "jepa_smoothl1": float(loss.detach().item()),
                "n_masked_positions": float(n_masked_total.detach().item()),
                "ema_momentum": float(self.ema_momentum),
            },
            "predicted_latent": predicted,
        }

    @torch.no_grad()
    def update_target_encoder_from(self, online_model: nn.Module) -> None:
        """Momentum-update the target encoder + frontend + pos-emb.

        Call after every ``optimizer.step()`` from the training loop.
        ``online_model`` is the full :class:`EEGSSLModel`; we pull its
        ``encoder``, ``frontend``, ``encoder_pos_emb`` and update the
        target's matching modules in place.
        """
        m = self.ema_momentum

        def _update(target_mod: nn.Module | None, online_mod: nn.Module | None) -> None:
            if target_mod is None or online_mod is None:
                return
            for online_p, target_p in zip(online_mod.parameters(),
                                          target_mod.parameters()):
                target_p.data.mul_(m).add_(online_p.data, alpha=1.0 - m)
            # Also momentum-update buffers (e.g. RMSNorm stats) so the
            # target encoder doesn't drift in batch statistics.
            for online_b, target_b in zip(online_mod.buffers(),
                                           target_mod.buffers()):
                if target_b.dtype.is_floating_point:
                    target_b.data.mul_(m).add_(online_b.data, alpha=1.0 - m)

        _update(self.target_encoder, getattr(online_model, "encoder", None))
        _update(self.target_frontend, getattr(online_model, "frontend", None))
        _update(self.target_pos_emb, getattr(online_model, "encoder_pos_emb", None))


# ---------------------------------------------------------------------------
# G4 — LeJEPA (Balestriero & LeCun, arXiv:2511.08544, Nov 2025)
# Vendored at vendor/lejepa/ (rbalestr-lab/lejepa, CC BY-NC 4.0).
# ---------------------------------------------------------------------------


class LeJEPAHead(ParadigmHead):
    """G4 paradigm head: LeJEPA's heuristics-free joint-embedding recipe.

    Despite the "JEPA" name shared with G3, this is a fundamentally
    different recipe from I-JEPA / EEG2Rep / our :class:`LatentJEPAHead`:
    no masking, no EMA target encoder, no stop-gradient, no predictor
    over masked positions. The only "predictive" element is an
    invariance loss between projections of multiple augmented views of
    the same input.

    The architecture is:

    * **Encoder**: shared with the rest of the model (``EEGSSLModel.encoder``)
      — runs over the FULL input window for every view. No masking.
    * **Projector**: a 3-layer MLP (BatchNorm + GELU) mapping the
      mean-pooled encoder output to a low-dim ``proj_dim``. Following
      the LeJEPA ``MINIMAL.md`` recipe.
    * **Multi-view input**: the trainer constructs ``V`` augmented views
      per sample via :class:`eegfm.data.MultiViewCollate`. The encoder
      processes all ``B*V`` views in one forward pass, then the head
      reshapes to ``(V, B, proj_dim)`` for loss computation.
    * **Loss** = ``λ * SIGReg(proj) + (1 - λ) * invariance(proj)``
      where:

      - ``invariance = ((proj.mean(0) - proj) ** 2).mean()`` — pulls all
        view projections toward their per-sample mean, like BYOL/DINO's
        consistency loss but **without** stop-gradients or a teacher.
      - ``SIGReg`` = Sketched Isotropic Gaussian Regularization — the
        statistical-test-based anti-collapse term from
        :mod:`lejepa.multivariate`. Concretely
        ``SlicingUnivariateTest(EppsPulley(...), num_slices=...)``
        applied to the flattened ``(V*B, proj_dim)`` projections.

    LeJEPA's pitch: a *single* trade-off hyperparameter ``λ`` (default
    0.02 per the reference recipe) replaces the dozen heuristics
    (momentum schedule, temperature schedule, stop-gradient, teacher
    update rule, etc.) that other SSL methods rely on. Empirically on
    ImageNet the same recipe matches or beats I-JEPA at 3x fewer epochs
    (`README.md` benchmark table). On EEG it is unvalidated — see
    Notion experiment 21 for our pre-registered hypothesis.

    References:

    * **LeJEPA paper** (Balestriero & LeCun, Nov 2025):
      https://arxiv.org/abs/2511.08544
    * **Repo** (CC BY-NC 4.0): https://github.com/rbalestr-lab/lejepa
      Vendored at ``vendor/lejepa/``.
    * **Minimal example** (130-line ViT-S/8 + Imagenette reference):
      ``vendor/lejepa/MINIMAL.md``.

    Inputs (from a refactored :meth:`EEGSSLModel.forward` for ``lejepa``):

        encoded_views   (V, B, D)   mean-pooled encoder output, one row per view

    Outputs:

        loss            scalar              total LeJEPA loss
        components      dict[str, float]    sigreg / invariance / lambda for wandb

    Implementation note: per LeJEPA's ``MINIMAL.md``, the projector
    must use BatchNorm (not LayerNorm) — the stats-based collapse
    prevention assumes BatchNorm's batch-level standardisation. We
    expose the choice via a flag for ablation, but BatchNorm is the
    default-and-recommended setting.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        d_model: int,
        proj_hidden: int = 2048,
        proj_dim: int = 128,
        proj_layers: int = 3,
        proj_norm: str = "batchnorm",
        sigreg_num_slices: int = 1024,
        sigreg_t_max: float = 3.0,
        sigreg_n_points: int = 17,
        sigreg_clip_value: float | None = None,
        lambda_sigreg: float = 0.02,
    ):
        super().__init__(cfg)
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.lambda_sigreg = lambda_sigreg

        # Build the projector MLP. Per LeJEPA `MINIMAL.md`:
        #     proj = MLP(d_model, [proj_hidden]*(proj_layers-1) + [proj_dim],
        #                norm_layer=BatchNorm1d)
        # We hand-roll it (vs torchvision.ops.MLP) to keep the einops/torch
        # dep surface minimal and to make the BN/LN swap explicit for ablation.
        if proj_norm == "batchnorm":
            norm_cls: Any = lambda d: nn.BatchNorm1d(d)
        elif proj_norm == "layernorm":
            norm_cls = lambda d: nn.LayerNorm(d)
        elif proj_norm == "none":
            norm_cls = lambda d: nn.Identity()
        else:
            raise ValueError(f"proj_norm must be batchnorm|layernorm|none, got {proj_norm!r}")

        layers: list[nn.Module] = []
        in_dim = d_model
        for _ in range(max(proj_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, proj_hidden))
            layers.append(norm_cls(proj_hidden))
            layers.append(nn.GELU())
            in_dim = proj_hidden
        layers.append(nn.Linear(in_dim, proj_dim))
        self.projector = nn.Sequential(*layers)

        # Build the SIGReg loss using the vendored lejepa package.
        # Lazy import so the rest of `paradigms` still works on a CPU
        # box without `vendor/lejepa` installed (e.g. unit tests of
        # MAEHead / ARNextPatchHead don't need lejepa).
        try:
            import lejepa  # noqa: F401
            from lejepa.univariate import EppsPulley
            from lejepa.multivariate import SlicingUnivariateTest
        except ImportError as exc:
            raise ImportError(
                "LeJEPA paradigm requires the `lejepa` package. Install "
                "the vendored submodule: `pip install -e vendor/lejepa` "
                "(or `uv pip install -e vendor/lejepa`). Submodule lives "
                "at vendor/lejepa/ — pinned to a specific commit of "
                "rbalestr-lab/lejepa (CC BY-NC 4.0)."
            ) from exc

        self.sigreg = SlicingUnivariateTest(
            univariate_test=EppsPulley(t_max=sigreg_t_max, n_points=sigreg_n_points),
            num_slices=sigreg_num_slices,
            reduction="mean",
            sampler="gaussian",
            clip_value=sigreg_clip_value,
        )

    def forward(
        self,
        *,
        encoded_views: torch.Tensor,            # (V, B, D)
    ) -> dict[str, Any]:
        """Compute LeJEPA's λ-weighted invariance + SIGReg loss.

        encoded_views is the mean-pooled encoder output for each view,
        already reshaped to (V, B, D) by :meth:`EEGSSLModel.forward`.
        We project to (V, B, proj_dim), compute invariance (variance of
        view projections around their per-sample mean), then SIGReg on
        the flattened (V*B, proj_dim) projections to enforce isotropic
        Gaussian distribution. Total loss is the convex combination
        ``λ·sigreg + (1-λ)·invariance``.
        """
        V, B, D = encoded_views.shape
        # Flatten (V, B, D) -> (V*B, D), project, reshape back.
        emb_flat = encoded_views.reshape(V * B, D)
        proj_flat = self.projector(emb_flat)            # (V*B, proj_dim)
        proj = proj_flat.reshape(V, B, self.proj_dim)   # (V, B, P)

        # Invariance: views of the same sample should have similar projections.
        # Per LeJEPA MINIMAL.md: `inv_loss = (proj.mean(0) - proj).square().mean()`
        # where the mean(0) is over the V views per sample.
        inv_loss = (proj.mean(dim=0, keepdim=True) - proj).square().mean()

        # SIGReg: all (V*B, proj_dim) projections together should follow
        # an isotropic Gaussian distribution. Anti-collapse without EMA
        # / stop-gradient.
        sigreg_loss = self.sigreg(proj_flat)

        loss = sigreg_loss * self.lambda_sigreg + inv_loss * (1.0 - self.lambda_sigreg)

        return {
            "loss": loss,
            "components": {
                "lejepa_total": float(loss.detach().item()),
                "lejepa_sigreg": float(sigreg_loss.detach().item()),
                "lejepa_invariance": float(inv_loss.detach().item()),
                "lejepa_lambda_sigreg": float(self.lambda_sigreg),
                "lejepa_proj_norm_mean": float(proj_flat.norm(dim=-1).mean().detach().item()),
            },
            "projections": proj,
            "embeddings": encoded_views,
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
    # G3 JEPA-only knobs (set up by EEGSSLModel.__init__ when paradigm.kind == "jepa")
    target_encoder: nn.Module | None = None,
    target_frontend: nn.Module | None = None,
    target_pos_emb: nn.Module | None = None,
    ema_momentum: float = 0.996,
    predictor_hidden: int = 1024,
    predictor_layers: int = 2,
    # G4 LeJEPA-only knobs (set up by EEGSSLModel.__init__ when paradigm.kind == "lejepa")
    lejepa_proj_hidden: int = 2048,
    lejepa_proj_dim: int = 128,
    lejepa_proj_layers: int = 3,
    lejepa_proj_norm: str = "batchnorm",
    lejepa_sigreg_num_slices: int = 1024,
    lejepa_sigreg_t_max: float = 3.0,
    lejepa_sigreg_n_points: int = 17,
    lejepa_lambda_sigreg: float = 0.02,
) -> ParadigmHead:
    """Build the paradigm head matching ``paradigm_cfg.kind``.

    The MAE head needs the existing decoder + recon_head + mask_token
    from :class:`EEGSSLModel` (we pass them in rather than re-creating
    so the encoder→decoder shape contract is enforced upstream). The
    AR and MAR heads create their own internal heads. The G3 JEPA head
    needs a *target encoder* (an EMA copy of the online encoder) plus
    the matching frontend + pos-emb; these are constructed in
    ``EEGSSLModel.__init__`` and passed in here.
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
        if target_encoder is None:
            raise ValueError(
                "JEPA paradigm requires a target_encoder (EMA copy of the "
                "online encoder). Built by EEGSSLModel.__init__ when "
                "cfg.paradigm.kind == 'jepa'. See LatentJEPAHead docstring "
                "for the full wire-up contract."
            )
        return LatentJEPAHead(
            paradigm_cfg,
            d_model=d_model,
            target_encoder=target_encoder,
            target_frontend=target_frontend,
            target_pos_emb=target_pos_emb,
            ema_momentum=ema_momentum,
            predictor_hidden=predictor_hidden,
            predictor_layers=predictor_layers,
        )
    if kind == "lejepa":
        return LeJEPAHead(
            paradigm_cfg,
            d_model=d_model,
            proj_hidden=lejepa_proj_hidden,
            proj_dim=lejepa_proj_dim,
            proj_layers=lejepa_proj_layers,
            proj_norm=lejepa_proj_norm,
            sigreg_num_slices=lejepa_sigreg_num_slices,
            sigreg_t_max=lejepa_sigreg_t_max,
            sigreg_n_points=lejepa_sigreg_n_points,
            lambda_sigreg=lejepa_lambda_sigreg,
        )
    raise ValueError(f"unknown paradigm kind: {kind!r}")
