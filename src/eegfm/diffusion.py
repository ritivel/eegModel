"""Minimal Gaussian diffusion utilities for the MAR (G2) paradigm.

This is a deliberately small port of facebookresearch/MAR's IDDPM-style
diffusion code (cosine noise schedule + ε-prediction MSE loss),
specialised to *representation learning*:

* We need only the **forward process** (``q_sample``) to noise the
  target and the **training loss** (ε-prediction MSE) — not the reverse
  process or sampling, since for SSL we never actually generate. That
  drops ~600 LOC of IDDPM's ``gaussian_diffusion.py`` to ~80 LOC here.
* We use the **cosine noise schedule** from `improved-diffusion`
  ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)), which
  is what MAR's ``create_diffusion(noise_schedule="cosine")`` produces
  by default.
* We skip the VLB (variational lower bound) head; pure ε-MSE is enough
  for representation learning and matches every IDDPM-style ablation
  that doesn't care about generation quality.

If we ever need actual sampling (e.g. for a generative-quality eval like
FID), we can either bring in the full ``diffusion`` package from MAR
(`pip install -e git+https://github.com/LTH14/mar#subdirectory=diffusion`)
or copy the ~30 LOC for the reverse process. Out of scope for exp17
(representation-only).

References:

    MAR (Li et al. 2024): https://github.com/LTH14/mar
        — official PyTorch implementation. Our :class:`DiffLossHead`
        in :mod:`paradigms` adapts ``models/diffloss.py`` (same MIT
        license; code structure preserved, comments rewritten for
        clarity).
    IDDPM (Nichol & Dhariwal, 2021):
        — cosine schedule definition; we ported the schedule constants.
    EDM (Karras et al. 2022, arXiv 2206.00364):
        — alternative noise schedule (log-normal σ instead of t∈[0,T]);
        not used here. MAR's authors found cosine works better for
        representation learning, so we follow MAR.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal 2021)
# ---------------------------------------------------------------------------


def _cosine_alpha_bar(t: torch.Tensor, *, s: float = 0.008) -> torch.Tensor:
    """Cosine ᾱ(t): α_bar at continuous time t∈[0, 1].

    Per `improved-diffusion`/IDDPM:

        ᾱ(t) = cos²((t + s) / (1 + s) · π/2)

    where s=0.008 is a small offset that prevents ᾱ(0) from being
    exactly 1 (avoiding division-by-zero issues in q_sample).
    """
    return torch.cos((t + s) / (1.0 + s) * math.pi * 0.5).pow(2)


class CosineDiffusion:
    """Discrete cosine-schedule diffusion with ε-prediction training loss.

    Usage::

        diff = CosineDiffusion(num_timesteps=1000, device="cuda")
        # Per training step, on a target tensor `x0` shape (B, ..., C):
        loss = diff.training_loss_eps(predict_eps_fn, x0)
        # where predict_eps_fn(x_t, t, **cond) -> eps_pred

    The schedule is built once at construction; sampling random
    timesteps + computing q_sample is O(1) per call.
    """

    def __init__(self, num_timesteps: int = 1000, *, s: float = 0.008,
                 max_beta: float = 0.999):
        self.num_timesteps = int(num_timesteps)
        self.s = float(s)
        self.max_beta = float(max_beta)

        # Pre-compute α_bar at every discrete timestep t ∈ {0, …, T-1}.
        # We use the IDDPM trick: build betas from the cosine schedule's
        # α_bar ratios, clip to max_beta=0.999 for numerical stability.
        ts = torch.arange(self.num_timesteps + 1, dtype=torch.float64)
        alpha_bar = _cosine_alpha_bar(ts / self.num_timesteps, s=s)
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, max_beta)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Cache as float32 buffers — we'll move to the model's device
        # lazily when the first batch arrives (see `_to_device`).
        self.alphas_cumprod: torch.Tensor = alphas_cumprod.to(torch.float32)
        self.sqrt_alphas_cumprod: torch.Tensor = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = (
            1.0 - self.alphas_cumprod
        ).sqrt()

        self._device: torch.device | None = None

    # ----- internals ---------------------------------------------------

    def _to_device(self, device: torch.device) -> None:
        """Move the schedule buffers to ``device`` once we know it."""
        if self._device == device:
            return
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = (
            self.sqrt_one_minus_alphas_cumprod.to(device)
        )
        self._device = device

    @staticmethod
    def _expand_for(t_buf: torch.Tensor, t: torch.Tensor,
                    target_shape: torch.Size) -> torch.Tensor:
        """Gather ``t_buf[t]`` and broadcast to ``target_shape``."""
        out = t_buf[t]                                   # (B,)
        # Add singleton dims to match target's trailing dims
        while out.dim() < len(target_shape):
            out = out.unsqueeze(-1)
        return out.expand(target_shape)

    # ----- public api --------------------------------------------------

    def sample_timesteps(self, B: int, device: torch.device) -> torch.Tensor:
        """Uniform sampling of integer timesteps in [0, num_timesteps)."""
        return torch.randint(0, self.num_timesteps, (B,), device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward (noising) process. Returns ``(x_t, noise)``.

        Args:
            x0: clean target, shape (B, ...) — any trailing dims OK.
            t:  integer timesteps, shape (B,).
            noise: optional pre-sampled noise of shape ``x0.shape``;
                if None, sampled from N(0, 1).

        Returns:
            x_t: noised target, same shape as x0.
            noise: the noise that was added (so the loss can compare).
        """
        self._to_device(x0.device)
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self._expand_for(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = self._expand_for(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    def training_loss_eps(
        self,
        predict_eps_fn,
        x0: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
        **cond_kwargs,
    ) -> torch.Tensor:
        """ε-prediction MSE training loss.

        Args:
            predict_eps_fn: callable ``(x_t, t, **cond_kwargs) -> eps_pred``.
            x0: clean target (B, ..., C).
            weight: optional per-sample / per-position weight, broadcastable
                to the per-element MSE map. Use this to mask out non-MAR
                positions (e.g. only compute the loss at masked tokens).
            **cond_kwargs: forwarded to ``predict_eps_fn`` (e.g. the
                conditioning vector ``z`` from the encoder).

        Returns:
            scalar tensor — the (optionally weighted) mean squared error
            between predicted ε and true ε.
        """
        B = x0.size(0)
        t = self.sample_timesteps(B, x0.device)
        x_t, noise = self.q_sample(x0, t)
        eps_pred = predict_eps_fn(x_t, t, **cond_kwargs)

        if eps_pred.shape != noise.shape:
            raise ValueError(
                f"eps_pred shape {tuple(eps_pred.shape)} != noise shape "
                f"{tuple(noise.shape)}; predict_eps_fn must return same "
                f"shape as the target."
            )

        per_element = (eps_pred - noise).pow(2)
        if weight is not None:
            # Weighted-mean over all dims: sum(w*err) / sum(w) (with eps).
            w = weight.to(per_element)
            return (per_element * w).sum() / w.sum().clamp(min=1e-8)
        return per_element.mean()
