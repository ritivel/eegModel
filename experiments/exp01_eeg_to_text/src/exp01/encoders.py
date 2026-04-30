"""Encoder loaders with a uniform ``(B, C, T) -> (B, T_seq, D)`` interface.

Three encoders are wired:
  REVE     — ``brain-bzh/reve-base``     (gated, requires HF_TOKEN + EUA)
  TFM      — ``Jathurshan/TFM-Tokenizer`` (open)
  DIVER-1  — checkpoint pulled from the cold bucket

Each encoder advertises:
  ``feature_dim`` (D)              — encoder hidden size
  ``native_sr``  (Hz | None)       — required input sampling rate, None=any
  ``encode(eeg, sr, channels)`` -> Tensor of shape (B, T_seq, D)

Discrete encoders (TFM by default; REVE/DIVER-1 if attached to an RVQ head)
also expose ``tokenize`` -> LongTensor (B, T_seq) for vocab-extension bridges.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from . import storage


# ============================================================================
# Common interface
# ============================================================================


@dataclass
class EncoderSpec:
    feature_dim: int
    native_sr: int | None        # required input rate, or None for any
    discrete: bool = False        # has a tokenize() method
    codebook_size: int | None = None


class EEGEncoder(nn.Module):
    spec: EncoderSpec

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        """eeg: (B, C, T) at ``sr`` Hz. Returns (B, T_seq, D)."""
        raise NotImplementedError

    def tokenize(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.LongTensor:
        raise NotImplementedError(f"{type(self).__name__} is not discrete")


# ============================================================================
# REVE  —  brain-bzh/reve-base
# ============================================================================


class REVEEncoder(EEGEncoder):
    spec = EncoderSpec(feature_dim=512, native_sr=200, discrete=False)

    def __init__(self, model_id: str = "brain-bzh/reve-base"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=str(storage.HF_CACHE)
        )
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", trust_remote_code=True, cache_dir=str(storage.HF_CACHE)
        )

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        eeg = _resample(eeg, sr, self.spec.native_sr)
        norm_channels = [_normalize_channel_for_reve(c) for c in channels]
        positions = self._safe_positions(norm_channels)            # (C, 3)
        positions = positions.to(eeg.device).unsqueeze(0).expand(eeg.size(0), -1, -1)
        out = self.model(eeg, positions)
        # REVE's forward returns either a raw Tensor or a HF ModelOutput.
        # Native shape is ``(B, C, T_patches, D)`` (one token per channel ×
        # patch); flatten to ``(B, C * T_patches, D)`` so downstream bridges
        # see a single sequence dim.
        if isinstance(out, torch.Tensor):
            feats = out
        elif isinstance(out, dict):
            feats = out.get("last_hidden_state", out.get("hidden_states", out))
        else:
            feats = getattr(out, "last_hidden_state", out)
        if feats.dim() == 4:
            B, C, T_p, D = feats.shape
            feats = feats.reshape(B, C * T_p, D)
        return feats


    def _safe_positions(self, channel_names: list[str]) -> torch.Tensor:
        """Return a (C, 3) position tensor for ``channel_names``.

        REVE's stock ``pos_bank`` filters unrecognised names *out* of the
        result, returning ``(len(matched), 3)`` instead of ``(C, 3)``. When
        ``len(matched) == 0`` the resulting empty tensor crashes with
        ``IndexError: tensors used as indices must be long`` (the inner
        ``self.embedding[indices]`` call fails on the empty float tensor
        from ``torch.tensor([])``). When ``0 < len(matched) < C`` the
        downstream encoder forward gets a position-vs-channel-count
        mismatch.

        Wrap it: look up each channel individually; for unknown channels
        substitute a fallback position. The fallback is the mean of all
        recognised positions in the batch (best informationless guess);
        if every channel is unknown (``eeg_sem_relev`` with anonymised
        ``ch01..ch32`` names), use the centroid of the entire pos_bank.
        """
        mapping = self.pos_bank.mapping
        embedding = self.pos_bank.embedding              # (N_known, 3)
        matched_idx = [mapping.get(c) for c in channel_names]
        recognised = [embedding[i] for i in matched_idx if i is not None]
        if recognised:
            fallback = torch.stack(recognised, dim=0).mean(dim=0)
        else:
            fallback = embedding.mean(dim=0)
        out = torch.stack([
            embedding[i] if i is not None else fallback
            for i in matched_idx
        ], dim=0)
        return out                                       # (C, 3) on pos_bank's device


def _normalize_channel_for_reve(name: str) -> str:
    """Map per-source channel naming quirks onto REVE's pos_bank registry.

    - ZuCo uses ``E001``..``E105`` (EGI HydroCel, zero-padded). REVE expects
      ``E1``..``E105`` (no zero-padding).
    - EMMT (Muse) channels are prefixed ``RAW_`` (e.g. ``RAW_TP9``); REVE
      knows the canonical 10-10 ``TP9``.
    - Other sources (DERCo) already use standard 10-10/10-20 names.
    Anything REVE doesn't recognise gets a zero-position; we keep the channel
    in the tensor anyway so shapes line up with the encoder's input.
    """
    s = name.strip()
    if s.startswith("RAW_"):
        s = s[4:]
    if s.startswith("E0") and s[1:].lstrip("0").isdigit():
        s = "E" + s[1:].lstrip("0")
    return s


# ============================================================================
# TFM-Tokenizer  —  Jathurshan/TFM-Tokenizer
# ============================================================================


class TFMEncoder(EEGEncoder):
    spec = EncoderSpec(feature_dim=64, native_sr=200, discrete=True, codebook_size=8192)

    def __init__(self):
        super().__init__()
        from huggingface_hub import hf_hub_download

        # Source repo holds the model definitions; clone into the HF cache.
        # We deliberately DON'T import from utils/utils.py because that file
        # pulls in pyhealth + matplotlib at module level. Inline get_stft_torch
        # below instead (its definition is identical to upstream).
        src = ensure_tfm_source()
        import sys
        sys.path.insert(0, str(src))
        from models.tfm_token import get_tfm_tokenizer_2x2x8  # type: ignore

        self._stft = _get_stft_torch
        self.tokenizer = get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)
        ckpt = hf_hub_download(
            repo_id="Jathurshan/TFM-Tokenizer",
            filename="pretrained/tfm_tokenizer_last.pth",
            cache_dir=str(storage.HF_CACHE),
        )
        self.tokenizer.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.tokenizer.eval()

    @torch.no_grad()
    def _tokenize_no_grad(self, eeg: torch.Tensor, sr: float):
        from einops import rearrange
        eeg = _resample(eeg, sr, self.spec.native_sr)
        B, C, T = eeg.shape
        x_stft = self._stft(eeg, resampling_rate=self.spec.native_sr)
        x_stft = rearrange(x_stft, "B C F T -> (B C) F T")
        x_temp = rearrange(eeg, "B C T -> (B C) T")
        _, tokens, embs = self.tokenizer.tokenize(x_stft, x_temp)
        # tokens: ((B*C), T_tok); embs: ((B*C), T_tok, D)
        tokens = rearrange(tokens, "(B C) T -> B (C T)", C=C)
        embs = rearrange(embs, "(B C) T D -> B (C T) D", C=C)
        return tokens, embs

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        _, embs = self._tokenize_no_grad(eeg, sr)
        return embs

    def tokenize(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.LongTensor:
        tokens, _ = self._tokenize_no_grad(eeg, sr)
        return tokens.long()


def _get_stft_torch(X: torch.Tensor, resampling_rate: int = 200) -> torch.Tensor:
    """Inline copy of TFM's ``utils.utils.get_stft_torch`` (avoids pyhealth dep).
    X: (B, C, T) at ``resampling_rate`` Hz. Returns: (B, C, F=rs//2, T_stft).
    """
    from einops import rearrange
    B, C, T = X.shape
    x_temp = rearrange(X, "B C T -> (B C) T")
    window = torch.hann_window(resampling_rate).to(x_temp.device)
    x_stft = torch.abs(
        torch.stft(
            x_temp,
            n_fft=resampling_rate,
            hop_length=resampling_rate // 2,
            onesided=True,
            return_complex=True,
            center=False,
            window=window,
        )[:, : resampling_rate // 2, :]
    )
    return rearrange(x_stft, "(B C) F T -> B C F T", B=B)


def ensure_tfm_source() -> Path:
    """Clone the TFM-Tokenizer GitHub repo on first use; cache on the volume."""
    target = storage.HF_CACHE / "src" / "TFM-Tokenizer"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.check_call(
        ["git", "clone", "--depth", "1",
         "https://github.com/Jathurshan0330/TFM-Tokenizer.git", str(target)]
    )
    return target


# ============================================================================
# DIVER-1  —  checkpoint pulled from the cold bucket
# ============================================================================


class DIVER1Encoder(EEGEncoder):
    """Loader stub for DIVER-1.

    The published artifact lives behind an anonymous URL (anonymous.4open.science)
    on Git LFS, which is not API-accessible. Drop the checkpoint and config at:
        $EXP01_DATA_ROOT/diver1/pytorch_model.bin
        $EXP01_DATA_ROOT/diver1/config.json
    Then this loader simply hydrates it. Keeps the run pipeline unchanged.
    """

    spec = EncoderSpec(feature_dim=512, native_sr=None, discrete=False)

    def __init__(self):
        super().__init__()
        ckpt_dir = storage.DATA_ROOT / "diver1"
        ckpt = ckpt_dir / "pytorch_model.bin"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"DIVER-1 checkpoint not found at {ckpt}. "
                "Drop the anonymous-repo weights there before training the diver1 cells."
            )
        # Until upstream releases a HF-loadable form, hold the state dict and
        # let bridges adapt feature_dim from config.
        import json as _json
        cfg_path = ckpt_dir / "config.json"
        cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.spec = EncoderSpec(
            feature_dim=int(cfg.get("hidden_size", 512)),
            native_sr=cfg.get("sampling_rate"),
            discrete=False,
        )
        # Placeholder forward; the run script for the diagonal cell with a
        # released DIVER-1 module would replace this.
        self._state = torch.load(ckpt, map_location="cpu")

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            "Plug DIVER-1's forward here once the upstream module is published. "
            "The state-dict is loaded and ready (self._state)."
        )


# ============================================================================
# Factory + helpers
# ============================================================================


def load_encoder(name: str) -> EEGEncoder:
    if name == "reve":
        return REVEEncoder()
    if name == "tfm":
        return TFMEncoder()
    if name == "diver1":
        return DIVER1Encoder()
    raise ValueError(f"unknown encoder: {name}")


def _resample(eeg: torch.Tensor, src_sr: float, target_sr: int | None) -> torch.Tensor:
    """Linear-interp resample on the time axis. ``target_sr=None`` is a no-op."""
    if target_sr is None or abs(src_sr - target_sr) < 0.5:
        return eeg
    B, C, T = eeg.shape
    new_T = max(1, int(round(T * target_sr / src_sr)))
    return torch.nn.functional.interpolate(eeg, size=new_T, mode="linear", align_corners=False)
