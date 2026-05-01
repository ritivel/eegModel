"""Encoder loaders with a uniform ``(B, C, T) -> (B, T_seq, D)`` interface.

Three encoders are wired:
  REVE     — ``brain-bzh/reve-base``     (gated, requires HF_TOKEN + EUA)
  TFM      — ``Jathurshan/TFM-Tokenizer`` (open)
  DIVER-1  — checkpoint pulled from the cold bucket

Each encoder advertises:
  ``feature_dim`` (D)              — encoder hidden size
  ``native_sr``  (Hz | None)       — required input sampling rate, None=any
  ``encode(eeg, sr, channels)`` -> Tensor of shape (B, T_seq, D)

Discrete encoders (TFM by default; REVE / DIVER-1 if attached to an RVQ head)
also expose ``tokenize`` -> LongTensor (B, T_seq) for vocab-extension bridges.

Encoders also expose ``attach_lora()`` so a trainer can unfreeze a small
adapter on the encoder's attention projections (Wav2Vec2-→-brain transfer
recipe; `arXiv 2501.09459 <https://arxiv.org/abs/2501.09459>`_ §3.2 Fig 5
shows full fine-tune > frozen by 15-20% absolute CER), and ``unfreeze_all``
for full encoder fine-tuning.

All loaders take a :class:`eeg_common.storage.Storage` argument so different
experiments can use different HF cache locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .storage import Storage


# ============================================================================
# Common interface
# ============================================================================


@dataclass
class EncoderSpec:
    feature_dim: int
    native_sr: int | None
    discrete: bool = False
    codebook_size: int | None = None


class EEGEncoder(nn.Module):
    spec: EncoderSpec

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        """eeg: (B, C, T) at ``sr`` Hz. Returns (B, T_seq, D)."""
        raise NotImplementedError

    def tokenize(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.LongTensor:
        raise NotImplementedError(f"{type(self).__name__} is not discrete")

    def attach_lora(self, *, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> list:
        """LoRA on attention projections. Default no-op; subclasses override."""
        return []

    def unfreeze_all(self) -> list:
        """Mark every encoder parameter as trainable; return the trainable list.

        Used for the full-fine-tune cell in exp02 (Wav2Vec2-style end-to-end
        ASR where the encoder is part of the optimisation, not a frozen
        feature extractor).
        """
        for p in self.parameters():
            p.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]


# ============================================================================
# REVE  —  brain-bzh/reve-base
# ============================================================================


class REVEEncoder(EEGEncoder):
    spec = EncoderSpec(feature_dim=512, native_sr=200, discrete=False)

    def __init__(self, storage: Storage, model_id: str = "brain-bzh/reve-base"):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=str(storage.hf_cache),
        )
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", trust_remote_code=True,
            cache_dir=str(storage.hf_cache),
        )

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        eeg = _resample(eeg, sr, self.spec.native_sr)
        norm_channels = [_normalize_channel_for_reve(c) for c in channels]
        positions = self._safe_positions(norm_channels)
        positions = positions.to(eeg.device).unsqueeze(0).expand(eeg.size(0), -1, -1)
        out = self.model(eeg, positions)
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

    def attach_lora(self, *, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> list:
        """Inject PEFT LoRA on REVE's attention projections.

        Targets ``transformer.layers.{i}.0.{to_qkv,to_out}`` (22 layers × 2 =
        44 adapter pairs). At r=8 ≈ 1 M trainable params vs REVE's 69 M total.

        Reference: `arXiv 2501.09459 <https://arxiv.org/abs/2501.09459>`_ §3.2.
        """
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=None,
            target_modules=r".*transformer\.layers\.\d+\.0\.(to_qkv|to_out)$",
        )
        wrapped = get_peft_model(self.model, cfg)
        self.model = wrapped
        new_params = [p for p in wrapped.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in new_params)
        n_total = sum(p.numel() for p in wrapped.parameters())
        print(f"[REVE-LoRA] r={r} alpha={alpha} dropout={dropout}; "
              f"trainable {n_trainable:,} / {n_total:,} "
              f"({100.0 * n_trainable / max(1, n_total):.2f}%)", flush=True)
        return new_params

    def _safe_positions(self, channel_names: list[str]) -> torch.Tensor:
        mapping = self.pos_bank.mapping
        embedding = self.pos_bank.embedding
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
        return out


def _normalize_channel_for_reve(name: str) -> str:
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

    def __init__(self, storage: Storage):
        super().__init__()
        from huggingface_hub import hf_hub_download

        src = ensure_tfm_source(storage)
        import sys
        sys.path.insert(0, str(src))
        from models.tfm_token import get_tfm_tokenizer_2x2x8  # type: ignore

        self._stft = _get_stft_torch
        self.tokenizer = get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)
        ckpt = hf_hub_download(
            repo_id="Jathurshan/TFM-Tokenizer",
            filename="pretrained/tfm_tokenizer_last.pth",
            cache_dir=str(storage.hf_cache),
        )
        self.tokenizer.load_state_dict(torch.load(ckpt, map_location="cpu"))
        # NOTE: do NOT call self.tokenizer.eval() here. The eval-mode flag is
        # automatically toggled by ``model.train()`` / ``model.eval()`` from
        # the trainer; forcing eval at construction time used to leak into
        # training and combined with the @torch.no_grad() decorator below
        # silently froze the entire encoder (see findings.md §2.1).

    def _tokenize(self, eeg: torch.Tensor, sr: float):
        """Continuous + discrete forward through the tokenizer.

        Respects ``self.training`` and the autograd context of the caller.
        Wrap externally with ``torch.no_grad()`` if a non-trainable inference
        path is desired.
        """
        from einops import rearrange
        eeg = _resample(eeg, sr, self.spec.native_sr)
        B, C, T = eeg.shape
        x_stft = self._stft(eeg, resampling_rate=self.spec.native_sr)
        x_stft = rearrange(x_stft, "B C F T -> (B C) F T")
        x_temp = rearrange(eeg, "B C T -> (B C) T")
        _, tokens, embs = self.tokenizer.tokenize(x_stft, x_temp)
        tokens = rearrange(tokens, "(B C) T -> B (C T)", C=C)
        embs = rearrange(embs, "(B C) T D -> B (C T) D", C=C)
        return tokens, embs

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        _, embs = self._tokenize(eeg, sr)
        return embs

    def tokenize(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.LongTensor:
        with torch.no_grad():
            tokens, _ = self._tokenize(eeg, sr)
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


def ensure_tfm_source(storage: Storage) -> Path:
    """Clone the TFM-Tokenizer GitHub repo on first use; cache on disk."""
    target = storage.hf_cache / "src" / "TFM-Tokenizer"
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

    The published artifact lives behind an anonymous URL on Git LFS, which is
    not API-accessible. Drop the checkpoint and config at::

        <storage.data_root>/diver1/pytorch_model.bin
        <storage.data_root>/diver1/config.json

    Then this loader hydrates it. Keeps the run pipeline unchanged.
    """

    spec = EncoderSpec(feature_dim=512, native_sr=None, discrete=False)

    def __init__(self, storage: Storage):
        super().__init__()
        ckpt_dir = storage.data_root / "diver1"
        ckpt = ckpt_dir / "pytorch_model.bin"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"DIVER-1 checkpoint not found at {ckpt}. "
                "Drop the anonymous-repo weights there before training the diver1 cells."
            )
        import json as _json
        cfg_path = ckpt_dir / "config.json"
        cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        self.spec = EncoderSpec(
            feature_dim=int(cfg.get("hidden_size", 512)),
            native_sr=cfg.get("sampling_rate"),
            discrete=False,
        )
        self._state = torch.load(ckpt, map_location="cpu")

    def encode(self, eeg: torch.Tensor, sr: float, channels: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            "Plug DIVER-1's forward here once the upstream module is published. "
            "The state-dict is loaded and ready (self._state)."
        )


# ============================================================================
# Factory + helpers
# ============================================================================


def load_encoder(name: str, storage: Storage) -> EEGEncoder:
    if name == "reve":
        return REVEEncoder(storage)
    if name == "tfm":
        return TFMEncoder(storage)
    if name == "diver1":
        return DIVER1Encoder(storage)
    raise ValueError(f"unknown encoder: {name}")


def _resample(eeg: torch.Tensor, src_sr: float, target_sr: int | None) -> torch.Tensor:
    """Linear-interp resample on the time axis. ``target_sr=None`` is a no-op."""
    if target_sr is None or abs(src_sr - target_sr) < 0.5:
        return eeg
    B, C, T = eeg.shape
    new_T = max(1, int(round(T * target_sr / src_sr)))
    return torch.nn.functional.interpolate(eeg, size=new_T, mode="linear", align_corners=False)
