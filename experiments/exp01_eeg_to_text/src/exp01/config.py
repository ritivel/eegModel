"""Per-cell run configuration.

The §5 run matrix is *3 encoders × 3 bridges × 3 input conditions × 5 LNSO folds*.
Each unique (encoder, bridge, input, fold) is one ``CellConfig``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# ---- Closed sets -------------------------------------------------------------

Encoder = Literal["reve", "diver1", "tfm"]
Bridge = Literal["linear", "qformer", "vocab"]
Input = Literal["eeg", "noise_train", "noise_test"]  # Jo et al. (2024) §4.3

# ---- Model registry ----------------------------------------------------------
# Names are the only place where remote IDs live. Swap them once,
# everything else picks up the new path.

ENCODER_HF = {
    "reve": "brain-bzh/reve-base",
    "tfm": "Jathurshan/TFM-Tokenizer",
    # DIVER-1 has no HF mirror; weights are pulled from the anon repo via a
    # tarball stored on the cold bucket (cold path resolved by encoders.py).
    "diver1": None,
}

# Default decoder per encoder (overridable from CellConfig).
DECODER_HF = {
    "reve": "google/gemma-4-E2B-it",
    "diver1": "google/gemma-4-E2B-it",
    "tfm": "google/gemma-4-E2B-it",
}

# Pre-registered diagonal of §2.3.
DIAGONAL: dict[Encoder, Bridge] = {
    "reve": "linear",
    "diver1": "qformer",
    "tfm": "vocab",
}


# ---- Cell --------------------------------------------------------------------


@dataclass(frozen=True)
class CellConfig:
    encoder: Encoder
    bridge: Bridge
    input: Input = "eeg"
    fold: int = 0  # 0..4 (5-fold LNSO)

    # Decoder choice
    decoder: str = "google/gemma-4-E2B-it"
    use_lora_in_stage3: bool = True

    # Bridge-specific knobs
    qformer_queries: int = 32
    rvq_codebook: int = 8192  # for off-diagonal vocab-extension cells

    # Stage-1 modality alignment (encoder + LM frozen)
    stage1_steps: int = 2_000
    stage1_lr: float = 1e-4

    # Stage-2 frozen-LM SFT
    stage2_steps: int = 6_000
    stage2_lr: float = 5e-5

    # Stage-3 LoRA SFT (optional)
    stage3_steps: int = 4_000
    stage3_lr: float = 2e-5
    lora_r: int = 16
    lora_alpha: int = 32

    # Optimisation
    batch_size: int = 8
    grad_accum: int = 4
    seed: int = 1234

    # Activation checkpointing trades ~30-40% throughput for ~50% less memory.
    # Soft-prompt cells (linear / qformer) fit comfortably on 80 GB H100 with
    # bs=8 *without* checkpointing — turn it off for those. Vocab cells need
    # it because the trainable extended embedding table eats 5+ GB extra of
    # AdamW state.
    use_gradient_checkpointing: bool = True

    # DataLoader worker processes. Bumped from 2 -> 4 because larger batches
    # need more parallel parquet decoding to keep the GPU fed.
    num_workers: int = 4

    # Reproducibility
    split_seed: int = 20260430

    @property
    def cell_id(self) -> str:
        return f"{self.encoder}_{self.bridge}_{self.input}_fold{self.fold}_dec-{self._dec_short()}"

    @property
    def cfg_key(self) -> str:
        """Round-trip key for `exp01 train CFG_KEY` etc."""
        return f"{self.encoder}.{self.bridge}.{self.input}.{self.fold}"

    def _dec_short(self) -> str:
        # google/gemma-4-E2B-it -> gemma4-e2b
        s = self.decoder.split("/")[-1].lower()
        return s.replace("gemma-4-", "gemma4-").replace("-it", "")

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Run matrix --------------------------------------------------------------


def all_cells(
    encoders: tuple[Encoder, ...] = ("reve", "diver1", "tfm"),
    bridges: tuple[Bridge, ...] = ("linear", "qformer", "vocab"),
    inputs: tuple[Input, ...] = ("eeg", "noise_train", "noise_test"),
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    decoder: str = "google/gemma-4-E2B-it",
) -> list[CellConfig]:
    """Cartesian product producing the §5 evaluation cells.

    Note: ``noise_test`` reuses the trained ``eeg`` checkpoint at eval time
    (see eval.py); we still emit a cell so the result is recorded.
    """
    out = []
    for e in encoders:
        for b in bridges:
            for i in inputs:
                for f in folds:
                    out.append(CellConfig(encoder=e, bridge=b, input=i, fold=f, decoder=decoder))
    return out


def pilot_cells(
    decoder: str = "google/gemma-4-E2B-it",
    encoders: tuple[Encoder, ...] = ("reve", "diver1", "tfm"),
) -> list[CellConfig]:
    """§5.1 pilot phase: fold 0, EEG only, all (encoder × bridge) cells.

    Default ``encoders`` reproduces the report's full 9-cell pilot. Pass a
    subset (e.g. ``("reve", "tfm")``) to skip encoders whose weights aren't
    cached yet.
    """
    return [
        CellConfig(encoder=e, bridge=b, input="eeg", fold=0, decoder=decoder)
        for e in encoders
        for b in ("linear", "qformer", "vocab")
    ]
