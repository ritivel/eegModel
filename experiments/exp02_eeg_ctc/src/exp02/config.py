"""Per-cell run configuration for exp02.

A cell is uniquely identified by ``(encoder, vocab, ctc_variant, input, fold)``.
Cell ID format::

    <encoder>_<vocab>_<variant>_<input>_fold<n>[_frozen]

Examples::

    reve_bpe1k_crctc_eeg_fold0
    reve_bpe1k_crctc_noise_train_fold0
    reve_char_ctc_eeg_fold0
    reve_bpe1k_ctcaed_eeg_fold0_frozen

A short ``cfg_key`` form is used by the CLI: ``encoder.vocab.variant.input.fold``
e.g. ``reve.bpe1k.crctc.eeg.0``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

# ---- Closed sets -------------------------------------------------------------

Encoder = Literal["reve", "diver1", "tfm"]
Vocab = Literal["char", "bpe1k"]
CTCVariant = Literal["ctc", "crctc", "intctc", "ctcaed"]
Input = Literal["eeg", "noise_train", "noise_test"]


# ---- Cell --------------------------------------------------------------------


@dataclass(frozen=True)
class CTCConfig:
    """One trainable cell. Defaults reproduce the headline recipe:
    REVE + full fine-tune + BPE-1k + CR-CTC + SpecAugment + 12k steps +
    beam-search + KenLM rescore at decode.
    """

    encoder: Encoder = "reve"
    vocab: Vocab = "bpe1k"
    variant: CTCVariant = "crctc"
    input: Input = "eeg"
    fold: int = 0

    # ----------- preprocessing -----------
    # Encoder-aware: ``v2`` resolves to v2_reve / v2_tfm / v2_dk25.
    preprocess: Literal["v1", "v2", "v2_reve", "v2_tfm", "v2_dk25"] = "v2"

    # ----------- encoder fine-tune -----------
    # Default is FULL end-to-end fine-tune of the encoder (Wav2Vec2 standard,
    # arXiv 2501.09459 §3.2 Fig 5: full > frozen by 15-20% absolute CER on
    # brain decoding). Override to ``frozen`` for the GROUP E ablation.
    encoder_finetune: Literal["full", "lora", "frozen"] = "full"
    encoder_lora_r: int = 8
    encoder_lora_alpha: int = 16
    encoder_lora_dropout: float = 0.05

    # First N steps the encoder is held frozen even for full / lora cells.
    # (Omnilingual ASR pattern: lets the head learn against stable encoder
    # features before the encoder starts moving.) Default is 10% of steps.
    encoder_warmup_freeze_steps: int = 1_200

    # ----------- training schedule -----------
    total_steps: int = 12_000
    warmup_steps: int = 1_000
    head_lr: float = 1e-3
    encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 2
    batch_size: int = 16
    seed: int = 1234
    precision: Literal["bf16", "fp32"] = "bf16"

    # ----------- CTC head -----------
    head_hidden: int = 512
    head_layers: int = 4
    head_heads: int = 8
    head_dropout: float = 0.1

    # ----------- CR-CTC (Yao et al. ICLR 2025) -----------
    # Two SpecAugmented views; KL between their CTC distributions.
    # Active iff variant in {"crctc"}; the trainer runs two forward passes.
    cr_ctc_kl_weight: float = 1.0
    cr_ctc_temperature: float = 1.0

    # ----------- Intermediate CTC (Komatsu 2022) -----------
    # Auxiliary CTC loss applied at intermediate head layers.
    # Active iff variant in {"intctc"}.
    intermediate_ctc_layers: tuple[int, ...] = (1, 2)
    intermediate_ctc_weight: float = 0.3

    # ----------- CTC + AED hybrid (Watanabe 2017) -----------
    # Joint loss: λ * CTC + (1-λ) * AED-cross-entropy.
    # Active iff variant in {"ctcaed"}.
    aed_weight: float = 0.3
    aed_layers: int = 4
    aed_heads: int = 8
    aed_dropout: float = 0.1
    aed_max_target_len: int = 96

    # ----------- Label-prior CTC (Zeyer 2021 §7) -----------
    # Subtract a learned label prior from logits to fight peaky behavior.
    # Cheap regulariser; on by default for vanilla CTC, off for CR-CTC
    # (which has its own anti-peakyness mechanism).
    label_prior_weight: float = 0.0  # set to 0.3 for the vanilla "ctc" variant
    label_prior_ema: float = 0.99

    # ----------- SpecAugment -----------
    specaugment: bool = True
    specaug_n_time_masks: int = 2
    specaug_time_mask_ms: int = 200
    specaug_n_chan_masks: int = 2
    specaug_chan_mask_max: int = 8

    # ----------- Decoder (eval-time) -----------
    decode_beam_width: int = 50
    decode_kenlm_alpha: float = 0.5
    decode_kenlm_beta: float = 1.5

    # ----------- Bookkeeping -----------
    num_workers: int = 4
    log_every: int = 10
    sample_every_frac: float = 0.1   # log dev samples every 10% of training
    save_every_frac: float = 0.5
    split_seed: int = 20260430

    # -----------------------------------------------------------------
    # Derived
    # -----------------------------------------------------------------

    @property
    def cell_id(self) -> str:
        prep = "" if self.preprocess == "v1" else f"_pp-{self.preprocess.replace('_', '-')}"
        suffix = ""
        if self.encoder_finetune == "frozen":
            suffix += "_frozen"
        elif self.encoder_finetune == "lora":
            suffix += "_elora"
        if not self.specaugment:
            suffix += "_nosa"
        return (f"{self.encoder}_{self.vocab}_{self.variant}_{self.input}"
                f"_fold{self.fold}{prep}{suffix}")

    @property
    def cfg_key(self) -> str:
        """Round-trip key for `exp02 train CFG_KEY` etc."""
        return f"{self.encoder}.{self.vocab}.{self.variant}.{self.input}.{self.fold}"

    def to_dict(self) -> dict:
        return asdict(self)


# ---- Run matrix --------------------------------------------------------------
#
# Default Track-C scope: 14 distinct training cells + 10 fold-extension cells
# = 24 cells total. See ``design.md`` for the full ablation matrix.


def _eeg_and_noise(cfg_eeg: CTCConfig) -> list[CTCConfig]:
    """Helper: emit the EEG cell and its matched noise twin (Jo §4.3)."""
    from dataclasses import replace
    return [cfg_eeg, replace(cfg_eeg, input="noise_train")]


def headline_cells() -> list[CTCConfig]:
    """GROUP A — REVE + BPE-1k + CR-CTC + matched §4.3 noise twin."""
    return _eeg_and_noise(CTCConfig(encoder="reve", vocab="bpe1k", variant="crctc"))


def encoder_ablation_cells(include_diver1: bool = False) -> list[CTCConfig]:
    """GROUP B — vary the encoder."""
    out: list[CTCConfig] = []
    out += _eeg_and_noise(CTCConfig(encoder="tfm", vocab="bpe1k", variant="crctc"))
    if include_diver1:
        out += _eeg_and_noise(CTCConfig(encoder="diver1", vocab="bpe1k", variant="crctc"))
    return out


def vocab_ablation_cells() -> list[CTCConfig]:
    """GROUP C — REVE + char vocab (vs default BPE-1k)."""
    return _eeg_and_noise(CTCConfig(encoder="reve", vocab="char", variant="crctc"))


def variant_ablation_cells() -> list[CTCConfig]:
    """GROUP D — vanilla CTC + intermediate CTC + CTC+AED hybrid."""
    out: list[CTCConfig] = []
    # vanilla CTC: turn ON label prior (anti-peakyness) to make this a fair
    # baseline against CR-CTC.
    from dataclasses import replace
    vanilla = CTCConfig(encoder="reve", vocab="bpe1k", variant="ctc")
    vanilla = replace(vanilla, label_prior_weight=0.3, cr_ctc_kl_weight=0.0)
    out += _eeg_and_noise(vanilla)
    out += _eeg_and_noise(CTCConfig(encoder="reve", vocab="bpe1k", variant="intctc"))
    out += _eeg_and_noise(CTCConfig(encoder="reve", vocab="bpe1k", variant="ctcaed"))
    return out


def freeze_ablation_cells() -> list[CTCConfig]:
    """GROUP E — REVE permanently frozen, head-only training."""
    from dataclasses import replace
    base = CTCConfig(encoder="reve", vocab="bpe1k", variant="crctc")
    base = replace(base, encoder_finetune="frozen",
                   encoder_warmup_freeze_steps=10**9)
    return _eeg_and_noise(base)


def fold_extension_cells(survivor: CTCConfig, *, n_folds: int = 5) -> list[CTCConfig]:
    """GROUP F — extend the survivor cell across folds 0..n_folds-1."""
    from dataclasses import replace
    out: list[CTCConfig] = []
    for f in range(n_folds):
        out += _eeg_and_noise(replace(survivor, fold=f))
    return out


def all_track_c_cells(include_diver1: bool = False) -> list[CTCConfig]:
    """Full Track-C scope: groups A + B + C + D + E. ~14 cells.

    GROUP F (5-fold extension) is decided after a survivor is identified, so
    it's not part of the default sweep.
    """
    return (headline_cells()
            + encoder_ablation_cells(include_diver1=include_diver1)
            + vocab_ablation_cells()
            + variant_ablation_cells()
            + freeze_ablation_cells())
