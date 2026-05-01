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
    # ``transformer`` (default): randomly-initialised TransformerEncoder stack.
    # ``lm_bridge``: a pretrained text Transformer (DistilBERT by default)
    # used as the bridge. Brings English priors that a from-scratch head
    # would need 100M+ training tokens to acquire on its own; see
    # ``exp02/lm_bridge_head.py`` and ``findings.md`` §2.4.
    head_type: Literal["transformer", "lm_bridge"] = "transformer"
    # Used by ``lm_bridge``: HuggingFace model id of the bridge transformer.
    head_lm_model_id: str = "distilbert-base-uncased"
    # REVE outputs (B, C * T_p, D) with C=105 channels and T_p depending on
    # input duration; this can exceed BERT's standard 512 cap. 8192 covers
    # 12-sec EEG at REVE's per-channel downsampling and is essentially free
    # (fixed sinusoidal buffer).
    head_lm_max_seq_len: int = 8192

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

    # ----------- SpecAugment (numpy-side, per-row, deterministic per row) -----------
    specaugment: bool = True
    specaug_n_time_masks: int = 2
    specaug_time_mask_ms: int = 200
    specaug_n_chan_masks: int = 2
    specaug_chan_mask_max: int = 8

    # ----------- Signal augmentation (GPU-side, per-step, stochastic) -----------
    # Each augmentation is applied with probability ``*_p`` (or scaled by its
    # magnitude) per training step. Defaults are OFF so wave-1 keeps its
    # current behaviour; wave-2 cells enable subsets via CLI flags.
    # See ``eeg_common.augment`` for the per-augmentation references.
    signal_aug_time_shift_max_frac: float = 0.0   # Brain Transformer 2025; recommended 0.05
    signal_aug_channel_dropout_p: float = 0.0     # Strumiłło 2026; recommended 0.5 with frac=0.1
    signal_aug_channel_dropout_frac: float = 0.1
    signal_aug_freq_mask_p: float = 0.0           # FFT-band mask; recommended 0.5 with max_hz=8
    signal_aug_freq_mask_n: int = 2
    signal_aug_freq_mask_max_hz: float = 8.0
    signal_aug_time_warp_p: float = 0.0           # Xu 2026; recommended 0.3
    signal_aug_time_warp_segments: int = 10
    signal_aug_time_warp_factor_low: float = 0.6
    signal_aug_time_warp_factor_high: float = 1.7
    signal_aug_gaussian_noise_sigma: float = 0.0  # additive on z-scored input; recommended 0.05
    signal_aug_fourier_surrogate_p: float = 0.0   # Strumiłło 2026; recommended 0.2
    signal_aug_mixup_alpha: float = 0.0           # Beta(α,α) lambda; recommended 0.2-1.0

    # ----------- Text-target augmentation (LLM paraphrase substitution) -----------
    # With probability ``text_aug_prob``, the trainer substitutes a random
    # paraphrase of the reference text as the CTC target. Build the
    # paraphrase parquet once via ``exp02 build-paraphrases``.
    # Default empty path / 0 prob = disabled.
    text_aug_paraphrase_path: str = ""
    text_aug_prob: float = 0.0

    # ----------- Data quality filters (May 1 audit, see findings.md §2.3) -----------
    # ``drop_sources``: comma-joined list, e.g. "derco_preprocessed,emmt".
    # The audit found DERCo (95% truncated by max_seconds) and EMMT
    # (4-channel, 96% zero-padded when batched with ZuCo) net-harmful.
    drop_sources: str = ""
    # Filter out rows whose label text length is outside the sane range.
    min_text_chars: int = 0
    max_text_chars: int = 0  # 0 disables the upper cap
    # Drop rows whose EEG duration exceeds this cap (rather than silently
    # truncating in the collator).
    max_seconds: float = 0.0
    drop_nan_rows: bool = False
    drop_zero_rows: bool = False

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

    # Free-form short tag appended to the cell_id (e.g. ``"aug"``,
    # ``"big-batch"``). Lets wave-2 augmented cells live alongside wave-1
    # baselines without colliding on the run / eval dirs.
    tag: str = ""

    # -----------------------------------------------------------------
    # Derived
    # -----------------------------------------------------------------

    @property
    def cell_id(self) -> str:
        prep = "" if self.preprocess == "v1" else f"_pp-{self.preprocess.replace('_', '-')}"
        suffix = ""
        if self.head_type != "transformer":
            suffix += f"_h-{self.head_type.replace('_', '-')}"
        if self.encoder_finetune == "frozen":
            suffix += "_frozen"
        elif self.encoder_finetune == "lora":
            suffix += "_elora"
        if not self.specaugment:
            suffix += "_nosa"
        if self.tag:
            suffix += f"_{self.tag}"
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

    NOTE (May 1 2026): TFM is excluded from the default sweep until the
    encoder-frozen bug in ``packages/eeg_common/src/eeg_common/encoders.py``
    is verified end-to-end (see ``findings.md`` §2.1). Pass
    ``include_diver1=True`` only when the DIVER-1 weights are present.
    """
    return (headline_cells()
            + vocab_ablation_cells()
            + variant_ablation_cells()
            + freeze_ablation_cells()
            + (encoder_ablation_cells(include_diver1=True) if include_diver1 else []))


# ============================================================================
# Wave-3 — May 1 2026, post-audit launch matrix
# ============================================================================
#
# Designed to factor the four wave-1 root causes (see ``findings.md``):
#   (1) data quality          — drop_sources + length filters + NaN/zero filter
#   (2) data sufficiency      — paraphrase text augmentation
#   (3) signal regularisation — GPU-side signal augmentations
#   (4) no English priors     — DistilBERT bridge as the head transformer
#
# 9 cells total, fanned across both boxes via the parallel orchestrator.

_DROP_SOURCES_DEFAULT = "derco_preprocessed,emmt"  # see findings.md §2.3
_PARAPHRASE_DEFAULT = "/home/ubuntu/data/exp02/text_aug/paraphrases.parquet"


def _clean_data_kwargs() -> dict:
    """Defaults applied to wave-3 'cleaned data' cells."""
    return dict(
        drop_sources=_DROP_SOURCES_DEFAULT,
        min_text_chars=10,
        max_text_chars=800,
        max_seconds=12.0,
        drop_nan_rows=True,
        drop_zero_rows=True,
    )


def _paraphrase_kwargs(prob: float = 0.5) -> dict:
    return dict(
        text_aug_prob=prob,
        text_aug_paraphrase_path=_PARAPHRASE_DEFAULT,
    )


def _signal_aug_kwargs() -> dict:
    """The wave-2 ``aug-signal`` recipe — 6 GPU-side augmentations."""
    return dict(
        signal_aug_time_shift_max_frac=0.05,
        signal_aug_channel_dropout_p=0.5,
        signal_aug_channel_dropout_frac=0.1,
        signal_aug_freq_mask_p=0.5,
        signal_aug_freq_mask_max_hz=8.0,
        signal_aug_time_warp_p=0.3,
        signal_aug_fourier_surrogate_p=0.2,
        signal_aug_mixup_alpha=0.4,
    )


def wave3_cells() -> list[CTCConfig]:
    """Wave-3 launch matrix — 9 cells across 9 GPUs.

    Pairings (4 EEG + 4 noise on Box A, 1 EEG (Box B) → noise sequential):

    | tag         | head        | data    | text-aug | signal-aug | hypothesis |
    | ----------- | ----------- | ------- | -------- | ---------- | --- |
    | clean       | transformer | cleaned | off      | off        | data quality alone moves the needle |
    | lm-clean    | lm_bridge   | cleaned | off      | off        | LM bridge alone moves the needle (no aug confound) |
    | lm-aug      | lm_bridge   | cleaned | 0.5      | on         | full stack — HEADLINE |
    | aug-clean   | transformer | cleaned | 0.5      | on         | aug helps WITHOUT lm bridge — controls for cell 3's lm contribution |
    | char-lm-aug | lm_bridge + char | cleaned | 0.5 | on        | char vocab beats BPE-1k under LM bridge (Box B) |

    Each EEG cell has a matched ``noise_train`` twin per Jo §4.3.
    """
    from dataclasses import replace
    base = CTCConfig(encoder="reve", vocab="bpe1k", variant="crctc")
    cells: list[CTCConfig] = []

    # 1+2: clean data only (transformer head, no aug)  ←  data-quality control
    c = replace(base, tag="clean", **_clean_data_kwargs())
    cells += _eeg_and_noise(c)

    # 3+4: lm-bridge only (no aug)  ←  isolated LM-bridge contribution
    c = replace(base, tag="lm-clean", head_type="lm_bridge",
                **_clean_data_kwargs())
    cells += _eeg_and_noise(c)

    # 5+6: HEADLINE — lm-bridge + paraphrase + signal aug + cleaned data
    c = replace(base, tag="lm-aug", head_type="lm_bridge",
                **_clean_data_kwargs(), **_paraphrase_kwargs(0.5),
                **_signal_aug_kwargs())
    cells += _eeg_and_noise(c)

    # 7+8: aug WITHOUT lm bridge — controls for lm contribution of cell 5
    c = replace(base, tag="aug-clean",
                **_clean_data_kwargs(), **_paraphrase_kwargs(0.5),
                **_signal_aug_kwargs())
    cells += _eeg_and_noise(c)

    # 9: lm-bridge + char vocab + everything (Box B's slot; sequential noise)
    c = replace(base, vocab="char", tag="char-lm-aug", head_type="lm_bridge",
                **_clean_data_kwargs(), **_paraphrase_kwargs(0.5),
                **_signal_aug_kwargs())
    cells += _eeg_and_noise(c)

    return cells
