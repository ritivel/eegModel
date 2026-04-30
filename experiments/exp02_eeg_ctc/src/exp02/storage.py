"""exp02 storage — bound to ``$EXP02_DATA_ROOT``.

Layout::

    $EXP02_DATA_ROOT/
      hf/            HuggingFace cache (REVE / TFM / dataset parquets)
      splits/        fold JSONs (Yin unique-sentence + LNSO)
      runs/          training checkpoints + jsonl logs
      eval/          eval metrics + predictions parquet
      wandb/         local wandb run dirs

      bpe/spm.model        sentencepiece BPE-1k (built by ``exp02 build-bpe``)
      bpe/spm.vocab        sentencepiece vocab listing
      kenlm/4gram.arpa     KenLM 4-gram (text form, built by ``exp02 build-kenlm``)
      kenlm/4gram.binary   KenLM 4-gram (binary form, faster to load)
      kenlm/corpus.txt     concatenated training-side corpus the KenLM was built on
"""

from __future__ import annotations

from pathlib import Path

from eeg_common import storage as _common
from eeg_common.storage import Storage

STORAGE: Storage = _common.from_env("EXP02_DATA_ROOT")

DATA_ROOT: Path = STORAGE.data_root
HF_CACHE: Path = STORAGE.hf_cache
WANDB_DIR: Path = STORAGE.wandb_dir
SPLITS: Path = STORAGE.splits
RUNS: Path = STORAGE.runs
EVAL: Path = STORAGE.eval

# CTC-specific shared artifacts (one set per data root, not one per cell).
BPE_DIR: Path = DATA_ROOT / "bpe"
BPE_MODEL: Path = BPE_DIR / "spm.model"
BPE_VOCAB: Path = BPE_DIR / "spm.vocab"

KENLM_DIR: Path = DATA_ROOT / "kenlm"
KENLM_ARPA: Path = KENLM_DIR / "4gram.arpa"
KENLM_BINARY: Path = KENLM_DIR / "4gram.binary"
KENLM_CORPUS: Path = KENLM_DIR / "corpus.txt"


def ensure_dirs() -> None:
    STORAGE.ensure_dirs()
    BPE_DIR.mkdir(parents=True, exist_ok=True)
    KENLM_DIR.mkdir(parents=True, exist_ok=True)


def cell_run_dir(cell_id: str) -> Path:
    return STORAGE.cell_run_dir(cell_id)


def cell_eval_dir(cell_id: str) -> Path:
    return STORAGE.cell_eval_dir(cell_id)
