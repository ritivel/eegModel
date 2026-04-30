"""One-shot sentencepiece BPE training script.

Run via ``exp02 build-bpe`` once per data root. Builds
``$EXP02_DATA_ROOT/bpe/spm.model`` from:

  1. ZuCo *training-fold* sentence references (avoids leaking dev / test text
     into the tokenizer vocabulary).
  2. A WikiText-103 subset (~1 M lines) for English-language coverage of
     words ZuCo doesn't see.

The training corpus is also written to ``$EXP02_DATA_ROOT/bpe/corpus.txt``
so it can be inspected and reused (e.g. by the KenLM build).
"""

from __future__ import annotations

from pathlib import Path

from . import storage


# ----------------------------------------------------------------------------
# Corpus assembly
# ----------------------------------------------------------------------------


def _assemble_zuco_corpus(out_path: Path, *, fold: int = 0) -> int:
    """Write ZuCo training-fold sentence references to ``out_path``.

    Reads ``sentence_text`` and ``participant_id`` columns *only* via parquet
    (not :class:`EEGSentenceDataset`, which would load + preprocess the full
    EEG per row → ~30 minutes for ZuCo train). Filters by the same
    train-subject and train-sentence-hash sets the dataset would use.

    Returns the number of unique lines written.
    """
    import pyarrow.parquet as pq

    from eeg_common.data import shard_paths, ZUCO_SOURCES
    from eeg_common.splits import load_fold, sent_hash

    fold_split = load_fold(storage.STORAGE, fold)
    train_subjects = set(fold_split.train_subjects)
    train_sent_hashes = set(fold_split.train_sent_hashes)

    seen: set[str] = set()
    n = 0
    with open(out_path, "a") as f:
        for src in ZUCO_SOURCES:
            for path in shard_paths(storage.STORAGE, src):
                t = pq.read_table(path,
                                  columns=["sentence_text", "participant_id"])
                texts = t["sentence_text"].to_pylist()
                pids = t["participant_id"].to_pylist()
                for text, pid in zip(texts, pids):
                    if not text:
                        continue
                    if str(pid) not in train_subjects:
                        continue
                    if sent_hash(text) not in train_sent_hashes:
                        continue
                    key = " ".join(text.lower().split())
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(text + "\n")
                    n += 1
    return n


def _assemble_wikitext(out_path: Path, *, max_lines: int = 1_000_000) -> int:
    """Stream WikiText-103 (HF datasets) into ``out_path``. Returns lines written."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                      streaming=True, cache_dir=str(storage.HF_CACHE))
    n = 0
    with open(out_path, "a") as f:
        for ex in ds:
            line = (ex.get("text") or "").strip()
            if not line:
                continue
            f.write(line + "\n")
            n += 1
            if n >= max_lines:
                break
    return n


def assemble_corpus(*, fold: int = 0, max_wiki_lines: int = 1_000_000) -> Path:
    """Write the full BPE training corpus and return its path."""
    storage.ensure_dirs()
    corpus_path = storage.BPE_DIR / "corpus.txt"
    if corpus_path.exists():
        corpus_path.unlink()
    n_zuco = _assemble_zuco_corpus(corpus_path, fold=fold)
    n_wiki = _assemble_wikitext(corpus_path, max_lines=max_wiki_lines)
    print(f"[bpe] wrote corpus: {n_zuco} ZuCo + {n_wiki} WikiText lines "
          f"-> {corpus_path}", flush=True)
    return corpus_path


# ----------------------------------------------------------------------------
# Sentencepiece training
# ----------------------------------------------------------------------------


def train_bpe_tokenizer(
    *,
    vocab_size: int = 1024,
    fold: int = 0,
    max_wiki_lines: int = 1_000_000,
    character_coverage: float = 0.9995,
) -> Path:
    """Train a sentencepiece BPE tokenizer and persist it under
    ``$EXP02_DATA_ROOT/bpe/spm.model``.

    Returns the path to the trained model.
    """
    import sentencepiece as spm

    storage.ensure_dirs()
    corpus_path = assemble_corpus(fold=fold, max_wiki_lines=max_wiki_lines)
    model_prefix = str(storage.BPE_DIR / "spm")

    print(f"[bpe] training sentencepiece BPE-{vocab_size} on {corpus_path}",
          flush=True)
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        # Emit lowercase, NFKC-normalised pieces. ZuCo refs are mixed-case
        # but we lowercase at encode time anyway (CTC vocab is case-folded).
        normalization_rule_name="nmt_nfkc_cf",
        # Standard reserved IDs. We don't use BOS / EOS / PAD in the CTC
        # path — only <unk>. Setting their IDs to -1 disables them.
        unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1,
        # Don't add a leading whitespace marker to the first token. CTC's
        # output is reassembled char-by-char so the marker is noise.
        add_dummy_prefix=False,
    )
    out = Path(model_prefix + ".model")
    print(f"[bpe] wrote {out} (vocab_size={vocab_size})", flush=True)
    return out
