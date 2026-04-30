"""One-shot KenLM 4-gram build.

Builds a word-level KenLM 4-gram from the same training-side text corpus
that fed BPE training (ZuCo train-fold sentence references + WikiText-103),
plus an optional cap on the number of WikiText lines for speed.

The build uses the ``kenlm`` system binaries (``lmplz`` + ``build_binary``).
If those aren't available on PATH, the build falls back to a pure-Python
``kenlm.Model`` builder via ``arpa`` if installed, or fails with a clear
message pointing the user at https://github.com/kpu/kenlm.

Usage::

    exp02 build-kenlm --order 4 --max-wiki-lines 1000000

Output::

    $EXP02_DATA_ROOT/kenlm/corpus.txt
    $EXP02_DATA_ROOT/kenlm/4gram.arpa
    $EXP02_DATA_ROOT/kenlm/4gram.binary    (only if build_binary is available)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import storage
from .tokenizer_build import _assemble_wikitext, _assemble_zuco_corpus


# ----------------------------------------------------------------------------
# Corpus assembly (re-uses tokenizer_build helpers but writes to KENLM_DIR)
# ----------------------------------------------------------------------------


def assemble_kenlm_corpus(*, fold: int = 0, max_wiki_lines: int = 1_000_000) -> Path:
    """Write the KenLM training corpus and return its path."""
    storage.ensure_dirs()
    out = storage.KENLM_CORPUS
    if out.exists():
        out.unlink()
    n_zuco = _assemble_zuco_corpus(out, fold=fold)
    n_wiki = _assemble_wikitext(out, max_lines=max_wiki_lines)
    print(f"[kenlm] wrote corpus: {n_zuco} ZuCo + {n_wiki} WikiText lines "
          f"-> {out}", flush=True)
    return out


# ----------------------------------------------------------------------------
# Binary-tooling KenLM build
# ----------------------------------------------------------------------------


def _have_kenlm_binaries() -> bool:
    return shutil.which("lmplz") is not None


def build_kenlm(
    *,
    order: int = 4,
    fold: int = 0,
    max_wiki_lines: int = 1_000_000,
    discount_fallback: bool = True,
) -> Path:
    """Build a KenLM ``order``-gram language model.

    Returns the path to the binary KenLM model (or the ARPA file if
    ``build_binary`` isn't available).
    """
    storage.ensure_dirs()
    corpus = assemble_kenlm_corpus(fold=fold, max_wiki_lines=max_wiki_lines)
    arpa = storage.KENLM_DIR / f"{order}gram.arpa"
    binary = storage.KENLM_DIR / f"{order}gram.binary"

    if not _have_kenlm_binaries():
        raise RuntimeError(
            "`lmplz` not found on PATH. Install KenLM (https://github.com/kpu/kenlm) "
            "and ensure its `bin/` is on PATH before running `exp02 build-kenlm`."
        )

    print(f"[kenlm] running lmplz --order={order} ...", flush=True)
    cmd = ["lmplz", "-o", str(order), "--text", str(corpus), "--arpa", str(arpa)]
    if discount_fallback:
        cmd += ["--discount_fallback"]
    subprocess.check_call(cmd)

    if shutil.which("build_binary"):
        print(f"[kenlm] running build_binary -> {binary}", flush=True)
        subprocess.check_call(["build_binary", str(arpa), str(binary)])
        out = binary
    else:
        print("[kenlm] build_binary not on PATH; leaving ARPA only.", flush=True)
        out = arpa

    print(f"[kenlm] done -> {out}", flush=True)
    return out
