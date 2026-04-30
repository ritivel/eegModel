"""LLM-based text augmentation: paraphrase ZuCo training references via OpenAI.

Use case: in CTC training, the same EEG row is shown many times across the 12k
training steps. Always pairing it with the *same* reference text overfits the
head onto the exact word sequence; pairing it with random paraphrases of that
sentence on each pass forces the head to learn semantic content rather than
memorise a specific surface form. This is the EEG analogue of the "neural
text augmentation for ASR" recipe ([Huang 2023, arXiv 2305.16333](https://arxiv.org/abs/2305.16333):
+9-15% relative WER improvement when neural-paraphrased transcripts are
TTS'd back into audio).

Output format: ``$EXP02_DATA_ROOT/text_aug/paraphrases.parquet`` with one row
per ZuCo training sentence::

    sentence_text  : str   — the canonical (lowercased) form
    sent_hash      : str   — Yin-style 16-char SHA1 (matches splits.py)
    paraphrase_1   : str
    paraphrase_2   : str
    paraphrase_3   : str
    ...

Idempotent: re-running picks up sentences without paraphrases and tops up
the existing parquet. Skips sentences whose paraphrase columns are already
populated.

Async batched calls keep wall-time low: 1107 unique ZuCo sentences × 3
paraphrases = 3321 generations; with concurrency=20 and gpt-4o-mini this
runs in ~3 min and costs <$0.10.

Usage::

    exp02 build-paraphrases --n-per-sentence 3 --concurrency 20 --model gpt-4o-mini
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from eeg_common.data import ZUCO_SOURCES, shard_paths
from eeg_common.splits import sent_hash, normalise

from . import storage


# ----------------------------------------------------------------------------
# Sentence collection
# ----------------------------------------------------------------------------


def collect_zuco_sentences() -> list[tuple[str, str]]:
    """Read all unique ZuCo sentences from the parquet shards.

    Returns ``[(sentence_text, sent_hash), ...]`` deduplicated by
    normalised text. We don't filter to the train fold here — paraphrasing
    *every* unique sentence costs the same as paraphrasing the train subset
    and gives us the option to use paraphrases at eval time too.
    """
    seen: dict[str, str] = {}
    for src in ZUCO_SOURCES:
        for path in shard_paths(storage.STORAGE, src):
            t = pq.read_table(path, columns=["sentence_text"])
            for s in t["sentence_text"].to_pylist():
                if not s:
                    continue
                key = normalise(s)
                if key in seen:
                    continue
                seen[key] = sent_hash(s)
    return [(text, h) for text, h in seen.items()]


# ----------------------------------------------------------------------------
# Existing paraphrases (for incremental top-up)
# ----------------------------------------------------------------------------


def load_existing_paraphrases() -> dict[str, dict]:
    """Return ``{sent_hash: {"sentence_text": ..., "paraphrase_1": ..., ...}}``.

    Empty dict if the parquet doesn't exist yet.
    """
    out_path = storage.DATA_ROOT / "text_aug" / "paraphrases.parquet"
    if not out_path.exists():
        return {}
    rows = pq.read_table(out_path).to_pylist()
    return {r["sent_hash"]: r for r in rows}


# ----------------------------------------------------------------------------
# OpenAI async paraphrase
# ----------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a paraphrase generator for English sentences from a reading-comprehension "
    "EEG dataset (ZuCo). The original sentences are mostly short news/biography/movie-"
    "review fragments. Generate paraphrases that:\n"
    "1. Preserve the SEMANTIC content exactly — same facts, same entities, same sentiment.\n"
    "2. Use DIFFERENT word choices, syntax, or sentence ordering.\n"
    "3. Stay roughly the SAME LENGTH as the original (within ±30%).\n"
    "4. Stay GRAMMATICAL English.\n"
    "5. Do NOT add information not in the original.\n"
    "6. Do NOT use markdown, quotes, numbering, or any formatting — return only "
    "the plain paraphrased sentence per line.\n"
    "Return exactly N paraphrases on separate lines, where N is given in the user message."
)


async def _paraphrase_one(client, *, model: str, sentence: str, n: int,
                          max_retries: int = 3) -> list[str]:
    """Async single-sentence paraphrase. Returns a list of ``n`` strings.

    Retries on transient errors. Returns ``[""] * n`` on terminal failure
    (logged to stderr); the trainer treats empty paraphrases as "fall back
    to original".
    """
    user_prompt = f"Generate exactly {n} paraphrases of this sentence, one per line:\n\n{sentence}"
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
                max_tokens=400,
                n=1,
            )
            text = resp.choices[0].message.content or ""
            # Split on lines, strip leading numbering / quotes / bullets.
            lines = [_clean_line(ln) for ln in text.splitlines() if ln.strip()]
            lines = [ln for ln in lines if ln]
            # Pad / trim to exactly ``n``.
            if len(lines) < n:
                lines = lines + [""] * (n - len(lines))
            return lines[:n]
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(2 ** attempt)
    print(f"[paraphrase] FAILED after {max_retries} attempts: "
          f"{type(last_err).__name__}: {last_err} | sentence={sentence[:80]!r}",
          flush=True)
    return [""] * n


def _clean_line(line: str) -> str:
    s = line.strip()
    # Strip "1. ", "1) ", "- ", "* ", "• ", "**" markers, surrounding quotes.
    s = re.sub(r"^\s*(?:[\-\*\u2022]|\d+[\.\)])\s+", "", s)
    s = re.sub(r"^[\"'\u201c\u2018\*]+|[\"'\u201d\u2019\*]+$", "", s)
    s = s.strip()
    return s


async def _paraphrase_all(sentences: list[tuple[str, str]], *,
                          model: str, n_per_sentence: int, concurrency: int
                          ) -> list[dict]:
    """Async-paraphrase all sentences with bounded concurrency."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set in environment. Add it to "
            "experiments/exp02_eeg_ctc/.env (which is gitignored) and "
            "`set -a && source .env && set +a` before running."
        )
    client = AsyncOpenAI(api_key=api_key)

    sem = asyncio.Semaphore(concurrency)
    pbar_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(sentences), "errors": 0}

    async def worker(idx: int, text: str, h: str) -> dict:
        async with sem:
            paraphrases = await _paraphrase_one(client, model=model,
                                                sentence=text, n=n_per_sentence)
        async with pbar_lock:
            progress["done"] += 1
            if any(p == "" for p in paraphrases):
                progress["errors"] += 1
            if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
                print(f"[paraphrase] {progress['done']}/{progress['total']} "
                      f"({progress['errors']} errors)", flush=True)
        out = {"sentence_text": text, "sent_hash": h}
        for i, p in enumerate(paraphrases, start=1):
            out[f"paraphrase_{i}"] = p
        return out

    tasks = [worker(i, t, h) for i, (t, h) in enumerate(sentences)]
    return await asyncio.gather(*tasks)


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------


def build_paraphrases(*, n_per_sentence: int = 3, concurrency: int = 20,
                      model: str = "gpt-4o-mini") -> Path:
    """Build the paraphrases parquet. Idempotent: skips sentences that already
    have a full set of paraphrases.

    Returns the path to the parquet.
    """
    out_dir = storage.DATA_ROOT / "text_aug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paraphrases.parquet"

    sentences = collect_zuco_sentences()
    print(f"[paraphrase] {len(sentences)} unique ZuCo sentences", flush=True)

    existing = load_existing_paraphrases()
    print(f"[paraphrase] {len(existing)} already paraphrased; "
          f"skipping those.", flush=True)

    todo = []
    for text, h in sentences:
        prev = existing.get(h)
        if prev is None:
            todo.append((text, h))
            continue
        # Top-up if any paraphrase column is missing or empty.
        complete = all(prev.get(f"paraphrase_{i}", "") for i in range(1, n_per_sentence + 1))
        if not complete:
            todo.append((text, h))

    if not todo:
        print(f"[paraphrase] nothing to do; {out_path} is already complete.",
              flush=True)
        return out_path

    print(f"[paraphrase] generating {len(todo) * n_per_sentence} paraphrases "
          f"(model={model}, concurrency={concurrency})...", flush=True)
    t0 = time.time()
    new_rows = asyncio.run(_paraphrase_all(todo, model=model,
                                            n_per_sentence=n_per_sentence,
                                            concurrency=concurrency))
    print(f"[paraphrase] done in {time.time() - t0:.1f}s.", flush=True)

    # Merge with existing rows.
    merged = {**existing, **{r["sent_hash"]: r for r in new_rows}}
    rows = list(merged.values())

    # Normalise to a single schema (in case n_per_sentence changed across runs).
    n_cols = max(
        max((int(k.split("_")[1]) for k in r if k.startswith("paraphrase_")),
            default=0)
        for r in rows
    )
    n_cols = max(n_cols, n_per_sentence)
    columns: list[str] = ["sentence_text", "sent_hash"] + [
        f"paraphrase_{i}" for i in range(1, n_cols + 1)
    ]
    aligned = []
    for r in rows:
        out = {c: r.get(c, "") for c in columns}
        aligned.append(out)

    table = pa.Table.from_pylist(aligned)
    pq.write_table(table, out_path)
    print(f"[paraphrase] wrote {len(aligned)} rows × {n_cols} paraphrases "
          f"to {out_path}", flush=True)
    return out_path


# ----------------------------------------------------------------------------
# Lookup helper used at training time
# ----------------------------------------------------------------------------


class ParaphraseLookup:
    """Loads the paraphrase parquet once, exposes a quick ``sample(text)``.

    The trainer's collator calls ``sample(ref)`` per row with probability
    ``text_aug_prob`` to substitute a random paraphrase as the CTC target.
    """

    def __init__(self, parquet_path: str):
        rows = pq.read_table(parquet_path).to_pylist()
        self._by_hash: dict[str, list[str]] = {}
        for r in rows:
            paras = []
            for k, v in r.items():
                if not k.startswith("paraphrase_"):
                    continue
                if v and isinstance(v, str) and v.strip():
                    paras.append(v.strip())
            if paras:
                self._by_hash[r["sent_hash"]] = paras
        print(f"[ParaphraseLookup] loaded {len(self._by_hash)} entries with "
              f"≥1 paraphrase from {parquet_path}", flush=True)

    def sample(self, original: str, *, rng) -> str:
        """Return a random paraphrase of ``original``, or the original if none."""
        h = sent_hash(original)
        paras = self._by_hash.get(h)
        if not paras:
            return original
        return paras[rng.integers(0, len(paras))]
