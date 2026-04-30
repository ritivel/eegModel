"""Evaluation: CER / WER / BLEU 1-4 / ROUGE-1-F / BERTScore-F1 per decode mode.

Per-cell artifacts:

  metrics.json       - summary mean / 95% CI per metric, per decode mode
  predictions.parquet - one row per test example with metadata, ref, and
                       hyp_<greedy|beam|beam_kenlm> + per-row metric values

W&B logging (when WANDB_API_KEY is set):
  - cell-level summary metrics
  - sample table of (ref, hyp) pairs (capped at 500 rows for quota safety)

Matched-pair §4.3 gap (Jo et al. 2024) is computed by
:func:`matched_pair_gap`, which the CLI ``exp02 gap CFG_KEY_eeg`` invokes
once both the ``eeg`` cell and its ``noise_train`` twin have been evaluated.

Per Jo et al. (2024): teacher forcing is disabled at evaluation. Greedy /
beam / beam+KenLM decoders are all run from the same ``log_probs`` so the
forward cost is paid once per cell.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_common import preprocessing, splits
from eeg_common.data import EEGSentenceDataset, ZUCO_SOURCES

from . import chars as chars_mod
from . import decode, storage
from .config import CTCConfig
from .model import EEG2CTC
from .train import _collate

BOOTSTRAP_N = 1000
PERMUTATION_N = 10_000
WANDB_TABLE_CAP = 500
DECODE_MODES = ("greedy", "beam", "beam_kenlm")


# ============================================================================
# Per-sentence metrics
# ============================================================================


def _sentence_bleu(hyp: str, ref: str, n: int) -> float:
    import sacrebleu
    return float(sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp")
                 .precisions[n - 1]) / 100.0


def _sentence_rouge1(hyp: str, ref: str, scorer) -> float:
    return float(scorer.score(ref, hyp)["rouge1"].fmeasure)


def _edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ai == b[j - 1] else 1),
            )
        prev, cur = cur, prev
    return prev[lb]


def _sentence_cer(hyp: str, ref: str) -> float:
    h = hyp.lower()
    r = ref.lower()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(list(h), list(r)) / max(1, len(r))


def _sentence_wer(hyp: str, ref: str) -> float:
    h = hyp.lower().split()
    r = ref.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(h, r) / max(1, len(r))


def per_sentence_scores(hyps: list[str], refs: list[str]) -> dict[str, np.ndarray]:
    """Per-sentence metrics. CER / WER / BLEU 1-4 / ROUGE-1-F / BERTScore-F1."""
    from rouge_score.rouge_scorer import RougeScorer
    rouge = RougeScorer(["rouge1"], use_stemmer=True)

    out: dict[str, list[float]] = {f"bleu{n}": [] for n in (1, 2, 3, 4)}
    out["rouge1_f"] = []
    out["cer"] = []
    out["wer"] = []
    for h, r in zip(hyps, refs):
        for n in (1, 2, 3, 4):
            out[f"bleu{n}"].append(_sentence_bleu(h, r, n))
        out["rouge1_f"].append(_sentence_rouge1(h, r, rouge))
        out["cer"].append(_sentence_cer(h, r))
        out["wer"].append(_sentence_wer(h, r))
    try:
        from bert_score import score as bertscore
        _, _, F1 = bertscore(hyps, refs, model_type="roberta-large", lang="en",
                             rescale_with_baseline=True, verbose=False)
        out["bertscore_f1"] = F1.numpy().tolist()
    except Exception as e:
        print(f"[eval] BERTScore failed ({type(e).__name__}: {e}); "
              f"setting bertscore_f1 to NaN.", flush=True)
        out["bertscore_f1"] = [float("nan")] * len(hyps)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# ============================================================================
# Bootstrap CIs + permutation test
# ============================================================================


def bootstrap_ci(values: np.ndarray, *, n: int = BOOTSTRAP_N,
                 alpha: float = 0.05, seed: int = 0):
    if len(values) == 0 or np.all(np.isnan(values)):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = np.nanmean(values[idx], axis=1)
    lo, hi = np.nanquantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.nanmean(values)), float(lo), float(hi)


def permutation_paired(eeg: np.ndarray, noise: np.ndarray, *,
                       n: int = PERMUTATION_N, seed: int = 0,
                       higher_is_better: bool = True) -> float:
    """Sign-flip permutation test for paired (eeg, noise) per-sentence scores.

    For ``higher_is_better=True`` (BLEU, ROUGE) we test ``eeg > noise``.
    For ``higher_is_better=False`` (CER, WER) we test ``eeg < noise`` by
    negating both inputs.
    """
    if not higher_is_better:
        eeg, noise = -eeg, -noise
    rng = np.random.default_rng(seed)
    delta = eeg - noise
    obs = delta.mean()
    signs = rng.choice([-1, 1], size=(n, len(delta)))
    null = (signs * delta).mean(axis=1)
    p = float(((null >= obs).sum() + 1) / (n + 1))
    return p


# ============================================================================
# Single-cell evaluation
# ============================================================================


def evaluate_cell(cfg: CTCConfig, *, ckpt_path: Path | None = None,
                  decode_modes: tuple[str, ...] = DECODE_MODES) -> dict:
    storage.ensure_dirs()

    if cfg.vocab == "bpe1k":
        vocab = chars_mod.load_vocab("bpe1k", bpe_model_path=str(storage.BPE_MODEL))
    else:
        vocab = chars_mod.load_vocab(cfg.vocab)

    fold = splits.load_fold(storage.STORAGE, cfg.fold)
    pp_spec = (preprocessing.for_encoder(cfg.preprocess, cfg.encoder)
               if cfg.preprocess != "v1" else None)
    test_ds = EEGSentenceDataset(
        storage.STORAGE,
        sources=ZUCO_SOURCES,
        subject_filter=fold.test_subjects,
        sentence_filter=fold.test_sent_hashes,
        noise="gauss" if cfg.input in ("noise_train", "noise_test") else None,
        preprocess=pp_spec,
    )

    model = EEG2CTC(cfg, vocab).to("cuda")
    if ckpt_path is None:
        ckpt_path = storage.RUNS / cfg.cell_id / "model.pt"
    state = torch.load(ckpt_path, map_location="cuda")
    sd = state["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"[eval] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}, "
            f"saved={len(sd)}",
            flush=True,
        )
        if len(missing) > 50 or len(unexpected) > 50:
            print(f"[eval]   first 5 missing:    {missing[:5]}", flush=True)
            print(f"[eval]   first 5 unexpected: {unexpected[:5]}", flush=True)
    model.eval()

    target_sr = model.encoder.spec.native_sr
    dl = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: _collate(b, target_sr=target_sr),
        pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )

    preds: list[dict] = []
    with torch.no_grad():
        for batch in dl:
            eeg = batch["eeg"].to("cuda", non_blocking=True)
            feats = model.encoder.encode(eeg, batch["sr"], batch["channels"])
            out = model.head_forward(feats)
            log_probs = F.log_softmax(out.logits.float(), dim=-1)
            hyps = decode.decode_all(
                log_probs, vocab,
                beam_width=cfg.decode_beam_width,
                kenlm_alpha=cfg.decode_kenlm_alpha,
                kenlm_beta=cfg.decode_kenlm_beta,
                enable_beam=("beam" in decode_modes),
                enable_beam_kenlm=("beam_kenlm" in decode_modes),
            )
            B = eeg.size(0)
            for i in range(B):
                row = {
                    "subject_id": batch["subject_ids"][i],
                    "dataset": batch["datasets"][i],
                    "ref": batch["text"][i],
                    "eeg_channels": int(batch["eeg"].shape[1]),
                    "eeg_time": int(batch["eeg"].shape[2]),
                    "sr": float(batch["sr"]),
                }
                for mode, hlist in hyps.items():
                    row[f"hyp_{mode}"] = hlist[i]
                preds.append(row)

    refs = [p["ref"] for p in preds]

    summary: dict = {
        "cell_id": cfg.cell_id,
        "cfg": asdict(cfg),
        "n_examples": len(preds),
        "ckpt": str(ckpt_path),
        "scores": {},
    }
    for mode in decode_modes:
        hyp_key = f"hyp_{mode}"
        if not preds or hyp_key not in preds[0]:
            continue
        hyps_list = [p[hyp_key] for p in preds]
        metrics = per_sentence_scores(hyps_list, refs)
        for k, arr in metrics.items():
            for i, v in enumerate(arr.tolist()):
                preds[i][f"{mode}__{k}"] = v
        summary["scores"][mode] = {}
        for k, v in metrics.items():
            m, lo, hi = bootstrap_ci(v)
            summary["scores"][mode][k] = {
                "mean": m, "ci_lo": lo, "ci_hi": hi,
                "values": v.tolist(),
            }

    cell_dir = storage.cell_eval_dir(cfg.cell_id)
    if preds:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(preds)
        pq.write_table(table, cell_dir / "predictions.parquet")
    (cell_dir / "metrics.json").write_text(json.dumps(summary, separators=(",", ":")))

    _wandb_log_eval(cfg, summary, preds)
    return summary


def _wandb_log_eval(cfg: CTCConfig, summary: dict, preds: list[dict]) -> None:
    if "WANDB_API_KEY" not in os.environ:
        return
    import wandb
    own_run = False
    run = wandb.run
    if run is None:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "exp02-eeg-ctc"),
            name=cfg.cell_id,
            group=f"{cfg.encoder}_{cfg.vocab}_{cfg.variant}",
            tags=[cfg.encoder, cfg.vocab, cfg.variant, cfg.input,
                  f"fold{cfg.fold}", "eval"],
            config=asdict(cfg),
            dir=str(storage.WANDB_DIR),
            resume="allow",
        )
        own_run = True

    run.summary["n_examples"] = len(preds)
    for mode, m_by_metric in summary["scores"].items():
        for k, v in m_by_metric.items():
            run.summary[f"eval/{mode}/{k}"] = v["mean"]
            run.summary[f"eval/{mode}/{k}_ci_lo"] = v["ci_lo"]
            run.summary[f"eval/{mode}/{k}_ci_hi"] = v["ci_hi"]

    if preds:
        cap = preds[:WANDB_TABLE_CAP]
        cols = ["subject_id", "dataset", "ref",
                "hyp_greedy", "hyp_beam", "hyp_beam_kenlm",
                "greedy__cer", "beam__cer", "beam_kenlm__cer",
                "greedy__bleu1", "beam__bleu1", "beam_kenlm__bleu1"]
        rows = [[p.get(c, None) for c in cols] for p in cap]
        run.log({"predictions": wandb.Table(columns=cols, data=rows)})

    if own_run:
        run.finish()


# ============================================================================
# Matched-pair §4.3 gap (Jo et al. 2024)
# ============================================================================


def matched_pair_gap(eeg_summary: dict, noise_summary: dict,
                     *, modes: tuple[str, ...] = DECODE_MODES) -> dict:
    """For each decode mode and metric, compute the (eeg - noise) bootstrap
    gap and the sign-flip permutation p-value.

    For CER and WER ``higher_is_better=False`` (the gap is reported as
    ``noise - eeg`` so a positive value = EEG wins).
    """
    out: dict = {}
    for mode in modes:
        if mode not in eeg_summary["scores"] or mode not in noise_summary["scores"]:
            continue
        out[mode] = {}
        for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge1_f", "bertscore_f1",
                  "cer", "wer"):
            if k not in eeg_summary["scores"][mode]:
                continue
            e = np.asarray(eeg_summary["scores"][mode][k]["values"])
            n = np.asarray(noise_summary["scores"][mode][k]["values"])
            higher_is_better = k not in ("cer", "wer")
            if higher_is_better:
                delta = e - n
            else:
                delta = n - e
            gap_m, gap_lo, gap_hi = bootstrap_ci(delta)
            p = permutation_paired(e, n, higher_is_better=higher_is_better)
            out[mode][k] = {
                "gap_mean": gap_m,
                "gap_ci_lo": gap_lo,
                "gap_ci_hi": gap_hi,
                "p_value": p,
                "passes_strict": (gap_lo > 0) and (p < 0.01),
                "passes_weak": gap_m > 0,
            }
    return out
