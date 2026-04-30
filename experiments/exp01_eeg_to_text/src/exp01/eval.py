"""Evaluation: BLEU-1..4, ROUGE-1-F, BERTScore-F1.

Per-cell artifacts:
  metrics.json       — summary mean / 95% CI per metric
  predictions.parquet — one row per test example with full metadata
                       (subject, dataset, sentence_id, sr, eeg_shape,
                        ref, hyp, all 6 metric values)

W&B logging (when WANDB_API_KEY is set):
  - cell-level summary metrics
  - sample table of (ref, hyp) pairs (capped at 500 rows for quota safety)

Per Jo et al. (2024): teacher forcing is disabled at evaluation. Greedy decode.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data, storage
from .config import CellConfig
from .model import EEG2Text
from .train import _collate

BOOTSTRAP_N = 1000
PERMUTATION_N = 10_000
WANDB_TABLE_CAP = 500          # cap per-cell W&B Table rows


# ============================================================================
# Per-sentence metrics
# ============================================================================


def _sentence_bleu(hyp: str, ref: str, n: int) -> float:
    import sacrebleu
    return float(sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp").precisions[n - 1]) / 100.0


def _sentence_rouge1(hyp: str, ref: str, scorer) -> float:
    return float(scorer.score(ref, hyp)["rouge1"].fmeasure)


def per_sentence_scores(hyps: list[str], refs: list[str]) -> dict[str, np.ndarray]:
    """All six metrics at the sentence level. BERTScore failures are
    tolerated — they just mark that metric as missing in the summary."""
    from rouge_score.rouge_scorer import RougeScorer

    rouge = RougeScorer(["rouge1"], use_stemmer=True)

    out: dict[str, list[float]] = {f"bleu{n}": [] for n in (1, 2, 3, 4)}
    out["rouge1_f"] = []

    for h, r in zip(hyps, refs):
        for n in (1, 2, 3, 4):
            out[f"bleu{n}"].append(_sentence_bleu(h, r, n))
        out["rouge1_f"].append(_sentence_rouge1(h, r, rouge))

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


def bootstrap_ci(values: np.ndarray, *, n: int = BOOTSTRAP_N, alpha: float = 0.05, seed: int = 0):
    if len(values) == 0 or np.all(np.isnan(values)):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = np.nanmean(values[idx], axis=1)
    lo, hi = np.nanquantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.nanmean(values)), float(lo), float(hi)


def permutation_paired(eeg: np.ndarray, noise: np.ndarray, *, n: int = PERMUTATION_N, seed: int = 0) -> float:
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


def evaluate_cell(cfg: CellConfig, *, ckpt_path: Path | None = None) -> dict:
    storage.ensure_dirs()
    fold = data.load_fold(cfg.fold)

    test_ds = data.EEGSentenceDataset(
        sources=data.ZUCO_SOURCES,
        subject_filter=fold.test_subjects,
        sentence_filter=fold.test_sent_hashes,
        noise="gauss" if cfg.input in ("noise_train", "noise_test") else None,
    )

    model = EEG2Text(cfg).to("cuda")
    if ckpt_path is None:
        if cfg.input == "noise_test":
            twin = storage.RUNS / cfg.cell_id.replace("_noise_test_", "_eeg_") / "model.pt"
            ckpt_path = twin
        else:
            ckpt_path = storage.RUNS / cfg.cell_id / "model.pt"
    state = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    tokenizer = model.dec.tokenizer
    target_sr = model.encoder.spec.native_sr
    dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2,
                    collate_fn=lambda b: _collate(b, tokenizer, target_sr=target_sr))

    # Collect predictions + metadata for every example.
    preds: list[dict] = []
    for batch in dl:
        b_gpu = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            gens = model.generate(eeg=b_gpu["eeg"], sr=b_gpu["sr"], channels=b_gpu["channels"])
        except Exception as e:
            gens = [f"<gen-error: {type(e).__name__}: {e}>"] * len(batch["text"])
        for i, (ref, hyp, sid, dset) in enumerate(zip(
                batch["text"], gens, batch["subject_ids"], batch["datasets"])):
            preds.append({
                "subject_id": sid,
                "dataset": dset,
                "ref": ref,
                "hyp": hyp,
                "eeg_channels": int(batch["eeg"].shape[1]),
                "eeg_time": int(batch["eeg"].shape[2]),
                "sr": float(batch["sr"]),
            })

    refs = [p["ref"] for p in preds]
    hyps = [p["hyp"] for p in preds]
    metrics = per_sentence_scores(hyps, refs) if preds else {}

    # Attach per-example metric values back to the prediction rows.
    for k, arr in metrics.items():
        for i, v in enumerate(arr.tolist()):
            preds[i][k] = v

    # ---- Persist ------------------------------------------------------------
    cell_dir = storage.cell_eval_dir(cfg.cell_id)

    # Predictions parquet (one row per test example, all metadata + metrics).
    if preds:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(preds)
        pq.write_table(table, cell_dir / "predictions.parquet")

    # Summary JSON.
    summary = {
        "cell_id": cfg.cell_id,
        "cfg": asdict(cfg),
        "n_examples": len(preds),
        "ckpt": str(ckpt_path),
        "scores": {},
    }
    for k, v in metrics.items():
        m, lo, hi = bootstrap_ci(v)
        summary["scores"][k] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "values": v.tolist()}

    (cell_dir / "metrics.json").write_text(json.dumps(summary, separators=(",", ":")))

    # ---- W&B ----------------------------------------------------------------
    _wandb_log_eval(cfg, summary, preds)

    return summary


def _wandb_log_eval(cfg: CellConfig, summary: dict, preds: list[dict]):
    if "WANDB_API_KEY" not in os.environ:
        return
    import wandb

    # Reuse the active run if training created one for this cell; otherwise
    # spin up a fresh eval-only run (e.g. when ``exp01 eval`` is called
    # standalone on a checkpoint).
    own_run = False
    run = wandb.run
    if run is None:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "exp01-eeg-to-text"),
            name=cfg.cell_id,
            group=f"{cfg.encoder}_{cfg.bridge}",
            tags=[cfg.encoder, cfg.bridge, cfg.input, f"fold{cfg.fold}", "eval"],
            config=asdict(cfg),
            dir=str(storage.WANDB_DIR),
            resume="allow",
        )
        own_run = True

    run.summary["n_examples"] = len(preds)
    for k, v in summary["scores"].items():
        run.summary[f"eval/{k}"] = v["mean"]
        run.summary[f"eval/{k}_ci_lo"] = v["ci_lo"]
        run.summary[f"eval/{k}_ci_hi"] = v["ci_hi"]

    if preds:
        cap = preds[:WANDB_TABLE_CAP]
        cols = ["subject_id", "dataset", "ref", "hyp",
                "bleu1", "bleu2", "bleu3", "bleu4", "rouge1_f", "bertscore_f1"]
        rows = [[p.get(c, None) for c in cols] for p in cap]
        run.log({"predictions": wandb.Table(columns=cols, data=rows)})

    if own_run:
        run.finish()


# ============================================================================
# EEG-vs-noise gap (per encoder × bridge × fold)
# ============================================================================


def eeg_noise_gap(eeg_summary: dict, noise_summary: dict) -> dict:
    out = {}
    for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge1_f", "bertscore_f1"):
        e = np.asarray(eeg_summary["scores"][k]["values"])
        n = np.asarray(noise_summary["scores"][k]["values"])
        gap_m, gap_lo, gap_hi = bootstrap_ci(e - n)
        p = permutation_paired(e, n)
        out[k] = {"gap_mean": gap_m, "gap_ci_lo": gap_lo, "gap_ci_hi": gap_hi, "p_value": p,
                  "passes": gap_lo > 0}
    return out
