"""Single CLI entry point.

  exp01 download-models      one-shot HF cache (REVE, REVE-positions, TFM, Gemma)
  exp01 download-data        one-shot HF cache for the unified dataset (~72 GB)
  exp01 make-splits          compute & persist the 5 LNSO folds
  exp01 smoke                end-to-end smoke (1 cell per code path, ~3 min)
  exp01 train CFG_KEY        train a single cell (CFG_KEY = encoder.bridge.input.fold)
  exp01 eval  CFG_KEY        evaluate a single cell
  exp01 pilot                Phase 1: fold 0, EEG, all enabled (encoder × bridge)
  exp01 full                 Phase 3: surviving cells × 3 inputs × 5 folds
  exp01 inspect-channels     Diagnostic: print channel names per source

Env:
  EXP01_DATA_ROOT            where everything lives (default ./data)
  HF_TOKEN                   gated HF repos
  WANDB_API_KEY              W&B logging (optional; runs offline if unset)
  WANDB_PROJECT              default exp01-eeg-to-text
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _cmd_download_models(args):
    import os as _os
    from huggingface_hub import snapshot_download

    from . import encoders, storage

    storage.ensure_dirs()
    repos = [
        ("brain-bzh/reve-base", "model"),
        ("brain-bzh/reve-positions", "model"),
        ("Jathurshan/TFM-Tokenizer", "model"),
        ("google/gemma-4-E2B-it", "model"),
    ]
    fail = []
    for repo_id, repo_type in repos:
        print(f"[download] {repo_id} ({repo_type})", flush=True)
        try:
            snapshot_download(repo_id=repo_id, repo_type=repo_type,
                              cache_dir=str(storage.HF_CACHE),
                              token=_os.environ.get("HF_TOKEN"))
        except Exception as e:
            print(f"[download][FAIL] {repo_id}: {type(e).__name__}: {e}", flush=True)
            fail.append(repo_id)
    print(f"[download] TFM source -> {encoders.ensure_tfm_source()}", flush=True)
    if fail:
        print("FAILED:", fail, file=sys.stderr); sys.exit(1)


def _cmd_download_data(args):
    from . import data
    data.download_dataset()


def _cmd_make_splits(args):
    from . import data
    data.write_splits()


def _cmd_inspect_channels(args):
    """Print channel name samples per dataset source — used to verify
    encoder-side normalisation is correct."""
    import glob
    import pyarrow.parquet as pq

    from . import data
    base = str(data._hf_dataset_snapshots_dir())
    print(f"=== shards under {base} ===", flush=True)
    for src in data.ALL_SOURCES:
        files = sorted(glob.glob(f"{base}/*/data/{src}*.parquet"))
        if not files:
            print(f"  {src}: NO FILES", flush=True); continue
        t = pq.ParquetFile(files[0]).read_row_group(0, columns=["channel_names"])
        chans = t["channel_names"].to_pylist()[0]
        print(f"  {src}: n={len(chans)} first={chans[:8]} last={chans[-3:]}", flush=True)


# ============================================================================
# Train / eval / smoke / pilot / full
# ============================================================================


def _parse_cfg_key(key: str):
    """``encoder.bridge.input.fold`` -> CellConfig kwargs.
    Example: ``reve.linear.eeg.0`` -> CellConfig(encoder='reve',...)."""
    parts = key.split(".")
    if len(parts) != 4:
        raise ValueError(f"expected encoder.bridge.input.fold, got {key!r}")
    enc, br, inp, fold = parts
    return {"encoder": enc, "bridge": br, "input": inp, "fold": int(fold)}


def _cmd_train(args):
    from .config import CellConfig
    from . import train as trainmod
    cfg = CellConfig(**_parse_cfg_key(args.cfg_key))
    trainmod.train(cfg)


def _cmd_eval(args):
    from .config import CellConfig
    from . import eval as evalmod
    cfg = CellConfig(**_parse_cfg_key(args.cfg_key))
    summary = evalmod.evaluate_cell(cfg)
    for k, v in summary["scores"].items():
        print(f"  {k:>15s}: {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")


def _cmd_smoke(args):
    """End-to-end micro-run. One cell per code path:
       - REVE × linear  (soft-prompt path)
       - TFM  × vocab   (vocab-extension path)
    Asserts every artifact exists + is non-empty.
    """
    import json

    from . import eval as evalmod
    from . import storage
    from . import train as trainmod
    from .config import CellConfig

    cells = [
        CellConfig(encoder="reve", bridge="linear", input="eeg", fold=0,
                   stage1_steps=20, stage2_steps=20, stage3_steps=10,
                   batch_size=2, grad_accum=1),
        CellConfig(encoder="tfm", bridge="vocab", input="eeg", fold=0,
                   stage1_steps=20, stage2_steps=20, stage3_steps=10,
                   batch_size=2, grad_accum=1),
    ]
    fails = []
    for cfg in cells:
        print(f"\n========== SMOKE: {cfg.cell_id} ==========\n", flush=True)
        try:
            ckpt = trainmod.train(cfg)
            summary = evalmod.evaluate_cell(cfg, ckpt_path=ckpt)
        except Exception as e:
            fails.append((cfg.cell_id, repr(e)))
            print(f"FAILED: {e}", flush=True)
            continue

        run_dir = storage.cell_run_dir(cfg.cell_id)
        eval_dir = storage.cell_eval_dir(cfg.cell_id)
        expected = [
            run_dir / "log.jsonl",
            run_dir / "sample_gens.jsonl",
            run_dir / "stats.jsonl",
            run_dir / "model_stage1.pt",
            run_dir / "model_stage2.pt",
            run_dir / "model.pt",
            eval_dir / "metrics.json",
            eval_dir / "predictions.parquet",
        ]
        missing = [p for p in expected if not p.exists() or p.stat().st_size == 0]
        if missing:
            fails.append((cfg.cell_id, f"missing/empty: {[str(p) for p in missing]}"))
            continue

        # Surface a sample (ref, hyp) pair so the user can eyeball it.
        import pyarrow.parquet as pq
        rows = pq.read_table(eval_dir / "predictions.parquet").to_pylist()
        if rows:
            print(f"  {len(rows)} test examples", flush=True)
            print(f"  first ref: {rows[0]['ref'][:80]!r}", flush=True)
            print(f"  first hyp: {rows[0]['hyp'][:80]!r}", flush=True)
        m = json.loads((eval_dir / "metrics.json").read_text())
        for k, v in m["scores"].items():
            print(f"  {k:>15s}: {v['mean']:.3f}", flush=True)

    if fails:
        print(f"\n=== Smoke summary: {len(cells)-len(fails)} passed, {len(fails)} failed ===")
        for cid, err in fails:
            print(f"  FAIL {cid}: {err}")
        sys.exit(1)
    print(f"\n=== Smoke summary: all {len(cells)} cells passed ===")


def _cmd_pilot(args):
    """§5.1 Phase 1: fold 0, EEG only, all (encoder × bridge) cells in the
    selected encoder set."""
    from . import eval as evalmod
    from . import train as trainmod
    from .config import pilot_cells

    encs = tuple(e.strip() for e in args.encoders.split(",") if e.strip())
    cells = pilot_cells(encoders=encs)
    print(f"Pilot: {len(cells)} cells (encoders={encs}), running sequentially.")

    results = []
    for cfg in cells:
        print(f"\n=========== {cfg.cell_id} ===========", flush=True)
        try:
            ckpt = trainmod.train(cfg)
            summary = evalmod.evaluate_cell(cfg, ckpt_path=ckpt)
            results.append((cfg.cell_id, summary["scores"]["bleu1"]["mean"],
                            summary["scores"]["rouge1_f"]["mean"]))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
            results.append((cfg.cell_id, None, None))

    print("\n=== Pilot ranking (BLEU-1) ===")
    results.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
    for cid, b, r in results:
        if b is None:
            print(f"  {cid:<55s}  FAILED")
        else:
            print(f"  {cid:<55s}  BLEU-1={b:.3f}  ROUGE-1-F={r:.3f}")


def _cmd_full(args):
    """§5.1 Phase 3: surviving (encoder, bridge) × 3 inputs × 5 folds."""
    import json

    from . import eval as evalmod
    from . import storage
    from . import train as trainmod
    from .config import CellConfig

    surviving = [tuple(p.split(":")) for p in args.surviving.split(",")]
    cells = []
    for enc, br in surviving:
        for inp in ("eeg", "noise_train", "noise_test"):
            for fold in range(5):
                cells.append(CellConfig(encoder=enc, bridge=br, input=inp, fold=fold))
    print(f"Full: {len(cells)} cells, sequential.")

    by_id = {}
    for cfg in cells:
        print(f"\n=========== {cfg.cell_id} ===========", flush=True)
        try:
            ckpt = trainmod.train(cfg)
            summary = evalmod.evaluate_cell(cfg, ckpt_path=ckpt)
            by_id[cfg.cell_id] = summary
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", flush=True)

    # EEG vs noise gap per (encoder, bridge, fold).
    report = {"cells": by_id, "gaps": {}}
    for cell_id, summary in by_id.items():
        if "_eeg_" not in cell_id:
            continue
        nt = cell_id.replace("_eeg_", "_noise_train_")
        nv = cell_id.replace("_eeg_", "_noise_test_")
        if nt in by_id and nv in by_id:
            report["gaps"][cell_id] = {
                "vs_noise_train": evalmod.eeg_noise_gap(summary, by_id[nt]),
                "vs_noise_test": evalmod.eeg_noise_gap(summary, by_id[nv]),
            }
    out = storage.EVAL / "full_report.json"
    out.write_text(json.dumps(report, separators=(",", ":")))
    print(f"\nWrote {out}")


# ============================================================================
# argparse wiring
# ============================================================================


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="exp01", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download-models").set_defaults(func=_cmd_download_models)
    sub.add_parser("download-data").set_defaults(func=_cmd_download_data)
    sub.add_parser("make-splits").set_defaults(func=_cmd_make_splits)
    sub.add_parser("inspect-channels").set_defaults(func=_cmd_inspect_channels)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    sp = sub.add_parser("train")
    sp.add_argument("cfg_key", help="encoder.bridge.input.fold (e.g. reve.linear.eeg.0)")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("eval")
    sp.add_argument("cfg_key")
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("pilot")
    sp.add_argument("--encoders", default="reve,tfm",
                    help="comma-separated subset of {reve,diver1,tfm}")
    sp.set_defaults(func=_cmd_pilot)

    sp = sub.add_parser("full")
    sp.add_argument("--surviving", default="reve:linear,tfm:vocab",
                    help="comma-separated <encoder>:<bridge> pairs")
    sp.set_defaults(func=_cmd_full)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
