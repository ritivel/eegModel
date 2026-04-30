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


def _finish_wandb():
    """Close any active W&B run (no-op if none)."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.finish()
    except Exception:
        pass


def _parse_cfg_key(key: str):
    """``encoder.bridge.input.fold`` -> CellConfig kwargs.
    Example: ``reve.linear.eeg.0`` -> CellConfig(encoder='reve',...)."""
    parts = key.split(".")
    if len(parts) != 4:
        raise ValueError(f"expected encoder.bridge.input.fold, got {key!r}")
    enc, br, inp, fold = parts
    return {"encoder": enc, "bridge": br, "input": inp, "fold": int(fold)}


def _step_overrides(args) -> dict:
    """CLI ``--stage{1,2,3}-steps`` -> kwargs for CellConfig."""
    out = {}
    for stage in (1, 2, 3):
        v = getattr(args, f"stage{stage}_steps", None)
        if v is not None:
            out[f"stage{stage}_steps"] = int(v)
    if getattr(args, "batch_size", None) is not None:
        out["batch_size"] = int(args.batch_size)
    if getattr(args, "no_lora", False):
        out["use_lora_in_stage3"] = False
    return out


def _cmd_train(args):
    from .config import CellConfig
    from . import train as trainmod
    cfg = CellConfig(**_parse_cfg_key(args.cfg_key), **_step_overrides(args))
    trainmod.train(cfg)


def _cmd_eval(args):
    from .config import CellConfig
    from . import eval as evalmod
    cfg = CellConfig(**_parse_cfg_key(args.cfg_key), **_step_overrides(args))
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
                   batch_size=1, grad_accum=1),
        CellConfig(encoder="tfm", bridge="vocab", input="eeg", fold=0,
                   stage1_steps=20, stage2_steps=20, stage3_steps=10,
                   batch_size=1, grad_accum=1),
    ]
    import traceback as _tb

    fails = []
    for cfg in cells:
        print(f"\n========== SMOKE: {cfg.cell_id} ==========\n", flush=True)
        try:
            ckpt = trainmod.train(cfg)
            summary = evalmod.evaluate_cell(cfg, ckpt_path=ckpt)
        except Exception as e:
            fails.append((cfg.cell_id, repr(e)))
            print(f"FAILED: {e}", flush=True)
            _tb.print_exc()
            _finish_wandb()
            continue
        _finish_wandb()

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
    selected encoder set. ``--parallel`` runs one cell per visible GPU.
    ``--cells`` overrides which cells to run (comma-separated cfg_keys)."""
    from .config import CellConfig, pilot_cells

    encs = tuple(e.strip() for e in args.encoders.split(",") if e.strip())
    if args.cells:
        keys = [c.strip() for c in args.cells.split(",") if c.strip()]
        cells = [CellConfig(**_parse_cfg_key(k), **_step_overrides(args)) for k in keys]
    else:
        cells = [
            CellConfig(encoder=c.encoder, bridge=c.bridge, input=c.input,
                       fold=c.fold, decoder=c.decoder, **_step_overrides(args))
            for c in pilot_cells(encoders=encs)
        ]

    if args.parallel:
        _run_parallel([c.cfg_key for c in cells], header="Pilot",
                      step_overrides=_step_overrides(args))
        return

    from . import eval as evalmod
    from . import train as trainmod
    import traceback as _tb

    print(f"Pilot: {len(cells)} cells, running sequentially.")
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
            _tb.print_exc()
            results.append((cfg.cell_id, None, None))
        _finish_wandb()

    print("\n=== Pilot ranking (BLEU-1) ===")
    results.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
    for cid, b, r in results:
        if b is None:
            print(f"  {cid:<55s}  FAILED")
        else:
            print(f"  {cid:<55s}  BLEU-1={b:.3f}  ROUGE-1-F={r:.3f}")


def _run_parallel(cfg_keys: list[str], *, header: str, step_overrides: dict | None = None):
    """Spawn one ``train+eval`` subprocess per GPU, sharded round-robin across
    visible GPUs. Each process writes its own log to
    ``$EXP01_DATA_ROOT/runs/<cell_id>/run.log``.

    Uses ``sys.executable -m exp01.cli`` so the subprocess works regardless of
    whether the ``exp01`` console script is on PATH.
    """
    import os
    import sys
    import subprocess
    import time

    n_gpus = _detect_gpu_count()
    if n_gpus == 0:
        raise SystemExit("No GPUs visible; --parallel requires at least one CUDA device.")
    print(f"{header}: {len(cfg_keys)} cells across {n_gpus} GPU(s) (round-robin).", flush=True)

    from . import storage
    storage.ensure_dirs()

    extra_args: list[str] = []
    for k, v in (step_overrides or {}).items():
        if k == "use_lora_in_stage3" and v is False:
            extra_args.append("--no-lora")
        elif k == "batch_size":
            extra_args += ["--batch-size", str(v)]
        elif k.startswith("stage") and k.endswith("_steps"):
            extra_args += [f"--{k.replace('_', '-')}", str(v)]

    procs: dict[int, tuple] = {}
    queue = list(cfg_keys)
    next_gpu = 0
    completed: list[tuple[str, bool]] = []

    while queue or procs:
        while queue and len(procs) < n_gpus:
            key = queue.pop(0)
            cell_id = _cfg_key_to_id(key)
            log_path = storage.RUNS / cell_id / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(next_gpu % n_gpus)
            train_cmd = [sys.executable, "-m", "exp01.cli", "train", key] + extra_args
            eval_cmd  = [sys.executable, "-m", "exp01.cli", "eval",  key] + extra_args
            f = open(log_path, "w")
            print(f"  [GPU{env['CUDA_VISIBLE_DEVICES']}] launching {cell_id}", flush=True)
            # Run train; on success, run eval. We do this via a tiny shell
            # one-liner so a single Popen tracks the whole pipeline.
            shell_cmd = " && ".join([
                _quote_argv(train_cmd),
                _quote_argv(eval_cmd),
            ])
            p = subprocess.Popen(shell_cmd, shell=True, env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
            procs[p.pid] = (p, cell_id, log_path, env["CUDA_VISIBLE_DEVICES"], f)
            next_gpu += 1

        time.sleep(10)
        done = [pid for pid, (p, *_rest) in procs.items() if p.poll() is not None]
        for pid in done:
            p, cell_id, log_path, gpu, f = procs.pop(pid)
            f.close()
            ok = p.returncode == 0
            print(f"  [GPU{gpu}] {'OK ' if ok else 'FAIL'} {cell_id} (rc={p.returncode}); "
                  f"log: {log_path}", flush=True)
            completed.append((cell_id, ok))

    print(f"\n=== {header} summary: {sum(1 for _, ok in completed if ok)}/{len(completed)} succeeded ===")
    for cid, ok in completed:
        print(f"  {'OK ' if ok else 'FAIL'} {cid}")


def _quote_argv(argv: list[str]) -> str:
    import shlex
    return " ".join(shlex.quote(a) for a in argv)


def _detect_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _cfg_key_to_id(key: str) -> str:
    """Build the same cell_id as CellConfig without instantiating it (so we
    don't pay model-load cost just to plan logs)."""
    from .config import CellConfig
    return CellConfig(**_parse_cfg_key(key)).cell_id


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

    def _step_flags(p):
        p.add_argument("--stage1-steps", type=int, default=None)
        p.add_argument("--stage2-steps", type=int, default=None)
        p.add_argument("--stage3-steps", type=int, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--no-lora", action="store_true",
                       help="Skip Stage 3 LoRA SFT")

    sp = sub.add_parser("train")
    sp.add_argument("cfg_key", help="encoder.bridge.input.fold (e.g. reve.linear.eeg.0)")
    _step_flags(sp)
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("eval")
    sp.add_argument("cfg_key")
    _step_flags(sp)
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("pilot")
    sp.add_argument("--encoders", default="reve,tfm",
                    help="comma-separated subset of {reve,diver1,tfm}")
    sp.add_argument("--cells", default=None,
                    help="comma-separated cfg_keys (overrides --encoders)")
    sp.add_argument("--parallel", action="store_true",
                    help="Run one cell per visible GPU (round-robin) as subprocesses")
    _step_flags(sp)
    sp.set_defaults(func=_cmd_pilot)

    sp = sub.add_parser("full")
    sp.add_argument("--surviving", default="reve:linear,tfm:vocab",
                    help="comma-separated <encoder>:<bridge> pairs")
    sp.set_defaults(func=_cmd_full)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
