"""Single CLI entry point for exp02.

Available subcommands::

  exp02 download-models      one-shot HF cache (REVE + REVE-positions + TFM)
  exp02 download-data        one-shot HF cache for the unified dataset (~72 GB)
  exp02 make-splits          compute & persist the 5 LNSO folds (shared with exp01)
  exp02 build-bpe            train sentencepiece BPE on ZuCo train + WikiText
  exp02 build-kenlm          build KenLM 4-gram on the BPE training corpus
  exp02 smoke                end-to-end smoke (~3 min on H100)
  exp02 train CFG_KEY        train a single cell (CFG_KEY = encoder.vocab.variant.input.fold)
  exp02 eval  CFG_KEY        evaluate a single cell (greedy + beam + beam_kenlm)
  exp02 gap   CFG_KEY        compute matched-pair §4.3 gap for an EEG cell vs its noise twin
  exp02 pilot                run the full Track-C scope (~14 cells)
  exp02 sweep                pilot + 5-fold extension on the survivor

Env vars::

  EXP02_DATA_ROOT            where everything lives (default ./data)
  HF_TOKEN                   gated HF repos
  WANDB_API_KEY              W&B logging (optional; runs offline if unset)
  WANDB_PROJECT              default exp02-eeg-ctc
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path


# ============================================================================
# Setup commands
# ============================================================================


def _cmd_download_models(args):
    from huggingface_hub import snapshot_download
    from eeg_common import encoders as _enc

    from . import storage
    storage.ensure_dirs()

    repos = [
        ("brain-bzh/reve-base", "model"),
        ("brain-bzh/reve-positions", "model"),
        ("Jathurshan/TFM-Tokenizer", "model"),
    ]
    fail: list[str] = []
    for repo_id, repo_type in repos:
        print(f"[download] {repo_id} ({repo_type})", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id, repo_type=repo_type,
                cache_dir=str(storage.HF_CACHE),
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as e:
            print(f"[download][FAIL] {repo_id}: {type(e).__name__}: {e}",
                  flush=True)
            fail.append(repo_id)
    print(f"[download] TFM source -> {_enc.ensure_tfm_source(storage.STORAGE)}",
          flush=True)
    if fail:
        print("FAILED:", fail, file=sys.stderr)
        sys.exit(1)


def _cmd_download_data(args):
    from eeg_common import data as _data
    from . import storage
    _data.download_dataset(storage.STORAGE)


def _cmd_make_splits(args):
    from eeg_common import splits as _splits
    from . import storage
    _splits.write_splits(storage.STORAGE)


def _cmd_build_bpe(args):
    from . import tokenizer_build
    tokenizer_build.train_bpe_tokenizer(
        vocab_size=args.vocab_size,
        fold=args.fold,
        max_wiki_lines=args.max_wiki_lines,
    )


def _cmd_build_kenlm(args):
    from . import kenlm_build
    kenlm_build.build_kenlm(
        order=args.order,
        fold=args.fold,
        max_wiki_lines=args.max_wiki_lines,
    )


def _cmd_build_paraphrases(args):
    from . import text_augment
    text_augment.build_paraphrases(
        n_per_sentence=args.n_per_sentence,
        concurrency=args.concurrency,
        model=args.model,
    )


# ============================================================================
# Train / eval / gap / smoke / pilot
# ============================================================================


def _finish_wandb():
    try:
        import wandb
        if wandb.run is not None:
            wandb.run.finish()
    except Exception:
        pass


def _parse_cfg_key(key: str) -> dict:
    """encoder.vocab.variant.input.fold -> CTCConfig kwargs."""
    parts = key.split(".")
    if len(parts) != 5:
        raise ValueError(
            f"expected encoder.vocab.variant.input.fold, got {key!r}")
    enc, vocab, variant, inp, fold = parts
    return {"encoder": enc, "vocab": vocab, "variant": variant,
            "input": inp, "fold": int(fold)}


_FLOAT_OVERRIDE_FIELDS = (
    "head_lr", "encoder_lr",
    "label_prior_weight", "cr_ctc_kl_weight",
    "intermediate_ctc_weight", "aed_weight",
    # Signal-aug knobs
    "signal_aug_time_shift_max_frac",
    "signal_aug_channel_dropout_p", "signal_aug_channel_dropout_frac",
    "signal_aug_freq_mask_p", "signal_aug_freq_mask_max_hz",
    "signal_aug_time_warp_p",
    "signal_aug_time_warp_factor_low", "signal_aug_time_warp_factor_high",
    "signal_aug_gaussian_noise_sigma",
    "signal_aug_fourier_surrogate_p",
    "signal_aug_mixup_alpha",
    # Text-aug
    "text_aug_prob",
)
_INT_OVERRIDE_FIELDS = (
    "total_steps", "warmup_steps", "batch_size", "grad_accum",
    "encoder_warmup_freeze_steps", "num_workers",
    "signal_aug_freq_mask_n", "signal_aug_time_warp_segments",
)
_STR_OVERRIDE_FIELDS = ("text_aug_paraphrase_path",)


def _cli_overrides(args) -> dict:
    out: dict = {}
    for name in _INT_OVERRIDE_FIELDS:
        v = getattr(args, name, None)
        if v is not None:
            out[name] = int(v)
    for name in _FLOAT_OVERRIDE_FIELDS:
        v = getattr(args, name, None)
        if v is not None:
            out[name] = float(v)
    for name in _STR_OVERRIDE_FIELDS:
        v = getattr(args, name, None)
        if v is not None:
            out[name] = str(v)
    if getattr(args, "encoder_finetune", None):
        out["encoder_finetune"] = args.encoder_finetune
    if getattr(args, "preprocess", None):
        out["preprocess"] = args.preprocess
    if getattr(args, "no_specaugment", False):
        out["specaugment"] = False
    return out


def _cmd_train(args):
    from .config import CTCConfig
    from . import train as trainmod
    cfg = CTCConfig(**_parse_cfg_key(args.cfg_key), **_cli_overrides(args))
    trainmod.train(cfg)


def _cmd_eval(args):
    from .config import CTCConfig
    from . import eval as evalmod
    cfg = CTCConfig(**_parse_cfg_key(args.cfg_key), **_cli_overrides(args))
    summary = evalmod.evaluate_cell(cfg)
    for mode, m_by_metric in summary["scores"].items():
        print(f"\n[{mode}]")
        for k, v in m_by_metric.items():
            print(f"  {k:>15s}: {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")


def _cmd_gap(args):
    """Compute the §4.3 matched-pair gap for an EEG cell vs its noise twin."""
    import json
    from .config import CTCConfig
    from . import eval as evalmod
    from . import storage as _storage

    eeg_cfg = CTCConfig(**_parse_cfg_key(args.cfg_key), **_cli_overrides(args))
    if eeg_cfg.input != "eeg":
        raise ValueError("`exp02 gap` expects an EEG cfg_key (input=eeg).")
    noise_cfg = replace(eeg_cfg, input="noise_train")

    eeg_path = _storage.cell_eval_dir(eeg_cfg.cell_id) / "metrics.json"
    noise_path = _storage.cell_eval_dir(noise_cfg.cell_id) / "metrics.json"
    if not eeg_path.exists():
        raise FileNotFoundError(f"missing eval metrics: {eeg_path}")
    if not noise_path.exists():
        raise FileNotFoundError(f"missing eval metrics: {noise_path}")
    eeg = json.loads(eeg_path.read_text())
    noise = json.loads(noise_path.read_text())
    gap = evalmod.matched_pair_gap(eeg, noise)
    print(json.dumps(gap, indent=2))

    out = _storage.cell_eval_dir(eeg_cfg.cell_id) / "gap_vs_noise.json"
    out.write_text(json.dumps(gap, separators=(",", ":")))
    print(f"\n[gap] wrote {out}", flush=True)


# ----------------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------------


def _cmd_smoke(args):
    """End-to-end micro-run. One cell per code path:
       - reve.char.ctc.eeg.0   (vanilla CTC + char vocab — no BPE / KenLM needed)
       - reve.char.crctc.eeg.0 (CR-CTC + char vocab)
    """
    import json
    import traceback as _tb

    from .config import CTCConfig
    from . import eval as evalmod
    from . import storage as _storage
    from . import train as trainmod

    cells = [
        CTCConfig(encoder="reve", vocab="char", variant="ctc",
                  total_steps=20, warmup_steps=5, batch_size=2, grad_accum=1,
                  encoder_warmup_freeze_steps=5,
                  encoder_finetune="frozen",
                  cr_ctc_kl_weight=0.0, label_prior_weight=0.3),
        CTCConfig(encoder="reve", vocab="char", variant="crctc",
                  total_steps=20, warmup_steps=5, batch_size=2, grad_accum=1,
                  encoder_warmup_freeze_steps=5,
                  encoder_finetune="frozen"),
    ]

    fails: list[tuple[str, str]] = []
    for cfg in cells:
        print(f"\n========== SMOKE: {cfg.cell_id} ==========\n", flush=True)
        try:
            ckpt = trainmod.train(cfg)
            evalmod.evaluate_cell(cfg, ckpt_path=ckpt,
                                  decode_modes=("greedy",))
        except Exception as e:
            fails.append((cfg.cell_id, repr(e)))
            print(f"FAILED: {e}", flush=True)
            _tb.print_exc()
            _finish_wandb()
            continue
        _finish_wandb()

        run_dir = _storage.cell_run_dir(cfg.cell_id)
        eval_dir = _storage.cell_eval_dir(cfg.cell_id)
        expected = [
            run_dir / "log.jsonl",
            run_dir / "sample_gens.jsonl",
            run_dir / "stats.jsonl",
            run_dir / "model.pt",
            eval_dir / "metrics.json",
            eval_dir / "predictions.parquet",
        ]
        missing = [p for p in expected if not p.exists() or p.stat().st_size == 0]
        if missing:
            fails.append((cfg.cell_id,
                          f"missing/empty: {[str(p) for p in missing]}"))
            continue

        import pyarrow.parquet as pq
        rows = pq.read_table(eval_dir / "predictions.parquet").to_pylist()
        if rows:
            print(f"  {len(rows)} test examples", flush=True)
            print(f"  first ref: {rows[0]['ref'][:80]!r}", flush=True)
            for k in ("hyp_greedy", "hyp_beam", "hyp_beam_kenlm"):
                if k in rows[0]:
                    print(f"  first {k}: {rows[0][k][:80]!r}", flush=True)
        m = json.loads((eval_dir / "metrics.json").read_text())
        for mode, by_metric in m["scores"].items():
            print(f"\n  [{mode}]", flush=True)
            for k, v in by_metric.items():
                print(f"    {k:>15s}: {v['mean']:.3f}", flush=True)

    if fails:
        print(f"\n=== Smoke summary: {len(cells) - len(fails)} passed, "
              f"{len(fails)} failed ===")
        for cid, err in fails:
            print(f"  FAIL {cid}: {err}")
        sys.exit(1)
    print(f"\n=== Smoke summary: all {len(cells)} cells passed ===")


# ----------------------------------------------------------------------------
# Pilot (Track-C: ~14 cells in parallel)
# ----------------------------------------------------------------------------


def _cmd_pilot(args):
    from .config import (
        all_track_c_cells,
        encoder_ablation_cells,
        freeze_ablation_cells,
        headline_cells,
        variant_ablation_cells,
        vocab_ablation_cells,
    )

    if args.group == "all":
        cells = all_track_c_cells(include_diver1=args.include_diver1)
    elif args.group == "headline":
        cells = headline_cells()
    elif args.group == "encoder":
        cells = encoder_ablation_cells(include_diver1=args.include_diver1)
    elif args.group == "vocab":
        cells = vocab_ablation_cells()
    elif args.group == "variant":
        cells = variant_ablation_cells()
    elif args.group == "freeze":
        cells = freeze_ablation_cells()
    else:
        raise ValueError(f"unknown pilot group: {args.group}")

    overrides = _cli_overrides(args)
    if overrides:
        cells = [replace(c, **overrides) for c in cells]

    if args.parallel:
        _run_parallel(cells, header=f"Pilot[{args.group}]")
        return

    from . import eval as evalmod
    from . import train as trainmod
    import traceback as _tb

    print(f"Pilot: {len(cells)} cells, running sequentially.")
    results: list[tuple[str, dict | None]] = []
    for cfg in cells:
        print(f"\n=========== {cfg.cell_id} ===========", flush=True)
        try:
            ckpt = trainmod.train(cfg)
            summary = evalmod.evaluate_cell(cfg, ckpt_path=ckpt)
            results.append((cfg.cell_id, summary))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
            _tb.print_exc()
            results.append((cfg.cell_id, None))
        _finish_wandb()

    print("\n=== Pilot ranking (best-decode CER on EEG cells) ===")
    ranked = []
    for cid, summary in results:
        if summary is None or "_eeg_" not in cid:
            continue
        best_cer = float("inf")
        best_mode = ""
        for mode, by_metric in summary["scores"].items():
            cer = by_metric.get("cer", {}).get("mean", float("inf"))
            if cer < best_cer:
                best_cer = cer
                best_mode = mode
        ranked.append((cid, best_cer, best_mode))
    ranked.sort(key=lambda r: r[1])
    for cid, cer, mode in ranked:
        print(f"  {cid:<70s}  CER={cer:.3f} ({mode})")


def _cmd_sweep(args):
    """Pilot + 5-fold extension on the best EEG cell from the pilot."""
    raise NotImplementedError(
        "exp02 sweep is a thin wrapper TODO: run pilot, identify survivor "
        "by best CER on the EEG twin, then call fold_extension_cells(survivor) "
        "and run those. Implement after the first pilot run identifies which "
        "cell is the survivor."
    )


# ----------------------------------------------------------------------------
# Parallel orchestration (one cell per visible GPU, round-robin)
# ----------------------------------------------------------------------------


def _run_parallel(cells, *, header: str) -> None:
    """Round-robin one cell per visible GPU. Each cell is a ``CTCConfig``;
    we compute its diff against the default config built from its ``cfg_key``
    and emit those as ``--<flag> value`` overrides so two cells with the same
    ``cfg_key`` but different bookkeeping (e.g. ``encoder_finetune=frozen``)
    end up in distinct run dirs.
    """
    n_gpus = _detect_gpu_count()
    if n_gpus == 0:
        raise SystemExit("No GPUs visible; --parallel requires at least one CUDA device.")
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_visible:
        physical_gpus = [g.strip() for g in parent_visible.split(",") if g.strip()]
    else:
        physical_gpus = [str(i) for i in range(n_gpus)]
    n_gpus = min(n_gpus, len(physical_gpus))
    print(f"{header}: {len(cells)} cells across {n_gpus} GPU(s) "
          f"(physical: {physical_gpus[:n_gpus]}).", flush=True)

    from . import storage
    storage.ensure_dirs()

    procs: dict[int, tuple] = {}
    queue = list(cells)
    next_gpu = 0
    completed: list[tuple[str, bool]] = []

    while queue or procs:
        while queue and len(procs) < n_gpus:
            cfg = queue.pop(0)
            extra_args = _diff_args(cfg)
            cell_id = cfg.cell_id
            log_path = storage.RUNS / cell_id / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = physical_gpus[next_gpu % n_gpus]
            train_cmd = ([sys.executable, "-m", "exp02.cli", "train", cfg.cfg_key]
                         + extra_args)
            eval_cmd = ([sys.executable, "-m", "exp02.cli", "eval", cfg.cfg_key]
                        + extra_args)
            f = open(log_path, "w")
            print(f"  [GPU{env['CUDA_VISIBLE_DEVICES']}] launching {cell_id}",
                  flush=True)
            shell_cmd = " && ".join([_quote_argv(train_cmd),
                                     _quote_argv(eval_cmd)])
            p = subprocess.Popen(shell_cmd, shell=True, env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
            procs[p.pid] = (p, cell_id, log_path,
                            env["CUDA_VISIBLE_DEVICES"], f)
            next_gpu += 1

        time.sleep(10)
        done = [pid for pid, (p, *_rest) in procs.items() if p.poll() is not None]
        for pid in done:
            p, cell_id, log_path, gpu, f = procs.pop(pid)
            f.close()
            ok = p.returncode == 0
            print(f"  [GPU{gpu}] {'OK ' if ok else 'FAIL'} {cell_id} "
                  f"(rc={p.returncode}); log: {log_path}", flush=True)
            completed.append((cell_id, ok))

    print(f"\n=== {header} summary: "
          f"{sum(1 for _, ok in completed if ok)}/{len(completed)} succeeded ===")
    for cid, ok in completed:
        print(f"  {'OK ' if ok else 'FAIL'} {cid}")


# Subset of CTCConfig fields the CLI knows how to override. Anything outside
# this set that diverges from the cfg-key default will trigger a warning in
# ``_diff_args`` so we don't silently drop important deltas.
_DIFF_ABLE_FIELDS = (
    {"preprocess", "encoder_finetune", "specaugment"}
    | set(_INT_OVERRIDE_FIELDS) | set(_FLOAT_OVERRIDE_FIELDS) | set(_STR_OVERRIDE_FIELDS)
)


def _diff_args(cfg) -> list[str]:
    """CLI flags needed to reconstruct ``cfg`` from ``CTCConfig(**parse_cfg_key(cfg.cfg_key))``.

    Returns the list of ``--flag value`` arguments that, when applied on top
    of the default cfg derived from the key, yield ``cfg``.
    """
    from .config import CTCConfig
    base = CTCConfig(**_parse_cfg_key(cfg.cfg_key))
    args: list[str] = []
    for fname in _DIFF_ABLE_FIELDS:
        bv = getattr(base, fname)
        cv = getattr(cfg, fname)
        if bv == cv:
            continue
        if fname == "specaugment" and cv is False:
            args.append("--no-specaugment")
        elif isinstance(cv, bool):
            # No CLI flag for True overrides; they should already be the default.
            continue
        else:
            args += ["--" + fname.replace("_", "-"), str(cv)]
    # Safety: surface any non-diff-able field that diverges from base, so we
    # don't silently launch the wrong cell.
    for fname, bv in vars(base).items():
        if fname in _DIFF_ABLE_FIELDS:
            continue
        cv = getattr(cfg, fname)
        if bv != cv:
            print(f"[parallel] WARNING: cfg field {fname!r} diverges from "
                  f"the cfg_key default ({bv!r} -> {cv!r}) but is not in "
                  f"_DIFF_ABLE_FIELDS; the spawned subprocess will get the "
                  f"default value, NOT the requested {cv!r}. Add {fname!r} "
                  f"to _DIFF_ABLE_FIELDS and the CLI's --flag set.",
                  flush=True)
    return args


def _quote_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in argv)


def _detect_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def _cfg_key_to_id(key: str) -> str:
    from .config import CTCConfig
    return CTCConfig(**_parse_cfg_key(key)).cell_id


# ============================================================================
# argparse
# ============================================================================


def _step_flags(p):
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--encoder-warmup-freeze-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--encoder-lr", type=float, default=None)
    p.add_argument("--encoder-finetune", default=None,
                   choices=("full", "lora", "frozen"))
    p.add_argument("--preprocess", default=None,
                   choices=("v1", "v2", "v2_reve", "v2_tfm", "v2_dk25"))
    p.add_argument("--no-specaugment", action="store_true")
    # CTC variant weights (used both as direct user overrides and as the
    # _diff_args targets for the parallel pilot orchestrator).
    p.add_argument("--label-prior-weight", type=float, default=None)
    p.add_argument("--cr-ctc-kl-weight", type=float, default=None)
    p.add_argument("--intermediate-ctc-weight", type=float, default=None)
    p.add_argument("--aed-weight", type=float, default=None)
    # Signal-aug knobs (default OFF; enable per cell)
    p.add_argument("--signal-aug-time-shift-max-frac", type=float, default=None)
    p.add_argument("--signal-aug-channel-dropout-p", type=float, default=None)
    p.add_argument("--signal-aug-channel-dropout-frac", type=float, default=None)
    p.add_argument("--signal-aug-freq-mask-p", type=float, default=None)
    p.add_argument("--signal-aug-freq-mask-n", type=int, default=None)
    p.add_argument("--signal-aug-freq-mask-max-hz", type=float, default=None)
    p.add_argument("--signal-aug-time-warp-p", type=float, default=None)
    p.add_argument("--signal-aug-time-warp-segments", type=int, default=None)
    p.add_argument("--signal-aug-time-warp-factor-low", type=float, default=None)
    p.add_argument("--signal-aug-time-warp-factor-high", type=float, default=None)
    p.add_argument("--signal-aug-gaussian-noise-sigma", type=float, default=None)
    p.add_argument("--signal-aug-fourier-surrogate-p", type=float, default=None)
    p.add_argument("--signal-aug-mixup-alpha", type=float, default=None)
    # Text-aug knobs
    p.add_argument("--text-aug-prob", type=float, default=None,
                   help="Probability of substituting a paraphrase as the CTC target per row.")
    p.add_argument("--text-aug-paraphrase-path", default=None,
                   help="Path to paraphrases parquet (default: $EXP02_DATA_ROOT/text_aug/paraphrases.parquet).")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="exp02", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download-models").set_defaults(func=_cmd_download_models)
    sub.add_parser("download-data").set_defaults(func=_cmd_download_data)
    sub.add_parser("make-splits").set_defaults(func=_cmd_make_splits)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    sp = sub.add_parser("build-bpe")
    sp.add_argument("--vocab-size", type=int, default=1024)
    sp.add_argument("--fold", type=int, default=0)
    sp.add_argument("--max-wiki-lines", type=int, default=1_000_000)
    sp.set_defaults(func=_cmd_build_bpe)

    sp = sub.add_parser("build-kenlm")
    sp.add_argument("--order", type=int, default=4)
    sp.add_argument("--fold", type=int, default=0)
    sp.add_argument("--max-wiki-lines", type=int, default=1_000_000)
    sp.set_defaults(func=_cmd_build_kenlm)

    sp = sub.add_parser("build-paraphrases",
                        help="LLM paraphrase ZuCo train sentences (OpenAI). "
                             "Idempotent; tops up missing entries.")
    sp.add_argument("--n-per-sentence", type=int, default=3)
    sp.add_argument("--concurrency", type=int, default=20)
    sp.add_argument("--model", default="gpt-4o-mini")
    sp.set_defaults(func=_cmd_build_paraphrases)

    sp = sub.add_parser("train")
    sp.add_argument("cfg_key",
                    help="encoder.vocab.variant.input.fold "
                         "(e.g. reve.bpe1k.crctc.eeg.0)")
    _step_flags(sp)
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("eval")
    sp.add_argument("cfg_key")
    _step_flags(sp)
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("gap")
    sp.add_argument("cfg_key", help="EEG cfg_key (e.g. reve.bpe1k.crctc.eeg.0). "
                                    "Looks up its noise_train twin and reports "
                                    "the matched §4.3 gap.")
    _step_flags(sp)
    sp.set_defaults(func=_cmd_gap)

    sp = sub.add_parser("pilot")
    sp.add_argument("--group", default="all",
                    choices=("all", "headline", "encoder", "vocab",
                             "variant", "freeze"),
                    help="Which subset of the Track-C scope to run.")
    sp.add_argument("--include-diver1", action="store_true",
                    help="Add DIVER-1 cells (requires checkpoint at "
                         "$EXP02_DATA_ROOT/diver1/pytorch_model.bin)")
    sp.add_argument("--parallel", action="store_true",
                    help="Round-robin one cell per visible GPU as subprocesses.")
    _step_flags(sp)
    sp.set_defaults(func=_cmd_pilot)

    sp = sub.add_parser("sweep")
    sp.set_defaults(func=_cmd_sweep)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
