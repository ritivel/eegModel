"""exp03 CLI — entrypoints for everything we run from the GPU box.

Top-level commands (each a ``typer`` subcommand):

    exp03 paths                  — print the resolved storage layout (sanity)
    exp03 list-releases          — list all HBN-EEG releases on s3://fcp-indi/
    exp03 list-subjects          — list subjects in one release
    exp03 download               — pull raw .set/.fdt + sidecars for a (release, subject) range
    exp03 audit                  — Phase-0 data audit (Karpathy step 1) on the local raw dir
    exp03 preprocess             — apply minimal (and/or v2_clean) pipelines → parquet shards
    exp03 sync-derived-up        — rclone the local derived/ tree to s3://eegmodel-warehouse/
    exp03 sync-derived-down      — rclone s3://eegmodel-warehouse/derived/ → local NVMe (bootstrap)

All commands respect ``$EXP03_DATA_ROOT`` (default ``/opt/dlami/nvme/eeg``).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import hbn, preprocess, storage

app = typer.Typer(
    name="exp03",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help=(
        "exp03 — self-supervised EEG pretraining on HBN-EEG. "
        "All paths and S3 prefixes resolve from $EXP03_DATA_ROOT (default "
        f"{storage.DEFAULT_DATA_ROOT})."
    ),
)
console = Console()


# ---------------------------------------------------------------------------
# `exp03 paths` — sanity check the resolved storage layout
# ---------------------------------------------------------------------------


@app.command("paths")
def paths_cmd():
    """Print the resolved storage paths and S3 prefixes."""
    s = storage.from_env()
    table = Table(title="exp03 storage layout", show_lines=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Path / URI")
    table.add_row("data_root", str(s.data_root))
    table.add_row("raw_hbn", str(s.raw_hbn))
    table.add_row(f"derived/{storage.PIPELINE_MINIMAL}",
                  str(s.derived_pipeline(storage.PIPELINE_MINIMAL)))
    table.add_row(f"derived/{storage.PIPELINE_V2_CLEAN}",
                  str(s.derived_pipeline(storage.PIPELINE_V2_CLEAN)))
    table.add_row("runs_root", str(s.runs_root))
    table.add_row("hf_cache", str(s.hf_cache))
    table.add_row("[s3] derived(minimal)", s.s3_derived(storage.PIPELINE_MINIMAL))
    table.add_row("[s3] derived(v2_clean)", s.s3_derived(storage.PIPELINE_V2_CLEAN))
    table.add_row("[s3] runs/exp03", s.s3_uri("runs", "exp03"))
    console.print(table)


# ---------------------------------------------------------------------------
# `exp03 list-releases` and `exp03 list-subjects`
# ---------------------------------------------------------------------------


@app.command("list-releases")
def list_releases_cmd():
    """List all HBN-EEG release folders under s3://fcp-indi/.../HBN/EEG/."""
    rels = hbn.list_releases()
    console.print(f"Found {len(rels)} releases: " + ", ".join(rels))


@app.command("list-subjects")
def list_subjects_cmd(
    release: str = typer.Argument(..., help="HBN release, e.g. 'NC' or 'R6'"),
    max_subjects: int = typer.Option(None, "--max", help="Cap output to N subjects"),
):
    """List subject IDs in a given release."""
    subs = hbn.list_subjects(release, max_subjects=max_subjects)
    for s in subs:
        print(s)
    console.print(f"\n[bold]{len(subs)}[/bold] subjects in release [cyan]{release}[/cyan]")


# ---------------------------------------------------------------------------
# `exp03 download`
# ---------------------------------------------------------------------------


@app.command("download")
def download_cmd(
    release: str = typer.Argument(..., help="HBN release, e.g. 'NC' or 'R6'"),
    max_subjects: int = typer.Option(5, "--max-subjects",
                                     help="Number of subjects to download (Tier 1 default = 5)"),
    overwrite: bool = typer.Option(False, "--overwrite",
                                   help="Re-download even if local file exists"),
    sidecars_only: bool = typer.Option(False, "--sidecars-only",
                                       help="Only download .tsv/.json sidecars (no .set/.fdt)"),
):
    """Download raw .set/.fdt + BIDS sidecars + release metadata to local NVMe.

    By default pulls 5 subjects (Tier 1 — enough for mini-experiment 01
    sanity baselines). Use --max-subjects 200 for the full 100h subset.
    """
    s = storage.from_env()
    s.ensure_dirs()

    console.print(f"[cyan]Release:[/cyan] {release}")
    console.print(f"[cyan]Local raw root:[/cyan] {s.raw_hbn}")

    # Release-level metadata (participants.tsv, dataset_description.json)
    console.print("\n[bold]Step 1: release metadata[/bold]")
    meta = hbn.download_release_metadata(release, s.raw_hbn, overwrite=overwrite)
    for fname, path in meta.items():
        console.print(f"  {fname}: {path}")

    # Subjects
    subjects = hbn.list_subjects(release, max_subjects=max_subjects)
    console.print(f"\n[bold]Step 2: download {len(subjects)} subjects[/bold]")

    from tqdm.auto import tqdm

    total_bytes = 0
    t0 = time.time()
    for sub in tqdm(subjects, desc="subjects", unit="sub"):
        # Sidecars first (small, parallelisable)
        sidecars = hbn.download_subject_sidecars(sub, release, s.raw_hbn, overwrite=overwrite)
        for p in sidecars:
            total_bytes += p.stat().st_size

        if sidecars_only:
            continue

        # Recordings
        recordings = hbn.list_subject_recordings(sub, release)
        for rec in recordings:
            set_path, fdt_path = hbn.download_recording(rec, s.raw_hbn, overwrite=overwrite)
            total_bytes += set_path.stat().st_size
            if fdt_path is not None:
                total_bytes += fdt_path.stat().st_size

    dt = time.time() - t0
    console.print(
        f"\n[green]done[/green]: {len(subjects)} subjects, "
        f"{total_bytes / 2**30:.2f} GB in {dt:.1f}s "
        f"({total_bytes / 2**20 / max(dt, 1e-3):.1f} MB/s)"
    )


# ---------------------------------------------------------------------------
# `exp03 audit` — Phase-0 data audit (per methodology.md §2 Phase 0)
# ---------------------------------------------------------------------------


@app.command("audit")
def audit_cmd(
    release: str = typer.Argument(..., help="HBN release to audit (must already be downloaded)"),
    n_subjects: int = typer.Option(None, "--n", help="Audit only the first N subjects"),
):
    """Phase-0 data audit: counts, hours, channels, NaN fraction, amplitude stats.

    Reads the local raw directory only — no model code touched. This is
    the "become one with the data" step from methodology.md §2 Phase 0.
    Per Karpathy: do not touch any neural net code until you can describe
    the data on one sheet of paper.
    """
    import numpy as np
    import pandas as pd

    s = storage.from_env()
    rel_dir = s.raw_hbn / hbn.release_s3_prefix(release)
    if not rel_dir.exists():
        console.print(f"[red]error:[/red] {rel_dir} not found — run "
                      f"`exp03 download {release}` first")
        raise typer.Exit(1)

    # Load participants.tsv if present
    participants_path = rel_dir / "participants.tsv"
    if participants_path.exists():
        df = hbn.load_participants(participants_path)
        n_total = len(df)
        n_adhd = int((df["adhd"] == 1).sum())
        n_no = int((df["adhd"] == 0).sum())
        n_missing = int((df["adhd"] == -1).sum())
    else:
        df = None
        n_total = n_adhd = n_no = n_missing = 0

    # Walk subject directories
    subject_dirs = sorted([p for p in rel_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if n_subjects:
        subject_dirs = subject_dirs[:n_subjects]

    rows = []
    from tqdm.auto import tqdm

    for sub_dir in tqdm(subject_dirs, desc="audit", unit="sub"):
        sub_id = sub_dir.name[len("sub-"):]
        eeg_dir = sub_dir / "eeg"
        if not eeg_dir.exists():
            continue
        for set_path in eeg_dir.glob("*.set"):
            try:
                eeg, sr, ch_names = hbn.load_recording(set_path)
            except Exception as e:  # noqa: BLE001
                rows.append({"subject_id": sub_id, "file": set_path.name,
                             "ok": False, "error": str(e)[:80]})
                continue

            n_nan = int(np.isnan(eeg).sum())
            rows.append({
                "subject_id": sub_id,
                "file": set_path.name,
                "ok": True,
                "n_channels": eeg.shape[0],
                "n_samples": eeg.shape[1],
                "duration_s": eeg.shape[1] / sr,
                "sr_hz": sr,
                "nan_frac": n_nan / max(eeg.size, 1),
                "amp_min": float(np.nanmin(eeg)),
                "amp_max": float(np.nanmax(eeg)),
                "amp_std": float(np.nanstd(eeg)),
            })

    audit_df = pd.DataFrame(rows)

    # ---- summary table -----------------------------------------------------
    table = Table(title=f"HBN-EEG release {release} — data audit", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    n_files = len(audit_df)
    n_ok = int(audit_df["ok"].sum()) if n_files else 0
    n_failed = n_files - n_ok
    total_hours = float(audit_df.loc[audit_df["ok"], "duration_s"].sum() / 3600.0) if n_ok else 0.0

    table.add_row("subjects in participants.tsv", str(n_total))
    table.add_row("  with ADHD label", str(n_adhd))
    table.add_row("  with no-diagnosis label", str(n_no))
    table.add_row("  with missing diagnosis", str(n_missing))
    table.add_row("audit subject count", str(len(subject_dirs)))
    table.add_row("recording files seen", str(n_files))
    table.add_row("  OK", str(n_ok))
    table.add_row("  failed", str(n_failed))
    table.add_row("total recording hours", f"{total_hours:.2f}")

    if n_ok:
        ok = audit_df[audit_df["ok"]]
        table.add_row("unique sample rates", str(sorted(ok["sr_hz"].unique().tolist())))
        table.add_row("unique channel counts", str(sorted(ok["n_channels"].unique().tolist())))
        table.add_row("median recording duration (s)", f"{ok['duration_s'].median():.1f}")
        table.add_row("median amp_std (V)", f"{ok['amp_std'].median():.3e}")
        table.add_row("max NaN fraction across files", f"{ok['nan_frac'].max():.3e}")

    console.print(table)

    # Dump audit_df to CSV for the record
    audit_csv = s.scratch / f"audit_{release}.csv"
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(audit_csv, index=False)
    console.print(f"\n[dim]Per-file audit written to {audit_csv}[/dim]")


# ---------------------------------------------------------------------------
# `exp03 preprocess`
# ---------------------------------------------------------------------------


@app.command("preprocess")
def preprocess_cmd(
    release: str = typer.Argument(..., help="HBN release, e.g. 'NC' or 'R6'"),
    pipeline: str = typer.Option("minimal", "--pipeline",
                                 help="Pipeline name: 'minimal' (primary), 'v2_clean' (F0-prep), or 'both'"),
    n_subjects: int = typer.Option(None, "--n",
                                   help="Process only the first N subjects (for Tier 1 sanity)"),
    overwrite: bool = typer.Option(False, "--overwrite",
                                   help="Re-write parquet shards even if they exist"),
):
    """Apply preprocess pipelines per `mini_experiments.md` §4.1 → parquet shards.

    Reads raw .set/.fdt from local NVMe, applies the chosen pipeline(s),
    windows, iid-expands, writes one parquet shard per (subject, recording)
    under ``$EXP03_DATA_ROOT/derived/<pipeline>/sub-<id>/``.

    Memory: each recording (~3 min × 128 ch × 500 Hz × float32) is ~46 MB
    raw and ~12 MB float16 after windowing — comfortably fits on one CPU
    worker. The CLI processes recordings sequentially to keep the code
    simple; for the 100h Tier-2 subset (~3000 recordings) that's about
    3 hours wall-clock single-threaded — parallelise via xargs / GNU
    parallel if you need it faster.
    """
    pipelines: list[str]
    if pipeline == "both":
        pipelines = ["minimal", "v2_clean"]
    elif pipeline in preprocess.SPECS:
        pipelines = [pipeline]
    else:
        console.print(f"[red]error:[/red] unknown pipeline {pipeline!r}; "
                      f"valid: minimal, v2_clean, both")
        raise typer.Exit(1)

    s = storage.from_env()
    s.ensure_dirs()

    rel_dir = s.raw_hbn / hbn.release_s3_prefix(release)
    if not rel_dir.exists():
        console.print(f"[red]error:[/red] no local data for release {release} at {rel_dir} — "
                      f"run `exp03 download {release}` first")
        raise typer.Exit(1)

    # Load participants.tsv (for adhd / age / sex / site labels)
    participants_path = rel_dir / "participants.tsv"
    if not participants_path.exists():
        console.print(f"[yellow]warning:[/yellow] {participants_path} missing; "
                      f"adhd/age/sex/site will be filled with sentinels")
        participants_df = None
    else:
        participants_df = hbn.load_participants(participants_path)

    # Subjects to process
    subject_dirs = sorted([p for p in rel_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if n_subjects:
        subject_dirs = subject_dirs[:n_subjects]

    from tqdm.auto import tqdm

    n_recordings = 0
    n_rows_total = 0
    t0 = time.time()
    for sub_dir in tqdm(subject_dirs, desc="subjects", unit="sub"):
        sub_id = sub_dir.name[len("sub-"):]
        eeg_dir = sub_dir / "eeg"
        if not eeg_dir.exists():
            continue

        meta_subject = (
            hbn.metadata_for_subject(sub_id, participants_df) if participants_df is not None
            else {"adhd": -1, "site": "", "age": float("nan"), "sex": ""}
        )

        for set_path in sorted(eeg_dir.glob("*.set")):
            parsed = hbn.parse_task_from_filename(set_path.stem)
            if parsed is None:
                continue
            _task, task_label = parsed
            recording_id = set_path.stem.replace(f"sub-{sub_id}_", "").replace("_eeg", "")

            # Compute provenance hash over .set + optional .fdt
            sha = hashlib.sha256()
            sha.update(set_path.read_bytes())
            fdt_sibling = set_path.with_suffix(".fdt")
            if fdt_sibling.exists():
                sha.update(fdt_sibling.read_bytes())
            src_sha8 = sha.hexdigest()[:8]

            try:
                eeg, sr_native, channel_names = hbn.load_recording(set_path)
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]skip[/yellow] {set_path.name}: {e}")
                continue

            for pl in pipelines:
                shard_dir = s.derived_pipeline(
                    storage.PIPELINE_MINIMAL if pl == "minimal" else storage.PIPELINE_V2_CLEAN
                ) / f"sub-{sub_id}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                shard_path = shard_dir / f"{recording_id}.parquet"
                if shard_path.exists() and not overwrite:
                    continue

                eeg_pp, sr_pp = preprocess.apply_pipeline(eeg, sr_native, pl)
                windows, starts = preprocess.window_4s(eeg_pp, sr_pp)
                if windows.shape[0] == 0:
                    console.print(f"[yellow]skip[/yellow] {set_path.name} ({pl}): "
                                  f"recording too short for one 4-s window "
                                  f"(samples={eeg_pp.shape[1]} sr={sr_pp})")
                    continue

                base_meta = {
                    "subject_id": sub_id,
                    "site": meta_subject["site"],
                    "recording_id": recording_id,
                    "task_label": task_label,
                    "sample_rate_hz": int(sr_pp),
                    "age": float(meta_subject["age"]),
                    "sex": meta_subject["sex"],
                    # CBCL Pearson-z factors per §4.3 Protocol A.1
                    "p_factor": float(meta_subject["p_factor"]),
                    "attention": float(meta_subject["attention"]),
                    "internalizing": float(meta_subject["internalizing"]),
                    "externalizing": float(meta_subject["externalizing"]),
                    "adhd": int(meta_subject["adhd"]),
                    "pipeline": pl,
                    "src_sha256_8": src_sha8,
                }

                rows = preprocess.iid_expand_rows(windows, starts, channel_names,
                                                  base_metadata=base_meta)
                n_written = preprocess.write_parquet_shard(rows, shard_path)
                n_rows_total += n_written

            n_recordings += 1

    dt = time.time() - t0
    console.print(
        f"\n[green]done[/green]: {n_recordings} recordings × {len(pipelines)} pipeline(s) "
        f"= {n_rows_total:,} rows in {dt:.1f}s "
        f"({n_rows_total / max(dt, 1e-3):.0f} rows/s)"
    )


# ---------------------------------------------------------------------------
# `exp03 sync-derived-up` / `exp03 sync-derived-down`
# ---------------------------------------------------------------------------


@app.command("sync-derived-up")
def sync_up_cmd(
    pipeline: str = typer.Option("both", "--pipeline",
                                 help="'minimal', 'v2_clean', or 'both'"),
):
    """Sync local derived/<pipeline>/ → s3://eegmodel-warehouse/derived/<pipeline>/.

    Idempotent (rclone copy). Use after `exp03 preprocess` to materialise
    the warehouse mirror.
    """
    import subprocess

    s = storage.from_env()
    pipelines = [storage.PIPELINE_MINIMAL, storage.PIPELINE_V2_CLEAN] if pipeline == "both" \
        else [storage.PIPELINE_MINIMAL if pipeline == "minimal" else storage.PIPELINE_V2_CLEAN]

    for pl in pipelines:
        local = s.derived_pipeline(pl)
        remote = s.s3_uri("derived", pl).replace("s3://", "s3:")
        console.print(f"[cyan]rclone copy[/cyan] {local} → {remote}")
        subprocess.run(
            ["rclone", "copy", str(local), remote, "--transfers", "32",
             "--checkers", "32", "--progress"],
            check=True,
        )


@app.command("sync-derived-down")
def sync_down_cmd(
    pipeline: str = typer.Option("both", "--pipeline",
                                 help="'minimal', 'v2_clean', or 'both'"),
):
    """Sync s3://eegmodel-warehouse/derived/<pipeline>/ → local derived/.

    The "preprocess once, ever" promise: on every new GPU box's bootstrap,
    pull the preprocessed shards instead of re-running preprocess.
    """
    import subprocess

    s = storage.from_env()
    s.ensure_dirs()
    pipelines = [storage.PIPELINE_MINIMAL, storage.PIPELINE_V2_CLEAN] if pipeline == "both" \
        else [storage.PIPELINE_MINIMAL if pipeline == "minimal" else storage.PIPELINE_V2_CLEAN]

    for pl in pipelines:
        local = s.derived_pipeline(pl)
        remote = s.s3_uri("derived", pl).replace("s3://", "s3:")
        console.print(f"[cyan]rclone copy[/cyan] {remote} → {local}")
        subprocess.run(
            ["rclone", "copy", remote, str(local), "--transfers", "32",
             "--checkers", "32", "--progress"],
            check=True,
        )


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint registered in pyproject.toml as `exp03`."""
    app(argv if argv is not None else sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
