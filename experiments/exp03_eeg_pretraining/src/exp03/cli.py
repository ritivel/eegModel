"""exp03 CLI — entrypoints for everything we run from the GPU box.

Top-level commands (each a ``typer`` subcommand):

    exp03 paths                  — print the resolved storage layout (sanity)
    exp03 list-releases          — list all HBN-EEG releases on s3://fcp-indi/
    exp03 list-subjects          — list subjects in one release
    exp03 download               — pull raw .set/.fdt + sidecars for a (release, subject) range
    exp03 audit                  — Phase-0 data audit (Karpathy step 1) on the local raw dir
    exp03 preprocess             — apply minimal (and/or v2_clean) pipelines → parquet shards (HBN)
    exp03 tuh-rsync              — rsync TUAB / TUEV from NEDC SFTP into local NVMe
    exp03 tuh-preprocess         — apply v2_clean pipeline → parquet shards (TUH)
    exp03 train                  — run one SSL pretraining cell (any paradigm) with wandb logging
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

from . import hbn, preprocess, storage, tuh

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
    table.add_row("raw_tuab", str(s.raw_tuab))
    table.add_row("raw_tuev", str(s.raw_tuev))
    for pl in storage.ALL_PIPELINES:
        table.add_row(f"derived/{pl}", str(s.derived_pipeline(pl)))
    table.add_row("runs_root", str(s.runs_root))
    table.add_row("hf_cache", str(s.hf_cache))
    for pl in storage.ALL_PIPELINES:
        table.add_row(f"[s3] derived({pl})", s.s3_derived(pl))
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
                    "corpus": "hbn",
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
                    # TUH-only sentinels (the default writer fills these,
                    # but being explicit matches the TUH-side code below).
                    "tuab_label": -1,
                    "tuev_label": -1,
                    "tuh_split": "",
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
# `exp03 tuh-rsync` — pull a TUH corpus from NEDC SFTP
# ---------------------------------------------------------------------------


@app.command("tuh-rsync")
def tuh_rsync_cmd(
    corpus: str = typer.Argument(..., help="Which corpus: 'tuab' or 'tuev'"),
    version: str = typer.Option(None, "--version",
                                help="Override version (default: latest pinned per tuh.py)"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Show what rsync would transfer without pulling"),
    extra: str = typer.Option("", "--extra-rsync-args",
                              help="Extra space-separated rsync args (advanced)"),
):
    """rsync a TUH corpus from NEDC SFTP into ``$EXP03_DATA_ROOT/raw/<corpus>/``.

    Requires the local SSH config alias ``nedc-tuh`` (set up 2026-05-04 to
    route through ``~/.ssh/id_ed25519_nedc``). On the GPU box, prefer
    ``ssh-agent`` forwarding (``ssh -A`` from the Mac) over copying the
    private key onto NVMe — the box's NVMe is wiped on instance stop and
    leaving a long-lived NEDC private key there is unnecessary risk.

    Sizes (rough, 2026-05-04 NEDC inventory):
    - TUAB v3.0.1: ~50 GB raw, ~25 min on g5.8xlarge's typical 30-50 MB/s
    - TUEV v2.0.1: ~30 GB raw, ~15 min
    """
    if corpus not in tuh.CORPORA:
        console.print(f"[red]error:[/red] unknown corpus {corpus!r}; valid: {tuh.CORPORA}")
        raise typer.Exit(1)
    s = storage.from_env()
    s.ensure_dirs()
    local_root = s.raw_tuh(corpus)
    extra_args = tuple(extra.split()) if extra.strip() else ()
    console.print(f"[cyan]rsync[/cyan] NEDC:{tuh.REMOTE_CORPUS_SUBDIR[corpus]}/"
                  f"{version or tuh.DEFAULT_VERSION_BY_CORPUS[corpus]}/ → {local_root}/")
    rc = tuh.rsync_corpus(corpus, local_root, version=version,
                          dry_run=dry_run, extra_rsync_args=extra_args)
    if rc != 0:
        console.print(f"[red]rsync exited with code {rc}[/red]")
        raise typer.Exit(rc)
    console.print(f"[green]done.[/green] Use `exp03 tuh-preprocess {corpus}` next.")


# ---------------------------------------------------------------------------
# `exp03 tuh-preprocess` — build TUH parquet shards (Protocol A.4 fuel)
# ---------------------------------------------------------------------------


@app.command("tuh-preprocess")
def tuh_preprocess_cmd(
    corpus: str = typer.Argument(..., help="Which corpus: 'tuab' or 'tuev'"),
    pipeline: str = typer.Option("v2_clean", "--pipeline",
                                 help="Only 'v2_clean' is supported for TUH (Protocol A.4 is "
                                      "the literature-comparable cell, by design)"),
    version: str = typer.Option(None, "--version",
                                help="Override version (default: latest pinned per tuh.py)"),
    splits: str = typer.Option("train,eval", "--splits",
                               help="Comma-separated NEDC splits to ingest. Default 'train,eval' "
                                    "ingests both; pass 'eval' to skip the train set if you "
                                    "only need the floor probe to land."),
    n_recordings: int = typer.Option(None, "--n",
                                     help="Cap to first N recordings per split (smoke-testing)"),
    overwrite: bool = typer.Option(False, "--overwrite",
                                   help="Re-write parquet shards even if they exist"),
):
    """Apply ``SPEC_V2_CLEAN`` to a TUH corpus → parquet shards under
    ``derived/{tuab,tuev}_v2_clean_250hz/sub-<id>/``.

    The Protocol A.4 evaluation cell is *literature-comparable* by design,
    so we use ``SPEC_V2_CLEAN`` (60 Hz notch + 0.5–100 Hz bandpass + 500
    →250 Hz polyphase resample) rather than the ``SPEC_MINIMAL`` we use
    for HBN — see ``mini_experiments.md`` §4.1 for the rationale (we
    want to compare against BENDR / LaBraM / CBraMod / REVE numbers
    measured on the same preprocessed input). Anything else is reserved
    for a future cell.

    For TUEV recordings we additionally read the sibling ``.rec`` event
    annotations to produce a per-window ``tuev_label`` (0..5; argmax-
    overlap-duration; falls back to BCKG=5 if no annotations overlap a
    window). For TUAB the label is read directly from the path.
    """
    if corpus not in tuh.CORPORA:
        console.print(f"[red]error:[/red] unknown corpus {corpus!r}; valid: {tuh.CORPORA}")
        raise typer.Exit(1)
    if pipeline != "v2_clean":
        console.print(f"[red]error:[/red] only 'v2_clean' is wired for TUH today.")
        raise typer.Exit(1)

    splits_t = tuple(s_.strip() for s_ in splits.split(",") if s_.strip())
    valid_splits = {"train", "eval"}
    bad = [s for s in splits_t if s not in valid_splits]
    if bad:
        console.print(f"[red]error:[/red] bad split(s) {bad}; valid: {sorted(valid_splits)}")
        raise typer.Exit(1)

    s = storage.from_env()
    s.ensure_dirs()
    corpus_root = s.raw_tuh(corpus)
    derived_pipeline_name = (
        storage.PIPELINE_TUAB_V2_CLEAN if corpus == "tuab" else storage.PIPELINE_TUEV_V2_CLEAN
    )
    derived_root = s.derived_pipeline(derived_pipeline_name)
    derived_root.mkdir(parents=True, exist_ok=True)

    recordings = tuh.iter_recordings(
        corpus_root, corpus, splits=splits_t,
        version=version, max_per_split=n_recordings,
    )
    if not recordings:
        console.print(f"[red]error:[/red] no recordings under {corpus_root} for splits {splits_t}; "
                      f"did `exp03 tuh-rsync {corpus}` complete?")
        raise typer.Exit(1)

    console.print(f"[cyan]TUH corpus:[/cyan] {corpus} ({len(recordings)} recordings)")
    console.print(f"[cyan]Output:[/cyan] {derived_root}")

    from tqdm.auto import tqdm

    n_recs_done = 0
    n_rows_total = 0
    n_skipped = 0
    t0 = time.time()
    for rec in tqdm(recordings, desc=f"{corpus}/{pipeline}", unit="rec"):
        sub_dir = derived_root / f"sub-{rec.subject_id}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        # The recording stem (e.g. "aaaaaaar_00000001") is unique within a
        # subject — collisions only happen if a subject has multiple sessions
        # *with the same filename*, which doesn't happen in TUH's layout.
        shard_path = sub_dir / f"{rec.basename}.parquet"
        if shard_path.exists() and not overwrite:
            n_skipped += 1
            continue

        # Provenance
        sha = hashlib.sha256()
        sha.update(rec.edf_path.read_bytes())
        if rec.rec_path is not None:
            sha.update(rec.rec_path.read_bytes())
        src_sha8 = sha.hexdigest()[:8]

        try:
            eeg, sr_native, channel_names = tuh.load_recording(rec.edf_path)
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]skip[/yellow] {rec.edf_path.name}: {e}")
            n_skipped += 1
            continue

        # TUEV: read .rec once and cache for per-window label derivation
        rec_events: list[tuh.RecEvent] = []
        if corpus == "tuev":
            if rec.rec_path is None:
                console.print(f"[yellow]skip[/yellow] {rec.edf_path.name}: missing .rec")
                n_skipped += 1
                continue
            try:
                rec_events = tuh.parse_rec(rec.rec_path)
            except (ValueError, FileNotFoundError) as e:
                console.print(f"[yellow]skip[/yellow] {rec.rec_path.name}: {e}")
                n_skipped += 1
                continue

        # Apply v2_clean and window
        eeg_pp, sr_pp = preprocess.apply_pipeline(eeg, sr_native, pipeline)
        windows, starts = preprocess.window_4s(eeg_pp, sr_pp)
        if windows.shape[0] == 0:
            console.print(f"[yellow]skip[/yellow] {rec.edf_path.name}: too short for one 4-s window")
            n_skipped += 1
            continue

        # Derive per-window labels for TUEV (one label per window, shared
        # across the n_channels iid rows from that window).
        per_window_tuev = None
        if corpus == "tuev":
            per_window_tuev = [
                tuh.tuev_window_label(rec_events, float(starts[w]),
                                      window_seconds=preprocess.WINDOW_SECONDS)
                for w in range(windows.shape[0])
            ]

        # Build per-row dicts. We extend `iid_expand_rows` with the TUH
        # specifics by post-processing — that keeps the HBN code path
        # untouched.
        base_meta = {
            "corpus": corpus,
            "subject_id": rec.subject_id,
            "site": "",
            "recording_id": rec.basename,
            "task_label": -1,
            "sample_rate_hz": int(sr_pp),
            "age": float("nan"),
            "sex": "",
            "p_factor": float("nan"),
            "attention": float("nan"),
            "internalizing": float("nan"),
            "externalizing": float("nan"),
            "adhd": -1,
            "tuab_label": int(rec.tuab_label) if corpus == "tuab" else -1,
            "tuev_label": -1,                # filled per-row below for TUEV
            "tuh_split": rec.split,
            "pipeline": pipeline,
            "src_sha256_8": src_sha8,
        }

        rows = preprocess.iid_expand_rows(windows, starts, channel_names,
                                          base_metadata=base_meta)
        if corpus == "tuev":
            # iid_expand_rows produces n_windows × n_channels rows in
            # window-major order: windows[0] across all channels, then
            # windows[1], etc. So row index = w * n_channels + c.
            n_channels = len(channel_names)
            for w, label in enumerate(per_window_tuev):
                base = w * n_channels
                for c in range(n_channels):
                    rows[base + c]["tuev_label"] = int(label)

        n_written = preprocess.write_parquet_shard(rows, shard_path)
        n_rows_total += n_written
        n_recs_done += 1

    dt = time.time() - t0
    console.print(
        f"\n[green]done[/green]: {n_recs_done} recordings ingested ({n_skipped} skipped), "
        f"{n_rows_total:,} rows in {dt:.1f}s "
        f"({n_rows_total / max(dt, 1e-3):.0f} rows/s)"
    )


# ---------------------------------------------------------------------------
# `exp03 train` — run one SSL pretraining cell (paradigm × seed)
# ---------------------------------------------------------------------------


@app.command("train")
def train_cmd(
    paradigm: str = typer.Option("mae", "--paradigm",
                                 help="Generative paradigm: 'mae' (G0), 'ar' (G1), 'mar' (G2), 'jepa' (G3, latent-prediction; v2)."),
    backbone_kind: str = typer.Option("mamba2", "--backbone-kind",
                                      help="Backbone: 'mamba2' (default) or 'transformer' (CPU/Mac fallback)"),
    backbone_layers: int = typer.Option(6, "--backbone-layers"),
    backbone_d_model: int = typer.Option(256, "--backbone-d-model"),
    decoder_layers: int = typer.Option(2, "--decoder-layers",
                                       help="Only used by G0 MAE."),
    mask_ratio: float = typer.Option(0.50, "--mask-ratio",
                                     help="Only used by G0/G2; G1 has no masking."),
    diffloss_d: int = typer.Option(3, "--diffloss-d", help="MAR DiffLoss MLP depth."),
    diffloss_w: int = typer.Option(1024, "--diffloss-w", help="MAR DiffLoss MLP width."),
    diffusion_batch_mul: int = typer.Option(1, "--diffusion-batch-mul",
                                            help="MAR per-sample replication for noise-level coverage. "
                                                 "MAR paper uses 4."),
    data_root: Path = typer.Option(None, "--data-root",
                                   help="Override $EXP03_DATA_ROOT/derived/<pipeline>/."),
    pipeline: str = typer.Option("hbn_minimal_500hz", "--pipeline",
                                 help="Which derived/<pipeline>/ folder to read from. "
                                      "Default 'hbn_minimal_500hz' is the §4.1 primary."),
    max_steps: int = typer.Option(1000, "--steps"),
    batch_size: int = typer.Option(32, "--batch-size"),
    accum_iter: int = typer.Option(1, "--accum-iter"),
    lr: float = typer.Option(1e-4, "--lr"),
    blr: float = typer.Option(None, "--blr",
                              help="Base LR; if set, lr = blr * eff_batch / 256 (MAE convention)."),
    weight_decay: float = typer.Option(0.05, "--weight-decay"),
    warmup_steps: int = typer.Option(100, "--warmup-steps"),
    grad_clip_norm: float = typer.Option(1.0, "--grad-clip-norm"),
    precision: str = typer.Option("bf16", "--precision",
                                  help="'bf16' (default), 'fp32', or 'fp16'. Mamba-2's segsum primitive "
                                       "can NaN under fp16 — prefer bf16 or fp32."),
    seed: int = typer.Option(0, "--seed"),
    output_dir: Path = typer.Option(None, "--output-dir",
                                    help="Where to write checkpoints + summary.json. Default: "
                                         "$EXP03_DATA_ROOT/runs/<wandb_run_name or auto>."),
    log_every: int = typer.Option(20, "--log-every"),
    ckpt_every: int = typer.Option(0, "--ckpt-every",
                                   help="0 = checkpoint at end only. Otherwise every N steps."),
    eval_at_end: bool = typer.Option(True, "--eval-at-end/--no-eval-at-end"),
    eval_max_subjects: int = typer.Option(50, "--eval-max-subjects"),
    num_workers: int = typer.Option(2, "--num-workers",
                                    help="DataLoader workers. Set 0 for single-process dev "
                                         "(macOS, stdin scripts) or to debug worker errors."),
    noise_twin: bool = typer.Option(False, "--noise-twin/--no-noise-twin",
                                    help="§3 control: replace the EEG signal with torch.randn_like at "
                                         "the model input. The matched-noise-twin cell of the §17 "
                                         "control matrix. Pretraining still happens normally; "
                                         "frozen-probe eval at end measures whether the encoder "
                                         "developed downstream-useful structure on pure noise."),
    wandb_project: str = typer.Option("exp03", "--wandb-project"),
    wandb_run_name: str = typer.Option(None, "--wandb-run-name"),
    wandb_mode: str = typer.Option("online", "--wandb-mode",
                                   help="'online' / 'offline' / 'disabled'"),
    wandb_tags: str = typer.Option("", "--wandb-tags",
                                   help="Comma-separated tags."),
    # ---- s3 checkpoint mirror -----------------------------------------
    s3_ckpt_bucket: str = typer.Option(None, "--s3-ckpt-bucket",
                                       help="If set, every ckpt is also pushed to "
                                            "s3://<bucket>/<prefix>/ via s3torchconnector. "
                                            "Recommended for capacity-block runs (so an EndDate "
                                            "force-stop doesn't lose the latest checkpoint)."),
    s3_ckpt_prefix: str = typer.Option(None, "--s3-ckpt-prefix",
                                       help="Default: runs/exp03/<wandb_run_name>"),
    s3_ckpt_region: str = typer.Option("us-west-2", "--s3-ckpt-region",
                                       help="Region of --s3-ckpt-bucket; default = warehouse."),
    s3_ckpt_resume: bool = typer.Option(True, "--s3-ckpt-resume/--no-s3-ckpt-resume",
                                        help="On train start, resume accelerate state from "
                                             "s3://<bucket>/<prefix>/accelerate/ if it exists."),
    use_dcp: bool = typer.Option(False, "--use-dcp/--no-use-dcp",
                                  help="Use PyTorch Distributed Checkpoint (DCP) for sharded "
                                       "multi-rank ckpt writes via s3torchconnector. Recommended "
                                       "for 8-way FSDP/DDP runs; equivalent to single-file save "
                                       "for single-rank training."),
    notion_experiment_id: str = typer.Option(None, "--notion-experiment-id",
                                              help="Notion page ID of the parent Experiment row "
                                                   "(seen via `eeg-ops config` or by opening the "
                                                   "Operations Hub). Used to relate the Run row "
                                                   "to its Experiment."),
    notion_session_id: str = typer.Option(None, "--notion-session-id",
                                           help="Notion page ID of the gpu_rental Session this "
                                                "run belongs to. Defaults to the active rental "
                                                "session if you've previously run "
                                                "`eeg-ops capacity buy`."),
):
    """Run one SSL pretraining cell.

    Examples:

    \b
        # Smoke-test G0 MAE for 100 steps with wandb disabled
        exp03 train --paradigm mae --steps 100 --wandb-mode disabled --no-eval-at-end

    \b
        # Real exp17 cell: G2 MAR, ~35M tokens (= 17500 steps × batch 32 × patch 8 / 2000 samples)
        exp03 train --paradigm mar --steps 17500 --batch-size 32 \\
            --diffusion-batch-mul 4 --blr 1e-4 \\
            --wandb-run-name exp17-g2-seed0 --wandb-tags exp17,g2,mar

    \b
        # Forward-only Mamba AR pretraining (G1)
        exp03 train --paradigm ar --steps 17500 --batch-size 32 \\
            --wandb-run-name exp17-g1-seed0 --wandb-tags exp17,g1,ar
    """
    from . import train as train_mod

    s = storage.from_env()
    if data_root is None:
        data_root = s.derived_pipeline(pipeline)
    if output_dir is None:
        run_id = wandb_run_name or f"exp17-{paradigm}-seed{seed}-{int(time.time())}"
        output_dir = s.runs_root / run_id

    cfg = train_mod.TrainConfig(
        paradigm=paradigm,
        backbone_kind=backbone_kind,
        backbone_layers=backbone_layers,
        backbone_d_model=backbone_d_model,
        decoder_layers=decoder_layers,
        mask_ratio=mask_ratio,
        diffloss_d=diffloss_d,
        diffloss_w=diffloss_w,
        diffusion_batch_mul=diffusion_batch_mul,
        data_root=data_root,
        max_steps=max_steps,
        batch_size=batch_size,
        accum_iter=accum_iter,
        lr=lr,
        blr=blr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        grad_clip_norm=grad_clip_norm,
        precision=precision,
        seed=seed,
        output_dir=output_dir,
        log_every=log_every,
        ckpt_every=ckpt_every,
        eval_at_end=eval_at_end,
        eval_max_subjects=eval_max_subjects,
        num_workers=num_workers,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
        wandb_tags=tuple(t.strip() for t in wandb_tags.split(",") if t.strip()),
        noise_twin=noise_twin,
        s3_ckpt_bucket=s3_ckpt_bucket,
        s3_ckpt_prefix=s3_ckpt_prefix,
        s3_ckpt_region=s3_ckpt_region,
        s3_ckpt_resume=s3_ckpt_resume,
        use_dcp=use_dcp,
        notion_experiment_id=notion_experiment_id,
        notion_session_id=notion_session_id or _resolve_active_rental_session(),
    )
    console.print(f"[cyan]train cell[/cyan] paradigm={paradigm} seed={seed} "
                  f"steps={max_steps} batch={batch_size} -> {output_dir}")
    # Use the lifecycle-aware wrapper so Notion gets a run_crashed event on
    # exception. ``train_mod.train`` is still the pure entry point for tests.
    result = train_mod.train_with_lifecycle(cfg)
    console.print(f"[green]done[/green]: {result}")


def _resolve_active_rental_session() -> str | None:
    """Look up the active gpu_rental Session ID from ~/.config/eeg-ops/state.toml.

    Returns None if eeg_ops isn't installed or the state file is missing —
    every code path that uses this is best-effort.
    """
    try:
        from eeg_ops.config import State
        return State.load().notion_rental_session_id
    except Exception:                                                       # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# `exp03 sync-derived-up` / `exp03 sync-derived-down`
# ---------------------------------------------------------------------------


_PIPELINE_GROUPS: dict[str, tuple[str, ...]] = {
    "all": storage.ALL_PIPELINES,
    "hbn": (storage.PIPELINE_MINIMAL, storage.PIPELINE_V2_CLEAN),
    "tuh": (storage.PIPELINE_TUAB_V2_CLEAN, storage.PIPELINE_TUEV_V2_CLEAN),
    "both": (storage.PIPELINE_MINIMAL, storage.PIPELINE_V2_CLEAN),  # legacy alias for HBN-only
    "minimal": (storage.PIPELINE_MINIMAL,),
    "v2_clean": (storage.PIPELINE_V2_CLEAN,),
    "tuab": (storage.PIPELINE_TUAB_V2_CLEAN,),
    "tuev": (storage.PIPELINE_TUEV_V2_CLEAN,),
}


def _resolve_pipelines(pipeline: str) -> list[str]:
    if pipeline in _PIPELINE_GROUPS:
        return list(_PIPELINE_GROUPS[pipeline])
    raise typer.BadParameter(
        f"unknown pipeline group {pipeline!r}; valid: {sorted(_PIPELINE_GROUPS)}"
    )


@app.command("sync-derived-up")
def sync_up_cmd(
    pipeline: str = typer.Option("hbn", "--pipeline",
                                 help=("Pipeline group to sync up: "
                                       "'all' (everything), 'hbn' (minimal+v2_clean), "
                                       "'tuh' (tuab+tuev), or one of "
                                       "'minimal'|'v2_clean'|'tuab'|'tuev'. "
                                       "Default 'hbn' matches the pre-2026-05-04 behaviour.")),
):
    """Sync local derived/<pipeline>/ → s3://eegmodel-warehouse/derived/<pipeline>/.

    Idempotent (rclone copy). Use after `exp03 preprocess` or
    `exp03 tuh-preprocess` to materialise the warehouse mirror.
    """
    import subprocess

    s = storage.from_env()
    for pl in _resolve_pipelines(pipeline):
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
    pipeline: str = typer.Option("hbn", "--pipeline",
                                 help=("Pipeline group to sync down: same set as "
                                       "sync-derived-up. Default 'hbn' matches "
                                       "the pre-2026-05-04 behaviour.")),
):
    """Sync s3://eegmodel-warehouse/derived/<pipeline>/ → local derived/.

    The "preprocess once, ever" promise: on every new GPU box's bootstrap,
    pull the preprocessed shards instead of re-running preprocess.
    """
    import subprocess

    s = storage.from_env()
    s.ensure_dirs()
    for pl in _resolve_pipelines(pipeline):
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
