"""HBN-EEG ingestion: catalog the public AWS bucket, download recordings,
read EEGLAB ``.set/.fdt`` via MNE, parse BIDS-iEEG sidecar metadata, and
derive the labels we need for the §4.3 eval suite (HBN ADHD-binary AUROC,
HBN 6-task classification BAC + WF1).

Source: ``s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_<release>/`` —
NIH-funded public mirror, no credentials needed for read; we use anonymous
boto3 (UNSIGNED) to avoid charging the user's AWS account for the
requester-pays-style listing.

Per-release layout (verified against the 2026-05 bucket state):

    s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_<release>/
        participants.tsv                                ← per-release
        dataset_description.json
        code/
        sub-NDARABCD/
            eeg/
                sub-NDARABCD_task-RestingState_channels.tsv
                sub-NDARABCD_task-RestingState_coordsystem.json
                sub-NDARABCD_task-RestingState_eeg.set     ← raw EEG (EEGLAB header)
                sub-NDARABCD_task-RestingState_eeg.fdt     ← raw EEG (EEGLAB binary)
                sub-NDARABCD_task-RestingState_eeg.json    ← BIDS sidecar
                sub-NDARABCD_task-RestingState_electrodes.tsv
                sub-NDARABCD_task-RestingState_events.tsv
                sub-NDARABCD_task-seqLearning8target_eeg.set
                sub-NDARABCD_task-symbolSearch_eeg.set
                sub-NDARABCD_task-surroundSupp_run-1_eeg.set
                sub-NDARABCD_task-surroundSupp_run-2_eeg.set
                sub-NDARABCD_task-contrastChangeDetection_run-1_eeg.set
                sub-NDARABCD_task-contrastChangeDetection_run-2_eeg.set
                sub-NDARABCD_task-contrastChangeDetection_run-3_eeg.set
                sub-NDARABCD_task-DespicableMe_eeg.set        ← Video category
                sub-NDARABCD_task-DiaryOfAWimpyKid_eeg.set    ← Video category
                sub-NDARABCD_task-FunwithFractals_eeg.set     ← Video category
                sub-NDARABCD_task-ThePresent_eeg.set          ← Video category

10 releases known: NC, R1..R9. Total ~2,639 subjects across all releases
(NC=447, R1=136, R2=150, R3=184, R4=324, R5=330, R6=135, R7=381,
R8=257, R9=295) per a 2026-05-02 listing.

References:
    HBN-EEG dataset:  Shirazi et al. 2024 bioRxiv,
        https://www.biorxiv.org/content/10.1101/2024.10.03.615261v2
    AWS public mirror layout (``s3://fcp-indi/.../HBN/BIDS_EEG/...``):
        https://fcp-indi.s3.amazonaws.com/index.html
    NeurIPS 2025 EEG Foundation Challenge (uses HBN):
        https://eeg2025.github.io/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HBN's six cognitive tasks → label index per `mini_experiments.md` §4.3
# Protocol A.2 (HBN 6-task classification). Task names as they appear in BIDS
# filenames are case-sensitive and have HBN-specific quirks:
#   - sequence learning ships in two variants per release (6-target and
#     8-target); each subject typically has one or the other, never both.
#     We collapse both to label 1 ("sequence learning").
#   - surroundSupp + contrastChangeDetection ship with multiple `_run-<N>`
#     runs per subject; the regex strips _run-N and we collapse all runs
#     to one label (3, 4 respectively).
#   - "Video" category splits across four distinct movie-clip task names
#     (DespicableMe, DiaryOfAWimpyKid, FunwithFractals, ThePresent), all
#     → label 5.
TASK_LABEL: dict[str, int] = {
    "RestingState":             0,
    "seqLearning6target":       1,   # ~half of subjects have this variant
    "seqLearning8target":       1,   # other half have this one
    "symbolSearch":             2,
    "surroundSupp":             3,
    "contrastChangeDetection":  4,
    # Video category — collapse the four movie-clip task names → label 5
    "DespicableMe":             5,
    "DiaryOfAWimpyKid":         5,
    "FunwithFractals":          5,
    "ThePresent":               5,
}

# Reverse map: label → canonical task name (we pick a representative
# string for display purposes; the actual filenames vary).
TASK_NAME_BY_LABEL: dict[int, str] = {
    0: "RestingState",
    1: "seqLearning",      # 6target or 8target depending on subject
    2: "symbolSearch",
    3: "surroundSupp",
    4: "contrastChangeDetection",
    5: "Video",            # one of {DespicableMe, DiaryOfAWimpyKid, FunwithFractals, ThePresent}
}

# Filename-parse regex. BIDS filenames look like:
#     sub-NDARxxx_task-<task>[_run-<N>]_<datatype>.<ext>
# datatype ∈ {eeg, channels, electrodes, coordsystem, events}.
# We capture the task name (alphanumeric, may contain digits) up to the
# optional ``_run-<N>`` and the trailing ``_<datatype>``.
_TASK_FILENAME_RE = re.compile(
    r"task-([A-Za-z][A-Za-z0-9]*?)(?:_run-\d+)?_(?:eeg|channels|electrodes|coordsystem|events)"
)

# HBN site code: provided in participants.tsv via the "ehq_total_completed"-
# adjacent "release_number" / "site" columns (release-dependent). We resolve
# site lazily from the participants.tsv frame at preprocess time.
HBN_SITES: tuple[str, ...] = ("RU", "CBIC", "CUNY", "SI")

# 128-channel EGI HydroCel naming convention; HBN's channels.tsv uses these
# literal names. Order matters — channel_idx in our parquet schema is the
# position in this list.
HYDROCEL_128_NAMES: tuple[str, ...] = tuple(f"E{i}" for i in range(1, 129))

# FCP-INDI HBN-EEG bucket + prefix (BIDS-formatted release tree).
HBN_S3_BUCKET = "fcp-indi"
HBN_S3_PREFIX = "data/Projects/HBN/BIDS_EEG"

# HBN release codes (the prefix we type when calling the CLI). The actual S3
# directory name is ``cmi_bids_<release>`` — see ``release_s3_prefix``.
KNOWN_RELEASES: tuple[str, ...] = (
    "NC", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
)


def release_s3_prefix(release: str) -> str:
    """Map user-facing release code (e.g. 'R1') → S3 prefix segment ('cmi_bids_R1')."""
    return f"cmi_bids_{release}"


def parse_task_from_filename(filename: str) -> tuple[str, int] | None:
    """Parse a BIDS-iEEG filename → (canonical_task_name, task_label) or None.

    Examples:
        ``sub-NDARxxx_task-RestingState_eeg.set``         → ("RestingState", 0)
        ``sub-NDARxxx_task-seqLearning8target_eeg.set``   → ("seqLearning8target", 1)
        ``sub-NDARxxx_task-surroundSupp_run-1_eeg.set``   → ("surroundSupp", 3)
        ``sub-NDARxxx_task-DespicableMe_eeg.set``         → ("DespicableMe", 5)
        ``sub-NDARxxx_task-Unknown_eeg.set``              → None  (not in TASK_LABEL)
    """
    m = _TASK_FILENAME_RE.search(filename)
    if not m:
        return None
    task = m.group(1)
    label = TASK_LABEL.get(task)
    if label is None:
        return None
    return task, label


# ---------------------------------------------------------------------------
# Subject + recording dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    """One EEGLAB ``.set`` (and optional ``.fdt`` sidecar) recording.

    Modern HBN BIDS releases (cmi_bids_R1..R9, NC) ship single-file ``.set``
    where the binary samples are embedded inside the .set file itself and
    *no* ``.fdt`` sidecar is present. Earlier EEGLAB versions used a paired
    ``.set`` (header) + ``.fdt`` (binary) layout. We support both: the
    ``.fdt`` key/path is Optional and may be ``None``.

    Identifiers in S3 + on local disk (the local layout mirrors the BIDS
    S3 layout including the ``cmi_bids_<release>/`` prefix, so a flat
    ``rclone`` between local and S3 is a no-op rename):

        s3_set:  s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_<release>/sub-<id>/eeg/<basename>.set
        s3_fdt:  ...<basename>.fdt   (None if absent)
        local:   <storage.raw_hbn>/cmi_bids_<release>/sub-<id>/eeg/<basename>.set
    """

    subject_id: str               # "NDARABCD1234"
    release: str                  # "NC", "R1", ...
    task: str                     # canonical task name from TASK_LABEL keys
    task_label: int               # 0..5 (label index for §4.3 Protocol A.2)
    basename: str                 # e.g. "sub-NDARABCD1234_task-RestingState_eeg"
    s3_set_key: str               # full S3 key of the .set (always present)
    s3_fdt_key: str | None = None # full S3 key of the .fdt; None for single-file .set

    @property
    def s3_set_uri(self) -> str:
        return f"s3://{HBN_S3_BUCKET}/{self.s3_set_key}"

    @property
    def s3_fdt_uri(self) -> str | None:
        return f"s3://{HBN_S3_BUCKET}/{self.s3_fdt_key}" if self.s3_fdt_key else None

    def _local_eeg_dir(self, raw_root: Path) -> Path:
        return raw_root / release_s3_prefix(self.release) / f"sub-{self.subject_id}" / "eeg"

    def local_set_path(self, raw_root: Path) -> Path:
        return self._local_eeg_dir(raw_root) / f"{self.basename}.set"

    def local_fdt_path(self, raw_root: Path) -> Path | None:
        if self.s3_fdt_key is None:
            return None
        return self._local_eeg_dir(raw_root) / f"{self.basename}.fdt"


# ---------------------------------------------------------------------------
# S3 anonymous client + listing
# ---------------------------------------------------------------------------


def _anonymous_s3_client():
    """boto3 client that doesn't sign requests (FCP-INDI bucket is public)."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_releases() -> list[str]:
    """List all HBN-EEG release codes available under ``BIDS_EEG/cmi_bids_*``.

    Returns user-facing release codes (e.g. ``["NC", "R1", "R2", ...]``),
    stripping the ``cmi_bids_`` prefix the bucket uses for the directory name.
    """
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    releases: set[str] = set()
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=f"{HBN_S3_PREFIX}/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            tail = cp["Prefix"].rstrip("/").split("/")[-1]
            if tail.startswith("cmi_bids_"):
                releases.add(tail[len("cmi_bids_"):])
    return sorted(releases)


def list_subjects(release: str, *, max_subjects: int | None = None) -> list[str]:
    """List subject IDs (without ``sub-`` prefix) under one release."""
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/"
    subjects: list[str] = []
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []) or []:
            tail = cp["Prefix"].rstrip("/").split("/")[-1]
            if tail.startswith("sub-"):
                subjects.append(tail[len("sub-"):])
                if max_subjects is not None and len(subjects) >= max_subjects:
                    return subjects
    return subjects


def list_subject_recordings(subject_id: str, release: str) -> list[Recording]:
    """List all recordings for one subject in one release.

    Single S3 listing pass; bucket .set and .fdt keys by basename so we can
    construct a Recording with the correct optional s3_fdt_key (HBN's modern
    BIDS releases are single-file .set with no .fdt sidecar).
    """
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/sub-{subject_id}/eeg/"

    # basename → {".set": s3_key, ".fdt": s3_key}
    by_basename: dict[str, dict[str, str]] = {}
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            ext = Path(key).suffix
            if ext not in (".set", ".fdt"):
                continue
            stem = Path(key).stem
            by_basename.setdefault(stem, {})[ext] = key

    recordings: list[Recording] = []
    for basename, ext_keys in sorted(by_basename.items()):
        if ".set" not in ext_keys:
            continue  # orphan .fdt without .set — skip
        parsed = parse_task_from_filename(basename)
        if parsed is None:
            continue
        task, task_label = parsed
        recordings.append(Recording(
            subject_id=subject_id,
            release=release,
            task=task,
            task_label=task_label,
            basename=basename,
            s3_set_key=ext_keys[".set"],
            s3_fdt_key=ext_keys.get(".fdt"),  # None if no sidecar
        ))
    return recordings


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_recording(rec: Recording, raw_root: Path,
                       *, overwrite: bool = False) -> tuple[Path, Path | None]:
    """Download a single recording's ``.set`` (and optional ``.fdt`` sidecar).

    Returns (local_set_path, local_fdt_path_or_None). Skips files that
    already exist unless ``overwrite=True``. The ``.fdt`` is only downloaded
    if the recording metadata indicates a sidecar exists (``rec.s3_fdt_key``
    is not None) — modern HBN BIDS releases are single-file .set with no
    .fdt.
    """
    s3 = _anonymous_s3_client()
    set_path = rec.local_set_path(raw_root)
    set_path.parent.mkdir(parents=True, exist_ok=True)

    # .set is mandatory
    if not (set_path.exists() and not overwrite):
        tmp_path = set_path.with_suffix(set_path.suffix + ".part")
        s3.download_file(HBN_S3_BUCKET, rec.s3_set_key, str(tmp_path))
        tmp_path.replace(set_path)

    # .fdt is optional
    fdt_path: Path | None = None
    if rec.s3_fdt_key is not None:
        fdt_path = rec.local_fdt_path(raw_root)
        if fdt_path is not None and not (fdt_path.exists() and not overwrite):
            tmp_path = fdt_path.with_suffix(fdt_path.suffix + ".part")
            s3.download_file(HBN_S3_BUCKET, rec.s3_fdt_key, str(tmp_path))
            tmp_path.replace(fdt_path)

    return set_path, fdt_path


def download_subject_sidecars(subject_id: str, release: str, raw_root: Path,
                              *, overwrite: bool = False) -> list[Path]:
    """Pull the BIDS sidecar files (channels.tsv, eeg.json, electrodes.tsv,
    coordsystem.json, events.tsv).

    These are small (~kB each) and needed for channel-name parsing + sample
    rate verification. We pull them all per-subject in one shot; the .set/.fdt
    pairs are pulled separately per-recording.
    """
    s3 = _anonymous_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    prefix = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/sub-{subject_id}/eeg/"
    out: list[Path] = []
    for page in paginator.paginate(Bucket=HBN_S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith((".set", ".fdt")):
                continue
            local = raw_root / release_s3_prefix(release) / f"sub-{subject_id}" / "eeg" / Path(key).name
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and not overwrite:
                out.append(local)
                continue
            s3.download_file(HBN_S3_BUCKET, key, str(local))
            out.append(local)
    return out


def download_release_metadata(release: str, raw_root: Path,
                              *, overwrite: bool = False) -> dict[str, Path]:
    """Pull release-level files: participants.tsv + dataset_description.json."""
    s3 = _anonymous_s3_client()
    out: dict[str, Path] = {}
    for fname in ("participants.tsv", "dataset_description.json"):
        key = f"{HBN_S3_PREFIX}/{release_s3_prefix(release)}/{fname}"
        local = raw_root / release_s3_prefix(release) / fname
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and not overwrite:
            out[fname] = local
            continue
        try:
            s3.download_file(HBN_S3_BUCKET, key, str(local))
            out[fname] = local
        except Exception as e:  # noqa: BLE001 — best-effort; release may lack one of these
            out.setdefault(fname, None)
            print(f"[hbn] release {release}: missing {fname} ({e})")
    return out


# ---------------------------------------------------------------------------
# .set/.fdt loader (MNE)
# ---------------------------------------------------------------------------


def load_recording(set_path: Path):
    """Read an EEGLAB ``.set`` file via MNE → return (eeg, sr, channel_names).

    Modern EEGLAB single-file ``.set`` embeds the binary samples; older
    paired-file format requires a sibling ``.fdt`` next to the ``.set``. MNE
    handles both transparently as long as the .fdt is co-located when needed
    — we don't pre-check, just let MNE raise if it's actually missing.

    Returns:
        eeg: float32 ndarray of shape (n_channels, n_samples), in **volts**
             (MNE's native unit; we keep volts here and let the per-channel
             z-score normalise downstream).
        sr:  float, sampling rate in Hz.
        channel_names: list[str], length n_channels.

    Raises FileNotFoundError if the .set itself is missing.
    """
    import mne

    set_path = Path(set_path)
    if not set_path.exists():
        raise FileNotFoundError(f"missing .set file: {set_path}")

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
    eeg = raw.get_data().astype("float32")  # (n_channels, n_samples), volts
    sr = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    return eeg, sr, channel_names


# ---------------------------------------------------------------------------
# participants.tsv parsing — derive HBN ADHD-binary label per §4.3 A.1
# ---------------------------------------------------------------------------


# HBN's continuous CBCL Pearson-z factor columns — the actual eval-target
# label set per `mini_experiments.md` §4.3 Protocol A.1. Verified against
# release R1's participants.tsv (2026-05-02): all 4 columns present with
# 132/136 non-NaN values; 4 missing rows are subjects where CBCL
# scoring was not completed.
CBCL_FACTOR_COLS: tuple[str, ...] = (
    "p_factor",       # general psychopathology
    "attention",      # ADHD-severity proxy (CBCL Attention factor)
    "internalizing",  # anxiety / depression
    "externalizing",  # conduct / aggression — NeurIPS 2025 EEG-FM Challenge C2 target
)

# Legacy ADHD-binary column. HBN releases verified 2026-05-02 ship NO DSM-V
# Dx columns of any kind, so this regex never matches anything; kept as a
# parquet-schema-compatibility sentinel that always evaluates to -1
# ("missing"). If a future HBN release adds Dx columns, this regex will
# light up automatically and `adhd` will start populating with 0/1.
ADHD_DX_PATTERN = re.compile(r"\b(?:ADHD|attention[\s\-_]*deficit)\b", re.IGNORECASE)


def load_participants(participants_tsv: Path):
    """Read participants.tsv → pandas DataFrame with HBN's actual eval-target
    columns standardised.

    HBN BIDS releases (verified R1, 2026-05-02) ship 24 columns including:
      participant_id, release_number, sex, age, ehq_total, commercial_use,
      full_pheno, p_factor, attention, internalizing, externalizing,
      RestingState, DespicableMe, ..., contrastChangeDetection_1, ..., etc.

    NO DSM-V Dx columns. The four CBCL Pearson-z factors (p_factor,
    attention, internalizing, externalizing) are the canonical eval-target
    columns for §4.3 Protocol A.1.

    Returned DataFrame has the original columns plus standardised:
        subject_id     string  (without "sub-" prefix)
        site           string
        age            float
        sex            string  ("M" / "F" / "")
        p_factor       float   (NaN if missing)
        attention      float   (NaN if missing)
        internalizing  float   (NaN if missing)
        externalizing  float   (NaN if missing)
        adhd           int8    (always -1 for current HBN; reserved for
                                schema-compat with downstream code that
                                expects the field — see ADHD_DX_PATTERN)
    """
    import pandas as pd

    df = pd.read_csv(participants_tsv, sep="\t", dtype=str,
                     keep_default_na=False, na_values=[""])

    # Subject ID (strip "sub-" prefix)
    if "participant_id" in df.columns:
        df["subject_id"] = df["participant_id"].str.replace(r"^sub-", "", regex=True)
    elif "subject_id" not in df.columns:
        raise KeyError(
            f"participants.tsv missing participant_id / subject_id: "
            f"cols={list(df.columns)}"
        )

    # Continuous CBCL factors (the actual eval targets per §4.3 Protocol A.1)
    for col in CBCL_FACTOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    # Legacy ADHD-binary slot — scan any Dx_* / diagnosis_* columns if any
    # appear in a future release. Current HBN has none, so this stays at -1.
    dx_cols = [c for c in df.columns if c.lower().startswith(("dx_", "diagnosis_"))]
    if dx_cols:
        adhd_vals = []
        for _, row in df[dx_cols].iterrows():
            non_empty = [v for v in row.values if isinstance(v, str) and v.strip()]
            if not non_empty:
                adhd_vals.append(-1)
            elif any(ADHD_DX_PATTERN.search(v) for v in non_empty):
                adhd_vals.append(1)
            else:
                adhd_vals.append(0)
        df["adhd"] = adhd_vals
    else:
        df["adhd"] = -1

    # Site (best-effort: HBN uses "site" or "Site" depending on release)
    site_col = next((c for c in df.columns if c.lower() == "site"), None)
    df["site"] = df[site_col].fillna("") if site_col else ""

    # Age + sex
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = float("nan")
    if "sex" not in df.columns:
        df["sex"] = ""
    df["sex"] = df["sex"].fillna("").str.upper().replace({"MALE": "M", "FEMALE": "F"})

    return df


def metadata_for_subject(subject_id: str, participants_df) -> dict:
    """Look up demographics + CBCL factors for one subject.

    Returns a dict with keys
        site, age, sex,
        p_factor, attention, internalizing, externalizing,
        adhd
    suitable as the parquet base_metadata. Missing values use sentinels:
        site = "", age = NaN, sex = "",
        p_factor / attention / internalizing / externalizing = NaN,
        adhd = -1.
    """
    import pandas as pd

    missing = {
        "site": "", "age": float("nan"), "sex": "",
        "p_factor": float("nan"),
        "attention": float("nan"),
        "internalizing": float("nan"),
        "externalizing": float("nan"),
        "adhd": -1,
    }
    if participants_df is None:
        return missing
    rows = participants_df[participants_df["subject_id"] == subject_id]
    if rows.empty:
        return missing
    r = rows.iloc[0]

    def _f(col: str) -> float:
        v = r.get(col, float("nan"))
        if isinstance(v, str):
            try:
                return float(v) if v.strip() else float("nan")
            except ValueError:
                return float("nan")
        try:
            return float(v) if pd.notna(v) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    return {
        "site":          str(r.get("site", "")),
        "age":           _f("age"),
        "sex":           str(r.get("sex", "")),
        "p_factor":      _f("p_factor"),
        "attention":     _f("attention"),
        "internalizing": _f("internalizing"),
        "externalizing": _f("externalizing"),
        "adhd":          int(r["adhd"]) if pd.notna(r.get("adhd")) else -1,
    }


# ---------------------------------------------------------------------------
# Iteration helpers (used by the CLI)
# ---------------------------------------------------------------------------


def iter_releases_and_subjects(releases: Iterable[str] | None = None,
                               max_subjects_per_release: int | None = None,
                               ) -> list[tuple[str, str]]:
    """Cross-product (release, subject_id) iteration helper.

    If ``releases`` is None, lists all releases. ``max_subjects_per_release``
    caps each release independently — useful for Tier-1 sanity processing.
    """
    rels = list(releases) if releases else list_releases()
    out: list[tuple[str, str]] = []
    for release in rels:
        subs = list_subjects(release, max_subjects=max_subjects_per_release)
        for sub in subs:
            out.append((release, sub))
    return out
